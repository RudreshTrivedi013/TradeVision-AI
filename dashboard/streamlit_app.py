
import sys
import os
import json
from pathlib import Path
from datetime import datetime, date, timedelta

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import yaml
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore
from src.data_pipeline import DataPipeline


# ===================================================================
# Page Config
# ===================================================================
st.set_page_config(
    page_title="TradeVision AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================================================================
# Custom CSS for premium look
# ===================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .main { background-color: #0e1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16192b 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-label {
        font-size: 13px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
    }
    .metric-up { color: #34d399; }
    .metric-down { color: #f87171; }

    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #e2e8f0;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #6366f1;
        display: inline-block;
    }

    .stSidebar > div:first-child {
        background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2235 0%, #171c2e 100%);
        border: 1px solid #2d3348;
        border-radius: 10px;
        padding: 12px 16px;
    }

    /* ── Final Prediction Card ───────────────────────────────── */
    .fp-card {
        background: #13172b;
        border: 1px solid #252b45;
        border-radius: 14px;
        padding: 22px 32px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 48px;
        flex-wrap: wrap;
    }
    .fp-label {
        font-size: 10px;
        font-weight: 600;
        color: #4b5680;
        text-transform: uppercase;
        letter-spacing: 1.8px;
        margin-bottom: 4px;
    }
    .fp-direction-up   { font-size: 30px; font-weight: 800; color: #34d399; letter-spacing: -0.5px; }
    .fp-direction-down { font-size: 30px; font-weight: 800; color: #f87171; letter-spacing: -0.5px; }
    .fp-sub { font-size: 11px; color: #4b5680; margin-top: 3px; }
    .fp-divider { width:1px; height:44px; background:#252b45; flex-shrink:0; }
    .fp-winrate { font-size: 30px; font-weight: 800; color: #818cf8; letter-spacing: -0.5px; }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# Load config
# ===================================================================
@st.cache_data
def load_config():
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ===================================================================
# Load models
# ===================================================================
@st.cache_resource
def load_models(config):
    models_dir = PROJECT_ROOT / config["training"]["models_dir"]
    models = {}
    for name in ["RandomForest", "XGBoost", "LogisticRegression", "isolation_forest"]:
        path = models_dir / f"{name}.pkl"
        if path.exists():
            models[name] = joblib.load(path)
    return models


# ===================================================================
# Sidebar
# ===================================================================
def sidebar():
    st.sidebar.markdown("# 📈 TradeVision AI")
    st.sidebar.markdown("---")

    config = load_config()
    default_tickers = config["data"]["default_tickers"]

    if "ticker_input" not in st.session_state:
        st.session_state.ticker_input = config["dashboard"]["default_ticker"]

    def set_ticker(t):
        st.session_state.ticker_input = t

    ticker = st.sidebar.text_input(
        "Enter Ticker Symbol",
        key="ticker_input",
        help="Enter any valid stock ticker (e.g. AAPL, NVDA, META)",
    ).upper().strip()

    st.sidebar.markdown("**Quick Select:**")
    cols = st.sidebar.columns(3)
    for i, t in enumerate(default_tickers):
        if cols[i % 3].button(t, key=f"quick_{t}", use_container_width=True, on_click=set_ticker, args=(t,)):
            pass

    st.sidebar.markdown("---")

    period = st.sidebar.selectbox(
        "Time Period",
        ["6mo", "1y", "2y", "5y"],
        index=1,
    )

    show_indicators = st.sidebar.multiselect(
        "Technical Indicators",
        ["RSI", "MACD", "Bollinger Bands", "Volume"],
        default=["RSI", "MACD"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Model:** Uses trained classifiers to predict next-day direction."
    )

    return ticker, period, show_indicators


# ===================================================================
# Fetch live data
# ===================================================================
@st.cache_data(ttl=300)
def fetch_live_data(ticker: str, period: str):
    if not ticker:
        return None, None, None
        
    df = None
    info = {}
    news = []
    
    # 1. Fetch live quotes/chart data via Twelve Data (highly reliable)
    api_key = os.environ.get("TWELVEDATA_API_KEY", "")
    if api_key:
        try:
            from twelvedata import TDClient
            td = TDClient(apikey=api_key)
            # Map Streamlit period to trading days (roughly)
            sizes = {"6mo": 130, "1y": 252, "2y": 504, "5y": 1260}
            size = sizes.get(period, 252)
            
            ts = td.time_series(
                symbol=ticker, 
                interval="1day", 
                outputsize=size, 
                timezone="America/New_York"
            )
            td_df = ts.as_pandas()
            if td_df is not None and not td_df.empty:
                td_df.index.name = "Date"
                td_df.index = pd.to_datetime(td_df.index)
                td_df = td_df.sort_index(ascending=True)
                col_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
                df = td_df.rename(columns={k: v for k, v in col_map.items() if k in td_df.columns})
        except Exception:
            pass
            
    # 2. Fetch Fundamentals & News via yfinance (and fallback for quotes if TD failed)
    try:
        stock = yf.Ticker(ticker)
        if df is None:
            df_yf = stock.history(period=period)
            if df_yf is not None and not df_yf.empty:
                df = df_yf
                
        info = stock.info if hasattr(stock, 'info') else {}
        news = stock.news if hasattr(stock, "news") else []
    except Exception:
        # Expected to fail on cloud due to Yahoo scraping blocks
        pass
        
    return df, info, news


# ===================================================================
# Fetch and Process Feature Data (Lazy Loading)
# ===================================================================
def get_feature_data(ticker, config):
    """
    Load feature data from disk OR compute it on the fly if missing.
    Ensures that custom tickers work automatically.
    """
    feature_file = PROJECT_ROOT / config["data"]["features_dir"] / f"{ticker}_features.parquet"

    if feature_file.exists():
        return pd.read_parquet(feature_file)

    with st.spinner(f"🚀 Running pipeline for {ticker}... (First time only)"):
        try:
            # 1. Fetch raw
            pipeline = DataPipeline(config)
            df_raw = pipeline.fetch(
                ticker,
                config["data"]["default_start_date"],
                config["data"]["default_end_date"]
            )
            if df_raw is None or df_raw.empty:
                return None

            # 2. Clean & Preprocess
            df_raw = pipeline.clean(df_raw, ticker)
            df_raw = pipeline.preprocess(df_raw, ticker)

            # 3. Engineer features
            feature_store = FeatureStore(config)
            df_feat = feature_store.engineer_features(df_raw, ticker)

            # 4. Save to feature store (so next time is instant)
            feature_store.save_features(df_feat, ticker)

            return df_feat
        except Exception as e:
            if "TWELVEDATA_API_KEY" in str(e):
                if not st.session_state.get("api_warned"):
                    st.warning("⚠️ **TWELVEDATA_API_KEY missing.** Cannot generate historical features for new tickers on-the-fly. Please select a pre-computed ticker or set your API key.")
                    st.session_state.api_warned = True
            else:
                st.error(f"Failed to process {ticker}: {e}")
            return None
def compute_display_indicators(df: pd.DataFrame, config: dict):
    cfg = config["features"]

    # RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(cfg["rsi_period"]).mean()
    avg_loss = loss.rolling(cfg["rsi_period"]).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df["Close"].ewm(span=cfg["macd_fast"], adjust=False).mean()
    ema_slow = df["Close"].ewm(span=cfg["macd_slow"], adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=cfg["macd_signal"], adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    sma = df["Close"].rolling(cfg["bollinger_window"]).mean()
    std = df["Close"].rolling(cfg["bollinger_window"]).std()
    df["BB_Upper"] = sma + cfg["bollinger_std"] * std
    df["BB_Middle"] = sma
    df["BB_Lower"] = sma - cfg["bollinger_std"] * std

    return df


# ===================================================================
# Charts
# ===================================================================
def plot_price_chart(df, ticker, show_indicators):
    n_rows = 1 + ("RSI" in show_indicators) + ("MACD" in show_indicators) + ("Volume" in show_indicators)
    heights = [0.5] + [0.17] * (n_rows - 1) if n_rows > 1 else [1.0]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=heights,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC",
        increasing_line_color="#34d399", decreasing_line_color="#f87171",
    ), row=1, col=1)

    # Bollinger Bands
    if "Bollinger Bands" in show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Upper"], name="BB Upper",
                                  line=dict(color="rgba(99,102,241,0.3)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["BB_Lower"], name="BB Lower",
                                  line=dict(color="rgba(99,102,241,0.3)", width=1),
                                  fill="tonexty", fillcolor="rgba(99,102,241,0.05)"), row=1, col=1)

    current_row = 2

    # RSI
    if "RSI" in show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI",
                                  line=dict(color="#818cf8", width=1.5)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(248,113,113,0.5)", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(52,211,153,0.5)", row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", row=current_row, col=1)
        current_row += 1

    # MACD
    if "MACD" in show_indicators:
        colors = ["#34d399" if v >= 0 else "#f87171" for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist",
                              marker_color=colors), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                  line=dict(color="#818cf8", width=1.5)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                                  line=dict(color="#f59e0b", width=1.5)), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)
        current_row += 1

    # Volume
    if "Volume" in show_indicators:
        colors = ["#34d399" if c >= o else "#f87171"
                  for c, o in zip(df["Close"].fillna(0), df["Open"].fillna(0))]
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                              marker_color=colors, opacity=0.7), row=current_row, col=1)
        fig.update_yaxes(title_text="Volume", row=current_row, col=1)

    fig.update_layout(
        title=f"{ticker} — Price & Indicators",
        template="plotly_dark",
        height=200 + 200 * n_rows,
        showlegend=False,
        xaxis_rangeslider_visible=False,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(family="Inter", color="#e2e8f0"),
    )
    return fig


# ===================================================================
# XGBoost Prediction helpers
# ===================================================================
@st.cache_data(ttl=3600)
def _backtest_xgboost(_model, ticker: str, config: dict) -> float:
    """
    Directional accuracy of XGBoost on the last 20% of the feature data
    for this ticker (time-ordered split — no leakage).
    Cached per ticker for 1 hour.
    """
    try:
        df_feat = get_feature_data(ticker, config)
        if df_feat is None or len(df_feat) < 60:
            return None
        exclude = ["Open", "High", "Low", "Close", "Adj Close", "Volume",
                   "target_direction", "ticker", "log_return"]
        fc = [c for c in df_feat.columns if c not in exclude]
        X, y = df_feat[fc].values, df_feat["target_direction"].values
        split = int(len(X) * 0.80)
        X_test, y_test = X[split:], y[split:]
        if len(X_test) < 10:
            return None
        from sklearn.metrics import accuracy_score
        preds = _model.predict(X_test)
        return round(accuracy_score(y_test, preds) * 100, 1)
    except Exception:
        return None



def _predict_with_model(ticker: str, config: dict, model) -> tuple:
    """
    Run a classifier on the latest feature row.
    Returns (direction, confidence) where confidence is max(p_up, p_down).
    """
    try:
        df_feat = get_feature_data(ticker, config)
        if df_feat is None:
            return None, None
        exclude = ["Open", "High", "Low", "Close", "Adj Close", "Volume",
                   "target_direction", "ticker", "log_return"]
        fc = [c for c in df_feat.columns if c not in exclude]
        latest = df_feat[fc].iloc[-1:].values
        pred = int(model.predict(latest)[0])
        conf = 0.5
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(latest)[0]
            conf = float(max(prob))
        return ("UP" if pred == 1 else "DOWN"), conf
    except Exception:
        return None, None


# Display name → dict key mapping
_MODEL_DISPLAY = {
    "XGBoost": "XGBoost",
    "RandomForest": "Random Forest",
    "LogisticRegression": "Logistic Regression",
}


def final_prediction_card(ticker: str, config: dict, models: dict):
    """
    Hero card: dynamically picks the best-backtested classifier for this
    specific ticker, then shows its next-day prediction and win-rate.
    """
    classifier_models = {k: v for k, v in models.items() if k != "isolation_forest"}
    if not classifier_models:
        st.warning("No models loaded.")
        return

    # Re-use the same cached backtest so we don't duplicate work
    all_acc = _backtest_all_models(
        tuple(sorted(classifier_models.keys())), classifier_models, ticker, config
    )
    best_key = all_acc.get("_best", "XGBoost")
    model = classifier_models.get(best_key)
    if model is None:
        model = classifier_models.get("XGBoost")
        best_key = "XGBoost"

    direction, conf = _predict_with_model(ticker, config, model)
    if direction is None:
        return

    acc = all_acc.get(best_key)
    acc_str = f"{acc:.1f}%" if acc is not None else "N/A"
    display_name = _MODEL_DISPLAY.get(best_key, best_key)

    is_up = direction == "UP"
    icon   = "\u2191" if is_up else "\u2193"
    dclass = "fp-direction-up" if is_up else "fp-direction-down"

    html = (
        '<div class="fp-card">'
        '<div>'
        f'<div class="fp-label">Final Prediction &middot; {display_name}</div>'
        f'<div class="{dclass}">{icon} {direction}</div>'
        f'<div class="fp-sub">{conf*100:.0f}% model confidence &middot; next-day direction</div>'
        '</div>'
        '<div class="fp-divider"></div>'
        '<div>'
        '<div class="fp-label">Backtested Win-Rate</div>'
        f'<div class="fp-winrate">{acc_str}</div>'
        '<div class="fp-sub">held-out last 20% of data</div>'
        '</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ===================================================================
# Predictions Tab — XGBoost summary
# ===================================================================
@st.cache_data(ttl=3600)
def _backtest_all_models(_models_dict_keys: tuple, _models: dict, ticker: str, config: dict) -> dict:
    """
    Compute per-ticker held-out accuracy for all loaded classifiers.

    Model training context:
        All classifiers (RandomForest, XGBoost, LogisticRegression) were trained
        ONCE on the pooled feature data from all 5 default tickers using a
        64% train / 16% val / 20% test time-ordered split (see src/train.py).
        There is no per-ticker retraining — a single global model is loaded.

    What this function does:
        For a given ticker, it takes that ticker's feature parquet, applies an
        80/20 time-ordered split, and evaluates the pre-trained (pooled) model
        on the last 20% of THAT ticker's data. This gives a ticker-specific
        accuracy estimate without any separate fitting.

    Caching:
        Results are cached per (model keys, ticker) tuple for 1 hour (ttl=3600).
        This means backtest accuracy shown in the UI may be up to 1 hour stale
        if the underlying feature file changes during a session.
    """
    results = {}
    try:
        df_feat = get_feature_data(ticker, config)
        if df_feat is None or len(df_feat) < 60:
            return results
        exclude = ["Open", "High", "Low", "Close", "Adj Close", "Volume",
                   "target_direction", "ticker", "log_return"]
        fc = [c for c in df_feat.columns if c not in exclude]
        X, y = df_feat[fc].values, df_feat["target_direction"].values
        split = int(len(X) * 0.80)
        X_test, y_test = X[split:], y[split:]
        if len(X_test) < 10:
            return results
        from sklearn.metrics import accuracy_score, recall_score
        for name, m in _models.items():
            preds = m.predict(X_test)
            results[name] = round(accuracy_score(y_test, preds) * 100, 1)

            results[f"{name}_DOWN_recall"] = round(recall_score(y_test, preds, pos_label=0) * 100, 0) if sum(y_test == 0) > 0 else 0.0
            results[f"{name}_UP_recall"] = round(recall_score(y_test, preds, pos_label=1) * 100, 0) if sum(y_test == 1) > 0 else 0.0

        baseline_preds = np.ones_like(y_test)
        results["Baseline"] = round(accuracy_score(y_test, baseline_preds) * 100, 1)

        # Record the best classifier key
        classifier_accs = {k: v for k, v in results.items()
                           if k in _models and isinstance(v, float)}
        if classifier_accs:
            results["_best"] = max(classifier_accs, key=classifier_accs.get)
    except Exception:
        pass
    return results

def prediction_panel(ticker, config, models):
    """
    Models Evaluated table — best model badge is computed dynamically per
    ticker from the backtest results. Hero card above already shows the headline.
    """
    classifier_models = {k: v for k, v in models.items() if k != "isolation_forest"}
    if not classifier_models:
        st.info("No models loaded.")
        return

    # Quick data check using any available model
    any_model = next(iter(classifier_models.values()))
    direction, _ = _predict_with_model(ticker, config, any_model)
    if direction is None:
        st.warning("Could not load feature data.")
        return

    st.markdown('<div style="font-size: 13px; font-weight: 600; color: #94a3b8; margin-bottom: 8px;">Models Evaluated</div>', unsafe_allow_html=True)
    st.markdown('<div style="border-top: 1px solid #252b45; margin-bottom: 12px;"></div>', unsafe_allow_html=True)

    all_acc = _backtest_all_models(tuple(sorted(classifier_models.keys())), classifier_models, ticker, config)

    xgb_acc = all_acc.get("XGBoost", "N/A")
    rf_acc  = all_acc.get("RandomForest", "N/A")
    lr_acc  = all_acc.get("LogisticRegression", "N/A")
    baseline_acc    = all_acc.get("Baseline", "N/A")
    best_key = all_acc.get("_best", "XGBoost")
    best_down_recall = int(all_acc.get(f"{best_key}_DOWN_recall", 0))
    best_up_recall   = int(all_acc.get(f"{best_key}_UP_recall", 0))
    best_display_name = _MODEL_DISPLAY.get(best_key, best_key)

    def _row(label, acc, key):
        acc_display = f"{acc}%" if acc != "N/A" else "N/A"
        if key == best_key:
            return (
                f'<tr>'
                f'<td style="padding: 4px 0; color: #e2e8f0;">{label}</td>'
                f'<td style="padding: 4px 0; text-align: right; color: #e2e8f0;">{acc_display}</td>'
                f'<td style="padding: 4px 0 4px 12px; color: #34d399; font-weight: 600;">&#10003; Best</td>'
                f'</tr>'
            )
        suffix = ' <span style="color:#64748b;font-size:11px;">(evaluated)</span>' if key == "LogisticRegression" else ''
        return (
            f'<tr>'
            f'<td style="padding: 4px 0;">{label}{suffix}</td>'
            f'<td style="padding: 4px 0; text-align: right;">{acc_display}</td>'
            f'<td style="padding: 4px 0 4px 12px;"></td>'
            f'</tr>'
        )

    rows = (
        _row("XGBoost",             xgb_acc, "XGBoost") +
        _row("Random Forest",       rf_acc,  "RandomForest") +
        _row("Logistic Regression", lr_acc,  "LogisticRegression")
    )

    table_html = f"""
    <table style="width: 100%; max-width: 480px; font-size: 13px; color: #94a3b8; text-align: left; border-collapse: collapse;">
        {rows}
        <tr>
            <td style="padding: 8px 0 4px 0; color: #475569; border-top: 1px solid #1e2438;">Naive baseline (always UP)</td>
            <td style="padding: 8px 0 4px 0; text-align: right; color: #475569;">{baseline_acc}%</td>
            <td style="padding: 8px 0 4px 12px; color: #475569;">0% DOWN recall</td>
        </tr>
    </table>
    <div style="font-size: 11px; color: #64748b; margin-top: 10px; line-height: 1.6; max-width: 480px;">
        {best_display_name} catches <strong style="color: #94a3b8;">{best_down_recall}% of DOWN days</strong>
        vs. {best_up_recall}% of UP days &mdash; the naive baseline catches zero.
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ===================================================================
# Fundamentals Panel
# ===================================================================
def fundamentals_panel(info):
    st.markdown('<p class="section-header">📊 Fundamentals</p>', unsafe_allow_html=True)

    # ── Disclaimer ──────────────────────────────────────────────────────
    st.info(
        "ℹ️ **Display only — not a model input.** "
        "These fundamental metrics (P/E, EPS, Market Cap, etc.) are fetched "
        "from Yahoo Finance for reference. They are **not** features in the "
        "prediction models, which rely solely on technical indicators and "
        "lagged price returns."
    )

    metrics = {
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "Market Cap": info.get("marketCap", "N/A"),
        "Profit Margin": info.get("profitMargins", "N/A"),
        "Revenue Growth": info.get("revenueGrowth", "N/A"),
        "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
        "52W Low": info.get("fiftyTwoWeekLow", "N/A"),
        "Avg Volume": info.get("averageVolume", "N/A"),
    }

    cols = st.columns(4)
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i % 4]:
            if isinstance(value, (int, float)):
                if label == "Market Cap":
                    if value >= 1e12:
                        display = f"${value/1e12:.2f}T"
                    elif value >= 1e9:
                        display = f"${value/1e9:.2f}B"
                    else:
                        display = f"${value/1e6:.0f}M"
                elif label in ["Profit Margin", "Revenue Growth"]:
                    display = f"{value*100:.1f}%"
                elif label == "Avg Volume":
                    display = f"{value/1e6:.1f}M"
                else:
                    display = f"{value:.2f}"
            else:
                display = str(value)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{display}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


# ===================================================================
# Anomaly Timeline
# ===================================================================
def anomaly_panel(ticker, config, models):
    if "isolation_forest" not in models:
        return

    st.markdown('<p class="section-header">🔍 Anomaly Detection</p>', unsafe_allow_html=True)

    df = get_feature_data(ticker, config)

    if df is None:
        st.info(f"No feature data for {ticker}. Check the ticker symbol.")
        return

    anomaly_features = ["volume_ratio", "price_change_pct"]
    anomaly_cols = [c for c in anomaly_features if c in df.columns]
    if not anomaly_cols:
        return

    iso = models["isolation_forest"]
    preds = iso.predict(df[anomaly_cols].values)
    df["anomaly"] = preds
    anomalies = df[df["anomaly"] == -1]

    # — Today's status pill (was previously orphaned in the Predictions tab) —
    today_pred = iso.predict(df[anomaly_cols].iloc[-1:].values)[0]
    if today_pred == -1:
        st.error("⚠️ **Today: ANOMALY detected** — unusual volume or price movement flagged")
    else:
        st.success("✅ **Today: Normal** — no anomalous activity detected")

    if len(anomalies) == 0:
        st.success("No anomalies detected in this period.")
        return

    st.warning(f"Found {len(anomalies)} anomalous day(s)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Close",
        line=dict(color="#818cf8", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=anomalies.index, y=anomalies["Close"], name="Anomaly",
        mode="markers", marker=dict(color="#f87171", size=8, symbol="x"),
    ))
    fig.update_layout(
        template="plotly_dark", height=300,
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(family="Inter", color="#e2e8f0"),
        title="Anomalous Trading Days",
    )
    st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# Sentiment Panel
# ===================================================================
def sentiment_panel(news):
    st.markdown('<p class="section-header">📰 News Sentiment</p>', unsafe_allow_html=True)

    # ── Disclaimer ──────────────────────────────────────────────────────
    st.warning(
        "⚠️ **Not used in prediction.** "
        "Sentiment scores below are computed from live Yahoo Finance headlines "
        "using VADER and are shown for informational context only. "
        "Sentiment was removed from the model pipeline to prevent look-ahead "
        "leakage (applying a current snapshot retroactively to historical dates). "
        "The prediction models use only technical indicators and lagged price returns."
    )

    if not news or len(news) == 0:
        st.info("No recent news available.")
        return

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()

        rows = []
        for article in news[:10]:
            content = article.get("content", article)
            title = content.get("title", "")
            if title:
                vs = analyzer.polarity_scores(title)
                rows.append({
                    "Headline": title[:80] + ("..." if len(title) > 80 else ""),
                    "Sentiment": vs["compound"],
                    "Signal": "🟢 Positive" if vs["compound"] > 0.05 else ("🔴 Negative" if vs["compound"] < -0.05 else "⚪ Neutral"),
                })

        if rows:
            avg_sent = np.mean([r["Sentiment"] for r in rows])
            signal_color = "metric-up" if avg_sent > 0 else "metric-down"

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value {signal_color}">{avg_sent:.3f}</div>
                <div class="metric-label">Average Sentiment Score</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("")

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    except ImportError:
        st.warning("vaderSentiment not installed.")


# ===================================================================
# Monitoring Panel
# ===================================================================
def monitoring_panel(ticker):
    st.markdown('<p class="section-header">📡 Monitoring</p>', unsafe_allow_html=True)

    log_path = PROJECT_ROOT / "logs" / "predictions.jsonl"
    if not log_path.exists():
        st.info("No prediction logs yet. Make some API requests to generate data.")
        return

    logs = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(json.loads(line))

    if not logs:
        st.info("Log file is empty.")
        return

    df_logs = pd.DataFrame(logs)
    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Requests", len(df_logs))
    with col2:
        up_pct = df_logs["prediction"].mean() * 100 if "prediction" in df_logs.columns else 0
        st.metric("UP Predictions", f"{up_pct:.1f}%")
    with col3:
        avg_conf = df_logs["confidence"].mean() * 100 if "confidence" in df_logs.columns else 0
        st.metric("Avg Confidence", f"{avg_conf:.1f}%")

    # Prediction distribution over time
    if len(df_logs) > 5:
        fig = px.histogram(df_logs, x="prediction", nbins=2,
                           title="Prediction Distribution",
                           template="plotly_dark",
                           color_discrete_sequence=["#818cf8"])
        fig.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(family="Inter", color="#e2e8f0"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Drift detection results
    drift_panel(ticker)


# ===================================================================
# Drift Panel
# ===================================================================
def drift_panel(ticker):
    st.markdown('<p class="section-header">🔄 Drift Detection</p>', unsafe_allow_html=True)

    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from monitoring.drift import DriftDetector

        config = load_config()
        detector = DriftDetector(config)

        detector = DriftDetector(config)

        # Use smart loader
        df = get_feature_data(ticker, config)

        if df is None:
            st.info(f"No feature data available for {ticker} drift analysis.")
            return

        report = detector.check_feature_drift(df)
        retrain = detector.should_retrain(report)

        if retrain:
            st.error("⚠️ Retraining recommended!")
        else:
            st.success("✅ No drift detected — model is stable.")

        # Show KS stats
        if "ks_results" in report:
            ks_df = pd.DataFrame(report["ks_results"])
            st.dataframe(ks_df, use_container_width=True, hide_index=True)

    except ImportError:
        st.info("Drift detection module not yet available.")
    except Exception as e:
        st.info(f"Drift analysis: {e}")


# ===================================================================
# Main App
# ===================================================================
def main():
    config = load_config()
    models = load_models(config)

    ticker, period, show_indicators = sidebar()

    if not ticker:
        st.info("👋 Welcome! Please enter a ticker symbol in the sidebar (e.g., AAPL, NVDA) to begin analysis.")
        return

    # Header
    st.markdown(f"# 📈 {ticker} Analysis")

    try:
        # Fetch live data (may fail on cloud servers due to Yahoo rate-limiting)
        with st.spinner(f"Fetching {ticker} data..."):
            df, info, news = fetch_live_data(ticker, period)

        has_live_data = df is not None and not df.empty

        if has_live_data:
            # Top metrics row from live data
            latest_close = df["Close"].iloc[-1]
            prev_close = df["Close"].iloc[-2] if len(df) > 1 else latest_close
            change = latest_close - prev_close
            change_pct = (change / prev_close) * 100

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Close", f"${latest_close:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")
            col2.metric("High", f"${df['High'].iloc[-1]:.2f}")
            col3.metric("Low", f"${df['Low'].iloc[-1]:.2f}")
            col4.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")

            # Compute indicators & show price chart
            df = compute_display_indicators(df, config)
            fig = plot_price_chart(df, ticker, show_indicators)
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Fallback: use pre-computed feature data for metrics
            st.info("ℹ️ **Live quotes unavailable in this environment — showing ML predictions from cached historical data.**")
            df_feat = get_feature_data(ticker, config)
            if df_feat is not None and "Close" in df_feat.columns:
                latest_close = df_feat["Close"].iloc[-1]
                prev_close = df_feat["Close"].iloc[-2] if len(df_feat) > 1 else latest_close
                change = latest_close - prev_close
                change_pct = (change / prev_close) * 100

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Close", f"${latest_close:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")
                col2.metric("High", f"${df_feat['High'].iloc[-1]:.2f}")
                col3.metric("Low", f"${df_feat['Low'].iloc[-1]:.2f}")
                col4.metric("Volume", f"{df_feat['Volume'].iloc[-1]:,.0f}")
            elif df_feat is None:
                st.error(f"No data available for {ticker}. Try one of the default tickers: AAPL, MSFT, GOOGL, TSLA, AMZN")
                return

        # ── Final Prediction Card (always visible, above tabs) ──────
        st.markdown("---")
        final_prediction_card(ticker, config, models)

        # Tabs — always show (work with pre-computed data)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🤖 Predictions", "📊 Fundamentals", "🔍 Anomalies",
            "📰 Sentiment", "📡 Monitoring"
        ])

        with tab1:
            prediction_panel(ticker, config, models)

        with tab2:
            if info:
                fundamentals_panel(info)
            else:
                st.info("ℹ️ **Fundamentals unavailable.** Live financial data access is restricted in this deployment.")

        with tab3:
            anomaly_panel(ticker, config, models)

        with tab4:
            if news:
                sentiment_panel(news)
            else:
                st.info("ℹ️ **Sentiment analysis requires live news access, unavailable in this deployment.**")

        with tab5:
            monitoring_panel(ticker)

    except Exception as e:
        st.error(f"Error: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
