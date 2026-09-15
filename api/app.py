"""
FastAPI App — REST API with 3 endpoints + Pydantic validation.

Endpoints:
    POST /analyze               — Full pipeline analysis for a ticker
    GET  /fundamentals/{ticker} — Live fundamentals from TwelveData
    GET  /anomalies/{ticker}    — Isolation Forest anomaly detection

Usage:
    uvicorn api.app:app --reload
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from twelvedata import TDClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import re

# Add project root to path (Rule 1: import same FeatureStore)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_store import FeatureStore
from src.data_pipeline import DataPipeline

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ===================================================================
# Load config + models at startup
# ===================================================================
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

MODELS_DIR = PROJECT_ROOT / CONFIG["training"]["models_dir"]
MODELS = {}
for name in ["RandomForest", "XGBoost", "LogisticRegression", "isolation_forest"]:
    path = MODELS_DIR / f"{name}.pkl"
    if path.exists():
        MODELS[name] = joblib.load(path)
        logger.info(f"Loaded model: {name}")

FEATURE_STORE = FeatureStore(CONFIG)
PIPELINE = DataPipeline(CONFIG)

# TwelveData client for fundamentals / statistics
_TD_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
_TD_CLIENT: Optional[TDClient] = TDClient(apikey=_TD_API_KEY) if _TD_API_KEY else None

LOG_PATH = PROJECT_ROOT / CONFIG["monitoring"]["log_file"]
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Pydantic Models
# ===================================================================
class AnalyzeRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (1-5 uppercase letters)")
    start_date: str = Field(..., description="Start date YYYY-MM-DD")
    end_date: str = Field(..., description="End date YYYY-MM-DD")

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v):
        v = v.upper().strip()
        if not re.match(r"^[A-Z]{1,5}$", v):
            raise ValueError("Ticker must be 1-5 uppercase letters")
        return v

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, v, info):
        end = datetime.strptime(v, "%Y-%m-%d").date()
        if end > date.today():
            raise ValueError("End date cannot be in the future")
        if "start_date" in info.data:
            start = datetime.strptime(info.data["start_date"], "%Y-%m-%d").date()
            if end <= start:
                raise ValueError("End date must be after start date")
        return v


class TechnicalSignals(BaseModel):
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    bollinger_position: Optional[str] = None
    volume_ratio: Optional[float] = None
    volatility_20d: Optional[float] = None


class Fundamentals(BaseModel):
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    market_cap: Optional[str] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None


class AnalyzeResponse(BaseModel):
    model_config = {"protected_namespaces": ()}  # allow 'model_used' field name

    ticker: str
    direction_prediction: str
    confidence: float
    anomaly_flag: bool
    sentiment_score: float
    technical_signals: TechnicalSignals
    fundamentals: Fundamentals
    model_used: str
    analysis_date: str


class AnomalyItem(BaseModel):
    date: str
    close_price: float
    volume: float
    volume_ratio: float
    price_change_pct: float


class AnomaliesResponse(BaseModel):
    ticker: str
    total_anomalies: int
    anomalies: list[AnomalyItem]


# ===================================================================
# Logging helper (Rule 3: log before you need it)
# ===================================================================
def log_prediction(ticker: str, features: dict, prediction: int,
                   confidence: float, latency_ms: float):
    """Write one JSON line to predictions.jsonl."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker": ticker,
        "features": features,
        "prediction": prediction,
        "confidence": confidence,
        "latency_ms": round(latency_ms, 2),
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ===================================================================
# FastAPI App
# ===================================================================
app = FastAPI(
    title="TradeVision AI API",
    description="End-to-end ML system for stock analysis: predictions, fundamentals, anomalies",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
# POST /analyze
# ===================================================================
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Full pipeline: fetch → feature engineering → all model predictions.
    Returns technical signals, sentiment, anomaly flag, and direction prediction.
    """
    start_time = time.time()
    ticker = request.ticker

    # --- Fetch & preprocess ---
    try:
        df = PIPELINE.fetch(ticker, request.start_date, request.end_date)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Data fetch failed for {ticker}: {str(e)}")

    df = PIPELINE.clean(df, ticker)
    if df is None:
        raise HTTPException(status_code=422, detail=f"Insufficient data for {ticker} — fewer than 30 usable days")

    if len(df) < 30:
        raise HTTPException(status_code=422, detail=f"Only {len(df)} days of data — need at least 30")

    df = PIPELINE.preprocess(df, ticker)

    # --- Feature engineering (Rule 1: same FeatureStore) ---
    try:
        df_feat = FEATURE_STORE.engineer_features(df, ticker)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {str(e)}")

    # --- Get feature vector ---
    exclude = ["Open", "High", "Low", "Close", "Adj Close", "Volume",
               "target_direction", "ticker", "log_return"]
    feature_cols = [c for c in df_feat.columns if c not in exclude]
    latest = df_feat[feature_cols].iloc[-1:]

    # --- Direction prediction (best classifier) ---
    best_model_name = "XGBoost"
    for name in ["XGBoost", "RandomForest", "LogisticRegression"]:
        if name in MODELS:
            best_model_name = name
            break

    model = MODELS[best_model_name]
    pred = model.predict(latest.values)[0]
    direction = "UP" if pred == 1 else "DOWN"

    confidence = 0.5
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(latest.values)[0]
        confidence = float(max(prob))

    # --- Anomaly detection ---
    anomaly_flag = False
    if "isolation_forest" in MODELS:
        anomaly_features = ["volume_ratio", "price_change_pct"]
        anomaly_cols = [c for c in anomaly_features if c in df_feat.columns]
        if anomaly_cols:
            anomaly_pred = MODELS["isolation_forest"].predict(
                df_feat[anomaly_cols].iloc[-1:].values
            )[0]
            anomaly_flag = anomaly_pred == -1

    # --- Technical signals ---
    last_row = df_feat.iloc[-1]
    bb_pos_val = last_row.get("bb_position", 0.5)
    if bb_pos_val > 0.8:
        bb_label = "upper"
    elif bb_pos_val < 0.2:
        bb_label = "lower"
    else:
        bb_label = "middle"

    signals = TechnicalSignals(
        rsi_14=round(float(last_row.get("rsi_14", 0)), 2),
        macd=round(float(last_row.get("macd", 0)), 4),
        macd_signal=round(float(last_row.get("macd_signal", 0)), 4),
        bollinger_position=bb_label,
        volume_ratio=round(float(last_row.get("volume_ratio", 1)), 4),
        volatility_20d=round(float(last_row.get("volatility_20d", 0)), 6),
    )

    # --- Fundamentals (TwelveData statistics) ---
    funds = _fetch_fundamentals(ticker)

    # --- Sentiment ---
    sentiment = float(last_row.get("sentiment_score", 0))

    # --- Log prediction (Rule 3) ---
    latency_ms = (time.time() - start_time) * 1000
    log_prediction(
        ticker=ticker,
        features={c: float(latest[c].values[0]) for c in feature_cols[:5]},
        prediction=int(pred),
        confidence=confidence,
        latency_ms=latency_ms,
    )

    return AnalyzeResponse(
        ticker=ticker,
        direction_prediction=direction,
        confidence=round(confidence, 4),
        anomaly_flag=anomaly_flag,
        sentiment_score=round(sentiment, 4),
        technical_signals=signals,
        fundamentals=funds,
        model_used=best_model_name,
        analysis_date=date.today().isoformat(),
    )


# ===================================================================
# Helper: fetch fundamentals via TwelveData statistics endpoint
# ===================================================================
def _fetch_fundamentals(ticker: str) -> "Fundamentals":
    """
    Fetch key fundamentals from TwelveData /statistics endpoint.
    Returns an empty Fundamentals() on any failure (free-tier may not include this).

    Actual TwelveData statistics JSON structure (verified):
      valuations_metrics.trailing_pe / market_capitalization
      financials.profit_margin
      financials.income_statement.diluted_eps_ttm / quarterly_revenue_growth
      stock_price_summary.fifty_two_week_high / fifty_two_week_low
    """
    if _TD_CLIENT is None:
        logger.warning("TwelveData client not initialised — TWELVEDATA_API_KEY missing")
        return Fundamentals()

    try:
        stats = _TD_CLIENT.get_statistics(symbol=ticker).as_json()

        # --- Valuation metrics ---
        valuation = stats.get("valuations_metrics", {})
        pe_ratio = _safe_float(valuation.get("trailing_pe"))
        mc = _safe_float(valuation.get("market_capitalization"))
        if mc and mc >= 1e12:
            mc_str = f"${mc/1e12:.2f}T"
        elif mc and mc >= 1e9:
            mc_str = f"${mc/1e9:.2f}B"
        elif mc:
            mc_str = f"${mc/1e6:.0f}M"
        else:
            mc_str = "N/A"

        # --- Financials ---
        financials = stats.get("financials", {})
        profit_margin = _safe_float(financials.get("profit_margin"))
        income = financials.get("income_statement", {})
        eps = _safe_float(income.get("diluted_eps_ttm"))
        revenue_growth = _safe_float(income.get("quarterly_revenue_growth"))

        # --- 52-week price summary ---
        price_summary = stats.get("stock_price_summary", {})
        fifty_two_wk_high = _safe_float(price_summary.get("fifty_two_week_high"))
        fifty_two_wk_low = _safe_float(price_summary.get("fifty_two_week_low"))

        return Fundamentals(
            pe_ratio=pe_ratio,
            eps=eps,
            market_cap=mc_str,
            profit_margin=profit_margin,
            revenue_growth=revenue_growth,
            fifty_two_week_high=fifty_two_wk_high,
            fifty_two_week_low=fifty_two_wk_low,
        )
    except Exception as exc:
        logger.warning(f"TwelveData fundamentals failed for {ticker}: {exc} — returning empty")
        return Fundamentals()


def _safe_float(value) -> Optional[float]:
    """Safely convert a value to float, returning None on failure."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# ===================================================================
# GET /fundamentals/{ticker}
# ===================================================================
@app.get("/fundamentals/{ticker}", response_model=Fundamentals)
def get_fundamentals(ticker: str):
    """Return live fundamentals from TwelveData statistics endpoint."""
    ticker = ticker.upper().strip()
    if not re.match(r"^[A-Z]{1,5}$", ticker):
        raise HTTPException(status_code=400, detail="Ticker must be 1-5 uppercase letters")

    return _fetch_fundamentals(ticker)


# ===================================================================
# GET /anomalies/{ticker}
# ===================================================================
@app.get("/anomalies/{ticker}", response_model=AnomaliesResponse)
def get_anomalies(ticker: str):
    """
    Run Isolation Forest on recent data and return flagged dates.
    """
    ticker = ticker.upper().strip()
    if not re.match(r"^[A-Z]{1,5}$", ticker):
        raise HTTPException(status_code=400, detail="Ticker must be 1-5 uppercase letters")

    if "isolation_forest" not in MODELS:
        raise HTTPException(status_code=503, detail="Anomaly model not loaded")

    # Try to use pre-computed features
    feature_file = PROJECT_ROOT / CONFIG["data"]["features_dir"] / f"{ticker}_features.parquet"

    if feature_file.exists():
        df = pd.read_parquet(feature_file)
    else:
        # Compute on the fly
        try:
            raw = PIPELINE.fetch(ticker, CONFIG["data"]["default_start_date"],
                                 CONFIG["data"]["default_end_date"])
            raw = PIPELINE.clean(raw, ticker)
            if raw is None:
                raise HTTPException(status_code=422, detail="Insufficient data")
            raw = PIPELINE.preprocess(raw, ticker)
            df = FEATURE_STORE.engineer_features(raw, ticker)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Pipeline failed: {str(e)}")

    anomaly_features = ["volume_ratio", "price_change_pct"]
    anomaly_cols = [c for c in anomaly_features if c in df.columns]
    if not anomaly_cols:
        raise HTTPException(status_code=422, detail="Required features not available")

    iso = MODELS["isolation_forest"]
    preds = iso.predict(df[anomaly_cols].values)
    df["anomaly"] = preds

    anomalies = df[df["anomaly"] == -1].sort_index(ascending=False)

    items = []
    for idx, row in anomalies.head(20).iterrows():
        items.append(AnomalyItem(
            date=str(idx.date()) if hasattr(idx, "date") else str(idx),
            close_price=round(float(row.get("Close", 0)), 2),
            volume=float(row.get("Volume", 0)),
            volume_ratio=round(float(row.get("volume_ratio", 0)), 4),
            price_change_pct=round(float(row.get("price_change_pct", 0)), 6),
        ))

    return AnomaliesResponse(
        ticker=ticker,
        total_anomalies=len(anomalies),
        anomalies=items,
    )


# ===================================================================
# Health check
# ===================================================================
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(MODELS.keys()),
        "timestamp": datetime.now().isoformat(),
    }
