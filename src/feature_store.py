

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
# FeatureStore Class
# ===================================================================
class FeatureStore:
    """
    Single feature pipeline used in both training and serving.

    All parameters come from config.yaml (Rule 2).

    Features:
        - Technical indicators: RSI, MACD, Bollinger Bands, volatility, volume ratio
        - Sentiment: VADER on yfinance news headlines
        - Lag features: 1-day, 3-day, 5-day lagged returns
        - Target: next-day direction (1 = up, 0 = down)
    """

    def __init__(self, config: dict):
        self.cfg = config["features"]
        self.data_cfg = config["data"]
        self.features_dir = Path(self.data_cfg["features_dir"])
        self.raw_dir = Path(self.data_cfg["raw_data_dir"])
        self.features_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # MAIN METHOD: engineer_features()
    # ---------------------------------------------------------------
    def engineer_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Compute all features for a single ticker DataFrame.

        Leakage rule: For day T, only use data available before market close on day T.
        The target (next-day direction) uses day T+1. These never mix.

        Args:
            df: DataFrame with OHLCV + log_return + volume_normalised columns
            ticker: Ticker symbol (for logging)

        Returns:
            DataFrame with all features + target column
        """
        logger.info(f"  Engineering features for {ticker} ...")

        feat = df.copy()

        # --- Technical Indicators ---
        feat = self._add_rsi(feat)
        feat = self._add_macd(feat)
        feat = self._add_bollinger_bands(feat)
        feat = self._add_volatility(feat)
        feat = self._add_volume_ratio(feat)

        # --- Sentiment ---
        feat = self._add_sentiment(feat, ticker)

        # --- Lag Features ---
        feat = self._add_lag_features(feat)

        # --- Target: next-day direction ---
        # 1 if tomorrow's close > today's close, else 0
        feat["target_direction"] = (feat["Close"].shift(-1) > feat["Close"]).astype(int)

        # --- Daily price change magnitude (for anomaly model) ---
        feat["price_change_pct"] = feat["Close"].pct_change().abs()

        # Drop rows with NaN from rolling calculations and the last row (no target)
        feat = feat.dropna()

        logger.info(
            f"  ✓ {ticker}: {len(feat)} rows, {len(feat.columns)} features"
        )
        return feat

    # ---------------------------------------------------------------
    # SAVE METHOD: save_features()
    # ---------------------------------------------------------------
    def save_features(self, df: pd.DataFrame, ticker: str) -> Path:
        """Save feature DataFrame as a parquet file in the feature store."""
        filepath = self.features_dir / f"{ticker}_features.parquet"
        df.to_parquet(filepath)
        logger.info(f"  ✓ Saved → {filepath}")
        return filepath

    # ---------------------------------------------------------------
    # LOAD METHOD: load latest raw data for a ticker
    # ---------------------------------------------------------------
    def load_latest_raw(self, ticker: str) -> pd.DataFrame:
        """Load the most recent raw parquet for a ticker from data/raw/."""
        files = sorted(self.raw_dir.glob(f"{ticker}_*.parquet"), reverse=True)
        if not files:
            raise FileNotFoundError(
                f"No raw data found for {ticker} in {self.raw_dir}"
            )
        latest = files[0]
        logger.info(f"  Loading {latest.name}")
        return pd.read_parquet(latest)

    # ===================================================================
    # TECHNICAL INDICATORS (private methods)
    # ===================================================================

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RSI (Relative Strength Index) — measures overbought/oversold.
        RSI > 70 = overbought, RSI < 30 = oversold.
        """
        period = self.cfg["rsi_period"]
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence) — momentum changes.
        MACD line, signal line, and histogram.
        """
        fast = self.cfg["macd_fast"]
        slow = self.cfg["macd_slow"]
        signal = self.cfg["macd_signal"]

        ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bollinger Bands — volatility and price extremes.
        Middle band = SMA, upper/lower = SMA ± N std.
        """
        window = self.cfg["bollinger_window"]
        n_std = self.cfg["bollinger_std"]

        sma = df["Close"].rolling(window=window).mean()
        std = df["Close"].rolling(window=window).std()

        df["bb_upper"] = sma + (n_std * std)
        df["bb_middle"] = sma
        df["bb_lower"] = sma - (n_std * std)

        # Position: where is price relative to the bands? (0 = lower, 1 = upper)
        df["bb_position"] = (df["Close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )
        return df

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling volatility — 20-day std of log returns. Risk signal."""
        window = self.cfg["volatility_window"]
        df[f"volatility_{window}d"] = df["log_return"].rolling(window=window).std()
        return df

    def _add_volume_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume ratio — today's volume vs 20-day average. Flags unusual activity."""
        window = self.cfg["volume_ratio_window"]
        rolling_mean = df["Volume"].rolling(window=window).mean()
        df["volume_ratio"] = df["Volume"] / rolling_mean
        return df

    # ===================================================================
    # SENTIMENT
    # ===================================================================

    def _add_sentiment(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Daily sentiment score from yfinance news headlines + VADER.

        If no news available for a day, fills with neutral (0.0).
        """
        logger.info(f"    Computing sentiment for {ticker} ...")

        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            import yfinance as yf

            analyzer = SentimentIntensityAnalyzer()
            stock = yf.Ticker(ticker)

            # Get news headlines
            news = stock.news if hasattr(stock, "news") else []

            if news and len(news) > 0:
                # Compute average sentiment from available headlines
                scores = []
                for article in news:
                    content = article.get("content", article)
                    title = content.get("title", "")
                    if title:
                        vs = analyzer.polarity_scores(title)
                        scores.append(vs["compound"])

                avg_sentiment = np.mean(scores) if scores else 0.0
                logger.info(
                    f"    Found {len(scores)} headlines, avg sentiment: {avg_sentiment:.3f}"
                )
            else:
                avg_sentiment = 0.0
                logger.info("    No news found, using neutral sentiment")

            # Apply same sentiment to all rows (news is a snapshot, not historical)
            df["sentiment_score"] = avg_sentiment

        except ImportError:
            logger.warning(
                "    vaderSentiment not installed — filling sentiment with 0.0"
            )
            df["sentiment_score"] = self.cfg["sentiment_neutral_fill"]
        except Exception as e:
            logger.warning(f"    Sentiment computation failed: {e} — filling with 0.0")
            df["sentiment_score"] = self.cfg["sentiment_neutral_fill"]

        return df

    # ===================================================================
    # LAG FEATURES
    # ===================================================================

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lagged returns — gives the model memory of recent price movement.
        Uses log_return shifted by 1, 3, and 5 days.
        """
        for lag in self.cfg["lag_periods"]:
            df[f"return_lag_{lag}d"] = df["log_return"].shift(lag)
        return df

    # ===================================================================
    # ORCHESTRATOR
    # ===================================================================
    def run(self, ticker: str) -> pd.DataFrame:
        """Load latest raw data → engineer features → save."""
        logger.info(f"\n{'='*60}")
        logger.info(f"  FEATURE ENGINEERING — {ticker}")
        logger.info(f"{'='*60}")

        df = self.load_latest_raw(ticker)
        df = self.engineer_features(df, ticker)
        self.save_features(df, ticker)

        logger.info(f"  ✅ FEATURES COMPLETE — {ticker}")
        return df


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stock ML Feature Store — engineer features from raw data"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--ticker", type=str, action="append", default=None,
        help="Ticker(s) to process. Defaults to config list."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    tickers = args.ticker if args.ticker else config["data"]["default_tickers"]

    store = FeatureStore(config)
    results = {}

    for ticker in tickers:
        ticker = ticker.upper().strip()
        try:
            df = store.run(ticker)
            results[ticker] = f"OK — {len(df)} rows, {len(df.columns)} cols"
        except Exception as e:
            logger.error(f"  ✗ {ticker} FAILED: {e}")
            results[ticker] = f"FAILED: {e}"

    logger.info(f"\n{'='*60}")
    logger.info("  FEATURE STORE SUMMARY")
    logger.info(f"{'='*60}")
    for t, status in results.items():
        icon = "✅" if status.startswith("OK") else "❌"
        logger.info(f"  {icon} {t}: {status}")


if __name__ == "__main__":
    main()
