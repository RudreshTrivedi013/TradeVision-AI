
import argparse
import json
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import yfinance as yf

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
# DataPipeline Class
# ===================================================================
class DataPipeline:
    """
    Reusable pipeline for any ticker: fetch → validate → clean → preprocess → log.

    All tuneable numbers are read from config.yaml (Rule 2).
    """

    def __init__(self, config: dict):
        self.cfg = config["data"]
        self.raw_dir = Path(self.cfg["raw_data_dir"])
        self.metadata_dir = Path(self.cfg["metadata_dir"])

        # Ensure output directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Tuneable parameters from config
        self.max_consecutive_fill = self.cfg["max_consecutive_fill"]
        self.max_missing_pct = self.cfg["max_missing_pct"]
        self.volume_cap_pct = self.cfg["volume_outlier_cap_percentile"]

    # ---------------------------------------------------------------
    # 1. FETCH — Download OHLCV data via yfinance
    # ---------------------------------------------------------------
    def fetch(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """
        Download OHLCV data for a single ticker.

        Args:
            ticker: Stock symbol (e.g. "AAPL")
            start:  Start date string "YYYY-MM-DD"
            end:    End date string "YYYY-MM-DD"

        Returns:
            Raw DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
        """
        logger.info(f"Fetching {ticker} from {start} to {end} ...")
        df = yf.download(ticker, start=start, end=end, progress=False)

        if df.empty:
            raise ValueError(f"No data returned for {ticker} ({start} → {end})")

        # yfinance sometimes returns MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

        df.index.name = "Date"
        logger.info(f"  ✓ Fetched {len(df)} rows for {ticker}")
        return df

    # ---------------------------------------------------------------
    # 2. VALIDATE — Data quality checks
    # ---------------------------------------------------------------
    def validate(self, df: pd.DataFrame, ticker: str) -> dict:
        """
        Run quality checks and return a report dict.

        Checks:
            - Trading-day gaps (missing rows)
            - Null or negative prices
            - Zero volume on trading days
            - Adjusted Close vs Close anomaly (bad split adjustment)
        """
        report = {
            "ticker": ticker,
            "rows": len(df),
            "issues": [],
        }

        # --- Check 1: Trading-day gaps ---
        expected_days = pd.bdate_range(start=df.index.min(), end=df.index.max())
        missing_days = expected_days.difference(df.index)
        if len(missing_days) > 0:
            report["issues"].append(
                f"{len(missing_days)} missing trading day(s) found"
            )
            logger.warning(f"  ⚠ {ticker}: {len(missing_days)} missing trading days")

        # --- Check 2: Null or negative prices ---
        price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
        existing_price_cols = [c for c in price_cols if c in df.columns]
        null_counts = df[existing_price_cols].isnull().sum()
        neg_counts = (df[existing_price_cols] < 0).sum()

        for col in existing_price_cols:
            if null_counts[col] > 0:
                report["issues"].append(f"{col} has {null_counts[col]} null(s)")
                logger.warning(f"  ⚠ {ticker}: {col} has {null_counts[col]} null(s)")
            if neg_counts[col] > 0:
                report["issues"].append(f"{col} has {neg_counts[col]} negative value(s)")
                logger.warning(f"  ⚠ {ticker}: {col} has {neg_counts[col]} negative(s)")

        # --- Check 3: Zero volume on trading days ---
        if "Volume" in df.columns:
            zero_vol = (df["Volume"] == 0).sum()
            if zero_vol > 0:
                report["issues"].append(f"Volume is zero on {zero_vol} trading day(s)")
                logger.warning(f"  ⚠ {ticker}: zero volume on {zero_vol} day(s)")

        # --- Check 4: Adjusted Close vs Close deviation ---
        if "Adj Close" in df.columns and "Close" in df.columns:
            ratio = (df["Adj Close"] / df["Close"]).dropna()
            # Flag if ratio deviates more than 50% from 1.0 on any day
            bad_adj = ((ratio < 0.5) | (ratio > 1.5)).sum()
            if bad_adj > 0:
                report["issues"].append(
                    f"Adj Close / Close ratio is extreme on {bad_adj} day(s) — possible bad split adjustment"
                )
                logger.warning(f"  ⚠ {ticker}: {bad_adj} day(s) with extreme Adj Close ratio")

        if not report["issues"]:
            logger.info(f"  ✓ {ticker}: All validation checks passed")
        else:
            logger.info(f"  ✓ {ticker}: Validation complete — {len(report['issues'])} issue(s)")

        return report

    # ---------------------------------------------------------------
    # 3. CLEAN — Fix problems found by validate()
    # ---------------------------------------------------------------
    def clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        """
        Clean the raw data.

        Steps:
            1. Forward-fill up to N consecutive missing days
            2. Drop ticker if >X% data still missing
            3. Cap extreme volume outliers at the Pth percentile

        Returns:
            Cleaned DataFrame, or None if ticker is too sparse.
        """
        logger.info(f"  Cleaning {ticker} ...")

        # --- Step 1: Reindex to full business-day range, then forward-fill ---
        full_range = pd.bdate_range(start=df.index.min(), end=df.index.max())
        df = df.reindex(full_range)
        df.index.name = "Date"

        # Forward-fill with a limit
        df = df.ffill(limit=self.max_consecutive_fill)

        # --- Step 2: Check remaining missing percentage ---
        missing_pct = df.isnull().mean().mean()  # average across all columns
        if missing_pct > self.max_missing_pct:
            logger.warning(
                f"  ✗ {ticker}: {missing_pct:.1%} missing after fill — exceeds "
                f"{self.max_missing_pct:.0%} threshold. DROPPING."
            )
            return None

        # Drop any remaining rows with nulls (edges)
        df = df.dropna()

        # --- Step 3: Cap extreme volume outliers ---
        if "Volume" in df.columns:
            cap = df["Volume"].quantile(self.volume_cap_pct / 100)
            before = (df["Volume"] > cap).sum()
            df["Volume"] = df["Volume"].clip(upper=cap)
            if before > 0:
                logger.info(f"    Capped {before} volume outlier(s) at {cap:,.0f}")

        logger.info(f"  ✓ {ticker}: Cleaned → {len(df)} rows")
        return df

    # ---------------------------------------------------------------
    # 4. PREPROCESS — Feature-ready transformations
    # ---------------------------------------------------------------
    def preprocess(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Transform cleaned data into modelling-ready format.

        Steps:
            1. Compute log returns (stationary)
            2. Normalise volume by 20-day rolling mean
        """
        logger.info(f"  Preprocessing {ticker} ...")

        # Log returns: log(close_t / close_t-1)
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        # Normalised volume: today's volume / 20-day rolling mean
        rolling_vol_mean = df["Volume"].rolling(window=20).mean()
        df["volume_normalised"] = df["Volume"] / rolling_vol_mean

        # Drop the initial NaN rows from rolling calculations
        df = df.dropna()

        logger.info(f"  ✓ {ticker}: Preprocessed → {len(df)} rows, "
                     f"{len(df.columns)} columns")
        return df

    # ---------------------------------------------------------------
    # 5. LOG METADATA — Save a JSON report per run
    # ---------------------------------------------------------------
    def log_metadata(
        self,
        df: pd.DataFrame,
        ticker: str,
        start: str,
        end: str,
        validation_report: dict,
    ) -> Path:
        """
        Save a JSON metadata file for this pipeline run.
        """
        meta = {
            "ticker": ticker,
            "date_range": {"start": start, "end": end},
            "row_count": len(df),
            "columns": list(df.columns),
            "null_counts": df.isnull().sum().to_dict(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "validation_issues": validation_report.get("issues", []),
            "run_timestamp": datetime.now().isoformat(),
        }

        filename = f"{ticker}_{date.today().isoformat()}_metadata.json"
        filepath = self.metadata_dir / filename
        with open(filepath, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        logger.info(f"  ✓ Metadata saved → {filepath}")
        return filepath

    # ---------------------------------------------------------------
    # ORCHESTRATOR — Run the full pipeline for one ticker
    # ---------------------------------------------------------------
    def run(self, ticker: str, start: str, end: str) -> pd.DataFrame | None:
        """
        Execute the full pipeline: fetch → validate → clean → preprocess → save → log.

        Returns:
            Final preprocessed DataFrame, or None if ticker was dropped.
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"  PIPELINE START — {ticker}")
        logger.info(f"{'='*60}")

        # Fetch
        df = self.fetch(ticker, start, end)

        # Validate
        validation_report = self.validate(df, ticker)

        # Clean
        df = self.clean(df, ticker)
        if df is None:
            return None

        # Preprocess
        df = self.preprocess(df, ticker)

        # Save versioned parquet: AAPL_2024-04-27.parquet
        parquet_name = f"{ticker}_{date.today().isoformat()}.parquet"
        parquet_path = self.raw_dir / parquet_name
        df.to_parquet(parquet_path)
        logger.info(f"  ✓ Saved → {parquet_path}")

        # Log metadata
        self.log_metadata(df, ticker, start, end, validation_report)

        logger.info(f"  ✅ PIPELINE COMPLETE — {ticker}")
        return df


# ===================================================================
# CLI Entry Point
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Stock ML Data Pipeline — fetch, validate, clean, preprocess"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--ticker", type=str, action="append", default=None,
        help="Ticker(s) to process. Can be repeated: --ticker AAPL --ticker NVDA. "
             "Defaults to the list in config.yaml."
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Determine tickers: CLI args override config defaults
    tickers = args.ticker if args.ticker else config["data"]["default_tickers"]

    start = config["data"]["default_start_date"]
    end = config["data"]["default_end_date"]

    # Run pipeline
    pipeline = DataPipeline(config)
    results = {}

    for ticker in tickers:
        ticker = ticker.upper().strip()
        try:
            df = pipeline.run(ticker, start, end)
            results[ticker] = "OK" if df is not None else "DROPPED (too sparse)"
        except Exception as e:
            logger.error(f"  ✗ {ticker} FAILED: {e}")
            results[ticker] = f"FAILED: {e}"

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("  PIPELINE SUMMARY")
    logger.info(f"{'='*60}")
    for t, status in results.items():
        icon = "✅" if status == "OK" else "❌"
        logger.info(f"  {icon} {t}: {status}")


if __name__ == "__main__":
    main()
