"""
conftest.py - shared pytest fixtures for the TradeVision AI test suite.

All fixtures are self-contained: no disk I/O, no API calls, no MLflow.
src/ is added to sys.path here so every test file can import production
modules directly without installing the package.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/ importable from any test file
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def sample_config() -> dict:
    """
    Minimal in-memory config dict that mirrors the production config.yaml
    schema consumed by FeatureStore and train.py helpers.
    """
    return {
        "data": {
            "raw_data_dir":    "data/raw",
            "features_dir":    "data/features",
            "metadata_dir":    "data/metadata",
            "default_tickers": ["AAPL", "MSFT"],
            "default_start_date": "2022-01-01",
            "default_end_date":   "2024-12-31",
            "max_consecutive_fill":          2,
            "max_missing_pct":             0.05,
            "volume_outlier_cap_percentile": 99,
        },
        "features": {
            "rsi_period":          14,
            "macd_fast":           12,
            "macd_slow":           26,
            "macd_signal":          9,
            "bollinger_window":    20,
            "bollinger_std":        2,
            "volatility_window":   20,
            "volume_ratio_window": 20,
            "lag_periods":         [1, 3, 5],
            "sentiment_neutral_fill": 0.0,
        },
        "training": {
            "test_size":      0.20,
            "val_size":       0.20,
            "random_seed":    42,
            "models_dir":     "models",
            "target_column":  "target_direction",
            "random_forest": {
                "n_estimators": 200, "max_depth": 10,
                "min_samples_split": 5, "min_samples_leaf": 2,
            },
            "xgboost": {
                "n_estimators": 200, "max_depth": 6,
                "learning_rate": 0.1, "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "logistic_regression": {
                "C": 1.0, "max_iter": 1000, "solver": "lbfgs",
            },
            "isolation_forest": {
                "n_estimators": 100, "contamination": 0.05,
                "random_state": 42,
            },
        },
        "mlflow": {
            "tracking_uri":    "mlruns",
            "experiment_name": "stock_ml_classifiers",
        },
    }


@pytest.fixture(scope="session")
def sample_ohlcv_df() -> pd.DataFrame:
    """
    300-row synthetic OHLCV DataFrame with a DatetimeIndex named Date.
    Columns match the schema expected by FeatureStore.engineer_features().
    Uses a deterministic seed so every test run is identical.
    """
    rng = np.random.default_rng(seed=42)
    n = 300

    bdays = pd.bdate_range(start="2022-01-03", periods=n, freq="B")

    log_returns = rng.normal(loc=0.0003, scale=0.012, size=n)
    close_prices = 150.0 * np.exp(np.cumsum(log_returns))

    opens  = close_prices * (1 + rng.normal(0, 0.002, n))
    highs  = np.maximum(opens, close_prices) * (1 + rng.uniform(0, 0.005, n))
    lows   = np.minimum(opens, close_prices) * (1 - rng.uniform(0, 0.005, n))
    vols   = rng.integers(1_000_000, 10_000_000, size=n).astype(float)

    df = pd.DataFrame(
        {
            "Open":      opens,
            "High":      highs,
            "Low":       lows,
            "Close":     close_prices,
            "Adj Close": close_prices,
            "Volume":    vols,
        },
        index=bdays,
    )
    df.index.name = "Date"

    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    rolling_vol_mean = df["Volume"].rolling(window=20).mean()
    df["volume_normalised"] = df["Volume"] / rolling_vol_mean
    df = df.dropna()

    return df