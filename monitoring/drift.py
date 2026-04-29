"""
Drift Detection — Stage 9: Feature drift, prediction drift, rolling accuracy,
and retraining triggers.

Drift checks:
    1. Feature drift: KS test per feature (last 30d vs training data)
    2. Prediction drift: proportion shift > 15pp
    3. Rolling accuracy: 30-day window, retrain if < 55%
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from scipy import stats

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
# DriftDetector
# ===================================================================
class DriftDetector:
    """
    Monitors feature drift, prediction drift, and rolling accuracy.
    All thresholds come from config.yaml (Rule 2).
    """

    def __init__(self, config: dict):
        self.cfg = config["monitoring"]
        self.lookback = self.cfg["drift_lookback_days"]
        self.ks_threshold = self.cfg["ks_pvalue_threshold"]
        self.pred_drift_threshold = self.cfg["prediction_drift_threshold"]
        self.accuracy_window = self.cfg["rolling_accuracy_window"]
        self.min_accuracy = self.cfg["min_accuracy_threshold"]

    # ---------------------------------------------------------------
    # 1. Feature Drift — KS test per feature
    # ---------------------------------------------------------------
    def check_feature_drift(
        self, df: pd.DataFrame, training_data: Optional[pd.DataFrame] = None
    ) -> dict:
        """
        Compare recent feature distributions vs training data using KS test.

        If no training_data provided, splits the dataframe:
        first 70% = 'training', last 30 days = 'recent'.
        """
        exclude = [
            "Open", "High", "Low", "Close", "Adj Close", "Volume",
            "target_direction", "ticker", "log_return",
        ]
        feature_cols = [c for c in df.columns if c not in exclude]

        if training_data is not None:
            reference = training_data
            recent = df.tail(self.lookback)
        else:
            split_idx = int(len(df) * 0.7)
            reference = df.iloc[:split_idx]
            recent = df.tail(self.lookback)

        ks_results = []
        drifted_features = []

        for col in feature_cols:
            if col not in reference.columns or col not in recent.columns:
                continue

            ref_vals = reference[col].dropna().values
            rec_vals = recent[col].dropna().values

            if len(ref_vals) < 10 or len(rec_vals) < 10:
                continue

            ks_stat, p_value = stats.ks_2samp(ref_vals, rec_vals)

            is_drifted = p_value < self.ks_threshold
            if is_drifted:
                drifted_features.append(col)

            ks_results.append({
                "feature": col,
                "ks_statistic": round(ks_stat, 4),
                "p_value": round(p_value, 4),
                "drifted": is_drifted,
            })

        report = {
            "total_features": len(feature_cols),
            "tested_features": len(ks_results),
            "drifted_features": drifted_features,
            "n_drifted": len(drifted_features),
            "ks_results": ks_results,
        }

        if drifted_features:
            logger.warning(f"  ⚠ Feature drift detected: {drifted_features}")
        else:
            logger.info("  ✓ No feature drift detected")

        return report

    # ---------------------------------------------------------------
    # 2. Prediction Drift — proportion shift
    # ---------------------------------------------------------------
    def check_prediction_drift(self, log_path: str | Path) -> dict:
        """
        Check if recent prediction distribution differs from historical.
        Flags if UP proportion shifted > threshold.
        """
        log_path = Path(log_path)
        if not log_path.exists():
            return {"drifted": False, "reason": "No log file found"}

        logs = []
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))

        if len(logs) < 20:
            return {"drifted": False, "reason": "Not enough predictions for drift check"}

        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Split into historical vs recent
        total = len(df)
        recent_n = min(self.lookback, total // 3)
        historical = df.iloc[:-recent_n]
        recent = df.iloc[-recent_n:]

        hist_up_pct = historical["prediction"].mean()
        recent_up_pct = recent["prediction"].mean()
        shift = abs(recent_up_pct - hist_up_pct)

        drifted = shift > self.pred_drift_threshold

        result = {
            "drifted": drifted,
            "historical_up_pct": round(hist_up_pct, 4),
            "recent_up_pct": round(recent_up_pct, 4),
            "shift": round(shift, 4),
            "threshold": self.pred_drift_threshold,
        }

        if drifted:
            logger.warning(
                f"  ⚠ Prediction drift: {shift:.1%} shift "
                f"(hist: {hist_up_pct:.1%}, recent: {recent_up_pct:.1%})"
            )
        else:
            logger.info(f"  ✓ No prediction drift (shift: {shift:.1%})")

        return result

    # ---------------------------------------------------------------
    # 3. Rolling Accuracy
    # ---------------------------------------------------------------
    def compute_rolling_accuracy(
        self, predictions: pd.Series, actuals: pd.Series
    ) -> pd.Series:
        """Compute rolling accuracy over a window."""
        correct = (predictions == actuals).astype(int)
        return correct.rolling(window=self.accuracy_window, min_periods=10).mean()

    # ---------------------------------------------------------------
    # 4. should_retrain() — The retraining trigger
    # ---------------------------------------------------------------
    def should_retrain(
        self,
        feature_drift_report: Optional[dict] = None,
        prediction_drift_report: Optional[dict] = None,
        rolling_accuracy: Optional[float] = None,
    ) -> bool:
        """
        Returns True if ANY condition is met:
            - Feature drift detected (any feature's KS p-value < 0.05)
            - Prediction drift > 15 percentage points
            - Rolling accuracy < 55%
        """
        reasons = []

        if feature_drift_report and feature_drift_report.get("n_drifted", 0) > 0:
            reasons.append(
                f"Feature drift in: {feature_drift_report['drifted_features']}"
            )

        if prediction_drift_report and prediction_drift_report.get("drifted", False):
            reasons.append(
                f"Prediction drift: {prediction_drift_report.get('shift', 0):.1%}"
            )

        if rolling_accuracy is not None and rolling_accuracy < self.min_accuracy:
            reasons.append(
                f"Rolling accuracy {rolling_accuracy:.1%} < {self.min_accuracy:.0%}"
            )

        if reasons:
            logger.warning(f"  🔄 RETRAIN RECOMMENDED — Reasons: {reasons}")
            return True

        logger.info("  ✓ No retraining needed")
        return False
