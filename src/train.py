
import argparse
import json
import logging
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
# Helper: Load & combine all feature files
# ===================================================================
def load_feature_data(config: dict) -> pd.DataFrame:
    """Load and concatenate all feature parquets into one training set."""
    features_dir = Path(config["data"]["features_dir"])
    files = sorted(features_dir.glob("*_features.parquet"))

    if not files:
        raise FileNotFoundError(f"No feature files found in {features_dir}")

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        ticker = f.stem.replace("_features", "")
        df["ticker"] = ticker
        dfs.append(df)
        logger.info(f"  Loaded {f.name}: {df.shape}")

    combined = pd.concat(dfs)
    logger.info(f"  Combined: {combined.shape}")
    return combined


# ===================================================================
# Helper: Get feature columns (exclude non-feature cols)
# ===================================================================
def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only model-input feature columns."""
    exclude = [
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "target_direction", "ticker", "log_return",
    ]
    return [c for c in df.columns if c not in exclude]


# ===================================================================
# Naive baseline
# ===================================================================
def compute_baseline(y_test: np.ndarray) -> dict:
    """Naive 'always predict UP' baseline."""
    y_naive = np.ones_like(y_test)
    return {
        "accuracy": accuracy_score(y_test, y_naive),
        "f1": f1_score(y_test, y_naive, zero_division=0),
        "precision": precision_score(y_test, y_naive, zero_division=0),
        "recall": recall_score(y_test, y_naive, zero_division=0),
    }


# ===================================================================
# XGBoost Hyperparameter Tuning on Validation Set
# ===================================================================
def tune_xgboost_on_val(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base_params: dict,
    seed: int,
) -> tuple[dict, float]:
    """
    Grid search over XGBoost max_depth and n_estimators using the validation set.

    Protocol:
        - Models are fit on X_train ONLY.
        - X_val is used purely for model selection — it never influences fitting.
        - The winning hyperparameters are used to re-train the final model on
          X_trainval (train + val combined), which is then evaluated on X_test.

    Grid searched:
        max_depth:    [3, 4, 5, 6]
        n_estimators: [100, 150, 200, 250]
        (all other params held at config defaults)

    Returns:
        best_params: dict of the winning hyperparameter values
        best_val_acc: validation accuracy of the winning configuration
    """
    max_depth_grid    = [3, 4, 5, 6]
    n_estimators_grid = [100, 150, 200, 250]

    best_val_acc = -1.0
    best_params  = {k: base_params[k] for k in base_params}  # start from config defaults

    n_configs = len(max_depth_grid) * len(n_estimators_grid)
    logger.info(
        f"\n  🔍 Tuning XGBoost on val set ({len(X_val)} samples) — "
        f"{n_configs} configurations ..."
    )

    for max_depth in max_depth_grid:
        for n_estimators in n_estimators_grid:
            m = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=base_params["learning_rate"],
                subsample=base_params["subsample"],
                colsample_bytree=base_params["colsample_bytree"],
                random_state=seed,
                eval_metric="logloss",
                verbosity=0,
            )
            m.fit(X_train, y_train)
            acc = accuracy_score(y_val, m.predict(X_val))

            if acc > best_val_acc:
                best_val_acc = acc
                best_params = {
                    "n_estimators":    n_estimators,
                    "max_depth":       max_depth,
                    "learning_rate":   base_params["learning_rate"],
                    "subsample":       base_params["subsample"],
                    "colsample_bytree": base_params["colsample_bytree"],
                }

    logger.info(
        f"  ✅ Best XGBoost — max_depth={best_params['max_depth']}, "
        f"n_estimators={best_params['n_estimators']} "
        f"(val acc: {best_val_acc:.4f})"
    )
    return best_params, best_val_acc


# ===================================================================
# Bootstrap Statistical Significance Test
# ===================================================================
def bootstrap_significance(
    model_a_name: str,
    model_b_name: str,
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap test: is model_a's test-set accuracy significantly
    better than model_b's on the same held-out rows?

    Method:
        For each bootstrap iteration, resample (with replacement) the
        n test rows, then compute the accuracy difference
        (acc_a - acc_b) on the resampled set.
        After n_boot iterations, the 95% CI on the difference is
        [2.5th, 97.5th] percentile of the bootstrap distribution.
        If the CI lower bound > 0, model_a is significantly better at α=0.05.

    Note on sample size:
        With ~114 test rows (20% of ~570), the CI will typically be
        wide (±5–10 pp), meaning a 1–3 pp raw advantage is unlikely
        to be significant. This is expected and documents the honest
        precision limit of this dataset size.

    Args:
        model_a_name: Label for the "challenger" model.
        model_b_name: Label for the "reference" model.
        y_true:       Ground-truth labels on the held-out test set.
        pred_a:       Predictions from model_a on the same test set.
        pred_b:       Predictions from model_b on the same test set.
        n_boot:       Number of bootstrap iterations (default 10 000).
        seed:         RNG seed for reproducibility.

    Returns:
        dict with observed_diff, ci_95 tuple, p_value (one-sided), and
        significant flag.
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)

    correct_a = (pred_a == y_true).astype(float)
    correct_b = (pred_b == y_true).astype(float)
    obs_diff  = float(correct_a.mean() - correct_b.mean())

    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_diffs[i] = correct_a[idx].mean() - correct_b[idx].mean()

    ci_low  = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    # One-sided p-value: fraction of bootstrap samples where A was NOT better
    p_val   = float((boot_diffs <= 0).mean())

    return {
        "model_a":              model_a_name,
        "model_b":              model_b_name,
        "n_test_samples":       n,
        "observed_diff":        round(obs_diff,  4),
        "ci_95_low":            round(ci_low,    4),
        "ci_95_high":           round(ci_high,   4),
        "p_value_one_sided":    round(p_val,     4),
        "significant_at_95pct": ci_low > 0,
    }


# ===================================================================
# Train a single classifier + log to MLflow
# ===================================================================
def train_and_log_classifier(
    model,
    model_name: str,
    X_trainval, X_test, y_trainval, y_test,
    params: dict,
    config: dict,
    feature_names: list,
    data_files_used: list,
    val_tuned: bool = False,
):
    """
    Fit one classifier on X_trainval (train + val = 80% of data),
    evaluate it on X_test (held-out 20%), and log everything to MLflow.

    Split context (set by caller):
        X_trainval: 64% train + 16% val combined — full fitting set.
        X_test:     Final 20% — strictly held-out, never seen during tuning.
    """

    with mlflow.start_run(run_name=model_name):
        # Log environment info for reproducibility
        mlflow.log_param("python_version",  platform.python_version())
        mlflow.log_param("model_type",      model_name)
        mlflow.log_param("random_seed",     config["training"]["random_seed"])
        mlflow.log_param("test_size",       config["training"]["test_size"])
        mlflow.log_param("val_size",        config["training"].get("val_size", 0.20))
        mlflow.log_param("split_strategy",  "64/16/20 time-ordered (train/val/test)")
        mlflow.log_param("val_tuned",       val_tuned)
        mlflow.log_param("n_features",      len(feature_names))
        mlflow.log_param("trainval_samples", len(X_trainval))
        mlflow.log_param("test_samples",    len(X_test))
        mlflow.log_param("data_files",      str(data_files_used))

        # Log all hyperparameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train with timing
        logger.info(f"\n  Training {model_name} on train+val ({len(X_trainval)} samples) ...")
        start_time = time.time()
        model.fit(X_trainval, y_trainval)
        train_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", round(train_time, 3))

        # Predict on held-out test set
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms/sample
        mlflow.log_metric("inference_time_ms_per_sample", round(inference_time, 4))

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        # Metrics (all computed on held-out test only)
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        roc  = roc_auc_score(y_test, y_prob)
        cm   = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("accuracy",  round(acc,  4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall",    round(rec,  4))
        mlflow.log_metric("f1_score",  round(f1,   4))
        mlflow.log_metric("roc_auc",   round(roc,  4))

        # Log confusion matrix as artifact
        cm_dict = {
            "true_neg":  int(cm[0][0]),
            "false_pos": int(cm[0][1]),
            "false_neg": int(cm[1][0]),
            "true_pos":  int(cm[1][1]),
        }
        cm_path = f"confusion_matrix_{model_name}.json"
        with open(cm_path, "w") as f:
            json.dump(cm_dict, f, indent=2)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            importance = dict(zip(feature_names, model.feature_importances_.tolist()))
            imp_path = f"feature_importance_{model_name}.json"
            with open(imp_path, "w") as f:
                json.dump(importance, f, indent=2)
            mlflow.log_artifact(imp_path)
            os.remove(imp_path)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path=model_name)

        # Save model locally as .pkl
        models_dir = Path(config["training"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)

        logger.info(
            f"  ✅ {model_name} — Acc: {acc:.4f} | F1: {f1:.4f} | "
            f"ROC-AUC: {roc:.4f} | Train: {train_time:.2f}s"
        )

        return {
            "model_name":    model_name,
            "accuracy":      acc,
            "precision":     prec,
            "recall":        rec,
            "f1":            f1,
            "roc_auc":       roc,
            "training_time": train_time,
            "inference_time_ms": inference_time,
            "y_pred":        y_pred,   # stored for bootstrap significance test
        }


# ===================================================================
# Train anomaly detector
# ===================================================================
def train_anomaly_model(
    df: pd.DataFrame,
    config: dict,
):
    """Train Isolation Forest on volume ratio + price change magnitude."""

    anomaly_features = ["volume_ratio", "price_change_pct"]
    X_anomaly = df[anomaly_features].dropna()

    params = config["training"]["isolation_forest"]

    with mlflow.start_run(run_name="IsolationForest"):
        mlflow.log_param("model_type", "IsolationForest")
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("features", str(anomaly_features))

        logger.info("\n  Training Isolation Forest (anomaly detection) ...")
        iso = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=params["contamination"],
            random_state=params["random_state"],
        )
        iso.fit(X_anomaly)

        # Predict: -1 = anomaly, 1 = normal
        predictions  = iso.predict(X_anomaly)
        n_anomalies  = (predictions == -1).sum()
        anomaly_pct  = n_anomalies / len(predictions) * 100

        mlflow.log_metric("n_anomalies", int(n_anomalies))
        mlflow.log_metric("anomaly_pct", round(anomaly_pct, 2))

        # Save model
        models_dir = Path(config["training"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path  = models_dir / "isolation_forest.pkl"
        joblib.dump(iso, model_path)

        mlflow.sklearn.log_model(iso, artifact_path="IsolationForest")

        logger.info(f"  ✅ IsolationForest — {n_anomalies} anomalies ({anomaly_pct:.1f}%)")

    return iso


# ===================================================================
# Fairness analysis
# ===================================================================
def fairness_analysis(
    model, model_name: str,
    df: pd.DataFrame, feature_cols: list, config: dict
) -> dict:
    """
    Compare model performance on high-vol (TSLA) vs low-vol (MSFT) stocks.
    Documents any accuracy gap > 10%.
    """
    results = {}
    for ticker in ["TSLA", "MSFT"]:
        subset = df[df["ticker"] == ticker]
        if len(subset) < 50:
            logger.warning(f"    Not enough data for {ticker} fairness check")
            continue

        X      = subset[feature_cols].values
        y      = subset[config["training"]["target_column"]].values
        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred, zero_division=0)
        results[ticker] = {"accuracy": acc, "f1": f1, "samples": len(subset)}

    if "TSLA" in results and "MSFT" in results:
        gap = abs(results["TSLA"]["accuracy"] - results["MSFT"]["accuracy"])
        results["accuracy_gap"] = gap
        if gap > 0.10:
            logger.warning(
                f"  ⚠ {model_name} FAIRNESS WARNING: {gap:.1%} accuracy gap "
                f"between TSLA ({results['TSLA']['accuracy']:.3f}) and "
                f"MSFT ({results['MSFT']['accuracy']:.3f})"
            )
        else:
            logger.info(f"  ✓ {model_name} fairness OK: {gap:.1%} gap")

    return results


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Train all models with MLflow logging")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config["training"]["random_seed"]

    # ---------------------------------------------------------------
    # Fix all random seeds for full reproducibility
    # Covers: Python stdlib, NumPy legacy API, and XGBoost (via random_state=seed).
    # ---------------------------------------------------------------
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"  🔒 Random seeds fixed — stdlib/numpy/xgboost/sklearn: seed={seed}")

    # MLflow setup
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Load data
    logger.info("\n" + "=" * 60)
    logger.info("  LOADING FEATURE DATA")
    logger.info("=" * 60)
    df = load_feature_data(config)

    feature_cols = get_feature_columns(df)
    logger.info(f"  Feature columns ({len(feature_cols)}): {feature_cols}")

    # Data files used (for reproducibility logging)
    features_dir = Path(config["data"]["features_dir"])
    data_files   = [f.name for f in features_dir.glob("*_features.parquet")]

    # Ensure chronological order before splitting
    df.sort_index(inplace=True)

    # Prepare data
    target = config["training"]["target_column"]
    X      = df[feature_cols].values
    y      = df[target].values

    # -------------------------------------------------------------------
    # 3-Way Time-Ordered Split: 64% train / 16% val / 20% test
    #
    # Step 1 — Hold out the final test set (last 20% chronologically).
    #           These rows are NEVER used during fitting or tuning.
    # Step 2 — From the remaining 80%, split off the last 20% (= 16% total)
    #           as the validation set used for hyperparameter selection.
    # Step 3 — After tuning, re-train final models on X_trainval (80% total)
    #           and report metrics on X_test only.
    #
    # shuffle=False preserves chronological order throughout — critical for
    # time-series data to prevent look-ahead leakage.
    # -------------------------------------------------------------------
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],   # 0.20 → last 20%
        random_state=seed,
        shuffle=False,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=config["training"].get("val_size", 0.20),  # 0.20 of trainval → 16% total
        random_state=seed,
        shuffle=False,
    )

    n_total = len(X)
    logger.info(
        f"\n  📐 DATA SPLIT (64/16/20 — time-ordered, no shuffle)\n"
        f"     Train:  {X_train.shape}   ({len(X_train)/n_total*100:.0f}% of total)\n"
        f"     Val:    {X_val.shape}    ({len(X_val)/n_total*100:.0f}% of total)  ← HP tuning only\n"
        f"     Test:   {X_test.shape}   ({len(X_test)/n_total*100:.0f}% of total)  ← never touched until final eval\n"
        f"     Final fit will use Train+Val = {len(X_trainval)} rows ({len(X_trainval)/n_total*100:.0f}%)"
    )
    logger.info(
        f"  Class distribution — Train: {np.mean(y_train):.3f} up, "
        f"Val: {np.mean(y_val):.3f} up, "
        f"Test: {np.mean(y_test):.3f} up"
    )

    # ---------------------------------------------------------------
    # Baseline (computed on held-out test set)
    # ---------------------------------------------------------------
    baseline = compute_baseline(y_test)
    logger.info(f"\n  📊 NAIVE BASELINE (always predict UP):")
    logger.info(f"     Accuracy: {baseline['accuracy']:.4f} | F1: {baseline['f1']:.4f}")

    # ---------------------------------------------------------------
    # Tune XGBoost on validation set
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  HYPERPARAMETER TUNING (VALIDATION SET)")
    logger.info("=" * 60)
    xgb_params_config = config["training"]["xgboost"]
    best_xgb_params, xgb_val_acc = tune_xgboost_on_val(
        X_train, y_train,
        X_val,   y_val,
        xgb_params_config, seed,
    )

    # ---------------------------------------------------------------
    # Train 3 classifiers (all fit on X_trainval — train + val)
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING CLASSIFIERS (train+val → test eval)")
    logger.info("=" * 60)
    results  = []
    cfg_t    = config["training"]

    # 1. Random Forest (config defaults — no val tuning)
    rf_params = cfg_t["random_forest"]
    rf = RandomForestClassifier(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"],
        min_samples_split=rf_params["min_samples_split"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        random_state=seed,
        n_jobs=-1,
    )
    result = train_and_log_classifier(
        rf, "RandomForest",
        X_trainval, X_test, y_trainval, y_test,
        rf_params, config, feature_cols, data_files,
        val_tuned=False,
    )
    results.append(result)

    # 2. XGBoost (val-tuned max_depth + n_estimators)
    xgb_model = xgb.XGBClassifier(
        n_estimators=best_xgb_params["n_estimators"],
        max_depth=best_xgb_params["max_depth"],
        learning_rate=best_xgb_params["learning_rate"],
        subsample=best_xgb_params["subsample"],
        colsample_bytree=best_xgb_params["colsample_bytree"],
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )
    result = train_and_log_classifier(
        xgb_model, "XGBoost",
        X_trainval, X_test, y_trainval, y_test,
        best_xgb_params, config, feature_cols, data_files,
        val_tuned=True,   # signals that params were chosen via val set
    )
    results.append(result)

    # 3. Logistic Regression (config defaults)
    lr_params = cfg_t["logistic_regression"]
    lr = LogisticRegression(
        C=lr_params["C"],
        max_iter=lr_params["max_iter"],
        solver=lr_params["solver"],
        random_state=seed,
    )
    result = train_and_log_classifier(
        lr, "LogisticRegression",
        X_trainval, X_test, y_trainval, y_test,
        lr_params, config, feature_cols, data_files,
        val_tuned=False,
    )
    results.append(result)

    # ---------------------------------------------------------------
    # Anomaly model
    # ---------------------------------------------------------------
    train_anomaly_model(df, config)

    # ---------------------------------------------------------------
    # Fairness analysis (on best classifier)
    # ---------------------------------------------------------------
    best = max(results, key=lambda r: r["f1"])
    logger.info(f"\n  🏆 BEST CLASSIFIER: {best['model_name']} (F1: {best['f1']:.4f})")

    best_model = joblib.load(
        Path(config["training"]["models_dir"]) / f"{best['model_name']}.pkl"
    )
    fairness_analysis(best_model, best["model_name"], df, feature_cols, config)

    # ---------------------------------------------------------------
    # Bootstrap Statistical Significance (XGBoost vs RF and LR)
    # ---------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("  STATISTICAL SIGNIFICANCE (paired bootstrap, n=10 000)")
    logger.info(f"{'='*60}")
    logger.info(
        f"  ⚠  NOTE: test set = {len(X_test)} rows. At this sample size a "
        f"1–3 pp accuracy gap typically has a wide CI (±5–10 pp) and will "
        f"NOT be statistically significant. This is expected and honest."
    )

    # Map model name → y_pred for easy lookup
    preds = {r["model_name"]: r["y_pred"] for r in results}

    sig_results = []
    for ref_name in ["RandomForest", "LogisticRegression"]:
        if "XGBoost" in preds and ref_name in preds:
            sig = bootstrap_significance(
                model_a_name="XGBoost",
                model_b_name=ref_name,
                y_true=y_test,
                pred_a=preds["XGBoost"],
                pred_b=preds[ref_name],
                n_boot=10_000,
                seed=seed,
            )
            sig_results.append(sig)
            flag = "✅ SIGNIFICANT" if sig["significant_at_95pct"] else "❌ NOT significant"
            logger.info(
                f"  XGBoost vs {ref_name}: "
                f"Δacc={sig['observed_diff']:+.4f} | "
                f"95% CI [{sig['ci_95_low']:+.4f}, {sig['ci_95_high']:+.4f}] | "
                f"p={sig['p_value_one_sided']:.4f} | {flag} at α=0.05"
            )

    # Save significance report as artifact
    sig_path = "bootstrap_significance.json"
    with open(sig_path, "w") as f:
        json.dump(sig_results, f, indent=2)
    # Log to MLflow as a standalone run for traceability
    with mlflow.start_run(run_name="BootstrapSignificance"):
        mlflow.log_artifact(sig_path)
        for sig in sig_results:
            prefix = f"xgb_vs_{sig['model_b'].lower()}_"
            mlflow.log_metric(prefix + "obs_diff",  sig["observed_diff"])
            mlflow.log_metric(prefix + "ci_95_low", sig["ci_95_low"])
            mlflow.log_metric(prefix + "ci_95_high",sig["ci_95_high"])
            mlflow.log_metric(prefix + "p_value",   sig["p_value_one_sided"])
    os.remove(sig_path)

    # ---------------------------------------------------------------
    # Per-ticker baseline + best-model verification
    # ---------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("  PER-TICKER VERIFICATION (all 5 tickers)")
    logger.info(f"{'='*60}")
    logger.info(
        "  Models are trained on pooled data from all 5 tickers.\n"
        "  Per-ticker accuracy below shows how the BEST model performs\n"
        "  on each ticker's own data — this is a fairness/consistency check,\n"
        "  NOT a separate per-ticker training run."
    )
    for ticker in sorted(df["ticker"].unique()):
        sub      = df[df["ticker"] == ticker]
        X_sub    = sub[feature_cols].values
        y_sub    = sub[config["training"]["target_column"]].values
        bl_acc   = accuracy_score(y_sub, np.ones_like(y_sub))
        best_acc = accuracy_score(y_sub, best_model.predict(X_sub))
        beats    = "✅" if best_acc > bl_acc else "❌"
        logger.info(
            f"  {ticker}: baseline={bl_acc:.4f} | {best['model_name']}={best_acc:.4f} "
            f"| {beats} {'beats' if best_acc > bl_acc else 'loses to'} baseline"
        )

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("  TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(
        f"  Split:  64% train / 16% val (HP tuning) / 20% test (final eval)\n"
        f"  Seed:   {seed}  (stdlib random + numpy — run twice to confirm identical results)"
    )
    logger.info(f"  {'Model':<22} {'Acc':>7} {'F1':>7} {'ROC-AUC':>8} {'Time':>7}")
    logger.info(f"  {'-'*52}")
    logger.info(
        f"  {'BASELINE (always UP)':<22} {baseline['accuracy']:>7.4f} "
        f"{baseline['f1']:>7.4f} {'N/A':>8} {'N/A':>7}"
    )
    for r in results:
        tuned_flag = " *" if r["model_name"] == "XGBoost" else "  "
        logger.info(
            f"  {r['model_name']:<22} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
            f"{r['roc_auc']:>8.4f} {r['training_time']:>6.2f}s{tuned_flag}"
        )
        beat = "✅ BEATS" if r["f1"] > baseline["f1"] else "❌ LOSES TO"
        logger.info(f"    → {beat} baseline on F1")

    logger.info(f"\n  * XGBoost max_depth + n_estimators val-tuned. All metrics on held-out test set.")
    logger.info(f"  🏆 Winner: {best['model_name']}")
    logger.info("  ✅ All models saved to models/ and logged to MLflow")


if __name__ == "__main__":
    main()
