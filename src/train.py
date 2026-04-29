
import argparse
import json
import logging
import os
import sys
import time
import platform
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

    combined = pd.concat(dfs, ignore_index=True)
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
# Train a single classifier + log to MLflow
# ===================================================================
def train_and_log_classifier(
    model,
    model_name: str,
    X_train, X_test, y_train, y_test,
    params: dict,
    config: dict,
    feature_names: list,
    data_files_used: list,
):
    """Train one classifier, evaluate it, and log everything to MLflow."""

    with mlflow.start_run(run_name=model_name):
        # Log environment info for reproducibility
        mlflow.log_param("python_version", platform.python_version())
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("random_seed", config["training"]["random_seed"])
        mlflow.log_param("test_size", config["training"]["test_size"])
        mlflow.log_param("n_features", len(feature_names))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("data_files", str(data_files_used))

        # Log all hyperparameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Train with timing
        logger.info(f"\n  Training {model_name} ...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        mlflow.log_metric("training_time_seconds", round(train_time, 3))

        # Predict with timing
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
        mlflow.log_metric("inference_time_ms_per_sample", round(inference_time, 4))

        # Probabilities for ROC-AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        mlflow.log_metric("accuracy", round(acc, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("roc_auc", round(roc, 4))

        # Log confusion matrix as artifact
        cm_dict = {
            "true_neg": int(cm[0][0]),
            "false_pos": int(cm[0][1]),
            "false_neg": int(cm[1][0]),
            "true_pos": int(cm[1][1]),
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

        logger.info(f"  ✅ {model_name} — Acc: {acc:.4f} | F1: {f1:.4f} | "
                     f"ROC-AUC: {roc:.4f} | Train: {train_time:.2f}s")

        return {
            "model_name": model_name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
            "training_time": train_time,
            "inference_time_ms": inference_time,
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
        predictions = iso.predict(X_anomaly)
        n_anomalies = (predictions == -1).sum()
        anomaly_pct = n_anomalies / len(predictions) * 100

        mlflow.log_metric("n_anomalies", int(n_anomalies))
        mlflow.log_metric("anomaly_pct", round(anomaly_pct, 2))

        # Save model
        models_dir = Path(config["training"]["models_dir"])
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "isolation_forest.pkl"
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

        X = subset[feature_cols].values
        y = subset[config["training"]["target_column"]].values
        y_pred = model.predict(X)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, zero_division=0)
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
    np.random.seed(seed)

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
    data_files = [f.name for f in features_dir.glob("*_features.parquet")]

    # Prepare data
    target = config["training"]["target_column"]
    X = df[feature_cols].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["training"]["test_size"],
        random_state=seed,
        stratify=y,
    )

    logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"  Class distribution — Train: {np.mean(y_train):.3f} up, "
                 f"Test: {np.mean(y_test):.3f} up")

    # ---------------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------------
    baseline = compute_baseline(y_test)
    logger.info(f"\n  📊 NAIVE BASELINE (always predict UP):")
    logger.info(f"     Accuracy: {baseline['accuracy']:.4f} | F1: {baseline['f1']:.4f}")

    # ---------------------------------------------------------------
    # Train 3 classifiers
    # ---------------------------------------------------------------
    results = []
    cfg_t = config["training"]

    # 1. Random Forest
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
        rf, "RandomForest", X_train, X_test, y_train, y_test,
        rf_params, config, feature_cols, data_files,
    )
    results.append(result)

    # 2. XGBoost
    xgb_params = cfg_t["xgboost"]
    xgb_model = xgb.XGBClassifier(
        n_estimators=xgb_params["n_estimators"],
        max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],
        subsample=xgb_params["subsample"],
        colsample_bytree=xgb_params["colsample_bytree"],
        random_state=seed,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    result = train_and_log_classifier(
        xgb_model, "XGBoost", X_train, X_test, y_train, y_test,
        xgb_params, config, feature_cols, data_files,
    )
    results.append(result)

    # 3. Logistic Regression
    lr_params = cfg_t["logistic_regression"]
    lr = LogisticRegression(
        C=lr_params["C"],
        max_iter=lr_params["max_iter"],
        solver=lr_params["solver"],
        random_state=seed,
    )
    result = train_and_log_classifier(
        lr, "LogisticRegression", X_train, X_test, y_train, y_test,
        lr_params, config, feature_cols, data_files,
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

    best_model = joblib.load(Path(config["training"]["models_dir"]) / f"{best['model_name']}.pkl")
    fairness = fairness_analysis(best_model, best["model_name"], df, feature_cols, config)

    # ---------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("  TRAINING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  {'Model':<22} {'Acc':>7} {'F1':>7} {'ROC-AUC':>8} {'Time':>7}")
    logger.info(f"  {'-'*52}")
    logger.info(f"  {'BASELINE (always UP)':<22} {baseline['accuracy']:>7.4f} {baseline['f1']:>7.4f} {'N/A':>8} {'N/A':>7}")
    for r in results:
        logger.info(
            f"  {r['model_name']:<22} {r['accuracy']:>7.4f} {r['f1']:>7.4f} "
            f"{r['roc_auc']:>8.4f} {r['training_time']:>6.2f}s"
        )
        beat = "✅ BEATS" if r["f1"] > baseline["f1"] else "❌ LOSES TO"
        logger.info(f"    → {beat} baseline on F1")

    logger.info(f"\n  🏆 Winner: {best['model_name']}")
    logger.info("  ✅ All models saved to models/ and logged to MLflow")


if __name__ == "__main__":
    main()
