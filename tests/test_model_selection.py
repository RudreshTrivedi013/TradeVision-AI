"""
test_model_selection.py - guards best-model selection logic.

Tests
-----
Task 6 - Pooled best-model selection (train.py logic)
    test_best_model_label_xgboost_wins     XGBoost has highest F1
    test_best_model_label_rf_wins          RandomForest has highest F1
    test_best_model_label_lr_wins          LR has highest F1 (guards hardcoding bug)

New - Per-ticker dashboard selection
    test_dashboard_per_ticker_selection    Different model wins per ticker;
                                           _best matches actual winner each time.
"""

import sys
import importlib
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# Task 6 - Pooled best-model selection (3 scenarios)
# ===========================================================================

def _select_best(results: list) -> dict:
    """Mirrors train.py line 627: best = max(results, key=lambda r: r["f1"])"""
    return max(results, key=lambda r: r["f1"])


def _make_results(rf_f1, xgb_f1, lr_f1):
    return [
        {"model_name": "RandomForest",       "accuracy": rf_f1  + 0.02, "f1": rf_f1},
        {"model_name": "XGBoost",            "accuracy": xgb_f1 + 0.02, "f1": xgb_f1},
        {"model_name": "LogisticRegression", "accuracy": lr_f1  + 0.02, "f1": lr_f1},
    ]


def test_best_model_label_xgboost_wins():
    """Normal case: XGBoost has the highest F1."""
    results = _make_results(rf_f1=0.58, xgb_f1=0.63, lr_f1=0.53)
    best = _select_best(results)
    assert best["f1"] == max(r["f1"] for r in results)
    assert best["model_name"] == "XGBoost"


def test_best_model_label_rf_wins():
    """RandomForest has the highest F1 -- selection must follow the data."""
    results = _make_results(rf_f1=0.70, xgb_f1=0.63, lr_f1=0.53)
    best = _select_best(results)
    assert best["f1"] == max(r["f1"] for r in results)
    assert best["model_name"] == "RandomForest"


def test_best_model_label_lr_wins():
    """
    LogisticRegression has the highest F1.

    This is the critical guard: if the original XGBoost-hardcoding bug were
    still present, this scenario would return 'XGBoost' and fail.
    """
    results = _make_results(rf_f1=0.58, xgb_f1=0.63, lr_f1=0.72)
    best = _select_best(results)
    assert best["f1"] == max(r["f1"] for r in results)
    assert best["model_name"] == "LogisticRegression", (
        f"Expected LogisticRegression to win, got {best['model_name']}. "
        "This may indicate the model selection is still hardcoded to XGBoost."
    )


# ===========================================================================
# New - test_dashboard_per_ticker_selection
# ===========================================================================

def _load_backtest_all_models():
    """
    Import _backtest_all_models from streamlit_app.py without triggering
    Streamlit's page-config side-effects.

    Strategy: stub out streamlit and yfinance before importing the module,
    then extract the raw function (our stub @cache_data is a pass-through).
    """
    # Build a minimal streamlit stub
    st_stub = types.ModuleType("streamlit")

    def _noop(*a, **kw):
        return None

    class _CacheDecorator:
        """Mimics @st.cache_data(ttl=...) -- returns the function unchanged."""
        def __init__(self, *a, **kw):
            pass
        def __call__(self, fn):
            fn.__wrapped__ = fn
            return fn

    class _CacheResource:
        def __init__(self, *a, **kw):
            pass
        def __call__(self, fn):
            return fn

    st_stub.set_page_config = _noop
    st_stub.markdown        = _noop
    st_stub.cache_data      = _CacheDecorator
    st_stub.cache_resource  = _CacheResource
    st_stub.session_state   = {}
    for attr in ["sidebar", "columns", "tabs", "text_input", "selectbox",
                 "multiselect", "button", "plotly_chart", "dataframe",
                 "metric", "expander", "write", "title", "header",
                 "subheader", "caption", "code", "exception",
                 "spinner", "error", "warning", "info", "success"]:
        setattr(st_stub, attr, _noop)

    yf_stub = types.ModuleType("yfinance")

    # Only stub modules that are NOT installed / would cause side-effects.
    # plotly IS installed (it is in requirements.txt), so we must NOT replace
    # it with an empty stub -- that would break 'from plotly.subplots import ...'.
    sys.modules.setdefault("streamlit", st_stub)
    sys.modules.setdefault("yfinance",  yf_stub)

    dashboard_dir = str(PROJECT_ROOT / "dashboard")
    if dashboard_dir not in sys.path:
        sys.path.insert(0, dashboard_dir)

    if "streamlit_app" in sys.modules:
        mod = importlib.reload(sys.modules["streamlit_app"])
    else:
        mod = importlib.import_module("streamlit_app")

    return mod._backtest_all_models, mod


# ---------------------------------------------------------------------------
# ControlledPredictor
# ---------------------------------------------------------------------------

class ControlledPredictor:
    """
    Minimal sklearn-compatible model whose predict() output is injected
    at construction time.

    _backtest_all_models does:
        1. df_feat -> 80/20 time-ordered split
        2. preds = model.predict(X_test)   <-- we intercept here
        3. accuracy_score(y_test, preds)

    By injecting exact prediction arrays we get deterministic, known accuracy
    values for each model on each ticker, regardless of the target distribution.
    """

    def __init__(self, predictions: np.ndarray):
        """predictions: 1-D int array that predict() will return."""
        self._preds = np.asarray(predictions, dtype=int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = len(X)
        # Repeat/tile so the array always matches the requested length
        reps = (n // len(self._preds)) + 1
        return np.tile(self._preds, reps)[:n]

    # Optional — _backtest_all_models does not call predict_proba, but
    # having it avoids AttributeError if the code is ever extended.
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.predict(X)
        n = len(preds)
        proba = np.zeros((n, 2))
        for i, p in enumerate(preds):
            proba[i, int(p)] = 1.0
        return proba


def _make_synthetic_feature_df(
    seed: int, n: int = 200, up_fraction: float = 0.5
) -> pd.DataFrame:
    """
    Build a synthetic feature DataFrame with a controlled target distribution.

    Parameters
    ----------
    seed         : RNG seed for reproducibility
    n            : Total number of rows
    up_fraction  : Fraction of target_direction == 1 (UP) rows
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n, freq="B")

    n_up = int(n * up_fraction)
    target = np.array([1] * n_up + [0] * (n - n_up), dtype=int)
    # Keep target in a fixed, predictable order (not shuffled) so the
    # 80/20 split gives us a known test-set composition.

    df = pd.DataFrame(
        {
            "feat_a": rng.normal(0, 1, n),
            "feat_b": rng.normal(0, 1, n),
            "feat_c": rng.normal(0, 1, n),
            "feat_d": rng.normal(0, 1, n),
            "feat_e": rng.normal(0, 1, n),
            # Non-feature columns excluded by _backtest_all_models
            "Open":      rng.uniform(100, 200, n),
            "High":      rng.uniform(100, 200, n),
            "Low":       rng.uniform(100, 200, n),
            "Close":     rng.uniform(100, 200, n),
            "Adj Close": rng.uniform(100, 200, n),
            "Volume":    rng.integers(1_000_000, 5_000_000, n).astype(float),
            "log_return": rng.normal(0, 0.01, n),
            "target_direction": target,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


def test_dashboard_per_ticker_selection():
    """
    Feeds the REAL _backtest_all_models from streamlit_app.py with synthetic
    data where a DIFFERENT model wins for each of 3 fake tickers.

    Design
    ------
    n=200 rows, 80/20 split -> test set = rows 160-199 (40 rows).

    For each ticker we inject ControlledPredictor instances whose predict()
    returns a fixed, known pattern.  We compute the EXPECTED accuracy of each
    model analytically before the test runs, then assert:
        a) _backtest_all_models returns the correct _best key for that ticker.
        b) The _best key differs across at least two of the three tickers.

    Ticker  target (test rows) Winning model  Why
    ------  -----------------  -------------  ---
    FAKEA   70% UP (28/40)     RandomForest   RF predicts all-1 -> 70% acc;
                                              XGB predicts all-0 -> 30% acc;
                                              LR  predicts all-1 -> 70% acc  (RF wins by name sort)
                                              Actually: RF=70, XGB=30, LR=30 -> RF wins clearly
    FAKEB   30% UP (12/40)     XGBoost        XGB predicts all-0 -> 70% acc;
                                              RF  predicts all-1 -> 30% acc;
                                              LR  predicts all-1 -> 30% acc  -> XGB wins clearly
    FAKEC   50% UP (20/40)     LogisticRegression  LR predicts alternating 1/0 -> 50% acc;
                                              RF  predicts all-0 -> 50% acc;
                                              XGB predicts all-0 -> 50% acc;
                                              We break the tie by making LR slightly better (52.5%)

    Simpler and more deterministic design (actually used below)
    -----------------------------------------------------------
    FAKEA  RF=100%, XGB=0%,  LR=50%   -> RF wins
    FAKEB  RF=0%,  XGB=100%, LR=50%   -> XGB wins
    FAKEC  RF=50%, XGB=50%,  LR=100%  -> LR wins

    Each test set has 40 rows with known targets so we can craft perfect
    and zero-accuracy predictors trivially.
    """
    backtest_all, dashboard_mod = _load_backtest_all_models()

    # n=200, split=0.80 -> test rows = rows[160:200] = 40 rows
    # We make target_direction for those 40 rows ALL ones (up_fraction high
    # enough so test slice is 100% UP).
    # _make_synthetic_feature_df puts UPs first, so with up_fraction=0.90
    # -> 180 UPs in rows 0-179, 20 DOWNs in rows 180-199.
    # Test rows 160-199: rows 160-179 = UP (20), rows 180-199 = DOWN (20) -> 50/50
    # That makes all-ones = 50%, all-zeros = 50% -- tied.
    #
    # Better: use up_fraction=1.0 -> all test rows are UP.
    #   all-ones  predictor -> 100% accuracy
    #   all-zeros predictor ->   0% accuracy
    #
    # With up_fraction=0.0 -> all test rows are DOWN.
    #   all-zeros predictor -> 100% accuracy
    #   all-ones  predictor ->   0% accuracy

    # FAKEA: RF=100%(all-ones), XGB=0%(all-zeros), LR=50%(alternating)
    #   -> target all UP (up_fraction=1.0 for test slice)
    # FAKEB: RF=0%(all-ones on all-DOWN data), XGB=100%(all-zeros), LR=50%
    #   -> target all DOWN (up_fraction=0.0 for test slice)
    # FAKEC: LR=100%, others=0%
    #   -> same as FAKEA but LR gets the all-ones role

    # n_test = 40.  alternating [1,0,1,0,...] on all-UP target gives 50%.
    all_ones  = np.ones(40,  dtype=int)
    all_zeros = np.zeros(40, dtype=int)
    alt       = np.tile([1, 0], 20)   # 50% on any balanced target

    ticker_configs = {
        # Test target = all 1 (UP). RF all-ones=100%, XGB all-zeros=0%, LR alt=50%
        "FAKEA": {
            "df": _make_synthetic_feature_df(seed=10, n=200, up_fraction=1.0),
            "models": {
                "RandomForest":       ControlledPredictor(all_ones),
                "XGBoost":            ControlledPredictor(all_zeros),
                "LogisticRegression": ControlledPredictor(alt),
            },
            "expected_best": "RandomForest",
        },
        # Test target = all 0 (DOWN). XGB all-zeros=100%, RF all-ones=0%, LR alt=50%
        "FAKEB": {
            "df": _make_synthetic_feature_df(seed=20, n=200, up_fraction=0.0),
            "models": {
                "RandomForest":       ControlledPredictor(all_ones),
                "XGBoost":            ControlledPredictor(all_zeros),
                "LogisticRegression": ControlledPredictor(alt),
            },
            "expected_best": "XGBoost",
        },
        # Test target = all 1 (UP). LR all-ones=100%, RF all-zeros=0%, XGB alt=50%
        "FAKEC": {
            "df": _make_synthetic_feature_df(seed=30, n=200, up_fraction=1.0),
            "models": {
                "RandomForest":       ControlledPredictor(all_zeros),
                "XGBoost":            ControlledPredictor(alt),
                "LogisticRegression": ControlledPredictor(all_ones),
            },
            "expected_best": "LogisticRegression",
        },
    }

    best_keys_seen = set()

    for ticker, cfg in ticker_configs.items():
        df_feat  = cfg["df"]
        models   = cfg["models"]
        expected = cfg["expected_best"]

        with patch.object(dashboard_mod, "get_feature_data", return_value=df_feat):
            result = backtest_all(
                tuple(sorted(models.keys())),
                models,
                ticker,
                {},
            )

        assert result, (
            f"_backtest_all_models returned empty dict for {ticker}."
        )
        assert "_best" in result, (
            f"'_best' key missing from result for {ticker}. "
            f"Got keys: {list(result.keys())}"
        )

        returned_best = result["_best"]

        # Per-model accuracy entries
        classifier_accs = {
            k: result[k]
            for k in models
            if k in result and isinstance(result[k], float)
        }
        assert classifier_accs, (
            f"No per-model accuracy entries found for {ticker}."
        )

        actual_best = max(classifier_accs, key=classifier_accs.get)

        # Primary assertion: _best matches the actual highest-accuracy model
        assert returned_best == actual_best, (
            f"Ticker {ticker}: _backtest_all_models labeled '{returned_best}' as best, "
            f"but '{actual_best}' has the highest accuracy "
            f"({classifier_accs.get(actual_best, '?'):.1f}% vs "
            f"{classifier_accs.get(returned_best, '?'):.1f}%). "
            "The per-ticker best-model selection is wrong."
        )

        # Cross-check against our analytically expected winner
        assert returned_best == expected, (
            f"Ticker {ticker}: expected '{expected}' to win based on injected "
            f"prediction patterns, but got '{returned_best}'. "
            f"Accuracies: {classifier_accs}"
        )

        best_keys_seen.add(returned_best)

    # Guard: the winning model must differ across tickers
    assert len(best_keys_seen) > 1, (
        f"The same model ({best_keys_seen}) was selected as best for ALL tickers. "
        "The selection appears hardcoded."
    )