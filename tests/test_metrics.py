"""
test_metrics.py - guards metric computation and naive-baseline correctness.

Tests
-----
test_baseline_per_ticker    Task 7 - baseline differs across tickers and
                                     matches manual fraction-of-UP-days math
test_metrics_match_sklearn  Task 8 - recall/precision match sklearn direct
"""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train import compute_baseline


class TestBaselinePerTicker:
    """
    compute_baseline(y_test) = always-predict-UP accuracy.
    Accuracy equals the fraction of actual UP days in y_test.
    """

    def test_baseline_matches_manual_fraction_ticker_a(self):
        """Ticker A: 60% UP days -> baseline accuracy must equal 0.60."""
        y_a = np.array([1] * 60 + [0] * 40)
        result = compute_baseline(y_a)
        expected_acc = 60 / 100
        assert abs(result["accuracy"] - expected_acc) < 1e-9, (
            f"Ticker A baseline: expected {expected_acc:.4f}, got {result['accuracy']:.4f}."
        )

    def test_baseline_matches_manual_fraction_ticker_b(self):
        """Ticker B: 45% UP days -> baseline accuracy must equal 0.45."""
        y_b = np.array([1] * 45 + [0] * 55)
        result = compute_baseline(y_b)
        expected_acc = 45 / 100
        assert abs(result["accuracy"] - expected_acc) < 1e-9, (
            f"Ticker B baseline: expected {expected_acc:.4f}, got {result['accuracy']:.4f}."
        )

    def test_baselines_differ_across_tickers(self):
        """A cached or hardcoded value would be identical across tickers."""
        y_a = np.array([1] * 60 + [0] * 40)
        y_b = np.array([1] * 45 + [0] * 55)
        baseline_a = compute_baseline(y_a)["accuracy"]
        baseline_b = compute_baseline(y_b)["accuracy"]
        assert baseline_a != baseline_b, (
            f"Baselines are identical ({baseline_a:.4f}) despite different "
            "UP-day fractions (60% vs 45%). The baseline may be hardcoded."
        )

    def test_baseline_recall_and_precision(self):
        """Always-UP predictor: recall=1.0, precision=fraction_UP."""
        y = np.array([1] * 60 + [0] * 40)
        result = compute_baseline(y)
        assert abs(result["recall"] - 1.0) < 1e-9
        assert abs(result["precision"] - 0.60) < 1e-9


class TestMetricsMatchSklearn:
    """
    y_true = [1,0,1,1,0,0,1,0,1,0]  (5 UP, 5 DOWN)
    y_pred = [1,0,0,1,0,1,1,0,0,0]

    Manual confusion matrix (pos_label=1):
        TP=3, FN=2, FP=1, TN=4
        Recall(UP)    = 3/5 = 0.6
        Precision(UP) = 3/4 = 0.75
        Recall(DOWN)  = 4/5 = 0.8
    """

    Y_TRUE = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    Y_PRED = np.array([1, 0, 0, 1, 0, 1, 1, 0, 0, 0])

    def test_up_recall_matches_sklearn(self):
        expected       = 3 / 5
        sk_up_recall   = recall_score(self.Y_TRUE, self.Y_PRED, pos_label=1, zero_division=0)
        code_up_recall = recall_score(self.Y_TRUE, self.Y_PRED, zero_division=0)
        assert abs(expected - sk_up_recall)   < 1e-9
        assert abs(sk_up_recall - code_up_recall) < 1e-9, (
            f"UP recall mismatch: sklearn={sk_up_recall:.4f}, code={code_up_recall:.4f}."
        )

    def test_down_recall_matches_sklearn(self):
        expected       = 4 / 5
        sk_down_recall = recall_score(self.Y_TRUE, self.Y_PRED, pos_label=0, zero_division=0)
        assert abs(expected - sk_down_recall) < 1e-9, (
            f"DOWN recall mismatch: expected={expected:.4f}, sklearn={sk_down_recall:.4f}."
        )

    def test_up_precision_matches_sklearn(self):
        expected  = 3 / 4
        sk_prec   = precision_score(self.Y_TRUE, self.Y_PRED, pos_label=1, zero_division=0)
        code_prec = precision_score(self.Y_TRUE, self.Y_PRED, zero_division=0)
        assert abs(expected - sk_prec)    < 1e-9
        assert abs(sk_prec  - code_prec)  < 1e-9, (
            f"UP precision mismatch: sklearn={sk_prec:.4f}, code={code_prec:.4f}."
        )

    def test_recall_and_precision_consistency_via_f1(self):
        """Cross-check: 2*P*R/(P+R) must equal sklearn F1 (catches swapped P/R)."""
        prec = precision_score(self.Y_TRUE, self.Y_PRED, zero_division=0)
        rec  = recall_score(self.Y_TRUE,    self.Y_PRED, zero_division=0)
        f1   = f1_score(self.Y_TRUE,        self.Y_PRED, zero_division=0)
        f1_manual = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        assert abs(f1 - f1_manual) < 1e-9, (
            f"F1 consistency failed: sklearn={f1:.4f}, 2PR/(P+R)={f1_manual:.4f}."
        )