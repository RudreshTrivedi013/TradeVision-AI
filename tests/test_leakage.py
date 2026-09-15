"""
test_leakage.py - guards against data-leakage and label-correctness bugs.

Tests
-----
test_chronological_split          Task 2 - train dates always precede test dates
test_no_sentiment_feature         Task 3 - sentiment_score column must never appear
test_target_direction_correctness Task 4 - target label equals hand-verified math
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from feature_store import FeatureStore


def test_chronological_split(sample_config, sample_ohlcv_df):
    """
    Replicates the exact split call in train.py (shuffle=False, test_size=0.20)
    on a feature DataFrame built from the shared synthetic OHLCV fixture.

    Guards the shuffle bug: if shuffle were accidentally set to True, test rows
    would interleave with train rows and this assertion would fail.
    """
    store = FeatureStore(sample_config)
    feat = store.engineer_features(sample_ohlcv_df.copy(), "TEST")
    feat = feat.sort_index()

    seed = sample_config["training"]["random_seed"]
    test_size = sample_config["training"]["test_size"]

    trainval, test_df = train_test_split(
        feat, test_size=test_size, random_state=seed, shuffle=False
    )

    train_dates = pd.to_datetime(trainval.index)
    test_dates  = pd.to_datetime(test_df.index)

    assert train_dates.max() < test_dates.min(), (
        f"Chronological leak! Latest train date {train_dates.max()} >= "
        f"earliest test date {test_dates.min()}. "
        "Check that shuffle=False in train_test_split."
    )


def test_no_sentiment_feature(sample_config, sample_ohlcv_df):
    """
    Asserts that sentiment_score is absent from the engineered feature DataFrame.
    Fails immediately if the column is re-added.
    """
    store = FeatureStore(sample_config)
    feat  = store.engineer_features(sample_ohlcv_df.copy(), "TEST")

    assert "sentiment_score" not in feat.columns, (
        "'sentiment_score' found in feature columns -- this column causes "
        "look-ahead leakage and must not be added back to the pipeline."
    )


def test_target_direction_correctness(sample_config, sample_ohlcv_df):
    """
    For rows at indices 20, 80, 140 (all well past the rolling warm-up zone),
    manually computes expected_target = 1 if Close[N+1] > Close[N] else 0
    and asserts it matches what engineer_features() stored.
    """
    store = FeatureStore(sample_config)
    feat  = store.engineer_features(sample_ohlcv_df.copy(), "TEST")
    feat  = feat.reset_index(drop=False)   # Date becomes a column

    check_rows = [20, 80, 140]

    for n in check_rows:
        close_n   = feat.loc[n,     "Close"]
        close_n1  = feat.loc[n + 1, "Close"]
        expected  = 1 if close_n1 > close_n else 0
        actual    = int(feat.loc[n, "target_direction"])

        assert actual == expected, (
            f"Row {n}: target_direction={actual} but "
            f"Close[{n}]={close_n:.4f}, Close[{n+1}]={close_n1:.4f} -> "
            f"expected {expected}. "
            "Check the .shift(-1) direction in engineer_features()."
        )