"""
test_indicators.py - proves technical indicators use no future data.

Tests
-----
test_rsi_no_lookahead  Task 5 - RSI at row N equals RSI recomputed on data[0..N]
"""

import numpy as np
import pandas as pd
import pytest

from feature_store import FeatureStore


def test_rsi_no_lookahead(sample_config, sample_ohlcv_df):
    """
    Proves that rsi_14 at row N of the full series equals the rsi_14 value
    produced when the same computation runs on a TRUNCATED series ending at N.

    If someone accidentally used .shift(-k) or a centred window, the truncated
    series would disagree with the full series at the boundary row.
    """
    store = FeatureStore(sample_config)
    raw   = sample_ohlcv_df.copy()

    full_feat  = store.engineer_features(raw.copy(), "FULL")
    full_feat  = full_feat.sort_index()

    N_idx       = 50
    target_date = full_feat.index[N_idx]

    raw_sorted  = raw.sort_index()
    raw_up_to_N = raw_sorted.loc[:target_date].copy()

    assert len(raw_up_to_N) >= 27, (
        "Partial raw frame too short to compute indicators."
    )

    partial_feat = store.engineer_features(raw_up_to_N, "PARTIAL")

    rsi_col      = f"rsi_{sample_config['features']['rsi_period']}"   # rsi_14
    rsi_full     = full_feat.loc[target_date, rsi_col]
    rsi_partial  = partial_feat[rsi_col].iloc[-1]

    assert abs(rsi_full - rsi_partial) < 1e-9, (
        f"RSI lookahead detected! "
        f"Full-series rsi_14 at {target_date.date()} = {rsi_full:.6f}, "
        f"but truncated-series rsi_14 = {rsi_partial:.6f}. "
        f"Difference = {abs(rsi_full - rsi_partial):.2e}. "
        "The RSI window is reading future data."
    )