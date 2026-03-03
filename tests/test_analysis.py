"""Tests for analysis.py — FR-01..FR-06, AC-01."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethusd_analyzer.analysis import add_features, add_strategy_features


class TestRollingZ:
    """AC-01: MAD-based z-score behaviour."""

    def test_normal_z_values_are_finite(self):
        from ethusd_analyzer.analysis import rolling_z
        s = pd.Series(np.random.default_rng(0).normal(0, 1, 300))
        z = rolling_z(s, window=50)
        assert z.dropna().notnull().all(), "z-scores should be finite after warm-up"

    def test_mad_zero_returns_zero(self):
        """When MAD == 0 (all values constant), z-score must be 0, never NaN/Inf."""
        from ethusd_analyzer.analysis import rolling_z
        s = pd.Series([5.0] * 200)
        z = rolling_z(s, window=50)
        assert (z.dropna() == 0.0).all(), "Constant series → z=0, not NaN/Inf"

    def test_z_stays_bounded_with_outliers(self):
        from ethusd_analyzer.analysis import rolling_z
        s = pd.Series([1.0] * 100 + [1e6])
        z = rolling_z(s, window=50)
        assert np.isfinite(z.dropna().values).all(), "Outliers must not produce inf z-scores"


class TestAddFeatures:
    """FR-01..FR-03: feature columns present and well-formed."""

    def test_required_columns_exist(self, df_1m_raw):
        df = add_features(df_1m_raw.copy(), z_window=50)
        # score is produced by add_features; score_mr is produced by add_strategy_features
        for col in ("imbalance", "imb_change", "log_ret", "score", "fwd_ret_1"):
            assert col in df.columns, f"Missing column: {col}"

    def test_score_mr_produced_by_strategy_features(self, df_1m_raw):
        """FR-03: score_mr must equal score after add_strategy_features."""
        df = add_features(df_1m_raw.copy(), z_window=50)
        strat = add_strategy_features(df, mom_span=20, vol_window=20, regime_corr_window=50)
        assert "score_mr" in strat.columns, "score_mr must exist after add_strategy_features"
        pd.testing.assert_series_equal(df["score"].reindex(strat.index),
                                       strat["score_mr"], check_names=False)

    def test_no_infinite_values_in_scores(self, df_1m_raw):
        df = add_features(df_1m_raw.copy(), z_window=50).dropna(subset=["fwd_ret_1"])
        for col in ("score", "imb_change_z"):
            bad = np.isinf(df[col].values)
            assert not bad.any(), f"Infinite values found in {col}"


class TestAddStrategyFeatures:
    """FR-04..FR-06: strategy feature columns present."""

    def test_strategy_columns_exist(self, df_1m_raw):
        df = add_features(df_1m_raw.copy(), z_window=50)
        strat = add_strategy_features(df, mom_span=20, vol_window=20, regime_corr_window=50)
        for col in ("score_mom", "volatility", "rc", "ar"):
            assert col in strat.columns, f"Missing strategy column: {col}"

    def test_rc_ar_are_bounded(self, df_strategy):
        """FR-05/06: rolling correlations must be in [-1, 1] where finite."""
        for col in ("rc", "ar"):
            vals = df_strategy[col].dropna().values
            assert (vals >= -1.0 - 1e-9).all() and (vals <= 1.0 + 1e-9).all(), \
                f"{col} out of [-1,1] bounds"
