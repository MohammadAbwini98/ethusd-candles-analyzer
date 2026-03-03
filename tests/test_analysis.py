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


class TestNoLookaheadRc:
    """Verify rc uses no forward information and latest bar is preserved."""

    def _build_full(self, n: int = 300):
        """Return add_strategy_features output WITHOUT dropping fwd_ret_1 first."""
        from tests.conftest import _make_1m_df
        raw = _make_1m_df(n=n)
        feat = add_features(raw, z_window=50)          # full frame, last row fwd_ret_1=NaN
        return add_strategy_features(feat, mom_span=20, vol_window=20, regime_corr_window=50)

    def test_rc_valid_at_last_bar(self):
        """rc must be non-NaN for the last bar once history >= regime_corr_window+1."""
        strat = self._build_full(n=300)
        assert pd.isna(strat["fwd_ret_1"].iloc[-1]), "last bar fwd_ret_1 must be NaN"
        last_rc = strat["rc"].iloc[-1]
        assert pd.notna(last_rc), (
            f"rc at last bar should be finite (got {last_rc}); "
            "rc must not depend on forward returns"
        )

    def test_rc_does_not_use_fwd_ret(self):
        """rc values must be identical whether fwd_ret_1 is present or zeroed out."""
        from tests.conftest import _make_1m_df
        raw = _make_1m_df(n=300)
        feat = add_features(raw, z_window=50)
        strat_normal = add_strategy_features(
            feat, mom_span=20, vol_window=20, regime_corr_window=50
        )
        # Corrupt fwd_ret_1 — rc must be unchanged
        feat_corrupted = feat.copy()
        feat_corrupted["fwd_ret_1"] = 999.0
        strat_corrupted = add_strategy_features(
            feat_corrupted, mom_span=20, vol_window=20, regime_corr_window=50
        )
        pd.testing.assert_series_equal(
            strat_normal["rc"].reset_index(drop=True),
            strat_corrupted["rc"].reset_index(drop=True),
            check_names=False,
            rtol=1e-10,
        )

    def test_build_timeframes_last_row_is_newest_candle(self):
        """build_timeframes() must return frames whose last row == newest input bar."""
        from ethusd_analyzer.run import build_timeframes
        from tests.conftest import _make_1m_df
        raw = _make_1m_df(n=300)
        # build_timeframes expects a "market_time" column (same as fetch_candles output)
        raw_mt = raw.rename(columns={"market_time": "market_time"})  # already correct
        feats = build_timeframes(raw_mt, z_window=50, resamples=[])
        df1m = feats["1m"]
        expected_last_time = raw_mt["market_time"].max()
        actual_last_time   = df1m["market_time"].iloc[-1]
        assert actual_last_time == expected_last_time, (
            f"Last row market_time {actual_last_time} != newest bar {expected_last_time}; "
            "build_timeframes() must not drop the latest bar"
        )

    def test_dropna_eval_is_one_shorter_than_full(self):
        """df_eval trimmed on fwd_ret_1 should be exactly 1 row shorter than full frame."""
        from ethusd_analyzer.run import build_timeframes
        from tests.conftest import _make_1m_df
        raw = _make_1m_df(n=300)
        feats = build_timeframes(raw, z_window=50, resamples=[])
        df_full = feats["1m"]
        df_eval = df_full.dropna(subset=["fwd_ret_1"])
        assert len(df_full) - len(df_eval) == 1, (
            f"Expected exactly 1 row difference (the latest bar), "
            f"got full={len(df_full)} eval={len(df_eval)}"
        )


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
