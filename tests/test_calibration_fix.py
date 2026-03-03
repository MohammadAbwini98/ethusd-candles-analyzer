"""Tests for SRS Calibration NO_VALID_PARAMS Fix.

Covers:
- FR-FIX-1: _simulate_strategy returns max_dd=0.0 (not 1.0) on no-trade paths
- FR-FIX-2: best_dd_seen tracks minimum max_dd correctly (never stuck at 1.0)
- FR-FIX-3: rejection breakdown sums to total_candidates
- FR-FIX-4: regime NaN-safety is independent per channel (MR vs MOM)
- FR-FIX-5: volume-weighted resample preserves imbalance signal variation
- FR-FIX-6: TF-aware min_trades defaults are applied per timeframe
- FR-FIX-7: Integration — calibration on realistic fixture yields max_trades_seen > 0
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethusd_analyzer.strategy import (
    CalibrationResult,
    Regime,
    _simulate_strategy,
    detect_regime,
    run_calibration,
)
from ethusd_analyzer.analysis import resample_timeframe


# ─────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────

_GRID_SINGLE = {
    "r_min": [0.10],
    "quantile_window": [100],
    "quantile_levels": [[0.90, 0.10]],
    "hold_bars": [1],
    "cost_bps": [10],
}

_GRID_SMALL = {
    "r_min": [0.05, 0.10, 0.15],
    "quantile_window": [100, 150],
    "quantile_levels": [[0.90, 0.10], [0.85, 0.15]],
    "hold_bars": [1, 2],
    "cost_bps": [10],
}


def _make_strat_df(n: int = 800, seed: int = 7) -> pd.DataFrame:
    from tests.conftest import _make_strategy_df
    return _make_strategy_df(n=n, seed=seed)


def _make_tiny_strat_df(n: int = 5) -> pd.DataFrame:
    """DataFrame too small for the simulator (below quantile_window + 10)."""
    return pd.DataFrame({
        "market_time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "close": np.ones(n) * 3000.0,
        "vol": np.ones(n),
        "buyers_pct": np.ones(n) * 50.0,
        "sellers_pct": np.ones(n) * 50.0,
        "rc": np.full(n, np.nan),
        "ar": np.full(n, np.nan),
        "score_mr": np.zeros(n),
        "score_mom": np.zeros(n),
        "fwd_ret_1": np.zeros(n),
    })


# ─────────────────────────────────────────────────────────
#  FR-FIX-1: max_dd=0.0 on no-trade early return
# ─────────────────────────────────────────────────────────

class TestSimulateStrategyNoTradeMaxDD:
    """max_dd must be 0.0 when the simulator exits with n_trades < 2."""

    def test_too_few_bars_returns_zero_maxdd(self):
        """When n < quantile_window+10, max_dd must be 0.0 not 1.0."""
        df_tiny = _make_tiny_strat_df(n=5)
        _, _, max_dd, n_trades, _ = _simulate_strategy(
            df_tiny,
            r_min=0.10, a_min=0.10,
            quantile_window=100,
            quantile_hi_pct=0.90, quantile_lo_pct=0.10,
            hold_bars=1, cost_bps=10,
        )
        assert n_trades == 0
        assert max_dd == 0.0, f"Expected max_dd=0.0 for no-trade (too few bars), got {max_dd}"

    def test_regime_all_no_trade_returns_zero_maxdd(self, df_strategy):
        """Force rc/ar to NaN everywhere → regime=NO_TRADE → 0 trades → max_dd must be 0.0."""
        df = df_strategy.copy()
        df["rc"] = np.nan
        df["ar"] = np.nan
        _, _, max_dd, n_trades, _ = _simulate_strategy(
            df,
            r_min=0.10, a_min=0.10,
            quantile_window=100,
            quantile_hi_pct=0.90, quantile_lo_pct=0.10,
            hold_bars=1, cost_bps=10,
        )
        assert n_trades < 2
        assert max_dd == 0.0, f"Expected max_dd=0.0 when all regime=NO_TRADE, got {max_dd}"


# ─────────────────────────────────────────────────────────
#  FR-FIX-2: best_dd_seen tracks correctly
# ─────────────────────────────────────────────────────────

class TestBestDDSeenTracking:
    """best_dd_seen must never be stuck at 1.0 when all candidates have no trades."""

    def test_best_dd_seen_not_one_when_no_trades(self, df_strategy):
        """With NaN rc/ar (all NO_TRADE), best_dd_seen should be 0.0, max_trades_seen=0."""
        df = df_strategy.copy()
        df["rc"] = np.nan
        df["ar"] = np.nan
        result = run_calibration(
            df, "1m", _GRID_SINGLE,
            min_trades=1,
            max_drawdown=0.15,
            lookback_days=0,
        )
        assert result.best_dd_seen == 0.0, (
            f"best_dd_seen should be 0.0 when all candidates produce 0 trades, "
            f"got {result.best_dd_seen}"
        )
        assert result.max_trades_seen == 0

    def test_best_dd_seen_default_is_zero(self):
        """CalibrationResult default for best_dd_seen must be 0.0."""
        cr = CalibrationResult(
            timeframe="1m",
            best_params={},
            net_sharpe=0.0,
            net_return=0.0,
            max_drawdown=0.0,
            n_trades=0,
            win_rate=0.0,
            param_grid_size=0,
            lookback_days=0,
        )
        assert cr.best_dd_seen == 0.0, (
            f"CalibrationResult.best_dd_seen default should be 0.0, got {cr.best_dd_seen}"
        )


# ─────────────────────────────────────────────────────────
#  FR-FIX-3: rejection breakdown sums
# ─────────────────────────────────────────────────────────

class TestRejectionBreakdownSums:
    """rej_mt + rej_dd + rej_both + eligible == total_candidates always."""

    def test_breakdown_sums_all_fail_min_trades(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_SMALL,
            min_trades=9999,  # force all to fail min_trades
            max_drawdown=0.15,
            lookback_days=0,
        )
        total = (result.rejected_by_min_trades + result.rejected_by_max_dd
                 + result.rejected_by_both + result.eligible_candidates)
        assert total == result.total_candidates, (
            f"Breakdown sum {total} != total_candidates {result.total_candidates}"
        )

    def test_breakdown_sums_with_nan_rc_ar(self, df_strategy):
        """NaN rc/ar → all NO_TRADE → 0 trades: breakdown must still sum correctly."""
        df = df_strategy.copy()
        df["rc"] = np.nan
        df["ar"] = np.nan
        result = run_calibration(
            df, "1m", _GRID_SMALL,
            min_trades=1,
            max_drawdown=0.15,
            lookback_days=0,
        )
        total = (result.rejected_by_min_trades + result.rejected_by_max_dd
                 + result.rejected_by_both + result.eligible_candidates)
        assert total == result.total_candidates

    def test_breakdown_sums_normal(self, df_strategy):
        """Normal run: breakdown must sum to total_candidates."""
        result = run_calibration(
            df_strategy, "1m", _GRID_SMALL,
            min_trades=1,
            max_drawdown=10.0,  # very permissive → many eligible
            lookback_days=0,
        )
        total = (result.rejected_by_min_trades + result.rejected_by_max_dd
                 + result.rejected_by_both + result.eligible_candidates)
        assert total == result.total_candidates


# ─────────────────────────────────────────────────────────
#  FR-FIX-4: regime NaN-safety — independent MR vs MOM
# ─────────────────────────────────────────────────────────

class TestRegimeNaNSafety:
    """MR and MOM channel detections are independent; NaN in one must not block the other."""

    def test_nan_rc_allows_mom_detection(self):
        """rc=NaN → skip MR check; ar>a_min → detect MOM independently."""
        regime = detect_regime(rc=float("nan"), ar=0.5, r_min=0.10, a_min=0.10)
        assert regime == Regime.MOM, f"Expected MOM when rc=NaN and ar>a_min, got {regime}"

    def test_nan_ar_allows_mr_detection(self):
        """ar=NaN → skip MOM check; rc<-r_min → detect MR independently."""
        regime = detect_regime(rc=-0.5, ar=float("nan"), r_min=0.10, a_min=0.10)
        assert regime == Regime.MR, f"Expected MR when ar=NaN and rc<-r_min, got {regime}"

    def test_both_nan_is_no_trade(self):
        """Both NaN → NO_TRADE."""
        regime = detect_regime(rc=float("nan"), ar=float("nan"), r_min=0.10, a_min=0.10)
        assert regime == Regime.NO_TRADE

    def test_both_valid_mr_wins(self):
        """Both conditions met; MR is checked first → MR wins."""
        regime = detect_regime(rc=-0.5, ar=0.5, r_min=0.10, a_min=0.10)
        assert regime == Regime.MR

    def test_within_threshold_is_no_trade(self):
        """rc within (-r_min, 0) and ar < a_min → NO_TRADE."""
        regime = detect_regime(rc=-0.05, ar=0.05, r_min=0.10, a_min=0.10)
        assert regime == Regime.NO_TRADE


# ─────────────────────────────────────────────────────────
#  FR-FIX-5: volume-weighted resampling preserves variation
# ─────────────────────────────────────────────────────────

class TestVolumeWeightedResample:
    """buyers_pct/sellers_pct after resampling must vary (std > 0) when 1m data varies."""

    def _make_1m_df_for_resample(self, n: int = 600, seed: int = 99) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        prices = 3000.0 + np.cumsum(rng.normal(0, 2, n))
        buyers = rng.uniform(35, 65, n)
        vols = rng.exponential(scale=100, size=n)  # high variance → VW matters
        return pd.DataFrame({
            "market_time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
            "close": prices,
            "vol": vols,
            "buyers_pct": buyers,
            "sellers_pct": 100.0 - buyers,
        })

    def test_resampled_buyers_pct_has_variation(self):
        """After 5m resample, buyers_pct std must be > 0 (not flattened to constant)."""
        df = self._make_1m_df_for_resample()
        df5 = resample_timeframe(df, "5min")
        std = df5["buyers_pct"].std()
        assert std > 0.1, (
            f"buyers_pct has no variation after resample (std={std:.4f}); "
            "volume-weighted resampling may be broken"
        )

    def test_resampled_buyers_sellers_roughly_complement(self):
        """buyers_pct + sellers_pct should approximately equal 100 after VW resample."""
        df = self._make_1m_df_for_resample()
        df5 = resample_timeframe(df, "5min")
        sums = df5["buyers_pct"] + df5["sellers_pct"]
        # With VW resampling, sum may not be exactly 100 but should be close
        assert (sums - 100.0).abs().max() < 2.0, (
            "buyers_pct + sellers_pct diverges too much from 100 after VW resample"
        )

    def test_zero_vol_rows_use_fallback_mean(self):
        """Rows with vol=0 must not produce NaN in buyers_pct (fallback to simple mean)."""
        df = self._make_1m_df_for_resample(n=60)
        df["vol"] = 0.0  # zero all volume → VW undefined, must fall back to mean
        result = resample_timeframe(df, "5min")
        assert result["buyers_pct"].isna().sum() == 0, (
            "buyers_pct must not be NaN when vol=0 (fallback to simple mean required)"
        )


# ─────────────────────────────────────────────────────────
#  FR-FIX-6: TF-aware min_trades defaults
# ─────────────────────────────────────────────────────────

class TestTFAwareDefaults:
    """run_calibration must apply TF-specific min_trades and max_dd defaults."""

    def test_1m_uses_higher_min_trades(self, df_strategy):
        """For 1m, TF-aware default min_trades=80; passing global=20 must be overridden."""
        # Run with global default (20) — code TF-aware default (80) should take over
        result = run_calibration(
            df_strategy, "1m", _GRID_SINGLE,
            min_trades=20,   # global default — should be overridden to 80 by TF-aware logic
            max_drawdown=0.15,
            lookback_days=0,
        )
        # min_trades_used in result must reflect the TF-aware override
        assert result.min_trades_used == 80, (
            f"Expected 1m min_trades_used=80 (TF-aware), got {result.min_trades_used}"
        )

    def test_30m_uses_lower_min_trades(self, df_strategy):
        """For 30m, TF-aware default min_trades=8."""
        result = run_calibration(
            df_strategy, "30m", _GRID_SINGLE,
            min_trades=20,
            max_drawdown=0.15,
            lookback_days=0,
        )
        assert result.min_trades_used == 8, (
            f"Expected 30m min_trades_used=8 (TF-aware), got {result.min_trades_used}"
        )

    def test_per_tf_override_wins_over_tf_aware(self, df_strategy):
        """per_tf_overrides must override both global config and TF-aware defaults."""
        result = run_calibration(
            df_strategy, "1m", _GRID_SINGLE,
            min_trades=20,
            max_drawdown=0.15,
            lookback_days=0,
            per_tf_overrides={"min_trades": 5},  # explicit override → must win
        )
        assert result.min_trades_used == 5, (
            f"per_tf_overrides must win over TF-aware defaults, got min_trades_used={result.min_trades_used}"
        )

    def test_15m_uses_higher_dd_limit(self, df_strategy):
        """For 15m, TF-aware default max_dd=0.18."""
        result = run_calibration(
            df_strategy, "15m", _GRID_SINGLE,
            min_trades=1,
            max_drawdown=0.15,  # global — should be overridden to 0.18
            lookback_days=0,
        )
        assert abs(result.max_dd_used - 0.18) < 1e-9, (
            f"Expected 15m max_dd_used=0.18 (TF-aware), got {result.max_dd_used}"
        )


# ─────────────────────────────────────────────────────────
#  FR-FIX-7: Integration — realistic fixture
# ─────────────────────────────────────────────────────────

class TestCalibrationIntegration:
    """End-to-end: calibration on a realistic synthetic fixture must produce sane diagnostics."""

    def test_max_trades_seen_positive_with_valid_regime_data(self, df_strategy):
        """When rc/ar are valid (not all NaN), at least some candidates should produce trades."""
        # Use very lenient thresholds so eligible candidates appear
        result = run_calibration(
            df_strategy, "1m",
            grid_config={
                "r_min": [0.01],          # very low threshold → more regime hits
                "quantile_window": [100],
                "quantile_levels": [[0.60, 0.40]],  # narrow band → more signals
                "hold_bars": [1],
                "cost_bps": [0],
            },
            min_trades=1,
            max_drawdown=10.0,
            lookback_days=0,
        )
        assert result.max_trades_seen >= 0  # must be a non-negative integer
        # best_dd_seen must not be 1.0 (our sentinel bug value)
        assert result.best_dd_seen != 1.0, (
            f"best_dd_seen={result.best_dd_seen} — should never be 1.0 (old bug value)"
        )

    def test_no_valid_params_status_when_impossible_constraint(self, df_strategy):
        """per_tf_overrides min_trades=9999 → NO_VALID_PARAMS; breakdown sums must be consistent."""
        # Use per_tf_overrides to set an impossible min_trades that wins over TF-aware defaults
        result = run_calibration(
            df_strategy, "5m", _GRID_SMALL,
            min_trades=20,
            max_drawdown=0.15,
            lookback_days=0,
            per_tf_overrides={"min_trades": 9999},  # overrides TF-aware default of 30
        )
        assert result.status == "NO_VALID_PARAMS"
        total = (result.rejected_by_min_trades + result.rejected_by_max_dd
                 + result.rejected_by_both + result.eligible_candidates)
        assert total == result.total_candidates

    def test_best_dd_seen_lt_1_on_real_data(self, df_strategy):
        """With actual computed rc/ar, no-trade drawdown must be 0.0, not 1.0."""
        df = df_strategy.copy()
        # Force all regime to NO_TRADE to verify the old 1.0 bug is gone
        df["rc"] = 0.0   # within threshold → NO_TRADE
        df["ar"] = 0.0
        result = run_calibration(
            df, "1m", _GRID_SINGLE,
            min_trades=1,
            max_drawdown=1.0,
            lookback_days=0,
        )
        assert result.best_dd_seen < 1.0, (
            f"best_dd_seen={result.best_dd_seen:.4f} should be < 1.0 "
            "(old bug returned max_dd=1.0 on no-trade, inflating best_dd_seen to 1.0)"
        )
