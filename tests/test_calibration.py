"""Tests for run_calibration() — FR-17..FR-22, FR-19, FR-20, AC-03."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethusd_analyzer.strategy import CalibrationResult, run_calibration


_GRID_TINY = {
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


def _make_strat_df(n: int = 600, seed: int = 0) -> pd.DataFrame:
    from tests.conftest import _make_strategy_df
    return _make_strategy_df(n=n, seed=seed)


class TestCalibrationInsufficientData:
    def test_insufficient_data_status(self):
        tiny = pd.DataFrame({"market_time": pd.date_range("2024-01-01", periods=5, freq="1min")})
        result = run_calibration(tiny, "1m", _GRID_TINY, lookback_days=0)
        assert result.status == "INSUFFICIENT_DATA"

    def test_insufficient_data_has_breakdown_zeros(self):
        tiny = pd.DataFrame({"market_time": pd.date_range("2024-01-01", periods=5, freq="1min")})
        result = run_calibration(tiny, "1m", _GRID_TINY, lookback_days=0)
        assert result.rejected_by_min_trades == 0
        assert result.rejected_by_max_dd == 0
        assert result.rejected_by_both == 0


class TestCalibrationRejectionBreakdown:
    """FR-19: rejection breakdown fields always populated."""

    def test_breakdown_sums_correctly(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_SMALL,
            min_trades=999,  # force all candidates to fail min_trades
            max_drawdown=0.15, lookback_days=0,
        )
        total = (result.rejected_by_min_trades
                 + result.rejected_by_max_dd
                 + result.rejected_by_both
                 + result.eligible_candidates)
        assert total == result.total_candidates, (
            f"Breakdown sum {total} != total_candidates {result.total_candidates}"
        )

    def test_max_trades_seen_is_nonnegative(self, df_strategy):
        result = run_calibration(df_strategy, "1m", _GRID_SMALL, lookback_days=0)
        assert result.max_trades_seen >= 0

    def test_best_dd_seen_is_bounded(self, df_strategy):
        result = run_calibration(df_strategy, "1m", _GRID_SMALL, lookback_days=0)
        assert 0.0 <= result.best_dd_seen <= 100.0, "best_dd_seen must be bounded"


class TestCalibrationNoValidParams:
    """FR-21: NO_VALID_PARAMS status when all candidates rejected."""

    def test_no_valid_params_when_min_trades_impossible(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_TINY,
            min_trades=99999,   # impossible threshold
            lookback_days=0,
        )
        assert result.status == "NO_VALID_PARAMS", f"Expected NO_VALID_PARAMS, got {result.status}"

    def test_no_valid_params_uses_default_params(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_TINY,
            min_trades=99999, lookback_days=0,
        )
        # Default params must be returned and should be a non-empty dict
        assert isinstance(result.best_params, dict) and result.best_params, \
            "Should fall back to default params"

    def test_rejection_reason_contains_counts(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_TINY,
            min_trades=99999, lookback_days=0,
        )
        assert result.rejection_reason is not None and len(result.rejection_reason) > 0


class TestCalibrationTieBreaking:
    """FR-20: tie-breaking by (Sharpe DESC, net_return DESC, max_dd ASC)."""

    def test_ok_result_has_params(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_SMALL,
            min_trades=1, max_drawdown=10.0, lookback_days=0,
        )
        if result.status == "OK":
            assert "r_min" in result.best_params
            assert "hold_bars" in result.best_params
            assert "quantile_hi" in result.best_params


class TestCalibrationMaxDdUsed:
    """FR-19/FR-28: max_dd_used field present and matches the constraint passed."""

    def test_max_dd_used_matches_argument(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_TINY,
            max_drawdown=0.12, lookback_days=0, min_trades=1,
        )
        assert result.max_dd_used == pytest.approx(0.12)


class TestCalibrationPerTfOverrides:
    """FR-18: per-timeframe overrides applied to min_trades."""

    def test_per_tf_override_raises_min_trades(self, df_strategy):
        """If per_tf_overrides sets min_trades=99999, should produce NO_VALID_PARAMS."""
        result = run_calibration(
            df_strategy, "1m", _GRID_TINY,
            lookback_days=0,
            per_tf_overrides={"min_trades": 99999},
        )
        assert result.status == "NO_VALID_PARAMS"


class TestWalkForwardCalibration:
    """FR-22: walk_forward_folds > 1 activates rolling walk-forward mode."""

    def test_walk_forward_returns_calibration_result(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_TINY,
            lookback_days=0, min_trades=1, max_drawdown=10.0,
            walk_forward_folds=2,
        )
        assert isinstance(result, CalibrationResult)

    def test_walk_forward_status_is_ok_or_no_valid(self, df_strategy):
        result = run_calibration(
            df_strategy, "1m", _GRID_TINY,
            lookback_days=0, min_trades=1, max_drawdown=10.0,
            walk_forward_folds=3,
        )
        assert result.status in ("OK", "NO_VALID_PARAMS")
