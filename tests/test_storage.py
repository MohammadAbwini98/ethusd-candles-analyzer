"""Tests for storage.py — FR-27, FR-28, FR-30, NFR-01, AC-04."""
from __future__ import annotations

import json
import math
import pytest


class TestSanitize:
    """FR-34 (via storage): NaN/Inf must not reach JSON payloads."""

    def test_nan_replaced_by_none(self):
        from ethusd_analyzer.dashboard_server import _sanitize
        result = _sanitize({"val": float("nan")})
        assert result["val"] is None

    def test_inf_replaced_by_none(self):
        from ethusd_analyzer.dashboard_server import _sanitize
        result = _sanitize({"val": float("inf")})
        assert result["val"] is None

    def test_negative_inf_replaced_by_none(self):
        from ethusd_analyzer.dashboard_server import _sanitize
        result = _sanitize({"val": float("-inf")})
        assert result["val"] is None

    def test_normal_float_intact(self):
        from ethusd_analyzer.dashboard_server import _sanitize
        result = _sanitize({"val": 3.14})
        assert result["val"] == pytest.approx(3.14)

    def test_nested_sanitization(self):
        from ethusd_analyzer.dashboard_server import _sanitize
        result = _sanitize({"a": {"b": float("nan")}, "c": [float("inf"), 1.0]})
        assert result["a"]["b"] is None
        assert result["c"][0] is None
        assert result["c"][1] == pytest.approx(1.0)

    def test_valid_json_after_sanitize(self):
        from ethusd_analyzer.dashboard_server import _sanitize
        dirty = {"x": float("nan"), "y": float("inf"), "z": 42.0}
        clean = _sanitize(dirty)
        # Must not raise
        encoded = json.dumps(clean)
        decoded = json.loads(encoded)
        assert decoded["z"] == 42.0


class TestCalibrationResultBreakdownFields:
    """FR-19: CalibrationResult dataclass has all required breakdown fields."""

    def test_calibration_result_has_breakdown_attrs(self):
        from ethusd_analyzer.strategy import CalibrationResult
        cr = CalibrationResult(
            timeframe="5m",
            best_params={},
            net_sharpe=1.0, net_return=0.05, max_drawdown=0.08,
            n_trades=50, win_rate=0.55,
            param_grid_size=81, lookback_days=7,
        )
        assert hasattr(cr, "rejected_by_min_trades")
        assert hasattr(cr, "rejected_by_max_dd")
        assert hasattr(cr, "rejected_by_both")
        assert hasattr(cr, "max_trades_seen")
        assert hasattr(cr, "best_dd_seen")
        assert hasattr(cr, "max_dd_used")

    def test_calibration_result_defaults_to_zero(self):
        from ethusd_analyzer.strategy import CalibrationResult
        cr = CalibrationResult(
            timeframe="5m", best_params={},
            net_sharpe=0.0, net_return=0.0, max_drawdown=0.0,
            n_trades=0, win_rate=0.0, param_grid_size=0, lookback_days=0,
        )
        assert cr.rejected_by_min_trades == 0
        assert cr.rejected_by_max_dd == 0
        assert cr.rejected_by_both == 0
        assert cr.max_trades_seen == 0


class TestWithRetry:
    """NFR-01: _with_retry wraps OperationalError with backoff."""

    def test_succeeds_on_first_try(self):
        from ethusd_analyzer.storage import _with_retry
        result = _with_retry(lambda: 42)
        assert result == 42

    def test_retries_on_operational_error(self):
        from ethusd_analyzer.storage import _with_retry
        from sqlalchemy.exc import OperationalError

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 2:
                raise OperationalError("connection", {}, Exception("transient"))
            return "ok"

        result = _with_retry(flaky, retries=3, base_delay=0.001)
        assert result == "ok"
        assert calls["n"] == 2

    def test_raises_after_all_retries_exhausted(self):
        from ethusd_analyzer.storage import _with_retry
        from sqlalchemy.exc import OperationalError

        def always_fail():
            raise OperationalError("fail", {}, Exception("permanent"))

        with pytest.raises(OperationalError):
            _with_retry(always_fail, retries=2, base_delay=0.001)


class TestParamsJsonField:
    """FR-27: params_json field must be serialisable."""

    def test_params_json_serialisable(self):
        from ethusd_analyzer.strategy import TradeRecommendation
        params = {"r_min": 0.10, "hold_bars": 2, "quantile_hi": 0.90}
        rec = TradeRecommendation(
            timeframe="5m", regime="MR", signal="BUY",
            confidence=0.7, entry_price=3000.0, stop_loss=2970.0,
            take_profit=3030.0, hold_bars=2, reason="test",
            conf_regime=0.5, conf_tail=0.3, conf_backtest=0.2,
            rc=-0.2, ar=0.1, score_mr=1.5, score_mom=0.0, volatility=0.01,
            params_json=params,
        )
        encoded = json.dumps(rec.params_json)
        decoded = json.loads(encoded)
        assert decoded["r_min"] == pytest.approx(0.10)
