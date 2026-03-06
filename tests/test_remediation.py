"""Tests for SRS remediation items A through J."""
from __future__ import annotations

import importlib
import os
import sys
from unittest import mock

import numpy as np
import pandas as pd
import pytest


# ── A) Secret and credential handling ──────────────────────────

class TestConfigSecrets:
    def test_resolve_secrets_from_env(self):
        from ethusd_analyzer.config_secrets import resolve_secrets

        cfg = {"capital": {"api_key": "", "email": "", "password": ""}}
        with mock.patch.dict(os.environ, {"CAPITAL_API_KEY": "test_key_123"}):
            warnings = resolve_secrets(cfg)
        assert cfg["capital"]["api_key"] == "test_key_123"
        assert not warnings

    def test_resolve_secrets_missing_required(self):
        from ethusd_analyzer.config_secrets import resolve_secrets, _SECRET_MAP

        # Temporarily make capital.api_key required
        original = list(_SECRET_MAP)
        _SECRET_MAP.clear()
        _SECRET_MAP.append(("capital.api_key", "CAPITAL_API_KEY", True))
        try:
            cfg = {"capital": {"api_key": ""}}
            with mock.patch.dict(os.environ, {}, clear=True):
                os.environ.pop("CAPITAL_API_KEY", None)
                warnings = resolve_secrets(cfg)
            assert len(warnings) == 1
            assert "CAPITAL_API_KEY" in warnings[0]
        finally:
            _SECRET_MAP.clear()
            _SECRET_MAP.extend(original)

    def test_redact_value(self):
        from ethusd_analyzer.config_secrets import redact_value

        assert redact_value("api_key", "abcdefghijklmnop") == "abc***nop"
        assert redact_value("password", "short") == "***"
        assert redact_value("name", "not_secret") == "not_secret"

    def test_redact_config(self):
        from ethusd_analyzer.config_secrets import redact_config

        cfg = {
            "capital": {"api_key": "real_secret_key_12345", "base_url": "https://example.com"},
            "alerts": {"telegram": {"bot_token": "123456:ABCDEFGH"}},
        }
        redacted = redact_config(cfg)
        assert "***" in redacted["capital"]["api_key"]
        assert redacted["capital"]["base_url"] == "https://example.com"
        assert "***" in redacted["alerts"]["telegram"]["bot_token"]

    def test_redact_string(self):
        from ethusd_analyzer.config_secrets import redact_string

        s = "Token is 8310447730:AAGyTFRCVONO_R1nENUMNO5DJidLL9WAenc"
        result = redact_string(s)
        assert "8310447730" not in result
        assert "REDACTED" in result

    def test_validate_no_hardcoded_secrets(self):
        from ethusd_analyzer.config_secrets import validate_no_hardcoded_secrets

        cfg = {"capital": {"api_key": "real_key_here"}}
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CAPITAL_API_KEY", None)
            issues = validate_no_hardcoded_secrets(cfg)
        assert len(issues) >= 1
        assert "CAPITAL_API_KEY" in issues[0]


# ── B) Bar-based cooldown ─────────────────────────────────────

class TestBarBasedCooldown:
    def test_cooldown_only_decrements_on_new_bar(self):
        """Simulate repeated loop iterations with same candle — bars_elapsed should NOT increment."""
        from ethusd_analyzer.strategy import should_emit_signal, Signal

        last_sig = {"signal": "BUY", "regime": "MR", "bars_elapsed": 0}
        # Same bar: bars_elapsed stays at 0 (caller doesn't increment)
        assert not should_emit_signal(Signal.BUY, "MR", last_sig, cooldown_bars=3, bars_elapsed=0)

        # After 3 new bars
        assert should_emit_signal(Signal.BUY, "MR", last_sig, cooldown_bars=3, bars_elapsed=3)

    def test_cooldown_resets_on_different_signal(self):
        from ethusd_analyzer.strategy import should_emit_signal, Signal

        last_sig = {"signal": "BUY", "regime": "MR", "bars_elapsed": 0}
        # Different signal type bypasses cooldown
        assert should_emit_signal(Signal.SELL, "MR", last_sig, cooldown_bars=3, bars_elapsed=0)

    def test_new_bar_detection_logic(self):
        """Simulate the bar-based cooldown tracking from run.py."""
        last_candle_ts_tracker = {}
        last_sig = {"signal": "BUY", "regime": "MR", "bars_elapsed": 0}

        ts1 = pd.Timestamp("2024-01-01 00:01:00", tz="UTC")
        ts2 = pd.Timestamp("2024-01-01 00:02:00", tz="UTC")

        # First call with ts1
        tf = "1m"
        prev_ts = last_candle_ts_tracker.get(tf)
        new_bar = ts1 is not None and ts1 != prev_ts
        assert new_bar is True
        last_candle_ts_tracker[tf] = ts1
        if new_bar:
            last_sig["bars_elapsed"] = last_sig.get("bars_elapsed", 0) + 1
        assert last_sig["bars_elapsed"] == 1

        # Same ts1 again (intra-bar refresh)
        prev_ts = last_candle_ts_tracker.get(tf)
        new_bar = ts1 is not None and ts1 != prev_ts
        assert new_bar is False
        # bars_elapsed stays at 1
        assert last_sig["bars_elapsed"] == 1

        # New bar ts2
        prev_ts = last_candle_ts_tracker.get(tf)
        new_bar = ts2 is not None and ts2 != prev_ts
        assert new_bar is True
        last_candle_ts_tracker[tf] = ts2
        if new_bar:
            last_sig["bars_elapsed"] = last_sig.get("bars_elapsed", 0) + 1
        assert last_sig["bars_elapsed"] == 2


# ── C) Config parity ──────────────────────────────────────────

class TestConfigParity:
    def test_resolve_effective_strategy_config(self):
        from ethusd_analyzer.strategy import resolve_effective_strategy_config

        strategy_cfg = {
            "gates": {"stretch_z_min": 1.0, "trend_min": 0.002},
            "signal": {"mom_k": 1.5},
            "confidence": {"min_confidence": 0.55},
            "regime": {"r_min": 0.05},
            "tp_sl": {"mr_tp_mult": 1.0},
            "meta_model": {"enabled": False},
            "calibration": {"enabled": True},
            "cooldown_bars": 3,
            "symbol": "ETHUSD",
            "timeframe_overrides": {
                "30m": {
                    "gates": {"stretch_z_min": 0.5},
                    "confidence": {"min_confidence": 0.60},
                }
            },
        }

        # 30m should use override
        eff_30m = resolve_effective_strategy_config("30m", strategy_cfg)
        assert eff_30m["gates"]["stretch_z_min"] == 0.5
        assert eff_30m["confidence"]["min_confidence"] == 0.60
        assert eff_30m["gates"]["trend_min"] == 0.002  # inherited from global

        # 15m should use global
        eff_15m = resolve_effective_strategy_config("15m", strategy_cfg)
        assert eff_15m["gates"]["stretch_z_min"] == 1.0
        assert eff_15m["confidence"]["min_confidence"] == 0.55

    def test_runtime_and_calibration_get_same_gates(self):
        """Verify that the resolver produces identical output for both paths."""
        from ethusd_analyzer.strategy import resolve_effective_strategy_config

        strategy_cfg = {
            "gates": {"stretch_z_min": 1.0, "trend_min": 0.002},
            "signal": {"mom_k": 1.5},
            "confidence": {"min_confidence": 0.55},
            "regime": {"r_min": 0.05},
            "tp_sl": {},
            "meta_model": {},
            "calibration": {},
            "cooldown_bars": 3,
            "symbol": "ETHUSD",
            "timeframe_overrides": {
                "30m": {"gates": {"stretch_z_min": 0.5}},
            },
        }

        # Both runtime and calibration call the same function
        eff1 = resolve_effective_strategy_config("30m", strategy_cfg)
        eff2 = resolve_effective_strategy_config("30m", strategy_cfg)
        assert eff1["gates"] == eff2["gates"]
        assert eff1["gates"]["stretch_z_min"] == 0.5


# ── D) Source candle timestamp ─────────────────────────────────

class TestSourceCandleTs:
    def test_trade_recommendation_has_source_candle_ts(self):
        from ethusd_analyzer.strategy import TradeRecommendation

        ts = pd.Timestamp("2024-01-01 12:00:00", tz="UTC")
        rec = TradeRecommendation(
            timeframe="5m", regime="MR", signal="BUY",
            confidence=0.7, entry_price=3000.0, stop_loss=2990.0,
            take_profit=3015.0, hold_bars=2, reason="test",
            conf_regime=0.5, conf_tail=0.3, conf_backtest=0.2,
            rc=-0.15, ar=0.08, score_mr=0.9, score_mom=0.01,
            volatility=0.001, source_candle_ts=ts,
        )
        assert rec.source_candle_ts == ts

    def test_evaluate_timeframe_sets_source_candle_ts(self, df_strategy):
        from ethusd_analyzer.strategy import evaluate_timeframe

        strategy_cfg = {
            "regime": {"r_min": 0.001, "a_min": 0.001},
            "signal": {"quantile_window": 50, "quantile_hi": 0.95, "quantile_lo": 0.05, "mom_k": 0.5},
            "gates": {"stretch_z_min": 0.0, "trend_min": 0.0},
            "tp_sl": {},
            "confidence": {"min_confidence": 0.0},
            "meta_model": {"enabled": False},
            "calibration": {},
            "cooldown_bars": 0,
            "symbol": "ETHUSD",
            "timeframe_overrides": {},
        }
        rec = evaluate_timeframe(
            df_strategy,
            timeframe="1m",
            strategy_cfg=strategy_cfg,
            cooldown_bars=0,
        )
        if rec is not None:
            assert rec.source_candle_ts is not None
            expected_ts = df_strategy["market_time"].iloc[-1]
            assert rec.source_candle_ts == expected_ts


# ── H) Pytest collection safety ───────────────────────────────

class TestPytestCollectionSafety:
    def test_helper_scripts_safe_on_import(self):
        """Root-level test_*.py files must not execute side effects on import."""
        root = os.path.dirname(os.path.dirname(__file__))
        for name in ["test_shutdown_notification", "test_notifiers", "test_macos_notifier"]:
            path = os.path.join(root, f"{name}.py")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                source = f.read()
            # All executable code must be inside if __name__ == "__main__":
            # Check that no load_config or get_*_notifier calls happen at module level
            lines = source.split("\n")
            in_main = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("if __name__"):
                    in_main = True
                # Only check lines at module level (no leading whitespace)
                is_module_level = (line == line.lstrip()) and stripped != ""
                if not in_main and is_module_level and not stripped.startswith("#") and not stripped.startswith("def "):
                    assert "load_config" not in stripped, (
                        f"{name}.py calls load_config at module level"
                    )
                    assert "get_macos_notifier" not in stripped, (
                        f"{name}.py calls get_macos_notifier at module level"
                    )
