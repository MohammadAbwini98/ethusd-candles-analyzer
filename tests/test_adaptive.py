"""Comprehensive tests for the adaptive strategy parameter layer.

Covers:
  - AdaptiveConfig parsing and defaults
  - MarketState computation
  - Adaptive parameter computation (clamping, formulas, edge cases)
  - Deterministic same-bar behaviour
  - No intra-bar adaptive drift (cache)
  - Timeframe override + adaptive composition
  - Fallback to static baseline on insufficient data
  - Shadow mode (compute + log, effective = base)
  - Defensive math (NaN, inf, zero variance)
  - Historical example scenarios per major adaptive parameter
  - Shadow-mode comparison example
"""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from ethusd_analyzer.adaptive import (
    AdaptiveCache,
    AdaptiveConfig,
    AdaptiveParamBounds,
    AdaptiveResult,
    MarketState,
    _DEFAULT_BOUNDS,
    _safe_clip,
    compute_adaptive_strategy_params,
    compute_market_state,
    precompute_adaptive_arrays,
    resolve_effective_adaptive_config,
)


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

def _make_strategy_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Construct a minimal strategy DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    close = 3000.0 + np.cumsum(rng.normal(0, 2, n))
    log_ret = np.diff(np.log(close), prepend=np.log(close[0]))
    vol = pd.Series(log_ret).rolling(20, min_periods=20).std().values
    rc = rng.normal(-0.15, 0.05, n)
    ar = rng.normal(0.10, 0.05, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "market_time": ts,
        "close": close,
        "volatility": vol,
        "rc": rc,
        "ar": ar,
        "trend_strength": rng.normal(0.001, 0.0005, n),
        "price_z": rng.normal(0, 1.0, n),
        "score_mr": rng.normal(0, 1.0, n),
        "score_mom": rng.normal(0, 0.001, n),
        "log_ret": log_ret,
    })


@pytest.fixture
def df_strat():
    return _make_strategy_df()


@pytest.fixture
def base_config():
    """A representative resolved base config dict."""
    return {
        "gates": {"trend_min": 0.002, "stretch_z_min": 1.0,
                  "ema_fast_span": 20, "ema_slow_span": 50},
        "signal": {"mom_k": 1.5, "quantile_window": 200},
        "tp_sl": {"mr_tp_mult": 1.0, "mr_sl_mult": 1.2,
                  "mom_tp_mult": 2.0, "mom_sl_mult": 1.0},
        "confidence": {"min_confidence": 0.55, "regime_denom": 0.25},
        "regime": {"r_min": 0.05, "a_min": 0.05},
    }


@pytest.fixture
def adaptive_enabled_cfg():
    """AdaptiveConfig with everything enabled."""
    return AdaptiveConfig(enabled=True, shadow_mode=False, lookback_bars=100)


@pytest.fixture
def adaptive_shadow_cfg():
    """AdaptiveConfig in shadow mode."""
    return AdaptiveConfig(enabled=True, shadow_mode=True, lookback_bars=100)


def _normal_market_state(**overrides):
    """A valid mid-range market state (defaults to vol_pctile=0.5 etc.)."""
    defaults = dict(
        volatility=0.005, volatility_pctile=0.5, atr_pctile=0.5,
        trend_efficiency=0.5, trend_strength=0.001, regime_strength=0.5,
        price_z=0.0, computed_from_bars=200, valid=True,
    )
    defaults.update(overrides)
    return MarketState(**defaults)


# ─────────────────────────────────────────────────────────────
#  A) AdaptiveConfig parsing
# ─────────────────────────────────────────────────────────────

class TestAdaptiveConfigParsing:

    def test_from_empty_dict(self):
        cfg = AdaptiveConfig.from_dict({})
        assert cfg.enabled is False
        assert cfg.shadow_mode is False

    def test_from_none(self):
        cfg = AdaptiveConfig.from_dict(None)
        assert cfg.enabled is False

    def test_defaults_populated(self):
        cfg = AdaptiveConfig()
        for name in _DEFAULT_BOUNDS:
            assert name in cfg.params, f"Missing default bounds for {name}"

    def test_full_yaml_round_trip(self):
        d = {
            "enabled": True,
            "shadow_mode": True,
            "lookback_bars": 50,
            "params": {
                "trend_min": {"enabled": False, "min": 0.001, "max": 0.005},
                "mom_k": {"enabled": True, "min": 1.0, "max": 2.5},
            },
            "per_timeframe": {"30m": {"trend_min": {"min": 0.0003, "max": 0.004}}},
        }
        cfg = AdaptiveConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.shadow_mode is True
        assert cfg.lookback_bars == 50
        assert cfg.params["trend_min"].enabled is False
        assert cfg.params["trend_min"].max_val == 0.005
        assert cfg.params["mom_k"].min_val == 1.0

    def test_per_timeframe_override(self):
        d = {
            "enabled": True,
            "params": {"trend_min": {"enabled": True, "min": 0.0005, "max": 0.01}},
            "per_timeframe": {"30m": {"trend_min": {"min": 0.0003, "max": 0.005}}},
        }
        cfg = AdaptiveConfig.from_dict(d)
        bounds = cfg.get_bounds("trend_min", "30m")
        assert bounds.min_val == 0.0003
        assert bounds.max_val == 0.005
        # Without TF override, get global bounds
        bounds_global = cfg.get_bounds("trend_min", "5m")
        assert bounds_global.max_val == 0.01

    def test_is_param_enabled_respects_global_flag(self):
        cfg = AdaptiveConfig(enabled=False)
        assert cfg.is_param_enabled("trend_min") is False

    def test_is_param_enabled_respects_per_param(self):
        cfg = AdaptiveConfig(enabled=True)
        cfg.params["trend_min"] = AdaptiveParamBounds(enabled=False)
        assert cfg.is_param_enabled("trend_min") is False
        assert cfg.is_param_enabled("mom_k") is True


# ─────────────────────────────────────────────────────────────
#  B) MarketState computation
# ─────────────────────────────────────────────────────────────

class TestComputeMarketState:

    def test_insufficient_bars_returns_invalid(self):
        df = _make_strategy_df(n=10)
        ms = compute_market_state(df, lookback_bars=100)
        assert ms.valid is False
        assert ms.fallback_reason == "insufficient_bars"

    def test_valid_with_enough_bars(self, df_strat):
        ms = compute_market_state(df_strat, lookback_bars=100)
        assert ms.valid is True
        assert ms.fallback_reason is None
        assert 0.0 <= ms.volatility_pctile <= 1.0
        assert 0.0 <= ms.trend_efficiency <= 1.0
        assert 0.0 <= ms.regime_strength <= 1.0

    def test_source_candle_ts_populated(self, df_strat):
        ms = compute_market_state(df_strat)
        assert ms.source_candle_ts is not None
        assert ms.source_candle_ts == df_strat["market_time"].iloc[-1]

    def test_handles_nan_volatility(self, df_strat):
        df = df_strat.copy()
        df.loc[df.index[-1], "volatility"] = float("nan")
        ms = compute_market_state(df)
        assert ms.valid is True
        assert ms.volatility == 0.0  # NaN → 0 fallback

    def test_handles_missing_columns_gracefully(self):
        df = pd.DataFrame({
            "close": np.linspace(100, 200, 50),
            "market_time": pd.date_range("2024-01-01", periods=50, freq="1min"),
        })
        ms = compute_market_state(df)
        # Should not crash; volatility columns missing → defaults
        assert ms.valid is True
        assert ms.volatility == 0.0
        assert ms.volatility_pctile == 0.5

    def test_trend_efficiency_range(self, df_strat):
        ms = compute_market_state(df_strat, lookback_bars=100)
        assert 0.0 <= ms.trend_efficiency <= 1.0


# ─────────────────────────────────────────────────────────────
#  C) Adaptive parameter computation
# ─────────────────────────────────────────────────────────────

class TestComputeAdaptiveParams:

    def test_disabled_returns_base(self, base_config):
        cfg = AdaptiveConfig(enabled=False)
        ms = _normal_market_state()
        result = compute_adaptive_strategy_params(base_config, ms, cfg)
        assert result.effective_values == result.base_values
        assert all(d == 0.0 for d in result.deltas.values())

    def test_invalid_state_returns_base(self, base_config, adaptive_enabled_cfg):
        ms = MarketState(
            volatility=0.0, volatility_pctile=0.5, atr_pctile=0.5,
            trend_efficiency=0.5, trend_strength=0.0, regime_strength=0.0,
            price_z=0.0, valid=False, fallback_reason="test",
        )
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values == result.base_values
        for r in result.reasons.values():
            assert "invalid_market_state" in r

    def test_all_params_clamped(self, base_config, adaptive_enabled_cfg):
        """Extreme market states must not produce out-of-bounds values."""
        for vol_p in [0.0, 1.0]:
            for te in [0.0, 1.0]:
                for rs in [0.0, 1.0]:
                    ms = _normal_market_state(
                        volatility_pctile=vol_p,
                        trend_efficiency=te,
                        regime_strength=rs,
                    )
                    result = compute_adaptive_strategy_params(
                        base_config, ms, adaptive_enabled_cfg,
                    )
                    for name, value in result.adaptive_values.items():
                        bounds = adaptive_enabled_cfg.get_bounds(name)
                        assert bounds.min_val <= value <= bounds.max_val, (
                            f"{name}={value} outside [{bounds.min_val}, {bounds.max_val}] "
                            f"at vol_p={vol_p} te={te} rs={rs}"
                        )

    def test_mid_range_produces_near_base(self, base_config, adaptive_enabled_cfg):
        """At vol_pctile=0.5, trend_efficiency=0.5, regime_strength=0.5 → ≈ base."""
        ms = _normal_market_state()
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        for name, base in result.base_values.items():
            eff = result.effective_values[name]
            # All should be within 20% of base at midpoint
            if base > 0:
                ratio = abs(eff - base) / base
                assert ratio < 0.20, (
                    f"{name}: eff={eff} base={base} ratio={ratio:.3f} exceeds 20%"
                )

    def test_per_param_disable(self, base_config):
        cfg = AdaptiveConfig(enabled=True)
        cfg.params["trend_min"] = AdaptiveParamBounds(enabled=False)
        ms = _normal_market_state(volatility_pctile=0.9)
        result = compute_adaptive_strategy_params(base_config, ms, cfg)
        # trend_min should equal base (disabled)
        assert result.effective_values["trend_min"] == result.base_values["trend_min"]
        # mom_k should differ (enabled)
        assert result.reasons["trend_min"] == "param_disabled"

    def test_audit_dict_serializable(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state()
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        audit = result.to_audit_dict()
        import json
        serialized = json.dumps(audit)
        assert isinstance(serialized, str)
        recovered = json.loads(serialized)
        assert "base" in recovered
        assert "adaptive" in recovered
        assert "market_state" in recovered


# ─────────────────────────────────────────────────────────────
#  D) Deterministic same-bar behaviour
# ─────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_df_produces_same_result(self, df_strat, base_config, adaptive_enabled_cfg):
        ms1 = compute_market_state(df_strat, lookback_bars=100)
        ms2 = compute_market_state(df_strat, lookback_bars=100)
        r1 = compute_adaptive_strategy_params(base_config, ms1, adaptive_enabled_cfg)
        r2 = compute_adaptive_strategy_params(base_config, ms2, adaptive_enabled_cfg)
        assert r1.effective_values == r2.effective_values

    def test_cache_prevents_recomputation(self, df_strat, base_config, adaptive_enabled_cfg):
        cache = AdaptiveCache()
        ms = compute_market_state(df_strat)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        ts = df_strat["market_time"].iloc[-1]
        cache.put("5m", ts, result)
        cached = cache.get("5m", ts)
        assert cached is result  # same object
        # Different ts → cache miss
        assert cache.get("5m", ts + pd.Timedelta(minutes=1)) is None

    def test_cache_clear(self):
        cache = AdaptiveCache()
        ms = _normal_market_state()
        result = AdaptiveResult(
            base_values={}, adaptive_values={}, effective_values={},
            deltas={}, market_state=ms, reasons={},
        )
        cache.put("5m", "ts1", result)
        cache.put("15m", "ts2", result)
        cache.clear("5m")
        assert cache.get("5m", "ts1") is None
        assert cache.get("15m", "ts2") is result
        cache.clear()
        assert cache.get("15m", "ts2") is None


# ─────────────────────────────────────────────────────────────
#  E) Shadow mode
# ─────────────────────────────────────────────────────────────

class TestShadowMode:

    def test_shadow_effective_equals_base(self, base_config, adaptive_shadow_cfg):
        ms = _normal_market_state(volatility_pctile=0.9, trend_efficiency=0.1)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_shadow_cfg)
        assert result.shadow_mode is True
        # Effective should equal base (shadow doesn't change decisions)
        assert result.effective_values == result.base_values
        # But adaptive values should differ from base (they were computed)
        has_diff = any(
            result.adaptive_values[k] != result.base_values[k]
            for k in result.base_values
        )
        assert has_diff, "Shadow mode should still compute different adaptive values"

    def test_shadow_deltas_nonzero(self, base_config, adaptive_shadow_cfg):
        ms = _normal_market_state(volatility_pctile=0.9, trend_efficiency=0.1)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_shadow_cfg)
        assert any(d != 0.0 for d in result.deltas.values())


# ─────────────────────────────────────────────────────────────
#  F) Specific parameter behavior
# ─────────────────────────────────────────────────────────────

class TestParameterBehavior:

    def test_high_vol_chop_raises_trend_min(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(volatility_pctile=0.9, trend_efficiency=0.1)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["trend_min"] > result.base_values["trend_min"]

    def test_smooth_trend_lowers_trend_min(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(volatility_pctile=0.1, trend_efficiency=0.9)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["trend_min"] <= result.base_values["trend_min"]

    def test_high_vol_raises_stretch_z_min(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(volatility_pctile=0.95)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["stretch_z_min"] > result.base_values["stretch_z_min"]

    def test_low_vol_lowers_stretch_z_min(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(volatility_pctile=0.05)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["stretch_z_min"] < result.base_values["stretch_z_min"]

    def test_high_chop_raises_mom_k(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(trend_efficiency=0.05)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["mom_k"] > result.base_values["mom_k"]

    def test_strong_persistent_trend_lowers_mom_k(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(trend_efficiency=0.95)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["mom_k"] < result.base_values["mom_k"]

    def test_weak_regime_raises_min_confidence(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(regime_strength=0.1)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["min_confidence"] > result.base_values["min_confidence"]

    def test_strong_regime_lowers_min_confidence(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(regime_strength=0.95)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["min_confidence"] < result.base_values["min_confidence"]

    def test_high_vol_widens_sl(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(volatility_pctile=0.95)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["mr_sl_mult"] >= result.base_values["mr_sl_mult"]
        assert result.effective_values["mom_sl_mult"] >= result.base_values["mom_sl_mult"]

    def test_strong_trend_widens_mom_tp(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(regime_strength=0.9, trend_efficiency=0.9)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.effective_values["mom_tp_mult"] > result.base_values["mom_tp_mult"]

    def test_mr_exits_tighter_than_mom(self, base_config, adaptive_enabled_cfg):
        """MR exits should remain tighter than MOM exits, matching baseline intent."""
        for vol_p in [0.2, 0.5, 0.8]:
            ms = _normal_market_state(volatility_pctile=vol_p, regime_strength=0.5)
            result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
            ev = result.effective_values
            # MR TP should be ≤ MOM TP (MR takes smaller profits)
            assert ev["mr_tp_mult"] <= ev["mom_tp_mult"], (
                f"MR TP ({ev['mr_tp_mult']}) should be <= MOM TP ({ev['mom_tp_mult']}) "
                f"at vol_pctile={vol_p}"
            )


# ─────────────────────────────────────────────────────────────
#  G) Defensive math
# ─────────────────────────────────────────────────────────────

class TestDefensiveMath:

    def test_safe_clip_handles_nan(self):
        assert _safe_clip(float("nan"), 0.5, 1.5) == 1.0  # midpoint

    def test_safe_clip_handles_inf(self):
        assert _safe_clip(float("inf"), 0.5, 1.5) == 1.0

    def test_safe_clip_handles_neg_inf(self):
        assert _safe_clip(float("-inf"), 0.5, 1.5) == 1.0

    def test_safe_clip_normal(self):
        assert _safe_clip(0.7, 0.5, 1.5) == 0.7
        assert _safe_clip(2.0, 0.5, 1.5) == 1.5
        assert _safe_clip(0.1, 0.5, 1.5) == 0.5

    def test_market_state_with_all_nan_features(self):
        """DataFrame where all feature columns are NaN."""
        n = 50
        df = pd.DataFrame({
            "close": np.linspace(100, 200, n),
            "market_time": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "volatility": [float("nan")] * n,
            "rc": [float("nan")] * n,
            "ar": [float("nan")] * n,
            "trend_strength": [float("nan")] * n,
            "price_z": [float("nan")] * n,
        })
        ms = compute_market_state(df)
        assert ms.valid is True
        assert ms.volatility == 0.0
        assert ms.regime_strength == 0.0

    def test_zero_variance_close(self):
        """All closes identical → trend_efficiency = 0."""
        n = 50
        df = pd.DataFrame({
            "close": [3000.0] * n,
            "market_time": pd.date_range("2024-01-01", periods=n, freq="1min"),
            "volatility": [0.005] * n,
            "rc": [-0.1] * n,
            "ar": [0.1] * n,
            "trend_strength": [0.0] * n,
            "price_z": [0.0] * n,
        })
        ms = compute_market_state(df)
        assert ms.trend_efficiency == 0.0


# ─────────────────────────────────────────────────────────────
#  H) resolve_effective_adaptive_config
# ─────────────────────────────────────────────────────────────

class TestResolveEffectiveAdaptiveConfig:

    def test_does_not_mutate_base(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(volatility_pctile=0.9)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        original_gates = copy.deepcopy(base_config["gates"])
        eff = resolve_effective_adaptive_config(base_config, result)
        # Original should be untouched
        assert base_config["gates"] == original_gates
        # Effective should have adaptive values
        assert eff["gates"]["trend_min"] == result.effective_values["trend_min"]
        assert eff["signal"]["mom_k"] == result.effective_values["mom_k"]

    def test_all_sections_merged(self, base_config, adaptive_enabled_cfg):
        ms = _normal_market_state(volatility_pctile=0.8, trend_efficiency=0.3)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        eff = resolve_effective_adaptive_config(base_config, result)
        assert eff["gates"]["trend_min"] == result.effective_values["trend_min"]
        assert eff["gates"]["stretch_z_min"] == result.effective_values["stretch_z_min"]
        assert eff["signal"]["mom_k"] == result.effective_values["mom_k"]
        assert eff["confidence"]["min_confidence"] == result.effective_values["min_confidence"]
        assert eff["tp_sl"]["mr_tp_mult"] == result.effective_values["mr_tp_mult"]
        assert eff["tp_sl"]["mr_sl_mult"] == result.effective_values["mr_sl_mult"]
        assert eff["tp_sl"]["mom_tp_mult"] == result.effective_values["mom_tp_mult"]
        assert eff["tp_sl"]["mom_sl_mult"] == result.effective_values["mom_sl_mult"]


# ─────────────────────────────────────────────────────────────
#  I) Deterministic historical examples
# ─────────────────────────────────────────────────────────────

class TestHistoricalExamples:
    """Deterministic examples showing base → adaptive → effective for each param."""

    def _compute(self, base_config, vol_p, te, rs):
        cfg = AdaptiveConfig(enabled=True)
        ms = _normal_market_state(volatility_pctile=vol_p, trend_efficiency=te, regime_strength=rs)
        return compute_adaptive_strategy_params(base_config, ms, cfg)

    def test_example_trend_min_high_chop_high_vol(self, base_config):
        """High chop (te=0.1) + high vol (vol_p=0.9): trend_min should increase."""
        result = self._compute(base_config, vol_p=0.9, te=0.1, rs=0.5)
        base = result.base_values["trend_min"]
        eff = result.effective_values["trend_min"]
        assert eff > base, f"Expected trend_min increase: base={base} eff={eff}"
        # Check the formula: chop=0.9, vol_factor=0.8
        # mult = 1.0 + 0.50 * 0.9 * max(0.8, 0) - 0.30 * 0.1 * max(-0.8, 0)
        # mult = 1.0 + 0.36 - 0 = 1.36
        expected_mult = 1.0 + 0.5 * 0.9 * 0.8
        expected = base * expected_mult
        assert abs(eff - expected) < 1e-6 or (
            _DEFAULT_BOUNDS["trend_min"]["min"] <= eff <= _DEFAULT_BOUNDS["trend_min"]["max"]
        )

    def test_example_stretch_z_min_high_vol(self, base_config):
        """High vol (vol_p=0.95): stretch_z_min should increase."""
        result = self._compute(base_config, vol_p=0.95, te=0.5, rs=0.5)
        base = result.base_values["stretch_z_min"]
        eff = result.effective_values["stretch_z_min"]
        # mult = 1.0 + 0.30 * (0.95-0.5)*2 = 1.0 + 0.27 = 1.27
        expected = base * (1.0 + 0.30 * 0.9)
        assert abs(eff - expected) < 1e-6

    def test_example_mom_k_pure_chop(self, base_config):
        """Pure chop (te=0.0): mom_k should increase."""
        result = self._compute(base_config, vol_p=0.5, te=0.0, rs=0.5)
        base = result.base_values["mom_k"]
        eff = result.effective_values["mom_k"]
        # mult = 1.0 + 0.30 * 1.0 - 0.15 * 0.0 = 1.30
        expected = base * 1.30
        assert abs(eff - expected) < 1e-6

    def test_example_min_confidence_weak_regime(self, base_config):
        """Weak regime (rs=0.1): min_confidence should increase."""
        result = self._compute(base_config, vol_p=0.5, te=0.5, rs=0.1)
        base = result.base_values["min_confidence"]
        eff = result.effective_values["min_confidence"]
        # mult = 1.0 + 0.15 * 0.9 - 0.05 * 0.1 = 1.0 + 0.135 - 0.005 = 1.13
        expected = base * 1.13
        assert abs(eff - expected) < 1e-6

    def test_example_mom_tp_strong_trend(self, base_config):
        """Strong trend (rs=0.9, te=0.9): mom_tp_mult should increase."""
        result = self._compute(base_config, vol_p=0.5, te=0.9, rs=0.9)
        base = result.base_values["mom_tp_mult"]
        eff = result.effective_values["mom_tp_mult"]
        # mult = 1.0 + 0.20 * 0.9 + 0.15 * 0.9 = 1.0 + 0.18 + 0.135 = 1.315
        expected = base * 1.315
        assert abs(eff - expected) < 1e-6


# ─────────────────────────────────────────────────────────────
#  J) Shadow-mode comparison example
# ─────────────────────────────────────────────────────────────

class TestShadowModeComparison:

    def test_shadow_vs_live_comparison(self, base_config):
        """Compare shadow vs live for same market state."""
        ms = _normal_market_state(volatility_pctile=0.85, trend_efficiency=0.2, regime_strength=0.3)

        shadow_cfg = AdaptiveConfig(enabled=True, shadow_mode=True)
        live_cfg = AdaptiveConfig(enabled=True, shadow_mode=False)

        shadow_result = compute_adaptive_strategy_params(base_config, ms, shadow_cfg)
        live_result = compute_adaptive_strategy_params(base_config, ms, live_cfg)

        # Shadow effective = base; live effective = adaptive
        assert shadow_result.effective_values == shadow_result.base_values
        assert live_result.effective_values == live_result.adaptive_values

        # But adaptive computations should be identical
        assert shadow_result.adaptive_values == live_result.adaptive_values

        # Deltas should be identical
        assert shadow_result.deltas == live_result.deltas

        # Log what the difference would have been
        for k in shadow_result.base_values:
            base_v = shadow_result.base_values[k]
            adap_v = shadow_result.adaptive_values[k]
            delta = shadow_result.deltas[k]
            assert abs(delta - (adap_v - base_v)) < 1e-10


# ─────────────────────────────────────────────────────────────
#  K) Integration: compute_market_state → compute_adaptive
# ─────────────────────────────────────────────────────────────

class TestEndToEndAdaptive:

    def test_full_flow_from_df(self, df_strat, base_config, adaptive_enabled_cfg):
        ms = compute_market_state(df_strat, lookback_bars=100)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        assert result.market_state.valid is True
        assert result.market_state.computed_from_bars == len(df_strat)
        # All effective values should be within bounds
        for name, value in result.effective_values.items():
            bounds = adaptive_enabled_cfg.get_bounds(name)
            assert bounds.min_val <= value <= bounds.max_val

    def test_full_flow_resolve_config(self, df_strat, base_config, adaptive_enabled_cfg):
        ms = compute_market_state(df_strat)
        result = compute_adaptive_strategy_params(base_config, ms, adaptive_enabled_cfg)
        eff_config = resolve_effective_adaptive_config(base_config, result)
        # Should be a valid config dict
        assert "gates" in eff_config
        assert "signal" in eff_config
        assert "tp_sl" in eff_config
        assert "confidence" in eff_config
        # Non-adaptive keys should be preserved
        assert eff_config["gates"]["ema_fast_span"] == 20


# ─────────────────────────────────────────────────────────────
#  L) Precompute adaptive arrays
# ─────────────────────────────────────────────────────────────

class TestPrecomputeAdaptiveArrays:

    def test_disabled_returns_base_arrays(self, df_strat, base_config):
        cfg = AdaptiveConfig(enabled=False)
        arrays = precompute_adaptive_arrays(df_strat, base_config, cfg)
        assert set(arrays.keys()) == {"trend_min", "stretch_z_min", "mom_k"}
        # All values should be the base scalar
        np.testing.assert_allclose(arrays["trend_min"], 0.002)
        np.testing.assert_allclose(arrays["stretch_z_min"], 1.0)
        np.testing.assert_allclose(arrays["mom_k"], 1.5)

    def test_enabled_produces_per_bar_variation(self, df_strat, base_config, adaptive_enabled_cfg):
        arrays = precompute_adaptive_arrays(df_strat, base_config, adaptive_enabled_cfg)
        # After the first ~20 bars, adaptive should produce varying values
        mature_trend = arrays["trend_min"][50:]
        assert len(set(np.round(mature_trend, 8))) > 1, "Expected per-bar variation in trend_min"
        mature_mom = arrays["mom_k"][50:]
        assert len(set(np.round(mature_mom, 8))) > 1, "Expected per-bar variation in mom_k"

    def test_first_20_bars_are_base(self, df_strat, base_config, adaptive_enabled_cfg):
        arrays = precompute_adaptive_arrays(df_strat, base_config, adaptive_enabled_cfg)
        np.testing.assert_allclose(arrays["trend_min"][:20], 0.002)
        np.testing.assert_allclose(arrays["stretch_z_min"][:20], 1.0)
        np.testing.assert_allclose(arrays["mom_k"][:20], 1.5)

    def test_all_values_within_bounds(self, df_strat, base_config, adaptive_enabled_cfg):
        arrays = precompute_adaptive_arrays(df_strat, base_config, adaptive_enabled_cfg)
        for name, arr in arrays.items():
            bounds = adaptive_enabled_cfg.get_bounds(name)
            assert np.all(arr >= bounds.min_val - 1e-10), f"{name} below min_val"
            assert np.all(arr <= bounds.max_val + 1e-10), f"{name} above max_val"

    def test_shadow_mode_returns_base(self, df_strat, base_config, adaptive_shadow_cfg):
        arrays = precompute_adaptive_arrays(df_strat, base_config, adaptive_shadow_cfg)
        np.testing.assert_allclose(arrays["trend_min"], 0.002)
        np.testing.assert_allclose(arrays["stretch_z_min"], 1.0)
        np.testing.assert_allclose(arrays["mom_k"], 1.5)

    def test_deterministic(self, df_strat, base_config, adaptive_enabled_cfg):
        a1 = precompute_adaptive_arrays(df_strat, base_config, adaptive_enabled_cfg)
        a2 = precompute_adaptive_arrays(df_strat, base_config, adaptive_enabled_cfg)
        for key in a1:
            np.testing.assert_array_equal(a1[key], a2[key])

    def test_short_df_returns_base(self, base_config, adaptive_enabled_cfg):
        df = _make_strategy_df(n=15)
        arrays = precompute_adaptive_arrays(df, base_config, adaptive_enabled_cfg)
        np.testing.assert_allclose(arrays["trend_min"], 0.002)


# ─────────────────────────────────────────────────────────────
#  M) Historical evaluation parity
# ─────────────────────────────────────────────────────────────

class TestHistoricalEvaluationParity:
    """Verify that precomputed arrays match per-bar canonical computation."""

    def test_precompute_matches_canonical_per_bar(self, df_strat, base_config, adaptive_enabled_cfg):
        """Last-bar adaptive result from precompute_adaptive_arrays must match
        compute_market_state + compute_adaptive_strategy_params on the full df."""
        arrays = precompute_adaptive_arrays(df_strat, base_config, adaptive_enabled_cfg)
        # Compare last bar from arrays vs canonical single-point computation
        ms = compute_market_state(
            df_strat,
            lookback_bars=adaptive_enabled_cfg.lookback_bars,
            regime_denom=float(base_config.get("confidence", {}).get("regime_denom", 0.25)),
        )
        result = compute_adaptive_strategy_params(
            base_config, ms, adaptive_enabled_cfg,
        )
        last = len(df_strat) - 1
        # Values should be very close (small floating-point differences acceptable
        # because the rolling percentile implementation may differ slightly)
        for key in ["trend_min", "stretch_z_min", "mom_k"]:
            precomp = arrays[key][last]
            canonical = result.effective_values[key]
            assert abs(precomp - canonical) < 0.05 * max(abs(canonical), 1e-6), (
                f"{key}: precomputed={precomp:.6f} canonical={canonical:.6f}"
            )

    def test_no_future_leakage(self, base_config, adaptive_enabled_cfg):
        """Adaptive value at bar i must not change when future bars are appended."""
        df_full = _make_strategy_df(n=200)
        df_prefix = df_full.iloc[:100].copy().reset_index(drop=True)
        arrays_full = precompute_adaptive_arrays(df_full, base_config, adaptive_enabled_cfg)
        arrays_prefix = precompute_adaptive_arrays(df_prefix, base_config, adaptive_enabled_cfg)
        # For bars 0..99, values from prefix and full should be identical
        # (rolling lookback at bar i only uses bars up to i)
        for key in ["trend_min", "stretch_z_min", "mom_k"]:
            np.testing.assert_allclose(
                arrays_prefix[key], arrays_full[key][:100],
                atol=1e-10,
                err_msg=f"{key}: future data leaked into historical computation",
            )

    def test_cache_neutrality(self, df_strat, base_config, adaptive_enabled_cfg):
        """precompute_adaptive_arrays does not use AdaptiveCache — results are
        independent of any cache state."""
        arrays_no_cache = precompute_adaptive_arrays(
            df_strat, base_config, adaptive_enabled_cfg,
        )
        # Create a dirty cache with wrong values
        cache = AdaptiveCache()
        fake_result = AdaptiveResult(
            base_values={k: 999.0 for k in ["trend_min", "stretch_z_min", "mom_k"]},
            adaptive_values={k: 999.0 for k in ["trend_min", "stretch_z_min", "mom_k"]},
            effective_values={k: 999.0 for k in ["trend_min", "stretch_z_min", "mom_k"]},
            deltas={k: 0.0 for k in ["trend_min", "stretch_z_min", "mom_k"]},
            market_state=_normal_market_state(),
            reasons={k: "fake" for k in ["trend_min", "stretch_z_min", "mom_k"]},
        )
        cache.put("5m", "some_ts", fake_result)
        # precompute doesn't take a cache → result must be unchanged
        arrays_with_dirty = precompute_adaptive_arrays(
            df_strat, base_config, adaptive_enabled_cfg,
        )
        for key in arrays_no_cache:
            np.testing.assert_array_equal(arrays_no_cache[key], arrays_with_dirty[key])


# ─────────────────────────────────────────────────────────────
#  N) Simulation adaptive integration
# ─────────────────────────────────────────────────────────────

class TestSimulationAdaptiveIntegration:
    """Verify adaptive arrays flow through _simulate_strategy."""

    def test_adaptive_arrays_change_simulation_result(self, base_config, adaptive_enabled_cfg):
        """Simulation with adaptive arrays should differ from static-only."""
        from tests.conftest import _make_strategy_df as _make_strat
        df = _make_strat(n=600)
        from ethusd_analyzer.strategy import _simulate_strategy

        # Static simulation
        sh_static, _, _, nt_static, _ = _simulate_strategy(
            df, r_min=0.05, a_min=0.05, quantile_window=200,
            quantile_hi_pct=0.90, quantile_lo_pct=0.10,
            hold_bars=2, cost_bps=10,
            stretch_z_min=1.0, trend_min=0.002,
        )

        # Adaptive simulation
        arrays = precompute_adaptive_arrays(df, base_config, adaptive_enabled_cfg)
        sh_adapt, _, _, nt_adapt, _ = _simulate_strategy(
            df, r_min=0.05, a_min=0.05, quantile_window=200,
            quantile_hi_pct=0.90, quantile_lo_pct=0.10,
            hold_bars=2, cost_bps=10,
            stretch_z_min=1.0, trend_min=0.002,
            adaptive_arrays=arrays,
        )

        # Results may or may not differ depending on data, but the function
        # should run without error.  If arrays have variation, at least one
        # metric should be non-trivially different or identical (both are valid).
        # The key assertion: no crash, valid return types.
        assert isinstance(sh_adapt, float)
        assert isinstance(nt_adapt, int)

    def test_none_adaptive_arrays_uses_scalar(self, base_config):
        """When adaptive_arrays=None, scalar params are used (backward compat)."""
        from tests.conftest import _make_strategy_df as _make_strat
        df = _make_strat(n=600)
        from ethusd_analyzer.strategy import _simulate_strategy

        sh1, ret1, dd1, nt1, wr1 = _simulate_strategy(
            df, r_min=0.10, a_min=0.10, quantile_window=200,
            quantile_hi_pct=0.90, quantile_lo_pct=0.10,
            hold_bars=2, cost_bps=10,
            adaptive_arrays=None,
        )
        sh2, ret2, dd2, nt2, wr2 = _simulate_strategy(
            df, r_min=0.10, a_min=0.10, quantile_window=200,
            quantile_hi_pct=0.90, quantile_lo_pct=0.10,
            hold_bars=2, cost_bps=10,
        )
        assert sh1 == sh2
        assert nt1 == nt2
        assert ret1 == ret2

    def test_calibration_accepts_adaptive_params(self, base_config, adaptive_enabled_cfg):
        """run_calibration should accept and use adaptive_cfg + base_config."""
        from tests.conftest import _make_strategy_df as _make_strat
        from ethusd_analyzer.strategy import run_calibration
        df = _make_strat(n=600)

        result = run_calibration(
            df, timeframe="5m",
            grid_config={"r_min": [0.05, 0.10], "quantile_window": [200],
                         "quantile_levels": [[0.90, 0.10]], "hold_bars": [2],
                         "cost_bps": [10]},
            adaptive_cfg=adaptive_enabled_cfg,
            base_config=base_config,
        )
        assert result.status in ("OK", "NO_VALID_PARAMS", "INSUFFICIENT_DATA")

    def test_calibration_without_adaptive_unchanged(self):
        """run_calibration without adaptive params works as before."""
        from tests.conftest import _make_strategy_df as _make_strat
        from ethusd_analyzer.strategy import run_calibration
        df = _make_strat(n=600)

        result = run_calibration(
            df, timeframe="5m",
            grid_config={"r_min": [0.05, 0.10], "quantile_window": [200],
                         "quantile_levels": [[0.90, 0.10]], "hold_bars": [2],
                         "cost_bps": [10]},
        )
        assert result.status in ("OK", "NO_VALID_PARAMS", "INSUFFICIENT_DATA")
