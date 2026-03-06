"""Adaptive strategy parameter layer.

Computes bounded, market-state-driven adjustments to strategy parameters.
Keeps the original static config as the baseline and produces effective values
deterministically from bar-close data only.

Design decisions:
  - Market-state features are computed from closed-bar data (no tick-level).
  - Every adaptive parameter is clamped to explicit [min, max] bounds.
  - A global enable/disable flag and per-parameter enable/disable flags exist.
  - Shadow mode: compute + log adaptive values without changing live decisions.
  - Fallback: if market state is invalid/insufficient, static baseline is used.
  - cooldown_bars is kept STATIC — bar-based cooldown is already deterministic;
    making it adaptive adds sequencing complexity for marginal benefit.

Adaptive parameters and their market-state drivers:
  ┌──────────────────┬───────────────────────────────┬──────────────────────┐
  │ Parameter        │ Primary drivers               │ Direction            │
  ├──────────────────┼───────────────────────────────┼──────────────────────┤
  │ trend_min        │ trend_efficiency, vol_pctile  │ ↑ in chop+high-vol  │
  │ stretch_z_min    │ vol_pctile                    │ ↑ in high-vol       │
  │ mom_k            │ trend_efficiency, vol_pctile  │ ↑ in chop           │
  │ min_confidence   │ regime_strength               │ ↑ in weak regime    │
  │ mr_tp_mult       │ vol_pctile                    │ widen in high-vol   │
  │ mr_sl_mult       │ vol_pctile                    │ widen in high-vol   │
  │ mom_tp_mult      │ regime_strength, trend_eff    │ widen in strong     │
  │ mom_sl_mult      │ vol_pctile                    │ widen in high-vol   │
  └──────────────────┴───────────────────────────────┴──────────────────────┘
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Market State ──────────────────────────────────────────────────────────────

@dataclass
class MarketState:
    """Snapshot of market-state features computed at one bar close.

    All fields are computed from historical closed-bar data only — no
    forward-looking or tick-level information leaks into these values.
    """
    volatility: float            # realized vol (rolling std of log returns)
    volatility_pctile: float     # percentile rank of vol in recent window [0, 1]
    atr_pctile: float            # percentile rank of ATR proxy [0, 1]
    trend_efficiency: float      # price efficiency ratio [0, 1]  (1=perfect trend, 0=chop)
    trend_strength: float        # (EMA_fast − EMA_slow) / close  (signed)
    regime_strength: float       # max(|rc|, |ar|) / regime_denom  capped [0, 1]
    price_z: float               # stretch z-score from MR baseline

    # Source metadata
    source_candle_ts: Optional[Any] = None
    computed_from_bars: int = 0
    valid: bool = True
    fallback_reason: Optional[str] = None


def compute_market_state(
    df: pd.DataFrame,
    lookback_bars: int = 100,
    regime_denom: float = 0.25,
) -> MarketState:
    """Compute market-state features from a strategy-ready DataFrame.

    Uses only bar-close data available at the current decision point.
    Returns a MarketState with ``valid=False`` and a ``fallback_reason``
    when there is insufficient data for reliable computation.
    """
    n = len(df)
    if n < 20:
        return MarketState(
            volatility=0.0, volatility_pctile=0.5, atr_pctile=0.5,
            trend_efficiency=0.5, trend_strength=0.0, regime_strength=0.0,
            price_z=0.0,
            computed_from_bars=n, valid=False,
            fallback_reason="insufficient_bars",
        )

    last = df.iloc[-1]

    # ── Volatility ────────────────────────────────────────────
    vol = float(last.get("volatility", 0.0))
    if math.isnan(vol) or math.isinf(vol):
        vol = 0.0

    # Volatility percentile
    vol_series = df["volatility"].dropna() if "volatility" in df.columns else pd.Series(dtype=float)
    window = min(lookback_bars, len(vol_series))
    if window >= 10 and vol > 0:
        recent_vol = vol_series.iloc[-window:]
        vol_pctile = float((recent_vol < vol).mean())
    else:
        vol_pctile = 0.5

    # ── ATR proxy percentile ─────────────────────────────────
    if "volatility" in df.columns and "close" in df.columns:
        atr_proxy = df["volatility"] * df["close"]
        atr_valid = atr_proxy.dropna()
        atr_window = min(lookback_bars, len(atr_valid))
        current_atr = vol * float(last["close"])
        if atr_window >= 10 and not math.isnan(current_atr) and current_atr > 0:
            recent_atr = atr_valid.iloc[-atr_window:]
            atr_pctile = float((recent_atr < current_atr).mean())
        else:
            atr_pctile = 0.5
    else:
        atr_pctile = 0.5

    # ── Trend efficiency ─────────────────────────────────────
    eff_window = min(lookback_bars, n)
    if eff_window >= 10 and "close" in df.columns:
        closes = np.asarray(df["close"].iloc[-eff_window:].values, dtype=np.float64)
        net_move = abs(float(closes[-1]) - float(closes[0]))
        bar_moves = float(np.nansum(np.abs(np.diff(closes))))
        trend_eff = net_move / bar_moves if bar_moves > 1e-12 else 0.0
        trend_eff = min(max(trend_eff, 0.0), 1.0)
    else:
        trend_eff = 0.5

    # ── Trend strength ───────────────────────────────────────
    ts_val = float(last.get("trend_strength", 0.0))
    if math.isnan(ts_val) or math.isinf(ts_val):
        ts_val = 0.0

    # ── Regime strength ──────────────────────────────────────
    rc = float(last.get("rc", 0.0))
    ar = float(last.get("ar", 0.0))
    rc = rc if not (math.isnan(rc) or math.isinf(rc)) else 0.0
    ar = ar if not (math.isnan(ar) or math.isinf(ar)) else 0.0
    regime_str = min(max(abs(rc), abs(ar)) / max(regime_denom, 1e-9), 1.0)

    # ── Price z ──────────────────────────────────────────────
    pz = float(last.get("price_z", 0.0))
    if math.isnan(pz) or math.isinf(pz):
        pz = 0.0

    source_ts = last.get("market_time") if "market_time" in df.columns else None

    return MarketState(
        volatility=vol,
        volatility_pctile=vol_pctile,
        atr_pctile=atr_pctile,
        trend_efficiency=trend_eff,
        trend_strength=ts_val,
        regime_strength=regime_str,
        price_z=pz,
        source_candle_ts=source_ts,
        computed_from_bars=n,
        valid=True,
    )


# ── Adaptive Config Schema ────────────────────────────────────────────────────

@dataclass
class AdaptiveParamBounds:
    """Per-parameter adaptive bounds and enable flag."""
    enabled: bool = True
    min_val: float = 0.0
    max_val: float = float("inf")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AdaptiveParamBounds:
        return cls(
            enabled=bool(d.get("enabled", True)),
            min_val=float(d.get("min", 0.0)),
            max_val=float(d.get("max", float("inf"))),
        )


# Default bounds for each adaptive parameter (used when config omits them)
_DEFAULT_BOUNDS: Dict[str, Dict[str, Any]] = {
    "trend_min":      {"enabled": True, "min": 0.0005, "max": 0.01},
    "stretch_z_min":  {"enabled": True, "min": 0.3,    "max": 2.5},
    "mom_k":          {"enabled": True, "min": 0.8,    "max": 3.0},
    "min_confidence":  {"enabled": True, "min": 0.35,   "max": 0.85},
    "mr_tp_mult":     {"enabled": True, "min": 0.5,    "max": 2.0},
    "mr_sl_mult":     {"enabled": True, "min": 0.5,    "max": 2.5},
    "mom_tp_mult":    {"enabled": True, "min": 1.0,    "max": 4.0},
    "mom_sl_mult":    {"enabled": True, "min": 0.5,    "max": 2.0},
}


@dataclass
class AdaptiveConfig:
    """Top-level adaptive configuration parsed from strategy.adaptive YAML."""
    enabled: bool = False
    shadow_mode: bool = False       # compute + log, don't apply to live decisions
    update_mode: str = "bar_close"  # only "bar_close" is supported
    lookback_bars: int = 100
    params: Dict[str, AdaptiveParamBounds] = field(default_factory=dict)
    per_timeframe: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure every known param has bounds (merge defaults)."""
        for name, defaults in _DEFAULT_BOUNDS.items():
            if name not in self.params:
                self.params[name] = AdaptiveParamBounds.from_dict(defaults)

    def is_param_enabled(self, param_name: str, timeframe: Optional[str] = None) -> bool:
        """Check if a specific parameter's adaptive logic is active."""
        if not self.enabled:
            return False
        bounds = self.get_bounds(param_name, timeframe)
        return bounds.enabled

    def get_bounds(self, param_name: str, timeframe: Optional[str] = None) -> AdaptiveParamBounds:
        """Get bounds for a parameter, with optional per-timeframe override."""
        base = self.params.get(param_name, AdaptiveParamBounds())
        if timeframe and timeframe in self.per_timeframe:
            tf_overrides = self.per_timeframe[timeframe]
            if param_name in tf_overrides:
                override = tf_overrides[param_name]
                if isinstance(override, dict):
                    return AdaptiveParamBounds(
                        enabled=bool(override.get("enabled", base.enabled)),
                        min_val=float(override.get("min", base.min_val)),
                        max_val=float(override.get("max", base.max_val)),
                    )
        return base

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> AdaptiveConfig:
        """Parse an AdaptiveConfig from the strategy.adaptive dict."""
        if not d:
            return cls()
        params: Dict[str, AdaptiveParamBounds] = {}
        for name, pdict in d.get("params", {}).items():
            if isinstance(pdict, dict):
                params[name] = AdaptiveParamBounds.from_dict(pdict)
        return cls(
            enabled=bool(d.get("enabled", False)),
            shadow_mode=bool(d.get("shadow_mode", False)),
            update_mode=str(d.get("update_mode", "bar_close")),
            lookback_bars=int(d.get("lookback_bars", 100)),
            params=params,
            per_timeframe=dict(d.get("per_timeframe", {})),
        )


# ── Adaptive Computation Result ───────────────────────────────────────────────

@dataclass
class AdaptiveResult:
    """Complete result of adaptive parameter computation.

    Carries base, adaptive, and effective values alongside the market state
    and per-parameter reason codes for full auditability.
    """
    base_values: Dict[str, float]
    adaptive_values: Dict[str, float]
    effective_values: Dict[str, float]   # = adaptive unless shadow_mode
    deltas: Dict[str, float]             # adaptive - base
    market_state: MarketState
    reasons: Dict[str, str]              # per-param reason code
    shadow_mode: bool = False

    def to_audit_dict(self) -> Dict[str, Any]:
        """Produce a JSON-serializable dict for logging and DB storage."""
        return {
            "base": {k: round(v, 6) for k, v in self.base_values.items()},
            "adaptive": {k: round(v, 6) for k, v in self.adaptive_values.items()},
            "effective": {k: round(v, 6) for k, v in self.effective_values.items()},
            "deltas": {k: round(v, 6) for k, v in self.deltas.items()},
            "shadow_mode": self.shadow_mode,
            "market_state": {
                "volatility": round(self.market_state.volatility, 8),
                "volatility_pctile": round(self.market_state.volatility_pctile, 4),
                "atr_pctile": round(self.market_state.atr_pctile, 4),
                "trend_efficiency": round(self.market_state.trend_efficiency, 4),
                "trend_strength": round(self.market_state.trend_strength, 8),
                "regime_strength": round(self.market_state.regime_strength, 4),
                "price_z": round(self.market_state.price_z, 4),
                "valid": self.market_state.valid,
                "fallback_reason": self.market_state.fallback_reason,
                "source_candle_ts": str(self.market_state.source_candle_ts)
                    if self.market_state.source_candle_ts else None,
                "computed_from_bars": self.market_state.computed_from_bars,
            },
            "reasons": self.reasons,
        }


# ── Core Computation ──────────────────────────────────────────────────────────

def _safe_clip(value: float, min_val: float, max_val: float) -> float:
    """Clip value to [min_val, max_val], NaN/inf-safe."""
    if math.isnan(value) or math.isinf(value):
        return (min_val + max_val) / 2.0  # midpoint fallback
    return float(np.clip(value, min_val, max_val))


def compute_adaptive_strategy_params(
    base_config: Dict[str, Any],
    market_state: MarketState,
    adaptive_cfg: AdaptiveConfig,
    timeframe: Optional[str] = None,
) -> AdaptiveResult:
    """Compute adaptive effective values from base config and market state.

    This is the **single canonical computation function**. All adaptive
    adjustments flow through here.

    Args:
        base_config: resolved strategy config (output of resolve_effective_strategy_config)
        market_state: current bar-close market state snapshot
        adaptive_cfg: parsed AdaptiveConfig from strategy.adaptive YAML
        timeframe: current timeframe label (for per-TF bound overrides)

    Returns:
        AdaptiveResult with base, adaptive, effective values and audit info.
    """
    # ── Extract base values from resolved config ──────────────
    gates = base_config.get("gates", {})
    signal_cfg = base_config.get("signal", {})
    tp_sl = base_config.get("tp_sl", {})
    confidence = base_config.get("confidence", {})

    base_values: Dict[str, float] = {
        "trend_min":      float(gates.get("trend_min", 0.002)),
        "stretch_z_min":  float(gates.get("stretch_z_min", 1.0)),
        "mom_k":          float(signal_cfg.get("mom_k", 1.5)),
        "min_confidence":  float(confidence.get("min_confidence", 0.55)),
        "mr_tp_mult":     float(tp_sl.get("mr_tp_mult", 1.0)),
        "mr_sl_mult":     float(tp_sl.get("mr_sl_mult", 1.2)),
        "mom_tp_mult":    float(tp_sl.get("mom_tp_mult", 2.0)),
        "mom_sl_mult":    float(tp_sl.get("mom_sl_mult", 1.0)),
    }

    # ── Fast path: adaptive disabled or invalid state → return base ──
    if not adaptive_cfg.enabled or not market_state.valid:
        reason = (
            "adaptive_disabled" if not adaptive_cfg.enabled
            else f"invalid_market_state:{market_state.fallback_reason}"
        )
        reasons = {k: reason for k in base_values}
        return AdaptiveResult(
            base_values=dict(base_values),
            adaptive_values=dict(base_values),
            effective_values=dict(base_values),
            deltas={k: 0.0 for k in base_values},
            market_state=market_state,
            reasons=reasons,
            shadow_mode=adaptive_cfg.shadow_mode,
        )

    ms = market_state
    adaptive_values: Dict[str, float] = {}
    reasons: Dict[str, str] = {}

    # ── Precompute common scaling factors ─────────────────────
    # vol_factor: [-1, 1]  — negative = low vol, positive = high vol
    vol_factor = (ms.volatility_pctile - 0.5) * 2.0
    # chop_factor: [0, 1]  — 0 = perfect trend, 1 = pure chop
    chop_factor = 1.0 - ms.trend_efficiency
    # regime_factor: [0, 1]  — 0 = weak regime, 1 = strong regime
    regime_factor = ms.regime_strength

    # ── 1. gates.trend_min ────────────────────────────────────
    # High chop + high vol → raise threshold (need stronger trend confirmation)
    # Smooth persistent trend + low vol → lower slightly
    # Multiplier range: ~0.70 (clean trend, low vol) to ~1.50 (chop, high vol)
    bounds = adaptive_cfg.get_bounds("trend_min", timeframe)
    if bounds.enabled:
        multiplier = 1.0 + (
            0.50 * chop_factor * max(vol_factor, 0.0)
            - 0.30 * ms.trend_efficiency * max(-vol_factor, 0.0)
        )
        raw = base_values["trend_min"] * multiplier
        adaptive_values["trend_min"] = _safe_clip(raw, bounds.min_val, bounds.max_val)
        reasons["trend_min"] = (
            f"mult={multiplier:.4f} chop_factor={chop_factor:.3f} "
            f"volatility_factor={vol_factor:.3f} trend_efficiency={ms.trend_efficiency:.3f}"
        )
    else:
        adaptive_values["trend_min"] = base_values["trend_min"]
        reasons["trend_min"] = "param_disabled"

    # ── 2. gates.stretch_z_min ────────────────────────────────
    # High vol → raise threshold (need more stretch for MR entries)
    # Low vol → lower modestly
    # Multiplier range: ~0.70 to ~1.30
    bounds = adaptive_cfg.get_bounds("stretch_z_min", timeframe)
    if bounds.enabled:
        multiplier = 1.0 + 0.30 * vol_factor
        raw = base_values["stretch_z_min"] * multiplier
        adaptive_values["stretch_z_min"] = _safe_clip(raw, bounds.min_val, bounds.max_val)
        reasons["stretch_z_min"] = f"mult={multiplier:.4f} volatility_factor={vol_factor:.3f}"
    else:
        adaptive_values["stretch_z_min"] = base_values["stretch_z_min"]
        reasons["stretch_z_min"] = "param_disabled"

    # ── 3. signal.mom_k ───────────────────────────────────────
    # High chop → raise (harder MOM entry); strong persistent trend → lower
    # Multiplier range: ~0.85 to ~1.30
    bounds = adaptive_cfg.get_bounds("mom_k", timeframe)
    if bounds.enabled:
        multiplier = 1.0 + 0.30 * chop_factor - 0.15 * ms.trend_efficiency
        raw = base_values["mom_k"] * multiplier
        adaptive_values["mom_k"] = _safe_clip(raw, bounds.min_val, bounds.max_val)
        reasons["mom_k"] = (
            f"mult={multiplier:.4f} chop_factor={chop_factor:.3f} trend_efficiency={ms.trend_efficiency:.3f}"
        )
    else:
        adaptive_values["mom_k"] = base_values["mom_k"]
        reasons["mom_k"] = "param_disabled"

    # ── 4. confidence.min_confidence ──────────────────────────
    # Weak regime → raise minimum (more selective)
    # Strong regime → allow slightly lower threshold
    # Multiplier range: ~0.95 to ~1.15
    bounds = adaptive_cfg.get_bounds("min_confidence", timeframe)
    if bounds.enabled:
        multiplier = 1.0 + 0.15 * (1.0 - regime_factor) - 0.05 * regime_factor
        raw = base_values["min_confidence"] * multiplier
        adaptive_values["min_confidence"] = _safe_clip(raw, bounds.min_val, bounds.max_val)
        reasons["min_confidence"] = f"mult={multiplier:.4f} regime_factor={regime_factor:.3f}"
    else:
        adaptive_values["min_confidence"] = base_values["min_confidence"]
        reasons["min_confidence"] = "param_disabled"

    # ── 5. TP/SL multipliers ─────────────────────────────────

    # MR TP mult: widen in high-vol (price swings further for MR)
    bounds = adaptive_cfg.get_bounds("mr_tp_mult", timeframe)
    if bounds.enabled:
        multiplier = 1.0 + 0.20 * vol_factor
        raw = base_values["mr_tp_mult"] * multiplier
        adaptive_values["mr_tp_mult"] = _safe_clip(raw, bounds.min_val, bounds.max_val)
        reasons["mr_tp_mult"] = f"mult={multiplier:.4f} volatility_factor={vol_factor:.3f}"
    else:
        adaptive_values["mr_tp_mult"] = base_values["mr_tp_mult"]
        reasons["mr_tp_mult"] = "param_disabled"

    # MR SL mult: widen SL in high-vol (more breathing room)
    bounds = adaptive_cfg.get_bounds("mr_sl_mult", timeframe)
    if bounds.enabled:
        multiplier = 1.0 + 0.25 * max(vol_factor, 0.0)
        raw = base_values["mr_sl_mult"] * multiplier
        adaptive_values["mr_sl_mult"] = _safe_clip(raw, bounds.min_val, bounds.max_val)
        reasons["mr_sl_mult"] = f"mult={multiplier:.4f} volatility_factor={vol_factor:.3f}"
    else:
        adaptive_values["mr_sl_mult"] = base_values["mr_sl_mult"]
        reasons["mr_sl_mult"] = "param_disabled"

    # MOM TP mult: allow larger TP in strong trends
    bounds = adaptive_cfg.get_bounds("mom_tp_mult", timeframe)
    if bounds.enabled:
        multiplier = 1.0 + 0.20 * regime_factor + 0.15 * ms.trend_efficiency
        raw = base_values["mom_tp_mult"] * multiplier
        adaptive_values["mom_tp_mult"] = _safe_clip(raw, bounds.min_val, bounds.max_val)
        reasons["mom_tp_mult"] = (
            f"mult={multiplier:.4f} regime_factor={regime_factor:.3f} trend_efficiency={ms.trend_efficiency:.3f}"
        )
    else:
        adaptive_values["mom_tp_mult"] = base_values["mom_tp_mult"]
        reasons["mom_tp_mult"] = "param_disabled"

    # MOM SL mult: slightly widen in high-vol
    bounds = adaptive_cfg.get_bounds("mom_sl_mult", timeframe)
    if bounds.enabled:
        multiplier = 1.0 + 0.20 * max(vol_factor, 0.0)
        raw = base_values["mom_sl_mult"] * multiplier
        adaptive_values["mom_sl_mult"] = _safe_clip(raw, bounds.min_val, bounds.max_val)
        reasons["mom_sl_mult"] = f"mult={multiplier:.4f} volatility_factor={vol_factor:.3f}"
    else:
        adaptive_values["mom_sl_mult"] = base_values["mom_sl_mult"]
        reasons["mom_sl_mult"] = "param_disabled"

    # ── Compute effective values (≡ adaptive unless shadow mode) ─
    effective_values = dict(base_values) if adaptive_cfg.shadow_mode else dict(adaptive_values)
    deltas = {k: adaptive_values[k] - base_values[k] for k in base_values}

    return AdaptiveResult(
        base_values=base_values,
        adaptive_values=adaptive_values,
        effective_values=effective_values,
        deltas=deltas,
        market_state=market_state,
        reasons=reasons,
        shadow_mode=adaptive_cfg.shadow_mode,
    )


# ── Adaptive Cache (bar-close determinism) ────────────────────────────────────

class AdaptiveCache:
    """Per-timeframe cache ensuring same bar → same adaptive result.

    Prevents repeated loop cycles from recomputing or altering adaptive
    parameters for the same closed candle.

    **Runtime-only**: this cache is NOT used during historical simulation
    or calibration.  Historical evaluation calls ``precompute_adaptive_arrays``
    which computes per-bar values from scratch using backward-looking data only.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[Any, AdaptiveResult]] = {}

    def get(self, timeframe: str, candle_ts: Any) -> Optional[AdaptiveResult]:
        """Return cached result if candle_ts matches; else None."""
        entry = self._cache.get(timeframe)
        if entry is not None and entry[0] == candle_ts:
            return entry[1]
        return None

    def put(self, timeframe: str, candle_ts: Any, result: AdaptiveResult) -> None:
        self._cache[timeframe] = (candle_ts, result)

    def clear(self, timeframe: Optional[str] = None) -> None:
        if timeframe:
            self._cache.pop(timeframe, None)
        else:
            self._cache.clear()


# ── Per-Bar Precomputation for Historical Evaluation ──────────────────────────

def precompute_adaptive_arrays(
    df: pd.DataFrame,
    base_config: Dict[str, Any],
    adaptive_cfg: AdaptiveConfig,
    timeframe: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Precompute per-bar adaptive effective values for historical simulation.

    For each bar *i*, computes a ``MarketState`` using only backward-looking
    data (no future leakage), then applies the canonical
    ``compute_adaptive_strategy_params`` to produce per-bar effective values.

    Returns a dict with arrays keyed by parameter name (``trend_min``,
    ``stretch_z_min``, ``mom_k``).  Bars with insufficient data (< 20)
    fall back to base (static) values.

    This function is called **once per DataFrame** and the returned arrays
    are reused across all calibration grid candidates, avoiding redundant
    computation.
    """
    n = len(df)

    # Extract base values for initialisation
    gates = base_config.get("gates", {})
    signal_cfg = base_config.get("signal", {})

    base_vals = {
        "trend_min": float(gates.get("trend_min", 0.002)),
        "stretch_z_min": float(gates.get("stretch_z_min", 1.0)),
        "mom_k": float(signal_cfg.get("mom_k", 1.5)),
    }

    # Initialise every bar to the static baseline
    arrays: Dict[str, np.ndarray] = {k: np.full(n, v) for k, v in base_vals.items()}

    if not adaptive_cfg.enabled or n < 20:
        return arrays

    lookback = adaptive_cfg.lookback_bars
    regime_denom = float(
        base_config.get("confidence", {}).get("regime_denom", 0.25)
    )

    # ── Precompute rolling market-state component arrays ──────

    # Volatility (NaN/inf → 0)
    vol_raw = df["volatility"].values if "volatility" in df.columns else np.zeros(n)
    vol_arr = np.where(
        np.isnan(vol_raw) | np.isinf(vol_raw), 0.0,
        np.asarray(vol_raw, dtype=np.float64),
    )

    # Close prices
    close_arr = (
        np.asarray(df["close"].values, dtype=np.float64)
        if "close" in df.columns else np.full(n, 3000.0)
    )

    # Volatility percentile (rolling rank, matches compute_market_state)
    vol_pctile_arr = (
        pd.Series(vol_arr)
        .rolling(lookback, min_periods=10)
        .apply(
            lambda x: float((x < x.iloc[-1]).mean()) if len(x) > 1 else 0.5,
            raw=False,
        )
        .fillna(0.5)
        .values
    )

    # ATR proxy percentile (rolling rank)
    atr_proxy = vol_arr * close_arr
    atr_pctile_arr = (
        pd.Series(atr_proxy)
        .rolling(lookback, min_periods=10)
        .apply(
            lambda x: float((x < x.iloc[-1]).mean()) if len(x) > 1 else 0.5,
            raw=False,
        )
        .fillna(0.5)
        .values
    )

    # Trend efficiency (rolling directional efficiency)
    trend_eff_arr = np.full(n, 0.5)
    if "close" in df.columns:
        for i in range(n):
            w = min(lookback, i + 1)
            if w < 10:
                continue
            start = max(0, i - lookback + 1)
            wc = close_arr[start:i + 1]
            net = abs(float(wc[-1]) - float(wc[0]))
            total = float(np.nansum(np.abs(np.diff(wc))))
            if total > 1e-12:
                trend_eff_arr[i] = min(max(net / total, 0.0), 1.0)

    # Trend strength (element-wise from column)
    if "trend_strength" in df.columns:
        ts_raw = df["trend_strength"].values
        ts_arr = np.where(
            np.isnan(ts_raw) | np.isinf(ts_raw), 0.0,
            np.asarray(ts_raw, dtype=np.float64),
        )
    else:
        ts_arr = np.zeros(n)

    # Regime strength: max(|rc|, |ar|) / regime_denom
    rc_raw = df["rc"].values if "rc" in df.columns else np.zeros(n)
    ar_raw = df["ar"].values if "ar" in df.columns else np.zeros(n)
    rc_safe = np.where(
        np.isnan(rc_raw) | np.isinf(rc_raw), 0.0,
        np.asarray(rc_raw, dtype=np.float64),
    )
    ar_safe = np.where(
        np.isnan(ar_raw) | np.isinf(ar_raw), 0.0,
        np.asarray(ar_raw, dtype=np.float64),
    )
    regime_str_arr = np.minimum(
        np.maximum(np.abs(rc_safe), np.abs(ar_safe)) / max(regime_denom, 1e-9),
        1.0,
    )

    # Price z (element-wise)
    if "price_z" in df.columns:
        pz_raw = df["price_z"].values
        pz_arr = np.where(
            np.isnan(pz_raw) | np.isinf(pz_raw), 0.0,
            np.asarray(pz_raw, dtype=np.float64),
        )
    else:
        pz_arr = np.zeros(n)

    # ── Apply canonical adaptive computation per bar ──────────
    for i in range(20, n):
        ms = MarketState(
            volatility=float(vol_arr[i]),
            volatility_pctile=float(vol_pctile_arr[i]),
            atr_pctile=float(atr_pctile_arr[i]),
            trend_efficiency=float(trend_eff_arr[i]),
            trend_strength=float(ts_arr[i]),
            regime_strength=float(regime_str_arr[i]),
            price_z=float(pz_arr[i]),
            computed_from_bars=min(i + 1, lookback),
            valid=True,
        )

        ar = compute_adaptive_strategy_params(
            base_config, ms, adaptive_cfg, timeframe=timeframe,
        )

        arrays["trend_min"][i] = ar.effective_values["trend_min"]
        arrays["stretch_z_min"][i] = ar.effective_values["stretch_z_min"]
        arrays["mom_k"][i] = ar.effective_values["mom_k"]

    return arrays


# ── Convenience: resolve effective config with adaptive overlay ───────────────

def resolve_effective_adaptive_config(
    base_resolved: Dict[str, Any],
    adaptive_result: AdaptiveResult,
) -> Dict[str, Any]:
    """Merge adaptive effective values back into a resolved config dict.

    Returns a NEW dict — the input ``base_resolved`` is never mutated.
    This dict is what the strategy pipeline should use for all decisions
    on the current bar.
    """
    import copy
    eff = copy.deepcopy(base_resolved)
    ev = adaptive_result.effective_values

    # Gates
    eff.setdefault("gates", {})
    eff["gates"]["trend_min"] = ev["trend_min"]
    eff["gates"]["stretch_z_min"] = ev["stretch_z_min"]

    # Signal
    eff.setdefault("signal", {})
    eff["signal"]["mom_k"] = ev["mom_k"]

    # Confidence
    eff.setdefault("confidence", {})
    eff["confidence"]["min_confidence"] = ev["min_confidence"]

    # TP/SL
    eff.setdefault("tp_sl", {})
    eff["tp_sl"]["mr_tp_mult"] = ev["mr_tp_mult"]
    eff["tp_sl"]["mr_sl_mult"] = ev["mr_sl_mult"]
    eff["tp_sl"]["mom_tp_mult"] = ev["mom_tp_mult"]
    eff["tp_sl"]["mom_sl_mult"] = ev["mom_sl_mult"]

    return eff
