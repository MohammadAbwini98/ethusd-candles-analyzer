from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Types ────────────────────────────────────────────────────

class Regime(str, Enum):
    MR = "MR"
    MOM = "MOM"
    NO_TRADE = "NO_TRADE"


class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NO_SIGNAL = "NO_SIGNAL"


@dataclass
class TradeRecommendation:
    timeframe: str
    regime: str
    signal: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    hold_bars: int
    reason: str
    conf_regime: float
    conf_tail: float
    conf_backtest: float
    rc: float
    ar: float
    score_mr: float
    score_mom: float
    volatility: float
    params_json: Optional[Dict[str, Any]] = None
    symbol: str = "ETHUSD"  # FR-26


@dataclass
class CalibrationResult:
    timeframe: str
    best_params: Dict[str, Any]
    net_sharpe: float
    net_return: float
    max_drawdown: float
    n_trades: int
    win_rate: float
    param_grid_size: int
    lookback_days: int
    status: str = "OK"  # "OK" | "NO_VALID_PARAMS" | "INSUFFICIENT_DATA"
    min_trades_used: int = 20
    max_dd_used: float = 0.15           # FR-19 / FR-28
    eligible_candidates: int = 0
    total_candidates: int = 0
    rejected_by_min_trades: int = 0     # FR-19
    rejected_by_max_dd: int = 0         # FR-19
    rejected_by_both: int = 0           # FR-19
    max_trades_seen: int = 0            # FR-19
    best_dd_seen: float = 0.0           # FR-19 (lowest max_dd observed; 0.0 = no trades, not 1.0)
    rejection_reason: Optional[str] = None


# ── Cooldown / Dedup ─────────────────────────────────────────

def should_emit_signal(
    signal: Signal,
    regime: str,
    last_signal: Optional[Dict[str, Any]],
    cooldown_bars: int = 3,
    bars_elapsed: int = 0,
) -> bool:
    """Return True if signal should be emitted (not a duplicate within cooldown)."""
    if last_signal is None:
        return True
    if last_signal.get("signal") != signal.value or last_signal.get("regime") != regime:
        return True  # different signal or regime → always emit
    if bars_elapsed >= cooldown_bars:
        return True  # enough bars have passed
    return False


# ── Step 1: Regime Detection ─────────────────────────────────

def detect_regime(
    rc: float,
    ar: float,
    r_min: float = 0.10,
    a_min: float = 0.10,
) -> Regime:
    # Check rc and ar independently — NaN in one doesn't block the other
    rc_ok = not np.isnan(rc)
    ar_ok = not np.isnan(ar)
    if rc_ok and rc < -r_min:
        return Regime.MR
    if ar_ok and ar > a_min:
        return Regime.MOM
    return Regime.NO_TRADE


def detect_regime_persistent(
    rc: float,
    ar: float,
    regime_history: List[str],
    r_min: float = 0.10,
    a_min: float = 0.10,
    persistence_k: int = 1,
    persistence_m: int = 1,
) -> Regime:
    """FR-09: K-of-M regime persistence gate.

    Appends the raw regime for this bar to *regime_history* (caller-owned list,
    capped at persistence_m entries) and returns the regime only when the same
    regime label appears in at least *persistence_k* of the last *persistence_m*
    observations (including the current one).  Falls back to NO_TRADE otherwise.
    If persistence_k <= 1 and persistence_m <= 1 this is equivalent to the plain
    detect_regime() call.
    """
    raw = detect_regime(rc, ar, r_min, a_min)
    # Maintain a rolling buffer of length persistence_m
    regime_history.append(raw.value)
    if len(regime_history) > max(persistence_m, 1):
        regime_history.pop(0)
    if persistence_k <= 1 and persistence_m <= 1:
        return raw
    # Require a full window of persistence_m observations before committing
    window = regime_history[-persistence_m:]
    if len(window) < persistence_m:
        return Regime.NO_TRADE  # not enough history yet
    for candidate in (Regime.MR, Regime.MOM):
        if window.count(candidate.value) >= persistence_k:
            return candidate
    return Regime.NO_TRADE


# ── Step 2: Signal Generation ────────────────────────────────

def generate_signal(
    regime: Regime,
    score_mr: float,
    score_mom: float,
    quantile_hi: float,
    quantile_lo: float,
    mom_threshold: float,
) -> Signal:
    if regime == Regime.NO_TRADE:
        return Signal.NO_SIGNAL
    if regime == Regime.MR:
        if np.isnan(score_mr) or np.isnan(quantile_hi) or np.isnan(quantile_lo):
            return Signal.NO_SIGNAL
        if score_mr >= quantile_hi:
            return Signal.SELL
        if score_mr <= quantile_lo:
            return Signal.BUY
        return Signal.NO_SIGNAL
    if regime == Regime.MOM:
        if np.isnan(score_mom) or np.isnan(mom_threshold) or mom_threshold <= 0:
            return Signal.NO_SIGNAL
        # MOM BUY: momentum positive AND MR score not strictly overbought (opposition check)
        if score_mom > mom_threshold:
            if not np.isnan(score_mr) and not np.isnan(quantile_hi) and score_mr > quantile_hi:
                return Signal.NO_SIGNAL  # MR says overbought → skip MOM BUY
            return Signal.BUY
        # MOM SELL: momentum negative AND MR score not strictly oversold
        if score_mom < -mom_threshold:
            if not np.isnan(score_mr) and not np.isnan(quantile_lo) and score_mr < quantile_lo:
                return Signal.NO_SIGNAL  # MR says oversold → skip MOM SELL
            return Signal.SELL
        return Signal.NO_SIGNAL
    return Signal.NO_SIGNAL


# ── Step 4: TP / SL ─────────────────────────────────────────

def compute_tp_sl(
    regime: Regime,
    signal: Signal,
    close: float,
    volatility: float,
    hold_bars: int = 1,
    mr_tp_mult: float = 1.0,
    mr_sl_mult: float = 1.2,
    mom_tp_mult: float = 2.0,
    mom_sl_mult: float = 1.0,
) -> Tuple[float, float]:
    # Scale move by sqrt(hold_bars) — volatility grows as sqrt(time)
    bar_scale = max(hold_bars, 1) ** 0.5
    move = close * max(volatility, 1e-9) * bar_scale
    if regime == Regime.MR:
        tp_mult, sl_mult = mr_tp_mult, mr_sl_mult
    else:
        tp_mult, sl_mult = mom_tp_mult, mom_sl_mult

    if signal == Signal.BUY:
        tp = close + tp_mult * move
        sl = close - sl_mult * move
    else:
        tp = close - tp_mult * move
        sl = close + sl_mult * move
    return tp, sl


# ── Step 5: Confidence Score ─────────────────────────────────

def compute_confidence(
    regime: Regime,
    rc: float,
    ar: float,
    score_mr: float,
    score_mom: float,
    score_mr_median: float,
    quantile_hi: float,
    quantile_lo: float,
    mom_threshold: float,
    sharpe_recent: float,
    weight_regime: float = 0.5,
    weight_tail: float = 0.3,
    weight_backtest: float = 0.2,
    regime_denom: float = 0.25,
) -> Tuple[float, float, float, float]:
    # conf_regime
    if regime == Regime.MR:
        conf_regime = min(max(abs(rc) / regime_denom, 0.0), 1.0) if not np.isnan(rc) else 0.0
    elif regime == Regime.MOM:
        conf_regime = min(max(abs(ar) / regime_denom, 0.0), 1.0) if not np.isnan(ar) else 0.0
    else:
        conf_regime = 0.0

    # conf_tail — regime-aware
    if regime == Regime.MR:
        spread = abs(quantile_hi - quantile_lo)
        if spread > 1e-9 and not np.isnan(score_mr_median):
            conf_tail = min(max((abs(score_mr) - abs(score_mr_median)) / spread, 0.0), 1.0)
        else:
            conf_tail = 0.0
    elif regime == Regime.MOM:
        # For MOM: how extreme is momentum relative to threshold
        if mom_threshold > 1e-9 and not np.isnan(score_mom):
            conf_tail = min(max((abs(score_mom) / mom_threshold - 1.0) / 2.0, 0.0), 1.0)
        else:
            conf_tail = 0.0
    else:
        conf_tail = 0.0

    # conf_bt
    if np.isnan(sharpe_recent):
        conf_bt = 0.33  # neutral default
    else:
        conf_bt = min(max((sharpe_recent + 1.0) / 3.0, 0.0), 1.0)

    confidence = weight_regime * conf_regime + weight_tail * conf_tail + weight_backtest * conf_bt
    return confidence, conf_regime, conf_tail, conf_bt


# ── Step 3: Calibration (Walk-Forward Grid Search) ───────────

def _simulate_strategy(
    df: pd.DataFrame,
    r_min: float,
    a_min: float,
    quantile_window: int,
    quantile_hi_pct: float,
    quantile_lo_pct: float,
    hold_bars: int,
    cost_bps: int,
    mom_k: float = 1.5,
) -> Tuple[float, float, float, int, float]:
    """Simulate strategy over historical data, return (sharpe, return, max_dd, n_trades, win_rate)."""
    n = len(df)
    min_bars = quantile_window + 10
    if n < min_bars:
        # max_dd=0.0: no trades executed, drawdown is undefined (not 1.0)
        return -999.0, 0.0, 0.0, 0, 0.0

    rc_arr = df["rc"].values
    ar_arr = df["ar"].values
    score_mr_arr = df["score_mr"].values
    score_mom_arr = df["score_mom"].values
    fwd_ret_arr = df["fwd_ret_1"].values

    # Pre-compute rolling quantiles
    score_mr_series = df["score_mr"]
    q_hi_series = score_mr_series.rolling(quantile_window, min_periods=quantile_window).quantile(quantile_hi_pct).values
    q_lo_series = score_mr_series.rolling(quantile_window, min_periods=quantile_window).quantile(quantile_lo_pct).values

    # Mom threshold series
    mom_std_series = df["score_mom"].rolling(quantile_window, min_periods=quantile_window).std().values

    cost_frac = cost_bps / 10000.0
    pnl_list: List[float] = []
    wins = 0
    i = quantile_window
    hold_remaining = 0
    # Diagnostic stage counters (used only when n_trades < 2)
    _dbg_total_bars = n - 1 - quantile_window
    _dbg_no_trade_regime = 0
    _dbg_no_signal = 0
    _dbg_cooldown_blocked = 0

    while i < n - 1:
        if hold_remaining > 0:
            hold_remaining -= 1
            _dbg_cooldown_blocked += 1
            i += 1
            continue

        rc_val = rc_arr[i]
        ar_val = ar_arr[i]
        regime = detect_regime(float(rc_val), float(ar_val), r_min, a_min)
        if regime == Regime.NO_TRADE:
            _dbg_no_trade_regime += 1
            i += 1
            continue

        q_hi_val = q_hi_series[i]
        q_lo_val = q_lo_series[i]
        mom_std_val = mom_std_series[i]
        mom_threshold = mom_k * mom_std_val if not np.isnan(mom_std_val) else 0.0

        signal = generate_signal(
            regime,
            float(score_mr_arr[i]),
            float(score_mom_arr[i]),
            float(q_hi_val),
            float(q_lo_val),
            mom_threshold,
        )
        if signal == Signal.NO_SIGNAL:
            _dbg_no_signal += 1
            i += 1
            continue

        # Accumulate PnL over hold_bars
        trade_ret = 0.0
        for h_offset in range(hold_bars):
            idx = i + h_offset
            if idx >= n - 1:
                break
            bar_ret = fwd_ret_arr[idx]
            if np.isnan(bar_ret):
                continue
            if signal == Signal.BUY:
                trade_ret += bar_ret
            else:
                trade_ret -= bar_ret

        net_ret = trade_ret - 2 * cost_frac  # round-trip cost
        pnl_list.append(net_ret)
        if net_ret > 0:
            wins += 1

        hold_remaining = hold_bars - 1
        i += 1

    n_trades = len(pnl_list)
    if n_trades < 2:
        # max_dd=0.0: insufficient trades, drawdown is 0 (not 1.0)
        # Returning 1.0 here falsely causes best_dd_seen=1.0 across all candidates.
        if _dbg_total_bars > 0:
            _pct_nt = 100.0 * _dbg_no_trade_regime / _dbg_total_bars
            _pct_ns = 100.0 * _dbg_no_signal / max(1, _dbg_total_bars - _dbg_no_trade_regime)
            logger.debug(
                "[_simulate_strategy] no-trade-diag r_min=%.2f q_win=%d hold=%d "
                "total_bars=%d no_trade_regime=%d(%.0f%%) blocked_cooldown=%d "
                "no_signal=%d(%.0f%% of tradable) n_trades=%d",
                r_min, quantile_window, hold_bars,
                _dbg_total_bars, _dbg_no_trade_regime, _pct_nt,
                _dbg_cooldown_blocked, _dbg_no_signal, _pct_ns, n_trades,
            )
        return -999.0, 0.0, 0.0, n_trades, 0.0

    pnl_arr = np.array(pnl_list)
    total_ret = float(np.sum(pnl_arr))
    mean_ret = float(np.mean(pnl_arr))
    std_ret = float(np.std(pnl_arr, ddof=1))

    # Penalize low trade counts: scale Sharpe by sqrt(n_trades)/sqrt(30)
    raw_sharpe = mean_ret / std_ret if std_ret > 1e-12 else 0.0
    trade_penalty = min(1.0, (n_trades / 30.0) ** 0.5)
    sharpe = raw_sharpe * trade_penalty

    # Proper equity-curve max drawdown
    equity = np.cumsum(pnl_arr)
    running_peak = np.maximum.accumulate(equity)
    drawdowns = running_peak - equity
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    win_rate = wins / n_trades if n_trades > 0 else 0.0
    return sharpe, total_ret, max_dd, n_trades, win_rate


def _default_params(cost_bps_default: int) -> Dict[str, Any]:
    return {
        "r_min": 0.10,
        "quantile_window": 200,
        "quantile_hi": 0.90,
        "quantile_lo": 0.10,
        "hold_bars": 2,
        "cost_bps": cost_bps_default,
    }


def run_calibration(
    df: pd.DataFrame,
    timeframe: str,
    grid_config: Dict[str, Any],
    cost_bps_default: int = 10,
    max_drawdown: float = 0.15,
    lookback_days: int = 7,
    oos_fraction: float = 0.25,
    min_trades: int = 20,
    walk_forward_folds: int = 1,          # FR-22: rolling walk-forward folds (1 = IS/OOS split)
    per_tf_overrides: Optional[Dict[str, Any]] = None,  # FR-18: per-timeframe param overrides
) -> CalibrationResult:
    """Walk-forward grid-search calibration with in-sample / out-of-sample split.

    Candidates with n_trades < min_trades are excluded (FR-17).
    Tie-breaking: Sharpe DESC → net_return DESC → max_dd ASC (FR-20).
    If no candidate passes all constraints, returns status=NO_VALID_PARAMS (FR-21).
    Rejection breakdown (rejected_by_min_trades, rejected_by_max_dd, rejected_by_both,
    max_trades_seen, best_dd_seen) are always populated (FR-19).
    """
    # FR-TFA: TF-aware defaults — activate only when caller passes the global defaults (20 / 0.15).
    # If caller passes a non-default value, it is respected as an explicit override.
    # per_tf_overrides (below) always wins over everything.
    # max_dd limits scale with trade frequency: 1m produces ~1800 trades/lookback;
    # equity-curve drawdown grows with sqrt(n_trades) so short TFs need larger limits.
    _GLOBAL_MT_DEFAULT = 20
    _GLOBAL_DD_DEFAULT = 0.15
    _TF_MIN_TRADES: Dict[str, int] = {"1m": 80, "5m": 30, "15m": 15, "30m": 8}
    _TF_MAX_DD: Dict[str, float] = {"1m": 1.50, "5m": 0.30, "15m": 0.18, "30m": 0.20}
    if min_trades == _GLOBAL_MT_DEFAULT and timeframe in _TF_MIN_TRADES:
        min_trades = _TF_MIN_TRADES[timeframe]
    if abs(max_drawdown - _GLOBAL_DD_DEFAULT) < 1e-9 and timeframe in _TF_MAX_DD:
        max_drawdown = _TF_MAX_DD[timeframe]

    # FR-18: merge per-timeframe overrides into grid_config (takes precedence over TF-aware defaults)
    effective_grid = dict(grid_config)
    if per_tf_overrides:
        effective_grid.update(per_tf_overrides.get("grid", {}))
        if "min_trades" in per_tf_overrides:
            min_trades = int(per_tf_overrides["min_trades"])
        if "max_drawdown" in per_tf_overrides:
            max_drawdown = float(per_tf_overrides["max_drawdown"])
        if "lookback_days" in per_tf_overrides:
            lookback_days = int(per_tf_overrides["lookback_days"])

    # Apply lookback window
    if "market_time" in df.columns and lookback_days > 0:
        cutoff = df["market_time"].max() - pd.Timedelta(days=lookback_days)
        df_cal = df[df["market_time"] >= cutoff].copy().reset_index(drop=True)
    else:
        df_cal = df

    if len(df_cal) < 100:
        return CalibrationResult(
            timeframe=timeframe,
            best_params=_default_params(cost_bps_default),
            net_sharpe=0.0, net_return=0.0, max_drawdown=0.0,
            n_trades=0, win_rate=0.0,
            param_grid_size=0, lookback_days=lookback_days,
            status="INSUFFICIENT_DATA",
            min_trades_used=min_trades, max_dd_used=max_drawdown,
            eligible_candidates=0, total_candidates=0,
            rejection_reason=f"Only {len(df_cal)} bars available (need >=100)",
        )

    r_min_vals = effective_grid.get("r_min", [0.05, 0.10, 0.15])
    q_window_vals = effective_grid.get("quantile_window", [150, 200, 300])
    q_level_pairs = effective_grid.get("quantile_levels", [[0.85, 0.15], [0.90, 0.10], [0.95, 0.05]])
    hold_bars_vals = effective_grid.get("hold_bars", [1, 2, 3])
    cost_bps_vals = effective_grid.get("cost_bps", [5, 10, 15])

    grid = list(product(r_min_vals, q_window_vals, q_level_pairs, hold_bars_vals, cost_bps_vals))
    grid_size = len(grid)

    # FR-22: walk-forward folds — if > 1 use rolling expanding IS windows
    effective_folds = max(1, int(walk_forward_folds))
    if effective_folds > 1:
        return _run_walk_forward_calibration(
            df_cal, timeframe, grid, grid_size, cost_bps_default,
            max_drawdown, lookback_days, min_trades, effective_folds,
        )

    # ── Single IS / OOS split (classic) ──────────────────────────────────────
    split_idx = int(len(df_cal) * (1.0 - oos_fraction))
    df_is = df_cal.iloc[:split_idx].copy().reset_index(drop=True)
    df_oos = df_cal.iloc[split_idx:].copy().reset_index(drop=True)

    best_tuple = (-np.inf, -np.inf, np.inf)   # (sharpe, net_return, -max_dd) — FR-20
    best_is_result: Tuple[float, float, float, int, float] = (-999.0, 0.0, 1.0, 0, 0.0)
    best_params: Dict[str, Any] = {}
    eligible_count = 0
    # FR-19 breakdown counters
    rejected_mt = 0
    rejected_dd = 0
    rejected_both = 0
    max_trades_seen = 0
    best_dd_seen = float("inf")

    for r_min, q_window, q_levels, hold_bars, cost_bps in grid:
        q_hi_pct = q_levels[0]
        q_lo_pct = q_levels[1]
        sharpe, total_ret, max_dd, n_trades, win_rate = _simulate_strategy(
            df_is, r_min, r_min, q_window, q_hi_pct, q_lo_pct, hold_bars, cost_bps,
        )
        # FR-19: track observed extremes
        max_trades_seen = max(max_trades_seen, n_trades)
        if max_dd < best_dd_seen:
            best_dd_seen = max_dd
        # FR-19: rejection breakdown
        fails_mt = n_trades < min_trades
        fails_dd = max_dd > max_drawdown
        if fails_mt and fails_dd:
            rejected_both += 1
            continue
        if fails_mt:
            rejected_mt += 1
            continue
        if fails_dd:
            rejected_dd += 1
            continue

        eligible_count += 1
        # FR-20: tie-break by (Sharpe DESC, net_return DESC, max_dd ASC)
        obj_tuple = (sharpe, total_ret, -max_dd)
        if obj_tuple > best_tuple:
            best_tuple = obj_tuple
            best_is_result = (sharpe, total_ret, max_dd, n_trades, win_rate)
            best_params = {
                "r_min": r_min,
                "quantile_window": q_window,
                "quantile_hi": q_hi_pct,
                "quantile_lo": q_lo_pct,
                "hold_bars": hold_bars,
                "cost_bps": cost_bps,
            }

    if best_dd_seen == float("inf"):
        best_dd_seen = 0.0  # no trades executed → drawdown is 0.0, not 1.0

    # FR-21: No valid candidates fallback
    if not best_params:
        # FR-DIAG: emit regime distribution on IS data to help root-cause the failure
        if "rc" in df_is.columns and "ar" in df_is.columns:
            _rc_nan = int(df_is["rc"].isna().sum())
            _ar_nan = int(df_is["ar"].isna().sum())
            _rc_valid = len(df_is) - _rc_nan
            _ar_valid = len(df_is) - _ar_nan
            logger.warning(
                "[calibration_diag] tf=%s is_rows=%d rc_nan=%d rc_valid=%d "
                "ar_nan=%d ar_valid=%d max_trades_seen=%d best_dd_seen=%.4f "
                "min_trades_used=%d max_dd_used=%.2f",
                timeframe, len(df_is), _rc_nan, _rc_valid,
                _ar_nan, _ar_valid, max_trades_seen, best_dd_seen,
                min_trades, max_drawdown,
            )
        reason_parts: List[str] = []
        reason_parts.append(
            f"All {grid_size} candidates rejected: "
            f"{rejected_mt} by min_trades<{min_trades}, "
            f"{rejected_dd} by max_dd>{max_drawdown:.2f}, "
            f"{rejected_both} by both. "
            f"max_trades_seen={max_trades_seen}, best_dd_seen={best_dd_seen:.4f}"
        )
        rejection_reason = " | ".join(reason_parts)
        logger.warning(f"[calibration][{timeframe}] NO_VALID_PARAMS: {rejection_reason}")
        return CalibrationResult(
            timeframe=timeframe,
            best_params=_default_params(cost_bps_default),
            net_sharpe=0.0, net_return=0.0, max_drawdown=0.0,
            n_trades=0, win_rate=0.0,
            param_grid_size=grid_size, lookback_days=lookback_days,
            status="NO_VALID_PARAMS",
            min_trades_used=min_trades, max_dd_used=max_drawdown,
            eligible_candidates=0, total_candidates=grid_size,
            rejected_by_min_trades=rejected_mt,
            rejected_by_max_dd=rejected_dd,
            rejected_by_both=rejected_both,
            max_trades_seen=max_trades_seen,
            best_dd_seen=best_dd_seen,
            rejection_reason=rejection_reason,
        )

    # Validate on OOS
    oos_sharpe, oos_ret, oos_dd, oos_trades, oos_wr = _simulate_strategy(
        df_oos,
        best_params["r_min"], best_params["r_min"],
        best_params["quantile_window"],
        best_params["quantile_hi"], best_params["quantile_lo"],
        best_params["hold_bars"], best_params["cost_bps"],
    )

    # Report OOS metrics (use OOS for final metrics — more honest)
    final_sharpe = oos_sharpe if oos_trades >= 2 else best_is_result[0]
    final_ret = oos_ret if oos_trades >= 2 else best_is_result[1]
    final_dd = oos_dd if oos_trades >= 2 else best_is_result[2]
    final_trades = oos_trades if oos_trades >= 2 else best_is_result[3]
    final_wr = oos_wr if oos_trades >= 2 else best_is_result[4]

    logger.info(
        f"[calibration][{timeframe}] IS sharpe={best_is_result[0]:.3f} "
        f"OOS sharpe={oos_sharpe:.3f} trades_oos={oos_trades} "
        f"eligible={eligible_count}/{grid_size} min_trades={min_trades} "
        f"rej_mt={rejected_mt} rej_dd={rejected_dd} rej_both={rejected_both} "
        f"max_trades_seen={max_trades_seen} best_dd_seen={best_dd_seen:.4f}"
    )

    return CalibrationResult(
        timeframe=timeframe,
        best_params=best_params,
        net_sharpe=final_sharpe,
        net_return=final_ret,
        max_drawdown=final_dd,
        n_trades=final_trades,
        win_rate=final_wr,
        param_grid_size=grid_size,
        lookback_days=lookback_days,
        status="OK",
        min_trades_used=min_trades,
        max_dd_used=max_drawdown,
        eligible_candidates=eligible_count,
        total_candidates=grid_size,
        rejected_by_min_trades=rejected_mt,
        rejected_by_max_dd=rejected_dd,
        rejected_by_both=rejected_both,
        max_trades_seen=max_trades_seen,
        best_dd_seen=best_dd_seen,
    )


def _run_walk_forward_calibration(
    df_cal: pd.DataFrame,
    timeframe: str,
    grid: list,
    grid_size: int,
    cost_bps_default: int,
    max_drawdown: float,
    lookback_days: int,
    min_trades: int,
    n_folds: int,
) -> CalibrationResult:
    """FR-22: Expanding-window walk-forward calibration with *n_folds* OOS periods.

    Divides df_cal into (n_folds + 1) equal chunks.  For fold i (0-indexed):
      IS  = chunks[0 .. i]
      OOS = chunks[i+1]
    The best params found on the *last* IS fold are returned; OOS metrics are
    averaged across all folds that produced ≥2 trades.
    """
    chunk_size = max(len(df_cal) // (n_folds + 1), 50)
    all_oos_sharpes: List[float] = []
    all_oos_rets: List[float] = []
    last_best_params: Dict[str, Any] = {}
    last_eligible = 0
    rejected_mt = rejected_dd = rejected_both = 0
    max_trades_seen = 0
    best_dd_seen = float("inf")

    for fold_i in range(n_folds):
        is_end = chunk_size * (fold_i + 1)
        oos_end = chunk_size * (fold_i + 2)
        df_is_f = df_cal.iloc[:is_end].copy().reset_index(drop=True)
        df_oos_f = df_cal.iloc[is_end:oos_end].copy().reset_index(drop=True)
        if len(df_is_f) < 100 or len(df_oos_f) < 10:
            continue

        best_t: tuple = (-np.inf, -np.inf, np.inf)
        bp: Dict[str, Any] = {}
        elig = 0
        for r_min, q_window, q_levels, hold_bars, cost_bps in grid:
            sh, tr, dd, nt, wr = _simulate_strategy(
                df_is_f, r_min, r_min, q_window, q_levels[0], q_levels[1], hold_bars, cost_bps,
            )
            max_trades_seen = max(max_trades_seen, nt)
            if dd < best_dd_seen:
                best_dd_seen = dd
            if nt < min_trades and dd > max_drawdown:
                rejected_both += 1; continue
            if nt < min_trades:
                rejected_mt += 1; continue
            if dd > max_drawdown:
                rejected_dd += 1; continue
            elig += 1
            t = (sh, tr, -dd)
            if t > best_t:
                best_t = t
                bp = {"r_min": r_min, "quantile_window": q_window, "quantile_hi": q_levels[0],
                      "quantile_lo": q_levels[1], "hold_bars": hold_bars, "cost_bps": cost_bps}
        last_eligible = elig
        if bp:
            last_best_params = bp
            oos_sh, oos_tr, _, oos_nt, _ = _simulate_strategy(
                df_oos_f, bp["r_min"], bp["r_min"], bp["quantile_window"],
                bp["quantile_hi"], bp["quantile_lo"], bp["hold_bars"], bp["cost_bps"],
            )
            if oos_nt >= 2 and not np.isnan(oos_sh) and oos_sh > -999.0:
                all_oos_sharpes.append(oos_sh)
                all_oos_rets.append(oos_tr)

    if best_dd_seen == float("inf"):
        best_dd_seen = 0.0  # no trades executed → drawdown is 0.0, not 1.0

    if not last_best_params:
        reason = (
            f"Walk-forward ({n_folds} folds) — no valid params across any fold. "
            f"rej_mt={rejected_mt} rej_dd={rejected_dd} rej_both={rejected_both}"
        )
        logger.warning(f"[calibration][{timeframe}] NO_VALID_PARAMS: {reason}")
        return CalibrationResult(
            timeframe=timeframe,
            best_params=_default_params(cost_bps_default),
            net_sharpe=0.0, net_return=0.0, max_drawdown=0.0,
            n_trades=0, win_rate=0.0,
            param_grid_size=grid_size, lookback_days=lookback_days,
            status="NO_VALID_PARAMS",
            min_trades_used=min_trades, max_dd_used=max_drawdown,
            eligible_candidates=0, total_candidates=grid_size,
            rejected_by_min_trades=rejected_mt, rejected_by_max_dd=rejected_dd,
            rejected_by_both=rejected_both,
            max_trades_seen=max_trades_seen, best_dd_seen=best_dd_seen,
            rejection_reason=reason,
        )

    avg_sharpe = float(np.mean(all_oos_sharpes)) if all_oos_sharpes else 0.0
    avg_ret = float(np.mean(all_oos_rets)) if all_oos_rets else 0.0
    logger.info(
        f"[calibration][{timeframe}] walk-forward({n_folds}) avg_oos_sharpe={avg_sharpe:.3f} "
        f"folds_with_trades={len(all_oos_sharpes)}/{n_folds} eligible_last={last_eligible}/{grid_size}"
    )
    return CalibrationResult(
        timeframe=timeframe,
        best_params=last_best_params,
        net_sharpe=avg_sharpe,
        net_return=avg_ret,
        max_drawdown=0.0,
        n_trades=0, win_rate=0.0,
        param_grid_size=grid_size,
        lookback_days=lookback_days,
        status="OK",
        min_trades_used=min_trades, max_dd_used=max_drawdown,
        eligible_candidates=last_eligible, total_candidates=grid_size,
        rejected_by_min_trades=rejected_mt, rejected_by_max_dd=rejected_dd,
        rejected_by_both=rejected_both,
        max_trades_seen=max_trades_seen, best_dd_seen=best_dd_seen,
    )


# ── Main Evaluator (per timeframe, per cycle) ────────────────

def evaluate_timeframe(
    df_feat: pd.DataFrame,
    timeframe: str,
    strategy_cfg: Dict[str, Any],
    calibration_params: Optional[Dict[str, Any]] = None,
    sharpe_recent: float = 0.0,
    last_signal_info: Optional[Dict[str, Any]] = None,
    cooldown_bars: int = 3,
    tf_overrides: Optional[Dict[str, Any]] = None,       # FR-14: per-timeframe threshold overrides
    regime_history: Optional[List[str]] = None,          # FR-09: K-of-M persistence buffer (mutated)
    persistence_k: int = 1,                              # FR-09: min occurrences required
    persistence_m: int = 1,                              # FR-09: rolling window size
    symbol: str = "ETHUSD",                              # FR-26
) -> Optional[TradeRecommendation]:
    """Run the full strategy pipeline for one timeframe. Returns a recommendation or None."""
    if len(df_feat) < 60:
        return None

    regime_cfg = strategy_cfg.get("regime", {})
    signal_cfg = strategy_cfg.get("signal", {})
    tp_sl_cfg = strategy_cfg.get("tp_sl", {})
    conf_cfg = strategy_cfg.get("confidence", {})

    # FR-14: apply per-timeframe overrides on top of global config
    if tf_overrides:
        regime_cfg = {**regime_cfg, **tf_overrides.get("regime", {})}
        signal_cfg = {**signal_cfg, **tf_overrides.get("signal", {})}
        tp_sl_cfg = {**tp_sl_cfg, **tf_overrides.get("tp_sl", {})}
        conf_cfg = {**conf_cfg, **tf_overrides.get("confidence", {})}

    # Use calibrated params if available, else config defaults
    cal = calibration_params or {}
    r_min = float(cal.get("r_min", regime_cfg.get("r_min", 0.10)))
    a_min = float(regime_cfg.get("a_min", r_min))
    q_window = int(cal.get("quantile_window", signal_cfg.get("quantile_window", 200)))
    q_hi_pct = float(cal.get("quantile_hi", signal_cfg.get("quantile_hi", 0.90)))
    q_lo_pct = float(cal.get("quantile_lo", signal_cfg.get("quantile_lo", 0.10)))
    hold_bars = int(cal.get("hold_bars", 2))

    last = df_feat.iloc[-1]
    rc_val = float(last["rc"]) if not np.isnan(float(last["rc"])) else float("nan")
    ar_val = float(last["ar"]) if not np.isnan(float(last["ar"])) else float("nan")

    # Step 1: Regime (with optional K-of-M persistence — FR-09)
    if regime_history is not None and persistence_k > 1 and persistence_m > 1:
        regime = detect_regime_persistent(
            rc_val, ar_val, regime_history, r_min, a_min, persistence_k, persistence_m,
        )
    else:
        regime = detect_regime(rc_val, ar_val, r_min, a_min)
    if regime == Regime.NO_TRADE:
        return None

    # Step 2: Thresholds + signal (enforce min_bars >= q_window)
    available_bars = len(df_feat)
    eff_q_window = min(q_window, available_bars)
    score_mr_series = df_feat["score_mr"]
    q_hi_val = float(score_mr_series.rolling(eff_q_window, min_periods=max(eff_q_window // 2, 20)).quantile(q_hi_pct).iloc[-1])
    q_lo_val = float(score_mr_series.rolling(eff_q_window, min_periods=max(eff_q_window // 2, 20)).quantile(q_lo_pct).iloc[-1])

    if np.isnan(q_hi_val) or np.isnan(q_lo_val):
        return None

    mom_k = float(signal_cfg.get("mom_k", 1.5))
    mom_std = float(df_feat["score_mom"].rolling(eff_q_window, min_periods=max(eff_q_window // 2, 20)).std().iloc[-1])
    mom_threshold = mom_k * mom_std if not np.isnan(mom_std) else 0.0

    score_mr_val = float(last["score_mr"])
    score_mom_val = float(last["score_mom"]) if not np.isnan(float(last["score_mom"])) else 0.0

    signal = generate_signal(
        regime, score_mr_val, score_mom_val,
        q_hi_val, q_lo_val, mom_threshold,
    )
    if signal == Signal.NO_SIGNAL:
        return None

    # Cooldown / dedup check
    bars_elapsed = last_signal_info.get("bars_elapsed", cooldown_bars) if last_signal_info else cooldown_bars
    if not should_emit_signal(signal, regime.value, last_signal_info, cooldown_bars, bars_elapsed):
        return None

    # Step 4: TP/SL (scaled by sqrt(hold_bars))
    close = float(last["close"])
    vol = float(last["volatility"]) if not np.isnan(float(last["volatility"])) else 1e-6

    tp, sl = compute_tp_sl(
        regime, signal, close, vol, hold_bars,
        tp_sl_cfg.get("mr_tp_mult", 1.0), tp_sl_cfg.get("mr_sl_mult", 1.2),
        tp_sl_cfg.get("mom_tp_mult", 2.0), tp_sl_cfg.get("mom_sl_mult", 1.0),
    )

    # Step 5: Confidence
    median_val = float(score_mr_series.rolling(eff_q_window, min_periods=max(eff_q_window // 2, 20)).median().iloc[-1])
    confidence, c_regime, c_tail, c_bt = compute_confidence(
        regime, rc_val, ar_val,
        score_mr_val, score_mom_val, median_val,
        q_hi_val, q_lo_val, mom_threshold, sharpe_recent,
        conf_cfg.get("weight_regime", 0.5),
        conf_cfg.get("weight_tail", 0.3),
        conf_cfg.get("weight_backtest", 0.2),
        conf_cfg.get("regime_denom", 0.25),
    )

    min_conf = float(conf_cfg.get("min_confidence", 0.55))
    if confidence < min_conf:
        return None

    # Build reason
    parts = [f"Regime={regime.value}(rc={rc_val:.3f},ar={ar_val:.3f})"]
    if regime == Regime.MR:
        parts.append(f"score_mr={score_mr_val:.3f} vs [{q_lo_val:.3f},{q_hi_val:.3f}]")
    else:
        parts.append(f"score_mom={score_mom_val:.6f} vs +/-{mom_threshold:.6f}")
    parts.append(f"conf={confidence:.2f}")

    # Active params for audit
    active_params = {
        "r_min": r_min, "a_min": a_min,
        "quantile_window": q_window,
        "quantile_hi": q_hi_pct, "quantile_lo": q_lo_pct,
        "hold_bars": hold_bars, "mom_k": mom_k,
    }

    return TradeRecommendation(
        timeframe=timeframe,
        regime=regime.value,
        signal=signal.value,
        confidence=round(confidence, 4),
        entry_price=round(close, 2),
        stop_loss=round(sl, 2),
        take_profit=round(tp, 2),
        hold_bars=hold_bars,
        reason=" | ".join(parts),
        conf_regime=round(c_regime, 4),
        conf_tail=round(c_tail, 4),
        conf_backtest=round(c_bt, 4),
        rc=round(rc_val, 6),
        ar=round(ar_val, 6),
        score_mr=round(score_mr_val, 6),
        score_mom=round(score_mom_val, 6),
        volatility=round(vol, 8),
        params_json=active_params,
        symbol=symbol,
    )
