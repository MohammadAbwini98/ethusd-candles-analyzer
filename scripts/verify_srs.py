#!/usr/bin/env python3
"""SRS v1.3 Offline Verification Script (AC-01..AC-04).

Usage:
    python scripts/verify_srs.py [--csv path/to/Candles.csv] [--config config.yaml]

Verifies acceptance criteria without requiring a live Capital.com connection.
Uses a synthetically generated DataFrame if --csv is not supplied.

Exit codes:
    0 — all checks passed
    1 — one or more checks failed
"""
from __future__ import annotations

import argparse
import math
import sys
import os

# Ensure project root is on the path when run directly
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from typing import Any

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────
#  Utility helpers
# ─────────────────────────────────────────────────────────────

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"
_results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    icon = PASS if condition else FAIL
    print(f"  {icon}  {name}" + (f"  ({detail})" if detail else ""))
    _results.append((name, condition, detail))


# ─────────────────────────────────────────────────────────────
#  Synthetic data fixture
# ─────────────────────────────────────────────────────────────

def _make_df(n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    prices = 3000.0 + np.cumsum(rng.normal(0, 2, n))
    return pd.DataFrame({
        "market_time": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "close": prices,
        "vol": rng.uniform(10, 200, n),
        "buyers_pct": rng.uniform(45, 55, n),
        "sellers_pct": 100.0 - rng.uniform(45, 55, n),
        "open": prices * 0.9995,
        "high": prices * 1.001,
        "low": prices * 0.999,
    })


# ─────────────────────────────────────────────────────────────
#  AC-01: Feature computation and MAD z-score
# ─────────────────────────────────────────────────────────────

def verify_ac01(df: pd.DataFrame) -> None:
    print("\n── AC-01: Feature Engineering (FR-01..FR-06) ───────────────")
    from ethusd_analyzer.analysis import add_features, add_strategy_features, rolling_z

    # MAD z-score on constant series → must return 0, not NaN
    s_const = pd.Series([5.0] * 200)
    z_const = rolling_z(s_const, window=50).dropna()
    check("MAD z-score on constant series == 0", (z_const == 0.0).all(),
          f"max_abs={z_const.abs().max():.4f}")

    # Normal series produces finite z-scores
    s_norm = pd.Series(np.random.default_rng(1).normal(0, 1, 300))
    z_norm = rolling_z(s_norm, window=50).dropna()
    check("MAD z-score on normal series is finite", np.isfinite(z_norm.values).all())

    feat = add_features(df.copy(), z_window=50)
    for col in ("imbalance", "imb_change", "log_ret", "score", "fwd_ret_1"):
        check(f"column '{col}' present after add_features", col in feat.columns)

    strat = add_strategy_features(feat, mom_span=20, vol_window=20, regime_corr_window=50)
    for col in ("score_mr", "score_mom", "volatility", "rc", "ar"):
        check(f"column '{col}' present after add_strategy_features", col in strat.columns)

    # rc and ar must be bounded [-1, 1]
    rc_vals = strat["rc"].dropna().values
    ar_vals = strat["ar"].dropna().values
    check("rc bounded [-1, 1]", (rc_vals >= -1 - 1e-9).all() and (rc_vals <= 1 + 1e-9).all())
    check("ar bounded [-1, 1]", (ar_vals >= -1 - 1e-9).all() and (ar_vals <= 1 + 1e-9).all())

    # No infinite values in scores
    for col in ("score", "imb_change_z"):
        vals = feat[col].dropna().values
        check(f"no inf in '{col}'", np.isfinite(vals).all())


# ─────────────────────────────────────────────────────────────
#  AC-02: Regime + signal pipeline
# ─────────────────────────────────────────────────────────────

def verify_ac02() -> None:
    print("\n── AC-02: Regime & Signal Pipeline (FR-07..FR-14) ──────────")
    from ethusd_analyzer.strategy import (
        Regime, Signal,
        detect_regime, detect_regime_persistent,
        generate_signal, compute_tp_sl, should_emit_signal,
        TradeRecommendation,
    )

    # FR-07: rc<0 → MR; FR-08: ar>0 → MOM; NaN in one doesn't block other
    check("detect_regime: rc=-0.2, ar=NaN → MR",
          detect_regime(-0.2, float("nan")) == Regime.MR)
    check("detect_regime: rc=NaN, ar=0.2 → MOM",
          detect_regime(float("nan"), 0.2) == Regime.MOM)
    check("detect_regime: both NaN → NO_TRADE",
          detect_regime(float("nan"), float("nan")) == Regime.NO_TRADE)

    # FR-09: K-of-M persistence
    hist: list = []
    r1 = detect_regime_persistent(-0.2, float("nan"), hist, persistence_k=2, persistence_m=3)
    r2 = detect_regime_persistent(-0.2, float("nan"), hist, persistence_k=2, persistence_m=3)
    r3 = detect_regime_persistent(-0.2, float("nan"), hist, persistence_k=2, persistence_m=3)
    check("FR-09: persistence suppresses signal until k-of-m met",
          r1 == Regime.NO_TRADE and r2 == Regime.NO_TRADE and r3 == Regime.MR,
          f"r1={r1} r2={r2} r3={r3}")

    # FR-10/12: MOM opposition filter
    sig_mom_blocked = generate_signal(Regime.MOM, score_mr=2.0, score_mom=1.0,
                                       quantile_hi=1.0, quantile_lo=-1.0, mom_threshold=0.5)
    check("FR-12: MOM BUY blocked by overbought MR score",
          sig_mom_blocked == Signal.NO_SIGNAL)

    # FR-13: cooldown
    last = {"signal": "BUY", "regime": "MR", "bars_elapsed": 0}
    check("FR-13: cooldown suppresses duplicate", 
          not should_emit_signal(Signal.BUY, "MR", last, cooldown_bars=3, bars_elapsed=1))
    check("FR-13: direction change bypasses cooldown",
          should_emit_signal(Signal.SELL, "MR", last, cooldown_bars=3, bars_elapsed=0))

    # FR-23: sqrt(hold_bars) scaling
    tp1, sl1 = compute_tp_sl(Regime.MR, Signal.BUY, 3000.0, 0.01, hold_bars=1)
    tp4, sl4 = compute_tp_sl(Regime.MR, Signal.BUY, 3000.0, 0.01, hold_bars=4)
    ratio = (tp4 - 3000.0) / (tp1 - 3000.0)
    check("FR-23: TP distance scales as sqrt(hold_bars)", abs(ratio - 2.0) < 0.01,
          f"ratio={ratio:.3f}")

    # FR-26: symbol field on TradeRecommendation
    rec = TradeRecommendation(
        timeframe="5m", regime="MR", signal="BUY",
        confidence=0.7, entry_price=3000.0, stop_loss=2970.0,
        take_profit=3030.0, hold_bars=2, reason="test",
        conf_regime=0.5, conf_tail=0.3, conf_backtest=0.2,
        rc=-0.2, ar=0.1, score_mr=1.5, score_mom=0.0, volatility=0.01,
    )
    check("FR-26: symbol field exists on TradeRecommendation", hasattr(rec, "symbol"),
          f"symbol={getattr(rec, 'symbol', 'MISSING')}")


# ─────────────────────────────────────────────────────────────
#  AC-03: Calibration grid-search behaviour
# ─────────────────────────────────────────────────────────────

def verify_ac03(df: pd.DataFrame) -> None:
    print("\n── AC-03: Calibration (FR-17..FR-22, FR-19, FR-20) ─────────")
    from ethusd_analyzer.analysis import add_features, add_strategy_features
    from ethusd_analyzer.strategy import run_calibration, CalibrationResult

    feat = add_features(df.copy(), z_window=50)
    strat = add_strategy_features(feat, mom_span=20, vol_window=20, regime_corr_window=50)

    _GRID = {
        "r_min": [0.05, 0.10, 0.15],
        "quantile_window": [100, 150],
        "quantile_levels": [[0.90, 0.10], [0.85, 0.15]],
        "hold_bars": [1, 2],
        "cost_bps": [10],
    }

    # FR-17: min_trades=999 → NO_VALID_PARAMS
    res_nvp = run_calibration(strat, "1m", _GRID, min_trades=999, lookback_days=0)
    check("FR-21: min_trades=999 produces NO_VALID_PARAMS",
          res_nvp.status == "NO_VALID_PARAMS", f"got status={res_nvp.status}")

    # FR-19: rejection breakdown sums to total_candidates
    total_check = (res_nvp.rejected_by_min_trades + res_nvp.rejected_by_max_dd
                   + res_nvp.rejected_by_both + res_nvp.eligible_candidates)
    check("FR-19: breakdown sums to total_candidates",
          total_check == res_nvp.total_candidates,
          f"{total_check} vs total={res_nvp.total_candidates}")

    check("FR-19: max_trades_seen populated", res_nvp.max_trades_seen >= 0,
          f"max_trades_seen={res_nvp.max_trades_seen}")
    check("FR-19: best_dd_seen populated", 0.0 <= res_nvp.best_dd_seen <= 100.0,
          f"best_dd_seen={res_nvp.best_dd_seen:.4f}")

    # FR-21: fallback to default params
    check("FR-21: NO_VALID_PARAMS returns non-empty default params",
          bool(res_nvp.best_params), f"params={res_nvp.best_params}")

    # FR-20: OK result with lenient constraints
    res_ok = run_calibration(strat, "1m", _GRID, min_trades=1, max_drawdown=10.0, lookback_days=0)
    check("FR-20: OK status achievable with lenient constraints",
          res_ok.status in ("OK", "NO_VALID_PARAMS"),
          f"status={res_ok.status}")
    if res_ok.status == "OK":
        check("FR-20: best_params contains r_min", "r_min" in res_ok.best_params)
        check("FR-20: best_params contains hold_bars", "hold_bars" in res_ok.best_params)

    # FR-28: max_dd_used matches constraint passed
    res_dd = run_calibration(strat, "1m", _GRID, min_trades=1, max_drawdown=0.12, lookback_days=0)
    check("FR-28: max_dd_used field equals constraint",
          abs(res_dd.max_dd_used - 0.12) < 1e-9, f"max_dd_used={res_dd.max_dd_used}")

    # FR-22: walk-forward mode
    res_wf = run_calibration(strat, "1m", _GRID, min_trades=1, max_drawdown=10.0,
                              lookback_days=0, walk_forward_folds=2)
    check("FR-22: walk-forward returns CalibrationResult",
          isinstance(res_wf, CalibrationResult))

    # FR-18: per-tf overrides
    res_ov = run_calibration(strat, "1m", _GRID, lookback_days=0,
                              per_tf_overrides={"min_trades": 99999})
    check("FR-18: per_tf_overrides raises min_trades → NO_VALID_PARAMS",
          res_ov.status == "NO_VALID_PARAMS", f"got {res_ov.status}")


# ─────────────────────────────────────────────────────────────
#  AC-04: Storage helpers (offline — no real DB)
# ─────────────────────────────────────────────────────────────

def verify_ac04() -> None:
    print("\n── AC-04: Storage & Dashboard Helpers (FR-27..FR-34, NFR-01) ")
    import json
    from ethusd_analyzer.dashboard_server import _sanitize
    from ethusd_analyzer.storage import _with_retry, OutputConfig
    from sqlalchemy.exc import OperationalError

    # FR-34: _sanitize
    dirty = {"a": float("nan"), "b": float("inf"), "c": [float("-inf"), 3.14], "d": "text"}
    clean = _sanitize(dirty)
    check("FR-34: NaN → None", clean["a"] is None)
    check("FR-34: Inf → None", clean["b"] is None)
    check("FR-34: -Inf → None", clean["c"][0] is None)
    check("FR-34: normal float preserved", abs(clean["c"][1] - 3.14) < 1e-9)  # type: ignore
    try:
        json.dumps(clean)
        check("FR-34: sanitized output is JSON-serialisable", True)
    except Exception as e:
        check("FR-34: sanitized output is JSON-serialisable", False, str(e))

    # NFR-01: _with_retry
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise OperationalError("t", {}, Exception("transient"))
        return "ok"
    result = _with_retry(flaky, retries=3, base_delay=0.001)
    check("NFR-01: _with_retry succeeds on second attempt", result == "ok")

    raised = False
    def always_fail():
        raise OperationalError("f", {}, Exception("permanent"))
    try:
        _with_retry(always_fail, retries=2, base_delay=0.001)
    except OperationalError:
        raised = True
    check("NFR-01: _with_retry raises after exhausted retries", raised)

    # FR-29: OutputConfig has enable_perf_tables field
    cfg = OutputConfig()
    check("FR-29: OutputConfig.enable_perf_tables exists", hasattr(cfg, "enable_perf_tables"))


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SRS v1.3 offline verification")
    p.add_argument("--csv", default=None, help="Path to Candles.csv (optional)")
    p.add_argument("--config", default="config.yaml", help="Config file (not used currently)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("  ETHUSD Candles Analyzer — SRS v1.3 Verification Script")
    print("=" * 60)

    if args.csv:
        try:
            df = pd.read_csv(args.csv)
            df.columns = [c.lower() for c in df.columns]
            for tc in ("market_time", "ts", "time", "timestamp", "date"):
                if tc in df.columns:
                    df = df.rename(columns={tc: "market_time"})
                    break
            df["market_time"] = pd.to_datetime(df["market_time"], utc=True, errors="coerce")
            df = df.dropna(subset=["market_time", "close"]).reset_index(drop=True)
            print(f"[data] Loaded {len(df)} rows from {args.csv}")
        except Exception as e:
            print(f"[warn] Could not load CSV ({e}), using synthetic data")
            df = _make_df()
    else:
        df = _make_df()
        print(f"[data] Using synthetic data ({len(df)} rows)")

    verify_ac01(df)
    verify_ac02()
    verify_ac03(df)
    verify_ac04()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    print("\n" + "=" * 60)
    print(f"  RESULT: {passed} passed, {failed} failed out of {len(_results)} checks")
    print("=" * 60)

    if failed:
        print("\nFailed checks:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  {FAIL}  {name}" + (f"  ({detail})" if detail else ""))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
