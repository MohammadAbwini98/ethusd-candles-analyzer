#!/usr/bin/env python3
"""
scripts/verify_calibration.py
──────────────────────────────
Diagnose calibration health for all timeframes and report a detailed
breakdown of regime coverage, signal generation, and candidate rejection.

Usage (from project root):
    ./ethusd_analyzer/.venv/bin/python3 scripts/verify_calibration.py
    ./ethusd_analyzer/.venv/bin/python3 scripts/verify_calibration.py --tf 1m --lookback 14
    ./ethusd_analyzer/.venv/bin/python3 scripts/verify_calibration.py --synthetic

Options:
    --tf TF            Only run for this timeframe (default: all 4)
    --lookback DAYS    Override lookback_days (default: from config.yaml)
    --synthetic        Use a synthetic test dataset instead of the database
    --verbose          Enable DEBUG logging from the strategy module
"""
from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

# Make sure the project root is in sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from ethusd_analyzer.analysis import add_features, add_strategy_features, resample_timeframe
from ethusd_analyzer.strategy import (
    Regime,
    _simulate_strategy,
    detect_regime,
    run_calibration,
)


# ─────────────────────────────────────────────────────────
#  Logging setup
# ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("verify_calibration")


# ─────────────────────────────────────────────────────────
#  Dataset helpers
# ─────────────────────────────────────────────────────────

def _load_from_db(tf: str, lookback_days: int) -> pd.DataFrame:
    """Load candle data from the configured database and build strategy features."""
    import yaml
    from ethusd_analyzer.db import DbConfig, fetch_candles, make_engine

    cfg_path = Path(__file__).parent.parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    db = cfg.get("db", cfg.get("database", {}))
    db_cfg = DbConfig(
        host=db.get("host", "localhost"),
        port=int(db.get("port", 5432)),
        database=db.get("database", db.get("name", "postgres")),
        username=db.get("username", db.get("user", "postgres")),
        password=db.get("password", ""),
    )
    engine = make_engine(db_cfg)
    out = cfg.get("db_output", cfg.get("output", {}))
    schema = out.get("schema", "ethusd_analytics")
    table = f"{schema}.candles"
    epic = cfg.get("filters", {}).get("epic", "ETHUSD")
    cols = {
        "time": "ts", "close": "close", "vol": "vol",
        "buyers_pct": "buyers_pct", "sellers_pct": "sellers_pct", "epic": "epic",
    }
    start_ts = pd.Timestamp.now("UTC") - pd.Timedelta(days=lookback_days + 2)
    df1 = fetch_candles(engine, table, cols, epic, start_ts, timeframe="1m")
    logger.info("Loaded %d 1m rows from DB", len(df1))

    if tf == "1m":
        df_tf = df1.copy()
    else:
        rule_map = {"5m": "5min", "15m": "15min", "30m": "30min"}
        rule = rule_map.get(tf)
        if rule is None:
            raise ValueError(f"Unknown timeframe: {tf}")
        df_tf = resample_timeframe(df1, rule)
        logger.info("Resampled to %s: %d rows", tf, len(df_tf))

    df_feat = add_features(df_tf, z_window=50)
    df_strat = add_strategy_features(
        df_feat, mom_span=20, vol_window=20, regime_corr_window=50
    )
    # Do NOT filter out rows with NaN rc/ar — run_calibration/_simulate_strategy
    # handle NaN rc/ar via NaN-safe detect_regime (same as production run.py path).
    return df_strat.dropna(subset=["fwd_ret_1"]).reset_index(drop=True)


def _make_synthetic_df(tf: str, n_per_tf: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic strategy DataFrame for offline verification."""
    rng = np.random.default_rng(seed)
    # Use more rows for lower-frequency TFs to compensate for aggregation
    freq_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min"}
    freq = freq_map.get(tf, "1min")
    prices = 3000.0 + np.cumsum(rng.normal(0, 2, n_per_tf))
    buyers = rng.uniform(35, 65, n_per_tf)
    df = pd.DataFrame({
        "market_time": pd.date_range("2024-01-01", periods=n_per_tf, freq=freq, tz="UTC"),
        "close": prices,
        "vol": rng.exponential(scale=100, size=n_per_tf),
        "buyers_pct": buyers,
        "sellers_pct": 100.0 - buyers,
    })
    df_feat = add_features(df, z_window=50)
    df_strat = add_strategy_features(
        df_feat, mom_span=20, vol_window=20, regime_corr_window=50
    )
    return df_strat.dropna(subset=["fwd_ret_1"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────
#  Diagnostic helpers
# ─────────────────────────────────────────────────────────

def _regime_distribution(df: pd.DataFrame, r_min: float = 0.10, a_min: float = 0.10) -> dict:
    """Count regime labels row-by-row over the DataFrame."""
    counts = {Regime.MR: 0, Regime.MOM: 0, Regime.NO_TRADE: 0}
    for rc, ar in zip(df["rc"].values, df["ar"].values):
        r = detect_regime(float(rc), float(ar), r_min, a_min)
        counts[r] += 1
    total = len(df)
    return {
        "total": total,
        "MR": counts[Regime.MR],
        "MOM": counts[Regime.MOM],
        "NO_TRADE": counts[Regime.NO_TRADE],
        "MR_pct": 100.0 * counts[Regime.MR] / max(1, total),
        "MOM_pct": 100.0 * counts[Regime.MOM] / max(1, total),
        "NO_TRADE_pct": 100.0 * counts[Regime.NO_TRADE] / max(1, total),
        "rc_nan": int(df["rc"].isna().sum()),
        "ar_nan": int(df["ar"].isna().sum()),
    }


def _simulate_with_diag(df: pd.DataFrame, r_min: float = 0.10) -> dict:
    """Run simulator with lenient params and return trade/dd diagnostics."""
    sharpe, total_ret, max_dd, n_trades, win_rate = _simulate_strategy(
        df,
        r_min=r_min, a_min=r_min,
        quantile_window=100,
        quantile_hi_pct=0.75, quantile_lo_pct=0.25,  # wider band → more signals
        hold_bars=2, cost_bps=0,
    )
    return {
        "n_trades": n_trades,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_ret": total_ret,
        "win_rate": win_rate,
    }


# ─────────────────────────────────────────────────────────
#  Core verify function
# ─────────────────────────────────────────────────────────

_GRID_VERIFY = {
    "r_min": [0.05, 0.10, 0.15],
    "quantile_window": [100, 150],
    "quantile_levels": [[0.90, 0.10], [0.85, 0.15]],
    "hold_bars": [1, 2],
    "cost_bps": [10],
}


def verify_timeframe(
    df: pd.DataFrame,
    tf: str,
    lookback_days: int = 30,
    r_min_diag: float = 0.10,
) -> dict:
    """Run full calibration + diagnostics for one timeframe."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  TIMEFRAME: {tf}   rows={len(df)}")
    print(sep)

    # 1. Regime distribution
    rd = _regime_distribution(df, r_min=r_min_diag)
    print(f"\n[1] Regime distribution (r_min={r_min_diag}):")
    print(f"    MR={rd['MR']} ({rd['MR_pct']:.1f}%)  "
          f"MOM={rd['MOM']} ({rd['MOM_pct']:.1f}%)  "
          f"NO_TRADE={rd['NO_TRADE']} ({rd['NO_TRADE_pct']:.1f}%)")
    print(f"    rc_nan={rd['rc_nan']}  ar_nan={rd['ar_nan']}")

    # 2. Simulator diagnostic (lenient params)
    sd = _simulate_with_diag(df, r_min=r_min_diag)
    print(f"\n[2] Simulator diag (lenient, q=[0.75,0.25], hold=2, cost=0):")
    print(f"    n_trades={sd['n_trades']}  max_dd={sd['max_dd']:.4f}  "
          f"sharpe={sd['sharpe']:.3f}  win_rate={sd['win_rate']:.2f}")

    # Old-bug check
    if sd['max_dd'] == 1.0 and sd['n_trades'] < 2:
        print("    *** OLD BUG DETECTED: max_dd=1.0 on no-trade — fix not applied! ***")
    elif sd['n_trades'] < 2:
        print(f"    NOTE: max_dd={sd['max_dd']:.4f} on no-trade (0.0 = correct)")

    # 3. Full calibration
    print(f"\n[3] Running run_calibration(grid={len(list(__import__('itertools').product(*_GRID_VERIFY.values())))} candidates):")
    result = run_calibration(
        df, tf,
        grid_config=_GRID_VERIFY,
        lookback_days=lookback_days,
        cost_bps_default=10,
        # Use TF-aware defaults (no per_tf_overrides → code applies TF defaults)
    )
    print(f"    status             = {result.status}")
    print(f"    total_candidates   = {result.total_candidates}")
    print(f"    eligible           = {result.eligible_candidates}")
    print(f"    rejected_by_mt     = {result.rejected_by_min_trades}  (min_trades={result.min_trades_used})")
    print(f"    rejected_by_dd     = {result.rejected_by_max_dd}  (max_dd={result.max_dd_used:.2f})")
    print(f"    rejected_by_both   = {result.rejected_by_both}")
    total_check = (result.rejected_by_min_trades + result.rejected_by_max_dd
                   + result.rejected_by_both + result.eligible_candidates)
    rej_ok = total_check == result.total_candidates
    print(f"    breakdown sum      = {total_check}  {'✓ correct' if rej_ok else '*** MISMATCH ***'}")
    print(f"    max_trades_seen    = {result.max_trades_seen}")
    print(f"    best_dd_seen       = {result.best_dd_seen:.4f}  "
          f"{'*** OLD BUG (=1.0) ***' if result.best_dd_seen == 1.0 else '✓'}")

    if result.status == "OK":
        print(f"\n    net_sharpe={result.net_sharpe:.3f}  net_return={result.net_return:.4f}  "
              f"n_trades={result.n_trades}  win_rate={result.win_rate:.2f}")
        p = result.best_params
        print(f"    best_params: r_min={p.get('r_min')} q_window={p.get('quantile_window')} "
              f"q_hi={p.get('quantile_hi')} q_lo={p.get('quantile_lo')} "
              f"hold={p.get('hold_bars')} cost={p.get('cost_bps')}")
    elif result.rejection_reason:
        print(f"\n    rejection_reason: {result.rejection_reason}")

    print()
    return {
        "tf": tf,
        "status": result.status,
        "regime_dist": rd,
        "simulator_diag": sd,
        "calibration": result,
        "breakdown_ok": rej_ok,
    }


# ─────────────────────────────────────────────────────────
#  CLI entrypoint
# ─────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify calibration health for ETH/USD strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/verify_calibration.py --synthetic
              python scripts/verify_calibration.py --tf 1m --lookback 7
              python scripts/verify_calibration.py --verbose
        """),
    )
    parser.add_argument("--tf", choices=["1m", "5m", "15m", "30m"],
                        help="Only verify this timeframe (default: all)")
    parser.add_argument("--lookback", type=int, default=30,
                        help="Lookback days for calibration data window (default: 30)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic test data instead of the database")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging from strategy module")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger("ethusd_analyzer.strategy").setLevel(logging.DEBUG)
        logging.getLogger("verify_calibration").setLevel(logging.DEBUG)

    tfs = [args.tf] if args.tf else ["1m", "5m", "15m", "30m"]

    print("=" * 60)
    print("  ETH/USD Calibration Verification")
    print(f"  mode={'synthetic' if args.synthetic else 'database'}  "
          f"lookback={args.lookback}d  tfs={','.join(tfs)}")
    print("=" * 60)

    all_results = []
    for tf in tfs:
        try:
            if args.synthetic:
                df = _make_synthetic_df(tf)
            else:
                df = _load_from_db(tf, args.lookback)
            result = verify_timeframe(df, tf, lookback_days=args.lookback)
            all_results.append(result)
        except Exception as e:
            logger.error("Failed to verify %s: %s", tf, e, exc_info=True)

    # Summary table
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    header = f"{'TF':>4}  {'Status':>16}  {'Trades':>8}  {'BestDD':>8}  {'Eligible':>8}  {'BDok':>5}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        cal = r["calibration"]
        print(
            f"{r['tf']:>4}  {cal.status:>16}  {cal.max_trades_seen:>8}  "
            f"{cal.best_dd_seen:>8.4f}  {cal.eligible_candidates:>8}  "
            f"{'Y' if r['breakdown_ok'] else 'N':>5}"
        )

    # Final verdict
    bugs_found = any(r["calibration"].best_dd_seen == 1.0 for r in all_results)
    breakdown_ok = all(r["breakdown_ok"] for r in all_results)
    print()
    if bugs_found:
        print("FAIL: best_dd_seen=1.0 detected — fix not applied correctly!")
        sys.exit(1)
    if not breakdown_ok:
        print("FAIL: rejection breakdown does not sum to total_candidates!")
        sys.exit(1)
    print("PASS: No known NO_VALID_PARAMS bugs detected.")


if __name__ == "__main__":
    main()
