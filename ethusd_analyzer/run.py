from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from typing import Any, Dict, Optional, Set

from .utils import load_config, save_json, now_ts, send_alert
from .db import DbConfig, make_engine, fetch_candles
from .analysis import add_features, add_strategy_features, resample_timeframe, correlation_table, lag_correlation, rolling_correlations
from .storage import OutputConfig, init_schema_and_tables, save_snapshot, save_signal_recommendation, save_calibration_result
from .strategy import evaluate_timeframe, run_calibration
from .dashboard_server import DashboardServer

logger = logging.getLogger(__name__)

# Module-level config snapshot set once in main() — used by alert helpers.
_cfg_raw: Dict[str, Any] = {}
# Timestamp of the last daily summary alert (epoch seconds).
_last_daily_summary: float = 0.0


# ── WhatsApp notifier auto-start ────────────────────────────────

def _start_notifier(alerts_cfg: Dict[str, Any]) -> Optional[subprocess.Popen]:
    """Spawn the Node.js notifier sidecar as a child process.

    Returns the Popen object so the caller can terminate it on exit,
    or None if alerts are disabled / node / notifier.js is not found.
    """
    if not alerts_cfg.get("enabled", False):
        return None

    wa_number = str(alerts_cfg.get("wa_number", "")).strip()
    if not wa_number:
        print("[notifier] alerts.wa_number not set in config — notifier not started")
        return None

    node_bin = shutil.which("node")
    if not node_bin:
        print("[notifier] 'node' not found in PATH — notifier not started")
        return None

    # notifier.js lives at <project_root>/notifier/notifier.js
    # The analyzer is always run from the project root.
    notifier_js = Path("notifier") / "notifier.js"
    if not notifier_js.exists():
        print(f"[notifier] {notifier_js} not found — notifier not started")
        return None

    env = os.environ.copy()
    env["WA_NUMBER"] = wa_number
    env["PORT"] = str(alerts_cfg.get("whatsapp_notifier_url", "http://127.0.0.1:3099/send").rsplit(":", 1)[-1].split("/")[0])

    proc = subprocess.Popen(
        [node_bin, str(notifier_js)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Stream notifier output in a daemon thread so it appears in the console
    import threading

    def _pipe():
        for line in proc.stdout:  # type: ignore[union-attr]
            print(f"[notifier] {line.rstrip()}")

    t = threading.Thread(target=_pipe, daemon=True)
    t.start()

    # Poll /health until the WhatsApp client is ready (session restore can take ~40s)
    import urllib.request
    import urllib.error

    port = env["PORT"]
    health_url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + 90  # wait up to 90 seconds
    print(f"[notifier] Sidecar started (pid={proc.pid}) \u2192 WA {wa_number} — waiting for WhatsApp client...")
    while time.time() < deadline:
        if proc.poll() is not None:
            print("[notifier] Sidecar exited unexpectedly during startup.")
            return proc
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                import json as _json
                data = _json.loads(r.read())
                if data.get("ready"):
                    print("[notifier] WhatsApp client ready \u2713")
                    return proc
        except Exception:
            pass
        time.sleep(2)
    print("[notifier] WARNING: WhatsApp client did not become ready within 90s — alerts will be queued until it connects")
    return proc

# ── CLI ─────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Continuous ETHUSD candles analyzer (PostgreSQL)")
    ap.add_argument("--config", required=True, help="Path to config.yaml")
    return ap.parse_args()

# ── shared helpers ──────────────────────────────────────────────

def sanity_checks(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"empty": True}
    dup = int(df["market_time"].duplicated().sum())
    monotonic = bool(df["market_time"].is_monotonic_increasing)
    bps = (df["buyers_pct"] + df["sellers_pct"] - 100.0).abs()
    return {
        "rows": int(len(df)),
        "duplicate_times": dup,
        "time_monotonic": monotonic,
        "buyers_sellers_mean_abs_diff": float(bps.mean()),
        "buyers_sellers_max_abs_diff": float(bps.max()),
    }

def write_files(tf: str, df_feat: pd.DataFrame, out_dir: Path, horizons, lag_range: int, roll_windows):
    summary = {
        "timeframe": tf,
        "rows": int(len(df_feat)),
        "start": df_feat["market_time"].min(),
        "end": df_feat["market_time"].max(),
        "latest_close": float(df_feat["close"].iloc[-1]),
        "latest_score": float(df_feat["score"].iloc[-1]),
        "buyers_plus_sellers_mean_abs_diff": float((df_feat["buyers_pct"] + df_feat["sellers_pct"] - 100.0).abs().mean()),
    }
    save_json(out_dir / f"summary_{tf}.json", summary)

    corr = correlation_table(df_feat, horizons=horizons)
    corr.to_csv(out_dir / f"corr_{tf}.csv", index=False)

    lag = lag_correlation(df_feat, horizon=1, feature="score", lag_range=lag_range)
    lag.to_csv(out_dir / f"lagcorr_{tf}.csv", index=False)

    roll = rolling_correlations(df_feat, horizon=1, feature="score", windows=roll_windows)
    roll.to_csv(out_dir / f"rollingcorr_{tf}.csv", index=False)

    return summary, corr, lag, roll

def build_timeframes(df_1m: pd.DataFrame, z_window: int, resamples):
    results = {}
    df1 = pd.DataFrame(df_1m[["market_time","close","vol","buyers_pct","sellers_pct"]])
    df1_feat = add_features(df1, z_window=z_window).dropna(subset=["fwd_ret_1"])
    results["1m"] = df1_feat

    for rule in resamples:
        dfr = resample_timeframe(df1, rule)
        dfr_feat = add_features(pd.DataFrame(dfr), z_window=z_window).dropna(subset=["fwd_ret_1"])
        tf_label = rule.replace("min", "m")
        results[tf_label] = dfr_feat
    return results

def _run_analysis_cycle(engine, schema, table, cols, epic_value, start_ts,
                        z_window, resamples, horizons, lag_range, roll_windows,
                        out_dir, out_cfg, timeframe_filter=None,
                        strategy_cfg: Optional[Dict[str, Any]] = None,
                        strategy_tfs: Optional[Set[str]] = None,
                        calibration_state: Optional[Dict[str, Any]] = None):
    """Read candles from DB, run analysis, save results."""
    df = fetch_candles(engine, table, cols, epic_value=epic_value,
                       start_ts=start_ts, timeframe=timeframe_filter)
    if df.empty:
        print("[analysis] No data yet, skipping cycle")
        return

    sc = sanity_checks(df)
    if sc.get("duplicate_times", 0) > 0 or not sc.get("time_monotonic", True):
        print(f"[sanity] duplicates={sc.get('duplicate_times')} monotonic={sc.get('time_monotonic')}")
        if sc.get("duplicate_times", 0) > 0:
            send_alert(
                f"\u26a0\ufe0f [Sanity] Duplicate timestamps detected: {sc['duplicate_times']}",
                _cfg_raw, "sanity_check_fail",
            )
    if sc.get("buyers_sellers_mean_abs_diff", 0.0) > 0.5:
        print(f"[sanity] buyers+sellers mean abs diff high: {sc.get('buyers_sellers_mean_abs_diff')}")
        send_alert(
            f"\u26a0\ufe0f [Sanity] buyers+sellers mean abs diff high: {sc.get('buyers_sellers_mean_abs_diff'):.4f}",
            _cfg_raw, "sanity_check_fail",
        )

    feats_by_tf = build_timeframes(df, z_window, resamples)

    for tf, df_feat in feats_by_tf.items():
        summary, corr, lag, roll = write_files(tf, df_feat, out_dir, horizons, lag_range, roll_windows)
        if out_cfg.enabled:
            save_snapshot(engine, out_cfg.schema, tf, summary, corr, lag, roll, out_cfg)

    print(f"[saved] files + db updated ({len(df)} rows)")

    # Strategy evaluation
    if strategy_cfg and strategy_cfg.get("enabled") and strategy_tfs and calibration_state is not None:
        _run_strategy_cycle(feats_by_tf, strategy_cfg, strategy_tfs, calibration_state, engine, out_cfg)


def _run_strategy_cycle(
    feats_by_tf: Dict[str, pd.DataFrame],
    strategy_cfg: Dict[str, Any],
    strategy_tfs: Set[str],
    calibration_state: Dict[str, Any],
    engine: Any,
    out_cfg: OutputConfig,
) -> None:
    """Run regime-gated strategy evaluation on applicable timeframes."""
    cal_cfg = strategy_cfg.get("calibration", {})
    cal_interval = float(cal_cfg.get("interval_minutes", 60)) * 60
    now = time.time()

    should_calibrate = (
        cal_cfg.get("enabled", True)
        and (now - calibration_state.get("last_calibration_time", 0.0)) >= cal_interval
    )

    regime_cfg = strategy_cfg.get("regime", {})
    cooldown_bars = int(strategy_cfg.get("cooldown_bars", 3))
    # FR-18: per-timeframe overrides map
    tf_overrides_map: Dict[str, Any] = strategy_cfg.get("timeframe_overrides", {})
    # FR-09: K-of-M regime persistence config
    persistence_k = int(regime_cfg.get("persistence_k", 1))
    persistence_m = int(regime_cfg.get("persistence_m", 1))
    # FR-22: walk-forward folds
    walk_forward_folds = int(cal_cfg.get("walk_forward_folds", 1))
    # FR-26: symbol
    epic_symbol: str = strategy_cfg.get("symbol", "ETHUSD")

    for tf, df_feat in feats_by_tf.items():
        if tf not in strategy_tfs:
            continue

        df_strat = add_strategy_features(
            df_feat,
            mom_span=int(regime_cfg.get("mom_span", 20)),
            vol_window=int(regime_cfg.get("vol_window", 20)),
            regime_corr_window=int(regime_cfg.get("regime_corr_window", 50)),
        )

        # FR-35: structured cycle log per timeframe
        n_valid_rc = len(df_strat["rc"].dropna())
        n_valid_ar = len(df_strat["ar"].dropna())
        logger.info(
            "[strategy_cycle] tf=%s rows=%d rc_valid=%d ar_valid=%d",
            tf, len(df_strat), n_valid_rc, n_valid_ar,
        )

        # Calibration (less frequent)
        if should_calibrate and len(df_strat) > 200:
            try:
                lookback_days = int(cal_cfg.get("lookback_days", 7))
                min_trades = int(cal_cfg.get("min_trades", 20))
                # FR-18: per-timeframe calibration overrides
                per_tf_cal = cal_cfg.get("per_timeframe", {}).get(tf)
                cal_result = run_calibration(
                    df_strat,
                    timeframe=tf,
                    grid_config=cal_cfg.get("grid", {}),
                    cost_bps_default=int(cal_cfg.get("cost_bps", 10)),
                    max_drawdown=float(cal_cfg.get("max_drawdown", 0.15)),
                    lookback_days=lookback_days,
                    min_trades=min_trades,
                    walk_forward_folds=walk_forward_folds,   # FR-22
                    per_tf_overrides=per_tf_cal,             # FR-18
                )
                # FR-35: structured calibration result log
                logger.info(
                    "[calibration_result] tf=%s status=%s sharpe=%.3f trades=%d "
                    "eligible=%d/%d rej_mt=%d rej_dd=%d rej_both=%d "
                    "max_trades_seen=%d best_dd_seen=%.4f",
                    tf, cal_result.status, cal_result.net_sharpe, cal_result.n_trades,
                    cal_result.eligible_candidates, cal_result.total_candidates,
                    cal_result.rejected_by_min_trades, cal_result.rejected_by_max_dd,
                    cal_result.rejected_by_both, cal_result.max_trades_seen,
                    cal_result.best_dd_seen,
                )
                # FR-4: Only update params if calibration succeeded
                if cal_result.status == "OK":
                    calibration_state["params"][tf] = cal_result.best_params
                    calibration_state["sharpe"][tf] = cal_result.net_sharpe
                else:
                    # FR-36: detailed rejection breakdown warning
                    logger.warning(
                        "[calibration_warning] tf=%s status=%s "
                        "rej_mt=%d rej_dd=%d rej_both=%d "
                        "max_trades_seen=%d best_dd_seen=%.4f reason=%s",
                        tf, cal_result.status,
                        cal_result.rejected_by_min_trades, cal_result.rejected_by_max_dd,
                        cal_result.rejected_by_both, cal_result.max_trades_seen,
                        cal_result.best_dd_seen, cal_result.rejection_reason,
                    )
                    send_alert(
                        f"\U0001f534 [Calibration][{tf}] {cal_result.status} \u2014 "
                        f"rej_mt={cal_result.rejected_by_min_trades} "
                        f"rej_dd={cal_result.rejected_by_max_dd} "
                        f"rej_both={cal_result.rejected_by_both} "
                        f"max_trades={cal_result.max_trades_seen} "
                        f"best_dd={cal_result.best_dd_seen:.4f}",
                        _cfg_raw, "calibration_warning",
                    )
                if out_cfg.enabled:
                    save_calibration_result(engine, out_cfg.schema, tf, cal_result, out_cfg)
                print(f"[calibration][{tf}] status={cal_result.status} sharpe={cal_result.net_sharpe:.3f} "
                      f"trades={cal_result.n_trades} eligible={cal_result.eligible_candidates}/{cal_result.total_candidates} "
                      f"min_trades={min_trades} rej_mt={cal_result.rejected_by_min_trades} "
                      f"rej_dd={cal_result.rejected_by_max_dd} rej_both={cal_result.rejected_by_both}")
            except Exception as e:
                logger.warning(f"[calibration][{tf}] failed: {e}")

        # Signal evaluation with cooldown
        try:
            last_sig = calibration_state.get("last_signal", {}).get(tf)
            # Increment bars_elapsed
            if last_sig is not None:
                last_sig["bars_elapsed"] = last_sig.get("bars_elapsed", 0) + 1

            # FR-09: per-TF regime history buffer
            regime_hist = calibration_state.setdefault("regime_hist", {})
            tf_regime_history: list = regime_hist.setdefault(tf, [])

            rec = evaluate_timeframe(
                df_strat,
                timeframe=tf,
                strategy_cfg=strategy_cfg,
                calibration_params=calibration_state.get("params", {}).get(tf),
                sharpe_recent=calibration_state.get("sharpe", {}).get(tf, 0.0),
                last_signal_info=last_sig,
                cooldown_bars=cooldown_bars,
                tf_overrides=tf_overrides_map.get(tf),       # FR-14
                regime_history=tf_regime_history,             # FR-09
                persistence_k=persistence_k,                  # FR-09
                persistence_m=persistence_m,                  # FR-09
                symbol=epic_symbol,                           # FR-26
            )
            if rec is not None:
                print(f"[signal][{tf}] {rec.signal} conf={rec.confidence:.2f} "
                      f"entry={rec.entry_price:.2f} TP={rec.take_profit:.2f} SL={rec.stop_loss:.2f} | {rec.reason}")
                # FR-35: structured signal log
                logger.info(
                    "[signal_fired] tf=%s signal=%s regime=%s conf=%.4f "
                    "entry=%.2f tp=%.2f sl=%.2f symbol=%s",
                    tf, rec.signal, rec.regime, rec.confidence,
                    rec.entry_price, rec.take_profit, rec.stop_loss, rec.symbol,
                )
                send_alert(
                    f"\U0001f7e2 [Signal][{tf}] {rec.signal} triggered \u2014 "
                    f"conf={rec.confidence:.2f} entry={rec.entry_price:.2f} "
                    f"TP={rec.take_profit:.2f} SL={rec.stop_loss:.2f}",
                    _cfg_raw, "signal_fired",
                )
                if out_cfg.enabled:
                    save_signal_recommendation(engine, out_cfg.schema, rec, out_cfg)
                # Update cooldown tracking
                calibration_state.setdefault("last_signal", {})[tf] = {
                    "signal": rec.signal,
                    "regime": rec.regime,
                    "bars_elapsed": 0,
                }
        except Exception as e:
            logger.warning(f"[signal][{tf}] evaluation failed: {e}")

    if should_calibrate:
        calibration_state["last_calibration_time"] = now

# ── Capital.com API mode ────────────────────────────────────────

def _run_capital_mode(cfg, engine, out_cfg, dash, out_dir,
                      z_window, resamples, horizons, lag_range, roll_windows,
                      strategy_cfg=None, strategy_tfs=None, calibration_state=None):
    from .capital_api import CapitalSession
    from .candle_builder import CandleBuilder, insert_candles

    cap_raw = cfg.raw.get("capital", {})
    epic = cap_raw.get("epic", cfg.filters.get("epic", "ETHUSD"))

    session = CapitalSession(
        api_key=cap_raw.get("api_key", ""),
        email=cap_raw.get("email", ""),
        password=cap_raw.get("password", ""),
        base_url=cap_raw.get("base_url", "https://api-capital.backend-capital.com"),
    )

    schema = out_cfg.schema
    table = f"{schema}.candles"
    cols = {"time": "ts", "close": "close", "vol": "vol",
            "buyers_pct": "buyers_pct", "sellers_pct": "sellers_pct", "epic": "epic"}

    history_hours = int(cfg.stream.get("history_hours", 720))
    save_every = int(cfg.stream.get("save_every_seconds", 60))
    tick_poll = float(cap_raw.get("tick_poll_seconds", 1))
    sentiment_poll = float(cap_raw.get("sentiment_poll_seconds", 30))

    start_ts = pd.Timestamp.now("UTC") - pd.Timedelta(hours=history_hours)

    # ── Phase 1: historical backfill ──
    backfill_enabled = bool(cap_raw.get("backfill_enabled", True))
    if not backfill_enabled:
        print("[backfill] Disabled via config (capital.backfill_enabled: false)")
    else:
        try:
            # Check latest existing candle to avoid re-fetching
            from sqlalchemy import text as sa_text
            from datetime import timedelta
            backfill_start: Optional[datetime] = None
            with engine.connect() as conn:
                latest_ts = conn.execute(
                    sa_text(f"SELECT MAX(ts) FROM {schema}.candles WHERE epic = :epic AND timeframe = '1m'"),
                    {"epic": epic},
                ).scalar()
            if latest_ts is not None:
                latest_dt = pd.Timestamp(latest_ts)
                if latest_dt.tzinfo is None:
                    latest_dt = latest_dt.tz_localize("UTC")
                else:
                    latest_dt = latest_dt.tz_convert("UTC")
                latest_dt_py = latest_dt.to_pydatetime()
                desired_start = datetime.now(timezone.utc) - timedelta(hours=history_hours)
                if latest_dt_py > desired_start:
                    # Only fetch from 1 minute after last candle
                    backfill_start = latest_dt_py + timedelta(minutes=1)
                    gap_hours = (datetime.now(timezone.utc) - latest_dt_py).total_seconds() / 3600
                    if gap_hours < 0.02:  # less than ~1 minute gap
                        print(f"[backfill] Already up to date (latest: {latest_dt_py.isoformat()}), skipping")
                    else:
                        print(f"[backfill] Resuming from {latest_dt_py.isoformat()} ({gap_hours:.1f}h gap)")
                else:
                    print(f"[backfill] Existing data too old ({latest_dt_py.isoformat()}), full backfill")
            else:
                print(f"[backfill] No existing candles, full backfill of {history_hours}h")

            if backfill_start is None or (datetime.now(timezone.utc) - backfill_start).total_seconds() > 60:
                raw_candles = session.backfill(
                    epic, "MINUTE", hours_back=history_hours, start_from=backfill_start,
                )
                if raw_candles:
                    sentiment = session.get_sentiment(epic)
                    candle_rows = [{
                        "ts": c["time"],
                        "epic": epic,
                        "timeframe": "1m",
                        "open": c["open"],
                        "high": c["high"],
                        "low": c["low"],
                        "close": c["close"],
                        "vol": c["volume"],
                        "buyers_pct": sentiment["buyers_pct"],
                        "sellers_pct": sentiment["sellers_pct"],
                    } for c in raw_candles]
                    n = insert_candles(engine, schema, candle_rows)
                    print(f"[backfill] Inserted {n} candles "
                          f"({raw_candles[0]['time']} → {raw_candles[-1]['time']})")
                else:
                    print("[backfill] No new candles returned")
        except Exception as e:
            print(f"[backfill] Failed: {e} — continuing with live data")

    # ── Phase 2: live loop ──
    print(f"[live] Starting tick poll every {tick_poll}s | sentiment every {sentiment_poll}s | analysis every {save_every}s")

    builder = CandleBuilder(epic=epic)
    sentiment_cache = {"buyers_pct": 50.0, "sellers_pct": 50.0}
    last_sentiment = 0.0
    last_analysis = 0.0

    try:
        while True:
            tnow = time.time()

            # refresh sentiment
            if (tnow - last_sentiment) >= sentiment_poll:
                try:
                    sentiment_cache = session.get_sentiment(epic)
                    last_sentiment = tnow
                except Exception as e:
                    logger.warning(f"Sentiment fetch failed: {e}")

            # poll live price
            try:
                live = session.get_live_price(epic)
                ts_now = datetime.now(timezone.utc)
                # Push live tick to dashboard
                if dash is not None:
                    dash.update_live_price(
                        price=live["mid"],
                        bid=live["bid"],
                        ask=live["ask"],
                        ts=ts_now.isoformat(),
                    )
                completed = builder.on_tick(
                    price=live["mid"],
                    volume=0.0,
                    ts=ts_now,
                    buyers_pct=sentiment_cache["buyers_pct"],
                    sellers_pct=sentiment_cache["sellers_pct"],
                )
                if completed:
                    insert_candles(engine, schema, completed)
                    for c in completed:
                        if c["timeframe"] != "tick":
                            print(f"  [{c['timeframe']}] candle closed: {c['ts']} close={c['close']:.2f}")
            except Exception as e:
                logger.warning(f"Tick poll failed: {e}")

            # analysis cycle
            if (tnow - last_analysis) >= save_every:
                last_analysis = tnow
                try:
                    _run_analysis_cycle(
                        engine, schema, table, cols, epic, start_ts,
                        z_window, resamples, horizons, lag_range, roll_windows,
                        out_dir, out_cfg, timeframe_filter="1m",
                        strategy_cfg=strategy_cfg, strategy_tfs=strategy_tfs,
                        calibration_state=calibration_state,
                    )
                    # Daily summary digest
                    global _last_daily_summary
                    now_dt = datetime.now(timezone.utc)
                    summary_hour = int(_cfg_raw.get("alerts", {}).get("daily_summary_hour_utc", 21))
                    if (
                        now_dt.hour == summary_hour
                        and (tnow - _last_daily_summary) > 3600
                        and _cfg_raw.get("alerts", {}).get("events", {}).get("daily_summary", True)
                    ):
                        _last_daily_summary = tnow
                        try:
                            from sqlalchemy import text as _sa_text
                            with engine.connect() as _conn:
                                _row_count = _conn.execute(
                                    _sa_text(f"SELECT COUNT(*) FROM {schema}.candles WHERE epic = :epic"),
                                    {"epic": epic},
                                ).scalar() or 0
                            send_alert(
                                f"\U0001f4ca [Daily Summary] {now_dt.strftime('%Y-%m-%d')} | "
                                f"rows={_row_count} | epic={epic}",
                                _cfg_raw, "daily_summary",
                            )
                        except Exception as _se:
                            logger.warning(f"Daily summary alert failed: {_se}")
                except Exception as e:
                    logger.warning(f"Analysis cycle failed: {e}")
                    send_alert(f"\U0001f4a5 [Error] Analysis cycle failed: {e}", _cfg_raw, "system_error")

            time.sleep(tick_poll)

    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
        if dash:
            dash.stop()

# ── Legacy DB-table mode ────────────────────────────────────────

def _run_db_table_mode(cfg, engine, out_cfg, dash, out_dir,
                       z_window, resamples, horizons, lag_range, roll_windows,
                       strategy_cfg=None, strategy_tfs=None, calibration_state=None):
    table = cfg.table
    cols = cfg.columns
    epic_value = cfg.filters.get("epic", "ETHUSD")

    poll_seconds = int(cfg.stream.get("poll_seconds", 10))
    history_hours = int(cfg.stream.get("history_hours", 24))
    max_bars = int(cfg.stream.get("max_bars", 20000))
    save_every = int(cfg.stream.get("save_every_seconds", 60))

    start_ts = pd.Timestamp.now("UTC") - pd.Timedelta(hours=history_hours)
    df = fetch_candles(engine, table, cols, epic_value=epic_value, start_ts=start_ts)

    if df.empty:
        print("[warn] No rows found on startup. Will keep polling…")
    else:
        print(f"Startup loaded rows: {len(df)}  range: {df.market_time.min()} -> {df.market_time.max()}")

    last_save = now_ts()

    try:
        while True:
            if df.empty:
                df = fetch_candles(engine, table, cols, epic_value=epic_value, start_ts=start_ts)
            else:
                last_time = df.market_time.max()
                new_df = fetch_candles(engine, table, cols, epic_value=epic_value, start_ts=start_ts, last_only_newer_than=last_time)
                if not new_df.empty:
                    df = pd.concat([df, new_df], ignore_index=True)
                    df = df.drop_duplicates(subset=["market_time"], keep="last").sort_values("market_time").reset_index(drop=True)
                    if len(df) > max_bars:
                        df = df.iloc[-max_bars:].reset_index(drop=True)
                    print(f"+{len(new_df)} new rows | total={len(df)} | last={df.market_time.max()}")

            tnow = now_ts()
            if (tnow - last_save) >= save_every and not df.empty:
                last_save = tnow

                sc = sanity_checks(df)
                if sc.get("duplicate_times", 0) > 0 or not sc.get("time_monotonic", True):
                    print(f"[sanity] duplicates={sc.get('duplicate_times')} monotonic={sc.get('time_monotonic')}")
                    if sc.get("duplicate_times", 0) > 0:
                        send_alert(
                            f"\u26a0\ufe0f [Sanity] Duplicate timestamps detected: {sc['duplicate_times']}",
                            _cfg_raw, "sanity_check_fail",
                        )
                if sc.get("buyers_sellers_mean_abs_diff", 0.0) > 0.5:
                    print(f"[sanity] buyers+sellers mean abs diff high: {sc.get('buyers_sellers_mean_abs_diff')}")
                    send_alert(
                        f"\u26a0\ufe0f [Sanity] buyers+sellers mean abs diff high: {sc.get('buyers_sellers_mean_abs_diff'):.4f}",
                        _cfg_raw, "sanity_check_fail",
                    )

                feats_by_tf = build_timeframes(df, z_window, resamples)

                for tf, df_feat in feats_by_tf.items():
                    summary, corr, lag, roll = write_files(tf, df_feat, out_dir, horizons, lag_range, roll_windows)
                    if out_cfg.enabled:
                        save_snapshot(engine, out_cfg.schema, tf, summary, corr, lag, roll, out_cfg)

                print("[saved] files + db updated")

                if strategy_cfg and strategy_cfg.get("enabled") and strategy_tfs and calibration_state is not None:
                    _run_strategy_cycle(feats_by_tf, strategy_cfg, strategy_tfs, calibration_state, engine, out_cfg)

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C).")
        if dash:
            dash.stop()

# ── main ────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    cfg = load_config(args.config)

    # Expose config to module-level alert helpers
    global _cfg_raw
    _cfg_raw = cfg.raw

    db_cfg = DbConfig(
        host=cfg.db.get("host","localhost"),
        port=int(cfg.db.get("port",5432)),
        database=cfg.db.get("database",""),
        username=cfg.db.get("username",""),
        password=str(cfg.db.get("password","")),
    )
    engine = make_engine(db_cfg)

    z_window = int(cfg.analysis.get("z_window", 50))
    roll_windows = list(cfg.analysis.get("rolling_corr_windows", [20, 50]))
    lag_range = int(cfg.analysis.get("lag_range", 12))
    horizons = list(cfg.analysis.get("forward_horizons", [1, 2]))
    resamples = list(cfg.analysis.get("resample_timeframes", ["5min", "15min"]))

    # DB output
    out_raw = cfg.raw.get("db_output", {})
    out_cfg = OutputConfig(
        enabled=bool(out_raw.get("enabled", True)),
        schema=str(out_raw.get("schema", "ethusd_analytics")),
        store_rolling_points=int(out_raw.get("store_rolling_points", 500)),
        retain_days=int(out_raw.get("retain_days", 14)),
    )
    if out_cfg.enabled:
        init_schema_and_tables(engine, out_cfg.schema)

    # Dashboard auto-start
    dash_raw = cfg.raw.get("dashboard", {})
    dash = None
    if bool(dash_raw.get("enabled", True)):
        dash = DashboardServer(
            engine=engine,
            schema=out_cfg.schema,
            host=str(dash_raw.get("host","127.0.0.1")),
            port=int(dash_raw.get("port",8787)),
            open_browser=bool(dash_raw.get("open_browser", True)),
        )
        dash.start()

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strategy config
    strategy_cfg = cfg.raw.get("strategy", {})
    strategy_tfs: Set[str] = set(strategy_cfg.get("timeframes", ["5m", "15m"])) if strategy_cfg.get("enabled") else set()
    calibration_state: Dict[str, Any] = {
        "last_calibration_time": 0.0,
        "params": {},
        "sharpe": {},
        "last_signal": {},
        "regime_hist": {},   # FR-09: per-TF regime classification history for K-of-M persistence
    }
    if strategy_cfg.get("enabled"):
        print(f"[strategy] Enabled for timeframes: {sorted(strategy_tfs)}")

    # Auto-start WhatsApp notifier sidecar
    alerts_cfg = cfg.raw.get("alerts", {})
    notifier_proc: Optional[subprocess.Popen] = _start_notifier(alerts_cfg)

    data_source = cfg.raw.get("data_source", "db_table")
    common = dict(
        cfg=cfg, engine=engine, out_cfg=out_cfg, dash=dash, out_dir=out_dir,
        z_window=z_window, resamples=resamples, horizons=horizons,
        lag_range=lag_range, roll_windows=roll_windows,
        strategy_cfg=strategy_cfg, strategy_tfs=strategy_tfs, calibration_state=calibration_state,
    )

    mode_label = "Capital.com API" if data_source == "capital_api" else "DB table"
    send_alert(
        f"\U0001f7e1 [Analyzer] Started \u2014 mode={mode_label} | "
        f"dashboard=http://{dash_raw.get('host','127.0.0.1')}:{dash_raw.get('port',8787)}",
        _cfg_raw, "system_error",
    )

    try:
        if data_source == "capital_api":
            print("[mode] Capital.com API — live data feed")
            _run_capital_mode(**common)
        else:
            print("[mode] DB table — polling external candles table")
            _run_db_table_mode(**common)
    except Exception as _fatal:
        send_alert(f"\U0001f4a5 [Analyzer] Fatal crash: {_fatal}", _cfg_raw, "system_error")
        raise
    finally:
        send_alert("\U0001f534 [Analyzer] Stopped.", _cfg_raw, "system_error")
        if notifier_proc is not None and notifier_proc.poll() is None:
            notifier_proc.terminate()
            try:
                notifier_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                notifier_proc.kill()
            print("[notifier] Sidecar stopped.")

if __name__ == "__main__":
    main()
