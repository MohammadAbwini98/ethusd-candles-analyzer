from __future__ import annotations

import argparse
import gc
import logging
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from typing import Any, Dict, Optional, Set

from .utils import load_config, save_json, now_ts, send_alert, set_telegram_notifier, set_email_notifier, set_macos_notifier, get_notification_tracker
from .db import DbConfig, make_engine, fetch_candles
from .analysis import add_features, add_strategy_features, resample_timeframe, correlation_table, lag_correlation, rolling_correlations
from .storage import OutputConfig, init_schema_and_tables, save_snapshot, save_signal_recommendation, save_calibration_result, save_meta_model_run
from .strategy import evaluate_timeframe, run_calibration, invalidate_meta_model_cache
from .meta_labeler import label_signals
from .meta_trainer import maybe_retrain
from .dashboard_server import DashboardServer
from .telegram_notifier import get_telegram_notifier, TelegramNotifier
from .email_notifier import get_email_notifier, EmailNotifier
from .macos_notifier import get_macos_notifier, MacOSNotifier

logger = logging.getLogger(__name__)

# Module-level config snapshot set once in main() — used by alert helpers.
_cfg_raw: Dict[str, Any] = {}
# Timestamp of the last daily summary alert (epoch seconds).
_last_daily_summary: float = 0.0
# Telegram notifier singleton (set by main after creation).
_telegram_notifier: Optional[TelegramNotifier] = None
# Email notifier singleton (set by main after creation).
_email_notifier: Optional[EmailNotifier] = None
# macOS notifier singleton (set by main after creation).
_macos_notifier: Optional[MacOSNotifier] = None
# Shutdown message sent flag (ensure sent only once)
_shutdown_msg_sent: bool = False
# Dedupe key cache for Telegram signal notifications per timeframe.
_last_sent_signal_key: Dict[str, str] = {}
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

    # Poll /health in a background daemon thread — never blocks the main analyzer loop.
    import urllib.request
    import urllib.error

    port = env["PORT"]
    health_url = f"http://127.0.0.1:{port}/health"
    print(f"[notifier] Sidecar started (pid={proc.pid}) \u2192 WA {wa_number} — WhatsApp client connecting in background...")

    def _wait_ready():
        deadline = time.time() + 120  # up to 2 minutes in background
        while time.time() < deadline:
            if proc.poll() is not None:
                print("[notifier] Sidecar exited unexpectedly.")
                return
            try:
                with urllib.request.urlopen(health_url, timeout=2) as r:
                    import json as _json
                    data = _json.loads(r.read())
                    if data.get("ready"):
                        print("[notifier] WhatsApp client ready \u2713")
                        return
            except Exception:
                pass
            time.sleep(3)
        print("[notifier] WARNING: WhatsApp client did not become ready within 120s — alerts will be delayed until it connects")

    threading.Thread(target=_wait_ready, daemon=True, name="wa-health-poll").start()
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

def write_files(tf: str, df_feat: pd.DataFrame, out_dir: Path, horizons, lag_range: int, roll_windows,
                df_eval: Optional[pd.DataFrame] = None):
    # Summary always from the FULL frame so the latest bar is reflected.
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

    # Offline correlation tables require non-null fwd_ret_* labels.
    # Use df_eval when provided; otherwise trim on the fly.
    _df_corr = df_eval if df_eval is not None else df_feat.dropna(subset=["fwd_ret_1"])

    corr = correlation_table(_df_corr, horizons=horizons)
    corr.to_csv(out_dir / f"corr_{tf}.csv", index=False)

    lag = lag_correlation(_df_corr, horizon=1, feature="score", lag_range=lag_range)
    lag.to_csv(out_dir / f"lagcorr_{tf}.csv", index=False)

    roll = rolling_correlations(_df_corr, horizon=1, feature="score", windows=roll_windows)
    roll.to_csv(out_dir / f"rollingcorr_{tf}.csv", index=False)

    return summary, corr, lag, roll

def build_timeframes(df_1m: pd.DataFrame, z_window: int, resamples):
    """Return FULL feature frames (latest bar included) for every timeframe.

    fwd_ret_1/fwd_ret_2 are kept as NaN on the last row so that offline
    correlation tables can operate on a trimmed copy while live strategy
    evaluation always sees the newest candle.
    """
    results = {}
    df1 = pd.DataFrame(df_1m[["market_time","close","vol","buyers_pct","sellers_pct"]])
    # No dropna — latest bar must be present for real-time signal generation.
    df1_feat = add_features(df1, z_window=z_window)
    results["1m"] = df1_feat

    for rule in resamples:
        dfr = resample_timeframe(df1, rule)
        dfr_feat = add_features(pd.DataFrame(dfr), z_window=z_window)
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
        # Eval frame: trim the last row (fwd_ret NaN) for offline label-based stats.
        df_eval = df_feat.dropna(subset=["fwd_ret_1"])
        summary, corr, lag, roll = write_files(
            tf, df_feat, out_dir, horizons, lag_range, roll_windows, df_eval=df_eval
        )
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
    # FR-22: walk-forward folds + OOS constraints
    walk_forward_folds = int(cal_cfg.get("walk_forward_folds", 1))
    min_trades_oos = int(cal_cfg.get("min_trades_oos", 0))
    min_folds_with_trades = int(cal_cfg.get("min_folds_with_trades", 1))
    # FR-26: symbol
    epic_symbol: str = strategy_cfg.get("symbol", "ETHUSD")

    for tf, df_feat in feats_by_tf.items():
        if tf not in strategy_tfs:
            continue

        gates_cfg = strategy_cfg.get("gates", {})
        df_strat = add_strategy_features(
            df_feat,
            mom_span=int(regime_cfg.get("mom_span", 20)),
            vol_window=int(regime_cfg.get("vol_window", 20)),
            regime_corr_window=int(regime_cfg.get("regime_corr_window", 50)),
            ema_fast_span=int(gates_cfg.get("ema_fast_span", 20)),
            ema_slow_span=int(gates_cfg.get("ema_slow_span", 50)),
            stretch_baseline_span=int(gates_cfg.get("stretch_baseline_span", 50)),
            stretch_window=int(gates_cfg.get("stretch_window", 200)),
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
                    stretch_z_min=float(gates_cfg.get("stretch_z_min", 0.0)),
                    trend_min=float(gates_cfg.get("trend_min", 0.0)),
                    min_trades_oos=min_trades_oos,
                    min_folds_with_trades=min_folds_with_trades,
                )
                # FR-35: structured calibration result log
                logger.info(
                    "[calibration_result] tf=%s status=%s sharpe=%.3f trades=%d "
                    "eligible=%d/%d rej_mt=%d rej_dd=%d rej_both=%d "
                    "max_trades_seen=%d best_dd_seen=%.4f "
                    "folds_used=%d oos_trades=%d oos_wr=%.2f worst_fold_dd=%.4f rej_oos_folds=%d",
                    tf, cal_result.status, cal_result.net_sharpe, cal_result.n_trades,
                    cal_result.eligible_candidates, cal_result.total_candidates,
                    cal_result.rejected_by_min_trades, cal_result.rejected_by_max_dd,
                    cal_result.rejected_by_both, cal_result.max_trades_seen,
                    cal_result.best_dd_seen,
                    cal_result.folds_used, cal_result.oos_trades,
                    cal_result.oos_win_rate, cal_result.worst_fold_dd,
                    cal_result.rejected_oos_folds,
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

            # Diagnostic: log evaluation attempt
            cal_params = calibration_state.get("params", {}).get(tf)
            logger.debug(
                "[signal_eval][%s] Evaluating: rows=%d cal_params=%s last_sig=%s",
                tf, len(df_strat), "present" if cal_params else "MISSING",
                last_sig.get("signal") if last_sig else "none"
            )
            
            rec = evaluate_timeframe(
                df_strat,
                timeframe=tf,
                strategy_cfg=strategy_cfg,
                calibration_params=cal_params,
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
                
                # ── Telegram: send detailed signal message ─────────────────────────
                if _telegram_notifier is not None:
                    try:
                        signal_time_val = datetime.now(timezone.utc).isoformat()
                        if "market_time" in df_strat.columns and not df_strat.empty:
                            signal_time_val = str(df_strat.iloc[-1]["market_time"])
                        dedupe_key = f"{tf}|{signal_time_val}|{rec.signal}"
                        if _last_sent_signal_key.get(tf) == dedupe_key:
                            logger.debug("[telegram_signal] Skipped duplicate signal key=%s", dedupe_key)
                        else:
                            _last_sent_signal_key[tf] = dedupe_key

                            # Compute hold duration in minutes from hold_bars
                            tf_minutes = {
                                "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                                "1h": 60, "4h": 240, "1d": 1440,
                            }.get(tf, 15)  # Default to 15m if unknown
                            hold_minutes = rec.hold_bars * tf_minutes

                            tg_msg = TelegramNotifier.format_signal_message(
                                symbol=rec.symbol,
                                timeframe=tf,
                                signal=rec.signal,
                                regime=rec.regime,
                                confidence=rec.confidence,
                                entry_price=rec.entry_price,
                                take_profit=rec.take_profit,
                                stop_loss=rec.stop_loss,
                                hold_bars=rec.hold_bars,
                                hold_minutes=hold_minutes,
                                rc=rec.rc,
                                ar=rec.ar,
                                volatility=rec.volatility,
                                price_z=(rec.params_json or {}).get("price_z"),
                                trend_strength=(rec.params_json or {}).get("trend_strength"),
                                timestamp=datetime.now(timezone.utc),
                            )
                            import threading
                            threading.Thread(
                                target=_telegram_notifier.send_message,
                                args=(tg_msg,),
                                daemon=True,
                            ).start()
                            logger.debug("[telegram_signal] Enqueued detailed signal message")
                    except Exception as e:
                        logger.warning("[telegram_signal] Failed to send detailed message: %s", e)
                
                # ── Email: send detailed signal message ───────────────────────────
                if _email_notifier is not None:
                    try:
                        signal_time_val = datetime.now(timezone.utc).isoformat()
                        if "market_time" in df_strat.columns and not df_strat.empty:
                            signal_time_val = str(df_strat.iloc[-1]["market_time"])
                        
                        # Compute hold duration in minutes from hold_bars
                        tf_minutes = {
                            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                            "1h": 60, "4h": 240, "1d": 1440,
                        }.get(tf, 15)  # Default to 15m if unknown
                        hold_minutes = rec.hold_bars * tf_minutes

                        subject, html_body = EmailNotifier.format_signal_message(
                            symbol=rec.symbol,
                            timeframe=tf,
                            signal=rec.signal,
                            regime=rec.regime,
                            confidence=rec.confidence,
                            entry_price=rec.entry_price,
                            take_profit=rec.take_profit,
                            stop_loss=rec.stop_loss,
                            hold_bars=rec.hold_bars,
                            hold_minutes=hold_minutes,
                            rc=rec.rc,
                            ar=rec.ar,
                            volatility=rec.volatility,
                            price_z=(rec.params_json or {}).get("price_z"),
                            trend_strength=(rec.params_json or {}).get("trend_strength"),
                            timestamp=datetime.now(timezone.utc),
                        )
                        import threading
                        threading.Thread(
                            target=_email_notifier.send_message,
                            args=(subject, html_body),
                            daemon=True,
                        ).start()
                        logger.debug("[email_signal] Enqueued detailed signal message")
                    except Exception as e:
                        logger.warning("[email_signal] Failed to send detailed message: %s", e)
                
                # ── macOS: send signal notification ───────────────────────────────
                if _macos_notifier is not None:
                    try:
                        # Compute hold duration in minutes from hold_bars
                        tf_minutes = {
                            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
                            "1h": 60, "4h": 240, "1d": 1440,
                        }.get(tf, 15)  # Default to 15m if unknown
                        hold_minutes = rec.hold_bars * tf_minutes

                        title, message, subtitle = MacOSNotifier.format_signal_message(
                            symbol=rec.symbol,
                            timeframe=tf,
                            signal=rec.signal,
                            confidence=rec.confidence,
                            entry_price=rec.entry_price,
                            take_profit=rec.take_profit,
                            stop_loss=rec.stop_loss,
                            hold_bars=rec.hold_bars,
                        )
                        # Use deduplication to avoid spamming same signal
                        _macos_notifier.notify_with_dedupe(tf, title, message, subtitle)
                        logger.debug("[macos_signal] Enqueued signal notification")
                    except Exception as e:
                        logger.warning("[macos_signal] Failed to send notification: %s", e)
                
                if out_cfg.enabled:
                    save_signal_recommendation(engine, out_cfg.schema, rec, out_cfg)
                # Update cooldown tracking
                calibration_state.setdefault("last_signal", {})[tf] = {
                    "signal": rec.signal,
                    "regime": rec.regime,
                    "bars_elapsed": 0,
                }
            else:
                # Diagnostic: log why no signal
                logger.debug(
                    "[signal_eval][%s] No signal generated (check strategy.py debug logs for filters)",
                    tf
                )
        except Exception as e:
            logger.warning(f"[signal][{tf}] evaluation failed: {e}", exc_info=True)

        # ── Meta-model: label past signals + conditionally retrain ─────────────
        meta_cfg: Dict[str, Any] = strategy_cfg.get("meta_model", {})
        if meta_cfg.get("enabled", False) or True:   # labeling always on (free)
            try:
                n_labeled = label_signals(engine, out_cfg.schema, tf)
                if n_labeled > 0:
                    logger.info("[meta_labeler][%s] labeled %d new outcomes", tf, n_labeled)
            except Exception as exc:
                logger.warning("[meta_labeler][%s] labeling failed: %s", tf, exc)

        if meta_cfg.get("enabled", False):
            try:
                tf_meta_state = calibration_state["meta"].setdefault(tf, {})
                meta_run = maybe_retrain(
                    engine, out_cfg.schema, tf, meta_cfg, tf_meta_state,
                )
                if meta_run is not None:
                    # Bust the in-process cache so next prediction loads new model
                    invalidate_meta_model_cache(tf)
                    logger.info(
                        "[meta_trainer][%s] Retrained — AUC=%.4f Brier=%.4f samples=%d",
                        tf, meta_run.get("auc", 0.0), meta_run.get("brier_score", 1.0),
                        meta_run.get("n_samples", 0),
                    )
                    if out_cfg.enabled:
                        save_meta_model_run(engine, out_cfg.schema, meta_run, out_cfg)
            except Exception as exc:
                logger.warning("[meta_trainer][%s] retrain cycle failed: %s", tf, exc)

    if should_calibrate:
        calibration_state["last_calibration_time"] = now

# ── Capital.com API mode ────────────────────────────────────────

def _run_capital_mode(cfg, engine, out_cfg, dash, out_dir,
                      z_window, resamples, horizons, lag_range, roll_windows,
                      strategy_cfg=None, strategy_tfs=None, calibration_state=None):
    from .capital_api import CapitalSession
    from .candle_builder import insert_candles   # still used for backfill
    from .ingestion import (
        fetch_and_upsert_1m,
        resample_and_upsert,
        insert_sentiment_tick,
    )

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
                    # Historical candles get neutral sentiment (50/50) and
                    # sentiment_ts=NULL — avoid smearing a single current
                    # sentiment value across all history.
                    candle_rows = [{
                        "ts":          c["time"],
                        "epic":        epic,
                        "timeframe":   "1m",
                        "open":        c["open"],
                        "high":        c["high"],
                        "low":         c["low"],
                        "close":       c["close"],
                        "vol":         c["volume"],
                        "buyers_pct":  50.0,
                        "sellers_pct": 50.0,
                        "sentiment_ts": None,
                    } for c in raw_candles]
                    n = insert_candles(engine, schema, candle_rows)
                    print(f"[backfill] Inserted {n} candles "
                          f"({raw_candles[0]['time']} → {raw_candles[-1]['time']})")
                else:
                    print("[backfill] No new candles returned")
        except Exception as e:
            print(f"[backfill] Failed: {e} — continuing with live data")

    # ── Phase 2: live polling ──
    candle_poll_seconds = float(cap_raw.get("candle_poll_seconds", 65))

    # Sentiment cache includes timestamp so as-of logic never smears
    sentiment_cache: Dict[str, Any] = {
        "ts": None,
        "buyers_pct": 50.0,
        "sellers_pct": 50.0,
    }
    zero_vol_state: Dict[str, int] = {"consecutive": 0}

    last_sentiment   = 0.0
    last_candle_poll = 0.0
    last_analysis    = 0.0

    print(
        f"[live] Starting — tick_poll={tick_poll}s | sentiment_poll={sentiment_poll}s "
        f"| candle_poll={candle_poll_seconds}s | analysis={save_every}s"
    )

    try:
        while True:
            tnow = time.time()

            # ── Sentiment: persist to DB + refresh in-memory cache ──────────
            if (tnow - last_sentiment) >= sentiment_poll:
                try:
                    sdata = session.get_sentiment(epic)
                    ts_now = datetime.now(timezone.utc)
                    sentiment_cache = {
                        "ts":          ts_now,
                        "buyers_pct":  sdata["buyers_pct"],
                        "sellers_pct": sdata["sellers_pct"],
                    }
                    insert_sentiment_tick(
                        engine, schema, epic, ts_now,
                        sdata["buyers_pct"], sdata["sellers_pct"],
                    )
                    logger.info(
                        "[sentiment] ts=%s buyers=%.1f%% sellers=%.1f%%",
                        ts_now.isoformat(),
                        sdata["buyers_pct"], sdata["sellers_pct"],
                    )
                    last_sentiment = tnow
                except Exception as e:
                    logger.warning(f"[sentiment] Fetch failed: {e}")

            # ── Live price: dashboard tick update ONLY (no candle building) ─
            try:
                live = session.get_live_price(epic)
                ts_now = datetime.now(timezone.utc)
                if dash is not None:
                    dash.update_live_price(
                        price=live["mid"],
                        bid=live["bid"],
                        ask=live["ask"],
                        ts=ts_now.isoformat(),
                    )
            except Exception as e:
                logger.warning(f"[tick] get_live_price failed: {e}")

            # ── Real OHLCV candle ingest + higher-TF resample ────────────────
            if (tnow - last_candle_poll) >= candle_poll_seconds:
                last_candle_poll = tnow
                try:
                    new_ts = fetch_and_upsert_1m(
                        session, engine, schema, epic, sentiment_cache,
                        zero_vol_state=zero_vol_state,
                    )
                    if new_ts:
                        resample_counts = resample_and_upsert(
                            engine, schema, epic, new_ts,
                        )
                        total_resampled = sum(resample_counts.values())
                        if total_resampled > 0:
                            logger.info(
                                "[resample] bars written: %s",
                                " | ".join(
                                    f"{tf}={n}"
                                    for tf, n in resample_counts.items()
                                ),
                            )
                except Exception as e:
                    logger.warning(f"[ingest] Candle poll cycle failed: {e}")

            # ── Analysis cycle ───────────────────────────────────────────────
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
        # Re-raise so main() finally block handles all teardown uniformly.
        raise

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
        # Re-raise so main() finally block handles all teardown uniformly.
        raise

# ── Startup/Shutdown Handlers ──────────────────────────────────

def _send_startup_alert(symbol: str = "ETH", mode: str = "LIVE") -> None:
    """Send startup notification to WhatsApp, Telegram, Email, and macOS."""
    events_cfg = _cfg_raw.get("alerts", {}).get("events", {})
    if not events_cfg.get("startup", True):
        return
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # WhatsApp (simple text via send_alert)
    whatsapp_msg = f"🟢 {symbol} Analyzer Started\n\nMode: {mode}\nTime: {timestamp}"
    send_alert(whatsapp_msg, _cfg_raw, "startup")
    
    # Telegram
    if _telegram_notifier is not None:
        msg = TelegramNotifier.format_startup_message(symbol=symbol, mode=mode)
        try:
            get_notification_tracker().record_attempt("telegram")
            _telegram_notifier.send_message(msg)
            get_notification_tracker().record_success("telegram")
            logger.info("[startup_alert] Telegram notification sent")
        except Exception as e:
            get_notification_tracker().record_error("telegram", str(e))
            logger.warning("[startup_alert] Failed to send Telegram: %s", e)
    
    # Email
    if _email_notifier is not None:
        subject, html_body = EmailNotifier.format_startup_message(symbol=symbol, mode=mode)
        try:
            import threading
            _em = _email_notifier  # capture for closure — Pyright can't narrow module globals
            get_notification_tracker().record_attempt("email")
            def _send_startup_email(_n: EmailNotifier = _em) -> None:
                ok = _n.send_message(subject, html_body)
                if ok:
                    get_notification_tracker().record_success("email")
                else:
                    get_notification_tracker().record_error("email", "send_message returned False")
            threading.Thread(target=_send_startup_email, daemon=True).start()
            logger.info("[startup_alert] Email notification sent")
        except Exception as e:
            get_notification_tracker().record_error("email", str(e))
            logger.warning("[startup_alert] Failed to send Email: %s", e)
    
    # macOS
    if _macos_notifier is not None:
        title, message, subtitle = MacOSNotifier.format_startup_message(symbol=symbol, mode=mode)
        try:
            get_notification_tracker().record_attempt("macos")
            _macos_notifier.notify(title, message, subtitle)
            get_notification_tracker().record_success("macos")
            logger.info("[startup_alert] macOS notification sent")
        except Exception as e:
            get_notification_tracker().record_error("macos", str(e))
            logger.warning("[startup_alert] Failed to send macOS: %s", e)


def _send_shutdown_alert(
    symbol: str = "ETH",
    reason: str = "normal",
    error_msg: Optional[str] = None,
) -> None:
    """Send shutdown notification (once only) to WhatsApp, Telegram, Email, and macOS."""
    global _shutdown_msg_sent

    events_cfg = _cfg_raw.get("alerts", {}).get("events", {})
    if not events_cfg.get("shutdown", True):
        return
    
    if _shutdown_msg_sent:
        return
    
    _shutdown_msg_sent = True
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # WhatsApp (simple text via send_alert)
    reason_emoji = {"normal": "✅", "error": "🚨", "signal": "🔔"}.get(reason, "⚪")
    whatsapp_msg = f"{reason_emoji} {symbol} Analyzer Stopped\n\nReason: {reason}\nTime: {timestamp}"
    if error_msg:
        whatsapp_msg += f"\nError: {error_msg}"
    send_alert(whatsapp_msg, _cfg_raw, "shutdown")
    
    # Telegram
    if _telegram_notifier is not None:
        msg = TelegramNotifier.format_shutdown_message(
            symbol=symbol, reason=reason, error_msg=error_msg
        )
        try:
            get_notification_tracker().record_attempt("telegram")
            _telegram_notifier.send_message(msg)
            get_notification_tracker().record_success("telegram")
            logger.info("[shutdown_alert] Telegram notification sent (reason=%s)", reason)
        except Exception as e:
            get_notification_tracker().record_error("telegram", str(e))
            logger.warning("[shutdown_alert] Failed to send Telegram: %s", e)
    
    # Email
    if _email_notifier is not None:
        subject, html_body = EmailNotifier.format_shutdown_message(
            symbol=symbol, reason=reason, error_msg=error_msg
        )
        try:
            import threading
            _em = _email_notifier  # capture for closure — Pyright can't narrow module globals
            get_notification_tracker().record_attempt("email")
            def _send_shutdown_email(_n: EmailNotifier = _em) -> None:
                ok = _n.send_message(subject, html_body)
                if ok:
                    get_notification_tracker().record_success("email")
                else:
                    get_notification_tracker().record_error("email", "send_message returned False")
            threading.Thread(target=_send_shutdown_email, daemon=True).start()
            logger.info("[shutdown_alert] Email notification sent (reason=%s)", reason)
        except Exception as e:
            get_notification_tracker().record_error("email", str(e))
            logger.warning("[shutdown_alert] Failed to send Email: %s", e)
    
    # macOS (use sync to ensure it completes before exit)
    if _macos_notifier is not None:
        title, message, subtitle = MacOSNotifier.format_shutdown_message(
            symbol=symbol, reason=reason, error_msg=error_msg
        )
        try:
            get_notification_tracker().record_attempt("macos")
            _macos_notifier.notify_sync(title, message, subtitle)
            get_notification_tracker().record_success("macos")
            logger.info("[shutdown_alert] macOS notification sent (reason=%s)", reason)
        except Exception as e:
            get_notification_tracker().record_error("macos", str(e))
            logger.warning("[shutdown_alert] Failed to send macOS: %s", e)


def _register_signal_handlers(symbol: str = "ETH") -> None:
    """Register SIGINT, SIGTERM (and SIGHUP on Unix) for graceful shutdown.

    Strategy:
    - SIGINT  → raise KeyboardInterrupt  (caught by main try/except)
    - SIGTERM → raise SystemExit(128+15)  (caught by main try/except)
    - SIGHUP  → same as SIGTERM (daemon restart / terminal close)
    - atexit  → last-resort flush in case Python skips the except handlers
    """
    def _on_sigint(signum, frame):
        # Ignore any further Ctrl+C so a second press can't interrupt cleanup
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logger.warning("[signal] SIGINT received — initiating graceful shutdown")
        _send_shutdown_alert(symbol=symbol, reason="SIGINT")
        raise KeyboardInterrupt()

    def _on_sigterm(signum, frame):
        # Restore default so a second SIGTERM falls through without re-entering
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        logger.warning("[signal] SIGTERM received — initiating graceful shutdown")
        _send_shutdown_alert(symbol=symbol, reason="SIGTERM")
        raise SystemExit(128 + 15)  # conventional SIGTERM exit code

    def _on_exit():
        # atexit fires even when sys.exit() is called; _send_shutdown_alert
        # is idempotent (flag-guarded) so duplicate calls are safe.
        _send_shutdown_alert(symbol=symbol, reason="normal")

    import atexit
    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigterm)
    # SIGHUP only exists on Unix; skip on Windows
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _on_sigterm)
    atexit.register(_on_exit)


# ── Logging setup ──────────────────────────────────────────────────

def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """Configure both console and file logging with daily rotation."""
    from pathlib import Path as _LogPath
    import logging.handlers
    
    _LogPath(log_dir).mkdir(exist_ok=True)
    log_file = _LogPath(log_dir) / f"analyzer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()  # Clear any existing handlers
    
    # Console handler (colorful, concise)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S"
    ))
    root.addHandler(console)
    
    # File handler (detailed, with module names)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=50 * 1024 * 1024,  # 50MB per file
        backupCount=14,              # Keep 2 weeks
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-8s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    root.addHandler(file_handler)
    
    logger.info("Logging initialized: console=%s file=%s", logging.getLevelName(console.level), log_file)


# ── main ────────────────────────────────────────────────────────

def main():
    setup_logging(log_dir="logs", level=logging.DEBUG)

    args = parse_args()
    cfg = load_config(args.config)

    # Expose config to module-level alert helpers
    global _cfg_raw, _telegram_notifier, _email_notifier, _macos_notifier
    _cfg_raw = cfg.raw

    # Initialize Telegram notifier (if configured)
    _telegram_notifier = get_telegram_notifier(cfg.raw)
    set_telegram_notifier(_telegram_notifier)  # Register in utils.py
    
    # Initialize Email notifier (if configured)
    _email_notifier = get_email_notifier(cfg.raw)
    set_email_notifier(_email_notifier)  # Register in utils.py
    
    # Initialize macOS notifier (if configured)
    _macos_notifier = get_macos_notifier(cfg.raw)
    set_macos_notifier(_macos_notifier)  # Register in utils.py
    
    # Get symbol for alerts
    epic_symbol = cfg.raw.get("strategy", {}).get("symbol", "ETH")

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
        alerts_cfg_for_dash = cfg.raw.get("alerts", {})
        _notifiers_for_dash = {
            "telegram": _telegram_notifier,
            "email": _email_notifier,
            "macos": _macos_notifier,
            "wa_url": alerts_cfg_for_dash.get("whatsapp_notifier_url", "http://127.0.0.1:3099/send"),
            "wa_enabled": alerts_cfg_for_dash.get("enabled", False),
        }
        dash = DashboardServer(
            engine=engine,
            schema=out_cfg.schema,
            host=str(dash_raw.get("host","127.0.0.1")),
            port=int(dash_raw.get("port",8787)),
            open_browser=bool(dash_raw.get("open_browser", True)),
            notification_tracker=get_notification_tracker(),
            notifiers=_notifiers_for_dash,
            cfg_raw=cfg.raw,
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
        "meta": {},          # meta-model trainer state per-TF (last_trained_count, last_trained_ts)
    }
    if strategy_cfg.get("enabled"):
        print(f"[strategy] Enabled for timeframes: {sorted(strategy_tfs)}")

    # Register signal handlers for graceful shutdown
    _register_signal_handlers(symbol=epic_symbol)

    # Auto-start WhatsApp notifier sidecar
    alerts_cfg = cfg.raw.get("alerts", {})
    notifier_proc: Optional[subprocess.Popen] = _start_notifier(alerts_cfg)

    # Send startup notification via Telegram
    data_source = cfg.raw.get("data_source", "db_table")
    mode_label = "Capital.com API" if data_source == "capital_api" else "DB table"
    _send_startup_alert(symbol=epic_symbol, mode=mode_label)

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
    except KeyboardInterrupt:
        logger.warning("[main] Interrupted by user (KeyboardInterrupt)")
        _send_shutdown_alert(symbol=epic_symbol, reason="SIGINT")
    except SystemExit:
        logger.warning("[main] SystemExit triggered")
        _send_shutdown_alert(symbol=epic_symbol, reason="SIGTERM")
        raise
    except Exception as _fatal:
        logger.critical("[main] Fatal exception: %s", _fatal, exc_info=True)
        _send_shutdown_alert(
            symbol=epic_symbol,
            reason="exception",
            error_msg=str(_fatal)[:200],
        )
        send_alert(
            f"\U0001f4a5 [Analyzer] Fatal crash: {_fatal}",
            _cfg_raw, "system_error"
        )
        raise
    finally:
        logger.info("[main] Cleanup: shutting down services...")
        send_alert(f"\U0001f534 [Analyzer] Stopped.", _cfg_raw, "system_error")

        # ── 1. Stop WhatsApp notifier sidecar ────────────────────────────
        if notifier_proc is not None and notifier_proc.poll() is None:
            notifier_proc.terminate()
            try:
                notifier_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("[main] Notifier did not exit in 5 s — SIGKILL")
                notifier_proc.kill()
                notifier_proc.wait()
            print("[notifier] Sidecar stopped.")

        # ── 2. Stop dashboard HTTP server ────────────────────────────────
        if dash is not None:
            try:
                dash.stop()
                logger.info("[main] Dashboard stopped.")
            except Exception as e:
                logger.warning("[main] Failed to stop dashboard: %s", e)

        # ── 3. Dispose SQLAlchemy engine (closes all pooled connections) ─
        try:
            engine.dispose()
            logger.info("[main] Database engine disposed — all connections closed.")
        except Exception as e:
            logger.warning("[main] engine.dispose() failed: %s", e)

        # ── 4. Final GC + log flush ───────────────────────────────────────
        gc.collect()
        logger.info("[main] Shutdown complete.")
        # Flush all log handlers and close them cleanly.
        logging.shutdown()

if __name__ == "__main__":
    main()
