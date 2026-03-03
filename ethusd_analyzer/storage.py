from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, TypeVar, cast

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

_T = TypeVar("_T")


@dataclass
class OutputConfig:
    enabled: bool = True
    schema: str = "ethusd_analytics"
    store_rolling_points: int = 500
    retain_days: int = 14
    # FR-29: optional performance tables
    enable_perf_tables: bool = False


def _with_retry(fn: Callable[[], _T], retries: int = 3, base_delay: float = 1.0) -> _T:
    """NFR-01: Execute *fn* with exponential-backoff retry on transient DB errors."""
    last_exc: Exception
    for attempt in range(retries):
        try:
            return fn()
        except OperationalError as exc:
            last_exc = exc
            if attempt < retries - 1:
                sleep = base_delay * (2 ** attempt)
                time.sleep(sleep)
    raise last_exc  # type: ignore[possibly-undefined]

def init_schema_and_tables(engine: Engine, schema: str) -> None:
    ddl_statements = [
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"""CREATE TABLE IF NOT EXISTS {schema}.snapshots (
              id BIGSERIAL PRIMARY KEY,
              timeframe TEXT NOT NULL,
              computed_at TIMESTAMPTZ NOT NULL,
              start_time TIMESTAMPTZ,
              end_time TIMESTAMPTZ,
              rows INTEGER,
              latest_close DOUBLE PRECISION,
              latest_score DOUBLE PRECISION,
              buyers_plus_sellers_mean_abs_diff DOUBLE PRECISION
            )""",
        f"CREATE INDEX IF NOT EXISTS snapshots_tf_time_idx ON {schema}.snapshots(timeframe, computed_at DESC)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.corr_results (
              id BIGSERIAL PRIMARY KEY,
              timeframe TEXT NOT NULL,
              computed_at TIMESTAMPTZ NOT NULL,
              horizon INTEGER NOT NULL,
              feature TEXT NOT NULL,
              n INTEGER,
              pearson_r DOUBLE PRECISION,
              pearson_p DOUBLE PRECISION,
              spearman_r DOUBLE PRECISION,
              spearman_p DOUBLE PRECISION
            )""",
        f"CREATE INDEX IF NOT EXISTS corr_tf_time_idx ON {schema}.corr_results(timeframe, computed_at DESC)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.lagcorr_results (
              id BIGSERIAL PRIMARY KEY,
              timeframe TEXT NOT NULL,
              computed_at TIMESTAMPTZ NOT NULL,
              horizon INTEGER NOT NULL,
              feature TEXT NOT NULL,
              lag INTEGER NOT NULL,
              n INTEGER,
              pearson_r DOUBLE PRECISION,
              pearson_p DOUBLE PRECISION
            )""",
        f"CREATE INDEX IF NOT EXISTS lagcorr_tf_time_idx ON {schema}.lagcorr_results(timeframe, computed_at DESC)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.rollingcorr_points (
              timeframe TEXT NOT NULL,
              horizon INTEGER NOT NULL,
              feature TEXT NOT NULL,
              "window" INTEGER NOT NULL,
              market_time TIMESTAMPTZ NOT NULL,
              value DOUBLE PRECISION,
              computed_at TIMESTAMPTZ NOT NULL,
              PRIMARY KEY (timeframe, horizon, feature, "window", market_time)
            )""",
        f"CREATE INDEX IF NOT EXISTS rolling_tf_time_idx ON {schema}.rollingcorr_points(timeframe, market_time DESC)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.candles (
              ts TIMESTAMPTZ NOT NULL,
              epic TEXT NOT NULL,
              timeframe TEXT NOT NULL,
              open DOUBLE PRECISION,
              high DOUBLE PRECISION,
              low DOUBLE PRECISION,
              close DOUBLE PRECISION NOT NULL,
              vol DOUBLE PRECISION DEFAULT 0,
              buyers_pct DOUBLE PRECISION DEFAULT 50,
              sellers_pct DOUBLE PRECISION DEFAULT 50,
              PRIMARY KEY (ts, epic, timeframe)
            )""",
        f"CREATE INDEX IF NOT EXISTS candles_tf_ts_idx ON {schema}.candles(timeframe, ts DESC)",
        f"CREATE INDEX IF NOT EXISTS candles_epic_tf_idx ON {schema}.candles(epic, timeframe, ts DESC)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.signal_recommendations (
              id BIGSERIAL PRIMARY KEY,
              timeframe TEXT NOT NULL,
              computed_at TIMESTAMPTZ NOT NULL,
              regime TEXT NOT NULL,
              signal TEXT NOT NULL,
              confidence DOUBLE PRECISION,
              entry_price DOUBLE PRECISION,
              stop_loss DOUBLE PRECISION,
              take_profit DOUBLE PRECISION,
              hold_bars INTEGER,
              reason TEXT,
              conf_regime DOUBLE PRECISION,
              conf_tail DOUBLE PRECISION,
              conf_backtest DOUBLE PRECISION,
              rc DOUBLE PRECISION,
              ar DOUBLE PRECISION,
              score_mr DOUBLE PRECISION,
              score_mom DOUBLE PRECISION,
              volatility DOUBLE PRECISION,
              params_json JSONB
            )""",
        f"CREATE INDEX IF NOT EXISTS signal_rec_tf_time_idx ON {schema}.signal_recommendations(timeframe, computed_at DESC)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.calibration_runs (
              id BIGSERIAL PRIMARY KEY,
              timeframe TEXT NOT NULL,
              computed_at TIMESTAMPTZ NOT NULL,
              best_params JSONB,
              net_sharpe DOUBLE PRECISION,
              net_return DOUBLE PRECISION,
              max_drawdown DOUBLE PRECISION,
              n_trades INTEGER,
              win_rate DOUBLE PRECISION,
              param_grid_size INTEGER,
              lookback_days INTEGER
            )""",
        f"CREATE INDEX IF NOT EXISTS calibration_tf_time_idx ON {schema}.calibration_runs(timeframe, computed_at DESC)",
        # FR-29: optional performance tracking tables
        f"""CREATE TABLE IF NOT EXISTS {schema}.strategy_runs (
              id BIGSERIAL PRIMARY KEY,
              timeframe TEXT NOT NULL,
              computed_at TIMESTAMPTZ NOT NULL,
              run_mode TEXT,
              net_sharpe DOUBLE PRECISION,
              net_return DOUBLE PRECISION,
              max_drawdown DOUBLE PRECISION,
              n_trades INTEGER,
              win_rate DOUBLE PRECISION,
              params JSONB
            )""",
        f"CREATE INDEX IF NOT EXISTS strategy_runs_tf_idx ON {schema}.strategy_runs(timeframe, computed_at DESC)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.strategy_trades (
              id BIGSERIAL PRIMARY KEY,
              run_id BIGINT REFERENCES {schema}.strategy_runs(id),
              timeframe TEXT NOT NULL,
              entered_at TIMESTAMPTZ NOT NULL,
              exited_at TIMESTAMPTZ,
              regime TEXT,
              signal TEXT,
              entry_price DOUBLE PRECISION,
              exit_price DOUBLE PRECISION,
              pnl DOUBLE PRECISION,
              hold_bars INTEGER
            )""",
        f"CREATE INDEX IF NOT EXISTS strategy_trades_run_idx ON {schema}.strategy_trades(run_id)",
        f"""CREATE TABLE IF NOT EXISTS {schema}.strategy_equity (
              run_id BIGINT NOT NULL REFERENCES {schema}.strategy_runs(id),
              bar_index INTEGER NOT NULL,
              equity DOUBLE PRECISION,
              PRIMARY KEY (run_id, bar_index)
            )""",
    ]
    # Migrations: add columns that may be missing on existing tables
    migrations = [
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.signal_recommendations ADD COLUMN params_json JSONB;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN min_trades INTEGER;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN eligible_candidates INTEGER;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN total_candidates INTEGER;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN status TEXT;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN rejection_reason TEXT;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        # FR-28: rejection breakdown columns
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN max_dd_used DOUBLE PRECISION;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN rejected_by_min_trades INTEGER;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN rejected_by_max_dd INTEGER;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN rejected_by_both INTEGER;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN max_trades_seen INTEGER;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
        f"""DO $$ BEGIN
            ALTER TABLE {schema}.calibration_runs ADD COLUMN best_dd_seen DOUBLE PRECISION;
        EXCEPTION WHEN duplicate_column THEN NULL; END $$""",
    ]

    def _init():
        with engine.begin() as conn:
            for stmt in ddl_statements:
                conn.execute(text(stmt))
            for stmt in migrations:
                conn.execute(text(stmt))

    _with_retry(_init)

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

def save_snapshot(
    engine: Engine,
    schema: str,
    timeframe: str,
    summary: dict,
    corr_df: pd.DataFrame,
    lag_df: pd.DataFrame,
    roll_df: pd.DataFrame,
    output_cfg: OutputConfig,
) -> None:
    if not output_cfg.enabled:
        return
    computed_at = _utcnow()

    with engine.begin() as conn:
        conn.execute(
            text(f"""
              INSERT INTO {schema}.snapshots
                (timeframe, computed_at, start_time, end_time, rows, latest_close, latest_score, buyers_plus_sellers_mean_abs_diff)
              VALUES
                (:tf, :computed_at, :start_time, :end_time, :rows, :latest_close, :latest_score, :bps_diff)
            """),
            {
                "tf": timeframe,
                "computed_at": computed_at,
                "start_time": summary.get("start"),
                "end_time": summary.get("end"),
                "rows": summary.get("rows"),
                "latest_close": summary.get("latest_close"),
                "latest_score": summary.get("latest_score"),
                "bps_diff": summary.get("buyers_plus_sellers_mean_abs_diff"),
            },
        )

        if corr_df is not None and not corr_df.empty:
            recs = cast(list[dict[str, Any]], corr_df.to_dict(orient="records"))
            params: list[dict[str, Any]] = [
                {"tf": timeframe, "computed_at": computed_at, **r} for r in recs
            ]
            conn.execute(
                text(f"""
                INSERT INTO {schema}.corr_results
                  (timeframe, computed_at, horizon, feature, n, pearson_r, pearson_p, spearman_r, spearman_p)
                VALUES
                  (:tf, :computed_at, :horizon, :feature, :n, :pearson_r, :pearson_p, :spearman_r, :spearman_p)
                """),
                params,
            )

        if lag_df is not None and not lag_df.empty:
            recs = cast(list[dict[str, Any]], lag_df.to_dict(orient="records"))
            lag_params: list[dict[str, Any]] = [
                {"tf": timeframe, "computed_at": computed_at, "horizon": 1, "feature": "score", **r} for r in recs
            ]
            conn.execute(
                text(f"""
                INSERT INTO {schema}.lagcorr_results
                  (timeframe, computed_at, horizon, feature, lag, n, pearson_r, pearson_p)
                VALUES
                  (:tf, :computed_at, :horizon, :feature, :lag, :n, :pearson_r, :pearson_p)
                """),
                lag_params,
            )

        if roll_df is not None and not roll_df.empty and output_cfg.store_rolling_points > 0:
            df = roll_df.copy()
            df["market_time"] = pd.to_datetime(df["market_time"], utc=True, errors="coerce")
            df = df.dropna(subset=["market_time"]).sort_values("market_time")
            df = df.tail(output_cfg.store_rolling_points)

            roll_cols = [c for c in df.columns if c.startswith("rollcorr_")]
            for col in roll_cols:
                try:
                    parts = col.split("_")
                    feature = parts[1]
                    horizon = int(parts[2].lstrip("h"))
                    window = int(parts[3].lstrip("w"))
                except Exception:
                    continue

                points = []
                mt_series = df["market_time"]
                val_series = df[col]
                for idx in range(len(df)):
                    v_val = val_series.iloc[idx]
                    if pd.isna(v_val):  # type: ignore[arg-type]
                        continue
                    points.append({
                        "tf": timeframe,
                        "h": horizon,
                        "feature": feature,
                        "w": window,
                        "mt": pd.Timestamp(mt_series.iloc[idx]).to_pydatetime(),
                        "value": float(v_val),
                        "computed_at": computed_at,
                    })
                if points:
                    conn.execute(  # type: ignore[call-overload]
                        text(f"""
                        INSERT INTO {schema}.rollingcorr_points
                          (timeframe, horizon, feature, "window", market_time, value, computed_at)
                        VALUES
                          (:tf, :h, :feature, :w, :mt, :value, :computed_at)
                        ON CONFLICT (timeframe, horizon, feature, "window", market_time)
                        DO UPDATE SET value = EXCLUDED.value, computed_at = EXCLUDED.computed_at
                        """),
                        points,
                    )

        if output_cfg.retain_days and output_cfg.retain_days > 0:
            cutoff = _utcnow() - timedelta(days=int(output_cfg.retain_days))
            conn.execute(  # type: ignore[call-overload]
                text(f"DELETE FROM {schema}.rollingcorr_points WHERE computed_at < :cutoff"),
                {"cutoff": cutoff},
            )
            conn.execute(  # type: ignore[call-overload]
                text(f"DELETE FROM {schema}.signal_recommendations WHERE computed_at < :cutoff"),
                {"cutoff": cutoff},
            )
            conn.execute(  # type: ignore[call-overload]
                text(f"DELETE FROM {schema}.calibration_runs WHERE computed_at < :cutoff"),
                {"cutoff": cutoff},
            )


def save_signal_recommendation(
    engine: Engine,
    schema: str,
    rec: Any,
    output_cfg: OutputConfig,
) -> None:
    if not output_cfg.enabled:
        return
    import json as _json2
    computed_at = _utcnow()

    def _do() -> None:
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.signal_recommendations
                        (timeframe, computed_at, regime, signal, confidence, entry_price,
                         stop_loss, take_profit, hold_bars, reason,
                         conf_regime, conf_tail, conf_backtest,
                         rc, ar, score_mr, score_mom, volatility, params_json)
                    VALUES
                        (:tf, :computed_at, :regime, :signal, :confidence, :entry_price,
                         :stop_loss, :take_profit, :hold_bars, :reason,
                         :conf_regime, :conf_tail, :conf_backtest,
                         :rc, :ar, :score_mr, :score_mom, :volatility, :params_json)
                """),
                {
                    "tf": rec.timeframe,
                    "computed_at": computed_at,
                    "regime": rec.regime,
                    "signal": rec.signal,
                    "confidence": rec.confidence,
                    "entry_price": rec.entry_price,
                    "stop_loss": rec.stop_loss,
                    "take_profit": rec.take_profit,
                    "hold_bars": rec.hold_bars,
                    "reason": rec.reason,
                    "conf_regime": rec.conf_regime,
                    "conf_tail": rec.conf_tail,
                    "conf_backtest": rec.conf_backtest,
                    "rc": rec.rc,
                    "ar": rec.ar,
                    "score_mr": rec.score_mr,
                    "score_mom": rec.score_mom,
                    "volatility": rec.volatility,
                    "params_json": _json2.dumps(rec.params_json) if rec.params_json else None,
                },
            )

    _with_retry(_do)  # NFR-01


def save_calibration_result(
    engine: Engine,
    schema: str,
    timeframe: str,
    result: Any,
    output_cfg: OutputConfig,
) -> None:
    if not output_cfg.enabled:
        return
    import json as _json
    computed_at = _utcnow()

    def _do() -> None:
        with engine.begin() as conn:
            conn.execute(
                text(f"""
                    INSERT INTO {schema}.calibration_runs
                        (timeframe, computed_at, best_params, net_sharpe, net_return,
                         max_drawdown, n_trades, win_rate, param_grid_size, lookback_days,
                         min_trades, eligible_candidates, total_candidates, status, rejection_reason,
                         max_dd_used, rejected_by_min_trades, rejected_by_max_dd,
                         rejected_by_both, max_trades_seen, best_dd_seen)
                    VALUES
                        (:tf, :computed_at, :best_params, :net_sharpe, :net_return,
                         :max_drawdown, :n_trades, :win_rate, :param_grid_size, :lookback_days,
                         :min_trades, :eligible_candidates, :total_candidates, :status, :rejection_reason,
                         :max_dd_used, :rejected_by_min_trades, :rejected_by_max_dd,
                         :rejected_by_both, :max_trades_seen, :best_dd_seen)
                """),
                {
                    "tf": timeframe,
                    "computed_at": computed_at,
                    "best_params": _json.dumps(result.best_params),
                    "net_sharpe": result.net_sharpe,
                    "net_return": result.net_return,
                    "max_drawdown": result.max_drawdown,
                    "n_trades": result.n_trades,
                    "win_rate": result.win_rate,
                    "param_grid_size": result.param_grid_size,
                    "lookback_days": result.lookback_days,
                    "min_trades": result.min_trades_used,
                    "eligible_candidates": result.eligible_candidates,
                    "total_candidates": result.total_candidates,
                    "status": result.status,
                    "rejection_reason": result.rejection_reason,
                    # FR-28: breakdown fields
                    "max_dd_used": getattr(result, "max_dd_used", None),
                    "rejected_by_min_trades": getattr(result, "rejected_by_min_trades", None),
                    "rejected_by_max_dd": getattr(result, "rejected_by_max_dd", None),
                    "rejected_by_both": getattr(result, "rejected_by_both", None),
                    "max_trades_seen": getattr(result, "max_trades_seen", None),
                    "best_dd_seen": getattr(result, "best_dd_seen", None),
                },
            )

    _with_retry(_do)  # NFR-01
