"""Outcome labeling for past signal_recommendations.

For each unlabeled recommendation older than (hold_bars + 1) * tf_minutes,
computes realized pnl from CLOSE-to-CLOSE log returns and labels WIN/LOSS.
The entry_price stored on the recommendation is used directly as the entry close;
the exit close is fetched from {schema}.candles at hold_bars ahead.

This module has NO side-effects on import and requires no ML dependencies.
"""
from __future__ import annotations

import json as _json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Minutes per candle timeframe label
_TF_MINUTES: Dict[str, int] = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}


def _tf_min(timeframe: str) -> int:
    return _TF_MINUTES.get(timeframe, 1)


def label_signals(
    engine: Engine,
    schema: str,
    timeframe: str,
    limit: int = 2000,
) -> int:
    """Fetch unlabeled recommendations for *timeframe*, compute pnl + WIN/LOSS,
    and write outcome/pnl/exit_price/exit_ts/label_computed_at back to DB.

    Returns the number of rows labeled in this call.
    """
    tf_minutes = _tf_min(timeframe)
    # Pre-filter: signal must be at least (2 * tf_min) old to have any exit data.
    # Per-row we check the exact hold_bars * tf_minutes requirement below.
    rough_cutoff = datetime.now(timezone.utc) - timedelta(minutes=2 * tf_minutes)

    with engine.connect() as conn:
        result = conn.execute(
            text(f"""
                SELECT id, computed_at, signal, hold_bars, entry_price, params_json
                FROM {schema}.signal_recommendations
                WHERE timeframe = :tf
                  AND outcome IS NULL
                  AND computed_at <= :cutoff
                ORDER BY computed_at ASC
                LIMIT :limit
            """),
            {"tf": timeframe, "cutoff": rough_cutoff, "limit": limit},
        )
        rows = result.fetchall()

    if not rows:
        return 0

    # Precise per-row filter: ensure (hold_bars + 1) bars have elapsed
    now_utc = datetime.now(timezone.utc)
    valid_rows = []
    for row in rows:
        hold_bars = int(row.hold_bars or 2)
        min_age = timedelta(minutes=(hold_bars + 1) * tf_minutes)
        entry_ts = row.computed_at
        if entry_ts.tzinfo is None:
            entry_ts = entry_ts.replace(tzinfo=timezone.utc)
        if now_utc >= entry_ts + min_age:
            valid_rows.append(row)

    if not valid_rows:
        return 0

    # ── Batch-fetch all candles needed ───────────────────────────────────────
    all_ts = [
        r.computed_at.replace(tzinfo=timezone.utc) if r.computed_at.tzinfo is None else r.computed_at
        for r in valid_rows
    ]
    min_ts = min(all_ts)
    max_hold = max(int(r.hold_bars or 2) for r in valid_rows)
    max_ts = max(all_ts) + timedelta(minutes=(max_hold + 2) * tf_minutes)

    with engine.connect() as conn:
        candle_result = conn.execute(
            text(f"""
                SELECT ts, close
                FROM {schema}.candles
                WHERE timeframe = :tf
                  AND ts >= :min_ts
                  AND ts <= :max_ts
                ORDER BY ts ASC
            """),
            {"tf": timeframe, "min_ts": min_ts, "max_ts": max_ts},
        )
        candle_rows = candle_result.fetchall()

    if not candle_rows:
        logger.debug("[labeler][%s] No candles found in range — skipping", timeframe)
        return 0

    candle_df = pd.DataFrame(candle_rows, columns=["ts", "close"])
    candle_df["ts"] = pd.to_datetime(candle_df["ts"], utc=True)
    candle_df = (
        candle_df.sort_values("ts")
        .drop_duplicates(subset="ts")
        .reset_index(drop=True)
    )
    tolerance = pd.Timedelta(minutes=2 * tf_minutes)

    # ── Label each row ────────────────────────────────────────────────────────
    updates: List[Dict[str, Any]] = []
    for row in valid_rows:
        try:
            signal_dir: str = str(row.signal)
            hold_bars  = int(row.hold_bars or 2)
            entry_price = float(row.entry_price)
            if entry_price <= 0:
                continue

            entry_ts = row.computed_at
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.replace(tzinfo=timezone.utc)
            entry_ts = pd.Timestamp(entry_ts)

            exit_target = entry_ts + pd.Timedelta(minutes=hold_bars * tf_minutes)

            # Find nearest candle to exit_target
            diffs = (candle_df["ts"] - exit_target).abs()
            nearest_idx = int(diffs.idxmin())
            if diffs.iloc[nearest_idx] > tolerance:
                continue  # exit candle not available yet

            exit_close = float(candle_df["close"].iloc[nearest_idx])
            exit_ts    = candle_df["ts"].iloc[nearest_idx]
            if exit_close <= 0:
                continue

            log_ret = math.log(exit_close / entry_price)

            # Extract cost_bps from params_json (fall back to 10 bps)
            cost_bps = 10
            if row.params_json:
                try:
                    params = (
                        _json.loads(row.params_json)
                        if isinstance(row.params_json, str)
                        else row.params_json
                    )
                    cost_bps = int(params.get("cost_bps", cost_bps))
                except Exception:
                    pass
            cost_frac = cost_bps / 10_000.0

            pnl = (log_ret if signal_dir == "BUY" else -log_ret) - 2 * cost_frac
            outcome = "WIN" if pnl > 0 else "LOSS"

            updates.append(
                {
                    "id":                row.id,
                    "outcome":           outcome,
                    "pnl":               round(pnl, 8),
                    "exit_price":        round(exit_close, 6),
                    "exit_ts":           exit_ts.to_pydatetime(),
                    "label_computed_at": datetime.now(timezone.utc),
                }
            )
        except Exception as exc:
            logger.warning("[labeler][%s] row id=%s failed: %s", timeframe, row.id, exc)

    if not updates:
        return 0

    with engine.begin() as conn:
        for upd in updates:
            conn.execute(
                text(f"""
                    UPDATE {schema}.signal_recommendations
                    SET outcome           = :outcome,
                        pnl               = :pnl,
                        exit_price        = :exit_price,
                        exit_ts           = :exit_ts,
                        label_computed_at = :label_computed_at
                    WHERE id = :id
                """),
                upd,
            )

    logger.info("[labeler][%s] Labeled %d new outcomes", timeframe, len(updates))
    return len(updates)


def get_label_stats(engine: Engine, schema: str, timeframe: str) -> Dict[str, Any]:
    """Return simple label statistics: total labeled, wins, losses, win_rate."""
    with engine.connect() as conn:
        row = conn.execute(
            text(f"""
                SELECT
                    COUNT(*)                                      AS total,
                    SUM(CASE WHEN outcome='WIN'  THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END) AS losses,
                    COUNT(CASE WHEN outcome IS NULL THEN 1 END)  AS unlabeled
                FROM {schema}.signal_recommendations
                WHERE timeframe = :tf
            """),
            {"tf": timeframe},
        ).fetchone()
    if row is None:
        return {}
    total, wins, losses, unlabeled = (
        int(row[0] or 0), int(row[1] or 0), int(row[2] or 0), int(row[3] or 0)
    )
    return {
        "timeframe": timeframe,
        "total":     total,
        "wins":      wins,
        "losses":    losses,
        "unlabeled": unlabeled,
        "win_rate":  round(wins / max(wins + losses, 1), 4),
    }
