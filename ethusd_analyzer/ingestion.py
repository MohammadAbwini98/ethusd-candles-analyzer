"""
Real OHLCV candle ingestion from Capital.com /api/v1/prices.

Replaces the synthetic tick-based CandleBuilder approach with:
  - Real 1m OHLCV candles from Capital.com /api/v1/prices/{epic}
  - Sentiment attached *as-of* each candle's close time (no smearing)
  - Higher-TF (5m/15m/30m/1H) resampling stored in DB
  - sentiment_ticks persistence for accurate historical as-of lookups
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# ── Resample configuration ─────────────────────────────────────────────────

# Timeframe label → bar duration in seconds
_TF_SECONDS: Dict[str, int] = {
    "5m":  300,
    "15m": 900,
    "30m": 1800,
    "1H":  3600,
}

# Timeframe label → pandas resample rule
_TF_RULE: Dict[str, str] = {
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1H":  "1h",
}


# ── Internal helpers ───────────────────────────────────────────────────────

def _floor_dt(dt: datetime, seconds: int) -> datetime:
    """Floor a UTC datetime to the nearest multiple of *seconds*."""
    epoch = int(dt.timestamp())
    return datetime.fromtimestamp((epoch // seconds) * seconds, tz=timezone.utc)


def _upsert_candles(engine: Engine, schema: str, rows: List[Dict]) -> int:
    """Upsert candle rows into {schema}.candles (includes sentiment_ts)."""
    if not rows:
        return 0
    # Ensure every row has sentiment_ts key (NULL if absent)
    for row in rows:
        row.setdefault("sentiment_ts", None)
    sql = text(f"""
        INSERT INTO {schema}.candles
            (ts, epic, timeframe, open, high, low, close, vol,
             buyers_pct, sellers_pct, sentiment_ts)
        VALUES
            (:ts, :epic, :timeframe, :open, :high, :low, :close, :vol,
             :buyers_pct, :sellers_pct, :sentiment_ts)
        ON CONFLICT (ts, epic, timeframe) DO UPDATE SET
            open         = EXCLUDED.open,
            high         = EXCLUDED.high,
            low          = EXCLUDED.low,
            close        = EXCLUDED.close,
            vol          = EXCLUDED.vol,
            buyers_pct   = EXCLUDED.buyers_pct,
            sellers_pct  = EXCLUDED.sellers_pct,
            sentiment_ts = EXCLUDED.sentiment_ts
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


# ── DB helpers ─────────────────────────────────────────────────────────────

def get_last_1m_ts(engine: Engine, schema: str, epic: str) -> Optional[datetime]:
    """Return the latest stored 1m candle timestamp for *epic*, or None."""
    with engine.connect() as conn:
        row = conn.execute(
            text(f"""
                SELECT MAX(ts)
                FROM   {schema}.candles
                WHERE  timeframe = '1m' AND epic = :epic
            """),
            {"epic": epic},
        ).fetchone()
    ts = row[0] if row else None
    if ts is None:
        return None
    ts_pd = pd.Timestamp(ts)
    if ts_pd.tzinfo is None:
        ts_pd = ts_pd.tz_localize("UTC")
    else:
        ts_pd = ts_pd.tz_convert("UTC")
    return ts_pd.to_pydatetime()


def insert_sentiment_tick(
    engine: Engine,
    schema: str,
    epic: str,
    ts: datetime,
    buyers_pct: float,
    sellers_pct: float,
) -> None:
    """Persist one sentiment observation into {schema}.sentiment_ticks."""
    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {schema}.sentiment_ticks (ts, epic, buyers_pct, sellers_pct)
                VALUES (:ts, :epic, :buyers_pct, :sellers_pct)
                ON CONFLICT (ts, epic) DO UPDATE SET
                    buyers_pct  = EXCLUDED.buyers_pct,
                    sellers_pct = EXCLUDED.sellers_pct
            """),
            {"ts": ts, "epic": epic,
             "buyers_pct": buyers_pct, "sellers_pct": sellers_pct},
        )


def sentiment_as_of(
    candle_end_ts: datetime,
    engine: Engine,
    schema: str,
    epic: str,
    memory_cache: Dict[str, Any],
) -> Tuple[float, float, Optional[datetime]]:
    """
    Return (buyers_pct, sellers_pct, sentiment_ts) valid as of *candle_end_ts*.

    Priority:
      1. In-memory cache — if its timestamp is <= candle_end_ts
      2. Latest DB row in sentiment_ticks with ts <= candle_end_ts
      3. Neutral fallback: (50.0, 50.0, None)
    """
    cache_ts: Optional[datetime] = memory_cache.get("ts")
    if cache_ts is not None and cache_ts <= candle_end_ts:
        return (
            float(memory_cache["buyers_pct"]),
            float(memory_cache["sellers_pct"]),
            cache_ts,
        )

    # Fall back to DB lookup
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(f"""
                    SELECT ts, buyers_pct, sellers_pct
                    FROM   {schema}.sentiment_ticks
                    WHERE  epic = :epic AND ts <= :cts
                    ORDER  BY ts DESC
                    LIMIT  1
                """),
                {"epic": epic, "cts": candle_end_ts},
            ).fetchone()
        if row:
            s_ts = row[0]
            if hasattr(s_ts, "to_pydatetime"):
                s_ts = s_ts.to_pydatetime()
            if s_ts.tzinfo is None:
                s_ts = s_ts.replace(tzinfo=timezone.utc)
            return float(row[1]), float(row[2]), s_ts
    except Exception as exc:
        logger.warning("[ingestion] sentiment_as_of DB query failed: %s", exc)

    return 50.0, 50.0, None


# ── 1m candle ingestion ─────────────────────────────────────────────────────

def fetch_and_upsert_1m(
    session: Any,                              # CapitalSession (Any avoids circular import)
    engine: Engine,
    schema: str,
    epic: str,
    sentiment_cache: Dict[str, Any],           # keys: ts, buyers_pct, sellers_pct
    overlap_minutes: int = 5,
    zero_vol_state: Optional[Dict[str, int]] = None,
    zero_vol_warn_threshold: int = 10,
) -> List[datetime]:
    """
    Fetch closed 1m candles from Capital.com (via /api/v1/prices) and upsert
    into {schema}.candles with as-of sentiment.

    *sentiment_cache* is updated externally by the live loop whenever
    get_sentiment() is polled. It contains:
        ``ts``          – datetime of last sentiment poll (or None)
        ``buyers_pct``  – float
        ``sellers_pct`` – float

    Returns a sorted list of newly upserted candle timestamps.
    An empty list means no new closed candles were available yet.
    """
    now_utc = datetime.now(timezone.utc)
    # A candle is "closed" only if its start timestamp is < the current minute
    current_minute_start = _floor_dt(now_utc, 60)

    last_ts = get_last_1m_ts(engine, schema, epic)
    if last_ts is None:
        logger.info(
            "[ingest_1m] No 1m candles in DB yet — backfill must run first"
        )
        return []

    start = last_ts - timedelta(minutes=overlap_minutes)
    end = now_utc

    try:
        raw = session.get_prices_range(epic, "MINUTE", start, end)
    except Exception as exc:
        logger.error("[ingest_1m] get_prices_range failed: %s", exc)
        return []

    # Keep only closed candles that are genuinely newer than what is in DB
    closed   = [c for c in raw if c["time"] < current_minute_start]
    truly_new = [c for c in closed if c["time"] > last_ts]

    if not truly_new:
        logger.debug(
            "[ingest_1m] No new closed 1m candles (last_ts=%s)",
            last_ts.isoformat(),
        )
        return []

    rows: List[Dict] = []
    for c in truly_new:
        buyers, sellers, s_ts = sentiment_as_of(
            c["time"], engine, schema, epic, sentiment_cache
        )
        rows.append({
            "ts":          c["time"],
            "epic":        epic,
            "timeframe":   "1m",
            "open":        c["open"],
            "high":        c["high"],
            "low":         c["low"],
            "close":       c["close"],
            "vol":         float(c["volume"]),
            "buyers_pct":  buyers,
            "sellers_pct": sellers,
            "sentiment_ts": s_ts,
        })

    _upsert_candles(engine, schema, rows)

    vols      = [r["vol"] for r in rows]
    newest_ts = max(r["ts"] for r in rows)
    logger.info(
        "[ingest_1m] +%d candles | newest=%s | vol min=%.4f max=%.4f mean=%.4f",
        len(rows), newest_ts.isoformat(),
        min(vols), max(vols), sum(vols) / len(vols),
    )

    # ── Consecutive zero-vol warning ────────────────────────────
    if zero_vol_state is not None:
        if all(v == 0.0 for v in vols):
            zero_vol_state["consecutive"] = (
                zero_vol_state.get("consecutive", 0) + len(rows)
            )
            if zero_vol_state["consecutive"] >= zero_vol_warn_threshold:
                logger.warning(
                    "[ingest_1m] WARN: %d consecutive 1m candles have vol=0 — "
                    "Capital.com API may not be returning volume data.",
                    zero_vol_state["consecutive"],
                )
        else:
            zero_vol_state["consecutive"] = 0

    return sorted(r["ts"] for r in rows)


# ── Resampler ──────────────────────────────────────────────────────────────

def _load_1m_window(
    engine: Engine,
    schema: str,
    epic: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load 1m candles from DB for the given half-open window [start, end)."""
    with engine.connect() as conn:
        df = pd.read_sql(
            text(f"""
                SELECT ts, open, high, low, close, vol,
                       buyers_pct, sellers_pct, sentiment_ts
                FROM   {schema}.candles
                WHERE  epic      = :epic
                  AND  timeframe = '1m'
                  AND  ts       >= :start
                  AND  ts        < :end
                ORDER  BY ts
            """),
            conn,
            params={"epic": epic, "start": start, "end": end},
        )
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        if "sentiment_ts" in df.columns:
            df["sentiment_ts"] = pd.to_datetime(
                df["sentiment_ts"], utc=True, errors="coerce"
            )
    return df


def resample_and_upsert(
    engine: Engine,
    schema: str,
    epic: str,
    affected_ts: List[datetime],
    target_timeframes: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Resample stored 1m candles into higher-TF candles for the window
    affected by *affected_ts* and upsert into {schema}.candles.

    Only fully-closed bars are upserted (bars whose end time < now).
    Sentiment is volume-weighted across constituent 1m bars; falls back
    to a simple mean when total vol = 0.

    Returns ``{timeframe: count_upserted}``.
    """
    if not affected_ts:
        return {}

    target_timeframes = target_timeframes or list(_TF_SECONDS.keys())
    results: Dict[str, int] = {}
    now_utc = datetime.now(timezone.utc)

    min_ts = min(affected_ts)
    max_ts = max(affected_ts)

    for tf in target_timeframes:
        tf_secs = _TF_SECONDS[tf]
        tf_rule = _TF_RULE[tf]
        tf_td   = timedelta(seconds=tf_secs)

        # Extend window by one bar on each side to ensure edge bars are complete
        window_start = _floor_dt(min_ts, tf_secs) - tf_td
        window_end   = _floor_dt(max_ts, tf_secs) + 2 * tf_td

        df1m = _load_1m_window(engine, schema, epic, window_start, window_end)
        if df1m.empty:
            results[tf] = 0
            continue

        df1m = df1m.set_index("ts").sort_index()

        # Standard OHLCV aggregation
        ohlcv = df1m.resample(tf_rule, label="left", closed="left").agg(
            open =("open",  "first"),
            high =("high",  "max"),
            low  =("low",   "min"),
            close=("close", "last"),
            vol  =("vol",   "sum"),
        ).dropna(subset=["open"])

        if ohlcv.empty:
            results[tf] = 0
            continue

        rows: List[Dict] = []
        for _bar_ts_raw, orow in ohlcv.iterrows():
            bar_ts = cast(pd.Timestamp, _bar_ts_raw)
            bar_end = bar_ts + tf_td
            if bar_end > now_utc:
                continue  # still forming — skip

            # Slice constituent 1m rows for this bar
            grp = df1m.loc[
                bar_ts : bar_ts + tf_td - timedelta(seconds=1), :
            ]
            if grp.empty:
                continue

            # Volume-weighted sentiment (fall back to simple mean)
            total_vol = float(grp["vol"].sum())
            if total_vol > 0:
                buyers  = float(
                    (grp["buyers_pct"]  * grp["vol"]).sum() / total_vol
                )
                sellers = float(
                    (grp["sellers_pct"] * grp["vol"]).sum() / total_vol
                )
            else:
                buyers  = float(grp["buyers_pct"].mean())
                sellers = float(grp["sellers_pct"].mean())

            # Latest non-null sentiment_ts of constituent bars
            s_ts: Optional[Any] = None
            if "sentiment_ts" in grp.columns:
                s_ts_series = grp["sentiment_ts"].dropna()
                if not s_ts_series.empty:
                    s_ts = s_ts_series.max()
                    if pd.isna(s_ts):
                        s_ts = None
                    elif hasattr(s_ts, "to_pydatetime"):
                        s_ts = s_ts.to_pydatetime()

            rows.append({
                "ts":          bar_ts.to_pydatetime(),
                "epic":        epic,
                "timeframe":   tf,
                "open":        float(orow["open"]),
                "high":        float(orow["high"]),
                "low":         float(orow["low"]),
                "close":       float(orow["close"]),
                "vol":         float(orow["vol"]),
                "buyers_pct":  buyers  if pd.notna(buyers)  else 50.0,
                "sellers_pct": sellers if pd.notna(sellers) else 50.0,
                "sentiment_ts": s_ts,
            })

        n = _upsert_candles(engine, schema, rows)
        results[tf] = n
        if n > 0:
            logger.info("[resample] tf=%-4s +%d bars upserted", tf, n)

    return results
