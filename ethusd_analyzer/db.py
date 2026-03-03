from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

@dataclass
class DbConfig:
    host: str
    port: int
    database: str
    username: str
    password: str = ""

def make_engine(cfg: DbConfig) -> Engine:
    url = f"postgresql+psycopg2://{cfg.username}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}"
    return create_engine(url, pool_pre_ping=True)

def _get_column_type(engine: Engine, table: str, column: str) -> Optional[str]:
    sql = """
    SELECT data_type
    FROM information_schema.columns
    WHERE table_name = :table AND column_name = :column
    LIMIT 1
    """
    with engine.connect() as conn:
        row = conn.execute(text(sql), {"table": table, "column": column}).fetchone()
    return str(row[0]).lower() if row else None

def _detect_epoch_unit(engine: Engine, table: str, time_col: str, epic_col: str, epic_value: str) -> Tuple[str, float]:
    sql = f"SELECT MAX({time_col}) FROM {table} WHERE {epic_col} = :epic"
    with engine.connect() as conn:
        max_ts = conn.execute(text(sql), {"epic": epic_value}).scalar()
    if max_ts is None:
        return ("seconds", 1.0)
    v = float(max_ts)
    if v > 1e14:
        return ("microseconds", 1_000_000.0)
    if v > 1e11:
        return ("milliseconds", 1000.0)
    return ("seconds", 1.0)

def fetch_candles(
    engine: Engine,
    table: str,
    cols: Dict[str, str],
    epic_value: str,
    start_ts,
    end_ts=None,
    last_only_newer_than=None,
    timeframe: Optional[str] = None,
) -> pd.DataFrame:
    time_col = cols["time"]
    epic_col = cols["epic"]
    close_col = cols["close"]
    vol_col = cols["vol"]
    buyers_col = cols["buyers_pct"]
    sellers_col = cols["sellers_pct"]

    col_type = _get_column_type(engine, table, time_col)
    is_numeric_time = col_type in {"bigint","integer","numeric","double precision","real","smallint"}

    where_parts = [f"{epic_col} = :epic"]
    params: Dict[str, Any] = {"epic": epic_value}

    if timeframe is not None:
        where_parts.append("timeframe = :timeframe")
        params["timeframe"] = timeframe

    if is_numeric_time:
        _unit, divisor = _detect_epoch_unit(engine, table, time_col, epic_col, epic_value)

        def to_epoch(x: object) -> int:
            ts = pd.Timestamp(x)  # type: ignore[arg-type]
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            return int(float(ts.timestamp()) * divisor)

        if last_only_newer_than is not None:
            where_parts.append(f"{time_col} > :last_ts")
            params["last_ts"] = to_epoch(last_only_newer_than)
        else:
            where_parts.append(f"{time_col} >= :start_ts")
            params["start_ts"] = to_epoch(start_ts)

        if end_ts is not None:
            where_parts.append(f"{time_col} < :end_ts")
            params["end_ts"] = to_epoch(end_ts)

        time_expr = f"to_timestamp({time_col} / {divisor}) AT TIME ZONE 'UTC'"
        order_expr = time_col
    else:
        if last_only_newer_than is not None:
            where_parts.append(f"{time_col} > :last_ts")
            params["last_ts"] = last_only_newer_than
        else:
            where_parts.append(f"{time_col} >= :start_ts")
            params["start_ts"] = start_ts

        if end_ts is not None:
            where_parts.append(f"{time_col} < :end_ts")
            params["end_ts"] = end_ts

        time_expr = time_col
        order_expr = time_col

    sql = f"""
    SELECT
      {time_expr} AS market_time,
      {close_col} AS close,
      {vol_col} AS vol,
      {buyers_col} AS buyers_pct,
      {sellers_col} AS sellers_pct,
      {epic_col} AS epic
    FROM {table}
    WHERE {' AND '.join(where_parts)}
    ORDER BY {order_expr} ASC
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    if not df.empty:
        df["market_time"] = pd.to_datetime(df["market_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["market_time"]).sort_values("market_time").reset_index(drop=True)
    return df

def inspect_columns(engine: Engine, table: str) -> List[str]:
    sql = """
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = :table
    ORDER BY ordinal_position
    """
    with engine.connect() as conn:
        rows = conn.execute(text(sql), {"table": table}).fetchall()
    return [f"{r[0]} ({r[1]})" for r in rows]
