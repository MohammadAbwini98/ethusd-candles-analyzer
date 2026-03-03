from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List

from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# timeframe label → seconds per candle
TIMEFRAMES: Dict[str, int] = {
    "tick": 1,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1H": 3600,
}


class _Accumulator:
    """Accumulates ticks within one candle period."""

    __slots__ = ("open", "high", "low", "close", "volume", "count")

    def __init__(self) -> None:
        self.open = 0.0
        self.high = -float("inf")
        self.low = float("inf")
        self.close = 0.0
        self.volume = 0.0
        self.count = 0

    def update(self, price: float, volume: float = 0.0) -> None:
        if self.count == 0:
            self.open = price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.close = price
        self.volume += volume
        self.count += 1


class CandleBuilder:
    """Builds candles for all configured timeframes from tick data."""

    def __init__(self, epic: str = "ETHUSD", timeframes: Dict[str, int] | None = None):
        self.epic = epic
        self.timeframes = timeframes or TIMEFRAMES
        self._accumulators: Dict[str, _Accumulator] = {}
        self._period_starts: Dict[str, datetime] = {}

    @staticmethod
    def _floor(ts: datetime, interval: int) -> datetime:
        """Floor timestamp to the beginning of its period."""
        epoch = int(ts.timestamp())
        floored = (epoch // interval) * interval
        return datetime.fromtimestamp(floored, tz=timezone.utc)

    def on_tick(
        self,
        price: float,
        volume: float,
        ts: datetime,
        buyers_pct: float,
        sellers_pct: float,
    ) -> List[Dict]:
        """Process one tick. Returns list of completed candle dicts."""
        completed: List[Dict] = []

        for tf_label, interval in self.timeframes.items():
            period_start = self._floor(ts, interval)

            if tf_label not in self._period_starts:
                self._period_starts[tf_label] = period_start
                self._accumulators[tf_label] = _Accumulator()

            # period boundary crossed → finalize the old candle
            if period_start > self._period_starts[tf_label]:
                acc = self._accumulators[tf_label]
                if acc.count > 0:
                    completed.append({
                        "ts": self._period_starts[tf_label],
                        "epic": self.epic,
                        "timeframe": tf_label,
                        "open": acc.open,
                        "high": acc.high,
                        "low": acc.low,
                        "close": acc.close,
                        "vol": acc.volume,
                        "buyers_pct": buyers_pct,
                        "sellers_pct": sellers_pct,
                    })
                self._period_starts[tf_label] = period_start
                self._accumulators[tf_label] = _Accumulator()

            self._accumulators[tf_label].update(price, volume)

        return completed


def insert_candles(engine: Engine, schema: str, candles: List[Dict]) -> int:
    """Upsert candle rows into {schema}.candles. Returns count inserted."""
    if not candles:
        return 0
    sql = text(f"""
        INSERT INTO {schema}.candles
            (ts, epic, timeframe, open, high, low, close, vol, buyers_pct, sellers_pct)
        VALUES
            (:ts, :epic, :timeframe, :open, :high, :low, :close, :vol, :buyers_pct, :sellers_pct)
        ON CONFLICT (ts, epic, timeframe)
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low  = EXCLUDED.low,
            close = EXCLUDED.close,
            vol   = EXCLUDED.vol,
            buyers_pct = EXCLUDED.buyers_pct,
            sellers_pct = EXCLUDED.sellers_pct
    """)
    with engine.begin() as conn:
        conn.execute(sql, candles)
    return len(candles)
