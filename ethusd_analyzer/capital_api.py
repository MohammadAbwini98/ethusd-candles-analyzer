from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class CapitalSession:
    """Capital.com REST API client with auto-refreshing session."""

    REFRESH_INTERVAL = 540  # 9 minutes (session expires at 10 min inactivity)

    def __init__(
        self,
        api_key: str,
        email: str,
        password: str,
        base_url: str = "https://api-capital.backend-capital.com",
    ):
        self.api_key = api_key
        self.email = email
        self.password = password
        self.base_url = base_url.rstrip("/")
        self._cst: Optional[str] = None
        self._security_token: Optional[str] = None
        self._last_auth: float = 0.0

    # ── authentication ──────────────────────────────────────────

    def _authenticate(self) -> None:
        resp = requests.post(
            f"{self.base_url}/api/v1/session",
            json={
                "identifier": self.email,
                "password": self.password,
                "encryptedPassword": False,
            },
            headers={
                "X-CAP-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        self._cst = resp.headers["CST"]
        self._security_token = resp.headers["X-SECURITY-TOKEN"]
        self._last_auth = time.time()
        logger.info("[capital] Session authenticated")

    def _ensure_session(self) -> None:
        if self._cst is None or (time.time() - self._last_auth) >= self.REFRESH_INTERVAL:
            self._authenticate()

    @property
    def _headers(self) -> Dict[str, str]:
        self._ensure_session()
        assert self._cst is not None, "Session not authenticated"
        assert self._security_token is not None, "Session not authenticated"
        return {
            "X-SECURITY-TOKEN": self._security_token,
            "CST": self._cst,
            "Content-Type": "application/json",
        }

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_time(s: str) -> datetime:
        """Parse Capital.com time string → UTC datetime."""
        s = s.replace("/", "-").replace(" ", "T")
        if not s.endswith("Z"):
            s += "Z"
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    @staticmethod
    def _mid(price_obj: Dict) -> float:
        return (price_obj["bid"] + price_obj["ask"]) / 2

    @staticmethod
    def _iso(dt: datetime) -> str:
        """Format datetime for Capital.com query params."""
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    # ── historical candles ──────────────────────────────────────

    def get_prices(self, epic: str, resolution: str, max_count: int = 1000) -> List[Dict]:
        """GET /api/v1/prices/{epic} — returns list of candle dicts."""
        resp = requests.get(
            f"{self.base_url}/api/v1/prices/{epic}",
            params={"resolution": resolution, "max": max_count},
            headers=self._headers,
            timeout=15,
        )
        resp.raise_for_status()
        return self._parse_prices(resp.json())

    def get_prices_range(
        self, epic: str, resolution: str, start: datetime, end: datetime, max_count: int = 1000
    ) -> List[Dict]:
        """GET /api/v1/prices/{epic} with from/to date range."""
        params = {
            "resolution": resolution,
            "max": max_count,
            "from": self._iso(start),
            "to": self._iso(end),
        }
        resp = requests.get(
            f"{self.base_url}/api/v1/prices/{epic}",
            params=params,
            headers=self._headers,
            timeout=15,
        )
        resp.raise_for_status()
        return self._parse_prices(resp.json())

    def _parse_prices(self, data: Dict) -> List[Dict]:
        candles = []
        for p in data.get("prices", []):
            time_str = p.get("snapshotTimeUTC") or p.get("snapshotTime", "")
            candles.append({
                "time": self._parse_time(time_str),
                "open": self._mid(p["openPrice"]),
                "high": self._mid(p["highPrice"]),
                "low": self._mid(p["lowPrice"]),
                "close": self._mid(p["closePrice"]),
                "volume": p.get("lastTradedVolume", 0) or 0,
            })
        candles.sort(key=lambda c: c["time"])
        return candles

    # ── live price ──────────────────────────────────────────────

    def get_live_price(self, epic: str) -> Dict:
        """GET /api/v1/markets/{epic} — current bid/ask/mid + market status."""
        resp = requests.get(
            f"{self.base_url}/api/v1/markets/{epic}",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        snap = resp.json().get("snapshot", {})
        bid = snap.get("bid", 0)
        ask = snap.get("offer", 0)
        return {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "status": snap.get("marketStatus", "UNKNOWN"),
        }

    # ── sentiment ───────────────────────────────────────────────

    def get_sentiment(self, epic: str) -> Dict:
        """GET /api/v1/clientsentiment/{epic} — buyers % / sellers %."""
        resp = requests.get(
            f"{self.base_url}/api/v1/clientsentiment/{epic}",
            headers=self._headers,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        # Capital.com may nest under "clientSentiment"
        d = data.get("clientSentiment", data)
        buyers = (
            d.get("longPositionPercentage")
            or d.get("buyersPercentage")
            or 50.0
        )
        sellers = (
            d.get("shortPositionPercentage")
            or d.get("sellersPercentage")
            or 50.0
        )
        return {"buyers_pct": float(buyers), "sellers_pct": float(sellers)}

    # ── paginated historical backfill ───────────────────────────

    def backfill(
        self,
        epic: str,
        resolution: str,
        hours_back: int = 720,
        chunk_minutes: int = 960,
        start_from: Optional[datetime] = None,
    ) -> List[Dict]:
        """Fetch historical candles in chunks to cover hours_back of data.

        If start_from is provided, fetch only from that point forward
        (used to skip already-fetched data).
        """
        end = datetime.now(timezone.utc)
        start = start_from if start_from is not None else (end - timedelta(hours=hours_back))
        chunk_td = timedelta(minutes=chunk_minutes)

        all_candles: List[Dict] = []
        cursor = start

        while cursor < end:
            chunk_end = min(cursor + chunk_td, end)
            try:
                batch = self.get_prices_range(epic, resolution, cursor, chunk_end)
                if batch:
                    all_candles.extend(batch)
                    logger.info(
                        f"[capital] Fetched {len(batch)} candles "
                        f"{batch[0]['time']} → {batch[-1]['time']}"
                    )
            except Exception as e:
                logger.warning(f"[capital] Backfill chunk failed ({cursor} → {chunk_end}): {e}")
            cursor = chunk_end
            time.sleep(0.15)  # stay well under 10 req/s

        # deduplicate by time
        seen = set()
        unique: List[Dict] = []
        for c in all_candles:
            if c["time"] not in seen:
                seen.add(c["time"])
                unique.append(c)
        unique.sort(key=lambda c: c["time"])
        return unique
