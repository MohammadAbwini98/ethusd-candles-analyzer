from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import json
import math
import webbrowser

from sqlalchemy import text
from sqlalchemy.engine import Engine


class _ReuseAddrHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that always sets SO_REUSEADDR + SO_REUSEPORT."""
    allow_reuse_address = True

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        except Exception:
            pass
        try:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass  # SO_REUSEPORT not available on all platforms
        super().server_bind()

def _sanitize(obj: object) -> object:
    """Replace float NaN/Inf with None so json.dumps produces valid JSON."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


class DashboardServer:
    def __init__(self, engine: Engine, schema: str, host: str, port: int, open_browser: bool = True):
        self.engine = engine
        self.schema = schema
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self._thread = None
        self._httpd = None
        self._static_dir = Path(__file__).parent / "static"
        self.live_price: dict = {}  # shared state: {"price": float, "bid": float, "ask": float, "ts": str}

    def update_live_price(self, price: float, bid: float, ask: float, ts: str) -> None:
        self.live_price = {"price": price, "bid": bid, "ask": ask, "ts": ts}

    @staticmethod
    def _free_port(port: int) -> None:
        """Kill any PID listening on *port* (excluding our own process)."""
        my_pid = os.getpid()
        try:
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True, text=True
            )
            pids = [p for p in result.stdout.strip().split() if p]
            for pid_str in pids:
                try:
                    pid = int(pid_str)
                    if pid != my_pid:
                        os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
        except Exception:
            pass

    def start(self):
        handler = self._make_handler()
        last_err = None
        for attempt in range(5):
            if attempt > 0:
                self._free_port(self.port)
                time.sleep(0.6 * attempt)  # 0.6s, 1.2s, 1.8s, 2.4s
            try:
                self._httpd = _ReuseAddrHTTPServer((self.host, self.port), handler)
                break
            except OSError as exc:
                last_err = exc
        else:
            raise OSError(
                f"[dashboard] Could not bind to port {self.port} after 5 attempts: {last_err}"
            ) from last_err

        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

        url = f"http://{self.host}:{self.port}/"
        print(f"[dashboard] running at {url}")
        if self.open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass

    def stop(self):
        if self._httpd:
            self._httpd.shutdown()

    def _make_handler(self):
        engine = self.engine
        schema = self.schema
        static_dir = self._static_dir
        server_ref = self  # capture reference for live_price access

        class Handler(BaseHTTPRequestHandler):
            def _send(self, code: int, body: bytes, content_type: str):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                parsed = urlparse(self.path)
                path = parsed.path
                qs = parse_qs(parsed.query)
                tf = qs.get("tf", ["15m"])[0]

                if path == "/" or path == "/index.html":
                    return self._send(200, (static_dir / "index.html").read_bytes(), "text/html; charset=utf-8")

                if path == "/api/snapshot":
                    payload = _sanitize(_latest_snapshot(engine, schema, tf))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/corr":
                    payload = _sanitize(_latest_corr(engine, schema, tf))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/lagcorr":
                    payload = _sanitize(_latest_lagcorr(engine, schema, tf))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/rolling":
                    payload = _sanitize(_latest_rolling(engine, schema, tf))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/signals":
                    payload = _sanitize(_latest_signals(engine, schema, tf))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/signals/history":
                    limit = int(qs.get("limit", ["50"])[0])
                    payload = _sanitize(_signal_history(engine, schema, tf, limit))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/calibration":
                    payload = _sanitize(_latest_calibration(engine, schema, tf))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/equity":  # FR-33: optional equity curve panel
                    payload = _sanitize(_latest_equity(engine, schema, tf))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/candles":
                    limit = int(qs.get("limit", ["200"])[0])
                    before = qs.get("before", [None])[0]
                    before_ts = int(before) if before else None
                    payload = _sanitize(_latest_candles(engine, schema, tf, limit, before_ts))
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                if path == "/api/price":
                    payload = server_ref.live_price or {"price": None}
                    return self._send(200, json.dumps(payload, default=str).encode("utf-8"), "application/json")

                return self._send(404, b"Not found", "text/plain")

            def log_message(self, format, *args):
                return

        return Handler

def _latest_snapshot(engine: Engine, schema: str, tf: str) -> dict:
    sql = text(f"""
      SELECT timeframe, computed_at, start_time, end_time, rows, latest_close, latest_score, buyers_plus_sellers_mean_abs_diff
      FROM {schema}.snapshots
      WHERE timeframe = :tf
      ORDER BY computed_at DESC
      LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"tf": tf}).mappings().first()
    return dict(row) if row else {"timeframe": tf, "error": "no snapshot yet"}

def _latest_corr(engine: Engine, schema: str, tf: str) -> dict:
    with engine.connect() as conn:
        latest = conn.execute(text(f"SELECT MAX(computed_at) FROM {schema}.corr_results WHERE timeframe=:tf"), {"tf": tf}).scalar()
        if latest is None:
            return {"timeframe": tf, "items": []}
        rows = conn.execute(text(f"""
            SELECT horizon, feature, n, pearson_r, pearson_p, spearman_r, spearman_p
            FROM {schema}.corr_results
            WHERE timeframe=:tf AND computed_at=:t
        """), {"tf": tf, "t": latest}).mappings().all()
    items = [dict(r) for r in rows]
    items.sort(key=lambda r: abs(r.get("pearson_r") or 0.0), reverse=True)
    return {"timeframe": tf, "computed_at": latest, "items": items}

def _latest_lagcorr(engine: Engine, schema: str, tf: str) -> dict:
    with engine.connect() as conn:
        latest = conn.execute(text(f"SELECT MAX(computed_at) FROM {schema}.lagcorr_results WHERE timeframe=:tf"), {"tf": tf}).scalar()
        if latest is None:
            return {"timeframe": tf, "items": []}
        rows = conn.execute(text(f"""
            SELECT lag, n, pearson_r, pearson_p
            FROM {schema}.lagcorr_results
            WHERE timeframe=:tf AND computed_at=:t
            ORDER BY lag ASC
        """), {"tf": tf, "t": latest}).mappings().all()
    return {"timeframe": tf, "computed_at": latest, "items": [dict(r) for r in rows]}

def _latest_rolling(engine: Engine, schema: str, tf: str) -> dict:
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT market_time, "window", value
            FROM {schema}.rollingcorr_points
            WHERE timeframe=:tf AND feature='score' AND horizon=1 AND "window" IN (20,50)
            ORDER BY market_time DESC
            LIMIT 400
        """), {"tf": tf}).mappings().all()
    if not rows:
        return {"timeframe": tf, "items": []}

    by_time = {}
    for r in rows:
        mt = r["market_time"]
        by_time.setdefault(mt, {"market_time": str(mt), "w20": None, "w50": None})
        if int(r["window"]) == 20:
            by_time[mt]["w20"] = r["value"]
        elif int(r["window"]) == 50:
            by_time[mt]["w50"] = r["value"]

    items = list(by_time.values())
    items.sort(key=lambda x: x["market_time"])
    items = items[-200:]
    return {"timeframe": tf, "items": items}


def _latest_signals(engine: Engine, schema: str, tf: str) -> dict:
    sql = text(f"""
        SELECT timeframe, computed_at, regime, signal, confidence,
               entry_price, stop_loss, take_profit, hold_bars, reason,
               conf_regime, conf_tail, conf_backtest,
               rc, ar, score_mr, score_mom, volatility, params_json
        FROM {schema}.signal_recommendations
        WHERE timeframe = :tf
        ORDER BY computed_at DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"tf": tf}).mappings().first()
    return dict(row) if row else {"timeframe": tf, "signal": "NONE", "reason": "no data yet"}


def _signal_history(engine: Engine, schema: str, tf: str, limit: int = 50) -> dict:
    sql = text(f"""
        SELECT computed_at, regime, signal, confidence, entry_price,
               stop_loss, take_profit, hold_bars, reason, rc, ar
        FROM {schema}.signal_recommendations
        WHERE timeframe = :tf
        ORDER BY computed_at DESC
        LIMIT :lim
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"tf": tf, "lim": limit}).mappings().all()
    items = [dict(r) for r in rows]
    items.reverse()
    return {"timeframe": tf, "items": items}


def _latest_calibration(engine: Engine, schema: str, tf: str) -> dict:
    sql = text(f"""
        SELECT timeframe, computed_at, best_params, net_sharpe, net_return,
               max_drawdown, n_trades, win_rate, param_grid_size, lookback_days,
               min_trades, eligible_candidates, total_candidates, status, rejection_reason,
               max_dd_used, rejected_by_min_trades, rejected_by_max_dd,
               rejected_by_both, max_trades_seen, best_dd_seen
        FROM {schema}.calibration_runs
        WHERE timeframe = :tf
        ORDER BY computed_at DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"tf": tf}).mappings().first()
    if row:
        result = dict(row)
        if isinstance(result.get("best_params"), str):
            result["best_params"] = json.loads(result["best_params"])
        return result
    return {"timeframe": tf, "error": "no calibration yet"}


def _latest_equity(engine: Engine, schema: str, tf: str) -> dict:
    """FR-33: Return the most recent strategy_equity curve for a timeframe."""
    try:
        sql = text(f"""
            SELECT se.bar_index, se.equity
            FROM {schema}.strategy_equity se
            JOIN {schema}.strategy_runs sr ON sr.id = se.run_id
            WHERE sr.timeframe = :tf
            AND sr.id = (
                SELECT id FROM {schema}.strategy_runs
                WHERE timeframe = :tf
                ORDER BY computed_at DESC LIMIT 1
            )
            ORDER BY se.bar_index ASC
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql, {"tf": tf}).mappings().all()
        return {"timeframe": tf, "items": [dict(r) for r in rows]}
    except Exception:
        return {"timeframe": tf, "items": []}


def _latest_candles(engine: Engine, schema: str, tf: str, limit: int = 200, before_ts: int | None = None) -> dict:
    """Return recent OHLCV bars for the candlestick chart.
    
    Args:
        engine: SQLAlchemy engine
        schema: Database schema name
        tf: Timeframe (e.g., "1m", "5m")
        limit: Maximum number of bars to return
        before_ts: Optional timestamp - if provided, only return bars before this time (for lazy loading)
    """
    try:
        if before_ts:
            # For lazy loading: fetch bars older than before_ts
            sql = text(f"""
                SELECT ts, open, high, low, close, vol
                FROM {schema}.candles
                WHERE timeframe = :tf
                  AND open IS NOT NULL
                  AND EXTRACT(EPOCH FROM ts) < :before
                ORDER BY ts DESC
                LIMIT :lim
            """)
            with engine.connect() as conn:
                rows = conn.execute(sql, {"tf": tf, "lim": limit, "before": before_ts}).mappings().all()
        else:
            # Normal fetch: most recent bars
            sql = text(f"""
                SELECT ts, open, high, low, close, vol
                FROM {schema}.candles
                WHERE timeframe = :tf
                  AND open IS NOT NULL
                ORDER BY ts DESC
                LIMIT :lim
            """)
            with engine.connect() as conn:
                rows = conn.execute(sql, {"tf": tf, "lim": limit}).mappings().all()
        
        items = []
        for r in reversed(list(rows)):
            ts = r["ts"]
            if hasattr(ts, "timestamp"):
                epoch = int(ts.timestamp())
            else:
                epoch = int(float(ts))
            items.append({
                "time": epoch,
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "vol": float(r["vol"]) if r["vol"] is not None else 0.0,
            })
        return {"timeframe": tf, "items": items}
    except Exception as exc:
        return {"timeframe": tf, "items": [], "error": str(exc)}
