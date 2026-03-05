from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import copy, yaml, time, json, requests, threading

@dataclass
class Config:
    raw: Dict[str, Any]

    @property
    def db(self) -> Dict[str, Any]:
        return self.raw.get("db", {})

    @property
    def table(self) -> str:
        return self.raw.get("table", "candles")

    @property
    def filters(self) -> Dict[str, Any]:
        return self.raw.get("filters", {})

    @property
    def columns(self) -> Dict[str, str]:
        return self.raw.get("columns", {})

    @property
    def stream(self) -> Dict[str, Any]:
        return self.raw.get("stream", {})

    @property
    def analysis(self) -> Dict[str, Any]:
        return self.raw.get("analysis", {})

def load_config(path: str) -> Config:
    p = Path(path)
    return Config(raw=yaml.safe_load(p.read_text(encoding="utf-8")))

def now_ts() -> float:
    return time.time()

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


# ── Notification Tracker ───────────────────────────────────────

class NotificationTracker:
    """Thread-safe per-channel delivery tracker for observability.

    Records last_attempt, last_success, and last_error per channel.
    Exposed via /api/health and POST /api/notify/test.
    """

    CHANNELS = ("whatsapp", "telegram", "email", "macos")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {ch: {} for ch in self.CHANNELS}

    def record_attempt(self, channel: str) -> None:
        with self._lock:
            self._data.setdefault(channel, {})["last_attempt"] = time.time()

    def record_success(self, channel: str) -> None:
        with self._lock:
            d = self._data.setdefault(channel, {})
            d["last_success"] = time.time()
            d["last_error"] = None

    def record_error(self, channel: str, err: str) -> None:
        with self._lock:
            self._data.setdefault(channel, {})["last_error"] = str(err)[:300]

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._data)


# Module-level singleton
_notif_tracker: NotificationTracker = NotificationTracker()


def get_notification_tracker() -> NotificationTracker:
    """Return the module-level NotificationTracker singleton."""
    return _notif_tracker


# ── Alert helpers (WhatsApp + Telegram) ───────────────────────

# Rate-limit cache: maps "alert_type_<window_bucket>" → count
_alert_cache: Dict[str, int] = {}

# Module-level Telegram notifier (singleton, set by run.py)
_telegram_notifier: Optional[Any] = None

# Module-level Email notifier (singleton, set by run.py)
_email_notifier: Optional[Any] = None

# Module-level macOS notifier (singleton, set by run.py)
_macos_notifier: Optional[Any] = None


def set_telegram_notifier(notifier: Optional[Any]) -> None:
    """Register the Telegram notifier instance (called by run.py)."""
    global _telegram_notifier
    _telegram_notifier = notifier


def set_email_notifier(notifier: Optional[Any]) -> None:
    """Register the Email notifier instance (called by run.py)."""
    global _email_notifier
    _email_notifier = notifier


def set_macos_notifier(notifier: Optional[Any]) -> None:
    """Register the macOS notifier instance (called by run.py)."""
    global _macos_notifier
    _macos_notifier = notifier


def _send_telegram_async(message: str, alert_type: str) -> None:
    """Send Telegram alert in background (non-blocking best-effort)."""
    notifier = _telegram_notifier
    if notifier is None:
        return

    def _worker() -> None:
        _notif_tracker.record_attempt("telegram")
        try:
            notifier.send_message(message)
            _notif_tracker.record_success("telegram")
        except Exception as exc:
            _notif_tracker.record_error("telegram", str(exc))
            print(f"[Alert] Telegram delivery failed ({alert_type}): {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def _send_email_async(subject: str, html_body: str, alert_type: str) -> None:
    """Send Email alert in background (non-blocking best-effort)."""
    notifier = _email_notifier
    if notifier is None:
        return

    def _worker() -> None:
        _notif_tracker.record_attempt("email")
        try:
            notifier.send_message(subject, html_body)
            _notif_tracker.record_success("email")
        except Exception as exc:
            _notif_tracker.record_error("email", str(exc))
            print(f"[Alert] Email delivery failed ({alert_type}): {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def _send_macos_async(title: str, message: str, alert_type: str) -> None:
    """Send macOS alert in background (non-blocking best-effort)."""
    notifier = _macos_notifier
    if notifier is None:
        return

    def _worker() -> None:
        _notif_tracker.record_attempt("macos")
        try:
            notifier.notify(title, message)
            _notif_tracker.record_success("macos")
        except Exception as exc:
            _notif_tracker.record_error("macos", str(exc))
            print(f"[Alert] macOS delivery failed ({alert_type}): {exc}")

    threading.Thread(target=_worker, daemon=True).start()


def send_alert(message: str, cfg: Dict[str, Any], alert_type: str = "general") -> None:
    """Send an alert via WhatsApp, Telegram, Email, and macOS (non-blocking best-effort).

    Args:
        message:    Text to send to the configured channels.
        cfg:        Full raw config dict (cfg.raw).
        alert_type: Category key used for per-type rate limiting.
                    Common values: "signal_fired", "calibration_warning",
                    "sanity_check_fail", "daily_summary", "system_error", "startup", "shutdown".
    """
    alerts_cfg: Dict[str, Any] = cfg.get("alerts", {})

    # Event-level switch — applies to all channels.
    event_flags: Dict[str, bool] = alerts_cfg.get("events", {})
    if alert_type in event_flags and not event_flags[alert_type]:
        return

    # Rate limiting — applies to all channels.
    rl = alerts_cfg.get("rate_limit", {})
    if rl.get("enabled", True):
        now = time.time()
        window: int = int(rl.get("window_seconds", 60))
        max_per_window: int = int(rl.get("max_per_window", 10))
        key = f"{alert_type}_{int(now // window)}"
        _alert_cache[key] = _alert_cache.get(key, 0) + 1
        if _alert_cache[key] > max_per_window:
            return  # silently drop — rate limited

        # Prune old cache entries to avoid unbounded growth
        current_bucket = int(now // window)
        stale = [k for k in list(_alert_cache) if int(k.split("_")[-1]) < current_bucket - 2]
        for k in stale:
            _alert_cache.pop(k, None)

    # ── WhatsApp: gated by alerts.enabled (master switch for WA sidecar) ─────
    if alerts_cfg.get("enabled", False):
        url: str = alerts_cfg.get("whatsapp_notifier_url", "http://127.0.0.1:3099/send")
        timeout: float = float(alerts_cfg.get("timeout_seconds", 3))

        def _whatsapp_post(msg: str) -> None:
            """Try up to 5 times with 2-second gaps. Handles the race where the
            notifier sidecar is just starting and hasn't bound its port yet."""
            for attempt in range(5):
                _notif_tracker.record_attempt("whatsapp")
                try:
                    resp = requests.post(url, json={"message": msg}, timeout=timeout)
                    resp.raise_for_status()
                    _notif_tracker.record_success("whatsapp")
                    return  # delivered (or queued by notifier)
                except requests.exceptions.ConnectionError:
                    err = f"ConnectionError (attempt {attempt + 1}/5)"
                    _notif_tracker.record_error("whatsapp", err)
                    if attempt < 4:
                        time.sleep(2)
                    else:
                        print(f"[Alert] WhatsApp unreachable after 5 attempts ({alert_type}) — sidecar not running?")
                    continue
                except requests.exceptions.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else 0
                    _notif_tracker.record_error("whatsapp", f"HTTP {status}")
                    if status == 503:
                        print(f"[Alert] WhatsApp 503 ({alert_type}) — notifier queue full or shutting down")
                    else:
                        print(f"[Alert] WhatsApp delivery failed ({alert_type}): {exc}")
                    return
                except Exception as exc:
                    _notif_tracker.record_error("whatsapp", str(exc)[:200])
                    print(f"[Alert] WhatsApp delivery failed ({alert_type}): {exc}")
                    return

        threading.Thread(target=_whatsapp_post, args=(message,), daemon=True, name=f"wa-alert-{alert_type}").start()

    # ── Telegram / Email / macOS: each controlled by its own enabled flag ─────
    # Alert types with dedicated rich formatters in run.py are excluded here
    # (startup/shutdown/signal_fired) to prevent duplicating plain-text messages.
    _DIRECT_TYPES: frozenset = frozenset({"startup", "shutdown", "signal_fired"})
    if alert_type not in _DIRECT_TYPES:
        _send_telegram_async(message, alert_type)

        if _email_notifier is not None:
            subject = f"Alert: {alert_type.replace('_', ' ').title()}"
            html_body = f"<html><body><pre>{message}</pre></body></html>"
            _send_email_async(subject, html_body, alert_type)

        if _macos_notifier is not None:
            alert_emoji = {
                "sanity_check_fail": "⚠️",
                "calibration_warning": "🟡",
                "system_error": "💥",
                "daily_summary": "📊",
            }.get(alert_type, "ℹ️")
            title = f"{alert_emoji} {alert_type.replace('_', ' ').title()}"
            short_message = message[:200] + "..." if len(message) > 200 else message
            _send_macos_async(title, short_message, alert_type)
