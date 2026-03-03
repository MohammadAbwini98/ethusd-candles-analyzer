from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml, time, json, requests

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


# ── WhatsApp alert helper ───────────────────────────────────────

# Rate-limit cache: maps "alert_type_<window_bucket>" → count
_alert_cache: Dict[str, int] = {}


def send_alert(message: str, cfg: Dict[str, Any], alert_type: str = "general") -> None:
    """Send a WhatsApp alert via the notifier sidecar (non-blocking best-effort).

    Args:
        message:    Text to send to the configured WA number.
        cfg:        Full raw config dict (cfg.raw).
        alert_type: Category key used for per-type rate limiting.
                    Common values: "signal_fired", "calibration_warning",
                    "sanity_check_fail", "daily_summary", "system_error".
    """
    alerts_cfg: Dict[str, Any] = cfg.get("alerts", {})

    if not alerts_cfg.get("enabled", False):
        return

    # Event-level switch
    event_flags: Dict[str, bool] = alerts_cfg.get("events", {})
    if alert_type in event_flags and not event_flags[alert_type]:
        return

    # Rate limiting
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

    url: str = alerts_cfg.get("whatsapp_notifier_url", "http://127.0.0.1:3099/send")
    timeout: float = float(alerts_cfg.get("timeout_seconds", 3))

    try:
        resp = requests.post(url, json={"message": message}, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        # Sidecar not running — silent no-op
        pass
    except requests.exceptions.HTTPError as exc:
        # 503 = WhatsApp client not ready yet (still authenticating) — silent no-op
        if exc.response is not None and exc.response.status_code == 503:
            pass
        else:
            print(f"[Alert] Failed to deliver ({alert_type}): {exc}")
    except Exception as exc:
        # Any other failure (timeout, etc.) — log briefly
        print(f"[Alert] Failed to deliver ({alert_type}): {exc}")
