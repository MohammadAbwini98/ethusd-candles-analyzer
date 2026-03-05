"""macOS Notification Center notifier for trading alerts.

Sends native macOS desktop notifications via osascript or terminal-notifier.
Complements Telegram and Email notifiers without replacing them.
"""

import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MacOSNotifier:
    """Send native macOS Notification Center alerts.
    
    Features:
    - Auto-detect Terminal Notifier or fall back to osascript
    - Rate limiting (min_interval_seconds between sends)
    - Deduplication for signal notifications
    - Non-blocking (threaded)
    - Silent failures (logs warnings, doesn't crash)
    """
    
    def __init__(
        self,
        enabled: bool = True,
        method: str = "auto",  # "auto" | "osascript" | "terminal-notifier"
        title_default: str = "Trading Bot",
        app_name: str = "ETH Analyzer",
        sound: str = "",  # e.g., "Ping" for osascript, empty = no sound
        timeout_seconds: float = 2.0,
        rate_limit_enabled: bool = True,
        rate_limit_min_interval: float = 1.0,
        dedupe_enabled: bool = True,
    ):
        """Initialize macOS notifier.
        
        Args:
            enabled: Master enable/disable
            method: "auto" (detect), "osascript", or "terminal-notifier"
            title_default: Default notification title
            app_name: App name displayed in notifications
            sound: Sound name (e.g., "Ping"; empty = silent)
            timeout_seconds: Subprocess timeout
            rate_limit_enabled: Enable rate limiting
            rate_limit_min_interval: Min seconds between notifications
            dedupe_enabled: Enable deduplication for signals
        """
        if not sys.platform.startswith("darwin"):
            logger.debug("[macos] Not running on macOS, disabling notifications")
            self.enabled = False
            return
        
        if not enabled:
            logger.debug("[macos] Disabled via config")
            self.enabled = False
            return
        
        self.enabled = True
        self.title_default = title_default
        self.app_name = app_name
        self.sound = sound.strip() if sound else ""
        self.timeout_seconds = timeout_seconds
        self.rate_limit_enabled = rate_limit_enabled
        self.rate_limit_min_interval = rate_limit_min_interval
        self.dedupe_enabled = dedupe_enabled
        
        # Select notification method
        self.method = self._select_method(method)
        logger.info(
            "[macos] Notifier initialized: method=%s app=%s sound=%s rate_limit=%s",
            self.method, app_name, sound or "none", rate_limit_enabled
        )
        
        # Rate limiting state
        self._last_send_time: float = 0.0
        self._lock = threading.Lock()
        
        # Deduplication state (per timeframe)
        self._last_signal_key: Dict[str, str] = {}
    
    def _select_method(self, method: str) -> str:
        """Select notification method based on availability.
        
        Args:
            method: "auto" | "osascript" | "terminal-notifier"
        
        Returns:
            Selected method name
        """
        if method == "auto":
            # Try terminal-notifier first (richer features)
            if self._check_command_exists("terminal-notifier"):
                return "terminal-notifier"
            return "osascript"  # Always available on macOS
        
        # Explicit method requested
        if method == "terminal-notifier":
            if self._check_command_exists("terminal-notifier"):
                return "terminal-notifier"
            logger.warning(
                "[macos] terminal-notifier requested but not found in PATH, "
                "falling back to osascript"
            )
            return "osascript"
        
        return "osascript"
    
    @staticmethod
    def _check_command_exists(cmd: str) -> bool:
        """Check if a command exists in PATH."""
        return subprocess.run(
            ["which", cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode == 0
    
    def _notify_osascript(
        self, title: str, message: str, subtitle: Optional[str] = None
    ) -> bool:
        """Send notification via osascript.
        
        Args:
            title: Notification title
            message: Notification message
            subtitle: Optional subtitle
        
        Returns:
            True if successful, False on failure
        """
        try:
            # Escape quotes for AppleScript
            title = title.replace('"', '\\"')
            message = message.replace('"', '\\"')
            subtitle = subtitle.replace('"', '\\"') if subtitle else ""
            
            # Limit length to avoid AppleScript errors
            title = title[:100]
            message = message[:500]
            subtitle = subtitle[:100] if subtitle else ""
            
            # Build AppleScript command
            parts = [f'display notification "{message}" with title "{title}"']
            if subtitle:
                parts.append(f'subtitle "{subtitle}"')
            if self.sound:
                parts.append(f'sound name "{self.sound}"')
            
            cmd = " ".join(parts)
            script = f'osascript -e \'{cmd}\''
            
            result = subprocess.run(
                script,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                text=True,
            )
            
            if result.returncode != 0:
                logger.warning(
                    "[macos] osascript failed: %s", result.stderr.strip()
                )
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.warning("[macos] osascript timed out")
            return False
        except Exception as e:
            logger.warning("[macos] osascript error: %s", e)
            return False
    
    def _notify_terminal_notifier(
        self, title: str, message: str, subtitle: Optional[str] = None
    ) -> bool:
        """Send notification via terminal-notifier.
        
        Args:
            title: Notification title
            message: Notification message
            subtitle: Optional subtitle
        
        Returns:
            True if successful, False on failure
        """
        try:
            cmd = ["terminal-notifier", "-title", title, "-message", message]
            
            if subtitle:
                cmd.extend(["-subtitle", subtitle])
            
            if self.sound:
                cmd.extend(["-sound", self.sound])
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_seconds,
                text=True,
            )
            
            if result.returncode != 0:
                logger.warning(
                    "[macos] terminal-notifier failed: %s", result.stderr.strip()
                )
                return False
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.warning("[macos] terminal-notifier timed out")
            return False
        except Exception as e:
            logger.warning("[macos] terminal-notifier error: %s", e)
            return False
    
    def _send_notification(
        self, title: str, message: str, subtitle: Optional[str] = None
    ) -> bool:
        """Internal: send notification (thread-safe).
        
        Args:
            title: Notification title
            message: Notification message
            subtitle: Optional subtitle
        
        Returns:
            True if sent, False if skipped or failed
        """
        if not self.enabled:
            return False
        
        # Rate limiting
        if self.rate_limit_enabled:
            with self._lock:
                now = time.time()
                if now - self._last_send_time < self.rate_limit_min_interval:
                    logger.debug("[macos] Skipped due to rate limit")
                    return False
                self._last_send_time = now
        
        # Dispatch to selected method
        if self.method == "terminal-notifier":
            return self._notify_terminal_notifier(title, message, subtitle)
        else:
            return self._notify_osascript(title, message, subtitle)
    
    def notify(
        self, title: str, message: str, subtitle: Optional[str] = None
    ) -> bool:
        """Send a notification (non-blocking via thread).
        
        Args:
            title: Notification title
            message: Notification message
            subtitle: Optional subtitle
        
        Returns:
            True if queued, False if skipped or disabled
        """
        if not self.enabled:
            return False
        
        def _worker():
            try:
                self._send_notification(title, message, subtitle)
            except Exception as e:
                logger.warning("[macos] Notification send error: %s", e)
        
        threading.Thread(target=_worker, daemon=True).start()
        return True
    
    def notify_sync(
        self, title: str, message: str, subtitle: Optional[str] = None
    ) -> bool:
        """Send a notification synchronously (blocking).
        
        Use this for shutdown notifications to ensure they complete before exit.
        
        Args:
            title: Notification title
            message: Notification message
            subtitle: Optional subtitle
        
        Returns:
            True if sent, False if skipped or failed
        """
        if not self.enabled:
            return False
        
        try:
            return self._send_notification(title, message, subtitle)
        except Exception as e:
            logger.warning("[macos] Notification send error: %s", e)
            return False
    
    @staticmethod
    def format_startup_message(symbol: str = "ETH", mode: str = "LIVE") -> tuple:
        """Format a startup notification.
        
        Returns:
            (title, message, subtitle) tuple
        """
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        title = "🟢 Analyzer Started"
        message = f"{symbol} | {mode} | {now_utc}"
        return title, message, None
    
    @staticmethod
    def format_shutdown_message(
        symbol: str = "ETH",
        reason: str = "normal",
        error_msg: Optional[str] = None,
    ) -> tuple:
        """Format a shutdown notification.
        
        Returns:
            (title, message, subtitle) tuple
        """
        now_utc = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        title = "🔴 Analyzer Stopped"
        message = f"{symbol} | {reason.upper()} | {now_utc}"
        
        if error_msg:
            message = f"{message}\nError: {error_msg[:80]}"
        
        return title, message, None
    
    @staticmethod
    def format_signal_message(
        symbol: str,
        timeframe: str,
        signal: str,
        confidence: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        hold_bars: int = 0,
    ) -> tuple:
        """Format a signal notification.
        
        Returns:
            (title, message, subtitle) tuple
        """
        signal_emoji = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
        
        title = f"{signal_emoji} {signal} | {symbol} | {timeframe}"
        
        risk_pct = abs((stop_loss - entry_price) / entry_price * 100)
        reward_pct = abs((take_profit - entry_price) / entry_price * 100)
        rr = reward_pct / risk_pct if risk_pct > 0 else 0.0
        
        message = (
            f"Conf: {confidence:.0%} | Entry: ${entry_price:.2f}\n"
            f"TP: ${take_profit:.2f} (+{reward_pct:.1f}%) | SL: ${stop_loss:.2f} (-{risk_pct:.1f}%)\n"
            f"R:R {rr:.2f}:1 | Hold: {hold_bars} bars"
        )
        
        subtitle = f"{symbol} {timeframe} signal detected"
        
        return title, message, subtitle
    
    @staticmethod
    def format_error_message(error_type: str, error_msg: str) -> tuple:
        """Format an error notification.
        
        Returns:
            (title, message, subtitle) tuple
        """
        title = f"⚠️ {error_type}"
        message = error_msg[:200]
        return title, message, None
    
    def notify_with_dedupe(
        self, timeframe: str, title: str, message: str, subtitle: Optional[str] = None
    ) -> bool:
        """Send notification with deduplication key.
        
        Useful for signal notifications to avoid spam when the same signal
        repeats in consecutive polling cycles.
        
        Args:
            timeframe: Timeframe identifier (for dedupe key)
            title: Notification title
            message: Notification message
            subtitle: Optional subtitle
        
        Returns:
            True if sent, False if deduplicated or skipped
        """
        if not self.enabled or not self.dedupe_enabled:
            return self.notify(title, message, subtitle)
        
        # Create dedupe key from title + message
        dedupe_key = f"{title}|{message}".replace("\n", "|")
        
        if self._last_signal_key.get(timeframe) == dedupe_key:
            logger.debug("[macos] Skipped duplicate signal (tf=%s)", timeframe)
            return False
        
        self._last_signal_key[timeframe] = dedupe_key
        return self.notify(title, message, subtitle)


def get_macos_notifier(cfg: Dict[str, Any]) -> Optional[MacOSNotifier]:
    """Factory function: create MacOSNotifier from config, or None if disabled.
    
    Args:
        cfg: Full raw config dict (cfg.raw)
    
    Returns:
        MacOSNotifier instance or None
    """
    alerts_cfg = cfg.get("alerts", {})
    macos_cfg = alerts_cfg.get("macos", {})
    
    if not alerts_cfg.get("enabled", False):
        logger.debug("[macos] Alerts disabled globally")
        return None
    
    if not macos_cfg.get("enabled", False):
        logger.debug("[macos] Disabled via config (or not on macOS)")
        return None
    
    try:
        rate_limit_cfg = macos_cfg.get("rate_limit", {})
        notifier = MacOSNotifier(
            enabled=True,
            method=macos_cfg.get("method", "auto"),
            title_default=macos_cfg.get("title", "Trading Bot"),
            app_name=macos_cfg.get("app_name", "ETH Analyzer"),
            sound=macos_cfg.get("sound", ""),
            timeout_seconds=float(macos_cfg.get("timeout_seconds", 2.0)),
            rate_limit_enabled=rate_limit_cfg.get("enabled", True),
            rate_limit_min_interval=float(rate_limit_cfg.get("min_interval_seconds", 1.0)),
            dedupe_enabled=macos_cfg.get("dedupe", {}).get("enabled", True),
        )
        
        # Return None if platform check disabled it
        if not notifier.enabled:
            return None
        
        return notifier
    except Exception as e:
        logger.error("[macos] Failed to initialize notifier: %s", e)
        return None


# CLI test entrypoint
if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Test macOS notifier")
    ap.add_argument("--test", default="Test notification", help="Message to send")
    args = ap.parse_args()
    
    logging.basicConfig(level=logging.DEBUG)
    
    notifier = MacOSNotifier(enabled=True, method="auto")
    if notifier.enabled:
        print(f"[test] Sending: {args.test}")
        notifier.notify("Test", args.test, "macOS Notifier Test")
        time.sleep(1)
        print("[test] Done")
    else:
        print("[test] macOS notifier disabled (not on macOS?)")
