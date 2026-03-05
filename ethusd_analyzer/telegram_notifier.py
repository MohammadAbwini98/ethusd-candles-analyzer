"""Telegram Bot Notifier for trading alerts.

Sends alerts to Telegram via bot API with retry logic, rate limiting, and graceful degradation.
Complements the WhatsApp notifier without replacing it.
"""

import logging
import os
import time
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send Telegram messages via bot API with retry logic and rate limiting.
    
    Features:
    - Broadcast to multiple chat IDs
    - Exponential backoff retry
    - Per-process rate limiting (min_interval_seconds)
    - Non-blocking (thread-based for startup/shutdown messages)
    - Silent failures (logs warnings, doesn't crash)
    """
    
    API_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"
    
    def __init__(
        self,
        bot_token: str,
        chat_ids: List[int],
        timeout_seconds: float = 5.0,
        parse_mode: str = "HTML",
        disable_web_page_preview: bool = True,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
        min_interval_seconds: float = 1.0,
        rate_limit_enabled: bool = True,
    ):
        """Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (from BotFather)
            chat_ids: List of recipient chat IDs (can be personal or group)
            timeout_seconds: HTTP request timeout
            parse_mode: "HTML" or "Markdown"
            disable_web_page_preview: Prevent link previews
            max_retries: Max retry attempts for transient failures
            retry_backoff_seconds: Base backoff multiplier (exponential)
            min_interval_seconds: Minimum seconds between messages (rate limit)
            rate_limit_enabled: Enable/disable local in-process rate limiting
        """
        if not bot_token or not chat_ids:
            raise ValueError("bot_token and chat_ids are required")
        
        self.bot_token = bot_token
        self.chat_ids = chat_ids
        self.timeout_seconds = timeout_seconds
        self.parse_mode = parse_mode
        self.disable_web_page_preview = disable_web_page_preview
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.min_interval_seconds = min_interval_seconds
        self.rate_limit_enabled = rate_limit_enabled
        
        # Rate limiting state (per-process in-memory)
        self._last_send_time: float = 0.0
        self._lock = threading.Lock()
    
    def _log_token_safe(self) -> str:
        """Return a safe token representation for logging."""
        if len(self.bot_token) > 20:
            return f"{self.bot_token[:10]}...{self.bot_token[-5:]}"
        return "***"
    
    def _post_send_message(
        self, chat_id: int, text: str, retry_count: int = 0
    ) -> Optional[Dict[str, Any]]:
        """Post a single message to one chat ID with retry logic.
        
        Args:
            chat_id: Recipient chat ID
            text: Message text (HTML or Markdown per parse_mode)
            retry_count: Current retry attempt
        
        Returns:
            Response dict if successful, None on persistent failure
        """
        url = self.API_URL_TEMPLATE.format(token=self.bot_token)
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": self.disable_web_page_preview,
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_seconds)
            
            if resp.status_code == 200:
                logger.debug(
                    "[telegram] Successfully sent to chat_id=%s (token=%s)",
                    chat_id, self._log_token_safe()
                )
                return resp.json()
            
            # Transient errors: 429 (rate limit), 5xx (server error)
            if resp.status_code in (429, 500, 502, 503, 504) and retry_count < self.max_retries:
                backoff = self.retry_backoff_seconds ** (retry_count + 1)
                logger.debug(
                    "[telegram] Transient error %d for chat_id=%s, retrying in %.1fs...",
                    resp.status_code, chat_id, backoff
                )
                time.sleep(backoff)
                return self._post_send_message(chat_id, text, retry_count + 1)
            
            # Permanent or final transient error
            try:
                error_body = resp.json()
                error_desc = error_body.get("description", "unknown error")
            except Exception:
                error_desc = resp.text[:200] if resp.text else "no response body"
            
            logger.warning(
                "[telegram] Failed to send to chat_id=%s: %d %s (token=%s)",
                chat_id, resp.status_code, error_desc, self._log_token_safe()
            )
            return None
        
        except requests.exceptions.Timeout:
            if retry_count < self.max_retries:
                backoff = self.retry_backoff_seconds ** (retry_count + 1)
                logger.debug(
                    "[telegram] Timeout for chat_id=%s, retrying in %.1fs...",
                    chat_id, backoff
                )
                time.sleep(backoff)
                return self._post_send_message(chat_id, text, retry_count + 1)
            logger.warning(
                "[telegram] Timeout (after %d retries) for chat_id=%s (token=%s)",
                self.max_retries, chat_id, self._log_token_safe()
            )
            return None
        
        except requests.exceptions.ConnectionError as e:
            if retry_count < self.max_retries:
                backoff = self.retry_backoff_seconds ** (retry_count + 1)
                logger.debug(
                    "[telegram] Connection error for chat_id=%s, retrying in %.1fs...",
                    chat_id, backoff
                )
                time.sleep(backoff)
                return self._post_send_message(chat_id, text, retry_count + 1)
            logger.warning(
                "[telegram] Connection failed (after %d retries) for chat_id=%s: %s (token=%s)",
                self.max_retries, chat_id, str(e)[:100], self._log_token_safe()
            )
            return None
        
        except Exception as e:
            logger.warning(
                "[telegram] Unexpected error sending to chat_id=%s: %s (token=%s)",
                chat_id, str(e)[:100], self._log_token_safe()
            )
            return None
    
    def send_message(self, text: str) -> Dict[int, Optional[Dict]]:
        """Send a message to all chat IDs with rate limiting.
        
        Returns:
            Dict mapping chat_id -> result (response dict or None)
        """
        # Rate limiting: enforce min_interval_seconds between any sends
        if self.rate_limit_enabled:
            with self._lock:
                now = time.time()
                elapsed = now - self._last_send_time
                if elapsed < self.min_interval_seconds:
                    sleep_time = self.min_interval_seconds - elapsed
                    logger.debug(
                        "[telegram] Rate limit: sleeping %.2fs before send",
                        sleep_time
                    )
                    time.sleep(sleep_time)
                self._last_send_time = time.time()
        
        results = {}
        for chat_id in self.chat_ids:
            results[chat_id] = self._post_send_message(chat_id, text)
        
        return results

    def send_message_to_all(self, text: str) -> Dict[int, Optional[Dict]]:
        """Broadcast helper alias for compatibility with integration code."""
        return self.send_message(text)
    
    @staticmethod
    def format_startup_message(symbol: str = "ETH", mode: str = "LIVE") -> str:
        """Format a startup notification (HTML)."""
        now_utc = datetime.now(timezone.utc)
        return (
            "<b>🟢 ETH Analyzer STARTED</b>\n"
            f"<b>Symbol:</b> <code>{symbol}</code>  |  <b>Mode:</b> {mode}\n"
            f"<b>Time:</b> {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
    
    @staticmethod
    def format_shutdown_message(
        symbol: str = "ETH",
        reason: str = "normal",
        error_msg: Optional[str] = None,
    ) -> str:
        """Format a shutdown notification (HTML)."""
        now_utc = datetime.now(timezone.utc)
        msg = (
            "<b>🔴 ETH Analyzer STOPPED</b>\n"
            f"<b>Symbol:</b> <code>{symbol}</code>  |  <b>Reason:</b> {reason}\n"
            f"<b>Time:</b> {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        if error_msg:
            msg += f"\n<b>Error:</b> <code>{error_msg[:200]}</code>"
        return msg
    
    @staticmethod
    def format_signal_message(
        symbol: str,
        timeframe: str,
        signal: str,  # "BUY", "SELL", "HOLD"
        regime: str,  # "MR", "MOM"
        confidence: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        hold_bars: int,
        hold_minutes: int,
        rc: float = float('nan'),
        ar: float = float('nan'),
        volatility: float = float('nan'),
        price_z: Optional[float] = None,
        trend_strength: Optional[float] = None,
        timestamp: Optional[datetime] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> str:
        """Format a signal notification (HTML).
        
        Args:
            symbol: Instrument (e.g. "ETHUSD")
            timeframe: Timeframe (e.g. "15m")
            signal: "BUY", "SELL", "HOLD"
            regime: "MR" (Mean Reversion) or "MOM" (Momentum)
            confidence: 0.0-1.0 confidence score
            entry_price: Entry level
            take_profit: TP level
            stop_loss: SL level
            hold_bars: Expected hold duration in bars
            hold_minutes: Expected hold duration in minutes
            rc, ar, volatility: Regime characteristics
            price_z: Price z-score (if available)
            trend_strength: Trend strength metric (if available)
            timestamp: Computation time (defaults to now UTC)
            bid, ask: Current bid/ask prices (for spread display)
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        emoji = "📈" if signal == "BUY" else "📉" if signal == "SELL" else "⏸️"
        msg = (
            f"<b>{emoji} Signal: {signal} ({regime})</b>\n"
            f"<b>Symbol:</b> <code>{symbol}</code>  <b>TF:</b> <code>{timeframe}</code>\n"
            f"<b>Confidence:</b> {confidence:.2%}\n"
        )
        
        # Bid/ask spread
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2
            spread = ask - bid
            msg += f"<b>Price:</b> {mid:.2f} (bid {bid:.2f} / ask {ask:.2f}, spread {spread:.2f})\n"
        
        # Entry / TP / SL
        msg += (
            f"<b>Entry:</b> {entry_price:.2f}\n"
            f"<b>TP:</b> {take_profit:.2f}   <b>SL:</b> {stop_loss:.2f}\n"
        )
        
        # Hold duration
        msg += f"<b>Hold:</b> {hold_bars} bars (~{hold_minutes}m)\n"
        
        # Metrics (if valid)
        import math
        metrics = []
        if not math.isnan(rc):
            metrics.append(f"rc={rc:.4f}")
        if not math.isnan(ar):
            metrics.append(f"ar={ar:.4f}")
        if not math.isnan(volatility):
            metrics.append(f"vol={volatility:.6f}")
        if price_z is not None and not math.isnan(price_z):
            metrics.append(f"price_z={price_z:.3f}")
        if trend_strength is not None and not math.isnan(trend_strength):
            metrics.append(f"trend={trend_strength:.6f}")
        
        if metrics:
            msg += f"<b>Metrics:</b> {' | '.join(metrics)}\n"
        
        msg += f"<b>Computed:</b> {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        return msg
    
    @staticmethod
    def format_error_message(title: str, error: str, symbol: str = "ETH") -> str:
        """Format an error notification (HTML)."""
        return (
            f"<b>⚠️ {title}</b>\n"
            f"<b>Symbol:</b> <code>{symbol}</code>\n"
            f"<b>Error:</b> <code>{error[:300]}</code>\n"
            f"<b>Time:</b> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )


def get_telegram_notifier(cfg: Dict[str, Any]) -> Optional[TelegramNotifier]:
    """Factory function: create TelegramNotifier from config, or None if disabled/invalid.
    
    Args:
        cfg: Full raw config dict (cfg.raw)
    
    Returns:
        TelegramNotifier instance or None
    """
    alerts_cfg = cfg.get("alerts", {})
    tg_cfg = alerts_cfg.get("telegram", {})
    
    if not alerts_cfg.get("enabled", False) or not tg_cfg.get("enabled", False):
        logger.debug("[telegram] Disabled via config")
        return None
    
    # Resolve bot token: env var first, then config, then None
    bot_token: Optional[str] = None
    if tg_cfg.get("bot_token_env"):
        env_var = tg_cfg["bot_token_env"]
        bot_token = os.environ.get(env_var, "").strip()
        if bot_token:
            logger.debug("[telegram] Token loaded from env var %s", env_var)
        else:
            logger.warning(
                "[telegram] Token env var '%s' not set or empty",
                env_var
            )
    
    if not bot_token and tg_cfg.get("bot_token"):
        bot_token = str(tg_cfg["bot_token"]).strip()
        logger.debug("[telegram] Token loaded from config (not recommended)")
    
    if not bot_token:
        logger.warning(
            "[telegram] No bot token found. Set TELEGRAM_BOT_TOKEN env var or "
            "add alerts.telegram.bot_token to config.yaml (NOT RECOMMENDED for production)"
        )
        return None
    
    chat_ids = tg_cfg.get("chat_ids", [])
    if not chat_ids:
        logger.warning("[telegram] No chat_ids configured")
        return None
    
    # Ensure chat_ids are ints
    try:
        chat_ids = [int(cid) for cid in chat_ids]
    except (TypeError, ValueError) as e:
        logger.error("[telegram] Invalid chat_ids: %s", e)
        return None
    
    try:
        notifier = TelegramNotifier(
            bot_token=bot_token,
            chat_ids=chat_ids,
            timeout_seconds=float(tg_cfg.get("timeout_seconds", 5.0)),
            parse_mode=str(tg_cfg.get("parse_mode", "HTML")),
            disable_web_page_preview=bool(tg_cfg.get("disable_web_page_preview", True)),
            max_retries=int(tg_cfg.get("max_retries", 3)),
            retry_backoff_seconds=float(tg_cfg.get("retry_backoff_seconds", 1.5)),
            min_interval_seconds=float(tg_cfg.get("rate_limit", {}).get("min_interval_seconds", 1.0)),
            rate_limit_enabled=bool(tg_cfg.get("rate_limit", {}).get("enabled", True)),
        )
        logger.info(
            "[telegram] Notifier initialized with %d chat_ids",
            len(chat_ids)
        )
        return notifier
    except Exception as e:
        logger.error("[telegram] Failed to initialize: %s", e)
        return None


if __name__ == "__main__":
    # Quick test: python -m ethusd_analyzer.telegram_notifier --test "hello world"
    import sys
    import yaml
    
    if "--test" in sys.argv:
        idx = sys.argv.index("--test")
        test_msg = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "Test message"
        config_path = "config.yaml"
        if "--config" in sys.argv:
            cidx = sys.argv.index("--config")
            if cidx + 1 < len(sys.argv):
                config_path = sys.argv[cidx + 1]

        cfg: Dict[str, Any] = {}
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        notifier = get_telegram_notifier(cfg)
        if notifier is None:
            print("ERROR: Telegram notifier is not configured. Check alerts.telegram and token/chat_ids.")
            sys.exit(1)

        logging.basicConfig(level=logging.DEBUG)

        print(f"[test] Initializing notifier: chats={len(notifier.chat_ids)}")
        print(f"[test] Sending message: {test_msg}")
        results = notifier.send_message_to_all(test_msg)

        print(f"[test] Results: {results}")
        success_count = sum(1 for r in results.values() if r is not None)
        print(f"[test] Delivered to {success_count}/{len(notifier.chat_ids)} chats")
        sys.exit(0 if success_count > 0 else 1)
