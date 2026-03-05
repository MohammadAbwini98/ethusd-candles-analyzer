"""Email Notifier for trading alerts.

Sends alerts via SMTP with retry logic and graceful degradation.
"""

import html
import logging
import os
import smtplib
import threading
import time
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class EmailNotifier:
    """Send email notifications via SMTP with retry logic.
    
    Features:
    - Multiple recipients
    - Exponential backoff retry
    - HTML formatted messages
    - Silent failures (logs warnings, doesn't crash)
    """
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        from_email: str,
        from_password: str,
        to_emails: List[str],
        use_tls: bool = True,
        subject_prefix: str = "[ETH Analyzer]",
        timeout_seconds: float = 10.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        """Initialize email notifier.
        
        Args:
            smtp_server: SMTP server hostname (e.g., smtp.gmail.com)
            smtp_port: SMTP port (587 for TLS, 465 for SSL)
            from_email: Sender email address
            from_password: Sender email password (app-specific for Gmail)
            to_emails: List of recipient email addresses
            use_tls: Use STARTTLS (set False for SSL)
            subject_prefix: Prefix for email subjects
            timeout_seconds: SMTP connection timeout
            max_retries: Max retry attempts for transient failures
            retry_backoff_seconds: Base backoff multiplier (exponential)
        """
        if not from_email or not from_password or not to_emails:
            raise ValueError("from_email, from_password, and to_emails are required")
        
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.from_email = from_email
        self.from_password = from_password
        self.to_emails = to_emails
        self.use_tls = use_tls
        self.subject_prefix = subject_prefix
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        
        self._lock = threading.Lock()
    
    def _send_smtp(
        self, subject: str, html_body: str, retry_count: int = 0
    ) -> bool:
        """Send email via SMTP with retry logic.
        
        Args:
            subject: Email subject
            html_body: HTML email body
            retry_count: Current retry attempt
        
        Returns:
            True if successful, False on persistent failure
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)
            msg["Subject"] = f"{self.subject_prefix} {subject}"
            
            # Plain text fallback (strip HTML tags)
            plain_text = html_body.replace("<b>", "").replace("</b>", "")
            plain_text = plain_text.replace("<code>", "").replace("</code>", "")
            plain_text = plain_text.replace("<i>", "").replace("</i>", "")
            plain_text = plain_text.replace("<br>", "\n").replace("<br/>", "\n")
            
            part1 = MIMEText(plain_text, "plain")
            part2 = MIMEText(html_body, "html")
            msg.attach(part1)
            msg.attach(part2)
            
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout_seconds)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=self.timeout_seconds)
            
            server.login(self.from_email, self.from_password)
            server.sendmail(self.from_email, self.to_emails, msg.as_string())
            server.quit()
            
            logger.debug(
                "[email] Sent to %d recipients: %s",
                len(self.to_emails), subject
            )
            return True
            
        except smtplib.SMTPException as e:
            if retry_count < self.max_retries:
                backoff = self.retry_backoff_seconds * (2 ** retry_count)
                logger.warning(
                    "[email] SMTP error, retrying in %.1fs (attempt %d/%d): %s",
                    backoff, retry_count + 1, self.max_retries, e
                )
                time.sleep(backoff)
                return self._send_smtp(subject, html_body, retry_count + 1)
            else:
                logger.error(
                    "[email] Failed to send after %d retries: %s",
                    self.max_retries, e
                )
                return False
                
        except Exception as e:
            logger.error("[email] Unexpected error: %s", e, exc_info=True)
            return False
    
    def send_message(self, subject: str, html_body: str) -> bool:
        """Send email notification (thread-safe).
        
        Args:
            subject: Email subject (prefix will be added)
            html_body: HTML formatted message body
        
        Returns:
            True if sent successfully
        """
        with self._lock:
            return self._send_smtp(subject, html_body)
    
    @staticmethod
    def format_startup_message(symbol: str = "ETH", mode: str = "LIVE") -> tuple:
        """Format a startup notification (HTML).
        
        Returns:
            (subject, html_body) tuple
        """
        now_utc = datetime.now(timezone.utc)
        subject = f"🟢 Analyzer STARTED - {symbol}"
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #22c55e;">🟢 ETH Analyzer STARTED</h2>
            <p><b>Symbol:</b> <code>{html.escape(symbol)}</code> &nbsp;|&nbsp; <b>Mode:</b> {html.escape(mode)}</p>
            <p><b>Time:</b> {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        </body>
        </html>
        """
        return subject, html_body
    
    @staticmethod
    def format_shutdown_message(
        symbol: str = "ETH",
        reason: str = "normal",
        error_msg: Optional[str] = None,
    ) -> tuple:
        """Format a shutdown notification (HTML).
        
        Returns:
            (subject, html_body) tuple
        """
        now_utc = datetime.now(timezone.utc)
        subject = f"🔴 Analyzer STOPPED - {symbol} ({reason})"
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #ef4444;">🔴 ETH Analyzer STOPPED</h2>
            <p><b>Symbol:</b> <code>{html.escape(symbol)}</code> &nbsp;|&nbsp; <b>Reason:</b> {html.escape(reason)}</p>
            <p><b>Time:</b> {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
        """
        if error_msg:
            html_body += f"<p><b>Error:</b> <code>{html.escape(error_msg[:200])}</code></p>"
        html_body += "</body></html>"
        return subject, html_body
    
    @staticmethod
    def format_signal_message(
        symbol: str,
        timeframe: str,
        signal: str,
        regime: str,
        confidence: float,
        entry_price: float,
        take_profit: float,
        stop_loss: float,
        hold_bars: int,
        hold_minutes: int,
        rc: Optional[float] = None,
        ar: Optional[float] = None,
        volatility: Optional[float] = None,
        price_z: Optional[float] = None,
        trend_strength: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> tuple:
        """Format a signal notification (HTML).
        
        Returns:
            (subject, html_body) tuple
        """
        ts = timestamp or datetime.now(timezone.utc)
        risk_pct = abs((stop_loss - entry_price) / entry_price * 100)
        reward_pct = abs((take_profit - entry_price) / entry_price * 100)
        rr_ratio = reward_pct / risk_pct if risk_pct > 0 else 0.0
        
        signal_color = "#22c55e" if signal == "BUY" else "#ef4444"
        signal_emoji = "🟢" if signal == "BUY" else "🔴"
        
        subject = f"{signal_emoji} {signal} Signal - {symbol} {timeframe}"
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: {signal_color};">{signal_emoji} {signal} SIGNAL</h2>
            <table style="border-collapse: collapse; width: 100%; max-width: 600px;">
                <tr style="background-color: #f3f4f6;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Symbol</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><code>{html.escape(symbol)}</code></td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Timeframe</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;"><code>{html.escape(timeframe)}</code></td>
                </tr>
                <tr style="background-color: #f3f4f6;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Regime</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{html.escape(regime)}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Confidence</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{confidence:.1%}</td>
                </tr>
                <tr style="background-color: #f3f4f6;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Entry</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">${entry_price:.2f}</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Take Profit</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">${take_profit:.2f} (+{reward_pct:.2f}%)</td>
                </tr>
                <tr style="background-color: #f3f4f6;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Stop Loss</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">${stop_loss:.2f} (-{risk_pct:.2f}%)</td>
                </tr>
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Risk/Reward</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{rr_ratio:.2f}:1</td>
                </tr>
                <tr style="background-color: #f3f4f6;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Hold Time</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{hold_bars} bars (~{hold_minutes} min)</td>
                </tr>
        """
        
        # Add optional metrics
        if rc is not None:
            html_body += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>RC</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{rc:.3f}</td>
                </tr>
            """
        
        if ar is not None:
            html_body += f"""
                <tr style="background-color: #f3f4f6;">
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>AR</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{ar:.3f}</td>
                </tr>
            """
        
        if volatility is not None:
            html_body += f"""
                <tr>
                    <td style="padding: 8px; border: 1px solid #ddd;"><b>Volatility</b></td>
                    <td style="padding: 8px; border: 1px solid #ddd;">{volatility:.4f}</td>
                </tr>
            """
        
        html_body += """
            </table>
        """
        
        html_body += f"""
            <p style="color: #6b7280; font-size: 0.875rem; margin-top: 16px;">
                <i>Generated at {ts.strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
            </p>
        </body>
        </html>
        """
        
        return subject, html_body
    
    @staticmethod
    def format_error_message(error_type: str, error_msg: str) -> tuple:
        """Format an error notification (HTML).
        
        Returns:
            (subject, html_body) tuple
        """
        now_utc = datetime.now(timezone.utc)
        subject = f"⚠️ Error: {error_type}"
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2 style="color: #ef4444;">⚠️ ERROR</h2>
            <p><b>Type:</b> {html.escape(error_type)}</p>
            <p><b>Message:</b></p>
            <pre style="background-color: #f3f4f6; padding: 12px; border-radius: 4px; overflow-x: auto;">{html.escape(error_msg[:500])}</pre>
            <p style="color: #6b7280; font-size: 0.875rem;">
                <i>{now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC</i>
            </p>
        </body>
        </html>
        """
        return subject, html_body


def get_email_notifier(cfg: Dict[str, Any]) -> Optional[EmailNotifier]:
    """Factory function: create EmailNotifier from config, or None if disabled/invalid.
    
    Args:
        cfg: Full raw config dict (cfg.raw)
    
    Returns:
        EmailNotifier instance or None
    """
    alerts_cfg = cfg.get("alerts", {})
    email_cfg = alerts_cfg.get("email", {})

    # Each channel is controlled by its own enabled flag.
    # The top-level alerts.enabled only gates WhatsApp (the sidecar).
    if not email_cfg.get("enabled", False):
        logger.debug("[email] Disabled via config (email.enabled=false)")
        return None
    
    # Resolve password from env var
    password: Optional[str] = None
    if email_cfg.get("from_password_env"):
        env_var = email_cfg["from_password_env"]
        password = os.environ.get(env_var, "").strip()
        if password:
            logger.debug("[email] Password loaded from env var %s", env_var)
        else:
            logger.warning("[email] Password env var '%s' not set or empty", env_var)
    
    if not password and email_cfg.get("from_password"):
        password = str(email_cfg["from_password"]).strip()
        logger.debug("[email] Password loaded from config (not recommended)")
    
    if not password:
        logger.warning(
            "[email] No password found. Set EMAIL_PASSWORD env var or "
            "add alerts.email.from_password to config.yaml"
        )
        return None
    
    from_email = email_cfg.get("from_email", "").strip()
    to_emails = email_cfg.get("to_emails", [])
    
    # Handle both list and single string format for to_emails
    if isinstance(to_emails, str):
        to_emails = [to_emails]
    
    if not from_email or not to_emails:
        logger.warning("[email] from_email or to_emails not configured")
        return None
    
    try:
        notifier = EmailNotifier(
            smtp_server=email_cfg.get("smtp_server", "smtp.gmail.com"),
            smtp_port=int(email_cfg.get("smtp_port", 587)),
            from_email=from_email,
            from_password=password,
            to_emails=to_emails,
            use_tls=bool(email_cfg.get("use_tls", True)),
            subject_prefix=email_cfg.get("subject_prefix", "[ETH Analyzer]"),
        )
        logger.info("[email] Notifier initialized for %d recipients", len(to_emails))
        return notifier
    except Exception as e:
        logger.error("[email] Failed to initialize notifier: %s", e)
        return None
