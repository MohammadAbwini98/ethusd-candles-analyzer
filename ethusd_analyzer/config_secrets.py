"""Secure config-loading layer: resolves secrets from environment variables.

All sensitive values are loaded from environment variables.  Config files
should contain only non-sensitive defaults and env-var key names.
Missing *required* secrets produce a clear error message that never leaks
the actual value.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Regex to detect strings that look like real secrets (not placeholders).
_PLACEHOLDER_RE = re.compile(
    r"^(YOUR_|CHANGE_ME|REPLACE|placeholder|xxx|___)", re.IGNORECASE
)

# Env-var mapping: (config_path, env_var_name, required)
_SECRET_MAP: List[tuple] = [
    # Capital.com
    ("capital.api_key", "CAPITAL_API_KEY", False),
    ("capital.email", "CAPITAL_EMAIL", False),
    ("capital.password", "CAPITAL_PASSWORD", False),
    # Database
    ("db.password", "DB_PASSWORD", False),
    # Telegram
    ("alerts.telegram.bot_token", "TELEGRAM_BOT_TOKEN", False),
    # Email
    ("alerts.email.from_password", "EMAIL_PASSWORD", False),
]

# Keys whose values must be redacted in logs/debug output
REDACT_KEYS = frozenset({
    "api_key", "password", "from_password", "bot_token",
    "cst", "security_token", "X-SECURITY-TOKEN", "CST",
    "X-CAP-API-KEY", "Authorization",
})

# Patterns to redact from log messages (token-like strings)
_REDACT_PATTERNS = [
    re.compile(r"(\d{8,12}:[A-Za-z0-9_-]{30,})"),  # Telegram bot tokens
    re.compile(r"(cbmr[a-z]{12})"),                   # Gmail app passwords
]


def _deep_get(d: Dict[str, Any], dotpath: str) -> Any:
    """Retrieve a nested dict value via dotted path."""
    keys = dotpath.split(".")
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _deep_set(d: Dict[str, Any], dotpath: str, value: Any) -> None:
    """Set a nested dict value via dotted path, creating intermediate dicts."""
    keys = dotpath.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def resolve_secrets(cfg: Dict[str, Any]) -> List[str]:
    """Overlay environment-variable secrets onto the raw config dict (in-place).

    Returns a list of warning messages for missing required secrets.
    """
    warnings: List[str] = []

    for dotpath, env_var, required in _SECRET_MAP:
        env_val = os.environ.get(env_var, "").strip()
        if env_val:
            _deep_set(cfg, dotpath, env_val)
            logger.debug("Secret %s loaded from env var %s", dotpath, env_var)
        else:
            existing = _deep_get(cfg, dotpath)
            if existing and isinstance(existing, str) and existing.strip():
                # Value already in config (legacy support) - warn but allow
                if not _PLACEHOLDER_RE.match(existing):
                    logger.warning(
                        "Secret '%s' loaded from config file. "
                        "Migrate to env var %s for production safety.",
                        dotpath, env_var,
                    )
            elif required:
                msg = f"Required secret missing: set env var {env_var} (config path: {dotpath})"
                warnings.append(msg)
                logger.error(msg)

    return warnings


def redact_value(key: str, value: Any) -> Any:
    """Return a redacted version of *value* if *key* is sensitive."""
    if not isinstance(value, str) or not value:
        return value
    low = key.lower().rstrip("_")
    for rk in REDACT_KEYS:
        if rk.lower() in low:
            if len(value) > 8:
                return f"{value[:3]}***{value[-3:]}"
            return "***"
    return value


def redact_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of *cfg* with all sensitive values masked."""
    out: Dict[str, Any] = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            out[k] = redact_config(v)
        elif isinstance(v, list):
            out[k] = v  # lists (e.g. chat_ids) are not redacted per-element
        else:
            out[k] = redact_value(k, v)
    return out


def redact_string(s: str) -> str:
    """Redact known secret patterns from a free-form string (for logs)."""
    result = s
    for pat in _REDACT_PATTERNS:
        result = pat.sub("***REDACTED***", result)
    return result


def validate_no_hardcoded_secrets(cfg: Dict[str, Any]) -> List[str]:
    """Check config for values that look like real secrets rather than placeholders.

    Returns list of warning strings.  Does NOT prevent startup — just warns.
    """
    issues: List[str] = []
    for dotpath, env_var, _req in _SECRET_MAP:
        val = _deep_get(cfg, dotpath)
        if val and isinstance(val, str) and val.strip():
            if not _PLACEHOLDER_RE.match(val) and not os.environ.get(env_var):
                issues.append(
                    f"Config '{dotpath}' appears to contain a real secret. "
                    f"Move to env var {env_var}."
                )
    return issues
