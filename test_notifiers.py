#!/usr/bin/env python3
"""Test notifier initialization.

Run directly: python test_notifiers.py
NOT a pytest test — all executable logic is under __main__.
"""
import os


def main() -> None:
    from ethusd_analyzer.utils import load_config
    from ethusd_analyzer.telegram_notifier import get_telegram_notifier
    from ethusd_analyzer.email_notifier import get_email_notifier

    cfg = load_config('config.yaml')
    print('Config loaded')

    # Test Telegram
    tg = get_telegram_notifier(cfg.raw)
    print(f'Telegram notifier: {"Initialized" if tg else "Not configured"}')
    if tg:
        print(f'  - Chat IDs: {len(tg.chat_ids)} configured')

    # Test Email
    print()
    email_cfg = cfg.raw.get('alerts', {}).get('email', {})
    print('Email configuration:')
    print(f'  - Enabled: {email_cfg.get("enabled", False)}')
    print(f'  - From: {"set" if email_cfg.get("from_email") else "not set"}')
    print(f'  - To: {len(email_cfg.get("to_emails", []))} recipients')
    pwd_env = email_cfg.get("from_password_env", "EMAIL_PASSWORD")
    print(f'  - Env var set: {"YES" if os.environ.get(pwd_env) else "NO"}')

    email = get_email_notifier(cfg.raw)
    print(f'Email notifier: {"Initialized" if email else "Not configured"}')


if __name__ == "__main__":
    main()
