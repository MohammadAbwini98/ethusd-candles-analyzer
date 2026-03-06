#!/usr/bin/env python3
"""Test macOS notification integration.

Run directly: python test_macos_notifier.py
NOT a pytest test — all executable logic is under __main__.
"""


def main() -> None:
    from ethusd_analyzer.utils import load_config
    from ethusd_analyzer.telegram_notifier import get_telegram_notifier
    from ethusd_analyzer.email_notifier import get_email_notifier
    from ethusd_analyzer.macos_notifier import get_macos_notifier

    cfg = load_config('config.yaml')

    print('='*60)
    print('NOTIFICATION SYSTEM STATUS')
    print('='*60)

    tg = get_telegram_notifier(cfg.raw)
    print(f'Telegram: {"Ready" if tg else "Disabled"}')

    email = get_email_notifier(cfg.raw)
    print(f'Email: {"Ready" if email else "Disabled"}')

    macos = get_macos_notifier(cfg.raw)
    status = "Ready (" + macos.method + ")" if macos and macos.enabled else "Disabled"
    print(f'macOS: {status}')

    print('='*60)
    print('MESSAGE FORMAT EXAMPLES')
    print('='*60)

    if macos:
        title, msg, subtitle = macos.format_startup_message('ETHUSD', 'LIVE')
        print(f'Startup:\n  {title}\n  {msg}\n')

        title, msg, subtitle = macos.format_signal_message(
            'ETHUSD', '15m', 'BUY', 0.85, 1970.50, 1995.00, 1950.00, 3
        )
        print(f'Signal:\n  {title}\n  {msg}\n  Subtitle: {subtitle}\n')

        title, msg, subtitle = macos.format_shutdown_message('ETHUSD', 'normal')
        print(f'Shutdown:\n  {title}\n  {msg}')

    print('='*60)


if __name__ == "__main__":
    main()
