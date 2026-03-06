#!/usr/bin/env python3
"""Test macOS shutdown notification blocking.

Run directly: python test_shutdown_notification.py
NOT a pytest test — all executable logic is under __main__.
"""
import sys
import time


def main() -> None:
    from ethusd_analyzer.utils import load_config
    from ethusd_analyzer.macos_notifier import get_macos_notifier

    cfg = load_config('config.yaml')
    macos = get_macos_notifier(cfg.raw)

    if not macos or not macos.enabled:
        print("[test] macOS notifier not available")
        sys.exit(1)

    print("[test] Testing synchronous (blocking) shutdown notification...")
    print("[test] This should block until the notification is sent...")

    start = time.time()
    title, message, subtitle = macos.format_shutdown_message("ETH", "test", None)
    result = macos.notify_sync(title, message, subtitle)
    elapsed = time.time() - start

    print(f"[test] Result: {result} (took {elapsed:.2f}s)")
    print("[test] Shutdown notification sent successfully!")


if __name__ == "__main__":
    main()
