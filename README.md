# ETHUSD Candles Analyzer (PostgreSQL) — Streaming / Continuous Analysis (v1.1)

This project continuously fetches candle rows from your PostgreSQL database (filtered by `epic='ETHUSD'`)
and runs analytics on a rolling window until you stop it (Ctrl+C).

## Fixes in v1.1
- Supports time columns that are **BIGINT epoch** (seconds / milliseconds / microseconds) **or** timestamp/timestamptz.
- Auto-detects epoch unit by magnitude and converts to `market_time` timestamp internally.

## What it does
- Continuously polls `candles` for new rows where time > last_seen_time.
- Filters strictly: `epic = 'ETHUSD'`
- Builds features (robust when buyers/sellers variance is small):
  - log returns, imbalance, **imbalance_change**, z-scores (window 50), interaction, score
- For each timeframe (1m + optional resamples 5m/15m):
  - Correlations vs forward returns (h=1,2)
  - Lag-correlation sweep (-12..+12)
  - Rolling correlations (windows 20/50)
- Saves outputs to `outputs/` periodically.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure
Edit `config.yaml` to match your DB schema.
Mandatory filter is already set:
- `epic: ETHUSD`

If your time column is `ts` (bigint), set:
- `columns.time: ts`

## Run
```bash
python -m ethusd_analyzer.run --config config.yaml
```

Stop with Ctrl+C.

## Inspect schema (optional)
```bash
python -m ethusd_analyzer.inspect_schema --config config.yaml
```

## Outputs
Inside `outputs/`:
- `summary_<tf>.json`
- `corr_<tf>.csv`
- `lagcorr_<tf>.csv`
- `rollingcorr_<tf>.csv`


## Web Dashboard (auto)
On startup, the program starts a local dashboard (default http://127.0.0.1:8787/) and opens your browser.

## DB output
All outputs are also written to Postgres under schema `ethusd_analytics`.


cd /Users/mohammadabwini/Desktop/Workplace/ethusd_candles_analyzer_v1_2
.venv/bin/python -m ethusd_analyzer.run --config config.yaml



# ==============================================================
# GoldBot — XAUUSD Trend-Following Bot
# ==============================================================

# ---- Capital.com API Credentials ----
CAPITAL_API_KEY=b7ZKfX5nISJiJyYf
CAPITAL_EMAIL=mohammad.abwini98@gmail.com
CAPITAL_PASSWORD=/Mohammad6598

# ---- Account Type ----
# "live" or "demo"
ACCOUNT_TYPE=live

# ---- Account ID (optional) ----
# Leave blank to use the preferred/default account.
# Set to your demo account ID to run in demo mode.
# Your accounts:
#   307978579944232132 → Live account      ($25.67)
#   308290261431431454 → Demo - Account    ($993.90)
#   310891246381248798 → Demo - CFD Account ($1,000.00)  ← active in browser
CAPITAL_ACCOUNT_ID=307978579944232132

# ---- Instrument ----
# Capital.com epic name. Examples: GOLD, SILVER, BTCUSD, ETHUSD, OIL_CRUDE, EURUSD, US500
INSTRUMENT=ETHUSD

# ---- Spread Filter ----
# Max allowed spread in price units (tune per instrument)
# GOLD=0.60 | ETHUSD=3.00 | BTCUSD=30.00 | SILVER=0.05 | EURUSD=0.0003
SPREAD_MAX=3.00

# ---- Strategy Toggles ----
# Enable H1+H4 swing mode (in addition to M5 scalp)
SWING_ENABLED=true

cd /Users/mohammadabwini/Desktop/Workplace/ethusd_candles_analyzer_v1_2
source .venv/bin/activate
export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN_HERE"
# ---- Telegram Notifications ----
# Bot token from @BotFather (set via TELEGRAM_BOT_TOKEN env var)
# Chat ID to receive messages. Use your personal numeric ID (get it from @userinfobot),
# or a public channel/group username like @mychannel where the bot is admin.
TELEGRAM_CHAT_ID=1495017760

# ---- Database (optional — ML data logging) ----
# Leave blank to run without a database (trading is unaffected).
# Example: postgresql://user:pass@localhost:5432/goldbot
DB_URL=postgresql://mohammadabwini@localhost:5432/goldbot

# ---- ML Confidence Gate (optional) ----
# Applied after BOS + M1 micro-confirm. No-op until models/current.json exists.
# BUY  entry: require p(up) >= ML_BUY_THRESHOLD
# SELL entry: require p(up) <= ML_SELL_THRESHOLD  (i.e. p(down) >= 1 - threshold)
ML_BUY_THRESHOLD=0.60
ML_SELL_THRESHOLD=0.40

export TELEGRAM_BOT_TOKEN="YOUR_BOT_TOKEN_HERE" && export PYTHONPATH="/Users/mohammadabwini/Desktop/Workplace/ethusd_candles_analyzer_v1_2" && /Users/mohammadabwini/Desktop/Workplace/ethusd_candles_analyzer_v1_2/.venv/bin/python -m ethusd_analyzer.telegram_notifier --test "hello" --config /Users/mohammadabwini/Desktop/Workplace/ethusd_candles_analyzer_v1_2/config.yaml


export TELEGRAM_BOT_TOKEN="YOUR_NEW_BOT_TOKEN" && export PYTHONPATH="/Users/mohammadabwini/Desktop/Workplace/ethusd_candles_analyzer_v1_2:$PYTHONPATH" && /Users/mohammadabwini/Desktop/Workplace/ethusd_candles_analyzer_v1_2/.venv/bin/python -m ethusd_analyzer.telegram_notifier --test "hello" --config /Users/mohammadabwini/Desktop/Workplace/ethusd_candles_analyzer_v1_2/config.yaml



Notification Coverage Matrix
Event Type	WhatsApp	Telegram	Email	macOS
Startup	✅ (simple)	✅ (rich)	✅ (rich)	✅ (rich)
Shutdown	✅ (simple)	✅ (rich)	✅ (rich)	✅ (sync)
Signal Fired	✅ (simple)	✅ (rich)	✅ (rich)	✅ (rich)
Sanity Check Fail	✅	✅	✅	✅
Calibration Warning	✅	✅	✅	✅
System Error	✅	✅	✅	✅
Daily Summary	✅	✅	✅	✅
