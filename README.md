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


## Secrets / Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env
# Edit .env with your values, then source it:
source .env
# Or export individually:
export CAPITAL_API_KEY="your_key"
export CAPITAL_EMAIL="your@email.com"
export CAPITAL_PASSWORD="your_password"
export TELEGRAM_BOT_TOKEN="your_bot_token"
export EMAIL_PASSWORD="your_app_password"
```

See `.env.example` for the full list of supported environment variables.

## Notification Coverage Matrix
Event Type	WhatsApp	Telegram	Email	macOS
Startup	✅ (simple)	✅ (rich)	✅ (rich)	✅ (rich)
Shutdown	✅ (simple)	✅ (rich)	✅ (rich)	✅ (sync)
Signal Fired	✅ (simple)	✅ (rich)	✅ (rich)	✅ (rich)
Sanity Check Fail	✅	✅	✅	✅
Calibration Warning	✅	✅	✅	✅
System Error	✅	✅	✅	✅
Daily Summary	✅	✅	✅	✅


Because this line disables it globally:

adaptive:
  enabled: false

With that value, the system stays in static-only mode even if all individual adaptive parameters are marked enabled: true.

How it works:

adaptive.enabled: false
= master switch off
= adaptive layer is not used in decisions

adaptive.params.<name>.enabled: true
= “this parameter is allowed to adapt” only if the global switch is on

So your current config means:

adaptive framework exists

parameter-level adaptation is configured

but the whole adaptive engine is intentionally disabled

Why this is usually done by default:

to preserve old behavior after deployment

to avoid changing live trading immediately

to let you test safely first

to use shadow_mode before activating real adaptive decisions

Your current state is effectively:

static mode

no adaptive values applied to entries/exits

likely no adaptive logs either, because shadow_mode: false

If you want adaptive logic active

Change it to:

adaptive:
  enabled: true
  shadow_mode: false

That means:

compute adaptive values

use them in real decisions

Safer first step

Use:

adaptive:
  enabled: true
  shadow_mode: true

That means:

compute adaptive values

log them

but keep live decisions on static baseline

This is usually the best first rollout step.

When each mode is useful
enabled: false, shadow_mode: false

Use when:

you want original behavior only

you are not testing adaptation yet

enabled: true, shadow_mode: true

Use when:

you want to inspect adaptive values

compare static vs adaptive behavior

validate logs and formulas safely

enabled: true, shadow_mode: false

Use when:

you already validated the adaptive layer

you want adaptive values to affect trading decisions

Important note

If the code was written correctly, the expected priority is:

if adaptive.enabled == false → ignore adaptive decisioning

if adaptive.enabled == true and shadow_mode == true → compute/log only

if adaptive.enabled == true and shadow_mode == false → compute and apply

So the reason it is “turned off” is not a bug by itself — it is a deliberate default safety setting.

Recommended next step for you:

adaptive:
  enabled: true
  shadow_mode: true

Run it first, inspect logs and signal outputs, then switch shadow_mode to false once the behavior looks correct.

If you want, I can also tell you exactly what logs/output you should expect in shadow mode versus fully active adaptive mode.