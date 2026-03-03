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
