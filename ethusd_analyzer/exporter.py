#!/usr/bin/env python3

from pathlib import Path
import psycopg2
from psycopg2 import sql


CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "goldbot",
    "username": "mohammadabwini",
    "password": "",
    "schema": "ethusd_analytics",
    "tables": [
         "candles",
         "signal_recommendations",
         "corr_results",
         "lagcorr_results",
         "sentiment_ticks",
         "snapshots",
         "rollingcorr_points",
         "calibration_runs",
         "meta_model_runs",
         "strategy_equity",
         "strategy_runs",
         "strategy_trades"
    ],
    "output_dir": "exports"
}


def main():
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = psycopg2.connect(
        host=CONFIG["host"],
        port=CONFIG["port"],
        dbname=CONFIG["database"],
        user=CONFIG["username"],
        password=CONFIG["password"]
    )

    try:
        with conn.cursor() as cur:
            for table_name in CONFIG["tables"]:
                output_file = output_dir / f"{table_name}.csv"

                copy_sql = sql.SQL(
                    "COPY (SELECT * FROM {}.{}) TO STDOUT WITH CSV HEADER"
                ).format(
                    sql.Identifier(CONFIG["schema"]),
                    sql.Identifier(table_name)
                )

                try:
                    with open(output_file, "w", encoding="utf-8", newline="") as f:
                        cur.copy_expert(copy_sql, f)
                    print(f"Exported: {CONFIG['schema']}.{table_name} -> {output_file}")
                except Exception as e:
                    print(f"Failed to export {CONFIG['schema']}.{table_name}: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()