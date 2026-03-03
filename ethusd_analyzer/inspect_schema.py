from __future__ import annotations
import argparse
from .utils import load_config
from .db import DbConfig, make_engine, inspect_columns

def parse_args():
    ap = argparse.ArgumentParser(description="Inspect candles table schema (columns)")
    ap.add_argument("--config", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    db_cfg = DbConfig(
        host=cfg.db.get("host","localhost"),
        port=int(cfg.db.get("port",5432)),
        database=cfg.db.get("database",""),
        username=cfg.db.get("username",""),
        password=str(cfg.db.get("password","")),
    )
    engine = make_engine(db_cfg)
    cols = inspect_columns(engine, cfg.table)
    print("Table:", cfg.table)
    print("Columns:")
    for c in cols:
        print(" -", c)

if __name__ == "__main__":
    main()
