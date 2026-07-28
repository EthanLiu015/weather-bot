"""Hourly forward-capture: top-of-book snapshots for target series.

Feeds two studies that cannot be backtested (Kalshi purges ticks):
  * crypto-bracket staleness — how long do quotes lag spot moves?
  * weather book depth/spread around model-release times

    PYTHONPATH=. python scripts/capture_books.py
Output: data/capture/books.parquet (append; one row per open market per run)
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd

from scripts.kalshi_market_scan import get

logger = logging.getLogger(__name__)

SERIES = ["KXBTCD", "KXETHD", "KXWTAMATCH", "KXATPMATCH", "KXTSAW"]
OUT = "data/capture/books.parquet"
THROTTLE_S = 0.12


def snapshot(series: str, ts: str) -> list[dict]:
    j = get("markets", series_ticker=series, status="open", limit=200)
    rows = []
    for m in j.get("markets", []):
        rows.append({
            "ts": ts, "series": series, "ticker": m["ticker"],
            "yes_bid": m.get("yes_bid_dollars"), "yes_ask": m.get("yes_ask_dollars"),
            "last_price": m.get("last_price_dollars"),
            "volume_fp": m.get("volume_fp"), "open_interest_fp": m.get("open_interest_fp"),
            "close_time": m.get("close_time"),
        })
    return rows


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ts = pd.Timestamp.utcnow().isoformat()
    rows = []
    for series in SERIES:
        got = snapshot(series, ts)
        rows += got
        logger.info("%s: %d open markets", series, len(got))
        time.sleep(THROTTLE_S)
    if not rows:
        return
    new = pd.DataFrame(rows)
    try:
        out = pd.concat([pd.read_parquet(OUT), new], ignore_index=True)
    except FileNotFoundError:
        out = new
    out.to_parquet(OUT)
    logger.info("appended %d rows -> %s (total %d)", len(new), OUT, len(out))


if __name__ == "__main__":
    main()
