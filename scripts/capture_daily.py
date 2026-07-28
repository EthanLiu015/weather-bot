"""Daily forward-capture: settled results + probe candles, before Kalshi purges.

Kalshi serves only ~10 weeks of history, so every backtest dies of small-n.
This job runs daily (cron) and appends, for each target series:
  * newly settled binary markets (ticker, result, open/close times)
  * the candle probe price at close-24h (clamped to mid-life) — captured NOW so
    the calibration sample keeps growing after the API window rolls off

Idempotent: already-captured tickers are skipped.

    PYTHONPATH=. python scripts/capture_daily.py
Output: data/capture/settled_probe.parquet (append)
Log:    data/capture/logs/capture_daily.log (via cron redirection)
"""
from __future__ import annotations

import logging
import os
import time

import pandas as pd

from scripts.kalshi_market_scan import get, probe_price

logger = logging.getLogger(__name__)

SERIES = ["KXBTCD", "KXETHD", "KXWTAMATCH", "KXATPMATCH", "KXATPCHALLENGERMATCH",
          "KXWTACHALLENGERMATCH", "KXCHALLENGERMATCH",
          "KXMLBGAME", "KXTSAW"]
OUT = "data/capture/settled_probe.parquet"
MAX_PER_SERIES = 1000  # BTC/ETH hourly ladders produce ~360 markets/day
THROTTLE_S = 0.12


def recent_settled(series: str) -> list[dict]:
    out, cursor = [], None
    while len(out) < MAX_PER_SERIES:
        params = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        j = get("markets", **params)
        ms = j.get("markets", [])
        out += [m for m in ms if m.get("market_type") == "binary"
                and m.get("result") in ("yes", "no")]
        cursor = j.get("cursor")
        if not ms or not cursor:
            break
        time.sleep(THROTTLE_S)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        existing = pd.read_parquet(OUT)
        seen = set(existing["ticker"])
    except FileNotFoundError:
        existing, seen = pd.DataFrame(), set()

    rows = []
    for series in SERIES:
        new = [m for m in recent_settled(series) if m["ticker"] not in seen]
        for m in new:
            p = probe_price(series, m)
            time.sleep(THROTTLE_S)
            rows.append({
                "series": series, "ticker": m["ticker"],
                "event_ticker": m.get("event_ticker"),
                "result": m["result"],
                "open_time": m.get("open_time"), "close_time": m.get("close_time"),
                "probe_price": p,
                "floor_strike": m.get("floor_strike"), "cap_strike": m.get("cap_strike"),
                "captured_at": pd.Timestamp.utcnow().isoformat(),
            })
        logger.info("%s: %d new settled", series, len(new))
    if rows:
        out = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        out.to_parquet(OUT)
        logger.info("appended %d rows -> %s (total %d)", len(rows), OUT, len(out))
    else:
        logger.info("nothing new")


if __name__ == "__main__":
    main()
