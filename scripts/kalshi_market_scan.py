"""Cross-category Kalshi inefficiency scan (Goal B, autoresearch 260706).

For hand-picked liquid series per category, pulls settled binary markets from
the (rolling ~10-week) API window, reads the candle price at probe time
(close - 24h, clamped to mid-life for short markets), and measures per-series
calibration: Brier, log loss, bias, and the price-band reliability that a
trader could actually exploit. Active-market execution quality (spread, depth
proxy) is read from the open books of the same series.

    PYTHONPATH=. python scripts/kalshi_market_scan.py
Output: autoresearch/orchestrator-260706-goalAB/market-scan-results.tsv + stdout
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
OUT_TSV = "autoresearch/orchestrator-260706-goalAB/market-scan-results.tsv"
MAX_MARKETS_PER_SERIES = 120
THROTTLE_S = 0.12

# series -> category label (hand-picked: liquid, retail-visible, binary)
SERIES = {
    "KXHIGHNY": "Weather daily high (benchmark)",
    "KXTEMPNYCH": "Weather HOURLY temp (new, TWC settle)",
    "KXBTCD": "Crypto BTC daily",
    "KXETHD": "Crypto ETH daily",
    "KXINXD": "S&P 500 daily",
    "KXNASDAQ100D": "Nasdaq daily",
    "KXMLBGAME": "Sports MLB game",
    "KXWTAMATCH": "Sports WTA match",
    "KXFED": "Fed funds decision",
    "KXCPIYOY": "CPI YoY",
    "KXROTTENT": "Culture Rotten Tomatoes",
    "KXTSAW": "TSA weekly passengers",
}


def get(path: str, **params) -> dict:
    for attempt in range(5):
        r = httpx.get(f"{BASE}/{path}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return {}
        time.sleep(1.5 * 2 ** attempt)
    return {}


def settled_markets(series: str) -> list[dict]:
    out, cursor = [], None
    while len(out) < MAX_MARKETS_PER_SERIES:
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
    return out[:MAX_MARKETS_PER_SERIES]


def probe_price(series: str, m: dict) -> float | None:
    """Candle mid at close-24h, clamped to the market's mid-life."""
    close = dt.datetime.fromisoformat(m["close_time"].replace("Z", "+00:00"))
    opent = dt.datetime.fromisoformat(m["open_time"].replace("Z", "+00:00"))
    probe = max(close - dt.timedelta(hours=24), opent + (close - opent) / 2)
    if probe >= close:
        return None
    life_h = (close - opent).total_seconds() / 3600
    interval = 1 if life_h < 4 else 60
    start = max(opent, probe - dt.timedelta(hours=6))
    j = get(f"series/{series}/markets/{m['ticker']}/candlesticks",
            start_ts=int(start.timestamp()),
            end_ts=int(probe.timestamp()), period_interval=interval)
    candles = j.get("candlesticks", [])
    if not candles:
        return None
    c = candles[-1]

    def dollars(d: dict | None, *keys) -> float | None:
        for k in keys:
            v = (d or {}).get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
        return None

    px = dollars(c.get("price"), "close_dollars", "previous_dollars")
    if px is not None and 0 < px < 1:
        return px
    bid = dollars(c.get("yes_bid"), "close_dollars")
    ask = dollars(c.get("yes_ask"), "close_dollars")
    if bid is not None and ask is not None and 0 <= bid <= ask <= 1 \
            and ask > 0 and (ask - bid) <= 0.10:
        return (bid + ask) / 2
    return None


def scan_series(series: str) -> pd.DataFrame:
    rows = []
    for m in settled_markets(series):
        p = probe_price(series, m)
        time.sleep(THROTTLE_S)
        if p is None or not (0.0 < p < 1.0):
            continue
        rows.append({"series": series, "ticker": m["ticker"], "p": p,
                     "y": 1.0 if m["result"] == "yes" else 0.0,
                     "close_time": m["close_time"]})
    return pd.DataFrame(rows)


def open_spread(series: str) -> tuple[float, int]:
    j = get("markets", series_ticker=series, status="open", limit=100)
    ms = j.get("markets", [])
    spreads = []
    for m in ms:
        try:
            bid = float(m.get("yes_bid_dollars") or 0)
            ask = float(m.get("yes_ask_dollars") or 0)
        except (TypeError, ValueError):
            continue
        if 0 <= bid <= ask <= 1 and ask > 0:
            spreads.append(ask - bid)
    return (float(np.median(spreads)) if spreads else np.nan, len(ms))


def stats(g: pd.DataFrame) -> dict:
    p, y = g["p"].values, g["y"].values
    eps = 1e-4
    pc = np.clip(p, eps, 1 - eps)
    brier = float(np.mean((p - y) ** 2))
    ref = float(np.mean((np.mean(y) - y) ** 2))
    longshot = g[p < 0.15]
    fav = g[p > 0.85]
    return {
        "n": len(g),
        "brier": round(brier, 4),
        "brier_skill": round(1 - brier / ref, 3) if ref > 0 else np.nan,
        "logloss": round(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), 4),
        "bias_p_minus_y": round(float(p.mean() - y.mean()), 4),
        "longshot_n": len(longshot),
        "longshot_p": round(float(longshot["p"].mean()), 3) if len(longshot) else np.nan,
        "longshot_y": round(float(longshot["y"].mean()), 3) if len(longshot) else np.nan,
        "fav_n": len(fav),
        "fav_p": round(float(fav["p"].mean()), 3) if len(fav) else np.nan,
        "fav_y": round(float(fav["y"].mean()), 3) if len(fav) else np.nan,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frames, summary = [], []
    for series, label in SERIES.items():
        df = scan_series(series)
        spread, n_open = open_spread(series)
        time.sleep(THROTTLE_S)
        if df.empty:
            logger.info("%s: no scoreable settled markets", series)
            summary.append({"series": series, "category": label, "n": 0,
                            "median_open_spread": spread, "n_open": n_open})
            continue
        frames.append(df)
        s = {"series": series, "category": label, **stats(df),
             "median_open_spread": spread, "n_open": n_open}
        summary.append(s)
        logger.info("%s: n=%d brier=%.4f", series, s["n"], s["brier"])
    res = pd.DataFrame(summary)
    res.to_csv(OUT_TSV, sep="\t", index=False)
    if frames:
        pd.concat(frames).to_csv(OUT_TSV.replace("results", "rows"), sep="\t", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
