"""Tennis favorite-longshot bias check: Kalshi vs sharp sportsbook lines.

Covers ATP + WTA tour and challenger match series (replaces the WTA-only
version; its rows were migrated into this output). The scan (autoresearch
260706) showed the classic FL-bias direction on KXWTAMATCH; before believing
it, cross-check Kalshi prices against de-vigged sharp-book consensus. Each run
logs matched (kalshi_mid, sharp_prob) pairs; settlement is scored later from
capture_daily's settled feed.

Odds source: The Odds API (https://the-odds-api.com), free tier 500 req/month.
Active tennis sport keys are discovered via the quota-free /sports listing.
Set ODDS_API_KEY in the environment; without it the script logs Kalshi books
only (still useful — pairs can be matched retroactively).

    ODDS_API_KEY=... PYTHONPATH=. python scripts/tennis_odds_compare.py
Output: data/capture/tennis_compare.parquet (append)
"""
from __future__ import annotations

import logging
import os
import re

import httpx
import pandas as pd

from scripts.kalshi_market_scan import get

logger = logging.getLogger(__name__)

SPORTS_URL = "https://api.the-odds-api.com/v4/sports"
ODDS_URL = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
FALLBACK_SPORTS = [
    "tennis_atp_wimbledon", "tennis_atp_french_open", "tennis_atp_us_open",
    "tennis_atp_aus_open_singles",
    "tennis_wta_wimbledon", "tennis_wta_french_open", "tennis_wta_us_open",
    "tennis_wta_aus_open_singles",
]
KALSHI_SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH",
                 "KXWTACHALLENGERMATCH", "KXCHALLENGERMATCH"]
OUT = "data/capture/tennis_compare.parquet"


def kalshi_tennis_books() -> pd.DataFrame:
    rows = []
    for series in KALSHI_SERIES:
        j = get("markets", series_ticker=series, status="open", limit=200)
        for m in j.get("markets", []):
            try:
                bid = float(m.get("yes_bid_dollars") or 0)
                ask = float(m.get("yes_ask_dollars") or 0)
            except (TypeError, ValueError):
                continue
            if not (0 < ask < 1):
                continue
            rows.append({"series": series, "ticker": m["ticker"],
                         "title": m.get("title") or "",
                         "yes_sub_title": m.get("yes_sub_title") or "",
                         "kalshi_bid": bid, "kalshi_ask": ask,
                         "kalshi_mid": (bid + ask) / 2 if bid > 0 else ask,
                         "close_time": m.get("close_time")})
        logger.info("%s: %d open priced markets", series,
                    sum(r["series"] == series for r in rows))
    return pd.DataFrame(rows)


def active_tennis_sports(api_key: str) -> list[str]:
    """Quota-free listing of currently active tennis sport keys."""
    try:
        r = httpx.get(SPORTS_URL, params={"apiKey": api_key}, timeout=30)
        r.raise_for_status()
        keys = [s["key"] for s in r.json()
                if s.get("group") == "Tennis" and s.get("active")]
        return keys or FALLBACK_SPORTS
    except httpx.HTTPError as e:
        logger.warning("sports listing failed (%s); using fallback", e)
        return FALLBACK_SPORTS


def sharp_probs(api_key: str) -> pd.DataFrame:
    """De-vigged two-way implied probabilities, sharp-book preference order."""
    rows = []
    for sport in active_tennis_sports(api_key):
        r = httpx.get(ODDS_URL.format(sport=sport),
                      params={"apiKey": api_key, "regions": "eu,us",
                              "markets": "h2h", "oddsFormat": "decimal"},
                      timeout=30)
        if r.status_code != 200:
            continue
        for game in r.json():
            books = {b["key"]: b for b in game.get("bookmakers", [])}
            book = next((books[k] for k in ("pinnacle", "betfair_ex_eu", "bet365")
                         if k in books), None)
            if book is None and books:
                book = next(iter(books.values()))
            if book is None:
                continue
            h2h = next((mk for mk in book.get("markets", []) if mk["key"] == "h2h"), None)
            if h2h is None or len(h2h.get("outcomes", [])) != 2:
                continue
            o1, o2 = h2h["outcomes"]
            inv1, inv2 = 1 / o1["price"], 1 / o2["price"]
            total = inv1 + inv2
            for o, inv in ((o1, inv1), (o2, inv2)):
                rows.append({"player": o["name"], "sharp_prob": inv / total,
                             "book": book["key"], "sport": sport,
                             "commence": game.get("commence_time")})
    return pd.DataFrame(rows)


def last_name(s: str) -> str:
    parts = re.sub(r"[^A-Za-z \-']", " ", s).split()
    return parts[-1].lower() if parts else ""


def match(kalshi: pd.DataFrame, sharp: pd.DataFrame) -> pd.DataFrame:
    if kalshi.empty or sharp.empty:
        kalshi["sharp_prob"] = None
        kalshi["sharp_book"] = None
        return kalshi
    sharp = sharp.assign(key=sharp["player"].map(last_name))
    by_key = sharp.drop_duplicates("key").set_index("key")
    probs, bookcol = [], []
    for _, row in kalshi.iterrows():
        key = last_name(row["yes_sub_title"] or row["title"])
        hit = by_key.loc[key] if key in by_key.index else None
        probs.append(float(hit["sharp_prob"]) if hit is not None else None)
        bookcol.append(str(hit["book"]) if hit is not None else None)
    kalshi["sharp_prob"] = probs
    kalshi["sharp_book"] = bookcol
    return kalshi


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    kalshi = kalshi_tennis_books()
    logger.info("kalshi tennis open markets: %d", len(kalshi))
    api_key = os.environ.get("ODDS_API_KEY", "")
    if api_key:
        merged = match(kalshi, sharp_probs(api_key))
        n = merged["sharp_prob"].notna().sum()
        logger.info("matched %d/%d to sharp lines", n, len(merged))
    else:
        merged = match(kalshi, pd.DataFrame())
        logger.warning("ODDS_API_KEY not set — logging Kalshi books only")
    if merged.empty:
        return
    merged["ts"] = pd.Timestamp.utcnow().isoformat()
    try:
        out = pd.concat([pd.read_parquet(OUT), merged], ignore_index=True)
    except FileNotFoundError:
        out = merged
    out.to_parquet(OUT)
    logger.info("appended %d rows -> %s (total %d)", len(merged), OUT, len(out))


if __name__ == "__main__":
    main()
