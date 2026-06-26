"""
Fetch historical Kalshi temperature market prices from the LIVE API (read-only).
Saves to data/historical/kalshi_prices.parquet for use in backtesting.

Uses LIVE credentials for authentication (read-only GET requests only).
No orders are placed — write methods are explicitly blocked.

Usage: PYTHONPATH=. python scripts/fetch_kalshi_history.py
"""
import asyncio
import base64
import logging
import re
import sys
import time
from datetime import datetime, date, timezone
from pathlib import Path

import httpx
import pandas as pd
import numpy as np
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LIVE_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
LIVE_API_KEY = "84f4ee55-7337-4391-a762-4810c9769bd2"
OUT_PATH = Path("data/historical/kalshi_prices.parquet")

# All temperature series mapped to ASOS station
SERIES_STATION: dict[str, str] = {
    "KXHIGHCHI":   "KORD",  "KXLOWTCHI":   "KORD",
    "KXHIGHNY":    "KLGA",  "KXHIGHNY0":   "KLGA",  "KXLOWNYC":    "KLGA",
    "KXHIGHLAX":   "KLAX",  "KXLOWTLAX":   "KLAX",
    "KXHIGHMIA":   "KMIA",  "KXLOWMIA":    "KMIA",
    "KXHIGHOU":    "KIAH",  "KXHIGHHOU":   "KIAH",  "KXLOWTHOU":   "KIAH",
    "KXHIGHPHIL":  "KPHL",  "KXLOWPHIL":   "KPHL",
    "KXHIGHATL":   "KATL",  "KXLOWTATL":   "KATL",
    "KXHIGHAUS":   "KAUS",  "KXLOWTAUS":   "KAUS",
    "KXDENHIGH":   "KDEN",  "KXHIGHDEN":   "KDEN",  "KXLOWDEN":    "KDEN",
    "KXHIGHTPHX":  "KPHX",  "KXLOWTPHX":   "KPHX",
    "KXHIGHTSFO":  "KSFO",  "KXLOWTSFO":   "KSFO",
    "KXHIGHTSEA":  "KSEA",  "KXLOWTSEA":   "KSEA",
    "KXHIGHTBOS":  "KBOS",  "KXLOWTBOS":   "KBOS",
    "KXHIGHTDAL":  "KDFW",  "KXLOWTDAL":   "KDFW",
    "KXHIGHTDC":   "KDCA",  "KXLOWTDC":    "KDCA",
    "KXHIGHTLV":   "KLAS",  "KXLOWTLV":    "KLAS",
    "KXHIGHTMIN":  "KMSP",  "KXLOWTMIN":   "KMSP",
    "KXHIGHTOKC":  "KOKC",  "KXLOWTOKC":   "KOKC",
    "KXHIGHTSATX": "KSAT",  "KXLOWTSATX":  "KSAT",
    "KXHIGHTNOLA": "KMSY",  "KXLOWTNOLA":  "KMSY",
}


# ---------------------------------------------------------------------------
# Read-only authenticated client
# ---------------------------------------------------------------------------

class ReadOnlyKalshiClient:
    """Live API client with write operations explicitly blocked."""

    def __init__(self, api_key: str, private_key_path: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        pem = Path(private_key_path).read_bytes()
        self._private_key = serialization.load_pem_private_key(pem, password=None)

    # --- Safety: block all write operations ---
    async def create_order(self, *args, **kwargs):
        raise RuntimeError("READ-ONLY CLIENT: create_order is disabled")

    async def cancel_order(self, *args, **kwargs):
        raise RuntimeError("READ-ONLY CLIENT: cancel_order is disabled")

    async def _post(self, *args, **kwargs):
        raise RuntimeError("READ-ONLY CLIENT: POST requests are disabled")

    async def _delete(self, *args, **kwargs):
        raise RuntimeError("READ-ONLY CLIENT: DELETE requests are disabled")

    # --- Auth ---
    def _sign(self, method: str, path: str) -> dict[str, str]:
        ts_ms = str(int(time.time() * 1000))
        msg = ts_ms + method.upper() + path
        sig = self._private_key.sign(msg.encode(), padding.PKCS1v15(), hashes.SHA256())
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict | None = None) -> dict:
        url = self._base_url + path
        headers = self._sign("GET", path)
        for attempt in range(4):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(url, headers=headers, params=params, timeout=30.0)
                if resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)
        return {}

    # --- Read endpoints ---
    async def get_settled_markets(self, series_ticker: str, cursor: str | None = None) -> dict:
        params: dict = {"series_ticker": series_ticker, "status": "settled", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        return await self._get("/markets", params=params)

    async def get_candlesticks(self, series: str, ticker: str, start_ts: int, end_ts: int) -> list[dict]:
        # Kalshi requires the series in the path: /series/{series}/markets/{ticker}/candlesticks.
        # The bare /markets/{ticker}/candlesticks form 404s.
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60}
        try:
            data = await self._get(f"/series/{series}/markets/{ticker}/candlesticks", params=params)
            return data.get("candlesticks", [])
        except Exception as exc:
            logger.debug("Candlestick fetch failed for %s: %s", ticker, exc)
            return []


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

def _compute_d1_mid(prev_bid, prev_ask, last_price=None) -> float:
    """Derive a D-1 mid from the prior session's yes bid/ask, returning NaN when
    the book is not a genuine two-sided quote.

    Settled markets report previous_yes_bid=0 / previous_yes_ask=1 (a collapsed
    book); its midpoint (0+1)/2 = 0.5 is a fabricated price that silently turns
    every backtest into 'beat a coin flip'. A real price requires 0 < bid <= ask < 1.
    `last_price` is intentionally NOT used as a fallback: for settled markets it is
    the near-resolution trade price and would leak the outcome into the backtest.
    """
    if prev_bid is None or prev_ask is None:
        return float("nan")
    bid, ask = float(prev_bid), float(prev_ask)
    if bid <= 0.0 or ask >= 1.0 or bid > ask:
        return float("nan")
    return (bid + ask) / 2.0


def _decision_price_from_candles(candles: list[dict], cutoff_ts: int) -> float | None:
    """Return the last traded price (price.close) from the candle at or before
    `cutoff_ts`, or None.

    `cutoff_ts` is the model's decision time (end of D-1). Candles after the
    cutoff are ignored so the backtest price carries no look-ahead toward the
    known outcome. A close of exactly 0 or 1 is a settled/degenerate value and is
    skipped — only genuine in-(0,1) trade prices count.
    """
    best_ts: int | None = None
    best_price: float | None = None
    for c in candles:
        ts = c.get("end_period_ts")
        if ts is None or ts > cutoff_ts:
            continue
        price = c.get("price") or {}
        raw = price.get("close_dollars")
        if raw in (None, ""):
            raw = price.get("mean_dollars")
        if raw in (None, ""):
            continue
        val = float(raw)
        if val <= 0.0 or val >= 1.0:
            continue
        if best_ts is None or ts > best_ts:
            best_ts, best_price = ts, val
    return best_price


def _parse_resolution_date(ticker: str) -> date | None:
    """Extract the resolution date (YYMMMDD, e.g. 26MAY31) from a ticker."""
    date_match = re.search(r"-(\d{2}[A-Z]{3}\d{2})-", ticker)
    if not date_match:
        return None
    try:
        return datetime.strptime(date_match.group(1), "%y%b%d").date()
    except ValueError:
        return None


def _strike_fields(m: dict) -> dict:
    """Extract the true bracket structure from a Kalshi market object.

    Kalshi temperature markets are mutually-exclusive brackets, NOT above/below
    contracts. The API gives the exact structure:
      * strike_type "greater": YES if high > floor_strike (e.g. ">84°")
      * strike_type "less":    YES if high < cap_strike   (e.g. "76° or below")
      * strike_type "between": YES if floor_strike ≤ high ≤ cap_strike
    floor_strike / cap_strike are absent for the side that is unbounded.
    """
    def _f(v):
        return None if v is None else float(v)

    return {
        "strike_type": m.get("strike_type"),
        "floor_strike": _f(m.get("floor_strike")),
        "cap_strike": _f(m.get("cap_strike")),
    }


# ---------------------------------------------------------------------------
# Fetch all settled markets for a series
# ---------------------------------------------------------------------------

async def fetch_series_markets(
    client: ReadOnlyKalshiClient,
    series: str,
    station: str,
) -> list[dict]:
    rows = []
    cursor = None
    pages = 0

    while pages < 50:
        data = await client.get_settled_markets(series, cursor)
        markets = data.get("markets", [])
        cursor = data.get("cursor")

        for m in markets:
            ticker = m.get("ticker", "")
            resolution_date = _parse_resolution_date(ticker)
            if resolution_date is None:
                continue
            strikes = _strike_fields(m)
            if strikes["strike_type"] is None:
                continue

            # Get D+1 mid from previous_yes_bid/ask (prices from prior trading
            # session). Empty/collapsed books (bid=0, ask=1) yield NaN — not a
            # fabricated 0.5 — so they are dropped rather than treated as real.
            d1_mid = _compute_d1_mid(
                m.get("previous_yes_bid_dollars"),
                m.get("previous_yes_ask_dollars"),
            )

            result = m.get("result")
            settlement = 1.0 if result == "yes" else 0.0 if result == "no" else float("nan")

            rows.append({
                "ticker":       ticker,
                "series":       series,
                "station":      station,
                "date":         resolution_date,
                "strike_type":  strikes["strike_type"],
                "floor_strike": strikes["floor_strike"],
                "cap_strike":   strikes["cap_strike"],
                "subtitle":     m.get("yes_sub_title") or m.get("subtitle"),
                "d1_mid":       d1_mid,
                "settlement":   settlement,
                "yes_bid":      m.get("yes_bid_dollars"),
                "yes_ask":      m.get("yes_ask_dollars"),
                "volume":       m.get("volume_fp"),
            })

        pages += 1
        if not cursor or not markets:
            break
        await asyncio.sleep(0.1)  # gentle rate limiting

    logger.info("  %s (%s): %d markets", series, station, len(rows))
    return rows


# ---------------------------------------------------------------------------
# Enrich with candlestick D+1 prices (authenticated)
# ---------------------------------------------------------------------------

# Decision cutoff: 14:00 UTC on resolution day ≈ 7–10am US local, before the
# afternoon max-temp high occurs. A price taken at/before this is one the bot
# could actually have traded on without knowing the outcome (no look-ahead).
DECISION_CUTOFF_HOUR_UTC = 14


async def enrich_with_candlesticks(
    client: ReadOnlyKalshiClient,
    df: pd.DataFrame,
    sample_size: int = 20000,
) -> pd.DataFrame:
    """Replace the (unreliable) bid/ask proxy with the real last-traded price
    from candlesticks, sampled at the model's decision time (no look-ahead).

    For settled markets the order book collapses to 0/1, so the bid/ask proxy is
    fabricated; the candle `price` (last trade) is the genuine price. We pull
    candles over [res-2d, decision cutoff] and take the last trade at or before
    the cutoff via _decision_price_from_candles.
    """
    df = df.copy()
    df["d1_candle_mid"] = float("nan")

    df["_vol_num"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    active = df[df["_vol_num"] > 0].head(sample_size)
    logger.info("Fetching candlestick decision-time prices for %d markets...", len(active))

    updated = 0
    for n, (_, row) in enumerate(active.iterrows()):
        ticker = row["ticker"]
        series = row["series"]
        rd = pd.Timestamp(row["date"]).date()
        res_midnight = int(datetime(rd.year, rd.month, rd.day, tzinfo=timezone.utc).timestamp())
        cutoff_ts = res_midnight + DECISION_CUTOFF_HOUR_UTC * 3600
        start_ts = res_midnight - 2 * 86400

        candles = await client.get_candlesticks(series, ticker, start_ts, cutoff_ts)
        price = _decision_price_from_candles(candles, cutoff_ts)
        if price is not None:
            df.loc[df["ticker"] == ticker, "d1_candle_mid"] = price
            updated += 1

        if n and n % 250 == 0:
            logger.info("  ...%d/%d processed, %d priced", n, len(active), updated)
        await asyncio.sleep(0.1)

    # Candlesticks are the ONLY genuine price source for settled markets —
    # overwrite d1_mid wholesale (the bid/ask proxy is NaN for empty books).
    df["d1_mid"] = df["d1_candle_mid"]
    df = df.drop(columns=["d1_candle_mid", "_vol_num"], errors="ignore")
    logger.info("Candlestick enrichment: %d/%d markets priced from real candles",
                int(df["d1_mid"].notna().sum()), len(active))
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    from config.settings import get_settings
    settings = get_settings()

    client = ReadOnlyKalshiClient(
        api_key=LIVE_API_KEY,
        private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
        base_url=LIVE_BASE_URL,
    )

    logger.info("Fetching historical Kalshi temperature markets (LIVE API, read-only)")
    logger.info("Base URL: %s", LIVE_BASE_URL)

    all_rows: list[dict] = []

    for series, station in SERIES_STATION.items():
        try:
            rows = await fetch_series_markets(client, series, station)
            all_rows.extend(rows)
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", series, exc)

    if not all_rows:
        logger.error("No data fetched — check credentials and network")
        return

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])

    # Enrich with candlestick prices where possible
    df = await enrich_with_candlesticks(client, df)

    # Drop rows with no usable mid price
    before = len(df)
    df = df[df["d1_mid"].notna() & (df["d1_mid"] > 0) & (df["d1_mid"] < 1)]
    logger.info("Filtered %d → %d rows with valid d1_mid", before, len(df))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    logger.info("Saved %d market records to %s", len(df), OUT_PATH)
    logger.info("Stations: %s", sorted(df['station'].unique().tolist()))
    logger.info("Date range: %s → %s", df['date'].min().date(), df['date'].max().date())
    logger.info("Avg d1_mid: %.3f | std: %.3f", df['d1_mid'].mean(), df['d1_mid'].std())


if __name__ == "__main__":
    asyncio.run(main())
