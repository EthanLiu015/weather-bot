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
from datetime import datetime, date
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

    async def get_candlesticks(self, ticker: str, start_ts: int, end_ts: int) -> list[dict]:
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60}
        try:
            data = await self._get(f"/markets/{ticker}/candlesticks", params=params)
            return data.get("candlesticks", [])
        except Exception as exc:
            logger.debug("Candlestick fetch failed for %s: %s", ticker, exc)
            return []


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------

def _parse_ticker(ticker: str, series: str) -> dict | None:
    """
    Parse ticker like KXHIGHCHI-26MAY31-T81 or KXHIGHCHI-26MAY31-B80.5
    Returns: {date, threshold, market_type ('above'/'below')}
    """
    # Extract date part: YYMMMDD e.g. 26MAY31
    date_match = re.search(r"-(\d{2}[A-Z]{3}\d{2})-", ticker)
    if not date_match:
        return None
    try:
        resolution_date = datetime.strptime(date_match.group(1), "%y%b%d").date()
    except ValueError:
        return None

    # Extract threshold: T75 or B80.5
    thresh_match = re.search(r"-([TB])([\d.]+)$", ticker)
    if not thresh_match:
        return None

    market_type = "above" if thresh_match.group(1) == "T" else "below"
    threshold = float(thresh_match.group(2))

    return {"date": resolution_date, "threshold": threshold, "market_type": market_type}


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
            parsed = _parse_ticker(ticker, series)
            if not parsed:
                continue

            # Get D+1 mid from previous_yes_bid/ask (prices from prior trading session)
            prev_bid = m.get("previous_yes_bid_dollars")
            prev_ask = m.get("previous_yes_ask_dollars")
            last_price = m.get("last_price_dollars")

            if prev_bid is not None and prev_ask is not None:
                d1_mid = (float(prev_bid) + float(prev_ask)) / 2.0
            elif last_price is not None:
                d1_mid = float(last_price)
            else:
                d1_mid = float("nan")

            result = m.get("result")
            settlement = 1.0 if result == "yes" else 0.0 if result == "no" else float("nan")

            rows.append({
                "ticker":       ticker,
                "series":       series,
                "station":      station,
                "date":         parsed["date"],
                "threshold":    parsed["threshold"],
                "market_type":  parsed["market_type"],
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

async def enrich_with_candlesticks(
    client: ReadOnlyKalshiClient,
    df: pd.DataFrame,
    sample_size: int = 200,
) -> pd.DataFrame:
    """
    For a sample of markets, fetch hourly candlestick data to get a more
    accurate D+1 price (market mid 24h before resolution).
    Used to validate / improve the prev_bid/ask proxy.
    """
    df = df.copy()
    df["d1_candle_mid"] = float("nan")

    # Sample markets where we have volume (actively traded)
    active = df[df["volume"].notna() & (df["volume"] > 0)].head(sample_size)
    logger.info("Fetching candlestick D+1 prices for %d markets...", len(active))

    for _, row in active.iterrows():
        ticker = row["ticker"]
        res_date = row["date"]
        # D+1: 24h window starting 48h before resolution midnight
        res_ts = int(datetime(res_date.year, res_date.month, res_date.day).timestamp())
        start_ts = res_ts - 48 * 3600
        end_ts = res_ts - 24 * 3600

        candles = await client.get_candlesticks(ticker, start_ts, end_ts)
        if candles:
            # Use the last candle's close price as D+1 mid
            last = candles[-1]
            close = last.get("close", {})
            yes_price = close.get("yes_price")
            if yes_price is not None:
                df.loc[df["ticker"] == ticker, "d1_candle_mid"] = float(yes_price) / 100.0

        await asyncio.sleep(0.15)

    # Use candlestick price where available, fall back to prev_bid/ask proxy
    has_candle = df["d1_candle_mid"].notna()
    df.loc[has_candle, "d1_mid"] = df.loc[has_candle, "d1_candle_mid"]
    df = df.drop(columns=["d1_candle_mid"])
    logger.info("Candlestick enrichment: %d markets updated", has_candle.sum())
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
