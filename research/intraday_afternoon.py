"""Afternoon intraday dataset: Kalshi price DURING the daily high, paired with the
settlement-aligned running max known at that moment.

The existing intraday_prices.parquet only reaches +14h UTC (~9am local) — before the
afternoon high, so the running-max signal is uninformative there. This module pulls
the traded price at AFTERNOON offsets (default +16/+18/+20/+22h UTC) from Kalshi
hourly candlesticks and, for each, records `run_max` = the settlement-aligned running
max (5-min-avg reconstruction from 1-minute ASOS) known at that time. That pairing is
what the obs-conditioned edge test needs: at time T, given run_max, is P(final max >
threshold) mispriced by the book?

No look-ahead: run_max at T ignores obs after T; settlement is the realized outcome we
predict, never an input.

    PYTHONPATH=. python -m research.intraday_afternoon --limit 60
Output: data/historical/intraday_afternoon.parquet
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd

from zoneinfo import ZoneInfo

from config.settings import get_settings
from config.series import is_low_temp_series
from config.stations import station_timezones
from ingestion.asos_1min import fetch_1min, settlement_max_at
from trading.kalshi_client import KalshiClient

_TZ = station_timezones()

logger = logging.getLogger(__name__)

OUT_PATH = "data/historical/intraday_afternoon.parquet"
PRICES_PATH = "data/historical/kalshi_prices.parquet"

# LOCAL hours into the settlement day at which to sample price + running max.
# 16/18/20/22 = 4/6/8/10pm local — straddling the daily high as it locks in.
# (Kalshi settles on the station's LOCAL calendar-day max, so everything is anchored
# to local midnight, not UTC midnight — a UTC anchor grabs the prior day's peak for
# west-coast stations.)
AFTERNOON_OFFSETS_H = [16, 18, 20, 22]


def _local_midnight_ts(date, station: str) -> int:
    """Unix ts of local midnight starting `date`'s settlement day at `station`."""
    d = pd.Timestamp(date)
    tz = ZoneInfo(_TZ.get(station, "UTC"))
    return int(datetime(d.year, d.month, d.day, tzinfo=tz).timestamp())


def price_at(candles: list[dict], cutoff_ts: int) -> float | None:
    """Last genuine traded YES price (in dollars) at or before `cutoff_ts`.

    Reads the most recent candle whose end_period_ts ≤ cutoff. Exact 0/1 closes are
    degenerate/settled values and skipped; only in-(0,1) prices count.
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


async def _prefetch_obs(station_days: set[tuple[str, str]], concurrency: int = 3) -> dict:
    """Fetch 1-minute obs once per unique (station, date) over the station's LOCAL
    settlement day. Bounded concurrency + single fetch per key — avoids the cache
    race that self-throttles IEM."""
    sem = asyncio.Semaphore(concurrency)
    cache: dict = {}

    async def one(station, date_str):
        mid = _local_midnight_ts(date_str, station)
        start = datetime.fromtimestamp(mid, tz=timezone.utc)
        end = start + pd.Timedelta(hours=24)
        async with sem:
            cache[(station, date_str)] = await asyncio.to_thread(fetch_1min, station, start, end)

    await asyncio.gather(*[one(st, ds) for st, ds in station_days])
    got = sum(1 for v in cache.values() if len(v))
    logger.info("Prefetched 1-min obs for %d/%d station-days", got, len(cache))
    return cache


async def _rows_for_market(client, row, obs_cache, sem, offsets) -> list[dict]:
    mid = _local_midnight_ts(row.date, row.station)
    async with sem:
        candles = await client.get_candlesticks_range(
            row.series, row.ticker, mid - 86400, mid + 30 * 3600
        )
    if not candles:
        return []
    obs = obs_cache.get((row.station, str(pd.Timestamp(row.date).date())), pd.DataFrame())
    rows = []
    for h in offsets:
        cutoff = mid + h * 3600
        price = price_at(candles, cutoff)
        if price is None:
            continue
        run_max = settlement_max_at(obs, pd.Timestamp(cutoff, unit="s", tz="UTC").tz_localize(None)) if len(obs) else None
        rows.append({
            "ticker": row.ticker, "station": row.station, "date": pd.Timestamp(row.date),
            "offset_h": h, "price": price, "run_max": run_max,
            "strike_type": row.strike_type, "floor_strike": row.floor_strike,
            "cap_strike": row.cap_strike, "settlement": float(row.settlement),
        })
    return rows


# Usable window = intersection of the candlestick history (~10 weeks back) and the
# IEM 1-minute archive (lags ~2 weeks). Dates are days-before-today.
CANDLE_MAX_AGE_D = 63
OBS_MIN_AGE_D = 15


async def build(limit: int, offsets=AFTERNOON_OFFSETS_H, concurrency: int = 6) -> pd.DataFrame:
    s = get_settings()
    client = KalshiClient(api_key=s.KALSHI_API_KEY,
                          private_key_path=s.KALSHI_PRIVATE_KEY_PATH,
                          base_url=s.KALSHI_BASE_URL)
    df = pd.read_parquet(PRICES_PATH)
    df = df[~df["series"].map(is_low_temp_series)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_vol"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df[df["settlement"].notna() & (df["_vol"] > 0) & df["strike_type"].notna()]

    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    lo, hi = today - pd.Timedelta(days=CANDLE_MAX_AGE_D), today - pd.Timedelta(days=OBS_MIN_AGE_D)
    df = df[(df["date"] >= lo) & (df["date"] <= hi)]
    # Prefer the most recent markets inside the window (best candle liquidity).
    df = df.sort_values("date").tail(limit)
    logger.info("Afternoon backfill: %d markets in [%s..%s], offsets=%s",
                len(df), lo.date(), hi.date(), offsets)

    station_days = {(r.station, str(pd.Timestamp(r.date).date())) for r in df.itertuples(index=False)}
    obs_cache = await _prefetch_obs(station_days)

    sem = asyncio.Semaphore(concurrency)
    batches = await asyncio.gather(
        *[_rows_for_market(client, r, obs_cache, sem, offsets) for r in df.itertuples(index=False)]
    )
    rows = [r for batch in batches for r in batch]
    out = pd.DataFrame(rows)
    out.to_parquet(OUT_PATH, index=False)
    logger.info("Wrote %d price×offset rows (%d markets) → %s",
                len(out), out["ticker"].nunique() if len(out) else 0, OUT_PATH)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--offsets", type=int, nargs="+", default=AFTERNOON_OFFSETS_H)
    args = ap.parse_args()
    out = asyncio.run(build(args.limit, offsets=args.offsets))
    if len(out):
        cov = out.groupby("offset_h").agg(n=("price", "size"),
                                          obs=("run_max", lambda s: s.notna().sum())).to_dict()
        print("\nrows per offset:", out.groupby("offset_h").size().to_dict())
        print("with run_max:", {h: int(v) for h, v in cov["obs"].items()})


if __name__ == "__main__":
    main()
