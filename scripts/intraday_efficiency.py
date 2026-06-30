"""Time-to-resolution efficiency probe for Kalshi temperature markets.

Every prior eval scored the market at a SINGLE moment (d1_mid, ~14:00 UTC on
resolution day). This probe extracts the traded price at SEVERAL times-to-
resolution from the same hourly-candle pull and measures how the market's
accuracy (Brier vs settlement) evolves through the trading window.

Question: is there a "stale window" — a time when the price is anchored/dumb and
hasn't yet incorporated good guidance — where a static forecast could have edge?
If market Brier is already low at the earliest cutoff, the market is efficient
end-to-end and there is no timing edge. If it plateaus early then drops, the gap
is where an edge (liquidity permitting) would live.

All cutoffs are <= +14h UTC (≈9am local), well before the afternoon high, so no
price here reflects the realizing outcome (no look-ahead).

Run: PYTHONPATH=. python scripts/intraday_efficiency.py [--limit N]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.series import is_low_temp_series
from config.settings import get_settings
from scripts.fetch_kalshi_history import (
    LIVE_API_KEY,
    LIVE_BASE_URL,
    ReadOnlyKalshiClient,
    _decision_price_from_candles,
)

logger = logging.getLogger(__name__)

# Hours relative to resolution-day 00:00 UTC. -12 = D-1 noon, 0 = res midnight,
# +14 = the existing decision cutoff (~9am local). All pre-afternoon-high.
CUTOFF_OFFSETS_H = [-12, -6, 0, 6, 12, 14]

PRICES_PATH = Path("data/historical/intraday_prices.parquet")


async def _fetch_market(client, row, sem) -> dict | None:
    rd = pd.Timestamp(row.date).date()
    res_midnight = int(datetime(rd.year, rd.month, rd.day, tzinfo=timezone.utc).timestamp())
    start_ts = res_midnight - 2 * 86400
    end_ts = res_midnight + 14 * 3600
    async with sem:
        try:
            candles = await client.get_candlesticks(row.series, row.ticker, start_ts, end_ts)
        except Exception as exc:
            logger.debug("candle fetch failed for %s: %s", row.ticker, exc)
            return None
    if not candles:
        return None
    out = {"ticker": row.ticker, "station": row.station, "settlement": float(row.settlement)}
    for h in CUTOFF_OFFSETS_H:
        out[f"p{h:+d}"] = _decision_price_from_candles(candles, res_midnight + h * 3600)
    return out


async def run(limit: int, concurrency: int = 6) -> pd.DataFrame:
    settings = get_settings()
    client = ReadOnlyKalshiClient(
        api_key=LIVE_API_KEY,
        private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
        base_url=LIVE_BASE_URL,
    )

    df = pd.read_parquet("data/historical/kalshi_prices.parquet")
    df = df[~df["series"].map(is_low_temp_series)].copy()
    df["_vol"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df[df["settlement"].notna() & (df["_vol"] > 0)]
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=0)
    logger.info("Probing %d high-temp settled markets across %d cutoffs", len(df), len(CUTOFF_OFFSETS_H))

    sem = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(*[_fetch_market(client, r, sem) for r in df.itertuples(index=False)])
    res = pd.DataFrame([r for r in results if r is not None])
    PRICES_PATH.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(PRICES_PATH, index=False)
    logger.info("Got intraday prices for %d/%d markets -> %s", len(res), len(df), PRICES_PATH)
    return res


def report(res: pd.DataFrame) -> None:
    s = res["settlement"].to_numpy(dtype=float)
    print("\n" + "=" * 64)
    print("MARKET ACCURACY vs TIME-TO-RESOLUTION  (n markets = %d)" % len(res))
    print("=" * 64)
    print("  cutoff(UTC h)   coverage   market Brier   mean|move to +14|")
    p_ref = res["p+14"]
    for h in CUTOFF_OFFSETS_H:
        col = f"p{h:+d}"
        have = res[col].notna()
        n = int(have.sum())
        brier = float(np.mean((res.loc[have, col].to_numpy(dtype=float) - s[have.to_numpy()]) ** 2)) if n else float("nan")
        both = have & p_ref.notna()
        move = float((res.loc[both, col].astype(float) - res.loc[both, "p+14"].astype(float)).abs().mean()) if both.any() else float("nan")
        label = f"{h:+d}h" + ("  (D-1 noon)" if h == -12 else "  (decision)" if h == 14 else "")
        print(f"   {label:<16} {n/len(res):>6.0%}     {brier:.4f}        {move:.3f}")
    print("-" * 64)
    print("  Read: if Brier is already low at -12h, the market is efficient")
    print("  end-to-end (no timing edge). A high-then-falling Brier + large")
    print("  move-to-decision = a stale early window worth a model edge.")
    print("=" * 64)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2500)
    args = ap.parse_args()
    res = asyncio.run(run(args.limit))
    if not res.empty:
        report(res)


if __name__ == "__main__":
    main()
