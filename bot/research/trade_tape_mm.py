"""Trades-only market-making viability probe for Kalshi temperature markets.

We have no historical order-book/quote data, but Kalshi's public /markets/trades
endpoint gives the full historical execution tape (time, price, size, taker_side)
for settled markets. That alone answers the gating question for market-making:
if we PASSIVELY absorbed every trade (always the maker, opposite the taker) and
held to settlement, would we make money net of the 25%-of-taker maker fee?

This is OPTIMISTIC on fills (assumes we win every passive fill — real queues and
competition take a cut) but HONEST on adverse selection and the overround
(informed takers picking us off, and longshots we sell that expire worthless are
both baked in). So it's a clean go/no-go: if maker P&L is negative even here,
market-making is dead; if clearly positive, it justifies building a depth logger
to measure real fill rates.

Maker fee (Kalshi, Feb 2026): 0.25 * 0.07 * P*(1-P) per contract.

Run: PYTHONPATH=. python scripts/trade_tape_mm.py [--limit N] [--fetch/--no-fetch]
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from bot.config.series import is_low_temp_series
from bot.config.settings import get_settings
from bot.research.fetch_kalshi_history import (
    LIVE_API_KEY,
    LIVE_BASE_URL,
    ReadOnlyKalshiClient,
)

logger = logging.getLogger(__name__)

TRADES_PATH = Path("data/historical/trades.parquet")
MAKER_FEE_COEF = 0.25 * 0.07  # 25% of the 0.07*P*(1-P) taker fee


def maker_fee_per_contract(price: float) -> float:
    """Kalshi maker fee per contract at executed price `price`."""
    return MAKER_FEE_COEF * price * (1.0 - price)


async def _fetch_trades_for_market(client, ticker: str, sem) -> list[dict]:
    """Paginate /markets/trades for one ticker (full historical tape)."""
    out: list[dict] = []
    cursor = None
    async with sem:
        for _ in range(50):  # hard page cap
            params = {"ticker": ticker, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            try:
                data = await client._get("/markets/trades", params=params)
            except Exception as exc:
                logger.debug("trades fetch failed for %s: %s", ticker, exc)
                break
            out.extend(data.get("trades", []))
            cursor = data.get("cursor")
            if not cursor:
                break
    return out


async def fetch_tape(limit: int, concurrency: int = 6) -> pd.DataFrame:
    settings = get_settings()
    client = ReadOnlyKalshiClient(
        api_key=LIVE_API_KEY,
        private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
        base_url=LIVE_BASE_URL,
    )
    p = pd.read_parquet("data/historical/kalshi_prices.parquet")
    p = p[~p["series"].map(is_low_temp_series)].copy()
    p["_vol"] = pd.to_numeric(p["volume"], errors="coerce").fillna(0)
    p = p[p["settlement"].notna() & (p["_vol"] > 0)].sort_values("date")
    if limit and len(p) > limit:
        p = p.tail(limit)  # most-recent markets (best trade availability)
    settle = dict(zip(p["ticker"], p["settlement"]))
    station = dict(zip(p["ticker"], p["station"]))
    logger.info("Fetching trade tape for %d markets...", len(p))

    sem = asyncio.Semaphore(concurrency)
    tapes = await asyncio.gather(*[_fetch_trades_for_market(client, t, sem) for t in p["ticker"]])

    rows = []
    for ticker, tape in zip(p["ticker"], tapes):
        for t in tape:
            rows.append({
                "ticker": ticker,
                "station": station[ticker],
                "settlement": float(settle[ticker]),
                "created_time": t.get("created_time"),
                "yes_price": float(t.get("yes_price_dollars", "nan")),
                "no_price": float(t.get("no_price_dollars", "nan")),
                "count": float(t.get("count_fp", t.get("count", 0)) or 0),
                "taker_side": t.get("taker_side"),
                "is_block": bool(t.get("is_block_trade", False)),
            })
    df = pd.DataFrame(rows)
    TRADES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TRADES_PATH, index=False)
    logger.info("Saved %d trades across %d markets -> %s", len(df), df["ticker"].nunique(), TRADES_PATH)
    return df


def maker_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Per-trade maker P&L assuming we are the passive side, held to settlement.

    taker buys YES @ yes_price  -> maker SELLS yes -> maker pnl = yes_price - settlement
    taker buys NO  @ no_price   -> maker SELLS no  -> maker pnl = settlement - yes_price
    (equivalently no_price - (1 - settlement)). Fee uses the executed contract price.
    """
    d = df.copy()
    is_yes = d["taker_side"] == "yes"
    # maker's executed YES-equivalent price and per-contract gross
    gross = np.where(is_yes, d["yes_price"] - d["settlement"], d["settlement"] - d["yes_price"])
    exec_price = np.where(is_yes, d["yes_price"], d["no_price"])
    fee = MAKER_FEE_COEF * exec_price * (1.0 - exec_price)
    d["maker_gross"] = gross
    d["maker_fee"] = fee
    d["maker_net"] = gross - fee
    d["exec_price"] = exec_price
    return d


def maker_markout(df: pd.DataFrame, horizons_s=(30, 60, 300, 1800)) -> dict:
    """Volume-weighted maker MARKOUT (cents/contract) at short horizons.

    For each fill the maker is the passive side: short YES if the taker bought
    YES, long YES if the taker bought NO, entered at the trade's yes_price. The
    markout at horizon h is the maker's mark-to-market if they unwound at the
    first trade >= h seconds later: maker_sign * (yes_price(t+h) - yes_price(t)).

    Positive => price drifts in the MAKER's favor after fills (capturable spread,
    benign flow). Negative => adverse selection (price chases the taker); a fast-
    unwinding MM still loses. This is the MM-relevant horizon, unlike hold-to-
    settlement. Net of the maker fee, an MM is viable iff captured-spread +
    markout - fee > 0.
    """
    d = df[df["count"] > 0].copy()
    d["ts"] = pd.to_datetime(d["created_time"]).astype("int64") // 10**9
    d["sign"] = np.where(d["taker_side"] == "yes", -1.0, 1.0)  # maker yes-position
    out = {}
    for h in horizons_s:
        num = 0.0
        den = 0.0
        for _, g in d.sort_values("ts").groupby("ticker", sort=False):
            ts = g["ts"].to_numpy()
            yp = g["yes_price"].to_numpy()
            sgn = g["sign"].to_numpy()
            cnt = g["count"].to_numpy()
            idx = np.searchsorted(ts, ts + h, side="left")
            valid = idx < len(ts)
            fut = np.where(valid, yp[np.clip(idx, 0, len(ts) - 1)], np.nan)
            mo = sgn * (fut - yp)
            ok = valid & np.isfinite(mo)
            num += float((mo[ok] * cnt[ok]).sum())
            den += float(cnt[ok].sum())
        out[h] = 1e2 * num / den if den else float("nan")
    return out


def report(df: pd.DataFrame) -> None:
    d = maker_pnl(df[df["count"] > 0].copy())
    c = d["count"].to_numpy()
    tot = c.sum()
    net_settle = 1e2 * float((d["maker_net"] * c).sum()) / tot
    fee_cct = 1e2 * float((d["maker_fee"] * c).sum()) / tot

    print("\n" + "=" * 66)
    print("TRADES-ONLY MARKET-MAKING PROBE")
    print("=" * 66)
    print(f"  trades: {len(d):,}   contracts: {tot:,.0f}   markets: {d['ticker'].nunique()}")
    print(f"  avg maker fee: {fee_cct:.2f} cents/contract")
    print("-" * 66)
    print("  (1) MAKER MARKOUT vs horizon  (passive side; +=favorable, -=adverse)")
    print("      the MM-relevant metric: can we capture spread before flow moves?")
    mo = maker_markout(df)
    for h, v in mo.items():
        label = f"{h}s" if h < 60 else f"{h//60}min"
        print(f"        +{label:<6} : {v:+.2f} cents/contract")
    print(f"      maker fee to beat: {fee_cct:.2f} c/ct each side")
    print("-" * 66)
    print(f"  (2) HOLD-TO-SETTLEMENT maker edge: {net_settle:+.2f} c/ct (net)")
    print("      NOT how MM works (no unwind) — measures whether the flow is")
    print("      informed at the SETTLEMENT horizon. Very negative => informed flow.")
    print("-" * 66)
    print("  READ: if markout stays >~ -fee at minute horizons, a fast-unwinding")
    print("  MM can capture spread despite settlement-horizon informedness. If")
    print("  markout is already sharply negative at +1min, the flow is toxic.")
    print("=" * 66)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--no-fetch", action="store_true", help="reuse saved trades.parquet")
    args = ap.parse_args()
    if args.no_fetch and TRADES_PATH.exists():
        df = pd.read_parquet(TRADES_PATH)
    else:
        df = asyncio.run(fetch_tape(args.limit))
    if not df.empty:
        report(df)


if __name__ == "__main__":
    main()
