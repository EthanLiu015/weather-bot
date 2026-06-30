"""Rough dollar P&L estimate for presence-based MM in the quiet niche.

daily P&L (per market) = daily_volume x capture_fraction x net_per_contract

  * capture_fraction & net_per_contract: from the queue fill model on the
    collected book+trades, at a given front-of-queue priority phi.
  * daily_volume: real per-market daily contracts from the historical Kalshi data.

The niche = thin + slow markets with positive net@5s at modest priority (where
you're front-of-queue by presence, no latency race). Reports conservative
(phi=0.25) and central (phi=0.5) cases. PARTIAL data -> order-of-magnitude only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.research.fill_model import HORIZONS, _side_events, simulate_side
from bot.research.mm_edge import BOOK_DIR, TRADE_DIR, _load, enrich

DEPTH_MAX = 200       # "thin" touch depth (contracts) — front-of-queue by presence
CHURN_MAX = 130       # "slow" best-quote updates/min — uncontested
DAYS_PER_MONTH = 26


def _market_sim(book_by, yes_by, no_by, tickers, q, phi):
    import random
    rng = random.Random(0)
    filled = touch = 0.0
    net = {h: 0.0 for h in HORIZONS}
    for tk in tickers:
        b = book_by.get(tk)
        if b is None:
            continue
        for side_trades, pcol, scol in ((yes_by.get(tk), "yes_ask", "yes_ask_sz"),
                                        (no_by.get(tk), "yes_bid", "yes_bid_sz")):
            if side_trades is None or side_trades.empty:
                continue
            f, n, tch = simulate_side(_side_events(b, side_trades, pcol, scol), q, phi=phi, rng=rng)
            filled += f
            touch += tch
            for h in HORIZONS:
                net[h] += n[h]
    return filled, touch, net


def _hist_daily_volume() -> pd.Series:
    """Typical per-market daily contracts from the historical settled markets."""
    try:
        p = pd.read_parquet("data/historical/kalshi_prices.parquet")
    except Exception:
        return pd.Series(dtype=float)
    p["vol"] = pd.to_numeric(p["volume"], errors="coerce")
    daily = p.groupby(["series", "date"])["vol"].sum().reset_index()
    return daily.groupby("series")["vol"].mean()


def run() -> None:
    book = _load(BOOK_DIR)
    trades = _load(TRADE_DIR)
    print("=" * 78)
    print("PRESENCE-BASED MM — ROUGH PROFIT ESTIMATE (quiet niche)")
    print("=" * 78)
    if book.empty or trades.empty:
        print("  not enough data yet")
        return
    t = enrich(book, trades, horizons=HORIZONS).dropna(subset=[f"net_{h}" for h in HORIZONS])
    book = book.copy(); book["market"] = book["ticker"].str.split("-").str[0]
    t = t.copy(); t["market"] = t["ticker"].str.split("-").str[0]
    book_by = {tk: g.sort_values("ts") for tk, g in book.groupby("ticker")}
    yes_by = {tk: g for tk, g in t[t["taker_side"] == "yes"].groupby("ticker")}
    no_by = {tk: g for tk, g in t[t["taker_side"] == "no"].groupby("ticker")}
    hist_vol = _hist_daily_volume()

    # identify the quiet niche markets (thin + slow)
    niche = []
    for mkt, mb in book.groupby("market"):
        span_min = (mb["ts"].max() - mb["ts"].min()) / 60.0
        if span_min <= 0:
            continue
        depth = ((mb["yes_bid_sz"].fillna(0) + mb["yes_ask_sz"].fillna(0)) / 2.0)
        depth = depth[depth > 0].mean()
        churn = len(mb) / span_min
        if depth < DEPTH_MAX and churn < CHURN_MAX:
            niche.append((mkt, list(mb["ticker"].unique())))

    for phi, label in ((0.25, "conservative phi=0.25"), (0.5, "central phi=0.5")):
        print(f"\n  --- {label} ---")
        print(f"    {'market':<12} {'day vol':>9} {'capture':>7} {'net¢/ct':>7} {'$/day':>8}")
        total_day = 0.0
        for mkt, tickers in niche:
            filled, touch, net = _market_sim(book_by, yes_by, no_by, tickers, 10, phi)
            if touch <= 0 or filled <= 0:
                continue
            cap = filled / touch
            e_c = net[5] / filled                       # net cents / filled contract @5s
            dv = float(hist_vol.get(mkt, np.nan))
            if not np.isfinite(dv):
                continue
            day_usd = dv * cap * e_c / 100.0            # $/day for this market
            total_day += day_usd
            print(f"    {mkt:<12} {dv:>9,.0f} {cap:>6.1%} {e_c:>+7.2f} {day_usd:>+8.2f}")
        print(f"    {'NICHE TOTAL':<12} {'':>9} {'':>7} {'':>7} {total_day:>+8.2f}  "
              f"(~${total_day * DAYS_PER_MONTH:,.0f}/month)")

    print("-" * 78)
    print("  Caveats: PARTIAL ~2h data; net@5s marks unwind at mid (real exit costs")
    print("  some spread); historical daily volume; ignores inventory risk + downtime.")
    print("  Order-of-magnitude only — the full multi-day collection will firm it up.")
    print("=" * 78)


if __name__ == "__main__":
    run()
