"""Map the 'quiet AND profitable' market-making niche.

The fill model showed passive MM loses by a hair but only ~18% front-of-queue
priority breaks even — and that priority is free in UNCONTESTED markets (you're
first by being present, no latency race). This asks, per market: how contested
is it, and is the edge positive at low priority?

Contestedness proxies (from the book feed):
  * touch depth  — mean size resting at the best bid/ask; deep = many makers
                   ahead of you (hard to be front).
  * quote churn  — best-quote updates per minute; fast = HFT-contested.
Edge (from the queue fill model), net@5s per contract at:
  * phi=0    passive / back-of-queue
  * phi=0.25 modest priority (achievable in quiet markets)

The target niche: low depth + low churn + positive net25 (and enough volume).

    PYTHONPATH=. python -m bot.research.contestedness
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.research.fill_model import HORIZONS, _simulate_all
from bot.research.mm_edge import BOOK_DIR, TRADE_DIR, _load, enrich


def run() -> None:
    book = _load(BOOK_DIR)
    trades = _load(TRADE_DIR)
    print("=" * 86)
    print("CONTESTEDNESS x EDGE BY MARKET  (find the quiet, profitable niche)")
    print("=" * 86)
    if book.empty or trades.empty:
        print("  not enough data yet")
        return
    t = enrich(book, trades, horizons=HORIZONS).dropna(subset=[f"net_{h}" for h in HORIZONS])
    book = book.copy()
    book["market"] = book["ticker"].str.split("-").str[0]
    t = t.copy()
    t["market"] = t["ticker"].str.split("-").str[0]

    book_by = {tk: g.sort_values("ts") for tk, g in book.groupby("ticker")}
    yes_by = {tk: g for tk, g in t[t["taker_side"] == "yes"].groupby("ticker")}
    no_by = {tk: g for tk, g in t[t["taker_side"] == "no"].groupby("ticker")}
    vol_by_mkt = t.groupby("market")["count"].sum()

    rows = []
    for mkt, mb in book.groupby("market"):
        span_min = (mb["ts"].max() - mb["ts"].min()) / 60.0
        if span_min <= 0:
            continue
        depth = ((mb["yes_bid_sz"].fillna(0) + mb["yes_ask_sz"].fillna(0)) / 2.0)
        depth = depth[depth > 0].mean()
        churn = len(mb) / span_min                      # best-quote updates / min
        tickers = mb["ticker"].unique()
        bb = {tk: book_by[tk] for tk in tickers if tk in book_by}
        yb = {tk: yes_by[tk] for tk in tickers if tk in yes_by}
        nb = {tk: no_by[tk] for tk in tickers if tk in no_by}
        if not yb and not nb:
            continue
        _, r0 = _simulate_all(bb, yb, nb, 10, phi=0.0)
        _, r25 = _simulate_all(bb, yb, nb, 10, phi=0.25)
        rows.append({
            "market": mkt,
            "contracts": float(vol_by_mkt.get(mkt, 0.0)),
            "depth": depth,
            "churn_min": churn,
            "net0_5s": r0[5],
            "net25_5s": r25[5],
        })

    df = pd.DataFrame(rows)
    df = df[df["contracts"] >= 500].sort_values("churn_min")  # ignore illiquid noise
    print(f"  {'market':<12} {'contracts':>9} {'depth':>7} {'churn/min':>9} "
          f"{'net@5s phi0':>12} {'net@5s phi.25':>13}")
    for _, r in df.iterrows():
        flag = "  <- quiet+positive" if (r.net25_5s > 0 and r.churn_min < df["churn_min"].median()
                                         and r.depth < df["depth"].median()) else ""
        print(f"  {r.market:<12} {r.contracts:>9,.0f} {r.depth:>7.0f} {r.churn_min:>9.0f} "
              f"{r.net0_5s:>+12.2f} {r.net25_5s:>+13.2f}{flag}")
    print("-" * 86)
    print("  low depth + low churn = uncontested (front-of-queue without a speed race).")
    print("  Target: 'quiet+positive' markets — net@5s phi.25 > 0 in the calmer half.")
    print("=" * 86)


if __name__ == "__main__":
    run()
