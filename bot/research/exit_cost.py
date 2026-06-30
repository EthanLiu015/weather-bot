"""Exit-cost model: the make-or-break assumption for MM viability.

mm_edge marks each position at the MID after Δ — i.e. assumes you exit for free
at fair value. Real MM must actually flatten, and you get adversely selected INTO
positions you then have to pay to exit. Modeled from the book directly:

  entry: passive at the touch (maker fee), in yes-terms at `price_c`, position
         sign = +1 long (taker bought no) / -1 short (taker bought yes).
  exit at Δ, three ways:
    * mid       — mark at mid (mm_edge's free-exit assumption)
    * passive   — post on the FAVOURABLE side, capture the 2nd half-spread
                  (maker fee) — the best case, IF your exit order fills
    * aggressive— CROSS to the unfavourable touch (taker fee = 4x maker) —
                  what you must do when the price ran away (adverse)
    * realistic — passive when the move was favourable, forced-cross when adverse
                  (the asymmetry that makes real edge < the mid mark)

If realistic net per contract is <= 0, presence-based MM here does not survive
exit costs.  PARTIAL data -> order-of-magnitude.

    PYTHONPATH=. python -m bot.research.exit_cost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.research.mm_edge import (BOOK_DIR, TRADE_DIR, HORIZONS_S, _load, enrich,
                                  maker_fee_cents)

HORIZONS = [1, 5, 30]
DEPTH_MAX, CHURN_MAX = 200, 130


def taker_fee_cents(price_cents: np.ndarray) -> np.ndarray:
    return 4.0 * maker_fee_cents(price_cents)  # taker = 0.07*P(1-P); maker = 25% of that


def exit_nets(t: pd.DataFrame, h: int):
    """Return (mid, passive, aggressive, realistic) net-cents/contract arrays at Δ=h."""
    sign = t["sign"].to_numpy()
    price = t["price_c"].to_numpy()
    bid = t[f"bid_{h}"].to_numpy()
    ask = t[f"ask_{h}"].to_numpy()
    mid = t[f"mid_{h}"].to_numpy()
    entry_fee = maker_fee_cents(price)

    long = sign > 0
    pass_mark = np.where(long, ask, bid)   # sell at ask (long) / buy at bid (short)
    aggr_mark = np.where(long, bid, ask)   # cross: sell at bid (long) / buy at ask (short)

    net_mid = sign * (mid - price) - entry_fee
    net_pass = sign * (pass_mark - price) - entry_fee - maker_fee_cents(pass_mark)
    net_aggr = sign * (aggr_mark - price) - entry_fee - taker_fee_cents(aggr_mark)
    net_real = np.where(net_mid >= 0, net_pass, net_aggr)
    return net_mid, net_pass, net_aggr, net_real


def _wavg(v, w):
    ok = np.isfinite(v)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else float("nan")


def run() -> None:
    book = _load(BOOK_DIR)
    trades = _load(TRADE_DIR)
    print("=" * 74)
    print("EXIT-COST MODEL  (net ¢/contract, capture-all; does the edge survive?)")
    print("=" * 74)
    if book.empty or trades.empty:
        print("  not enough data yet")
        return
    need = [f"{p}_{h}" for h in HORIZONS for p in ("mid", "bid", "ask")]
    t = enrich(book, trades, horizons=sorted(set(HORIZONS_S + HORIZONS))).dropna(subset=need).copy()
    t["market"] = t["ticker"].str.split("-").str[0]
    w = t["count"].to_numpy()

    print("  exit at Δ    mid    passive   aggressive   REALISTIC")
    for h in HORIZONS:
        m, p, a, r = exit_nets(t, h)
        print(f"    {h:>3}s     {_wavg(m,w):>+5.2f}   {_wavg(p,w):>+6.2f}   {_wavg(a,w):>+8.2f}   "
              f"{_wavg(r,w):>+8.2f}")
    print("-" * 74)

    # Does the quiet niche survive exit costs? (realistic net @5s, by market)
    print("  quiet-niche markets — realistic net@5s vs the mid mark:")
    print(f"    {'market':<12} {'mid':>6} {'passive':>8} {'realistic':>10}")
    h = 5
    for mkt, g in t.groupby("market"):
        mb = book[book["ticker"].str.startswith(mkt)]
        span = (mb["ts"].max() - mb["ts"].min()) / 60.0
        if span <= 0:
            continue
        depth = ((mb["yes_bid_sz"].fillna(0) + mb["yes_ask_sz"].fillna(0)) / 2.0)
        depth = depth[depth > 0].mean()
        churn = len(mb) / span
        if not (depth < DEPTH_MAX and churn < CHURN_MAX):
            continue
        gw = g["count"].to_numpy()
        m, p, a, r = exit_nets(g, h)
        print(f"    {mkt:<12} {_wavg(m,gw):>+6.2f} {_wavg(p,gw):>+8.2f} {_wavg(r,gw):>+10.2f}")
    print("-" * 74)
    print("  REALISTIC = passive exit when the move was favourable, forced cross when")
    print("  adverse. If it's <= 0, presence-based MM does not survive exit costs here.")
    print("=" * 74)


if __name__ == "__main__":
    run()
