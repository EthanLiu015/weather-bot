"""Full round-trip market-making replay backtest — the decisive number.

Simulates a two-sided maker per ticker against the recorded book+trade tape,
combining every effect the earlier analyses isolated:
  * quote both sides at the touch (size q), joining the back of the queue
    (front with prob phi = queue priority);
  * fills are FIFO/queue-aware (a trade clears the queue ahead, then us);
  * a bid fill buys yes (inventory+), an ask fill sells yes (inventory-) — the
    opposite quote is the natural PASSIVE exit that captures the spread;
  * if |inventory| hits a limit, FORCE-FLATTEN by crossing (taker fee) — the
    adverse-selection / inventory cost;
  * maker fee on passive fills, taker fee on forced flattens; residual inventory
    marked at the final mid.

Our small quotes are assumed not to move the tape (standard small-size replay).
Quoting only AT existing touches (not improving empty sides) makes this a
CONSERVATIVE estimate of the presence edge. PARTIAL data -> order-of-magnitude.

    PYTHONPATH=. python -m bot.research.backtest
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from bot.research.exit_cost import taker_fee_cents
from bot.research.mm_edge import BOOK_DIR, TRADE_DIR, _load, maker_fee_cents

Q = 10            # quote size per side (contracts)
MAX_INV = 50      # inventory limit before a forced flatten
PHIS = [0.0, 0.25, 0.5]
DEPTH_MAX, CHURN_MAX, DAYS = 200, 130, 26


def _mfee(price_c):
    return float(maker_fee_cents(np.array([price_c]))[0])


def _tfee(price_c):
    return float(taker_fee_cents(np.array([price_c]))[0])


def backtest_ticker(events, q=Q, max_inv=MAX_INV, phi=0.0, rng=None):
    """Returns (pnl_cents, traded_contracts, fees_cents)."""
    rng = rng or random.Random(0)

    def join(depth):
        return 0.0 if (phi > 0 and rng.random() < phi) else depth

    inv = cash = fees = traded = 0.0
    bid_lvl = ask_lvl = None
    bid_qa = ask_qa = bid_rem = ask_rem = 0.0
    bid_depth = ask_depth = 0.0
    last_mid = last_bid = last_ask = None

    for ev in events:
        if ev[0] == "q":
            _, bid, bsz, ask, asz = ev
            last_bid, last_ask, last_mid = bid, ask, (bid + ask) / 2.0
            bid_depth, ask_depth = bsz, asz
            if bid != bid_lvl:
                bid_lvl, bid_qa, bid_rem = bid, join(bsz), q
            if ask != ask_lvl:
                ask_lvl, ask_qa, ask_rem = ask, join(asz), q
            continue

        _, price, side, size = ev
        if side == "no" and price == bid_lvl:        # hits our bid -> we BUY yes
            s = size
            eat = min(bid_qa, s); bid_qa -= eat; s -= eat
            while s > 0 and bid_rem > 0:
                f = min(bid_rem, s); bid_rem -= f; s -= f
                inv += f; cash -= f * bid_lvl; fees += f * _mfee(bid_lvl); traded += f
                if bid_rem <= 0:
                    bid_qa, bid_rem = join(bid_depth), q
        elif side == "yes" and price == ask_lvl:      # hits our ask -> we SELL yes
            s = size
            eat = min(ask_qa, s); ask_qa -= eat; s -= eat
            while s > 0 and ask_rem > 0:
                f = min(ask_rem, s); ask_rem -= f; s -= f
                inv -= f; cash += f * ask_lvl; fees += f * _mfee(ask_lvl); traded += f
                if ask_rem <= 0:
                    ask_qa, ask_rem = join(ask_depth), q

        if abs(inv) > max_inv and last_bid is not None:   # forced flatten (cross)
            if inv > 0:
                cash += inv * last_bid; fees += inv * _tfee(last_bid)
            else:
                cash += inv * last_ask; fees += (-inv) * _tfee(last_ask)
            inv = 0.0

    pnl = cash + (inv * last_mid if last_mid is not None else 0.0) - fees
    return pnl, traded, fees


def _ticker_events(book_t, trades_t):
    evs = []
    for ts, bb, bs, aa, as_ in zip(book_t["ts"], book_t["yes_bid"], book_t["yes_bid_sz"],
                                   book_t["yes_ask"], book_t["yes_ask_sz"]):
        if pd.notna(bb) and pd.notna(aa):
            evs.append((ts, "q", int(bb), float(bs or 0), int(aa), float(as_ or 0)))
    for r in trades_t.itertuples(index=False):
        evs.append((r.ts, "x", int(round(float(r.yes_price) * 100)), r.taker_side, float(r.count)))
    evs.sort(key=lambda e: e[0])
    return [e[1:] for e in evs]


def _hist_daily_volume():
    try:
        p = pd.read_parquet("data/historical/kalshi_prices.parquet")
        p["v"] = pd.to_numeric(p["volume"], errors="coerce")
        return p.groupby(["series", "date"])["v"].sum().groupby("series").mean()
    except Exception:
        return pd.Series(dtype=float)


def run() -> None:
    book = _load(BOOK_DIR)
    trades = _load(TRADE_DIR)
    print("=" * 76)
    print("FULL ROUND-TRIP MM BACKTEST (two-sided, queue+inventory+exit; quiet niche)")
    print("=" * 76)
    if book.empty or trades.empty:
        print("  not enough data yet")
        return
    book = book.copy(); book["market"] = book["ticker"].str.split("-").str[0]
    trades = trades.dropna(subset=["yes_price", "count", "taker_side"]).copy()
    trades = trades[trades["count"] > 0]
    trades["market"] = trades["ticker"].str.split("-").str[0]
    span_h = (book["ts"].max() - book["ts"].min()) / 3600

    # quiet niche markets
    niche = []
    for mkt, mb in book.groupby("market"):
        span = (mb["ts"].max() - mb["ts"].min()) / 60.0
        if span <= 0:
            continue
        depth = ((mb["yes_bid_sz"].fillna(0) + mb["yes_ask_sz"].fillna(0)) / 2.0)
        depth = depth[depth > 0].mean()
        if depth < DEPTH_MAX and len(mb) / span < CHURN_MAX:
            niche.append(mkt)

    book_by = {tk: g.sort_values("ts") for tk, g in book.groupby("ticker")}
    trades_by = {tk: g for tk, g in trades.groupby("ticker")}
    niche_tickers = [tk for tk in book_by if tk.split("-")[0] in niche]
    sample_vol = float(trades[trades["market"].isin(niche)]["count"].sum())
    hist = _hist_daily_volume()
    daily_vol = float(sum(hist.get(m, 0.0) for m in niche))

    print(f"  niche markets: {len(niche)} | tickers: {len(niche_tickers)} | ~{span_h:.1f}h (PARTIAL)")
    print(f"  size {Q}/side, inventory limit {MAX_INV}")
    print("   phi    sample P&L   traded   ¢/contract    ~$/month (scaled)")
    for phi in PHIS:
        rng = random.Random(0)
        pnl = traded = 0.0
        for tk in niche_tickers:
            tt = trades_by.get(tk)
            if tt is None or tt.empty:
                continue
            p, tr, _ = backtest_ticker(_ticker_events(book_by[tk], tt), phi=phi, rng=rng)
            pnl += p; traded += tr
        per_c = pnl / traded if traded else float("nan")
        sample_usd = pnl / 100.0
        # scale: realized edge per contract * niche daily volume * days
        monthly = (per_c / 100.0) * daily_vol * DAYS if np.isfinite(per_c) else float("nan")
        print(f"   {phi:>4.0%}   ${sample_usd:>+8.2f}   {traded:>7,.0f}   {per_c:>+8.2f}    ${monthly:>+10,.0f}")
    print("-" * 76)
    print("  ¢/contract = realized round-trip edge after queue, inventory, exits, fees.")
    print("  monthly scales that edge by real niche daily volume x %d days. PARTIAL data;" % DAYS)
    print("  conservative (quotes only at existing touches, full-cross flattens).")
    print("=" * 76)


if __name__ == "__main__":
    run()
