"""Queue-based fill model: turn the optimistic MM edge into a realistic one.

mm_edge assumes you capture EVERY passive fill. In reality you rest at the back
of the queue and only get filled after the size ahead of you trades through —
before the price moves away. This simulates a maker continuously resting `q`
contracts at the best bid/ask (FIFO, re-joining the back after each fill / price
move) and credits the REAL per-fill edge (from mm_edge.enrich) only on contracts
actually filled. That exposes two effects:

  * capacity   — what fraction of touch volume you actually capture
  * selection  — whether the flow you DO fill is more toxic (lower net/contract)
                 than the capture-all optimistic average

Data caveat: we log top-of-book only, so the queue is approximated (depth ahead
decremented by trades; other makers' cancels ignored = conservative). First-order.

    PYTHONPATH=. python -m bot.research.fill_model
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd

from bot.research.mm_edge import BOOK_DIR, TRADE_DIR, _load, enrich

HORIZONS = [0, 1, 5]
MAKER_SIZES = [1, 10, 50]
PHIS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]  # front-of-queue (win-the-race) probabilities


def simulate_side(events, q: float, horizons=HORIZONS, phi: float = 0.0, rng=None):
    """FIFO queue sim for one side. `events` is time-ordered:
      ('q', price, depth)            best price on this side is `price`, size `depth`
      ('x', price, size, net_dict)   a trade of `size` hit this side at `price`

    `phi` is the probability that on (re)joining a level we win the race to the
    FRONT (no queue ahead); otherwise we join the back (behind `depth`). phi=0 is
    a pure passive maker; phi=1 captures everything (the optimistic bound).
    Returns (filled_contracts, {h: net_credited}, total_touch_volume)."""
    rng = rng or random.Random(0)

    def join_qa(depth):
        return 0.0 if (phi > 0 and rng.random() < phi) else depth

    filled = 0.0
    net = {h: 0.0 for h in horizons}
    touch = 0.0
    level = None
    qa = 0.0          # contracts ahead of our resting order
    rem = 0.0         # our unfilled resting size
    last_depth = 0.0  # most recent displayed depth at this level
    seen = False
    for ev in events:
        if ev[0] == "q":
            _, price, depth = ev
            seen = True
            last_depth = depth
            if price != level:           # price moved -> join the new level
                level, qa, rem = price, join_qa(depth), q
        else:
            if not seen:
                continue
            _, price, size, net_d = ev
            touch += size
            if price != level:
                continue
            s = size
            while s > 0:
                if qa > 0:               # clear the queue ahead first (FIFO)
                    eat = min(qa, s); qa -= eat; s -= eat
                elif rem > 0:            # then our resting order fills
                    f = min(rem, s); rem -= f; s -= f; filled += f
                    for h in horizons:
                        net[h] += f * net_d[h]
                    if rem <= 0:         # filled -> re-join
                        rem, qa = q, join_qa(last_depth)
                else:
                    break
    return filled, net, touch


def _simulate_all(book_by, yes_by, no_by, q, phi):
    filled = touch = 0.0
    net = {h: 0.0 for h in HORIZONS}
    rng = random.Random(0)
    for tk, b in book_by.items():
        for side_trades, pcol, scol in ((yes_by.get(tk), "yes_ask", "yes_ask_sz"),
                                        (no_by.get(tk), "yes_bid", "yes_bid_sz")):
            if side_trades is None or side_trades.empty:
                continue
            evs = _side_events(b, side_trades, pcol, scol)
            f, n, tch = simulate_side(evs, q, phi=phi, rng=rng)
            filled += f
            touch += tch
            for h in HORIZONS:
                net[h] += n[h]
    cap = filled / touch if touch else 0.0
    realized = {h: (net[h] / filled if filled else float("nan")) for h in HORIZONS}
    return cap, realized


def _side_events(book_t: pd.DataFrame, trades_t: pd.DataFrame, price_col, size_col):
    evs = []
    for ts, p, d in zip(book_t["ts"], book_t[price_col], book_t[size_col]):
        if pd.notna(p) and pd.notna(d):
            evs.append((ts, "q", int(p), float(d)))
    for r in trades_t.itertuples(index=False):
        net_d = {h: getattr(r, f"net_{h}") for h in HORIZONS}
        evs.append((r.ts, "x", int(round(r.price_c)), float(r.count), net_d))
    evs.sort(key=lambda e: e[0])
    return [e[1:] for e in evs]


def run() -> None:
    book = _load(BOOK_DIR)
    trades = _load(TRADE_DIR)
    print("=" * 70)
    print("FILL-RATE MODEL  (back-of-queue maker; capacity + selection)")
    print("=" * 70)
    if book.empty or trades.empty:
        print("  not enough data yet")
        return
    t = enrich(book, trades, horizons=HORIZONS)
    # Drop trades we can't score (no book mid within the horizon — boundary cases)
    # so the optimistic and queue-filled edges are compared on the same set.
    t = t.dropna(subset=[f"net_{h}" for h in HORIZONS]).copy()
    w = t["count"].to_numpy()
    opt = {h: float(np.average(t[f"net_{h}"].to_numpy(), weights=w)) for h in HORIZONS}
    span_h = (book["ts"].max() - book["ts"].min()) / 3600
    print(f"  {len(book):,} book rows, {len(t):,} trades, ~{span_h:.1f}h  (PARTIAL)")
    print(f"  optimistic (capture-all) net/contract:  " +
          "  ".join(f"{h}s {opt[h]:+.2f}" for h in HORIZONS))
    print("-" * 70)

    book_by = {tk: g.sort_values("ts") for tk, g in book.groupby("ticker")}
    yes_by = {tk: g for tk, g in t[t["taker_side"] == "yes"].groupby("ticker")}
    no_by = {tk: g for tk, g in t[t["taker_side"] == "no"].groupby("ticker")}

    print("  PASSIVE (back-of-queue, phi=0): capacity + fill selection")
    print("  size  capture%   realized net/contract (filled only)")
    for q in MAKER_SIZES:
        cap, realized = _simulate_all(book_by, yes_by, no_by, q, phi=0.0)
        rstr = "  ".join(f"{h}s {realized[h]:+.2f}" for h in HORIZONS)
        print(f"  {q:>4}  {cap:>6.1%}   {rstr}")
    print("-" * 70)

    # Break-even: how much queue priority (front-of-queue fraction phi) is needed?
    q = 10
    print(f"  QUEUE-PRIORITY SWEEP (size {q}): realized net vs front-of-queue prob phi")
    print("   phi    capture%   net@1s   net@5s")
    curve = []
    for phi in PHIS:
        cap, realized = _simulate_all(book_by, yes_by, no_by, q, phi=phi)
        curve.append((phi, realized[5]))
        print(f"   {phi:>4.0%}   {cap:>6.1%}   {realized[1]:>+6.2f}   {realized[5]:>+6.2f}")
    # interpolate the phi where net@5s crosses 0
    be = None
    for (p0, n0), (p1, n1) in zip(curve, curve[1:]):
        if (n0 < 0) != (n1 < 0) and n1 != n0:
            be = p0 + (0 - n0) * (p1 - p0) / (n1 - n0)
            break
    print("-" * 70)
    if be is None:
        sign = "always positive" if curve[0][1] >= 0 else "never positive"
        print(f"  break-even front-of-queue phi: {sign} across the sweep")
    else:
        print(f"  break-even front-of-queue phi (net@5s = 0):  ~{be:.0%}")
        print(f"  => you must win the race to the front ~{be:.0%} of the time just to break even.")
    print("  phi achievable WITHOUT speed only where the queue is thin / uncontested")
    print("  (quiet markets/hours) — otherwise it is a latency race.")
    print("=" * 70)


if __name__ == "__main__":
    run()
