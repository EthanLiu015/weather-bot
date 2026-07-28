"""Size/depth cost for the order-flow taker edge (plans/tennis-mm-next-steps.md
Phase 4a). tennis_taker_pnl.py only ever tests 1 contract against the best
bid/ask — the same "assumes infinite depth at the top price" optimism that
mm_feasibility.py's queue simulator was built to avoid on the maker side.
This walks the actual resting-order ladder to get a size-weighted average
execution price, for a range of sizes, and checks whether the edge found at
1 contract (plans doc: "no" side cluster t-stat 7.2-8.0, holds under latency)
survives being large enough to matter.

Same trade-sign detection as tennis_taker_pnl.py, latency fixed at 0 (already
shown not to matter — this isolates the size question). Single forward pass
per ticker: when a signal fires, immediately walk the current ladder for the
entry VWAP at each size; push exit targets (entry_ts + horizon) onto a
per-ticker min-heap; as the replay crosses each target time, walk the ladder
*then* for the exit VWAP. No pre-storage of ladder history — snapshots are
taken exactly when needed and discarded.

  side "yes" (buy yes): consumes the NO book from its best level down
    (yes_ask = 1 - no_price per level). Exit sells into the YES book.
  side "no"  (buy no):  consumes the YES book from its best level down
    (no_ask = 1 - yes_price per level). Exit sells into the NO book.
  A size with insufcient resting depth to fill is recorded as unfillable
  (NaN), not silently filled at a worse price than exists.

    PYTHONPATH=. python scripts/tennis_size_cost.py [--days ...] [--top N] [--sizes 1,5,20,50,100]
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json

import numpy as np
import pandas as pd

from backtest.track_b import TAKER_FEE_COEF, kalshi_fee
from scripts.mm_feasibility import TICKS_DIR, available_days

HORIZONS_S = [5, 30, 120]
DEFAULT_SIZES = [1, 5, 20, 50, 100]


def ladder_vwap(book: dict[float, float], size: float) -> float | None:
    """Walk resting levels best-first; None if total depth < size."""
    remaining = size
    cost = 0.0
    for price in sorted(book, reverse=True):
        avail = book[price]
        take = min(avail, remaining)
        cost += take * price
        remaining -= take
        if remaining <= 0:
            return cost / size
    return None


def simulate_day(df: pd.DataFrame, sizes: list[int], horizons: list[int]) -> list[dict]:
    df = df.sort_values("ts")
    yes_books: dict[str, dict[float, float]] = {}
    no_books: dict[str, dict[float, float]] = {}
    prev_volume: dict[str, float] = {}
    heaps: dict[str, list] = {}
    signals: dict[int, dict] = {}
    counter = itertools.count()

    for row in df.itertuples():
        t = row.ticker
        yes_book = yes_books.setdefault(t, {})
        no_book = no_books.setdefault(t, {})
        heap = heaps.setdefault(t, [])
        ts = row.ts
        ts64 = pd.Timestamp(ts).to_datetime64()

        # resolve any pending exit targets whose time has arrived, using the
        # book state as of just before this row's own update
        while heap and heap[0][0] <= ts64:
            _, sig_idx, size, h = heapq.heappop(heap)
            sig = signals[sig_idx]
            book_for_exit = yes_book if sig["side"] == "yes" else no_book
            px = ladder_vwap(book_for_exit, size)
            sig.setdefault("exit", {})[(size, h)] = px

        if row.type == "snapshot":
            yes_books[t] = {round(float(p), 2): s for p, s in json.loads(row.yes_book_json)}
            no_books[t] = {round(float(p), 2): s for p, s in json.loads(row.no_book_json)}
        elif row.type == "delta":
            book = yes_books[t] if row.side == "yes" else no_books[t]
            p = round(float(row.price), 2)
            new_size = book.get(p, 0.0) + row.delta_fp
            if new_size <= 0:
                book.pop(p, None)
            else:
                book[p] = new_size
        elif row.type == "ticker" and row.last_price is not None and row.volume_fp is not None:
            pv = prev_volume.get(t)
            if pv is not None:
                trade_size = row.volume_fp - pv
                if trade_size > 0:
                    yb, nb = yes_books[t], no_books[t]
                    tp = round(row.last_price, 2)
                    bp_yes = max(yb) if yb else None
                    bp_no = max(nb) if nb else None
                    sign = 0
                    if bp_yes is not None and abs(tp - bp_yes) < 1e-6:
                        sign = -1
                    elif bp_no is not None and abs(tp - round(1 - bp_no, 2)) < 1e-6:
                        sign = 1
                    if sign != 0:
                        side = "yes" if sign == 1 else "no"
                        # entry always crosses the *opposite* book (buying yes
                        # matches resting no-buys and vice versa), so the raw
                        # ladder price is always the complement of what we pay
                        entry_book = nb if side == "yes" else yb
                        entries = {}
                        for sz in sizes:
                            px = ladder_vwap(entry_book, sz)
                            entries[sz] = None if px is None else round(1 - px, 2)
                        sig_idx = next(counter)
                        signals[sig_idx] = {"ticker": t, "side": side, "ts": ts, "entry": entries}
                        for h in horizons:
                            target = ts64 + np.timedelta64(h, "s")
                            for sz in sizes:
                                heapq.heappush(heap, (target, sig_idx, sz, h))
            prev_volume[t] = row.volume_fp

    return list(signals.values())


def build_pnl_frame(signals: list[dict], sizes: list[int], horizons: list[int]) -> pd.DataFrame:
    rows = []
    for sig in signals:
        exits = sig.get("exit", {})
        for sz in sizes:
            entry = sig["entry"].get(sz)
            for h in horizons:
                exit_px = exits.get((sz, h))
                fillable = entry is not None and exit_px is not None
                pnl = None
                if fillable:
                    entry_fee = kalshi_fee(sz, entry, TAKER_FEE_COEF)
                    exit_fee = kalshi_fee(sz, exit_px, TAKER_FEE_COEF)
                    pnl = (exit_px - entry) - (entry_fee + exit_fee) / sz
                rows.append({"ticker": sig["ticker"], "side": sig["side"], "size": sz,
                             "horizon": h, "entry": entry, "exit": exit_px,
                             "fillable": fillable, "pnl_per_contract": pnl})
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, sizes: list[int], horizons: list[int]) -> None:
    for h in horizons:
        print(f"\n=== horizon {h}s ===")
        for sz in sizes:
            d = df[(df["size"] == sz) & (df["horizon"] == h)]
            if d.empty:
                continue
            fill_rate = d["fillable"].mean()
            for side in ["yes", "no"]:
                ds = d[(d["side"] == side) & d["fillable"]]["pnl_per_contract"]
                n_signals = (d["side"] == side).sum()
                if ds.empty:
                    print(f"  size {sz:>4} side {side:>3}: 0/{n_signals} fillable")
                    continue
                fr = len(ds) / n_signals if n_signals else float("nan")
                print(f"  size {sz:>4} side {side:>3}: fillable {len(ds):>6}/{n_signals:<6} ({fr:.1%})  "
                      f"mean={ds.mean():+.5f} median={ds.median():+.5f} win_rate={(ds>0).mean():.1%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=None)
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    days = args.days.split(",") if args.days else available_days()
    all_signals: list[dict] = []
    total_signals_detected = 0

    for day in days:
        path = f"{TICKS_DIR}/date={day}/ticks.parquet"
        df = pd.read_parquet(path)
        if "type" not in df.columns:
            print(f"{day}: no depth data (pre-v2 schema), skipping")
            continue
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
        if args.top:
            keep = df["ticker"].value_counts().head(args.top).index
            df = df[df["ticker"].isin(keep)]
        print(f"{day}: {len(df):,} rows, {df['ticker'].nunique()} markets")
        sigs = simulate_day(df, sizes, HORIZONS_S)
        for s in sigs:
            s["ticker"] = f"{day}:{s['ticker']}"
        total_signals_detected += len(sigs)
        all_signals.extend(sigs)

    print(f"\n{total_signals_detected} signals detected total")
    pnl = build_pnl_frame(all_signals, sizes, HORIZONS_S)
    if pnl.empty:
        print("no signals")
        return
    report(pnl, sizes, HORIZONS_S)
    pnl.to_parquet("data/capture/tennis_size_cost.parquet")
    print("\nsaved -> data/capture/tennis_size_cost.parquet")


if __name__ == "__main__":
    main()
