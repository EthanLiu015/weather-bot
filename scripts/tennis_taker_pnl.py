"""Taker P&L for the order-flow momentum signal (plans/tennis-mm-next-steps.md
Phase 4a). tennis_momentum_signal.py found real mid-to-mid predictability
(corr ~0.46 @ 30s) — this asks whether it survives actually trading it: enter
by crossing the spread (pay ask, not mid) `--latency_ms` after a trade prints
with a sign, exit by crossing the spread again `H` seconds after *that*, net
of `TAKER_FEE_COEF = 0.07` (backtest/track_b.py) on both legs.

`--latency_ms 0` (default) is the original optimistic-instant-reaction case.
Positive latency simulates the delay between seeing the signal and actually
getting an order into the book (network + API + matching-queue time) by
looking up the entry price at `signal_ts + latency` instead of `signal_ts` —
using the same real captured book states, just a later one. If the edge is
other fast reactors exploiting slow ones, realistic latency should visibly
erode it; if it barely moves, the edge is more likely a genuine multi-second
drift big enough to survive a slow reaction.

  side "yes" (trade_sign +1, yes bought): buy yes at ask, sell yes at bid
    later. pnl = bid_exit - ask_entry - fee(ask_entry) - fee(bid_exit)
  side "no"  (trade_sign -1, yes sold): buy no at (1 - yes_bid), sell no at
    (1 - yes_ask) later, same fee treatment.

    PYTHONPATH=. python scripts/tennis_taker_pnl.py [--days ...] [--top N] [--latency_ms 0]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from backtest.track_b import TAKER_FEE_COEF, kalshi_fee
from scripts.mm_feasibility import TICKS_DIR, available_days

HORIZONS_S = [5, 30, 120]


class TakerState:
    __slots__ = ("yes_book", "no_book", "prev_volume", "bids", "asks", "ts_list", "signals")

    def __init__(self) -> None:
        self.yes_book: dict[float, float] = {}
        self.no_book: dict[float, float] = {}
        self.prev_volume: float | None = None
        self.bids: list[float] = []
        self.asks: list[float] = []
        self.ts_list: list = []
        self.signals: list[dict] = []


def simulate_day(df: pd.DataFrame) -> dict[str, TakerState]:
    df = df.sort_values("ts")
    states: dict[str, TakerState] = {}

    for row in df.itertuples():
        st = states.get(row.ticker)
        if st is None:
            st = states[row.ticker] = TakerState()
        ts = row.ts

        if row.type == "snapshot":
            st.yes_book = {round(float(p), 2): s for p, s in json.loads(row.yes_book_json)}
            st.no_book = {round(float(p), 2): s for p, s in json.loads(row.no_book_json)}
        elif row.type == "delta":
            book = st.yes_book if row.side == "yes" else st.no_book
            p = round(float(row.price), 2)
            new_size = book.get(p, 0.0) + row.delta_fp
            if new_size <= 0:
                book.pop(p, None)
            else:
                book[p] = new_size
        elif row.type == "ticker" and row.last_price is not None and row.volume_fp is not None:
            if st.prev_volume is not None:
                trade_size = row.volume_fp - st.prev_volume
                if trade_size > 0 and row.yes_bid is not None and row.yes_ask is not None:
                    tp = round(row.last_price, 2)
                    bp_yes = max(st.yes_book) if st.yes_book else None
                    bp_no = max(st.no_book) if st.no_book else None
                    sign = 0
                    if bp_yes is not None and abs(tp - bp_yes) < 1e-6:
                        sign = -1
                    elif bp_no is not None and abs(tp - round(1 - bp_no, 2)) < 1e-6:
                        sign = 1
                    if sign != 0:
                        side = "yes" if sign == 1 else "no"
                        st.signals.append({"ticker": row.ticker, "side": side, "ts": ts})
            st.prev_volume = row.volume_fp

        if row.yes_bid is not None and row.yes_ask is not None:
            st.bids.append(row.yes_bid)
            st.asks.append(row.yes_ask)
            st.ts_list.append(ts)

    return states


def compute_pnl(states: dict[str, TakerState], horizons: list[int], latency_ms: int = 0) -> pd.DataFrame:
    out = []
    latency_delta = np.timedelta64(latency_ms, "ms")
    for ticker, st in states.items():
        if len(st.ts_list) < 2 or not st.signals:
            continue
        ts_arr = pd.DatetimeIndex(st.ts_list).values
        bids = np.array(st.bids)
        asks = np.array(st.asks)
        last_ts = ts_arr[-1]
        for sig in st.signals:
            signal_ts64 = pd.Timestamp(sig["ts"]).to_datetime64()
            entry_ts64 = signal_ts64 + latency_delta
            if entry_ts64 > last_ts:
                continue
            entry_idx = min(np.searchsorted(ts_arr, entry_ts64), len(bids) - 1)
            side = sig["side"]
            entry = asks[entry_idx] if side == "yes" else round(1 - bids[entry_idx], 2)
            row = dict(sig, entry=entry, entry_ts_actual=entry_ts64)
            entry_fee = kalshi_fee(1.0, entry, TAKER_FEE_COEF)
            for h in horizons:
                target = entry_ts64 + np.timedelta64(h, "s")
                if target > last_ts:
                    row[f"pnl_{h}s"] = np.nan
                    continue
                idx = min(np.searchsorted(ts_arr, target), len(bids) - 1)
                if side == "yes":
                    exit_price = bids[idx]
                else:
                    exit_price = round(1 - asks[idx], 2)
                exit_fee = kalshi_fee(1.0, exit_price, TAKER_FEE_COEF)
                row[f"pnl_{h}s"] = (exit_price - entry) - entry_fee - exit_fee
            out.append(row)
    return pd.DataFrame(out)


def report(df: pd.DataFrame, horizons: list[int]) -> None:
    print(f"\n=== {len(df)} taker signals across {df['ticker'].nunique()} market-days ===")
    for h in horizons:
        col = f"pnl_{h}s"
        d = df[col].dropna()
        if d.empty:
            continue
        win_rate = (d > 0).mean()
        se = d.std() / np.sqrt(len(d))
        t_stat = d.mean() / se if se > 0 else np.nan
        print(f"horizon {h}s: n={len(d)} mean={d.mean():+.5f} median={d.median():+.5f} "
              f"win_rate={win_rate:.1%} t_stat={t_stat:+.2f}")
        by_side = df.dropna(subset=[col]).groupby("side")[col].agg(["mean", "count"])
        print(f"  by side:\n{by_side.to_string()}".replace("\n", "\n  "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=None)
    ap.add_argument("--top", type=int, default=None)
    ap.add_argument("--latency_ms", type=int, default=0)
    args = ap.parse_args()

    days = args.days.split(",") if args.days else available_days()
    all_states: dict[str, TakerState] = {}

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
        states = simulate_day(df)
        for k, v in states.items():
            all_states[f"{day}:{k}"] = v

    pnl = compute_pnl(all_states, HORIZONS_S, latency_ms=args.latency_ms)
    if pnl.empty:
        print("no signals")
        return
    print(f"\n[latency_ms={args.latency_ms}]")
    report(pnl, HORIZONS_S)
    out_path = f"data/capture/tennis_taker_pnl_lat{args.latency_ms}ms.parquet"
    pnl.to_parquet(out_path)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
