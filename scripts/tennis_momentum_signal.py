"""Order-flow momentum screening test — path (a) of plans/tennis-mm-next-steps.md
Phase 4. Passive MM lost to adverse selection (scripts/mm_feasibility.py):
fills cluster right before the book reprices against the resting side. This
asks the mirror question — is that same reprice-pressure predictable a few
seconds ahead from public order-flow alone, i.e. takeable? No external odds,
no live score feed, just the existing tick capture.

At every trade print, record two signals from the book *as of that moment*:
  - imbalance: (top-of-book yes-side size - no-side size) / their sum
  - momentum: sum of the signs of the last 5 trades (this one included),
    +1 = trade printed at the ask (yes bought), -1 = printed at the bid
Then look up mid `H` seconds later and correlate. Positive correlation with
sensible magnitude = a real screening hit, worth building 4b's execution
cost model around; near-zero = the market already prices new flow near-
instantly and this path is a dead end too.

    PYTHONPATH=. python scripts/tennis_momentum_signal.py [--days ...] [--top N]
"""
from __future__ import annotations

import argparse
import json
from collections import deque

import numpy as np
import pandas as pd

from scripts.mm_feasibility import TICKS_DIR, available_days

HORIZONS_S = [5, 30, 120]
MOMENTUM_WINDOW = 5


class MomentumState:
    __slots__ = ("yes_book", "no_book", "prev_volume", "trade_signs", "mids", "ts_list", "obs")

    def __init__(self) -> None:
        self.yes_book: dict[float, float] = {}
        self.no_book: dict[float, float] = {}
        self.prev_volume: float | None = None
        self.trade_signs: deque[int] = deque(maxlen=MOMENTUM_WINDOW)
        self.mids: list[float] = []
        self.ts_list: list = []
        self.obs: list[dict] = []


def simulate_day(df: pd.DataFrame) -> dict[str, MomentumState]:
    df = df.sort_values("ts")
    states: dict[str, MomentumState] = {}

    for row in df.itertuples():
        st = states.get(row.ticker)
        if st is None:
            st = states[row.ticker] = MomentumState()
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
                if trade_size > 0:
                    tp = round(row.last_price, 2)
                    bp_yes = max(st.yes_book) if st.yes_book else None
                    bp_no = max(st.no_book) if st.no_book else None
                    sign = 0
                    if bp_yes is not None and abs(tp - bp_yes) < 1e-6:
                        sign = -1  # printed at bid: yes sold, downward pressure
                    elif bp_no is not None and abs(tp - round(1 - bp_no, 2)) < 1e-6:
                        sign = 1  # printed at ask: yes bought, upward pressure
                    if sign != 0:
                        top_yes = st.yes_book.get(bp_yes, 0.0) if bp_yes is not None else 0.0
                        top_no = st.no_book.get(bp_no, 0.0) if bp_no is not None else 0.0
                        denom = top_yes + top_no
                        imbalance = (top_yes - top_no) / denom if denom > 0 else np.nan
                        st.trade_signs.append(sign)
                        momentum = sum(st.trade_signs)
                        if row.yes_bid is not None and row.yes_ask is not None:
                            mid_now = (row.yes_bid + row.yes_ask) / 2.0
                            st.obs.append({"ticker": row.ticker, "ts": ts, "imbalance": imbalance,
                                           "momentum": momentum, "trade_sign": sign, "mid_at_obs": mid_now})
            st.prev_volume = row.volume_fp

        if row.yes_bid is not None and row.yes_ask is not None:
            st.mids.append((row.yes_bid + row.yes_ask) / 2.0)
            st.ts_list.append(ts)

    return states


def forward_returns(states: dict[str, MomentumState], horizons: list[int]) -> pd.DataFrame:
    out = []
    for ticker, st in states.items():
        if len(st.ts_list) < 2 or not st.obs:
            continue
        ts_arr = pd.DatetimeIndex(st.ts_list).values
        mids = np.array(st.mids)
        for o in st.obs:
            row = dict(o)
            obs_ts64 = pd.Timestamp(o["ts"]).to_datetime64()
            for h in horizons:
                target = obs_ts64 + np.timedelta64(h, "s")
                if target > ts_arr[-1]:
                    row[f"fwd_ret_{h}s"] = np.nan
                    continue
                idx = np.searchsorted(ts_arr, target)
                mid_after = mids[min(idx, len(mids) - 1)]
                row[f"fwd_ret_{h}s"] = mid_after - o["mid_at_obs"]
            out.append(row)
    return pd.DataFrame(out)


def report(df: pd.DataFrame, horizons: list[int]) -> None:
    print(f"\n=== {len(df)} trade-flow observations across {df['ticker'].nunique()} market-days ===")
    for h in horizons:
        col = f"fwd_ret_{h}s"
        d = df[["imbalance", "momentum", "trade_sign", col]].dropna()
        if len(d) < 30:
            continue
        n = len(d)
        corr_imb = d["imbalance"].corr(d[col])
        corr_mom = d["momentum"].corr(d[col])
        corr_sign = d["trade_sign"].corr(d[col])
        print(f"\n--- horizon {h}s (n={n}) ---")
        print(f"  corr(imbalance, fwd_ret)   = {corr_imb:+.4f}")
        print(f"  corr(momentum,  fwd_ret)   = {corr_mom:+.4f}")
        print(f"  corr(trade_sign, fwd_ret)  = {corr_sign:+.4f}")
        q = pd.qcut(d["imbalance"], 5, duplicates="drop")
        print("  fwd_ret by imbalance quintile:")
        print(d.groupby(q, observed=True)[col].agg(["mean", "count"]).to_string().replace("\n", "\n    "))
        by_sign = d.groupby("trade_sign", observed=True)[col].agg(["mean", "count"])
        print("  fwd_ret by trade_sign:")
        print(by_sign.to_string().replace("\n", "\n    "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=None)
    ap.add_argument("--top", type=int, default=None)
    args = ap.parse_args()

    days = args.days.split(",") if args.days else available_days()
    all_states: dict[str, MomentumState] = {}

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

    fr = forward_returns(all_states, HORIZONS_S)
    if fr.empty:
        print("no observations")
        return
    report(fr, HORIZONS_S)
    fr.to_parquet("data/capture/tennis_momentum_obs.parquet")
    print("\nsaved -> data/capture/tennis_momentum_obs.parquet")


if __name__ == "__main__":
    main()
