"""Market-making feasibility: queue-aware markout analysis, no external odds needed.

MM edge = spread captured minus adverse selection minus fees. None of that needs a
"true probability" — it only needs our own book. This simulates, for every market on
every captured day: rest a passive order at the current best price the instant it
becomes best (join-the-queue, FIFO price-time priority). Queue position = whatever
size is already resting there when we join. Walk the trade tape forward — each trade
printed AT that exact price drains the queue FIFO; once cumulative drain passes our
queue position, we're filled at that price. If the level's best status changes away
without draining through us (a cancel-driven move, not a trade), we requeue at the new
best with no fill — that's the queue-priority answer: fill rate tells you how often you'd
actually win the level, not just quote it.

For every fill, markout = signed move in mid `H` seconds later minus the Kalshi maker
fee (backtest/track_b.py MAKER_FEE_COEF, resting orders only pay this, not taker).
Positive average markout across enough fills = the spread would net positive after
adverse selection; <=0 means the market moves against fills faster than the spread
pays for it.

    PYTHONPATH=. python scripts/mm_feasibility.py [--days 2026-07-16,2026-07-17] [--top N]
"""
from __future__ import annotations

import argparse
import glob
import json

import numpy as np
import pandas as pd

from backtest.track_b import MAKER_FEE_COEF, kalshi_fee

TICKS_DIR = "data/capture/tennis_ticks"
HORIZONS_S = [5, 30, 120]


def available_days() -> list[str]:
    return sorted(
        d.split("date=")[1] for d in glob.glob(f"{TICKS_DIR}/date=*")
        if glob.glob(f"{d}/ticks.parquet")
    )


class MarketState:
    __slots__ = ("yes_book", "no_book", "quote", "prev_volume", "mids", "ts_list")

    def __init__(self) -> None:
        self.yes_book: dict[float, float] = {}
        self.no_book: dict[float, float] = {}
        self.quote: dict[str, dict | None] = {"yes": None, "no": None}
        self.prev_volume: float | None = None
        self.mids: list[float] = []
        self.ts_list: list[np.datetime64] = []


def simulate_day(df: pd.DataFrame) -> tuple[list[dict], dict[str, MarketState]]:
    """Single chronological pass. Returns (fill events, per-ticker state incl. mid series)."""
    df = df.sort_values("ts")
    states: dict[str, MarketState] = {}
    fills: list[dict] = []

    for row in df.itertuples():
        st = states.get(row.ticker)
        if st is None:
            st = states[row.ticker] = MarketState()
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
                    q = st.quote["yes"]
                    if q is not None and abs(tp - q["price"]) < 1e-6:
                        q["queue_pos"] -= trade_size
                        if q["queue_pos"] <= 0:
                            fills.append({"ticker": row.ticker, "side": "yes", "fill_ts": ts,
                                          "fill_price": tp, "wait_s": (ts - q["join_ts"]).total_seconds()})
                            st.quote["yes"] = None
                    q = st.quote["no"]
                    if q is not None:
                        no_equiv = round(1 - tp, 2)
                        if abs(no_equiv - q["price"]) < 1e-6:
                            q["queue_pos"] -= trade_size
                            if q["queue_pos"] <= 0:
                                fills.append({"ticker": row.ticker, "side": "no", "fill_ts": ts,
                                              "fill_price": q["price"], "wait_s": (ts - q["join_ts"]).total_seconds()})
                                st.quote["no"] = None
            st.prev_volume = row.volume_fp

        # (re)join queue at current best, for both sides, every event
        bp_yes = max(st.yes_book) if st.yes_book else None
        if bp_yes is not None and (st.quote["yes"] is None or st.quote["yes"]["price"] != bp_yes):
            st.quote["yes"] = {"price": bp_yes, "queue_pos": st.yes_book.get(bp_yes, 0.0), "join_ts": ts}
        elif bp_yes is None:
            st.quote["yes"] = None
        bp_no = max(st.no_book) if st.no_book else None
        if bp_no is not None and (st.quote["no"] is None or st.quote["no"]["price"] != bp_no):
            st.quote["no"] = {"price": bp_no, "queue_pos": st.no_book.get(bp_no, 0.0), "join_ts": ts}
        elif bp_no is None:
            st.quote["no"] = None

        if row.yes_bid is not None and row.yes_ask is not None:
            st.mids.append((row.yes_bid + row.yes_ask) / 2.0)
            st.ts_list.append(ts)

    return fills, states


def markout(fills: list[dict], states: dict[str, MarketState], horizons: list[int]) -> pd.DataFrame:
    out = []
    arrays_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f in fills:
        st = states[f["ticker"]]
        if len(st.ts_list) < 2:
            continue
        if f["ticker"] not in arrays_cache:
            arrays_cache[f["ticker"]] = (
                pd.DatetimeIndex(st.ts_list).values,
                np.array(st.mids),
            )
        ts_arr, mids = arrays_cache[f["ticker"]]
        fill_price = f["fill_price"]
        side = f["side"]
        fee = kalshi_fee(1.0, fill_price, MAKER_FEE_COEF)
        row = {"ticker": f["ticker"], "side": side, "fill_ts": f["fill_ts"],
               "fill_price": fill_price, "wait_s": f["wait_s"]}
        fill_ts64 = pd.Timestamp(f["fill_ts"]).to_datetime64()
        for h in horizons:
            target = fill_ts64 + np.timedelta64(h, "s")
            if target > ts_arr[-1]:
                row[f"markout_{h}s"] = np.nan
                continue
            idx = np.searchsorted(ts_arr, target)
            mid_after = mids[min(idx, len(mids) - 1)]
            pnl = (mid_after - fill_price) if side == "yes" else ((1 - mid_after) - fill_price)
            row[f"markout_{h}s"] = pnl - fee
        out.append(row)
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default=None, help="comma-separated YYYY-MM-DD; default all compacted days")
    ap.add_argument("--top", type=int, default=None, help="limit to top-N markets by row count, for a quick pass")
    args = ap.parse_args()

    days = args.days.split(",") if args.days else available_days()
    all_fills: list[dict] = []
    all_states: dict[str, MarketState] = {}

    for day in days:
        path = f"{TICKS_DIR}/date={day}/ticks.parquet"
        df = pd.read_parquet(path)
        df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
        if args.top:
            keep = df["ticker"].value_counts().head(args.top).index
            df = df[df["ticker"].isin(keep)]
        print(f"{day}: {len(df):,} rows, {df['ticker'].nunique()} markets")
        fills, states = simulate_day(df)
        for f in fills:
            f["ticker"] = f"{day}:{f['ticker']}"
        for k, v in states.items():
            all_states[f"{day}:{k}"] = v
        all_fills.extend(fills)
        print(f"  -> {len(fills)} simulated fills")

    mo = markout(all_fills, all_states, HORIZONS_S)
    if mo.empty:
        print("no fills simulated")
        return

    print(f"\n=== {len(mo)} total fills across {mo['ticker'].nunique()} market-days ===")
    print("fill wait time (s): median %.1f, p90 %.1f" % (mo["wait_s"].median(), mo["wait_s"].quantile(0.9)))
    for h in HORIZONS_S:
        col = f"markout_{h}s"
        valid = mo[col].dropna()
        if valid.empty:
            continue
        win_rate = (valid > 0).mean()
        print(f"markout @ {h}s: n={len(valid)} mean={valid.mean():+.5f} "
              f"median={valid.median():+.5f} win_rate={win_rate:.1%}")

    mo.to_parquet("data/capture/mm_feasibility_fills.parquet")
    print("\nsaved per-fill detail -> data/capture/mm_feasibility_fills.parquet")


if __name__ == "__main__":
    main()
