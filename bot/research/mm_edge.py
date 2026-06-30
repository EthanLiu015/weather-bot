"""Estimate market-making edge from the collected book + trade data.

The trades-only probe could only measure markout vs the LAST TRADE price; with the
depth logger's book we can measure against the real MID, which cleanly separates
the two MM economics:

  net_maker(Δ) per contract = sign * (mid_after_Δ - trade_price) - maker_fee

where the maker is the PASSIVE side of each trade (sign=+1 if the taker bought NO
=> maker is long yes; -1 if the taker bought YES => maker is short yes). At Δ=0
this is just the half-spread captured (gross). As Δ grows, the adverse mid-move
(adverse selection) is subtracted. If net stays above 0 at the seconds-to-minutes
horizons a maker unwinds over, market-making is viable.

This is OPTIMISTIC on fills (assumes you win the passive fill) but now HONEST on
both spread and adverse selection. Run on whatever has been collected so far:

    PYTHONPATH=. python -m bot.research.mm_edge
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

BOOK_DIR = Path("data/marketdata/book")
TRADE_DIR = Path("data/marketdata/trades")
HORIZONS_S = [0, 1, 5, 30, 60]
MAKER_FEE_COEF = 0.25 * 0.07  # Kalshi maker fee = 0.0175 * P*(1-P) per contract


def maker_fee_cents(price_cents: np.ndarray) -> np.ndarray:
    p = price_cents / 100.0
    return 100.0 * MAKER_FEE_COEF * p * (1.0 - p)


def net_maker_cents(trade_price_c, sign, mid_after_c, fee_c) -> np.ndarray:
    """Per-contract maker P&L in cents: capture (trade away from mid) minus the
    adverse mid-move by horizon, minus the maker fee. Pure; unit-tested."""
    return sign * (mid_after_c - trade_price_c) - fee_c


def _load(d: Path) -> pd.DataFrame:
    fs = sorted(glob.glob(str(d / "*.parquet")))
    frames = []
    for f in fs:
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute(book: pd.DataFrame, trades: pd.DataFrame, horizons=HORIZONS_S) -> pd.DataFrame:
    """Volume-weighted net maker edge (cents/contract) per unwind horizon."""
    b = book.dropna(subset=["yes_bid", "yes_ask"]).copy()
    b["mid"] = (b["yes_bid"] + b["yes_ask"]) / 2.0
    b = b[["ts", "ticker", "mid"]].sort_values("ts")

    t = trades.dropna(subset=["yes_price", "count", "taker_side"]).copy()
    t = t[t["count"] > 0].sort_values("ts")
    t["price_c"] = pd.to_numeric(t["yes_price"], errors="coerce") * 100.0
    t["sign"] = np.where(t["taker_side"] == "yes", -1.0, 1.0)  # maker yes-position
    t = t.dropna(subset=["price_c"])
    t["fee_c"] = maker_fee_cents(t["price_c"].to_numpy())

    rows = []
    for h in horizons:
        tt = t.copy()
        tt["look_ts"] = tt["ts"] + h
        # mid at (or just before) the unwind time, matched within each ticker
        merged = pd.merge_asof(
            tt.sort_values("look_ts"), b.rename(columns={"ts": "mid_ts"}),
            left_on="look_ts", right_on="mid_ts", by="ticker", direction="backward",
        ).dropna(subset=["mid"])
        if merged.empty:
            continue
        net = net_maker_cents(merged["price_c"].to_numpy(), merged["sign"].to_numpy(),
                              merged["mid"].to_numpy(), merged["fee_c"].to_numpy())
        w = merged["count"].to_numpy()
        rows.append({
            "horizon_s": h,
            "trades": len(merged),
            "contracts": float(w.sum()),
            "gross_c": float(np.average(net + merged["fee_c"].to_numpy(), weights=w)),
            "fee_c": float(np.average(merged["fee_c"].to_numpy(), weights=w)),
            "net_c_per_contract": float(np.average(net, weights=w)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    book, trades = _load(BOOK_DIR), _load(TRADE_DIR)
    print("=" * 64)
    print("MARKET-MAKING EDGE (book mid; passive side of every fill)")
    print("=" * 64)
    if book.empty or trades.empty:
        print("  not enough data yet (need book + trades)")
        return
    span_h = (book["ts"].max() - book["ts"].min()) / 3600
    print(f"  data so far: {len(book):,} book rows, {len(trades):,} trades, ~{span_h:.1f}h  (PARTIAL)")
    res = compute(book, trades)
    print("  horizon   trades  contracts   gross   fee    NET ¢/contract")
    for _, r in res.iterrows():
        print(f"   {int(r.horizon_s):>3}s     {int(r.trades):>6}  {r.contracts:>9,.0f}  "
              f"{r.gross_c:>+6.2f}  {r.fee_c:>4.2f}   {r.net_c_per_contract:>+6.2f}")
    print("-" * 64)
    print("  Δ=0 ≈ gross half-spread captured; longer Δ subtracts adverse")
    print("  selection. NET > 0 at the minute horizon ⇒ MM viable (modulo fill rate).")
    print("=" * 64)


if __name__ == "__main__":
    main()
