"""Estimate market-making edge from the collected book + trade data.

The trades-only probe could only measure markout vs the LAST TRADE price; with the
depth logger's book we measure against the real MID, which cleanly separates the
two MM economics:

  net_maker(Δ) per contract = sign * (mid_after_Δ - trade_price) - maker_fee

where the maker is the PASSIVE side of each trade (sign=+1 if the taker bought NO
=> maker is long yes; -1 if the taker bought YES => maker is short yes). At Δ=0
this is the half-spread captured (gross); as Δ grows the adverse mid-move
(adverse selection) is subtracted. Net > 0 at the seconds-to-minutes horizon a
maker unwinds over ⇒ viable (modulo fill rate).

This OPTIMISTICALLY assumes you win the passive fill, but is HONEST on spread and
adverse selection. The segment breakdown shows WHERE the edge concentrates.

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
SEG_HORIZONS = [0, 1, 5]  # gross / fast-unwind / realistic-unwind for the segment tables
MAKER_FEE_COEF = 0.25 * 0.07  # Kalshi maker fee = 0.0175 * P*(1-P) per contract


def maker_fee_cents(price_cents: np.ndarray) -> np.ndarray:
    p = price_cents / 100.0
    return 100.0 * MAKER_FEE_COEF * p * (1.0 - p)


def net_maker_cents(trade_price_c, sign, mid_after_c, fee_c) -> np.ndarray:
    """Per-contract maker P&L in cents: capture (trade away from mid) minus the
    adverse mid-move by horizon, minus the maker fee. Pure; unit-tested."""
    return sign * (mid_after_c - trade_price_c) - fee_c


def _load(d: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(d / "*.parquet"))):
        try:
            frames.append(pd.read_parquet(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def enrich(book: pd.DataFrame, trades: pd.DataFrame, horizons=HORIZONS_S) -> pd.DataFrame:
    """One row per trade with the resting mid/spread before it, net_maker at each
    horizon, and segment columns (price bucket, spread bucket, market, hour)."""
    b = book.dropna(subset=["yes_bid", "yes_ask"]).copy()
    b["mid"] = (b["yes_bid"] + b["yes_ask"]) / 2.0
    b["spr"] = b["yes_ask"] - b["yes_bid"]
    b["bid"] = b["yes_bid"]
    b["ask"] = b["yes_ask"]
    b = b[["ts", "ticker", "mid", "spr", "bid", "ask"]].sort_values("ts")

    t = trades.dropna(subset=["yes_price", "count", "taker_side"]).copy()
    t = t[t["count"] > 0].copy()
    t["price_c"] = pd.to_numeric(t["yes_price"], errors="coerce") * 100.0
    t = t.dropna(subset=["price_c"])
    t["sign"] = np.where(t["taker_side"] == "yes", -1.0, 1.0)
    t["fee_c"] = maker_fee_cents(t["price_c"].to_numpy())
    t = t.sort_values("ts").reset_index(drop=True)
    t["tid"] = np.arange(len(t))

    # resting mid + quoted spread just before each trade
    t = pd.merge_asof(t, b.rename(columns={"mid": "mid_before", "spr": "spr_before"}),
                      on="ts", by="ticker", direction="backward")

    # mid at each unwind horizon -> net_maker at that horizon
    for h in horizons:
        look = t[["tid", "ticker", "ts", "price_c", "sign", "fee_c"]].copy()
        look["look"] = look["ts"] + h
        look = look.sort_values("look")
        m = pd.merge_asof(look, b.rename(columns={"ts": "bts"})[["bts", "ticker", "mid", "bid", "ask"]],
                          left_on="look", right_on="bts", by="ticker", direction="backward")
        m[f"net_{h}"] = net_maker_cents(m["price_c"].to_numpy(), m["sign"].to_numpy(),
                                        m["mid"].to_numpy(), m["fee_c"].to_numpy())
        cols = {"mid": f"mid_{h}", "bid": f"bid_{h}", "ask": f"ask_{h}"}
        t = t.merge(m[["tid", f"net_{h}"] + list(cols)].rename(columns=cols), on="tid", how="left")

    t["market"] = t["ticker"].str.split("-").str[0]
    t["hour_utc"] = pd.to_datetime(t["ts"], unit="s").dt.hour
    t["price_bkt"] = pd.cut(t["price_c"], [0, 15, 35, 65, 85, 100],
                            labels=["1-15", "15-35", "35-65", "65-85", "85-99"])
    t["spr_bkt"] = pd.cut(t["spr_before"], [0, 1.5, 3.5, 6.5, 100],
                          labels=["1c", "2-3c", "4-6c", "7c+"])
    return t


def _wavg(values: np.ndarray, weights: np.ndarray) -> float:
    ok = np.isfinite(values)
    return float(np.average(values[ok], weights=weights[ok])) if ok.any() else float("nan")


def overall(t: pd.DataFrame, horizons=HORIZONS_S) -> pd.DataFrame:
    w = t["count"].to_numpy()
    rows = []
    for h in horizons:
        n = t[f"net_{h}"].to_numpy()
        rows.append({"horizon_s": h, "contracts": float(w.sum()),
                     "net_c": _wavg(n, w)})
    return pd.DataFrame(rows)


def seg_table(t: pd.DataFrame, by: str, horizons=SEG_HORIZONS) -> pd.DataFrame:
    rows = []
    for key, g in t.groupby(by, observed=True):
        w = g["count"].to_numpy()
        row = {by: key, "trades": len(g), "contracts": float(w.sum())}
        for h in horizons:
            row[f"net_{h}s"] = _wavg(g[f"net_{h}"].to_numpy(), w)
        rows.append(row)
    return pd.DataFrame(rows)


def _print_seg(title: str, df: pd.DataFrame, key: str) -> None:
    print(f"  by {title}:")
    print(f"    {'segment':<10} {'contracts':>10}   net@0s  net@1s  net@5s")
    for _, r in df.iterrows():
        print(f"    {str(r[key]):<10} {r['contracts']:>10,.0f}   "
              f"{r['net_0s']:>+5.2f}   {r['net_1s']:>+5.2f}   {r['net_5s']:>+5.2f}")
    print()


def main() -> None:
    book, trades = _load(BOOK_DIR), _load(TRADE_DIR)
    print("=" * 70)
    print("MARKET-MAKING EDGE  (book mid; passive side of every fill; ¢/contract)")
    print("=" * 70)
    if book.empty or trades.empty:
        print("  not enough data yet (need book + trades)")
        return
    t = enrich(book, trades)
    span_h = (book["ts"].max() - book["ts"].min()) / 3600
    print(f"  {len(book):,} book rows, {len(t):,} priced trades, ~{span_h:.1f}h  (PARTIAL)\n")

    print("  OVERALL net vs unwind horizon:")
    for _, r in overall(t).iterrows():
        print(f"    {int(r.horizon_s):>3}s   net {r.net_c:>+5.2f}")
    print()
    _print_seg("trade-price bucket", seg_table(t, "price_bkt"), "price_bkt")
    _print_seg("quoted-spread bucket", seg_table(t, "spr_bkt"), "spr_bkt")
    _print_seg("market", seg_table(t, "market").sort_values("contracts", ascending=False).head(10), "market")
    _print_seg("hour (UTC)", seg_table(t, "hour_utc"), "hour_utc")
    print("-" * 70)
    print("  net@1s/5s > 0 marks segments where a fast-unwinding maker has edge")
    print("  (still optimistic on fill rate). Tight-spread liquid segments expected best.")
    print("=" * 70)


if __name__ == "__main__":
    main()
