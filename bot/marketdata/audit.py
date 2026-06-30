"""Audit the persisted depth-logger data for correctness problems.

`--validate` checks the live ingest pipeline; this checks what actually landed on
disk. Re-runnable any time during a collection:

    PYTHONPATH=. python -m bot.marketdata.audit

Flags any of: crossed books (ask < bid), out-of-range prices, bad spreads,
non-positive sizes, invalid trade prices/sides, unreadable shards, suspicious
timestamps, and multi-minute holes in the feed (possible drops). Exit code is
non-zero if any hard problem is found.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BOOK_DIR = Path("data/marketdata/book")
TRADE_DIR = Path("data/marketdata/trades")
HOLE_SECONDS = 300  # a gap with zero book updates across ALL markets this long is suspect


def _load(d: Path) -> tuple[pd.DataFrame, int]:
    frames, bad = [], 0
    for f in sorted(glob.glob(str(d / "*.parquet"))):
        try:
            frames.append(pd.read_parquet(f))
        except Exception as exc:  # a shard mid-write or corrupt
            print(f"  WARN unreadable shard {f}: {exc}")
            bad += 1
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), bad


def _check(label: str, n_bad: int, *, hard: bool = True) -> int:
    status = "OK" if n_bad == 0 else ("*** PROBLEM" if hard else "note")
    print(f"  {label:<42} {n_bad:>8}   {status}")
    return n_bad if hard else 0


def audit_book(df: pd.DataFrame) -> int:
    problems = 0
    if df.empty:
        print("BOOK: no rows yet")
        return 0
    ts = df["ts"].to_numpy(dtype=float)
    span_h = (ts.max() - ts.min()) / 3600
    print(f"BOOK: {len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{pd.to_datetime(ts.min(), unit='s')} → {pd.to_datetime(ts.max(), unit='s')} ({span_h:.1f}h)")

    both = df["yes_bid"].notna() & df["yes_ask"].notna()
    problems += _check("crossed books (yes_ask < yes_bid)",
                       int((df.loc[both, "yes_ask"] < df.loc[both, "yes_bid"]).sum()))
    problems += _check("spread != ask - bid (inconsistent)",
                       int((df.loc[both, "spread"] != df.loc[both, "yes_ask"] - df.loc[both, "yes_bid"]).sum()))
    for col in ("yes_bid", "yes_ask"):
        v = df[col].dropna()
        problems += _check(f"{col} out of [1,99]", int(((v < 1) | (v > 99)).sum()))
    for col in ("yes_bid_sz", "yes_ask_sz"):
        v = df[col].dropna()
        problems += _check(f"{col} <= 0", int((v <= 0).sum()))
    problems += _check("timestamps <= 0 or in the future",
                       int(((ts <= 0) | (ts > pd.Timestamp.utcnow().timestamp() + 60)).sum()))

    # Soft/diagnostic (not failures):
    nl = df["yes_bid"].isna() & df["yes_ask"].isna()
    _check("rows with no quotes (both null)", int(nl.sum()), hard=False)
    _check("one-sided rows (exactly one of bid/ask null)",
           int((df["yes_bid"].isna() ^ df["yes_ask"].isna()).sum()), hard=False)
    # Feed holes: longest stretch with zero book updates across all markets.
    s = np.sort(ts)
    max_gap = float(np.max(np.diff(s))) if len(s) > 1 else 0.0
    big = int((np.diff(s) > HOLE_SECONDS).sum()) if len(s) > 1 else 0
    _check(f"feed holes > {HOLE_SECONDS}s (no updates at all)", big, hard=False)
    print(f"  longest no-update stretch: {max_gap:.0f}s")
    return problems


def audit_trades(df: pd.DataFrame) -> int:
    problems = 0
    if df.empty:
        print("TRADES: no rows yet")
        return 0
    p = pd.to_numeric(df["yes_price"], errors="coerce")
    c = pd.to_numeric(df["count"], errors="coerce")
    print(f"TRADES: {len(df):,} rows | {df['ticker'].nunique()} tickers")
    problems += _check("yes_price out of (0,1)", int(((p <= 0) | (p >= 1)).sum()))
    problems += _check("count <= 0 or null", int((c.isna() | (c <= 0)).sum()))
    problems += _check("taker_side not in {yes,no}",
                       int((~df["taker_side"].isin(["yes", "no"])).sum()))
    problems += _check("null ticker", int(df["ticker"].isna().sum()))
    return problems


def main() -> int:
    print("=" * 64)
    print("DEPTH-LOGGER DATA AUDIT")
    print("=" * 64)
    book, bad_b = _load(BOOK_DIR)
    trades, bad_t = _load(TRADE_DIR)
    problems = bad_b + bad_t
    problems += audit_book(book)
    print("-" * 64)
    problems += audit_trades(trades)
    print("=" * 64)
    if problems == 0:
        print("VERDICT: CLEAN — no correctness problems found")
    else:
        print(f"VERDICT: {problems} PROBLEM(S) FOUND — investigate before trusting the data")
    print("=" * 64)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
