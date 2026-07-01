"""Intraday market-efficiency diagnostic.

Does the Kalshi temperature book converge to the settled outcome as the trading
day progresses, or does it stay mispriced while markets are still live? We score
the market's OWN price at each intraday snapshot against the binary settlement:

  * brier         — mean (price - settlement)^2. Falls toward 0 as the book learns.
  * resolved_frac — share of markets whose price is within 0.1 of 0 or 1 (i.e. the
                    outcome is effectively decided in the price).
  * mean_price    — average YES price (a base-rate calibration check).

If, late in the day, the Brier stays well above 0 while a large share of markets
are still unresolved, the book is leaving information on the table intraday — the
opening the edge-gated thesis needs. This needs NO model and NO training: it is
the cheap go/no-go test for the whole intraday programme.

Run:  PYTHONPATH=. python -m research.intraday_lag
Data: data/historical/intraday_prices.parquet
"""
from __future__ import annotations

import numpy as np
import pandas as pd

INTRADAY_PATH = "data/historical/intraday_prices.parquet"

# Snapshot columns in the intraday parquet, earliest → latest relative to the
# market's reference time. p-12 is empty in the current data but kept for shape.
SNAPSHOTS = ["p-12", "p-6", "p+0", "p+6", "p+12", "p+14"]

RESOLVED_TOL = 0.1


def _nan_stats() -> dict:
    return {
        "n": 0,
        "brier": float("nan"),
        "resolved_frac": float("nan"),
        "mean_price": float("nan"),
    }


def snapshot_efficiency(
    df: pd.DataFrame,
    snapshots: list[str] = SNAPSHOTS,
    settlement_col: str = "settlement",
    resolved_tol: float = RESOLVED_TOL,
) -> dict[str, dict]:
    """Per-snapshot market efficiency stats keyed by snapshot column name.

    A snapshot with no non-null prices (paired with a non-null settlement) reports
    n=0 and NaN stats rather than raising, so the fully-empty p-12 column is fine.
    """
    out: dict[str, dict] = {}
    settle = pd.to_numeric(df[settlement_col], errors="coerce")
    for snap in snapshots:
        if snap not in df.columns:
            out[snap] = _nan_stats()
            continue
        price = pd.to_numeric(df[snap], errors="coerce")
        mask = price.notna() & settle.notna()
        p = price[mask].to_numpy(dtype=float)
        y = settle[mask].to_numpy(dtype=float)
        if p.size == 0:
            out[snap] = _nan_stats()
            continue
        out[snap] = {
            "n": int(p.size),
            "brier": float(np.mean((p - y) ** 2)),
            "resolved_frac": float(np.mean(np.minimum(p, 1.0 - p) < resolved_tol)),
            "mean_price": float(p.mean()),
        }
    return out


def _print_report(stats: dict[str, dict], base_rate: float) -> None:
    print("\n" + "=" * 66)
    print("INTRADAY MARKET EFFICIENCY  (market price vs settlement, by snapshot)")
    print("=" * 66)
    print(f"  base rate P(settle=1) = {base_rate:.3f}")
    print(f"  {'snapshot':8s} {'n':>5s}  {'brier':>7s}  {'resolved':>8s}  {'mean_p':>7s}")
    for snap, s in stats.items():
        if s["n"] == 0:
            print(f"  {snap:8s} {s['n']:>5d}  {'--':>7s}  {'--':>8s}  {'--':>7s}")
            continue
        print(f"  {snap:8s} {s['n']:>5d}  {s['brier']:>7.4f}  "
              f"{s['resolved_frac']:>8.2f}  {s['mean_price']:>7.3f}")
    print("-" * 66)
    print("  Reading: brier falling toward 0 = book converging; brier staying high")
    print("  with resolved<1.0 late = intraday mispricing left on the table.")


def main() -> None:
    df = pd.read_parquet(INTRADAY_PATH)
    base_rate = float(pd.to_numeric(df["settlement"], errors="coerce").mean())
    stats = snapshot_efficiency(df)
    _print_report(stats, base_rate)


if __name__ == "__main__":
    main()
