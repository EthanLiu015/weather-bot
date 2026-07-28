"""Score the forward-capture programs once data has accumulated.

Run after >=2 weeks of cron capture:
    PYTHONPATH=. python scripts/analyze_capture.py

  1. TENNIS: Kalshi mid vs de-vigged sharp prob divergence, scored vs settlement.
  2. CALIBRATION: growing per-series reliability from settled_probe.parquet
     (the sample the API purge would otherwise destroy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAP = "data/capture"


def load(name: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(f"{CAP}/{name}.parquet")
    except FileNotFoundError:
        return pd.DataFrame()


def settled_map() -> pd.Series:
    s = load("settled_probe")
    if s.empty:
        return pd.Series(dtype=float)
    return s.set_index("ticker")["result"].map({"yes": 1.0, "no": 0.0})


def analyze_tennis(results: pd.Series) -> None:
    w = load("tennis_compare")
    if w.empty or w["sharp_prob"].notna().sum() == 0:
        print("\ntennis: no matched sharp lines yet (set ODDS_API_KEY)")
        return
    w = w[w["sharp_prob"].notna()].copy()
    w["y"] = w["ticker"].map(results)
    scored = w[w["y"].notna()]
    print(f"\n=== TENNIS: {len(w)} matched, {len(scored)} settled ===")
    if len(scored):
        bk = np.mean((scored["kalshi_mid"] - scored["y"]) ** 2)
        bs = np.mean((scored["sharp_prob"] - scored["y"]) ** 2)
        print(f"  Brier kalshi {bk:.4f} vs sharp {bs:.4f} "
              f"({'sharp better' if bs < bk else 'kalshi better'})")
        print(f"  mean |divergence| {np.mean(np.abs(scored['kalshi_mid'] - scored['sharp_prob'])):.4f}")


def analyze_calibration(_: pd.Series) -> None:
    s = load("settled_probe")
    s = s[s["probe_price"].notna()] if not s.empty else s
    if s.empty:
        print("\ncalibration: no data yet")
        return
    s = s.copy()
    s["y"] = s["result"].map({"yes": 1.0, "no": 0.0})
    s["event"] = s["ticker"].str.rsplit("-", n=1).str[0]
    print("\n=== ACCUMULATED CALIBRATION (probe at close-24h/mid-life) ===")
    g = s.groupby("series").apply(
        lambda x: pd.Series({
            "n": len(x), "events": x["event"].nunique(),
            "brier": np.mean((x["probe_price"] - x["y"]) ** 2),
            "bias": x["probe_price"].mean() - x["y"].mean(),
        }), include_groups=False)
    print(g.round(4).to_string())


def main() -> None:
    results = settled_map()
    analyze_tennis(results)
    analyze_calibration(results)


if __name__ == "__main__":
    main()
