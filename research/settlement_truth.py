"""Cycle 4: settle the settlement-truth question.

Recovers the tmax interval implied by which KXHIGH bracket settled YES
(HIGH-temp series only — the throwaway version mixed KXLOWT overnight-min
brackets in, manufacturing the fake -18F "ERA5 corruption" at KATL/KIAH),
fetches official NWS CLI daily highs from IEM (what Kalshi actually settles
on), and reports where ERA5 `actual_tmax` disagrees with settlement truth.

Usage:
    PYTHONPATH=. python -m research.settlement_truth
"""

from __future__ import annotations

import json
import time
import urllib.request

import pandas as pd

from config.series import is_low_temp_series

PRICES_PATH = "data/historical/kalshi_prices.parquet"
FEATURES_PATH = "data/historical/features.parquet"
CLI_TRUTH_PATH = "data/historical/cli_truth.parquet"
IEM_CLI_URL = "https://mesonet.agron.iastate.edu/json/cli.py?station={station}&year={year}"


def implied_interval(brackets: pd.DataFrame) -> tuple[float, float] | None:
    """Tmax interval [lo, hi] implied by the single YES bracket of one
    station-day's KXHIGH markets. Subtitle semantics: between B83.5
    (floor 83, cap 84) = "83 to 84" -> [83, 84]; greater T84 (floor 84) =
    "85 or above" -> [85, inf); less T77 (cap 77) = "76 or below" ->
    (-inf, 76]. None if zero or multiple YES brackets (bad/missing data)."""
    yes = brackets[brackets["settlement"] == 1.0]
    if len(yes) != 1:
        return None
    row = yes.iloc[0]
    if row["strike_type"] == "between":
        return (row["floor_strike"], row["cap_strike"])
    if row["strike_type"] == "greater":
        return (row["floor_strike"] + 1.0, float("inf"))
    if row["strike_type"] == "less":
        return (float("-inf"), row["cap_strike"] - 1.0)
    return None


def implied_intervals(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per KXHIGH station-day: implied tmax interval from settlement."""
    high = prices[~prices["series"].map(is_low_temp_series)]
    rows = []
    for (station, date), grp in high.groupby(["station", "date"]):
        interval = implied_interval(grp)
        if interval is None:
            continue
        rows.append(
            {"station": station, "date": date, "implied_lo": interval[0], "implied_hi": interval[1]}
        )
    return pd.DataFrame(rows)


def fetch_cli_highs(stations: list[str], years: list[int]) -> pd.DataFrame:
    """Official NWS CLI daily highs from IEM, one row per station-date.
    CLI corrections appear as later products for the same valid date; the
    last product (lexicographically greatest id = latest issuance) wins."""
    rows = []
    for station in stations:
        for year in years:
            url = IEM_CLI_URL.format(station=station, year=year)
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = json.load(resp)
            for rec in payload.get("results", []):
                if not isinstance(rec.get("high"), int):
                    continue
                rows.append(
                    {
                        "station": station,
                        "date": rec["valid"],
                        "cli_high": float(rec["high"]),
                        "product": rec.get("product", ""),
                    }
                )
            time.sleep(0.5)
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("product").groupby(["station", "date"], as_index=False).last()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.drop(columns=["product"])


def era5_truth(features_path: str = FEATURES_PATH) -> pd.DataFrame:
    feats = pd.read_parquet(features_path)[["station", "date", "actual_tmax"]].dropna()
    feats["date"] = pd.to_datetime(feats["date"])
    return feats.drop_duplicates(["station", "date"])


def consistency(truth: pd.DataFrame) -> pd.DataFrame:
    """Per station-day: does each truth source fall inside the implied interval?"""
    out = truth.copy()
    for col in ("cli_high", "actual_tmax"):
        out[f"{col}_consistent"] = (out[col] >= out["implied_lo"]) & (
            out[col] <= out["implied_hi"]
        )
    out["era5_cli_diff"] = out["actual_tmax"] - out["cli_high"]
    return out


def main() -> None:
    prices = pd.read_parquet(PRICES_PATH)
    prices["date"] = pd.to_datetime(prices["date"])
    implied = implied_intervals(prices)
    print(f"KXHIGH station-days with clean single-YES recovery: {len(implied)}")

    stations = sorted(implied["station"].unique())
    years = sorted(implied["date"].dt.year.unique())
    try:
        cli = pd.read_parquet(CLI_TRUTH_PATH)
    except FileNotFoundError:
        cli = fetch_cli_highs(stations, years)
        cli.to_parquet(CLI_TRUTH_PATH)
        print(f"fetched {len(cli)} CLI reports -> {CLI_TRUTH_PATH}")

    merged = implied.merge(cli, on=["station", "date"], how="left").merge(
        era5_truth(), on=["station", "date"], how="left"
    )
    scored = consistency(merged.dropna(subset=["cli_high", "actual_tmax"]))
    print(f"station-days with implied + CLI + ERA5: {len(scored)}\n")

    print("=== Does official CLI high fall in the settled bracket? (validates recovery) ===")
    print(scored.groupby("station")["cli_high_consistent"].agg(["mean", "count"]).to_string())
    print(f"overall: {scored['cli_high_consistent'].mean():.3f}\n")

    print("=== Does ERA5 actual_tmax fall in the settled bracket? ===")
    print(scored.groupby("station")["actual_tmax_consistent"].agg(["mean", "count"]).to_string())
    print(f"overall: {scored['actual_tmax_consistent'].mean():.3f}\n")

    print("=== ERA5 minus CLI (deg F) per station ===")
    stats = scored.groupby("station")["era5_cli_diff"].agg(["mean", "std", "count"])
    stats["abs_ge_1"] = scored.assign(big=scored["era5_cli_diff"].abs() >= 1.0).groupby(
        "station"
    )["big"].mean()
    print(stats.round(2).to_string())
    print(
        f"overall: mean {scored['era5_cli_diff'].mean():+.2f}, "
        f"MAE {scored['era5_cli_diff'].abs().mean():.2f}, "
        f"|diff|>=1F share {(scored['era5_cli_diff'].abs() >= 1.0).mean():.2%}"
    )


if __name__ == "__main__":
    main()
