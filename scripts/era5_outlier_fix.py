"""
Fix ERA5 outliers in features.parquet.

Two-step process:
1. Clip gefs_tmax_mean at each station's 1st-percentile floor. This removes
   physically impossible values (e.g. KSEA -16°F) while preserving systematic
   cold bias the model relies on.
2. Recompute obs_minus_model_lag1/2/3 and roll_mean/roll_std, since those
   features are derived from gefs_tmax_mean and the clipping changes their
   source values.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROLLING_WINDOW_DAYS = 7


def compute_station_floors(df: pd.DataFrame) -> dict[str, float]:
    """Return per-station 1st-percentile of gefs_tmax_mean."""
    return {
        station: float(group["gefs_tmax_mean"].quantile(0.01))
        for station, group in df.groupby("station")
    }


def clip_gefs_outliers(
    df: pd.DataFrame,
    floors: dict[str, float],
) -> pd.DataFrame:
    """Clip gefs_tmax_mean per station at *floors*, then recompute
    ecmwf_gefs_tmax_delta = |ecmwf_tmax - gefs_tmax_mean|."""
    result = df.copy()
    for station, floor in floors.items():
        mask = result["station"] == station
        result.loc[mask, "gefs_tmax_mean"] = result.loc[mask, "gefs_tmax_mean"].clip(lower=floor)

    has_both = result["ecmwf_tmax"].notna() & result["gefs_tmax_mean"].notna()
    result.loc[has_both, "ecmwf_gefs_tmax_delta"] = (
        result.loc[has_both, "ecmwf_tmax"] - result.loc[has_both, "gefs_tmax_mean"]
    ).abs()

    return result


def recompute_lag_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute obs_minus_model_lag1/2/3 and roll_mean/roll_std from the
    current gefs_tmax_mean (post-clip) and actual_tmax values.

    The D+1 (lead_hour=24) row for each (station, date) defines the residual
    for that calendar day: residual = actual_tmax - gefs_tmax_mean. All lead
    hours for the following dates then inherit this residual as lag values,
    matching the logic in build_feature_matrix.py.
    """
    result = df.copy()
    result["_date_ts"] = pd.to_datetime(result["date"])

    residual_frames: list[pd.DataFrame] = []

    for station, group in result.groupby("station"):
        d1 = (
            group[group["lead_hour"] == 24]
            .set_index("_date_ts")[["actual_tmax", "gefs_tmax_mean"]]
            .sort_index()
        )
        res = (d1["actual_tmax"] - d1["gefs_tmax_mean"]).rename("_residual")
        # Expand residual to a daily DatetimeIndex, then shift-join for lags
        res_df = res.to_frame()
        res_df["_lag1"] = res.shift(1, freq="D")
        res_df["_lag2"] = res.shift(2, freq="D")
        res_df["_lag3"] = res.shift(3, freq="D")
        # Rolling stats: min 2 observations from the previous ROLLING_WINDOW_DAYS days
        res_df["_roll_mean"] = (
            res.rolling(window=f"{_ROLLING_WINDOW_DAYS}D", min_periods=2).mean().shift(1, freq="D")
        )
        res_df["_roll_std"] = (
            res.rolling(window=f"{_ROLLING_WINDOW_DAYS}D", min_periods=2).std().shift(1, freq="D")
        )
        res_df["station"] = station
        residual_frames.append(res_df.reset_index().rename(columns={"_date_ts": "_date_ts"}))

    residuals = pd.concat(residual_frames, ignore_index=True)

    result = result.merge(
        residuals[["station", "_date_ts", "_lag1", "_lag2", "_lag3", "_roll_mean", "_roll_std"]],
        on=["station", "_date_ts"],
        how="left",
    )
    result["obs_minus_model_lag1"] = result["_lag1"]
    result["obs_minus_model_lag2"] = result["_lag2"]
    result["obs_minus_model_lag3"] = result["_lag3"]
    result["obs_minus_model_roll_mean"] = result["_roll_mean"]
    result["obs_minus_model_roll_std"] = result["_roll_std"]

    return result.drop(columns=["_date_ts", "_lag1", "_lag2", "_lag3", "_roll_mean", "_roll_std"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default="data/historical/features.parquet")
    args = parser.parse_args()

    df = pd.read_parquet(args.features)
    logger.info("Loaded %d rows from %s", len(df), args.features)

    floors = compute_station_floors(df)
    logger.info("Station floors: %s", {k: round(v, 2) for k, v in sorted(floors.items())})

    n_clipped = sum(
        int(((df["station"] == st) & (df["gefs_tmax_mean"] < floor)).sum())
        for st, floor in floors.items()
    )
    logger.info("Rows to clip: %d", n_clipped)

    df = clip_gefs_outliers(df, floors)
    logger.info("Clipped gefs_tmax_mean and recomputed ecmwf_gefs_tmax_delta")

    df = recompute_lag_residuals(df)
    logger.info("Recomputed lag and rolling residual features")

    df.to_parquet(args.features, index=False)
    logger.info("Saved %d rows back to %s", len(df), args.features)


if __name__ == "__main__":
    main()
