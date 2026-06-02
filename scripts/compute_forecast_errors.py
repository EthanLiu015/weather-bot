"""
Compute empirical GFS-proxy forecast error distributions from existing data.

Uses ERA5 reanalysis (at D-lead initialization time) as a proxy for what a
numerical model would have predicted, then computes the error vs actual ASOS
Tmax on the verification date.

Output: data/historical/forecast_error_distributions.parquet
  Columns: station, month, lead_hours, mean_error_f, std_error_f, n_samples
  One row per (station, month, lead_hours) combination.

Usage: PYTHONPATH=. python scripts/compute_forecast_errors.py
"""
import io
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.stations import STATION_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ERA5_DIR  = Path("data/era5")
HIST_DIR  = Path("data/historical")
OUT_PATH  = HIST_DIR / "forecast_error_distributions.parquet"

LEAD_HOURS = [24, 48, 72, 96, 120, 168]


# ---------------------------------------------------------------------------
# Load ERA5 station time series from zip archives
# ---------------------------------------------------------------------------

def load_era5_station_ts(station: str) -> pd.DataFrame:
    meta = STATION_REGISTRY[station]
    lat, lon = meta.lat, meta.lon
    lon_360 = lon % 360

    frames = []
    era5_files = sorted(ERA5_DIR.glob("era5_*.nc"))

    for path in era5_files:
        try:
            with zipfile.ZipFile(path) as z:
                with tempfile.TemporaryDirectory() as tmp:
                    z.extractall(tmp)
                    nc_path = os.path.join(tmp, "data_stream-oper_stepType-instant.nc")
                    ds = xr.open_dataset(nc_path, engine="netcdf4")
                    t2m = ds["t2m"].sel(
                        latitude=lat, longitude=lon_360, method="nearest"
                    )
                    df = pd.DataFrame({
                        "valid_time": pd.to_datetime(t2m.valid_time.values),
                        "t2m_k": t2m.values.flatten(),
                    })
                    df["t2m_f"] = (df["t2m_k"] - 273.15) * 9 / 5 + 32
                    frames.append(df[["valid_time", "t2m_f"]])
        except Exception as exc:
            logger.warning("ERA5 load failed for %s / %s: %s", station, path.name, exc)

    if not frames:
        return pd.DataFrame(columns=["valid_time", "t2m_f"])

    ts = pd.concat(frames).drop_duplicates("valid_time").sort_values("valid_time")
    ts = ts.set_index("valid_time")
    logger.info("ERA5 ts for %s: %d timesteps (%s → %s)",
                station, len(ts), ts.index.min(), ts.index.max())
    return ts


# ---------------------------------------------------------------------------
# Load ASOS actual daily Tmax
# ---------------------------------------------------------------------------

def load_asos_daily_tmax(station: str) -> pd.Series:
    path = HIST_DIR / f"{station}_hourly.parquet"
    if not path.exists():
        return pd.Series(dtype=float)

    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    tz = STATION_REGISTRY[station].timezone
    df["datetime_local"] = df["datetime"].dt.tz_convert(tz)
    df["date"] = df["datetime_local"].dt.date
    df = df.dropna(subset=["tmpf"])
    daily = df.groupby("date")["tmpf"].max()
    daily.index = pd.DatetimeIndex(daily.index)
    return daily


# ---------------------------------------------------------------------------
# Compute errors per (station, month, lead_hours)
# ---------------------------------------------------------------------------

def compute_station_errors(station: str) -> list[dict]:
    era5_ts  = load_era5_station_ts(station)
    asos_max = load_asos_daily_tmax(station)

    if era5_ts.empty or asos_max.empty:
        logger.warning("Insufficient data for %s", station)
        return []

    rows = []
    for lead_h in LEAD_HOURS:
        lag = pd.Timedelta(hours=lead_h)
        errors_by_month: dict[int, list[float]] = {m: [] for m in range(1, 13)}

        for date, actual_tmax in asos_max.items():
            if pd.isna(actual_tmax):
                continue

            # ERA5 at init time = verification_date midnight UTC - lead_hours
            init_time = pd.Timestamp(date, tz="UTC") - lag
            init_time_naive = init_time.tz_localize(None)

            # Find nearest ERA5 timestep within 4 hours
            if era5_ts.empty:
                continue
            diffs = abs(era5_ts.index - init_time_naive)
            if diffs.min() > pd.Timedelta(hours=4):
                continue

            era5_val = float(era5_ts.iloc[diffs.argmin()]["t2m_f"])
            error = era5_val - actual_tmax  # positive = model ran warm
            month = date.month
            errors_by_month[month].append(error)

        for month, errs in errors_by_month.items():
            if len(errs) < 10:
                continue
            arr = np.array(errs)
            rows.append({
                "station":      station,
                "month":        month,
                "lead_hours":   lead_h,
                "mean_error_f": float(np.mean(arr)),
                "std_error_f":  float(np.std(arr)),
                "p10_error_f":  float(np.percentile(arr, 10)),
                "p90_error_f":  float(np.percentile(arr, 90)),
                "n_samples":    len(errs),
            })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_rows: list[dict] = []
    stations = list(STATION_REGISTRY.keys())
    logger.info("Computing forecast error distributions for %d stations × %d lead times",
                len(stations), len(LEAD_HOURS))

    for station in stations:
        logger.info("--- %s (%s) ---", station, STATION_REGISTRY[station].city)
        rows = compute_station_errors(station)
        all_rows.extend(rows)
        if rows:
            sample = [r for r in rows if r["month"] == 7 and r["lead_hours"] == 24]
            if sample:
                r = sample[0]
                logger.info("  July D+1: mean_err=%.1f°F std=%.1f°F n=%d",
                            r["mean_error_f"], r["std_error_f"], r["n_samples"])

    df = pd.DataFrame(all_rows)
    HIST_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    logger.info("Saved %d rows to %s", len(df), OUT_PATH)
    logger.info("Buckets per station: %d", len(df) // len(stations) if stations else 0)

    # Print summary
    print("\n=== Error Distribution Summary ===")
    summary = df.groupby("lead_hours")[["mean_error_f", "std_error_f", "n_samples"]].mean().round(2)
    print(summary.to_string())
    print("\nLarger std = more forecast uncertainty at that lead time")


if __name__ == "__main__":
    main()
