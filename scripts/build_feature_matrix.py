"""
Build historical training feature matrix from ERA5 + ASOS.
Run after bootstrap_history.py completes.

Output: data/historical/features.parquet
One row per (station, date, lead_hour) with all model features + actual_tmax target.

ERA5 is used as a reanalysis proxy for model forecasts:
  - t2m, d2m, u10, v10, tcc → forecast analogs at lead-time initialization
  - Spread proxies computed from ±3-day temporal window (captures synoptic variability)
  - NBM features left NaN (no historical NBM available)
"""
import json
import logging
import os
import sys
import tempfile
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.stations import STATION_REGISTRY, ALL_ICAO
from processing.climatology import CLIMATOLOGY_PATH, climo_tmax_normal
from processing.features import (
    get_feature_columns,
    _quantile_skewness,
    _quantile_kurtosis,
    _onshore_wind_component,
    _ROLLING_WINDOW_DAYS,
)
from processing.bias_correction import BiasCorrectionRegistry, get_lead_bucket, get_season

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ERA5_DIR = Path("data/era5")
HIST_DIR = Path("data/historical")
BIAS_DIR = Path("data/bias_correctors")

LEAD_HOURS = [24, 48, 72, 96, 120, 168]
SPREAD_WINDOW_HOURS = 72   # ±3 days for spread proxy


# ---------------------------------------------------------------------------
# ERA5 loading
# ---------------------------------------------------------------------------

def _load_era5_zip(path: Path, tmp_dir: str) -> tuple[xr.Dataset, xr.Dataset]:
    with zipfile.ZipFile(path) as z:
        z.extractall(tmp_dir)
    instant = xr.open_dataset(
        os.path.join(tmp_dir, "data_stream-oper_stepType-instant.nc"),
        engine="netcdf4",
    )
    accum = xr.open_dataset(
        os.path.join(tmp_dir, "data_stream-oper_stepType-accum.nc"),
        engine="netcdf4",
    )
    return instant, accum


def _extract_station_series(instant: xr.Dataset, accum: xr.Dataset, station: str) -> pd.DataFrame:
    meta = STATION_REGISTRY[station]
    lat, lon = meta.lat, meta.lon
    lon_360 = lon % 360

    def nearest(da: xr.DataArray) -> pd.Series:
        vals = da.sel(latitude=lat, longitude=lon_360, method="nearest")
        return pd.Series(vals.values, index=pd.to_datetime(da.valid_time.values))

    rows = pd.DataFrame({
        "t2m":  nearest(instant["t2m"]),
        "d2m":  nearest(instant["d2m"]),
        "u10":  nearest(instant["u10"]),
        "v10":  nearest(instant["v10"]),
        "tcc":  nearest(instant["tcc"]),
        "sp":   nearest(instant["sp"]),
        "tp":   nearest(accum["tp"]),
    })
    rows.index.name = "valid_time"
    return rows


def build_era5_station_ts(stations: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load all ERA5 zip files and extract a continuous time series per station.
    Returns dict[station] -> DataFrame indexed by UTC timestamp.
    """
    era5_files = sorted(ERA5_DIR.glob("era5_*.nc"))
    if not era5_files:
        raise FileNotFoundError(f"No ERA5 files found in {ERA5_DIR}")

    logger.info("Loading %d ERA5 files for %d stations...", len(era5_files), len(stations))
    station_frames: dict[str, list[pd.DataFrame]] = {s: [] for s in stations}

    with tempfile.TemporaryDirectory() as tmp:
        for era5_path in era5_files:
            logger.info("  %s", era5_path.name)
            try:
                instant, accum = _load_era5_zip(era5_path, tmp)
                for station in stations:
                    df = _extract_station_series(instant, accum, station)
                    station_frames[station].append(df)
                instant.close()
                accum.close()
                # Clean up extracted files for next iteration
                for f in Path(tmp).glob("*.nc"):
                    f.unlink()
            except Exception as exc:
                logger.warning("Failed to load %s: %s", era5_path.name, exc)

    result = {}
    for station, frames in station_frames.items():
        if frames:
            combined = pd.concat(frames).sort_index()
            combined = combined[~combined.index.duplicated(keep="first")]
            result[station] = combined
            logger.info("ERA5 ts for %s: %d timesteps", station, len(combined))
        else:
            logger.warning("No ERA5 data for %s", station)
    return result


# ---------------------------------------------------------------------------
# ASOS actual Tmax
# ---------------------------------------------------------------------------

# IEM ASOS daily history is deep; this start safely precedes the ERA5 span
# (2021-05-27 →). End is open (today) so re-runs pick up new days.
_OFFICIAL_TMAX_START = date(2021, 1, 1)


def build_daily_tmax(station: str) -> pd.Series:
    """Daily Tmax (°F) target for `station`, sourced from the OFFICIAL NWS daily
    max at the station Kalshi settles on (IEM ASOS daily summary). Falls back to
    the hourly-METAR max if IEM is unavailable.

    This is the temperature Kalshi resolves on — using it as the training target
    closes the source mismatch that handicapped the model (see
    plans/nws-settlement-source.md).
    """
    from ingestion.nws_daily import official_daily_tmax_series, SETTLEMENT_STATION

    if station in SETTLEMENT_STATION:
        try:
            s = official_daily_tmax_series(station, _OFFICIAL_TMAX_START, date.today())
            if not s.empty:
                return s
            logger.warning("IEM returned no data for %s — falling back to hourly METAR max", station)
        except Exception as exc:
            logger.warning("IEM fetch failed for %s (%s) — falling back to hourly METAR max", station, exc)
    return build_asos_daily_tmax(station, STATION_REGISTRY[station].timezone)


def build_asos_daily_tmax(station: str, tz: str) -> pd.Series:
    """Return daily Tmax (°F) in local station time, indexed by date."""
    path = HIST_DIR / f"{station}_hourly.parquet"
    if not path.exists():
        logger.warning("No ASOS file for %s", station)
        return pd.Series(dtype=float)

    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["datetime_local"] = df["datetime"].dt.tz_convert(tz)
    df["date_local"] = df["datetime_local"].dt.date
    df = df.dropna(subset=["tmpf"])

    daily_tmax = df.groupby("date_local")["tmpf"].max()
    daily_tmax.index = pd.to_datetime(daily_tmax.index)
    return daily_tmax


# ---------------------------------------------------------------------------
# Feature computation from ERA5 time series
# ---------------------------------------------------------------------------

def _kelvin_to_f(k: float) -> float:
    return (k - 273.15) * 9 / 5 + 32 if not np.isnan(k) else float("nan")


def _cyclical(val: float, period: float) -> tuple[float, float]:
    return float(np.sin(2 * np.pi * val / period)), float(np.cos(2 * np.pi * val / period))


def _spread_proxy(ts: pd.DataFrame, center_time: pd.Timestamp, col: str, window_hours: int = 72) -> dict:
    lo = center_time - pd.Timedelta(hours=window_hours)
    hi = center_time + pd.Timedelta(hours=window_hours)
    window = ts.loc[lo:hi, col].dropna()
    if len(window) < 3:
        return {"std": float("nan"), "p10": float("nan"), "p25": float("nan"), "p50": float("nan"),
                "p75": float("nan"), "p90": float("nan"), "iqr": float("nan"),
                "range": float("nan")}
    # vals_f is already converted Kelvin -> Fahrenheit, so every stat below is
    # already in the right units — do not re-apply _kelvin_to_f to these.
    vals_f = np.array([_kelvin_to_f(v) for v in window])
    return {
        "std":   float(np.std(vals_f)),
        "p10":   float(np.percentile(vals_f, 10)),
        "p25":   float(np.percentile(vals_f, 25)),
        "p50":   float(np.percentile(vals_f, 50)),
        "p75":   float(np.percentile(vals_f, 75)),
        "p90":   float(np.percentile(vals_f, 90)),
        "iqr":   float(np.percentile(vals_f, 75) - np.percentile(vals_f, 25)),
        "range": float(np.percentile(vals_f, 90) - np.percentile(vals_f, 10)),
    }


def build_feature_rows(
    station: str,
    era5_ts: pd.DataFrame,
    daily_tmax: pd.Series,
) -> list[dict]:
    meta = STATION_REGISTRY[station]
    tz = meta.timezone
    rows = []

    # Pre-compute lag residuals: actual_tmax - era5 mean at D+1 (lead=24h)
    # Used for obs_minus_model_lag features
    lag_residuals: dict[date, float] = {}

    for verification_date, actual_tmax in daily_tmax.items():
        if pd.isna(actual_tmax):
            continue
        vdate = verification_date.date() if hasattr(verification_date, "date") else verification_date

        for lead_hour in LEAD_HOURS:
            # ERA5 initialization time: lead_hour before midnight UTC of verification date
            # Use tz-naive to match the ERA5 index (which comes out of netCDF4 as tz-naive UTC)
            init_utc = pd.Timestamp(vdate) - pd.Timedelta(hours=lead_hour)

            # Find closest ERA5 timestep
            if era5_ts.empty:
                continue
            time_diffs = abs(era5_ts.index - init_utc)
            if time_diffs.min() > pd.Timedelta(hours=7):
                continue
            closest_idx = time_diffs.argmin()
            closest_time = era5_ts.index[closest_idx]
            row_era5 = era5_ts.iloc[closest_idx]

            t2m_k  = row_era5.get("t2m", float("nan"))
            d2m_k  = row_era5.get("d2m", float("nan"))
            u10    = row_era5.get("u10", float("nan"))
            v10    = row_era5.get("v10", float("nan"))
            tcc    = row_era5.get("tcc", float("nan"))

            t2m_f  = _kelvin_to_f(t2m_k)
            d2m_f  = _kelvin_to_f(d2m_k)
            wind   = float(np.sqrt(u10**2 + v10**2)) if not (np.isnan(u10) or np.isnan(v10)) else float("nan")
            wdir_sin = float(np.sin(np.arctan2(u10, v10))) if not (np.isnan(u10) or np.isnan(v10)) else float("nan")
            wdir_cos = float(np.cos(np.arctan2(u10, v10))) if not (np.isnan(u10) or np.isnan(v10)) else float("nan")
            tcc_pct  = float(tcc) * 100.0 if not np.isnan(tcc) else float("nan")
            dp_dep   = t2m_f - d2m_f if not (np.isnan(t2m_f) or np.isnan(d2m_f)) else float("nan")

            # Cloud cover total
            cloud_total = min(max(float(tcc), 0.0), 1.0) if not np.isnan(tcc) else float("nan")

            # Spread proxies from ±SPREAD_WINDOW_HOURS of ERA5 (already in °F)
            sp_f = _spread_proxy(era5_ts, closest_time, "t2m", SPREAD_WINDOW_HOURS)
            skewness = _quantile_skewness(sp_f["p10"], sp_f["p50"], sp_f["p90"])
            kurtosis = _quantile_kurtosis(sp_f["p10"], sp_f["p25"], sp_f["p75"], sp_f["p90"])

            # Temporal features
            month = vdate.month
            doy   = vdate.timetuple().tm_yday
            doy_sin, doy_cos = _cyclical(doy, 365)
            lead_time_sqrt = float(np.sqrt(lead_hour))

            # Climatology anomaly
            climo_normal = climo_tmax_normal(station, month)
            climo_anomaly = (
                t2m_f - climo_normal
                if not (np.isnan(t2m_f) or np.isnan(climo_normal))
                else float("nan")
            )

            # Onshore/offshore wind component
            onshore_wind_component = _onshore_wind_component(
                wdir_sin, wdir_cos, meta.onshore_bearing_deg
            )

            # Lag residuals: last 3 days of (actual - era5_proxy) at D+1 lead
            lag1 = lag_residuals.get(vdate - timedelta(days=1), float("nan"))
            lag2 = lag_residuals.get(vdate - timedelta(days=2), float("nan"))
            lag3 = lag_residuals.get(vdate - timedelta(days=3), float("nan"))

            # Rolling residual stats over the last _ROLLING_WINDOW_DAYS days
            roll_vals = [
                lag_residuals[d]
                for d in (vdate - timedelta(days=n) for n in range(1, _ROLLING_WINDOW_DAYS + 1))
                if d in lag_residuals and not np.isnan(lag_residuals[d])
            ]
            if len(roll_vals) >= 2:
                roll_mean = float(np.mean(roll_vals))
                roll_std = float(np.std(roll_vals))
            else:
                roll_mean = roll_std = float("nan")

            # Store D+1 residual for future use
            if lead_hour == 24:
                lag_residuals[vdate] = actual_tmax - t2m_f

            row: dict = {
                "station":   station,
                "date":      vdate,
                "lead_hour": lead_hour,
                "actual_tmax": actual_tmax,
                # GEFS proxies (ERA5 as deterministic + spread from temporal window)
                "gefs_tmax_mean":          t2m_f,
                "gefs_tmax_std":           sp_f["std"],
                "gefs_tmax_range":         sp_f["range"],
                "gefs_tmax_iqr":           sp_f["iqr"],
                "gefs_tmax_p10":           sp_f["p10"],
                "gefs_tmax_p25":           sp_f["p25"],
                "gefs_tmax_p75":           sp_f["p75"],
                "gefs_tmax_p90":           sp_f["p90"],
                "gefs_ensemble_skewness":  skewness,
                "gefs_ensemble_kurtosis":  kurtosis,
                "gefs_tmax_climo_anomaly": climo_anomaly,
                # ECMWF = ERA5 (ERA5 is ECMWF's reanalysis)
                "ecmwf_tmax":              t2m_f,
                "ecmwf_tmin":             t2m_f,
                "ecmwf_gefs_tmax_delta":   0.0,
                # NBM — unavailable historically
                "nbm_t10":   float("nan"), "nbm_t25": float("nan"),
                "nbm_t50":   float("nan"), "nbm_t75": float("nan"),
                "nbm_t90":   float("nan"),
                "nbm_tmin":  float("nan"), "nbm_pop12": float("nan"),
                "nbm_spread": float("nan"), "nbm_gefs_delta": float("nan"),
                # Atmospheric
                "cloud_cover_total":          cloud_total,
                "surface_wind_speed":         wind,
                "wind_dir_sin":               wdir_sin,
                "wind_dir_cos":               wdir_cos,
                "onshore_wind_component":     onshore_wind_component,
                "surface_dew_point_depression": dp_dep,
                # Temporal
                "lead_time_sqrt": lead_time_sqrt,
                "day_of_year_sin": doy_sin, "day_of_year_cos": doy_cos,
                # Physical meta
                "elevation_delta_m":   meta.elevation_m,
                "uhi_index":           meta.uhi_index,
                "coastal_distance_km": meta.coastal_distance_km,
                # Residual lags + rolling stats
                "obs_minus_model_lag1": lag1,
                "obs_minus_model_lag2": lag2,
                "obs_minus_model_lag3": lag3,
                "obs_minus_model_roll_mean": roll_mean,
                "obs_minus_model_roll_std": roll_std,
                # Engineered: signed deltas, cross-model spread, normalized spread, accel
                # (ecmwf_tmax == gefs_tmax_mean == t2m_f in historical, so signed delta = 0)
                "gefs_ecmwf_delta_signed": 0.0 if not np.isnan(t2m_f) else float("nan"),
                "nbm_ecmwf_delta_signed":  float("nan"),  # NBM unavailable historically
                "multi_model_spread":      float("nan"),  # only 1 model available historically
                "ecmwf_climo_anomaly": (
                    t2m_f - climo_normal
                    if not (np.isnan(t2m_f) or np.isnan(climo_normal))
                    else float("nan")
                ),
                "gefs_spread_per_lead": (
                    sp_f["std"] / lead_time_sqrt
                    if (not np.isnan(sp_f["std"]) and lead_time_sqrt > 0)
                    else float("nan")
                ),
                "nbm_spread_per_lead": float("nan"),  # NBM spread unavailable historically
                "obs_minus_model_accel": (
                    lag1 - lag3
                    if not (np.isnan(lag1) or np.isnan(lag3))
                    else float("nan")
                ),
                # ecmwf_diurnal_range: historically t2m_f used for both tmax and tmin → 0
                "ecmwf_diurnal_range": 0.0 if not np.isnan(t2m_f) else float("nan"),
            }

            rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    out_path = HIST_DIR / "features.parquet"
    if out_path.exists():
        logger.info("features.parquet already exists — delete it to rebuild")
        return

    # Load ERA5 time series for all stations
    era5_ts = build_era5_station_ts(ALL_ICAO)

    # Pre-compute daily Tmax history per station (used for both climatology
    # normals below and the per-station feature rows in the main loop)
    daily_tmax_by_station = {
        station: build_daily_tmax(station)
        for station in ALL_ICAO
    }

    # Monthly Tmax climatology normals (station -> month -> mean Tmax °F),
    # written before any build_feature_rows call so climo_tmax_normal can load them
    normals: dict[str, dict[str, float]] = {}
    for station, daily_tmax in daily_tmax_by_station.items():
        if daily_tmax.empty:
            continue
        monthly_means = daily_tmax.groupby(daily_tmax.index.month).mean()
        normals[station] = {str(m): float(v) for m, v in monthly_means.items()}

    CLIMATOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLIMATOLOGY_PATH, "w") as f:
        json.dump(normals, f, indent=2)
    logger.info("Wrote climatology normals for %d stations to %s", len(normals), CLIMATOLOGY_PATH)

    all_rows = []
    for station in ALL_ICAO:
        logger.info("=== %s (%s) ===", station, STATION_REGISTRY[station].city)

        if station not in era5_ts:
            logger.warning("No ERA5 data for %s — skipping", station)
            continue

        daily_tmax = daily_tmax_by_station[station]
        if daily_tmax.empty:
            logger.warning("No ASOS data for %s — skipping", station)
            continue

        rows = build_feature_rows(station, era5_ts[station], daily_tmax)
        logger.info("%s: %d feature rows", station, len(rows))
        all_rows.extend(rows)

    if not all_rows:
        logger.error("No rows produced — check ERA5 and ASOS data")
        return

    df = pd.DataFrame(all_rows)

    # Ensure all feature columns present (fill missing with NaN)
    for col in get_feature_columns():
        if col not in df.columns:
            df[col] = float("nan")

    # Fill NBM NaNs with 0 for training (model learns they're absent)
    nbm_cols = [c for c in df.columns if c.startswith("nbm_")]
    df[nbm_cols] = df[nbm_cols].fillna(0.0)

    df = df.sort_values(["station", "date", "lead_hour"]).reset_index(drop=True)
    df.to_parquet(out_path, index=False)

    logger.info(
        "Saved features.parquet: %d rows × %d cols | stations=%d | dates %s → %s",
        len(df), len(df.columns), df["station"].nunique(),
        df["date"].min(), df["date"].max(),
    )
    logger.info("Target stats: actual_tmax mean=%.1f std=%.1f min=%.1f max=%.1f",
                df["actual_tmax"].mean(), df["actual_tmax"].std(),
                df["actual_tmax"].min(), df["actual_tmax"].max())


if __name__ == "__main__":
    main()
