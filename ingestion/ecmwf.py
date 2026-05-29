import asyncio
import logging
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import cfgrib
import numpy as np
import pandas as pd
import xarray as xr

from config.stations import station_coords, STATION_REGISTRY

logger = logging.getLogger(__name__)

STATION_COORDS = station_coords()

# Parameters to fetch from ECMWF open data
# 2t = 2m temperature (K), 10u/10v = 10m wind (m/s), tcc = total cloud cover (0-1), tp = precip (m)
ECMWF_PARAMS = ["2t", "10u", "10v", "tcc", "tp"]

# Forecast steps to download (hours). 3-hourly out to D6, then D7.
FORECAST_STEPS = [3, 6, 9, 12, 15, 18, 21, 24,
                  27, 30, 33, 36, 39, 42, 45, 48,
                  54, 60, 66, 72, 78, 84, 90, 96,
                  102, 108, 114, 120, 132, 144, 168]


def _nearest_val(da: xr.DataArray, lat: float, lon: float) -> float:
    lon_360 = lon % 360
    try:
        return float(da.sel(latitude=lat, longitude=lon_360, method="nearest").values)
    except Exception:
        try:
            return float(da.sel(lat=lat, lon=lon_360, method="nearest").values)
        except Exception:
            return float("nan")


def _parse_grib_file(path: str) -> list[xr.Dataset]:
    try:
        return cfgrib.open_datasets(path, backend_kwargs={"indexing_time": "valid_time"})
    except Exception as exc:
        logger.warning("cfgrib parse failed for %s: %s", path, exc)
        return []


def _extract_station_temp_series(
    datasets: list[xr.Dataset],
    station: str,
) -> pd.Series:
    lat, lon = STATION_COORDS[station]
    records: dict[pd.Timestamp, float] = {}

    for ds in datasets:
        t2m_var = None
        for candidate in ("t2m", "2t", "tmp2m"):
            if candidate in ds.data_vars:
                t2m_var = candidate
                break
        if t2m_var is None:
            continue

        da = ds[t2m_var]
        time_dim = None
        for dim in ("valid_time", "time", "step"):
            if dim in da.dims:
                time_dim = dim
                break

        if time_dim is None:
            val_k = _nearest_val(da, lat, lon)
            if not np.isnan(val_k):
                ts = pd.Timestamp.now()
                records[ts] = (val_k - 273.15) * 9 / 5 + 32
            continue

        for idx in range(len(da[time_dim])):
            try:
                slice_da = da.isel({time_dim: idx})
                val_k = _nearest_val(slice_da, lat, lon)
                if np.isnan(val_k):
                    continue
                val_f = (val_k - 273.15) * 9 / 5 + 32
                ts_val = slice_da[time_dim].values
                ts = pd.Timestamp(ts_val)
                records[ts] = val_f
            except Exception:
                continue

    if not records:
        return pd.Series(dtype=float)
    return pd.Series(records).sort_index()


def _compute_daily_tmax_tmin(
    temp_series: pd.Series,
    run_dt: datetime,
    lead_days: int = 7,
) -> dict[int, dict[str, float]]:
    if temp_series.empty:
        return {}

    result: dict[int, dict[str, float]] = {}
    for day_offset in range(1, lead_days + 1):
        target_date = (run_dt + timedelta(days=day_offset)).date()
        day_mask = temp_series.index.date == target_date  # type: ignore[union-attr]
        day_temps = temp_series[day_mask]
        if day_temps.empty:
            result[day_offset] = {"tmax": float("nan"), "tmin": float("nan")}
        else:
            result[day_offset] = {
                "tmax": float(day_temps.max()),
                "tmin": float(day_temps.min()),
            }
    return result


async def _download_ecmwf_opendata(data_dir: Path, run_dt: datetime) -> Path | None:
    from ecmwf.opendata import Client

    data_dir.mkdir(parents=True, exist_ok=True)
    date_str = run_dt.strftime("%Y%m%d")
    cycle = run_dt.hour  # 0 or 12
    out_path = data_dir / f"ecmwf_{date_str}_{cycle:02d}z.grib2"

    if out_path.exists() and out_path.stat().st_size > 10_000:
        logger.info("ECMWF cache hit: %s", out_path)
        return out_path

    def _download() -> bool:
        try:
            client = Client(source="ecmwf", beta=True)
            client.retrieve(
                date=date_str,
                time=cycle,
                step=FORECAST_STEPS,
                param=ECMWF_PARAMS,
                target=str(out_path),
            )
            logger.info("ECMWF downloaded to %s (%d bytes)", out_path, out_path.stat().st_size)
            return True
        except Exception as exc:
            logger.warning("ECMWF opendata download failed: %s", exc)
            if out_path.exists():
                out_path.unlink()
            return False

    success = await asyncio.get_event_loop().run_in_executor(None, _download)
    return out_path if success else None


def _latest_available_run(now: datetime) -> datetime:
    # ECMWF runs 00z and 12z; data typically available ~7h after run time
    for hours_back in range(0, 48, 12):
        candidate = now - timedelta(hours=hours_back)
        cycle = 12 if candidate.hour >= 19 else 0 if candidate.hour >= 7 else None
        if cycle is None:
            continue
        run_dt = candidate.replace(hour=cycle, minute=0, second=0, microsecond=0)
        if run_dt <= now:
            return run_dt
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


async def fetch_latest_ecmwf_run(data_dir: str = "data/ecmwf") -> dict:
    now = datetime.utcnow()
    run_dt = _latest_available_run(now)
    logger.info("Fetching ECMWF run %sz", run_dt.strftime("%Y%m%d %Hz"))

    grib_path = await _download_ecmwf_opendata(Path(data_dir), run_dt)

    if grib_path is None or not grib_path.exists():
        logger.warning("ECMWF download failed; returning NaN for all stations")
        return {s: {"tmax_forecast": float("nan"), "tmin_forecast": float("nan"),
                     "daily": {}, "run_date": run_dt.strftime("%Y%m%d"),
                     "cycle": f"{run_dt.hour:02d}"} for s in STATION_COORDS}

    datasets = _parse_grib_file(str(grib_path))
    if not datasets:
        logger.warning("ECMWF GRIB parse returned no datasets from %s", grib_path)
        return {s: {"tmax_forecast": float("nan"), "tmin_forecast": float("nan"),
                     "daily": {}, "run_date": run_dt.strftime("%Y%m%d"),
                     "cycle": f"{run_dt.hour:02d}"} for s in STATION_COORDS}

    result: dict[str, dict] = {}
    for station in STATION_COORDS:
        temp_series = _extract_station_temp_series(datasets, station)
        daily = _compute_daily_tmax_tmin(temp_series, run_dt)

        # D+1 is the primary forecast (closest to what Kalshi resolves on)
        d1 = daily.get(1, {})
        tmax = d1.get("tmax", float("nan"))
        tmin = d1.get("tmin", float("nan"))

        result[station] = {
            "tmax_forecast": tmax,
            "tmin_forecast": tmin,
            "daily": daily,
            "run_date": run_dt.strftime("%Y%m%d"),
            "cycle": f"{run_dt.hour:02d}",
            "temp_series_len": len(temp_series),
        }
        logger.info(
            "ECMWF %s D+1: tmax=%.1f°F tmin=%.1f°F (series len=%d)",
            station, tmax, tmin, len(temp_series),
        )

    return result


def extract_station_values(ds: xr.Dataset, station_coords: dict) -> pd.DataFrame:
    rows = []
    for station, (lat, lon) in station_coords.items():
        row = {"station": station}
        for var in ds.data_vars:
            row[var] = _nearest_val(ds[var], lat, lon)
        rows.append(row)
    return pd.DataFrame(rows).set_index("station")
