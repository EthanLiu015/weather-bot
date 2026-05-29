import logging
from datetime import datetime
import httpx
import xarray as xr
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

ECMWF_OPEN_DATA_BASE = "https://data.ecmwf.int/forecasts"
STATION_COORDS = {
    "KORD": (41.978611, -87.904722),
    "KJFK": (40.639722, -73.778889),
    "KLAX": (33.942536, -118.408075),
}
FORECAST_HOURS = [24, 48, 72, 96, 120, 144, 168]


def _ecmwf_open_data_url(run_date: str, run_hour: str, step: int, stream: str = "oper") -> str:
    return (
        f"{ECMWF_OPEN_DATA_BASE}/{run_date}/{run_hour}z/{stream}/0p25/oper/"
        f"{run_date}{run_hour}0000-{step}h-oper-fc.grib2"
    )


def extract_station_values(ds: xr.Dataset, station_coords: dict) -> pd.DataFrame:
    rows = []
    for station, (lat, lon) in station_coords.items():
        lon_360 = lon % 360
        row = {"station": station}
        for var in ds.data_vars:
            try:
                val = float(ds[var].sel(latitude=lat, longitude=lon_360, method="nearest").values)
                row[var] = val
            except Exception:
                row[var] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("station")


async def fetch_latest_ecmwf_run(data_dir: str = "data/ecmwf") -> dict:
    now = datetime.utcnow()
    for cycle in ["12", "00"]:
        run_date = now.strftime("%Y%m%d")
        frames_by_station: dict[str, list[pd.DataFrame]] = {s: [] for s in STATION_COORDS}
        success = False

        async with httpx.AsyncClient() as client:
            for step in FORECAST_HOURS:
                url = _ecmwf_open_data_url(run_date, cycle, step)
                try:
                    resp = await client.head(url, timeout=10.0)
                    if resp.status_code != 200:
                        continue
                    logger.info("ECMWF file available: %s", url)
                    success = True
                except Exception as exc:
                    logger.warning("ECMWF probe failed for step %d: %s", step, exc)

        if success:
            result = {}
            for station in STATION_COORDS:
                result[station] = {
                    "tmax_forecast": np.nan,
                    "tmin_forecast": np.nan,
                    "run_date": run_date,
                    "cycle": cycle,
                }
            logger.info("ECMWF run %s %sz detected", run_date, cycle)
            return result

    logger.warning("No ECMWF run detected; returning empty")
    return {s: {"tmax_forecast": np.nan, "tmin_forecast": np.nan} for s in STATION_COORDS}
