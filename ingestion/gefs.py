import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import httpx
import xarray as xr
import pandas as pd
import numpy as np
import cfgrib  # noqa: F401 — ensures eccodes backend available

from config.stations import station_coords

logger = logging.getLogger(__name__)

NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod"
FORECAST_HOURS = [6, 12, 24, 48, 72, 96, 120, 168]
STATION_COORDS = station_coords()

# cfgrib filter keys for each variable group in the pgrb2s file
_FILTER_GROUPS = [
    {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t"},
    {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2d"},
    {"typeOfLevel": "heightAboveGround", "level": 10, "shortName": "10u"},
    {"typeOfLevel": "heightAboveGround", "level": 10, "shortName": "10v"},
    {"typeOfLevel": "atmosphere",  "shortName": "tcc"},
    {"typeOfLevel": "surface",     "shortName": "tp"},
    {"typeOfLevel": "surface",     "shortName": "sp"},
]


def _extract_var(ds: xr.Dataset, station: str) -> dict[str, float]:
    lat, lon = STATION_COORDS[station]
    lon_360 = lon % 360
    out: dict[str, float] = {}
    for var in ds.data_vars:
        try:
            val = float(
                ds[var].sel(latitude=lat, longitude=lon_360, method="nearest").values
            )
            out[var] = val
        except Exception:
            out[var] = float("nan")
    return out


def _parse_member_file(path: str, station: str) -> dict[str, float]:
    combined: dict[str, float] = {}
    # Try each variable group separately — cfgrib requires homogeneous messages
    for fkeys in _FILTER_GROUPS:
        try:
            ds = cfgrib.open_dataset(path, filter_by_keys=fkeys, indexpath=None)
            combined.update(_extract_var(ds, station))
        except Exception:
            pass
    # Fallback: open_datasets without filters
    if not combined:
        try:
            for ds in cfgrib.open_datasets(path, indexpath=None):
                combined.update(_extract_var(ds, station))
        except Exception:
            pass
    return combined


def _kelvin_to_f(k: float) -> float:
    return (k - 273.15) * 9 / 5 + 32 if not np.isnan(k) else float("nan")


def _wind_speed(u: float, v: float) -> float:
    return float(np.sqrt(u**2 + v**2)) if not (np.isnan(u) or np.isnan(v)) else float("nan")


def _wind_dir_sin(u: float, v: float) -> float:
    if np.isnan(u) or np.isnan(v):
        return float("nan")
    angle = float(np.arctan2(u, v))
    return float(np.sin(angle))


def _wind_dir_cos(u: float, v: float) -> float:
    if np.isnan(u) or np.isnan(v):
        return float("nan")
    angle = float(np.arctan2(u, v))
    return float(np.cos(angle))


async def _download_file(url: str, dest: Path, client: httpx.AsyncClient) -> bool:
    for attempt in range(5):
        try:
            async with client.stream("GET", url, timeout=120.0) as resp:
                if resp.status_code == 404:
                    return False
                resp.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
            return True
        except (httpx.HTTPError, OSError) as exc:
            wait = 2**attempt
            logger.warning("Download attempt %d failed for %s: %s", attempt + 1, url, exc)
            await asyncio.sleep(wait)
    return False


def _build_member_url(date_str: str, cycle: str, member: str, fhour: int) -> str:
    return (
        f"{NOMADS_BASE}/gefs.{date_str}/{cycle}/atmos/pgrb2sp25/"
        f"ge{member}.t{cycle}z.pgrb2s.0p25.f{fhour:03d}"
    )


async def fetch_latest_gefs_run(data_dir: str = "data/gefs") -> dict:
    """
    Returns:
        dict[station, dict[lead_hour, list[dict]]]
        where each inner list has one dict per member with keys:
          temp_f, dewpoint_f, wind_speed, wind_dir_sin, wind_dir_cos, tcc, tp, sp
    """
    now = datetime.utcnow()
    for hours_back in range(0, 24, 6):
        run_dt = now - timedelta(hours=hours_back)
        cycle_hour = (run_dt.hour // 6) * 6
        run_dt = run_dt.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
        date_str = run_dt.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"

        members = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]

        # result[station][lead_hour] = list of per-member dicts
        result: dict[str, dict[int, list[dict]]] = {
            s: {lh: [] for lh in FORECAST_HOURS} for s in STATION_COORDS
        }

        async with httpx.AsyncClient() as client:
            for member in members:
                for fhour in FORECAST_HOURS:
                    url = _build_member_url(date_str, cycle_str, member, fhour)
                    dest = Path(data_dir) / date_str / cycle_str / f"ge{member}_f{fhour:03d}.grib2"

                    if not dest.exists():
                        ok = await _download_file(url, dest, client)
                        if not ok:
                            logger.debug("Member %s fhour %d unavailable", member, fhour)
                            continue

                    for station in STATION_COORDS:
                        raw = _parse_member_file(str(dest), station)
                        if not raw:
                            continue

                        t2m = raw.get("t2m", float("nan"))
                        d2m = raw.get("d2m", float("nan"))
                        u10 = raw.get("u10", float("nan"))
                        v10 = raw.get("v10", float("nan"))

                        member_data = {
                            "member": member,
                            "temp_f": _kelvin_to_f(t2m),
                            "dewpoint_f": _kelvin_to_f(d2m),
                            "wind_speed": _wind_speed(u10, v10),
                            "wind_dir_sin": _wind_dir_sin(u10, v10),
                            "wind_dir_cos": _wind_dir_cos(u10, v10),
                            "tcc": raw.get("tcc", float("nan")),
                            "tp": raw.get("tp", float("nan")),
                            "sp": raw.get("sp", float("nan")),
                        }
                        result[station][fhour].append(member_data)

        members_found = sum(
            len(result[s][lh]) for s in STATION_COORDS for lh in FORECAST_HOURS
        )
        if members_found > 0:
            logger.info(
                "GEFS run %s %sz: %d member×station×hour records",
                date_str, cycle_str, members_found,
            )
            return result

    logger.error("No GEFS data available in last 24 hours")
    return {s: {lh: [] for lh in FORECAST_HOURS} for s in STATION_COORDS}


async def detect_new_run(last_run_time: datetime) -> bool:
    now = datetime.utcnow()
    cycle_hour = (now.hour // 6) * 6
    latest_run_dt = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    if latest_run_dt > last_run_time:
        date_str = latest_run_dt.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"
        probe_url = _build_member_url(date_str, cycle_str, "c00", 6)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.head(probe_url, timeout=10.0)
                return resp.status_code == 200
        except Exception:
            return False
    return False


def extract_station_values(ds: xr.Dataset, station_coords_: dict) -> pd.DataFrame:
    rows = []
    for station, (lat, lon) in station_coords_.items():
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
