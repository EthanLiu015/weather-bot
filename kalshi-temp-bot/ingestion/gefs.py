import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path
import httpx
import xarray as xr
import pandas as pd
import numpy as np
import cfgrib  # noqa: F401 — ensures eccodes backend available

logger = logging.getLogger(__name__)

NOMADS_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod"
FORECAST_HOURS = [6, 12, 24, 48, 72, 96, 120, 168]
GRIB_VARS = {
    "TMP:2 m": ("2m_temperature", "t2m"),
    "TCDC:entire atmosphere": ("total_cloud_cover", "tcc"),
    "UGRD:10 m": ("10m_u_component_of_wind", "u10"),
    "VGRD:10 m": ("10m_v_component_of_wind", "v10"),
    "DPT:2 m": ("2m_dewpoint_temperature", "d2m"),
    "APCP:surface": ("total_precipitation", "tp"),
    "PRES:surface": ("surface_pressure", "sp"),
}
STATION_COORDS = {
    "KORD": (41.978611, -87.904722),
    "KJFK": (40.639722, -73.778889),
    "KLAX": (33.942536, -118.408075),
}


def _build_member_url(date_str: str, cycle: str, member: str, fhour: int) -> str:
    fhour_str = f"f{fhour:03d}"
    return (
        f"{NOMADS_BASE}/gefs.{date_str}/{cycle}/atmos/pgrb2sp25/"
        f"ge{member}.t{cycle}z.pgrb2s.0p25.{fhour_str}"
    )


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
            logger.warning("Download attempt %d failed for %s: %s; retrying in %ds", attempt + 1, url, exc, wait)
            await asyncio.sleep(wait)
    logger.error("All download attempts failed for %s", url)
    return False


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


async def fetch_latest_gefs_run(
    data_dir: str = "data/gefs",
    max_wait_minutes: int = 120,
) -> dict:
    now = datetime.utcnow()
    for hours_back in range(0, 24, 6):
        run_dt = now - timedelta(hours=hours_back)
        run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
        cycle_hour = (run_dt.hour // 6) * 6
        run_dt = run_dt.replace(hour=cycle_hour)
        date_str = run_dt.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"

        members = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]
        result: dict[str, dict[str, pd.DataFrame]] = {s: {} for s in STATION_COORDS}

        async with httpx.AsyncClient() as client:
            for member in members:
                member_frames = []
                member_ok = True
                for fhour in FORECAST_HOURS:
                    url = _build_member_url(date_str, cycle_str, member, fhour)
                    dest = Path(data_dir) / date_str / cycle_str / f"ge{member}_f{fhour:03d}.grib2"
                    if not dest.exists():
                        ok = await _download_file(url, dest, client)
                        if not ok:
                            logger.warning("Member %s fhour %d unavailable", member, fhour)
                            member_ok = False
                            continue
                    try:
                        ds_list = cfgrib.open_datasets(str(dest))
                        station_dfs = []
                        for ds in ds_list:
                            sdf = extract_station_values(ds, STATION_COORDS)
                            station_dfs.append(sdf)
                        combined = pd.concat(station_dfs, axis=1)
                        combined["lead_hour"] = fhour
                        member_frames.append(combined)
                    except Exception as exc:
                        logger.warning("GRIB parse failed for member %s fhour %d: %s", member, fhour, exc)

                if member_frames:
                    df_member = pd.concat(member_frames)
                    for station in STATION_COORDS:
                        if station in df_member.index:
                            result[station][member] = df_member.loc[[station]]

        if any(result[s] for s in STATION_COORDS):
            logger.info("GEFS run %s %sz ingested with %d members", date_str, cycle_str, len(members))
            return result

    logger.error("Could not fetch any GEFS run in the last 24 hours")
    return {s: {} for s in STATION_COORDS}


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
