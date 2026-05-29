import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

import cfgrib
import httpx
import numpy as np
import xarray as xr

from config.stations import station_coords

logger = logging.getLogger(__name__)

NBM_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"
STATION_COORDS = station_coords()

# NBM runs 4× daily; data available ~1.5h after run time
NBM_CYCLES = [0, 6, 12, 18]

# Forecast hours to pull (these align with daily Tmax windows)
NBM_FORECAST_HOURS = [24, 48, 72, 96, 120]

# cfgrib filter configs for NBM probabilistic temperature percentiles
_PERCENTILE_FILTERS = {
    "t10":  {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t", "percentileValue": 10},
    "t25":  {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t", "percentileValue": 25},
    "t50":  {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t", "percentileValue": 50},
    "t75":  {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t", "percentileValue": 75},
    "t90":  {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t", "percentileValue": 90},
}

_DETERMINISTIC_FILTERS = {
    "tmax": {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "tmax"},
    "tmin": {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "tmin"},
    "pop12": {"typeOfLevel": "surface", "shortName": "pop"},
    "t2m":  {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t"},
}


def _nbm_url(date_str: str, cycle: str, fhour: int) -> str:
    return (
        f"{NBM_BASE}/blend.{date_str}/{cycle}/core/"
        f"blend.t{cycle}z.core.f{fhour:03d}.co.grib2"
    )


async def _download_file(url: str, dest: Path, client: httpx.AsyncClient) -> bool:
    for attempt in range(3):
        try:
            async with client.stream("GET", url, timeout=90.0) as resp:
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
            logger.warning("NBM download attempt %d failed: %s", attempt + 1, exc)
            await asyncio.sleep(wait)
    return False


def _nearest_val(da: xr.DataArray, lat: float, lon: float) -> float:
    lon_360 = lon % 360
    try:
        # NBM uses a Lambert conformal grid — try x/y dims first, fall back to lat/lon
        if "latitude" in da.coords and "longitude" in da.coords:
            return float(da.sel(latitude=lat, longitude=lon_360, method="nearest").values)
        # For projected grids, find nearest point manually
        lats = da.coords.get("latitude", da.coords.get("lat"))
        lons = da.coords.get("longitude", da.coords.get("lon"))
        if lats is None or lons is None:
            return float("nan")
        dist = (lats - lat) ** 2 + (lons - lon_360) ** 2
        idx = int(dist.argmin())
        return float(da.values.flat[idx])
    except Exception:
        return float("nan")


def _kelvin_to_f(k: float) -> float:
    return (k - 273.15) * 9 / 5 + 32 if not np.isnan(k) else float("nan")


def _parse_nbm_file(path: str, station: str) -> dict[str, float]:
    lat, lon = STATION_COORDS[station]
    result: dict[str, float] = {}

    # Try each percentile filter
    for key, fkeys in _PERCENTILE_FILTERS.items():
        try:
            ds = cfgrib.open_dataset(path, filter_by_keys=fkeys, indexpath=None)
            var = list(ds.data_vars)[0]
            val_k = _nearest_val(ds[var], lat, lon)
            result[key] = _kelvin_to_f(val_k)
        except Exception:
            result[key] = float("nan")

    # Deterministic fields
    for key, fkeys in _DETERMINISTIC_FILTERS.items():
        if key in result:
            continue
        try:
            ds = cfgrib.open_dataset(path, filter_by_keys=fkeys, indexpath=None)
            var = list(ds.data_vars)[0]
            val = _nearest_val(ds[var], lat, lon)
            if key in ("tmax", "tmin", "t2m"):
                result[key] = _kelvin_to_f(val)
            else:
                result[key] = float(val) if not np.isnan(val) else float("nan")
        except Exception:
            result[key] = float("nan")

    # Fallback: open_datasets without filters if nothing parsed
    if all(np.isnan(v) for v in result.values()):
        try:
            for ds in cfgrib.open_datasets(path, indexpath=None):
                for var in ds.data_vars:
                    if var not in result:
                        result[var] = _nearest_val(ds[var], lat, lon)
        except Exception:
            pass

    return result


def _latest_nbm_run(now: datetime) -> tuple[str, str] | None:
    for hours_back in range(0, 12, 1):
        candidate = now - timedelta(hours=hours_back)
        cycle = max(c for c in NBM_CYCLES if c <= candidate.hour)
        run_dt = candidate.replace(hour=cycle, minute=0, second=0, microsecond=0)
        # NBM available ~1.5h after run time
        if now >= run_dt + timedelta(hours=2):
            return run_dt.strftime("%Y%m%d"), f"{cycle:02d}"
    return None


async def fetch_latest_nbm(data_dir: str = "data/nbm") -> dict:
    """
    Returns:
        dict[station, dict[lead_hour, dict]]
        where each inner dict has keys: t10, t25, t50, t75, t90, tmax, tmin, pop12, spread
    """
    now = datetime.utcnow()
    run_info = _latest_nbm_run(now)

    empty = {
        s: {
            lh: {"t10": float("nan"), "t25": float("nan"), "t50": float("nan"),
                 "t75": float("nan"), "t90": float("nan"), "tmax": float("nan"),
                 "tmin": float("nan"), "pop12": float("nan"), "spread": float("nan")}
            for lh in NBM_FORECAST_HOURS
        }
        for s in STATION_COORDS
    }

    if run_info is None:
        logger.warning("No NBM run available yet")
        return empty

    date_str, cycle_str = run_info
    result: dict[str, dict[int, dict]] = {
        s: {} for s in STATION_COORDS
    }

    async with httpx.AsyncClient() as client:
        for fhour in NBM_FORECAST_HOURS:
            url = _nbm_url(date_str, cycle_str, fhour)
            dest = Path(data_dir) / date_str / cycle_str / f"nbm_f{fhour:03d}.grib2"

            if not dest.exists():
                ok = await _download_file(url, dest, client)
                if not ok:
                    logger.warning("NBM f%03d unavailable", fhour)
                    for station in STATION_COORDS:
                        result[station][fhour] = empty[station][fhour]
                    continue

            for station in STATION_COORDS:
                parsed = _parse_nbm_file(str(dest), station)

                t10 = parsed.get("t10", float("nan"))
                t90 = parsed.get("t90", float("nan"))
                spread = t90 - t10 if not (np.isnan(t10) or np.isnan(t90)) else float("nan")

                result[station][fhour] = {
                    "t10":   t10,
                    "t25":   parsed.get("t25", float("nan")),
                    "t50":   parsed.get("t50", float("nan")),
                    "t75":   parsed.get("t75", float("nan")),
                    "t90":   t90,
                    "tmax":  parsed.get("tmax", float("nan")),
                    "tmin":  parsed.get("tmin", float("nan")),
                    "pop12": parsed.get("pop12", float("nan")),
                    "spread": spread,
                }

    filled = sum(
        1 for s in STATION_COORDS for lh in NBM_FORECAST_HOURS
        if not np.isnan(result[s].get(lh, {}).get("t50", float("nan")))
    )
    logger.info("NBM %s %sz: %d station×hour records with valid t50", date_str, cycle_str, filled)
    return result
