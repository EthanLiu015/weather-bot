import asyncio
import logging
from datetime import datetime, timedelta
from pathlib import Path

import eccodes
import httpx
import numpy as np
from gribapi.errors import GribInternalError

from config.stations import station_coords
from ingestion.cache_pruning import prune_run_cache

logger = logging.getLogger(__name__)

NBM_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"
STATION_COORDS = station_coords()

# NBM runs 4× daily; data available ~1.5h after run time
NBM_CYCLES = [0, 6, 12, 18]

# Default lead-hour keys for fetch_latest_nbm's result shape when no run is
# available yet (the actual keys depend on cycle — see _nbm_lead_fhours).
NBM_FORECAST_HOURS = [24, 48, 72, 96, 120]

# GRIB key filters for the NBM "core" fields. The ensemble mean and ensemble
# std-dev share these keys and are distinguished only by `derivedForecast`
# (absent for the mean, =2 for std dev — WMO table 4.7 "Standard deviation").
_TMAX_FILTER = {"shortName": "tmax", "typeOfLevel": "heightAboveGround", "level": 2, "typeOfStatisticalProcessing": 2}
_TMIN_FILTER = {"shortName": "tmin", "typeOfLevel": "heightAboveGround", "level": 2, "typeOfStatisticalProcessing": 3}
# NBM core files carry separate APCP probability messages for the 1h, 6h, and
# 12h accumulation windows that share every other key — lengthOfTimeRange is
# the only field distinguishing the true POP12 message from POP1/POP6.
_POP12_FILTER = {"shortName": "tp", "typeOfLevel": "surface", "typeOfStatisticalProcessing": 1, "probabilityType": 1, "lowerLimit": 255.0, "lengthOfTimeRange": 12}

# Normal-quantile z-scores used to derive percentile temperatures from
# NBM's tmax mean + std-dev fields (NBM does not publish 2m-temp percentiles directly).
_PERCENTILE_Z_SCORES = {
    "t10": -1.2815515655446004,
    "t25": -0.6744897501960817,
    "t75": 0.6744897501960817,
    "t90": 1.2815515655446004,
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


def _kelvin_to_f(k: float) -> float:
    return (k - 273.15) * 9 / 5 + 32 if not np.isnan(k) else float("nan")


def _kelvin_delta_to_f(k: float) -> float:
    """Convert a delta/spread quantity (e.g. std dev) from Kelvin to Fahrenheit
    scale — no offset, since deltas don't shift with the zero point."""
    return k * 9 / 5 if not np.isnan(k) else float("nan")


def _grib_safe_get(gid, key):
    try:
        return eccodes.codes_get(gid, key)
    except Exception:
        return None


# Keys identifying a GRIB grid's layout. NBM's CONUS grid is identical across
# the whole archive (2020-present), so flat-array indices computed once via
# _grid_nearest_indices can be reused for every later file's codes_get_values
# array — verified against this signature by _grib_values_at_indices.
_GRID_SIGNATURE_KEYS = ["Ni", "Nj", "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees", "gridType"]


def _grid_signature(gid) -> tuple:
    return tuple(eccodes.codes_get(gid, k) for k in _GRID_SIGNATURE_KEYS)


def _grid_nearest_indices(path: str, points: dict[str, tuple[float, float]]) -> tuple[dict[str, int], tuple]:
    """Return ({name: flat grid-array index nearest to (lat, lon)}, grid
    signature) for every point, computed from `path`'s first message.
    `codes_grib_find_nearest`'s `index` is a single O(grid) search done once
    per point here — the result is reused across many files via
    _grib_values_at_indices instead of repeating the search per file."""
    with open(path, "rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
    try:
        signature = _grid_signature(gid)
        indices = {}
        for name, (lat, lon) in points.items():
            nearest = eccodes.codes_grib_find_nearest(gid, lat, lon % 360)
            indices[name] = nearest[0]["index"]
        return indices, signature
    finally:
        eccodes.codes_release(gid)


def _unflip_alternating_rows(values, ni: int, nj: int):
    """Un-flip a boustrophedon (alternating-row) scanned value array to plain
    row-major order. Odd rows are stored reversed; reverse them back so the
    array matches the grid's latitudes/longitudes arrays (and find_nearest
    indices). Assumes i-consecutive rows of width `ni`."""
    grid = np.asarray(values, dtype=float).reshape(nj, ni).copy()
    grid[1::2] = grid[1::2, ::-1]
    return grid.ravel()


def _normalize_scan_order(values, gid):
    """Reorder `values` to match the grid's latitudes/longitudes arrays.

    NBM's CONUS grid uses alternating-row (boustrophedon) scanning
    (scanningMode bit 0x10): codes_get_values returns odd rows reversed, while
    codes_get_array('latitudes') and codes_grib_find_nearest use plain row-major
    order — so a geographically-correct flat index reads the WRONG cell's value
    (e.g. KATL read 98.5°F instead of 82.7°F). Un-flip the odd rows to fix this.
    No-op for normally-scanned grids (e.g. the synthetic test fixtures)."""
    scanning_mode = _grib_safe_get(gid, "scanningMode") or 0
    alternating = bool(scanning_mode & 0x10)
    i_consecutive = not (scanning_mode & 0x20)
    if not (alternating and i_consecutive):
        return values
    ni = eccodes.codes_get(gid, "Ni")
    nj = eccodes.codes_get(gid, "Nj")
    return _unflip_alternating_rows(values, ni, nj)


def _grib_values_at_indices(
    path: str,
    match_keys: dict,
    indices: dict[str, int],
    grid_signature: tuple,
    derived_forecast: int | None = None,
) -> dict[str, float]:
    """Scan GRIB2 messages in `path` for the first message whose keys match
    `match_keys` and whose `derivedForecast` equals `derived_forecast`, and
    return {name: value at that grid index} for each (name, index) in
    `indices`. Returns NaN for every name if the file is missing, no message
    matches, or the message's grid doesn't match `grid_signature` (the
    cached `indices` would be meaningless on a different grid)."""
    nan_result = {name: float("nan") for name in indices}
    try:
        with open(path, "rb") as f:
            while True:
                gid = eccodes.codes_grib_new_from_file(f)
                if gid is None:
                    break
                try:
                    if _grib_safe_get(gid, "derivedForecast") != derived_forecast:
                        continue
                    if not all(_grib_safe_get(gid, k) == v for k, v in match_keys.items()):
                        continue
                    if _grid_signature(gid) != grid_signature:
                        logger.error("NBM grid signature mismatch for %s; skipping", path)
                        return nan_result
                    values = _normalize_scan_order(eccodes.codes_get_values(gid), gid)
                    return {name: float(values[idx]) for name, idx in indices.items()}
                finally:
                    eccodes.codes_release(gid)
    except (OSError, GribInternalError) as exc:
        # A range fetch that failed mid-transfer can leave a partial/corrupted
        # GRIB2 file on disk; eccodes raises GribInternalError subclasses
        # (PrematureEndOfFileError, WrongLengthError, ...) when reading these.
        logger.warning("Failed to read GRIB message from %s: %s", path, exc)
    return nan_result


def _derive_nbm_percentiles(t50: float, spread: float) -> dict[str, float]:
    """Derive t10/t25/t75/t90 from the NBM tmax mean (t50) and std dev
    (spread), assuming a normal distribution — NBM does not publish 2m-temp
    percentile fields directly."""
    if np.isnan(t50) or np.isnan(spread):
        return {key: float("nan") for key in _PERCENTILE_Z_SCORES}
    return {key: t50 + z * spread for key, z in _PERCENTILE_Z_SCORES.items()}


def _parse_nbm_files_for_stations(
    tmax_path: str,
    tmin_path: str | None,
    indices: dict[str, int],
    grid_signature: tuple,
) -> dict[str, dict[str, float]]:
    """Extract tmax/tmin/pop12 from NBM core GRIB2 files for every station in
    `indices` and derive the percentile fields from tmax's mean and std dev.

    `tmax_path` is the core file for the TMAX window (also carries POP12);
    `tmin_path` is the core file for the paired TMIN window, or None if
    unavailable (tmin is left as NaN in that case). `indices`/`grid_signature`
    come from _grid_nearest_indices.
    """
    tmax_k = _grib_values_at_indices(tmax_path, _TMAX_FILTER, indices, grid_signature, derived_forecast=None)
    tmax_std_k = _grib_values_at_indices(tmax_path, _TMAX_FILTER, indices, grid_signature, derived_forecast=2)
    pop12 = _grib_values_at_indices(tmax_path, _POP12_FILTER, indices, grid_signature, derived_forecast=None)
    tmin_k = (
        _grib_values_at_indices(tmin_path, _TMIN_FILTER, indices, grid_signature, derived_forecast=None)
        if tmin_path is not None
        else {name: float("nan") for name in indices}
    )

    result = {}
    for name in indices:
        t50 = _kelvin_to_f(tmax_k[name])
        spread = _kelvin_delta_to_f(tmax_std_k[name])
        result[name] = {
            **_derive_nbm_percentiles(t50, spread),
            "t50": t50,
            "tmax": t50,
            "tmin": _kelvin_to_f(tmin_k[name]),
            "pop12": pop12[name],
            "spread": spread,
        }
    return result


def _nbm_lead_fhours(cycle: int, n_days: int = 5) -> list[tuple[int, int, int]]:
    """Return `n_days` (lead_hour, tmax_fhour, tmin_fhour) tuples for the
    given NBM cycle.

    NBM core files carry TMAX/TMIN window messages every 12h, with each
    window covering either 12-24Z (TMAX) or 00-12Z (TMIN). Which forecast
    hour holds which field depends on the cycle — e.g. for a 00z run, f024
    covers 12-24Z (TMAX) and f012 covers 00-12Z (TMIN); for a 12z run this is
    reversed (f012=TMAX, f024=TMIN). `lead_hour` (the larger forecast hour of
    each pair) is used as the dict key in fetch_latest_nbm's result.
    """
    base = (12 - cycle) % 12 or 12
    pairs = [(base + 12 * (2 * k), base + 12 * (2 * k + 1)) for k in range(n_days)]
    result = []
    for low, high in pairs:
        if (cycle + low) % 24 == 0:
            tmax_fhour, tmin_fhour = low, high
        else:
            tmax_fhour, tmin_fhour = high, low
        # The TMAX (daytime-max) window must cover the VERIFICATION day, not the
        # init day. TMAX/TMIN windows recur every 24h, so advancing both forecast
        # hours by 24 moves to the verification day's windows while keeping
        # `lead_hour` (= hours from init to that day's 00z) as the key. Without
        # this the result held the init day's high — one day too early.
        result.append((high, tmax_fhour + 24, tmin_fhour + 24))
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


async def _ensure_downloaded(url: str, dest: Path, client: httpx.AsyncClient) -> bool:
    if dest.exists():
        return True
    return await _download_file(url, dest, client)


async def fetch_latest_nbm(data_dir: str = "data/nbm") -> dict:
    """
    Returns:
        dict[station, dict[lead_hour, dict]]
        where each inner dict has keys: t10, t25, t50, t75, t90, tmax, tmin, pop12, spread
    """
    now = datetime.utcnow()
    run_info = _latest_nbm_run(now)

    empty_slot = {
        "t10": float("nan"), "t25": float("nan"), "t50": float("nan"),
        "t75": float("nan"), "t90": float("nan"), "tmax": float("nan"),
        "tmin": float("nan"), "pop12": float("nan"), "spread": float("nan"),
    }
    empty = {
        s: {lh: dict(empty_slot) for lh in NBM_FORECAST_HOURS}
        for s in STATION_COORDS
    }

    if run_info is None:
        logger.warning("No NBM run available yet")
        return empty

    date_str, cycle_str = run_info
    lead_fhours = _nbm_lead_fhours(int(cycle_str))
    result: dict[str, dict[int, dict]] = {
        s: {} for s in STATION_COORDS
    }

    # Grid indices/signature are computed once from the first downloaded
    # file and reused for every lead_hour — NBM's CONUS grid is identical
    # across runs, so this avoids an O(grid) nearest-neighbor search per
    # station per file (see _grid_nearest_indices).
    grid_indices: dict[str, int] | None = None
    grid_signature: tuple | None = None

    async with httpx.AsyncClient() as client:
        for lead_hour, tmax_fhour, tmin_fhour in lead_fhours:
            tmax_dest = Path(data_dir) / date_str / cycle_str / f"nbm_f{tmax_fhour:03d}.grib2"
            tmin_dest = Path(data_dir) / date_str / cycle_str / f"nbm_f{tmin_fhour:03d}.grib2"

            tmax_ok = await _ensure_downloaded(_nbm_url(date_str, cycle_str, tmax_fhour), tmax_dest, client)
            if not tmax_ok:
                logger.warning("NBM f%03d unavailable", tmax_fhour)
                for station in STATION_COORDS:
                    result[station][lead_hour] = dict(empty_slot)
                continue

            tmin_ok = await _ensure_downloaded(_nbm_url(date_str, cycle_str, tmin_fhour), tmin_dest, client)
            if not tmin_ok:
                logger.warning("NBM f%03d (tmin) unavailable", tmin_fhour)

            if grid_indices is None:
                grid_indices, grid_signature = _grid_nearest_indices(str(tmax_dest), STATION_COORDS)

            parsed = _parse_nbm_files_for_stations(
                str(tmax_dest),
                str(tmin_dest) if tmin_ok else None,
                grid_indices, grid_signature,
            )
            for station in STATION_COORDS:
                result[station][lead_hour] = parsed[station]

    filled = sum(
        1 for s in STATION_COORDS for lead_hour, _, _ in lead_fhours
        if not np.isnan(result[s].get(lead_hour, {}).get("t50", float("nan")))
    )
    logger.info("NBM %s %sz: %d station×hour records with valid t50", date_str, cycle_str, filled)

    if filled > 0:
        removed = prune_run_cache(data_dir, keep=2)
        if removed:
            logger.info("Pruned old NBM run cache dirs: %s", removed)

    return result
