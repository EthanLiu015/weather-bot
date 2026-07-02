"""Backfill GEFS 31-member daily highs from dynamical.org's cloud Zarr (cycle 7).

FAIR: only the 00Z init of settlement day d is used — GEFS 00Z posts ~04-07 UTC,
well before the 14:00 UTC decision cutoff. Point reads against the
analysis-ready Zarr fetch only the chunks covering each station (KB-scale), so
nothing near a full GRIB field ever touches local memory or disk.

`maximum_temperature_2m` is the 3-hour window max, matched to the station's
local calendar day (Kalshi settles on the CLI local-day high). Coordinates are
the TRUE settlement stations (KNYC / KMDW for NY / Chicago — cycle 4 finding).

    PYTHONPATH=. python scripts/backfill_gefs_members_zarr.py
Output: data/historical/ensemble_members.parquet
  (station, date, model='gefs', member, tmax_f)
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr

from config.stations import get_station

logger = logging.getLogger(__name__)

ZARR_URL = "https://data.dynamical.org/noaa/gefs/forecast-35-day/latest.zarr"
START, END = "2026-04-01", "2026-06-24"
LEAD_SLICE = slice(1, 12)  # 3h..33h — covers every station's local calendar day d
STATIONS = ["KAUS", "KBOS", "KDCA", "KDEN", "KDFW", "KLAS", "KLAX", "KLGA",
            "KMIA", "KMSP", "KMSY", "KOKC", "KORD", "KPHL", "KPHX", "KSAT",
            "KSEA", "KSFO"]
# Kalshi settles NY on Central Park and Chicago on Midway, not the ICAO airport
SETTLEMENT_COORDS = {"KLGA": (40.7789, -73.9692), "KORD": (41.7861, -87.7522)}
OUT = "data/historical/ensemble_members.parquet"
N_WORKERS = 4


def station_coords(station: str) -> tuple[float, float, str]:
    st = get_station(station)
    lat, lon = SETTLEMENT_COORDS.get(station, (st.lat, st.lon))
    return lat, lon, st.timezone


def local_day_max(vals: np.ndarray, valid_utc: pd.DatetimeIndex, tz: str,
                  day: pd.Timestamp) -> np.ndarray | None:
    """Per-member max over the 3h-window steps whose valid time falls on the
    station's local calendar date `day`. vals: (member, lead)."""
    local_dates = valid_utc.tz_localize("UTC").tz_convert(tz).date
    mask = local_dates == day.date()
    if not mask.any():
        return None
    return np.nanmax(vals[:, mask], axis=1)


def fetch_station(station: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Each worker opens its own dataset handle — xarray's lazy pandas indexes
    are not safe to build concurrently on a shared object."""
    ds = xr.open_zarr(ZARR_URL, decode_timedelta=True, chunks=None)
    da = ds["maximum_temperature_2m"]
    lat, lon, tz = station_coords(station)
    point = da.sel(latitude=lat, longitude=lon, method="nearest")
    rows = []
    for day in days:
        sel = point.sel(init_time=day).isel(lead_time=LEAD_SLICE)
        vals = sel.values  # (member, lead), deg C — a few KB
        valid = pd.DatetimeIndex(day + pd.to_timedelta(sel.lead_time.values))
        tmax_c = local_day_max(vals, valid, tz, day)
        if tmax_c is None:
            continue
        for member, v in enumerate(tmax_c):
            if np.isnan(v):
                continue
            rows.append({"station": station, "date": day, "model": "gefs",
                         "member": member, "tmax_f": float(v) * 9 / 5 + 32})
    logger.info("%s: %d member-days", station, len(rows))
    return pd.DataFrame(rows, columns=["station", "date", "model", "member", "tmax_f"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    days = pd.date_range(START, END, freq="D")
    frames = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(fetch_station, s, days): s for s in STATIONS}
        for fut in as_completed(futures):
            frames.append(fut.result())
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT)
    logger.info("wrote %d rows (%d station-days) -> %s", len(out),
                out.groupby(["station", "date"]).ngroups, OUT)


if __name__ == "__main__":
    main()
