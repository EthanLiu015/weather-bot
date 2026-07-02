"""Backfill ECMWF IFS ENS 51-member daily highs from dynamical.org Icechunk (cycle 7).

FAIR: dataset has 00Z inits only; ECMWF open-data 00Z posts ~07:55 UTC, before
the 14:00 UTC cutoff. `temperature_2m` is instantaneous 3-hourly — the sampled
daily max sits a touch below the true high, a level shift the walk-forward bias
fit absorbs. Point reads fetch KB-scale chunks; no full fields locally.

Appends model='ecmwf_ens' rows into ensemble_members.parquet next to the GEFS
rows (replacing any prior ecmwf_ens rows).

    PYTHONPATH=. python scripts/backfill_ecmwf_members_icechunk.py
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from scripts.backfill_gefs_members_zarr import (OUT, START, END, STATIONS,
                                                local_day_max, station_coords)

logger = logging.getLogger(__name__)

DATASET = "ecmwf-ifs-ens-forecast-15-day-0-25-degree"
LEAD_SLICE = slice(1, 12)  # 3h..33h, 3-hourly — covers every station's local day


def fetch_station(station: str, days: pd.DatetimeIndex) -> pd.DataFrame:
    import dynamical_catalog

    ds = dynamical_catalog.open(DATASET, chunks=None)
    da = ds["temperature_2m"]
    lat, lon, tz = station_coords(station)
    point = da.sel(latitude=lat, longitude=lon, method="nearest")
    rows = []
    for day in days:
        sel = point.sel(init_time=day).isel(lead_time=LEAD_SLICE)
        vals = sel.values  # (member, lead) or (lead, member) — check dims
        if sel.dims != ("ensemble_member", "lead_time"):
            vals = vals.T
        valid = pd.DatetimeIndex(day + pd.to_timedelta(sel.lead_time.values))
        tmax_c = local_day_max(vals, valid, tz, day)
        if tmax_c is None:
            continue
        for member, v in enumerate(tmax_c):
            if np.isnan(v):
                continue
            rows.append({"station": station, "date": day, "model": "ecmwf_ens",
                         "member": member, "tmax_f": float(v) * 9 / 5 + 32})
    logger.info("%s: %d member-days", station, len(rows))
    return pd.DataFrame(rows, columns=["station", "date", "model", "member", "tmax_f"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    days = pd.date_range(START, END, freq="D")
    frames = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(fetch_station, s, days): s for s in STATIONS}
        for fut in as_completed(futures):
            frames.append(fut.result())
    new = pd.concat(frames, ignore_index=True)
    try:
        existing = pd.read_parquet(OUT)
        existing = existing[existing["model"] != "ecmwf_ens"]
    except FileNotFoundError:
        existing = pd.DataFrame(columns=new.columns)
    out = pd.concat([existing, new], ignore_index=True)
    out.to_parquet(OUT)
    logger.info("wrote %d rows (%d ecmwf_ens) -> %s", len(out), len(new), OUT)


if __name__ == "__main__":
    main()
