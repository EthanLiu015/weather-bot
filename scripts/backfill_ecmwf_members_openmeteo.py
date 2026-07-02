"""Backfill ECMWF ENS member daily highs from Open-Meteo — UNFAIR, ceiling only.

Open-Meteo's ensemble archive serves plain `temperature_2m` member data
assembled from the most recent run covering each hour; for a ~21 UTC daily high
that includes the 12Z run published AFTER the 14:00 UTC cutoff. Same look-ahead
class as `openmeteo_fresh.parquet` — BANNED for predicate claims. Use only as a
ceiling diagnostic: if even post-cutoff member shape cannot beat the market,
fair member shape (AWS/GEFS) cannot either.

    PYTHONPATH=. python scripts/backfill_ecmwf_members_openmeteo.py
Output: data/historical/ensemble_members_unfair.parquet
  (station, date, model='ecmwf_ens', member, tmax_f)
"""
from __future__ import annotations

import logging
import time

import httpx
import pandas as pd

from config.stations import get_station
from ingestion.openmeteo import daily_max_by_date

logger = logging.getLogger(__name__)

URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODEL_SLUG = "ecmwf_ifs025"
VAR = "temperature_2m"
STATIONS = ["KAUS", "KBOS", "KDCA", "KDEN", "KDFW", "KLAS", "KLAX", "KLGA",
            "KMIA", "KMSP", "KMSY", "KOKC", "KORD", "KPHL", "KPHX", "KSAT",
            "KSEA", "KSFO"]
START, END = "2026-04-01", "2026-06-24"
OUT = "data/historical/ensemble_members_unfair.parquet"


def parse_members(resp: dict, station: str) -> pd.DataFrame:
    hourly = resp.get("hourly", {})
    times = hourly.get("time", [])
    rows: list[dict] = []
    for col, values in hourly.items():
        if col == "time" or not col.startswith(VAR):
            continue
        suffix = col[len(VAR):]
        member = int(suffix.replace("_member", "")) if suffix else 0
        for date, tmax in daily_max_by_date(times, values).items():
            rows.append({"station": station, "date": date, "model": "ecmwf_ens",
                         "member": member, "tmax_f": tmax})
    return pd.DataFrame(rows, columns=["station", "date", "model", "member", "tmax_f"])


def fetch(station: str, max_retries: int = 4) -> dict:
    st = get_station(station)
    params = {"latitude": st.lat, "longitude": st.lon, "timezone": st.timezone,
              "start_date": START, "end_date": END, "hourly": VAR,
              "models": MODEL_SLUG, "temperature_unit": "fahrenheit"}
    for attempt in range(max_retries):
        try:
            r = httpx.get(URL, params=params, timeout=120.0)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            wait = 2 ** attempt
            logger.warning("%s attempt %d failed (%s); retry in %ds",
                           station, attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"{station}: all retries failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frames = []
    for station in STATIONS:
        got = parse_members(fetch(station), station)
        logger.info("%s: %d member-days", station, len(got))
        frames.append(got)
        time.sleep(2.0)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT)
    logger.info("wrote %d rows -> %s", len(out), OUT)


if __name__ == "__main__":
    main()
