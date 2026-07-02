"""Backfill raw ensemble-member daily highs (cycle 5) — FAIR lead.

Open-Meteo Ensemble API `temperature_2m_previous_day1` gives, for each hour,
the value predicted by the run issued ~24h earlier — every run is pre-14:00 UTC
on settlement day, so this is legal under the fairness rule (same staleness
class as openmeteo_multimodel.parquet). Requesting one model per call keeps
member columns unsuffixed (`..._memberNN`).

    PYTHONPATH=. python scripts/backfill_ensemble_members.py
Output: data/historical/ensemble_members.parquet
  (station, date, model, member, tmax_f)   member 0 = control
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
MODELS = {"ecmwf_ens": "ecmwf_ifs025", "gefs": "gfs025"}
STATIONS = ["KATL", "KAUS", "KBOS", "KDCA", "KDEN", "KDFW", "KIAH", "KLAS",
            "KLAX", "KLGA", "KMIA", "KMSP", "KMSY", "KOKC", "KORD", "KPHL",
            "KPHX", "KSAT", "KSEA", "KSFO"]
START, END = "2026-04-01", "2026-06-24"
VAR = "temperature_2m_previous_day1"
OUT = "data/historical/ensemble_members.parquet"


def parse_members(resp: dict, station: str, model: str) -> pd.DataFrame:
    """Long-form (station, date, model, member, tmax_f); member 0 = control."""
    hourly = resp.get("hourly", {})
    times = hourly.get("time", [])
    rows: list[dict] = []
    for col, values in hourly.items():
        if col == "time" or not col.startswith(VAR):
            continue
        suffix = col[len(VAR):]
        member = int(suffix.replace("_member", "")) if suffix else 0
        for date, tmax in daily_max_by_date(times, values).items():
            rows.append({"station": station, "date": date, "model": model,
                         "member": member, "tmax_f": tmax})
    return pd.DataFrame(rows, columns=["station", "date", "model", "member", "tmax_f"])


def fetch(station: str, model_slug: str, max_retries: int = 4) -> dict:
    st = get_station(station)
    params = {"latitude": st.lat, "longitude": st.lon, "timezone": st.timezone,
              "start_date": START, "end_date": END, "hourly": VAR,
              "models": model_slug, "temperature_unit": "fahrenheit"}
    for attempt in range(max_retries):
        try:
            r = httpx.get(URL, params=params, timeout=120.0)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            wait = 2 ** attempt
            logger.warning("%s/%s attempt %d failed (%s); retry in %ds",
                           station, model_slug, attempt + 1, e, wait)
            time.sleep(wait)
    raise RuntimeError(f"{station}/{model_slug}: all retries failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frames = []
    for station in STATIONS:
        for model, slug in MODELS.items():
            got = parse_members(fetch(station, slug), station, model)
            logger.info("%s %s: %d member-days", station, model, len(got))
            frames.append(got)
            time.sleep(1.0)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT)
    logger.info("wrote %d rows -> %s", len(out), OUT)


if __name__ == "__main__":
    main()
