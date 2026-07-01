"""Open-Meteo Previous Runs multi-model daily-high fetcher (roadmap 3).

Pulls lead-specific, leakage-free archived forecasts for five models the current
feature set does not have — crucially the AI models AIFS and GraphCast — so we can
measure multi-model *disagreement* as a new source of signal.

The Previous Runs API (https://previous-runs-api.open-meteo.com/v1/forecast)
exposes `temperature_2m_previous_dayN` = the forecast valid at each hour but issued
N*24h earlier. We take the daily maximum per LOCAL date to get the N-day-lead
daily-high forecast, aligned to how Kalshi settles (local calendar day). Archive
starts Jan 2024, so the Apr–Jun 2026 eval window is fully covered.

Pure parsing (`daily_max_by_date`, `parse_previous_runs`) is unit-tested; the
network call is a thin wrapper.

    PYTHONPATH=. python -m ingestion.openmeteo   # smoke-fetch one station
"""
from __future__ import annotations

import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# short name -> Open-Meteo model slug. AIFS + GraphCast are the new (AI) signal;
# gfs/icon/ecmwf give physics baselines to disagree against.
MODEL_SLUGS: dict[str, str] = {
    "aifs": "ecmwf_aifs025_single",
    "graphcast": "gfs_graphcast025",
    "gfs": "gfs_seamless",
    "icon": "icon_seamless",
    "ecmwf": "ecmwf_ifs025",
}

DEFAULT_LEADS = (1, 2, 3)  # previous_dayN → 24/48/72h; short leads = decision zone


def daily_max_by_date(times: list[str], values: list[float | None]) -> dict[str, float]:
    """Max value per local calendar date from parallel hourly (time, value) lists.
    None values are skipped; a date with only None yields no entry."""
    by_date: dict[str, float] = {}
    for t, v in zip(times, values):
        if v is None:
            continue
        date = t[:10]
        cur = by_date.get(date)
        if cur is None or v > cur:
            by_date[date] = float(v)
    return by_date


def parse_previous_runs(
    resp: dict, station: str, model: str, leads=DEFAULT_LEADS
) -> pd.DataFrame:
    """Long-form (station, date, lead_hour, model, tmax_c) from one API response.

    Reads `temperature_2m_previous_day{lead}_{slug}` for each lead; a missing
    column (model/lead not archived) contributes no rows rather than erroring."""
    hourly = resp.get("hourly", {})
    times = hourly.get("time", [])
    slug = MODEL_SLUGS[model]
    rows: list[dict] = []
    for lead in leads:
        col = f"temperature_2m_previous_day{lead}_{slug}"
        if col not in hourly:
            continue
        for date, tmax in daily_max_by_date(times, hourly[col]).items():
            rows.append({
                "station": station,
                "date": date,
                "lead_hour": lead * 24,
                "model": model,
                "tmax_c": tmax,
            })
    return pd.DataFrame(rows, columns=["station", "date", "lead_hour", "model", "tmax_c"])


def fetch_station(
    station: str, lat: float, lon: float, tz: str,
    start_date: str, end_date: str, leads=DEFAULT_LEADS,
    models=tuple(MODEL_SLUGS), timeout: float = 60.0, max_retries: int = 4,
) -> pd.DataFrame:
    """Fetch all models for one station/date-range in ONE request. Requesting
    several models makes Open-Meteo suffix each column with its slug (a single
    model returns unsuffixed columns), which is what `parse_previous_runs`
    expects. Rate-limited + retried on 429/5xx."""
    import httpx

    hourly_vars = ",".join(f"temperature_2m_previous_day{lead}" for lead in leads)
    params = {
        "latitude": lat, "longitude": lon, "timezone": tz,
        "start_date": start_date, "end_date": end_date,
        "hourly": hourly_vars,
        "models": ",".join(MODEL_SLUGS[m] for m in models),
        "temperature_unit": "celsius",
    }
    for attempt in range(max_retries):
        try:
            r = httpx.get(BASE_URL, params=params, timeout=timeout)
            if r.status_code == 429 or r.status_code >= 500:
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            r.raise_for_status()
            resp = r.json()
            frames = [parse_previous_runs(resp, station, m, leads) for m in models]
            frames = [f for f in frames if not f.empty]  # drop models with no archive
            return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
                columns=["station", "date", "lead_hour", "model", "tmax_c"])
        except httpx.HTTPError as e:
            wait = 2 ** attempt
            logger.warning("%s attempt %d failed (%s); retry in %ds",
                           station, attempt + 1, e, wait)
            time.sleep(wait)
    logger.error("%s: giving up after %d attempts", station, max_retries)
    return pd.DataFrame(columns=["station", "date", "lead_hour", "model", "tmax_c"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from config.stations import get_station
    st = get_station("KLAX")
    df = fetch_station("KLAX", st.lat, st.lon, st.timezone, "2026-05-01", "2026-05-05")
    print(df.pivot_table(index=["date", "lead_hour"], columns="model", values="tmax_c").round(1))


if __name__ == "__main__":
    main()
