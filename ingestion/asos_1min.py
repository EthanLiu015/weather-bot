"""1-minute ASOS temperature fetcher (Iowa Environmental Mesonet).

The freshest running-max signal we can trade on. The daily high is set in a short
window; 1-minute obs let the model know the running max precisely and in near-real
time, ahead of a lagging retail book. Existing ingestion tops out at hourly METAR
(ingestion/asos.py) and daily max (ingestion/nws_daily.py) — this is the only
sub-hourly source.

Endpoint (verified): the IEM ASOS 1-minute service. Temperatures come as whole °F
with 'M' for missing minutes (common overnight; afternoon — when the max occurs —
is essentially complete).

    fetch_1min("KMSP", start, end) -> DataFrame[valid_utc, tmpf]
    running_max(obs)              -> DataFrame[valid_utc, tmpf, run_max]

Settlement caveat: Kalshi settles on the OFFICIAL NWS daily max for a specific
station (e.g. NYC = Central Park). Map ICAO → the correct settlement station before
fetching; icao_to_1min_code only strips the leading 'K'.
"""
from __future__ import annotations

import io
import logging
from datetime import datetime

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

ONEMIN_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py"
_MISSING = {"M", "", "None", "NaN"}


def icao_to_1min_code(station: str) -> str:
    """IEM's 1-minute service keys US stations by the 3-char FAA id (MSP), so
    strip a leading 'K' from a 4-char ICAO id. Non-K / 3-char ids pass through."""
    s = station.strip().upper()
    if len(s) == 4 and s.startswith("K"):
        return s[1:]
    return s


def parse_1min_csv(text: str) -> pd.DataFrame:
    """Parse the IEM 1-minute CSV into clean numeric obs.

    Returns columns [valid_utc (datetime64), tmpf (float)] with missing-temp
    minutes ('M') dropped. An empty body yields an empty, correctly-typed frame.
    """
    raw = pd.read_csv(io.StringIO(text))
    empty = pd.DataFrame({"valid_utc": pd.to_datetime([]), "tmpf": pd.Series([], dtype=float)})
    if raw.empty or "valid(UTC)" not in raw.columns:
        return empty
    tmpf = pd.to_numeric(raw["tmpf"].astype(str).where(~raw["tmpf"].astype(str).isin(_MISSING)),
                         errors="coerce")
    out = pd.DataFrame({
        "valid_utc": pd.to_datetime(raw["valid(UTC)"], errors="coerce"),
        "tmpf": tmpf,
    }).dropna(subset=["valid_utc", "tmpf"]).reset_index(drop=True)
    return out


def running_max(obs: pd.DataFrame) -> pd.DataFrame:
    """Add a cumulative-max `run_max` column over time-ordered obs — the running
    daily high as it would be known minute by minute."""
    if obs.empty:
        return obs.assign(run_max=pd.Series([], dtype=float))
    ordered = obs.sort_values("valid_utc").reset_index(drop=True)
    ordered["run_max"] = ordered["tmpf"].cummax()
    return ordered


def fetch_1min(station: str, start: datetime, end: datetime, timeout: float = 90.0) -> pd.DataFrame:
    """Fetch 1-minute tmpf for `station` (ICAO) over [start, end] UTC.

    Thin network wrapper around parse_1min_csv; kept sync for simple batch backfills.
    """
    code = icao_to_1min_code(station)
    params = {
        "station": code,
        "vars": "tmpf",
        "tz": "UTC",
        "year1": start.year, "month1": start.month, "day1": start.day,
        "hour1": start.hour, "minute1": start.minute,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "hour2": end.hour, "minute2": end.minute,
        "sample": "1min",
        "what": "download",
        "delim": "comma",
        "gis": "no",
    }
    resp = httpx.get(ONEMIN_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    df = parse_1min_csv(resp.text)
    logger.info("1-min %s %s→%s: %d obs", code, start, end, len(df))
    return df
