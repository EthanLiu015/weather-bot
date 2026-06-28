"""Official NWS daily-max ingestion via the IEM ASOS daily summary.

Kalshi temperature markets settle on the National Weather Service's
"Climatological Report (Daily)" max at a specific station. Our historical
training target used to be the max of hourly METAR temps, which underestimates
the true daily peak and disagreed with Kalshi settlements ~12% of the time —
fatal on 2°-wide brackets. The Iowa Environmental Mesonet (IEM) ASOS daily
summary exposes the official `max_temp_f`; verified to agree with Kalshi
settlements 100% across stations when keyed to the correct settlement station.

`SETTLEMENT_STATION` maps our ICAO station to the IEM (network, station_id) that
Kalshi actually settles on — own airport for 18 cities, but Chicago→Midway and
New York→Central Park (Kalshi does NOT use O'Hare / LaGuardia).
"""

import io
import logging
from datetime import date

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

IEM_DAILY_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"

# icao -> (iem_network, iem_station_id). 18 settle on their own airport; Chicago
# and New York settle elsewhere (verified against Kalshi rules_primary +
# settlements, 2026-06-28).
SETTLEMENT_STATION: dict[str, tuple[str, str]] = {
    "KLGA": ("NY_ASOS", "NYC"),   # New York — Central Park (not LaGuardia)
    "KORD": ("IL_ASOS", "MDW"),   # Chicago — Midway (not O'Hare)
    "KLAX": ("CA_ASOS", "LAX"),
    "KMIA": ("FL_ASOS", "MIA"),
    "KIAH": ("TX_ASOS", "IAH"),
    "KPHL": ("PA_ASOS", "PHL"),
    "KATL": ("GA_ASOS", "ATL"),
    "KAUS": ("TX_ASOS", "AUS"),
    "KDEN": ("CO_ASOS", "DEN"),
    "KPHX": ("AZ_ASOS", "PHX"),
    "KSFO": ("CA_ASOS", "SFO"),
    "KSEA": ("WA_ASOS", "SEA"),
    "KBOS": ("MA_ASOS", "BOS"),
    "KDFW": ("TX_ASOS", "DFW"),
    "KDCA": ("VA_ASOS", "DCA"),
    "KLAS": ("NV_ASOS", "LAS"),
    "KMSP": ("MN_ASOS", "MSP"),
    "KOKC": ("OK_ASOS", "OKC"),
    "KSAT": ("TX_ASOS", "SAT"),
    "KMSY": ("LA_ASOS", "MSY"),
}


def _parse_iem_daily(csv_text: str) -> dict[str, float]:
    """Parse IEM daily-summary CSV into {day (YYYY-MM-DD) -> max_temp_f}.

    Rows whose max_temp_f is missing ("None"/empty) are skipped.
    """
    if not csv_text.strip():
        return {}
    df = pd.read_csv(io.StringIO(csv_text))
    if "day" not in df.columns or "max_temp_f" not in df.columns:
        return {}
    df["max_temp_f"] = pd.to_numeric(df["max_temp_f"], errors="coerce")
    df = df.dropna(subset=["max_temp_f"])
    return {str(d): float(v) for d, v in zip(df["day"], df["max_temp_f"])}


def fetch_official_daily_tmax(
    network: str, station_id: str, start: date, end: date, timeout: float = 60.0
) -> dict[str, float]:
    """Fetch official daily max temps (°F) for one station over [start, end]."""
    params = {
        "network": network,
        "stations": station_id,
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end.year, "month2": end.month, "day2": end.day,
        "format": "comma",
    }
    resp = httpx.get(IEM_DAILY_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return _parse_iem_daily(resp.text)


def fetch_official_daily_tmax_for_icao(
    icao: str, start: date, end: date
) -> dict[str, float]:
    """Official daily max for one of our ICAO stations, routed to the IEM
    settlement station Kalshi uses (e.g. KORD -> Midway)."""
    if icao not in SETTLEMENT_STATION:
        raise KeyError(f"No settlement station mapped for {icao}")
    network, station_id = SETTLEMENT_STATION[icao]
    return fetch_official_daily_tmax(network, station_id, start, end)


def official_daily_tmax_series(icao: str, start: date, end: date) -> pd.Series:
    """Official daily max as a date-indexed Series (°F), drop-in replacement for
    the old hourly-METAR `build_asos_daily_tmax`. Empty Series if none."""
    d = fetch_official_daily_tmax_for_icao(icao, start, end)
    if not d:
        return pd.Series(dtype=float)
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()
