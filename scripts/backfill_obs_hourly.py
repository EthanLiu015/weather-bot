"""Backfill hourly METAR temperature obs (cycle 5: morning-obs conditioning).

IEM ASOS archive, routine hourly METARs (report_type=3). Obs are keyed to the
KALSHI station label but fetched from the station Kalshi actually settles on
(cycle-4 finding): NY -> NYC (Central Park), Chicago -> MDW (Midway).
Obs at/before the 14:00 UTC decision cutoff are fair by construction.

    PYTHONPATH=. python scripts/backfill_obs_hourly.py
Output: data/historical/obs_hourly.parquet  (station, valid_utc, tmpf)
"""
from __future__ import annotations

import logging
import time
from io import StringIO

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# kalshi station label -> IEM ASOS id of the settlement station
OBS_ID = {"KAUS": "AUS", "KBOS": "BOS", "KDCA": "DCA", "KDEN": "DEN",
          "KDFW": "DFW", "KLAS": "LAS", "KLAX": "LAX", "KLGA": "NYC",
          "KMIA": "MIA", "KMSP": "MSP", "KMSY": "MSY", "KOKC": "OKC",
          "KORD": "MDW", "KPHL": "PHL", "KPHX": "PHX", "KSAT": "SAT",
          "KSEA": "SEA", "KSFO": "SFO"}
START = {"year1": 2026, "month1": 4, "day1": 1}
END = {"year2": 2026, "month2": 6, "day2": 25}
OUT = "data/historical/obs_hourly.parquet"


def fetch_station(iem_id: str, max_retries: int = 5) -> pd.DataFrame:
    params = {"station": iem_id, "data": "tmpf", "tz": "Etc/UTC",
              "format": "onlycomma", "latlon": "no", "missing": "empty",
              "trace": "empty", "report_type": 3, **START, **END}
    for attempt in range(max_retries):
        r = httpx.get(URL, params=params, timeout=120.0)
        if r.status_code == 200 and r.text.startswith("station"):
            df = pd.read_csv(StringIO(r.text))
            df = df[df["tmpf"].notna()]
            return df
        wait = 5 * 2 ** attempt
        logger.warning("%s attempt %d got %s; retry in %ds",
                       iem_id, attempt + 1, r.status_code, wait)
        time.sleep(wait)
    raise RuntimeError(f"{iem_id}: all retries failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frames = []
    for kalshi_st, iem_id in OBS_ID.items():
        df = fetch_station(iem_id)
        logger.info("%s (%s): %d obs", kalshi_st, iem_id, len(df))
        frames.append(pd.DataFrame({
            "station": kalshi_st,
            "valid_utc": pd.to_datetime(df["valid"]),
            "tmpf": df["tmpf"].astype(float),
        }))
        time.sleep(5.0)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT)
    logger.info("wrote %d rows -> %s", len(out), OUT)


if __name__ == "__main__":
    main()
