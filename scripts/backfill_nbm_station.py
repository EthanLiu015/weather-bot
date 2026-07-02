"""Backfill NBM station-bulletin (NBS) daytime-max forecasts from the IEM MOS archive.

For every station/day in the Kalshi eval window, keep the latest NBM run at or
before 07Z (posted ~08:30 UTC, comfortably before the 14:00 UTC decision cutoff)
and extract the daytime max `txn` + NBM's own max-temp stdev `xnd` at the
ftime 00Z of the following day.

    PYTHONPATH=. python scripts/backfill_nbm_station.py
Output: data/historical/nbm_station.parquet
  (station, date, runtime, nbm_max_f, nbm_sigma_f)
"""
from __future__ import annotations

import logging
import time

import httpx
import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"
STATIONS = ["KATL", "KAUS", "KBOS", "KDCA", "KDEN", "KDFW", "KIAH", "KLAS",
            "KLAX", "KLGA", "KMIA", "KMSP", "KMSY", "KOKC", "KORD", "KPHL",
            "KPHX", "KSAT", "KSEA", "KSFO"]
START, END = "2026-04-10", "2026-06-25"
MAX_RUN_HOUR_UTC = 7
OUT = "data/historical/nbm_station.parquet"


def extract_daytime_max(raw: pd.DataFrame) -> pd.DataFrame:
    """Latest run <= MAX_RUN_HOUR_UTC per day; txn/xnd at ftime 00Z next day."""
    df = raw.copy()
    df["runtime"] = pd.to_datetime(df["runtime"])
    df["ftime"] = pd.to_datetime(df["ftime"])
    df = df[df["runtime"].dt.hour <= MAX_RUN_HOUR_UTC]
    df["date"] = df["runtime"].dt.normalize()
    target = df["date"] + pd.Timedelta(days=1)
    df = df[(df["ftime"] == target) & df["txn"].notna()]
    df = df.sort_values("runtime").groupby(["station", "date"], as_index=False).last()
    out = df[["station", "date", "runtime", "txn", "xnd"]].rename(
        columns={"txn": "nbm_max_f", "xnd": "nbm_sigma_f"})
    return out.reset_index(drop=True)


def fetch_station(station: str, timeout: float = 120.0) -> pd.DataFrame:
    params = {"station": station, "model": "NBS",
              "sts": f"{START}T00:00Z", "ets": f"{END}T08:00Z", "format": "csv"}
    r = httpx.get(URL, params=params, timeout=timeout)
    r.raise_for_status()
    from io import StringIO
    return pd.read_csv(StringIO(r.text))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    frames = []
    for st in STATIONS:
        raw = fetch_station(st)
        got = extract_daytime_max(raw)
        logger.info("%s: %d station-days (of %d raw rows)", st, len(got), len(raw))
        frames.append(got)
        time.sleep(1.0)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT, index=False)
    logger.info("wrote %s: %d rows, %d stations, %s..%s",
                OUT, len(out), out["station"].nunique(), out["date"].min(), out["date"].max())


if __name__ == "__main__":
    main()
