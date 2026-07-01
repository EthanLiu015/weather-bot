"""Backfill Open-Meteo multi-model daily-high forecasts for the eval window.

One request per station covers all models, all leads, the whole date range, so
the full backfill is ~20 cheap requests. Output feeds the multi-model
disagreement study (roadmap 3).

    PYTHONPATH=. python scripts/backfill_openmeteo.py
    PYTHONPATH=. python scripts/backfill_openmeteo.py --start 2026-04-01 --end 2026-06-01
Output: data/historical/openmeteo_multimodel.parquet
    (station, date, lead_hour, model, tmax_c, tmax_f)
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd

from config.stations import STATION_REGISTRY
from ingestion.openmeteo import fetch_station

logger = logging.getLogger(__name__)

OUT_PATH = "data/historical/openmeteo_multimodel.parquet"


def backfill(start: str, end: str, pause: float = 1.0) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for i, (icao, st) in enumerate(STATION_REGISTRY.items(), 1):
        df = fetch_station(icao, st.lat, st.lon, st.timezone, start, end)
        logger.info("[%d/%d] %s: %d rows (%s models)", i, len(STATION_REGISTRY),
                    icao, len(df), df["model"].nunique() if not df.empty else 0)
        if not df.empty:
            frames.append(df)
        time.sleep(pause)
    out = pd.concat(frames, ignore_index=True)
    out["tmax_f"] = out["tmax_c"] * 9.0 / 5.0 + 32.0
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-04-01")
    ap.add_argument("--end", default="2026-06-01")
    args = ap.parse_args()

    df = backfill(args.start, args.end)
    df.to_parquet(OUT_PATH, index=False)
    logger.info("Saved %d rows to %s", len(df), OUT_PATH)
    print(df.groupby(["model", "lead_hour"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
