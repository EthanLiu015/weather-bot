import logging
import asyncio
from datetime import datetime
import httpx
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

NBM_BASE = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod"
STATION_COORDS = {
    "KORD": (41.978611, -87.904722),
    "KJFK": (40.639722, -73.778889),
    "KLAX": (33.942536, -118.408075),
}


async def fetch_latest_nbm(data_dir: str = "data/nbm") -> dict:
    now = datetime.utcnow()
    cycle_hour = (now.hour // 6) * 6
    date_str = now.strftime("%Y%m%d")
    cycle_str = f"{cycle_hour:02d}"

    base_url = f"{NBM_BASE}/blend.{date_str}/{cycle_str}/core/"

    result: dict[str, dict] = {}
    for station in STATION_COORDS:
        result[station] = {
            "tmax_blend": np.nan,
            "tmin_blend": np.nan,
            "pop12": np.nan,
            "run_date": date_str,
            "cycle": cycle_str,
        }

    probe_url = f"{base_url}blend.t{cycle_str}z.core.f024.co.grib2"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.head(probe_url, timeout=10.0)
            if resp.status_code == 200:
                logger.info("NBM run %s %sz available", date_str, cycle_str)
            else:
                logger.warning("NBM run %s %sz not yet available (HTTP %d)", date_str, cycle_str, resp.status_code)
        except Exception as exc:
            logger.warning("NBM probe failed: %s", exc)

    return result
