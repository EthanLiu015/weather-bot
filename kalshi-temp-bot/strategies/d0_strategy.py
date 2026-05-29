import json
import logging
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np

from ingestion.asos import fetch_metar, compute_running_tmax, estimate_remaining_hours, STATION_TZ

logger = logging.getLogger(__name__)

DIURNAL_DIR = Path("data/diurnal_climatology")
NEAR_CERTAIN_YES = 0.985
NEAR_CERTAIN_NO = 0.015


class D0Strategy:
    def __init__(self, shared_state, settings) -> None:
        self._state = shared_state
        self._settings = settings
        self._diurnal_cache: dict[str, dict] = {}

    async def run_cycle(self, station: str, ticker: str, threshold: float) -> None:
        if not await self.is_resolution_day(ticker):
            return

        obs_list = await fetch_metar(station, hours=24)
        if not obs_list:
            logger.warning("No METAR data for %s", station)
            return

        running_tmax = compute_running_tmax(obs_list)
        tz = STATION_TZ.get(station, "UTC")
        import zoneinfo
        now_local = datetime.now(zoneinfo.ZoneInfo(tz))
        hour_local = now_local.hour
        month = now_local.month

        remaining_hours = estimate_remaining_hours(datetime.utcnow().hour, tz)
        max_possible = self._estimate_max_possible(running_tmax, hour_local, remaining_hours, station, month)

        if not math.isnan(running_tmax) and running_tmax > threshold:
            fair_value = NEAR_CERTAIN_YES
        elif remaining_hours < 3 and max_possible < threshold:
            fair_value = NEAR_CERTAIN_NO
        else:
            fair_value = self.conditional_prob(running_tmax, threshold, hour_local, station, month)

        self._state.update_fair_b(ticker, fair_value)
        logger.info(
            "D0 %s %s: running_tmax=%.1f threshold=%.1f remaining=%dh → fair_b=%.3f",
            station, ticker, running_tmax, threshold, remaining_hours, fair_value,
        )

    def conditional_prob(
        self,
        running_tmax: float,
        threshold: float,
        hour_local: int,
        station: str,
        month: int,
    ) -> float:
        clim = self._load_diurnal(station, month)
        gap = threshold - running_tmax

        if gap <= 0:
            return NEAR_CERTAIN_YES

        # For each remaining hour, compute P(T_h - running_tmax >= gap)
        # where T_h - running_tmax ~ historical(Tmax - T_at_hour_h)
        # P(at least one hour exceeds) = 1 - prod(1 - P_h)
        prob_none_exceeds = 1.0
        for h in range(hour_local + 1, 24):
            hour_key = str(h)
            if hour_key not in clim:
                continue
            dist = clim[hour_key]
            mean_diff = dist.get("mean", 5.0)
            std_diff = dist.get("std", 3.0)
            if std_diff <= 0:
                continue
            # P(Tmax - T_at_h >= gap) = P(additional gain >= gap)
            z = (gap - mean_diff) / std_diff
            p_exceed = max(0.0, 1.0 - self._normal_cdf(z))
            prob_none_exceeds *= (1.0 - p_exceed)

        return float(np.clip(1.0 - prob_none_exceeds, 0.0, 1.0))

    def _estimate_max_possible(
        self,
        running_tmax: float,
        hour_local: int,
        remaining_hours: int,
        station: str,
        month: int,
    ) -> float:
        clim = self._load_diurnal(station, month)
        max_gain = 0.0
        for h in range(hour_local + 1, 24):
            dist = clim.get(str(h), {})
            mean = dist.get("mean", 0.0)
            std = dist.get("std", 3.0)
            max_gain = max(max_gain, mean + 2 * std)
        return running_tmax + max_gain

    def _load_diurnal(self, station: str, month: int) -> dict:
        key = f"{station}_{month}"
        if key not in self._diurnal_cache:
            path = DIURNAL_DIR / f"{station}_{month}.json"
            if path.exists():
                with open(path) as f:
                    self._diurnal_cache[key] = json.load(f)
            else:
                # Fallback: synthetic climatology
                self._diurnal_cache[key] = {
                    str(h): {"mean": max(0.0, (h - 6) * 0.5), "std": 2.5}
                    for h in range(24)
                }
        return self._diurnal_cache[key]

    @staticmethod
    def _normal_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))

    async def is_resolution_day(self, ticker: str) -> bool:
        import re
        m = re.search(r"(\d{8})", ticker)
        if not m:
            return True
        try:
            ticker_date = datetime.strptime(m.group(1), "%Y%m%d").date()
            return ticker_date == date.today()
        except ValueError:
            return False
