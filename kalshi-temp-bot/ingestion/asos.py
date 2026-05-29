import re
import asyncio
import logging
from datetime import datetime, timezone
import httpx
import pandas as pd

logger = logging.getLogger(__name__)

METAR_BASE = "https://aviationweather.gov/cgi-bin/data/metar.php"
STATION_TZ = {
    "KORD": "America/Chicago",
    "KJFK": "America/New_York",
    "KLAX": "America/Los_Angeles",
}


async def fetch_metar(station: str, hours: int = 24) -> list[dict]:
    url = f"{METAR_BASE}?ids={station}&hours={hours}&format=raw"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=20.0)
                resp.raise_for_status()
            lines = [ln.strip() for ln in resp.text.splitlines() if ln.strip()]
            observations = []
            for line in lines:
                if line.startswith(station):
                    parsed = parse_metar_string(line)
                    if parsed:
                        observations.append(parsed)
            return observations
        except Exception as exc:
            wait = 2**attempt
            logger.warning("METAR fetch attempt %d for %s failed: %s; retry in %ds", attempt + 1, station, exc, wait)
            await asyncio.sleep(wait)
    logger.error("All METAR fetch attempts failed for %s", station)
    return []


def parse_metar_string(raw: str) -> dict:
    parts = raw.split()
    obs: dict = {"raw": raw, "station": None, "temp_c": None, "dewpoint_c": None,
                 "temp_f": None, "dewpoint_f": None, "wind_dir": None,
                 "wind_speed_kt": None, "visibility_sm": None,
                 "sky_cover": [], "present_weather": [],
                 "observation_time": None, "fog": False, "inversion_proxy": False}

    if not parts:
        return obs

    obs["station"] = parts[0]

    # Parse observation time (DDHHMM Z)
    time_pattern = re.compile(r"^(\d{6})Z$")
    for part in parts[1:3]:
        m = time_pattern.match(part)
        if m:
            try:
                now = datetime.utcnow()
                day = int(m.group(1)[:2])
                hour = int(m.group(1)[2:4])
                minute = int(m.group(1)[4:6])
                obs["observation_time"] = now.replace(day=day, hour=hour, minute=minute,
                                                       second=0, microsecond=0,
                                                       tzinfo=timezone.utc)
            except ValueError:
                pass

    # Wind: dddssKT or dddssGssKT or VRB
    wind_pattern = re.compile(r"^(\d{3}|VRB)(\d{2,3})(G\d{2,3})?KT$")
    for part in parts:
        m = wind_pattern.match(part)
        if m:
            obs["wind_dir"] = None if m.group(1) == "VRB" else int(m.group(1))
            obs["wind_speed_kt"] = int(m.group(2))
            break

    # Visibility
    vis_pattern = re.compile(r"^(\d+(?:/\d+)?|\d+\s+\d+/\d+)SM$")
    for i, part in enumerate(parts):
        if part.endswith("SM"):
            try:
                vis_str = part.replace("SM", "")
                if "/" in vis_str:
                    num, den = vis_str.split("/")
                    obs["visibility_sm"] = float(num) / float(den)
                else:
                    obs["visibility_sm"] = float(vis_str)
            except ValueError:
                pass
            break

    # Temp / dewpoint: TT/DD or M-prefix for negative
    temp_pattern = re.compile(r"^(M?\d{2})/(M?\d{2})$")
    for part in parts:
        m = temp_pattern.match(part)
        if m:
            def parse_temp(s: str) -> float:
                return -float(s[1:]) if s.startswith("M") else float(s)
            obs["temp_c"] = parse_temp(m.group(1))
            obs["dewpoint_c"] = parse_temp(m.group(2))
            obs["temp_f"] = obs["temp_c"] * 9 / 5 + 32
            obs["dewpoint_f"] = obs["dewpoint_c"] * 9 / 5 + 32
            break

    # Sky cover layers: CLR, SKC, FEW, SCT, BKN, OVC
    sky_pattern = re.compile(r"^(CLR|SKC|FEW|SCT|BKN|OVC)(\d{3})?$")
    sky_layers = []
    for part in parts:
        m = sky_pattern.match(part)
        if m:
            layer = {"cover": m.group(1), "height_ft": int(m.group(2)) * 100 if m.group(2) else None}
            sky_layers.append(layer)
    obs["sky_cover"] = sky_layers

    # Present weather codes
    wx_codes = {"FG", "BR", "MIFG", "BCFG", "RA", "SN", "DZ", "TS", "SH", "FZ"}
    present_wx = []
    for part in parts:
        for code in wx_codes:
            if code in part:
                present_wx.append(code)
    obs["present_weather"] = list(set(present_wx))

    # Fog/mist detection
    obs["fog"] = any(code in obs["present_weather"] for code in ("FG", "BR", "MIFG", "BCFG"))

    # Inversion proxy: temp > dewpoint + 20°F AND wind < 5kt AND clear
    clear_sky = all(lyr["cover"] in ("CLR", "SKC") for lyr in sky_layers) or not sky_layers
    if (obs["temp_f"] is not None and obs["dewpoint_f"] is not None
            and obs["wind_speed_kt"] is not None):
        obs["inversion_proxy"] = (
            obs["temp_f"] > obs["dewpoint_f"] + 20
            and obs["wind_speed_kt"] < 5
            and clear_sky
        )

    return obs


def compute_running_tmax(obs_list: list[dict]) -> float:
    temps = [o["temp_f"] for o in obs_list if o.get("temp_f") is not None]
    return max(temps) if temps else float("nan")


def estimate_remaining_hours(current_hour_utc: int, station_tz: str) -> int:
    import zoneinfo
    tz = zoneinfo.ZoneInfo(station_tz)
    now_local = datetime.now(tz)
    local_hour = now_local.hour
    remaining = max(0, 23 - local_hour)
    return remaining
