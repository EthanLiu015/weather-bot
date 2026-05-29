from dataclasses import dataclass


@dataclass(frozen=True)
class StationMeta:
    icao: str
    city: str
    lat: float
    lon: float
    elevation_m: float
    uhi_index: float
    coastal_distance_km: float
    timezone: str


# All stations with active Kalshi temperature markets (high and/or low).
# Source: Kalshi series API — verified 2026-05-29.
STATION_REGISTRY: dict[str, StationMeta] = {
    s.icao: s for s in [
        StationMeta("KLGA",  "New York City",    40.7769, -73.8740,    3,  2.5,    5, "America/New_York"),
        StationMeta("KORD",  "Chicago",          41.9786, -87.9047,  203,  1.8, 1500, "America/Chicago"),
        StationMeta("KLAX",  "Los Angeles",      33.9425,-118.4081,   38,  2.1,    5, "America/Los_Angeles"),
        StationMeta("KMIA",  "Miami",            25.7959, -80.2870,    3,  1.9,    3, "America/New_York"),
        StationMeta("KIAH",  "Houston",          29.9844, -95.3414,   30,  2.2,   80, "America/Chicago"),
        StationMeta("KPHL",  "Philadelphia",     39.8744, -75.2424,   11,  2.0,   50, "America/New_York"),
        StationMeta("KATL",  "Atlanta",          33.6367, -84.4281,  313,  2.3,  400, "America/New_York"),
        StationMeta("KAUS",  "Austin",           30.1945, -97.6699,  149,  1.7,  300, "America/Chicago"),
        StationMeta("KDEN",  "Denver",           39.8561,-104.6737, 1655,  1.5, 1800, "America/Denver"),
        StationMeta("KMSY",  "New Orleans",      29.9934, -90.2580,    1,  1.8,    5, "America/Chicago"),
        StationMeta("KPHX",  "Phoenix",          33.4373,-112.0078,  337,  3.5,  600, "America/Phoenix"),
        StationMeta("KSFO",  "San Francisco",    37.6213,-122.3790,    4,  1.4,    1, "America/Los_Angeles"),
        StationMeta("KSEA",  "Seattle",          47.4489,-122.3094,  131,  1.6,   15, "America/Los_Angeles"),
        StationMeta("KBOS",  "Boston",           42.3630, -71.0064,    9,  2.2,    1, "America/New_York"),
        StationMeta("KDFW",  "Dallas",           32.8998, -97.0403,  182,  2.0,  500, "America/Chicago"),
        StationMeta("KDCA",  "Washington DC",    38.8521, -77.0377,    5,  2.4,   10, "America/New_York"),
        StationMeta("KLAS",  "Las Vegas",        36.0840,-115.1522,  664,  2.8,  450, "America/Los_Angeles"),
        StationMeta("KMSP",  "Minneapolis",      44.8848, -93.2223,  278,  1.7, 2000, "America/Chicago"),
        StationMeta("KOKC",  "Oklahoma City",    35.3931, -97.6008,  397,  1.5,  700, "America/Chicago"),
        StationMeta("KSAT",  "San Antonio",      29.5337, -98.4698,  242,  2.0,  250, "America/Chicago"),
    ]
}

ALL_ICAO = list(STATION_REGISTRY.keys())


def get_station(icao: str) -> StationMeta:
    if icao not in STATION_REGISTRY:
        raise KeyError(f"Station {icao!r} not in registry. Add it to config/stations.py.")
    return STATION_REGISTRY[icao]


def station_coords() -> dict[str, tuple[float, float]]:
    return {icao: (s.lat, s.lon) for icao, s in STATION_REGISTRY.items()}


def station_timezones() -> dict[str, str]:
    return {icao: s.timezone for icao, s in STATION_REGISTRY.items()}
