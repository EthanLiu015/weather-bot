import numpy as np
import pandas as pd
import xarray as xr

from config.stations import STATION_REGISTRY

STATION_META = {
    icao: {
        "lat": s.lat,
        "lon": s.lon,
        "elevation_m": s.elevation_m,
        "uhi_index": s.uhi_index,
        "coastal_distance_km": s.coastal_distance_km,
    }
    for icao, s in STATION_REGISTRY.items()
}

LAPSE_RATE_C_PER_M = 0.0065


def bilinear_interp(ds: xr.Dataset, lat: float, lon: float, var: str) -> float:
    lon_360 = lon % 360
    try:
        val = float(ds[var].sel(latitude=lat, longitude=lon_360, method="nearest").values)
        return val
    except Exception:
        return float("nan")


def elevation_correct_temp(
    temp_k: float,
    grid_elevation_m: float,
    station_elevation_m: float,
) -> float:
    delta_elev = station_elevation_m - grid_elevation_m
    correction_k = -LAPSE_RATE_C_PER_M * delta_elev
    return temp_k + correction_k


def apply_uhi(temp_f: float, station: str, hour_local: int) -> float:
    meta = STATION_META.get(station, {})
    uhi = meta.get("uhi_index", 0.0)
    # UHI stronger at night (hours 22-06 local), minimal at midday
    if 6 <= hour_local <= 18:
        scale = 0.3
    else:
        scale = 1.0
    return temp_f + uhi * scale


def downscale_gefs_to_station(
    gefs_data: dict,
    station: str,
    grid_elevation_m: float = 0.0,
) -> pd.DataFrame:
    meta = STATION_META.get(station, {})
    station_elev = meta.get("elevation_m", 0.0)
    rows = []
    for member_id, df in gefs_data.items():
        for _, row in df.iterrows():
            temp_k = row.get("t2m", float("nan"))
            if not np.isnan(temp_k):
                temp_k_corrected = elevation_correct_temp(temp_k, grid_elevation_m, station_elev)
                temp_f = (temp_k_corrected - 273.15) * 9 / 5 + 32
            else:
                temp_f = float("nan")
            rows.append({
                "member": member_id,
                "lead_hour": row.get("lead_hour"),
                "temp_f": temp_f,
                "u10": row.get("u10", float("nan")),
                "v10": row.get("v10", float("nan")),
                "tcc": row.get("tcc", float("nan")),
                "tp": row.get("tp", float("nan")),
                "sp": row.get("sp", float("nan")),
            })
    return pd.DataFrame(rows)
