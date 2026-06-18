import datetime as dt_mod

import numpy as np
import pandas as pd

from config.stations import STATION_REGISTRY, ALL_ICAO
from processing.climatology import climo_tmax_normal

STATION_META = {
    icao: {
        "elevation_delta_m": s.elevation_m,
        "uhi_index": s.uhi_index,
        "coastal_distance_km": s.coastal_distance_km,
        "onshore_bearing_deg": s.onshore_bearing_deg,
    }
    for icao, s in STATION_REGISTRY.items()
}
ALL_STATIONS = ALL_ICAO


def _cyclical(val: float, period: float) -> tuple[float, float]:
    return np.sin(2 * np.pi * val / period), np.cos(2 * np.pi * val / period)


def _cloud_fractions(tcc: float) -> tuple[float, float, float, float]:
    if np.isnan(tcc):
        return float("nan"), float("nan"), float("nan"), float("nan")
    total = min(max(float(tcc) / 100.0, 0.0), 1.0)
    return total, total * 0.5, total * 0.3, total * 0.2


def _safe(vals: list[float], fn):
    clean = [v for v in vals if not np.isnan(v)]
    if not clean:
        return float("nan")
    try:
        return float(fn(clean))
    except Exception:
        return float("nan")


def _quantile_skewness(p10: float, p50: float, p90: float) -> float:
    """Bowley-style quantile skewness. ~0 for symmetric, robust to outlier members."""
    if np.isnan(p10) or np.isnan(p50) or np.isnan(p90) or (p90 - p10) == 0:
        return 0.0
    return float((p90 + p10 - 2 * p50) / (p90 - p10))


def _quantile_kurtosis(p10: float, p25: float, p75: float, p90: float) -> float:
    """Tail-heaviness ratio (p90-p10)/(p75-p25). ~2.91 for a normal distribution,
    higher = heavier tails. Robust to outlier members."""
    if np.isnan(p10) or np.isnan(p25) or np.isnan(p75) or np.isnan(p90) or (p75 - p25) == 0:
        return 0.0
    return float((p90 - p10) / (p75 - p25))


def _aggregate_members(member_list: list[dict]) -> dict[str, float]:
    """Compute full distribution statistics across all 31 GEFS members."""
    temps   = [m["temp_f"]     for m in member_list if not np.isnan(m.get("temp_f", float("nan")))]
    dpts    = [m["dewpoint_f"] for m in member_list if not np.isnan(m.get("dewpoint_f", float("nan")))]
    winds   = [m["wind_speed"] for m in member_list if not np.isnan(m.get("wind_speed", float("nan")))]
    dir_sin = [m["wind_dir_sin"] for m in member_list if not np.isnan(m.get("wind_dir_sin", float("nan")))]
    dir_cos = [m["wind_dir_cos"] for m in member_list if not np.isnan(m.get("wind_dir_cos", float("nan")))]
    tccs    = [m["tcc"]        for m in member_list if not np.isnan(m.get("tcc", float("nan")))]

    def pct(arr, q):
        return float(np.percentile(arr, q)) if arr else float("nan")

    n = len(temps)
    p10, p25, p50, p75, p90 = pct(temps, 10), pct(temps, 25), pct(temps, 50), pct(temps, 75), pct(temps, 90)

    return {
        # Central tendency
        "gefs_tmax_mean":          _safe(temps, np.mean),
        # Spread
        "gefs_tmax_std":           _safe(temps, np.std) if n > 1 else float("nan"),
        "gefs_tmax_range":         (max(temps) - min(temps)) if n > 1 else float("nan"),
        "gefs_tmax_iqr":           p75 - p25 if n > 1 else float("nan"),
        # Quantiles
        "gefs_tmax_p10":           p10,
        "gefs_tmax_p25":           p25,
        "gefs_tmax_p75":           p75,
        "gefs_tmax_p90":           p90,
        # Distribution shape (quantile-based, robust to outlier members)
        "gefs_ensemble_skewness":  _quantile_skewness(p10, p50, p90) if n > 2 else 0.0,
        "gefs_ensemble_kurtosis":  _quantile_kurtosis(p10, p25, p75, p90) if n > 3 else 0.0,
        # Atmospheric
        "surface_wind_speed":      _safe(winds, np.mean),
        "wind_dir_sin":            _safe(dir_sin, np.mean),
        "wind_dir_cos":            _safe(dir_cos, np.mean),
        "surface_dew_point_depression": (
            _safe(temps, np.mean) - _safe(dpts, np.mean)
            if temps and dpts else float("nan")
        ),
        "cloud_cover_raw":         _safe(tccs, np.mean),
    }


_MAX_LAG_STALENESS_DAYS = 3


def _filter_and_sort_asos(hist: pd.Series, ref_ts: pd.Timestamp) -> pd.Series:
    if isinstance(hist.index, pd.DatetimeIndex):
        hist = hist[hist.index <= ref_ts]
    return hist.sort_index()


_ROLLING_WINDOW_DAYS = 7


def _extract_lags(
    vals: np.ndarray,
    index: pd.Index,
    ref_ts: pd.Timestamp,
) -> tuple[float, float, float]:
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    if isinstance(index, pd.DatetimeIndex) and len(index) > 0:
        most_recent = index[-1]
        staleness = (ref_ts - most_recent).days
        if staleness > _MAX_LAG_STALENESS_DAYS:
            return float("nan"), float("nan"), float("nan")
    lag1 = float(vals[-1]) if len(vals) >= 1 else float("nan")
    lag2 = float(vals[-2]) if len(vals) >= 2 else float("nan")
    lag3 = float(vals[-3]) if len(vals) >= 3 else float("nan")
    return lag1, lag2, lag3


def _extract_rolling_stats(
    vals: np.ndarray,
    index: pd.Index,
    ref_ts: pd.Timestamp,
) -> tuple[float, float]:
    if len(vals) < 2:
        return float("nan"), float("nan")
    if isinstance(index, pd.DatetimeIndex) and len(index) > 0:
        most_recent = index[-1]
        staleness = (ref_ts - most_recent).days
        if staleness > _MAX_LAG_STALENESS_DAYS:
            return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _onshore_wind_component(
    wind_dir_sin: float,
    wind_dir_cos: float,
    onshore_bearing_deg: float | None,
) -> float:
    """+1 when wind blows directly from the adjacent water (onshore), -1 when
    directly offshore, 0 for inland stations or missing wind data."""
    if onshore_bearing_deg is None or np.isnan(wind_dir_sin) or np.isnan(wind_dir_cos):
        return 0.0
    bearing_rad = np.radians(onshore_bearing_deg)
    return float(wind_dir_sin * np.sin(bearing_rad) + wind_dir_cos * np.cos(bearing_rad))


def build_feature_matrix(
    gefs_data: dict,
    ecmwf_data: dict,
    asos_history: pd.DataFrame,
    nbm_data: dict | None = None,
    station_meta: dict | None = None,
    reference_date: dt_mod.date | None = None,
) -> pd.DataFrame:
    rows = []
    if station_meta is None:
        station_meta = STATION_META

    ref_date = reference_date or dt_mod.date.today()
    month = ref_date.month
    doy = ref_date.timetuple().tm_yday
    doy_sin, doy_cos = _cyclical(doy, 365)

    for station in ALL_STATIONS:
        station_gefs = gefs_data.get(station, {})
        if not station_gefs:
            continue

        lead_hours = sorted(lh for lh, members in station_gefs.items() if members)
        if not lead_hours:
            continue

        for lead_hour in lead_hours:
            member_list = station_gefs[lead_hour]
            if not member_list:
                continue

            gefs = _aggregate_members(member_list)

            # Cloud cover from raw TCC
            tcc = gefs["cloud_cover_raw"]
            cloud_total = float("nan") if np.isnan(tcc) else min(max(float(tcc) / 100.0, 0.0), 1.0)

            # ECMWF — per-day lookup
            lead_day = max(1, round(lead_hour / 24))
            ecmwf_station = ecmwf_data.get(station, {})
            daily_ecmwf = ecmwf_station.get("daily", {})
            if daily_ecmwf and lead_day in daily_ecmwf:
                ecmwf_tmax = float(daily_ecmwf[lead_day].get("tmax", float("nan")))
                ecmwf_tmin = float(daily_ecmwf[lead_day].get("tmin", float("nan")))
            else:
                ecmwf_tmax = float(ecmwf_station.get("tmax_forecast", float("nan")))
                ecmwf_tmin = float(ecmwf_station.get("tmin_forecast", float("nan")))

            gefs_mean = gefs["gefs_tmax_mean"]
            ecmwf_gefs_delta = (
                abs(ecmwf_tmax - gefs_mean)
                if not (np.isnan(ecmwf_tmax) or np.isnan(gefs_mean))
                else float("nan")
            )

            # NBM features — match closest forecast hour
            nbm_row: dict[str, float] = {
                "nbm_t10": float("nan"), "nbm_t25": float("nan"),
                "nbm_t50": float("nan"), "nbm_t75": float("nan"),
                "nbm_t90": float("nan"),
                "nbm_tmin": float("nan"), "nbm_pop12": float("nan"),
                "nbm_spread": float("nan"), "nbm_gefs_delta": float("nan"),
            }
            if nbm_data and station in nbm_data:
                nbm_station = nbm_data[station]
                # Find closest available NBM lead hour
                nbm_hours = list(nbm_station.keys())
                if nbm_hours:
                    closest = min(nbm_hours, key=lambda h: abs(h - lead_hour))
                    nbm_slot = nbm_station[closest]
                    t50 = nbm_slot.get("t50", float("nan"))
                    nbm_row = {
                        "nbm_t10":       nbm_slot.get("t10", float("nan")),
                        "nbm_t25":       nbm_slot.get("t25", float("nan")),
                        "nbm_t50":       t50,
                        "nbm_t75":       nbm_slot.get("t75", float("nan")),
                        "nbm_t90":       nbm_slot.get("t90", float("nan")),
                        "nbm_tmin":      nbm_slot.get("tmin", float("nan")),
                        "nbm_pop12":     nbm_slot.get("pop12", float("nan")),
                        "nbm_spread":    nbm_slot.get("spread", float("nan")),
                        "nbm_gefs_delta": (
                            abs(t50 - gefs_mean)
                            if not (np.isnan(t50) or np.isnan(gefs_mean))
                            else float("nan")
                        ),
                    }

            # Lag residuals + rolling stats from ASOS history
            lag1 = lag2 = lag3 = float("nan")
            roll_mean = roll_std = float("nan")
            if asos_history is not None and not asos_history.empty:
                try:
                    ref_ts = pd.Timestamp(ref_date)
                    if isinstance(asos_history.index, pd.MultiIndex):
                        if station in asos_history.index.get_level_values(0):
                            hist = asos_history.loc[station]
                            hist = _filter_and_sort_asos(hist, ref_ts)
                            vals = hist.dropna().values[-_ROLLING_WINDOW_DAYS:]
                            lag1, lag2, lag3 = _extract_lags(vals, hist.dropna().index, ref_ts)
                            roll_mean, roll_std = _extract_rolling_stats(vals, hist.dropna().index, ref_ts)
                    elif station in asos_history.columns:
                        hist = asos_history[station]
                        hist = _filter_and_sort_asos(hist, ref_ts)
                        vals = hist.dropna().values[-_ROLLING_WINDOW_DAYS:]
                        lag1, lag2, lag3 = _extract_lags(vals, hist.dropna().index, ref_ts)
                        roll_mean, roll_std = _extract_rolling_stats(vals, hist.dropna().index, ref_ts)
                except Exception:
                    pass

            meta = station_meta.get(station, {})
            lead_time_sqrt = float(np.sqrt(lead_hour))

            climo_normal = climo_tmax_normal(station, month)
            climo_anomaly = (
                gefs_mean - climo_normal
                if not (np.isnan(gefs_mean) or np.isnan(climo_normal))
                else float("nan")
            )

            onshore_wind_component = _onshore_wind_component(
                gefs["wind_dir_sin"], gefs["wind_dir_cos"], meta.get("onshore_bearing_deg")
            )

            row: dict = {
                "station":   station,
                "lead_hour": lead_hour,
                # GEFS ensemble distribution
                "gefs_tmax_mean":         gefs["gefs_tmax_mean"],
                "gefs_tmax_std":          gefs["gefs_tmax_std"],
                "gefs_tmax_range":        gefs["gefs_tmax_range"],
                "gefs_tmax_iqr":          gefs["gefs_tmax_iqr"],
                "gefs_tmax_p10":          gefs["gefs_tmax_p10"],
                "gefs_tmax_p25":          gefs["gefs_tmax_p25"],
                "gefs_tmax_p75":          gefs["gefs_tmax_p75"],
                "gefs_tmax_p90":          gefs["gefs_tmax_p90"],
                "gefs_ensemble_skewness": gefs["gefs_ensemble_skewness"],
                "gefs_ensemble_kurtosis": gefs["gefs_ensemble_kurtosis"],
                "gefs_tmax_climo_anomaly": climo_anomaly,
                # ECMWF
                "ecmwf_tmax":             ecmwf_tmax,
                "ecmwf_tmin":             ecmwf_tmin,
                "ecmwf_gefs_tmax_delta":  ecmwf_gefs_delta,
                # NBM blend
                **nbm_row,
                # Cloud
                "cloud_cover_total":      cloud_total,
                # Wind
                "surface_wind_speed":     gefs["surface_wind_speed"],
                "wind_dir_sin":           gefs["wind_dir_sin"],
                "wind_dir_cos":           gefs["wind_dir_cos"],
                "onshore_wind_component": onshore_wind_component,
                # Moisture
                "surface_dew_point_depression": gefs["surface_dew_point_depression"],
                # Temporal
                "lead_time_sqrt":         lead_time_sqrt,
                "day_of_year_sin":        doy_sin,
                "day_of_year_cos":        doy_cos,
                # Station physical meta
                "elevation_delta_m":      meta.get("elevation_delta_m", 0.0),
                "uhi_index":              meta.get("uhi_index", 0.0),
                "coastal_distance_km":    meta.get("coastal_distance_km", 0.0),
                # Observed residuals
                "obs_minus_model_lag1":   lag1,
                "obs_minus_model_lag2":   lag2,
                "obs_minus_model_lag3":   lag3,
                "obs_minus_model_roll_mean": roll_mean,
                "obs_minus_model_roll_std":  roll_std,
            }

            rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns() -> list[str]:
    return [
        # GEFS distribution
        "gefs_tmax_mean", "gefs_tmax_std",
        "gefs_tmax_range", "gefs_tmax_iqr",
        "gefs_tmax_p10", "gefs_tmax_p25", "gefs_tmax_p75", "gefs_tmax_p90",
        "gefs_ensemble_skewness", "gefs_ensemble_kurtosis",
        "gefs_tmax_climo_anomaly",
        # ECMWF
        "ecmwf_tmax", "ecmwf_tmin", "ecmwf_gefs_tmax_delta",
        # NBM
        "nbm_t10", "nbm_t25", "nbm_t50", "nbm_t75", "nbm_t90",
        "nbm_tmin", "nbm_pop12", "nbm_spread", "nbm_gefs_delta",
        # Cloud
        "cloud_cover_total",
        # Wind
        "surface_wind_speed", "wind_dir_sin", "wind_dir_cos", "onshore_wind_component",
        # Moisture
        "surface_dew_point_depression",
        # Temporal
        "lead_time_sqrt", "day_of_year_sin", "day_of_year_cos",
        # Physical meta
        "elevation_delta_m", "uhi_index", "coastal_distance_km",
        # Residual lags + rolling stats
        "obs_minus_model_lag1", "obs_minus_model_lag2", "obs_minus_model_lag3",
        "obs_minus_model_roll_mean", "obs_minus_model_roll_std",
    ]
