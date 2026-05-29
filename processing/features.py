import datetime as dt_mod

import numpy as np
import pandas as pd
from scipy import stats

from config.stations import STATION_REGISTRY, ALL_ICAO

STATION_META = {
    icao: {
        "elevation_delta_m": s.elevation_m,
        "uhi_index": s.uhi_index,
        "coastal_distance_km": s.coastal_distance_km,
    }
    for icao, s in STATION_REGISTRY.items()
}
ALL_STATIONS = ALL_ICAO
N_REGIME_CLUSTERS = 12


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


def _aggregate_members(member_list: list[dict]) -> dict[str, float]:
    """Compute full distribution statistics across all 31 GEFS members."""
    temps   = [m["temp_f"]     for m in member_list if not np.isnan(m.get("temp_f", float("nan")))]
    dpts    = [m["dewpoint_f"] for m in member_list if not np.isnan(m.get("dewpoint_f", float("nan")))]
    winds   = [m["wind_speed"] for m in member_list if not np.isnan(m.get("wind_speed", float("nan")))]
    dir_sin = [m["wind_dir_sin"] for m in member_list if not np.isnan(m.get("wind_dir_sin", float("nan")))]
    dir_cos = [m["wind_dir_cos"] for m in member_list if not np.isnan(m.get("wind_dir_cos", float("nan")))]
    tccs    = [m["tcc"]        for m in member_list if not np.isnan(m.get("tcc", float("nan")))]
    tps     = [m["tp"]         for m in member_list if not np.isnan(m.get("tp", float("nan")))]

    def pct(arr, q):
        return float(np.percentile(arr, q)) if arr else float("nan")

    t_arr = np.array(temps) if temps else np.array([float("nan")])
    n = len(temps)

    return {
        # Central tendency
        "gefs_tmax_mean":          _safe(temps, np.mean),
        "gefs_tmax_median":        pct(temps, 50),
        # Spread
        "gefs_tmax_std":           _safe(temps, np.std) if n > 1 else float("nan"),
        "gefs_tmax_range":         (max(temps) - min(temps)) if n > 1 else float("nan"),
        "gefs_tmax_iqr":           pct(temps, 75) - pct(temps, 25) if n > 1 else float("nan"),
        # Quantiles
        "gefs_tmax_p10":           pct(temps, 10),
        "gefs_tmax_p25":           pct(temps, 25),
        "gefs_tmax_p75":           pct(temps, 75),
        "gefs_tmax_p90":           pct(temps, 90),
        # Distribution shape
        "gefs_ensemble_skewness":  float(stats.skew(t_arr)) if n > 2 else 0.0,
        "gefs_ensemble_kurtosis":  float(stats.kurtosis(t_arr)) if n > 3 else 0.0,
        # Low tail (for Tmin markets)
        "gefs_tmin_mean":          _safe(temps, np.min),
        "gefs_tmin_std":           _safe(temps, np.std) if n > 1 else float("nan"),
        # Atmospheric
        "surface_wind_speed":      _safe(winds, np.mean),
        "wind_dir_sin":            _safe(dir_sin, np.mean),
        "wind_dir_cos":            _safe(dir_cos, np.mean),
        "surface_dew_point_depression": (
            _safe(temps, np.mean) - _safe(dpts, np.mean)
            if temps and dpts else float("nan")
        ),
        "cloud_cover_raw":         _safe(tccs, np.mean),
        "convective_precip_prob":  float(np.mean([1.0 if tp > 0 else 0.0 for tp in tps])) if tps else 0.0,
        "total_precip_mm":         _safe(tps, np.mean),
    }


def build_feature_matrix(
    gefs_data: dict,
    ecmwf_data: dict,
    asos_history: pd.DataFrame,
    regime_labels: pd.Series,
    nbm_data: dict | None = None,
    station_meta: dict | None = None,
) -> pd.DataFrame:
    rows = []
    if station_meta is None:
        station_meta = STATION_META

    today = dt_mod.date.today()
    month = today.month
    doy = today.timetuple().tm_yday
    month_sin, month_cos = _cyclical(month, 12)
    doy_sin, doy_cos = _cyclical(doy, 365)

    regime_vec = [0.0] * N_REGIME_CLUSTERS
    if regime_labels is not None and len(regime_labels) > 0:
        try:
            latest_label = int(regime_labels.iloc[-1])
            if 0 <= latest_label < N_REGIME_CLUSTERS:
                regime_vec[latest_label] = 1.0
        except Exception:
            pass

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

            # Cloud fractions from raw TCC
            cloud_total, cloud_low, cloud_mid, cloud_high = _cloud_fractions(gefs["cloud_cover_raw"])

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
                "nbm_t90": float("nan"), "nbm_tmax": float("nan"),
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
                        "nbm_tmax":      nbm_slot.get("tmax", float("nan")),
                        "nbm_tmin":      nbm_slot.get("tmin", float("nan")),
                        "nbm_pop12":     nbm_slot.get("pop12", float("nan")),
                        "nbm_spread":    nbm_slot.get("spread", float("nan")),
                        "nbm_gefs_delta": (
                            abs(t50 - gefs_mean)
                            if not (np.isnan(t50) or np.isnan(gefs_mean))
                            else float("nan")
                        ),
                    }

            # Lag residuals from ASOS history
            lag1 = lag2 = lag3 = float("nan")
            if asos_history is not None and not asos_history.empty:
                try:
                    if isinstance(asos_history.index, pd.MultiIndex):
                        if station in asos_history.index.get_level_values(0):
                            hist = asos_history.loc[station]
                            vals = hist.dropna().values[-3:]
                            if len(vals) >= 1: lag1 = float(vals[-1])
                            if len(vals) >= 2: lag2 = float(vals[-2])
                            if len(vals) >= 3: lag3 = float(vals[-3])
                    elif station in asos_history.columns:
                        hist = asos_history[station]
                        vals = hist.dropna().values[-3:]
                        if len(vals) >= 1: lag1 = float(vals[-1])
                        if len(vals) >= 2: lag2 = float(vals[-2])
                        if len(vals) >= 3: lag3 = float(vals[-3])
                except Exception:
                    pass

            meta = station_meta.get(station, {})
            lead_sin, lead_cos = _cyclical(lead_hour, 168)

            row: dict = {
                "station":   station,
                "lead_hour": lead_hour,
                # GEFS ensemble distribution
                "gefs_tmax_mean":         gefs["gefs_tmax_mean"],
                "gefs_tmax_median":       gefs["gefs_tmax_median"],
                "gefs_tmax_std":          gefs["gefs_tmax_std"],
                "gefs_tmax_range":        gefs["gefs_tmax_range"],
                "gefs_tmax_iqr":          gefs["gefs_tmax_iqr"],
                "gefs_tmax_p10":          gefs["gefs_tmax_p10"],
                "gefs_tmax_p25":          gefs["gefs_tmax_p25"],
                "gefs_tmax_p75":          gefs["gefs_tmax_p75"],
                "gefs_tmax_p90":          gefs["gefs_tmax_p90"],
                "gefs_ensemble_skewness": gefs["gefs_ensemble_skewness"],
                "gefs_ensemble_kurtosis": gefs["gefs_ensemble_kurtosis"],
                "gefs_tmin_mean":         gefs["gefs_tmin_mean"],
                "gefs_tmin_std":          gefs["gefs_tmin_std"],
                # ECMWF
                "ecmwf_tmax":             ecmwf_tmax,
                "ecmwf_tmin":             ecmwf_tmin,
                "ecmwf_gefs_tmax_delta":  ecmwf_gefs_delta,
                # NBM blend
                **nbm_row,
                # Cloud
                "cloud_cover_total":      cloud_total,
                "cloud_low_frac":         cloud_low,
                "cloud_mid_frac":         cloud_mid,
                "cloud_high_frac":        cloud_high,
                # Wind
                "surface_wind_speed":     gefs["surface_wind_speed"],
                "wind_dir_sin":           gefs["wind_dir_sin"],
                "wind_dir_cos":           gefs["wind_dir_cos"],
                # Moisture
                "surface_dew_point_depression": gefs["surface_dew_point_depression"],
                # Precip
                "convective_precip_prob": gefs["convective_precip_prob"],
                "total_precip_mm":        gefs["total_precip_mm"],
                # Temporal
                "lead_time_hours":        float(lead_hour),
                "lead_sin":               lead_sin,
                "lead_cos":               lead_cos,
                "month_sin":              month_sin,
                "month_cos":              month_cos,
                "day_of_year_sin":        doy_sin,
                "day_of_year_cos":        doy_cos,
                # Station one-hots (fixed to full registry → stable feature space)
                **{f"station_{icao.lower()}": 1.0 if station == icao else 0.0
                   for icao in ALL_ICAO},
                # Station physical meta
                "elevation_delta_m":      meta.get("elevation_delta_m", 0.0),
                "uhi_index":              meta.get("uhi_index", 0.0),
                "coastal_distance_km":    meta.get("coastal_distance_km", 0.0),
                # Observed residuals
                "obs_minus_model_lag1":   lag1,
                "obs_minus_model_lag2":   lag2,
                "obs_minus_model_lag3":   lag3,
            }
            for i, v in enumerate(regime_vec):
                row[f"regime_cluster_{i}"] = v

            rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns() -> list[str]:
    return [
        # GEFS distribution
        "gefs_tmax_mean", "gefs_tmax_median", "gefs_tmax_std",
        "gefs_tmax_range", "gefs_tmax_iqr",
        "gefs_tmax_p10", "gefs_tmax_p25", "gefs_tmax_p75", "gefs_tmax_p90",
        "gefs_ensemble_skewness", "gefs_ensemble_kurtosis",
        "gefs_tmin_mean", "gefs_tmin_std",
        # ECMWF
        "ecmwf_tmax", "ecmwf_tmin", "ecmwf_gefs_tmax_delta",
        # NBM
        "nbm_t10", "nbm_t25", "nbm_t50", "nbm_t75", "nbm_t90",
        "nbm_tmax", "nbm_tmin", "nbm_pop12", "nbm_spread", "nbm_gefs_delta",
        # Cloud
        "cloud_cover_total", "cloud_low_frac", "cloud_mid_frac", "cloud_high_frac",
        # Wind
        "surface_wind_speed", "wind_dir_sin", "wind_dir_cos",
        # Moisture / precip
        "surface_dew_point_depression", "convective_precip_prob", "total_precip_mm",
        # Temporal
        "lead_time_hours", "lead_sin", "lead_cos",
        "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
        # Station one-hots
    ] + [f"station_{icao.lower()}" for icao in ALL_ICAO] + [
        # Physical meta
        "elevation_delta_m", "uhi_index", "coastal_distance_km",
        # Residual lags
        "obs_minus_model_lag1", "obs_minus_model_lag2", "obs_minus_model_lag3",
    ] + [f"regime_cluster_{i}" for i in range(N_REGIME_CLUSTERS)]
