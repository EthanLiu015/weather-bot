import numpy as np
import pandas as pd
from scipy import stats

STATION_META = {
    "KORD": {"elevation_delta_m": 203.0, "uhi_index": 1.8, "coastal_distance_km": 1500.0},
    "KJFK": {"elevation_delta_m": 4.0,   "uhi_index": 2.5, "coastal_distance_km": 2.0},
    "KLAX": {"elevation_delta_m": 38.0,  "uhi_index": 2.1, "coastal_distance_km": 5.0},
}
ALL_STATIONS = ["KORD", "KJFK", "KLAX"]
N_REGIME_CLUSTERS = 12


def _cyclical(val: float, period: float) -> tuple[float, float]:
    return np.sin(2 * np.pi * val / period), np.cos(2 * np.pi * val / period)


def _cloud_fractions(tcc: float) -> tuple[float, float, float, float]:
    if np.isnan(tcc):
        return float("nan"), float("nan"), float("nan"), float("nan")
    total = min(max(tcc / 100.0, 0.0), 1.0)
    low = total * 0.5
    mid = total * 0.3
    high = total * 0.2
    return total, low, mid, high


def build_feature_matrix(
    gefs_data: dict,
    ecmwf_data: dict,
    asos_history: pd.DataFrame,
    regime_labels: pd.Series,
    station_meta: dict | None = None,
) -> pd.DataFrame:
    rows = []
    if station_meta is None:
        station_meta = STATION_META

    for station in ALL_STATIONS:
        station_gefs = gefs_data.get(station, {})
        if not station_gefs:
            continue

        member_ids = list(station_gefs.keys())
        all_dfs = [station_gefs[m] for m in member_ids]
        lead_hours_available = sorted(set(
            lh for df in all_dfs for lh in df["lead_hour"].dropna().unique()
        ))

        for lead_hour in lead_hours_available:
            member_temps = []
            member_winds = []
            member_tcc = []
            member_tp = []

            for df in all_dfs:
                subset = df[df["lead_hour"] == lead_hour]
                if subset.empty:
                    continue
                t = subset["temp_f"].values[0] if "temp_f" in subset.columns else float("nan")
                u = subset["u10"].values[0] if "u10" in subset.columns else float("nan")
                v = subset["v10"].values[0] if "v10" in subset.columns else float("nan")
                tc = subset["tcc"].values[0] if "tcc" in subset.columns else float("nan")
                tp = subset["tp"].values[0] if "tp" in subset.columns else float("nan")
                member_temps.append(t)
                member_winds.append(np.sqrt(u**2 + v**2) if not (np.isnan(u) or np.isnan(v)) else float("nan"))
                member_tcc.append(tc)
                member_tp.append(tp)

            temps = [t for t in member_temps if not np.isnan(t)]
            winds = [w for w in member_winds if not np.isnan(w)]
            tccs = [t for t in member_tcc if not np.isnan(t)]
            tps = [t for t in member_tp if not np.isnan(t)]

            gefs_tmax_mean = float(np.mean(temps)) if temps else float("nan")
            gefs_tmax_std = float(np.std(temps)) if len(temps) > 1 else float("nan")
            gefs_tmax_p10 = float(np.percentile(temps, 10)) if temps else float("nan")
            gefs_tmax_p90 = float(np.percentile(temps, 90)) if temps else float("nan")
            gefs_tmin_mean = float(np.min(temps)) if temps else float("nan")
            gefs_tmin_std = float(np.std(temps)) if len(temps) > 1 else float("nan")
            gefs_skew = float(stats.skew(temps)) if len(temps) > 2 else 0.0

            ecmwf_station = ecmwf_data.get(station, {})
            ecmwf_tmax = float(ecmwf_station.get("tmax_forecast", float("nan")))
            ecmwf_tmin = float(ecmwf_station.get("tmin_forecast", float("nan")))
            ecmwf_gefs_delta = abs(ecmwf_tmax - gefs_tmax_mean) if not (np.isnan(ecmwf_tmax) or np.isnan(gefs_tmax_mean)) else float("nan")

            avg_tcc = float(np.mean(tccs)) if tccs else float("nan")
            cloud_total, cloud_low, cloud_mid, cloud_high = _cloud_fractions(avg_tcc)

            avg_wind = float(np.mean(winds)) if winds else float("nan")
            avg_tp = float(np.mean(tps)) if tps else float("nan")
            convective_precip_prob = float(np.mean([1.0 if tp > 0 else 0.0 for tp in tps])) if tps else 0.0

            lead_sin, lead_cos = _cyclical(lead_hour, 168)
            import datetime as dt_mod
            today = dt_mod.date.today()
            month = today.month
            doy = today.timetuple().tm_yday
            month_sin, month_cos = _cyclical(month, 12)
            doy_sin, doy_cos = _cyclical(doy, 365)

            meta = station_meta.get(station, {})
            elevation_delta = meta.get("elevation_delta_m", 0.0)
            uhi = meta.get("uhi_index", 0.0)
            coastal = meta.get("coastal_distance_km", 0.0)

            # Lag residuals from asos_history
            lag1 = lag2 = lag3 = float("nan")
            if asos_history is not None and not asos_history.empty and station in asos_history.index.get_level_values(0) if isinstance(asos_history.index, pd.MultiIndex) else station in asos_history.columns:
                try:
                    hist = asos_history[station] if station in asos_history.columns else asos_history.loc[station]
                    vals = hist.dropna().values[-3:]
                    if len(vals) >= 1:
                        lag1 = float(vals[-1])
                    if len(vals) >= 2:
                        lag2 = float(vals[-2])
                    if len(vals) >= 3:
                        lag3 = float(vals[-3])
                except Exception:
                    pass

            # Regime clusters (one-hot)
            regime_vec = [0.0] * N_REGIME_CLUSTERS
            if regime_labels is not None and len(regime_labels) > 0:
                try:
                    latest_label = int(regime_labels.iloc[-1])
                    if 0 <= latest_label < N_REGIME_CLUSTERS:
                        regime_vec[latest_label] = 1.0
                except Exception:
                    pass

            row: dict = {
                "station": station,
                "lead_hour": lead_hour,
                "gefs_tmax_mean": gefs_tmax_mean,
                "gefs_tmax_std": gefs_tmax_std,
                "gefs_tmax_p10": gefs_tmax_p10,
                "gefs_tmax_p90": gefs_tmax_p90,
                "gefs_tmin_mean": gefs_tmin_mean,
                "gefs_tmin_std": gefs_tmin_std,
                "ecmwf_tmax": ecmwf_tmax,
                "ecmwf_tmin": ecmwf_tmin,
                "ecmwf_gefs_tmax_delta": ecmwf_gefs_delta,
                "gefs_ensemble_skewness": gefs_skew,
                "cloud_cover_total": cloud_total,
                "cloud_low_frac": cloud_low,
                "cloud_mid_frac": cloud_mid,
                "cloud_high_frac": cloud_high,
                "wind_850hpa_speed": avg_wind,
                "wind_850hpa_dir_sin": float("nan"),
                "wind_850hpa_dir_cos": float("nan"),
                "surface_wind_speed": avg_wind,
                "surface_dew_point_depression": float("nan"),
                "convective_precip_prob": convective_precip_prob,
                "total_precip_mm": avg_tp,
                "lead_time_hours": lead_hour,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "day_of_year_sin": doy_sin,
                "day_of_year_cos": doy_cos,
                "station_ord": 1.0 if station == "KORD" else 0.0,
                "station_jfk": 1.0 if station == "KJFK" else 0.0,
                "station_lax": 1.0 if station == "KLAX" else 0.0,
                "elevation_delta_m": elevation_delta,
                "uhi_index": uhi,
                "coastal_distance_km": coastal,
                "obs_minus_model_lag1": lag1,
                "obs_minus_model_lag2": lag2,
                "obs_minus_model_lag3": lag3,
            }
            for i, v in enumerate(regime_vec):
                row[f"regime_cluster_{i}"] = v

            rows.append(row)

    return pd.DataFrame(rows)


def get_feature_columns() -> list[str]:
    base = [
        "gefs_tmax_mean", "gefs_tmax_std", "gefs_tmax_p10", "gefs_tmax_p90",
        "gefs_tmin_mean", "gefs_tmin_std", "ecmwf_tmax", "ecmwf_tmin",
        "ecmwf_gefs_tmax_delta", "gefs_ensemble_skewness",
        "cloud_cover_total", "cloud_low_frac", "cloud_mid_frac", "cloud_high_frac",
        "wind_850hpa_speed", "wind_850hpa_dir_sin", "wind_850hpa_dir_cos",
        "surface_wind_speed", "surface_dew_point_depression",
        "convective_precip_prob", "total_precip_mm",
        "lead_time_hours", "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
        "station_ord", "station_jfk", "station_lax",
        "elevation_delta_m", "uhi_index", "coastal_distance_km",
        "obs_minus_model_lag1", "obs_minus_model_lag2", "obs_minus_model_lag3",
    ] + [f"regime_cluster_{i}" for i in range(N_REGIME_CLUSTERS)]
    return base
