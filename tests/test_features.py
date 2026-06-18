import numpy as np
import pandas as pd
import pytest
from processing.features import build_feature_matrix, get_feature_columns


def _make_member(temp_f: float = 72.0) -> dict:
    rng = np.random.default_rng(42)
    return {
        "member": "p00",
        "temp_f": temp_f,
        "dewpoint_f": temp_f - 15.0,
        "wind_speed": float(rng.uniform(3, 15)),
        "wind_dir_sin": float(rng.uniform(-1, 1)),
        "wind_dir_cos": float(rng.uniform(-1, 1)),
        "tcc": float(rng.uniform(0, 100)),
        "tp": float(rng.uniform(0, 3)),
        "sp": 101325.0,
    }


def _make_gefs(station: str, lead_hours: list[int], n_members: int = 10) -> dict:
    rng = np.random.default_rng(0)
    station_data: dict[int, list[dict]] = {}
    for lh in lead_hours:
        members = []
        for i in range(n_members):
            m = _make_member(float(rng.normal(72, 5)))
            m["member"] = f"p{i:02d}"
            members.append(m)
        station_data[lh] = members
    return {station: station_data}


def test_feature_matrix_has_expected_shape():
    gefs = _make_gefs("KORD", [24, 48, 72])
    ecmwf = {"KORD": {"tmax_forecast": 75.0, "tmin_forecast": 60.0}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
    )
    assert not df.empty
    assert "gefs_tmax_mean" in df.columns
    assert "station" in df.columns
    assert len(df) == 3  # one row per lead hour


def test_no_nans_in_core_cyclical_features():
    gefs = _make_gefs("KLGA", [24])
    ecmwf = {"KLGA": {"tmax_forecast": 70.0, "tmin_forecast": 58.0}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
    )
    for col in ["day_of_year_sin", "day_of_year_cos"]:
        assert col in df.columns, f"Missing: {col}"
        vals = df[col].dropna()
        assert len(vals) > 0, f"All NaN in {col}"
        assert vals.between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_feature_matrix_has_no_station_one_hot_columns():
    gefs = _make_gefs("KLAX", [24])
    ecmwf = {"KLAX": {"tmax_forecast": 80.0, "tmin_forecast": 65.0}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
    )
    assert not df.empty
    assert not any(c.startswith("station_k") for c in df.columns), (
        "station one-hots are constant within each per-station model "
        "(train_final_models trains one model per station) — removed entirely"
    )


def test_onshore_wind_component_present_for_coastal_station_and_zero_inland():
    gefs = _make_gefs("KSFO", [24])
    ecmwf = {"KSFO": {}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
    )
    assert not df.empty
    row = df.iloc[0]
    assert "onshore_wind_component" in df.columns
    assert -1.0 <= row["onshore_wind_component"] <= 1.0

    gefs_inland = _make_gefs("KORD", [24])
    df_inland = build_feature_matrix(
        gefs_data=gefs_inland,
        ecmwf_data={"KORD": {}},
        asos_history=pd.DataFrame(),
    )
    assert df_inland.iloc[0]["onshore_wind_component"] == pytest.approx(0.0)


def test_climatology_anomaly_feature_present():
    gefs = _make_gefs("KORD", [24])
    ecmwf = {"KORD": {}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
    )
    assert not df.empty
    assert "gefs_tmax_climo_anomaly" in df.columns


def test_lead_time_sqrt_replaces_lead_cyclical_features():
    gefs = _make_gefs("KORD", [24, 96])
    ecmwf = {"KORD": {}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
    )
    assert not df.empty
    assert "lead_time_sqrt" in df.columns
    assert "lead_sin" not in df.columns
    assert "lead_cos" not in df.columns
    row24 = df[df["lead_hour"] == 24].iloc[0]
    row96 = df[df["lead_hour"] == 96].iloc[0]
    assert row24["lead_time_sqrt"] == pytest.approx(24 ** 0.5)
    assert row96["lead_time_sqrt"] == pytest.approx(96 ** 0.5)


def test_full_31_member_statistics_computed():
    rng = np.random.default_rng(7)
    members = [_make_member(float(rng.normal(72, 5))) for _ in range(31)]
    for i, m in enumerate(members):
        m["member"] = f"p{i:02d}"
    gefs = {"KORD": {24: members}}
    ecmwf = {"KORD": {}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
    )
    assert not df.empty
    row = df.iloc[0]
    assert not np.isnan(row["gefs_tmax_iqr"])
    assert not np.isnan(row["gefs_tmax_range"])
    assert not np.isnan(row["gefs_ensemble_kurtosis"])
    assert row["gefs_tmax_p25"] <= row["gefs_tmax_p75"]


def test_nbm_features_populated():
    gefs = _make_gefs("KORD", [24])
    ecmwf = {"KORD": {}}
    nbm = {
        "KORD": {
            24: {"t10": 65.0, "t25": 68.0, "t50": 72.0, "t75": 76.0,
                 "t90": 79.0, "tmax": 80.0, "tmin": 62.0, "pop12": 0.1, "spread": 14.0}
        }
    }
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
        nbm_data=nbm,
    )
    assert not df.empty
    row = df.iloc[0]
    assert row["nbm_t50"] == pytest.approx(72.0)
    assert row["nbm_spread"] == pytest.approx(14.0)
    assert not np.isnan(row["nbm_gefs_delta"])


def test_feature_matrix_has_no_regime_cluster_columns():
    gefs = _make_gefs("KORD", [24])
    ecmwf = {"KORD": {}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
    )
    assert not df.empty
    assert not any(c.startswith("regime_cluster") for c in df.columns), (
        "regime_cluster_* is built from synthetic random noise (not real ERA5 "
        "data) and was always-zero at inference — removed entirely"
    )


def test_get_feature_columns_returns_list():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) == 40
    assert "nbm_t50" in cols
    assert "gefs_tmax_iqr" in cols
    assert "gefs_ensemble_kurtosis" in cols
    assert "nbm_gefs_delta" in cols
    assert not any(c.startswith("regime_cluster") for c in cols)
    assert not any(c.startswith("station_k") for c in cols)
    assert "cloud_cover_total" in cols
    assert "cloud_low_frac" not in cols
    assert "cloud_mid_frac" not in cols
    assert "cloud_high_frac" not in cols
    assert "gefs_tmin_mean" not in cols
    assert "gefs_tmin_std" not in cols
    assert "total_precip_mm" not in cols
    assert "convective_precip_prob" not in cols
    assert "month_sin" not in cols
    assert "month_cos" not in cols
    assert "lead_sin" not in cols
    assert "lead_cos" not in cols
    assert "lead_time_sqrt" in cols
    assert "gefs_tmax_climo_anomaly" in cols
    assert "onshore_wind_component" in cols
    assert "obs_minus_model_roll_mean" in cols
    assert "obs_minus_model_roll_std" in cols
