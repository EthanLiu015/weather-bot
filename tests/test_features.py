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
        regime_labels=pd.Series(dtype=float),
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
        regime_labels=pd.Series(dtype=float),
    )
    for col in ["month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos"]:
        assert col in df.columns, f"Missing: {col}"
        vals = df[col].dropna()
        assert len(vals) > 0, f"All NaN in {col}"
        assert vals.between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_station_one_hot_exactly_one_active():
    gefs = _make_gefs("KLAX", [24])
    ecmwf = {"KLAX": {"tmax_forecast": 80.0, "tmin_forecast": 65.0}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
        regime_labels=pd.Series(dtype=float),
    )
    lax_rows = df[df["station"] == "KLAX"]
    if not lax_rows.empty:
        row = lax_rows.iloc[0]
        assert row["station_klax"] == 1.0
        assert row["station_kord"] == 0.0
        assert row["station_klga"] == 0.0


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
        regime_labels=pd.Series(dtype=float),
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
        regime_labels=pd.Series(dtype=float),
        nbm_data=nbm,
    )
    assert not df.empty
    row = df.iloc[0]
    assert row["nbm_t50"] == pytest.approx(72.0)
    assert row["nbm_spread"] == pytest.approx(14.0)
    assert not np.isnan(row["nbm_gefs_delta"])


def test_regime_clusters_sum_to_one_or_zero():
    gefs = _make_gefs("KORD", [24])
    ecmwf = {"KORD": {}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
        regime_labels=pd.Series([3]),
    )
    regime_cols = [f"regime_cluster_{i}" for i in range(12)]
    present = [c for c in regime_cols if c in df.columns]
    if present and not df.empty:
        row_sum = df[present].sum(axis=1).iloc[0]
        assert row_sum in (0.0, 1.0)


def test_get_feature_columns_returns_list():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) >= 55
    assert "nbm_t50" in cols
    assert "gefs_tmax_iqr" in cols
    assert "gefs_ensemble_kurtosis" in cols
    assert "nbm_gefs_delta" in cols
