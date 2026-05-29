import numpy as np
import pandas as pd
import pytest
from processing.features import build_feature_matrix, get_feature_columns


def _make_minimal_gefs(station: str, lead_hours: list[int]) -> dict:
    rng = np.random.default_rng(42)
    members = {}
    for i in range(5):
        rows = []
        for lh in lead_hours:
            rows.append({
                "lead_hour": lh,
                "temp_f": float(rng.normal(72, 5)),
                "u10": float(rng.normal(5, 2)),
                "v10": float(rng.normal(3, 2)),
                "tcc": float(rng.uniform(0, 100)),
                "tp": float(rng.uniform(0, 5)),
                "sp": 101325.0,
            })
        members[f"p{i:02d}"] = pd.DataFrame(rows)
    return {station: members}


def test_feature_matrix_has_expected_shape():
    gefs = _make_minimal_gefs("KORD", [24, 48, 72])
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


def test_no_nans_in_core_cyclical_features():
    gefs = _make_minimal_gefs("KJFK", [24])
    ecmwf = {"KJFK": {"tmax_forecast": 70.0, "tmin_forecast": 58.0}}
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
        regime_labels=pd.Series(dtype=float),
    )
    cyclical_cols = ["month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos"]
    for col in cyclical_cols:
        assert col in df.columns, f"Missing column: {col}"
        vals = df[col].dropna()
        assert len(vals) > 0, f"All NaN in {col}"
        assert vals.between(-1.0, 1.0).all(), f"Cyclical values out of [-1,1] in {col}"


def test_station_one_hot_exactly_one_active():
    gefs = _make_minimal_gefs("KLAX", [24])
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
        assert row["station_lax"] == 1.0
        assert row["station_ord"] == 0.0
        assert row["station_jfk"] == 0.0


def test_regime_clusters_sum_to_one_or_zero():
    gefs = _make_minimal_gefs("KORD", [24])
    ecmwf = {"KORD": {"tmax_forecast": 72.0, "tmin_forecast": 55.0}}
    regime = pd.Series([3])
    df = build_feature_matrix(
        gefs_data=gefs,
        ecmwf_data=ecmwf,
        asos_history=pd.DataFrame(),
        regime_labels=regime,
    )
    regime_cols = [f"regime_cluster_{i}" for i in range(12)]
    present = [c for c in regime_cols if c in df.columns]
    if present and not df.empty:
        row_sum = df[present].sum(axis=1).iloc[0]
        assert row_sum in (0.0, 1.0), f"Regime one-hot sum is {row_sum}"


def test_get_feature_columns_returns_list():
    cols = get_feature_columns()
    assert isinstance(cols, list)
    assert len(cols) >= 40
