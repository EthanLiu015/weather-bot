"""Tests for temporal correctness of lag features."""
import datetime as dt
import numpy as np
import pandas as pd
import pytest

from processing.features import build_feature_matrix


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


def _make_gefs(station: str = "KORD") -> dict:
    rng = np.random.default_rng(0)
    members = [_make_member(float(rng.normal(72, 5))) for _ in range(5)]
    return {station: {24: members}}


def _asos_column_based(station: str, dates: list[dt.date], values: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.DataFrame({station: values}, index=index)


# ── Gap 5: lag features set to NaN when ASOS data is stale ───────────────────

def test_lag_features_are_nan_when_asos_data_is_stale():
    reference_date = dt.date(2024, 6, 15)
    stale_dates = [dt.date(2024, 6, 1), dt.date(2024, 6, 2), dt.date(2024, 6, 3)]
    asos = _asos_column_based("KORD", stale_dates, [70.0, 71.0, 72.0])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert np.isnan(row["obs_minus_model_lag1"]), "Stale lags should be NaN"
    assert np.isnan(row["obs_minus_model_lag2"]), "Stale lags should be NaN"
    assert np.isnan(row["obs_minus_model_lag3"]), "Stale lags should be NaN"


def test_lag_features_populated_when_asos_data_is_recent():
    reference_date = dt.date(2024, 6, 15)
    recent_dates = [dt.date(2024, 6, 13), dt.date(2024, 6, 14), dt.date(2024, 6, 15)]
    asos = _asos_column_based("KORD", recent_dates, [70.0, 71.0, 72.0])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert not np.isnan(row["obs_minus_model_lag1"]), "Recent lags should not be NaN"


def test_rolling_residual_stats_populated_with_enough_history():
    reference_date = dt.date(2024, 6, 15)
    dates = [dt.date(2024, 6, d) for d in range(9, 16)]  # 7 days
    values = [70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0]
    asos = _asos_column_based("KORD", dates, values)

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert row["obs_minus_model_roll_mean"] == pytest.approx(np.mean(values))
    assert row["obs_minus_model_roll_std"] == pytest.approx(np.std(values))


def test_rolling_residual_stats_nan_with_single_observation():
    reference_date = dt.date(2024, 6, 15)
    asos = _asos_column_based("KORD", [dt.date(2024, 6, 15)], [70.0])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert np.isnan(row["obs_minus_model_roll_mean"])
    assert np.isnan(row["obs_minus_model_roll_std"])


def test_rolling_residual_stats_nan_when_asos_data_is_stale():
    reference_date = dt.date(2024, 6, 15)
    stale_dates = [dt.date(2024, 6, 1), dt.date(2024, 6, 2), dt.date(2024, 6, 3)]
    asos = _asos_column_based("KORD", stale_dates, [70.0, 71.0, 72.0])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert np.isnan(row["obs_minus_model_roll_mean"])
    assert np.isnan(row["obs_minus_model_roll_std"])


def test_lag_features_only_use_asos_entries_on_or_before_reference_date():
    reference_date = dt.date(2024, 6, 10)
    dates = [dt.date(2024, 6, 8), dt.date(2024, 6, 9), dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
    asos = _asos_column_based("KORD", dates, [70.0, 71.0, 72.0, 999.0])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert row["obs_minus_model_lag1"] != pytest.approx(999.0), "Future entry must not be used as lag1"
