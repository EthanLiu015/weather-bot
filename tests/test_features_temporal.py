"""Tests for temporal correctness of lag features and regime label lookup."""
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


def _regime_series_with_dates(dates: list[dt.date], labels: list[int]) -> pd.Series:
    index = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    return pd.Series(labels, index=index)


# ── Gap 5: lag features set to NaN when ASOS data is stale ───────────────────

def test_lag_features_are_nan_when_asos_data_is_stale():
    reference_date = dt.date(2024, 6, 15)
    stale_dates = [dt.date(2024, 6, 1), dt.date(2024, 6, 2), dt.date(2024, 6, 3)]
    asos = _asos_column_based("KORD", stale_dates, [70.0, 71.0, 72.0])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        regime_labels=pd.Series(dtype=float),
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
        regime_labels=pd.Series(dtype=float),
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert not np.isnan(row["obs_minus_model_lag1"]), "Recent lags should not be NaN"


def test_lag_features_only_use_asos_entries_on_or_before_reference_date():
    reference_date = dt.date(2024, 6, 10)
    dates = [dt.date(2024, 6, 8), dt.date(2024, 6, 9), dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
    asos = _asos_column_based("KORD", dates, [70.0, 71.0, 72.0, 999.0])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        regime_labels=pd.Series(dtype=float),
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert row["obs_minus_model_lag1"] != pytest.approx(999.0), "Future entry must not be used as lag1"


# ── Gap 6: regime label selected by reference_date, not iloc[-1] ─────────────

def test_regime_label_uses_reference_date_not_latest():
    reference_date = dt.date(2024, 6, 10)
    dates = [dt.date(2024, 6, 9), dt.date(2024, 6, 10), dt.date(2024, 6, 11)]
    # label on reference_date is 3; latest (Jun 11) is 7
    regime = _regime_series_with_dates(dates, [2, 3, 7])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=pd.DataFrame(),
        regime_labels=regime,
        reference_date=reference_date,
    )

    assert not df.empty
    row = df.iloc[0]
    assert row["regime_cluster_3"] == 1.0, "Should use label for reference_date (cluster 3)"
    assert row["regime_cluster_7"] == 0.0, "Should NOT use latest label (cluster 7)"


def test_regime_label_falls_back_to_most_recent_when_no_date_index():
    regime = pd.Series([2, 3, 7])

    df = build_feature_matrix(
        gefs_data=_make_gefs("KORD"),
        ecmwf_data={"KORD": {}},
        asos_history=pd.DataFrame(),
        regime_labels=regime,
        reference_date=dt.date(2024, 6, 10),
    )

    assert not df.empty
    row = df.iloc[0]
    assert row["regime_cluster_7"] == 1.0, "Should fall back to iloc[-1] when no date index"
