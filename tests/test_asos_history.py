"""Tests for building live obs_minus_model lag history from ASOS observations
and persisted forecasts.

The NGBoost/QRF models were trained with obs_minus_model_lag1/2/3 as their
single most important feature group (~38% combined importance), but live
inference always feeds 0.0 because run_cycle hardcodes asos_history=pd.DataFrame().
These helpers build a real asos_history DataFrame from (a) recent actual daily
Tmax observed via METAR and (b) the GEFS D+1 forecast persisted on a prior cycle.
"""
import datetime as dt
import numpy as np
import pandas as pd
import pytest

from processing.asos_history import compute_daily_residuals, build_asos_history_df
from processing.features import build_feature_matrix


# ── compute_daily_residuals ──────────────────────────────────────────────────

def test_compute_daily_residuals_returns_actual_minus_forecast_for_overlapping_dates():
    actual = {dt.date(2026, 6, 8): 75.0, dt.date(2026, 6, 9): 80.0}
    forecast = {dt.date(2026, 6, 8): 72.0, dt.date(2026, 6, 9): 83.0}

    residuals = compute_daily_residuals(actual, forecast)

    assert residuals[dt.date(2026, 6, 8)] == pytest.approx(3.0)
    assert residuals[dt.date(2026, 6, 9)] == pytest.approx(-3.0)


def test_compute_daily_residuals_skips_dates_missing_a_forecast():
    actual = {dt.date(2026, 6, 8): 75.0, dt.date(2026, 6, 9): 80.0}
    forecast = {dt.date(2026, 6, 8): 72.0}

    residuals = compute_daily_residuals(actual, forecast)

    assert residuals == {dt.date(2026, 6, 8): pytest.approx(3.0)}


def test_compute_daily_residuals_empty_inputs_returns_empty():
    assert compute_daily_residuals({}, {}) == {}


# ── build_asos_history_df ────────────────────────────────────────────────────

def test_build_asos_history_df_indexes_by_date_with_one_column_per_station():
    residuals = {
        "KORD": {dt.date(2026, 6, 8): 3.0, dt.date(2026, 6, 9): -2.0},
        "KLAX": {dt.date(2026, 6, 8): 1.0},
    }

    df = build_asos_history_df(residuals)

    assert "KORD" in df.columns
    assert "KLAX" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.loc[pd.Timestamp(dt.date(2026, 6, 8)), "KORD"] == pytest.approx(3.0)
    assert df.loc[pd.Timestamp(dt.date(2026, 6, 9)), "KORD"] == pytest.approx(-2.0)


def test_build_asos_history_df_empty_input_returns_empty_dataframe():
    df = build_asos_history_df({})

    assert df.empty


# ── End-to-end: feeds obs_minus_model_lag1/2/3 in build_feature_matrix ──────

def _make_member(temp_f: float = 72.0) -> dict:
    return {
        "member": "p00", "temp_f": temp_f, "dewpoint_f": temp_f - 15.0,
        "wind_speed": 5.0, "wind_dir_sin": 0.0, "wind_dir_cos": 1.0,
        "tcc": 50.0, "tp": 0.0, "sp": 101325.0,
    }


def test_live_residuals_populate_lag_features_via_build_feature_matrix():
    reference_date = dt.date(2026, 6, 10)
    residuals = {
        "KORD": {
            dt.date(2026, 6, 9): 3.0,
            dt.date(2026, 6, 8): -1.5,
            dt.date(2026, 6, 7): 0.5,
        }
    }
    asos = build_asos_history_df(residuals)

    df = build_feature_matrix(
        gefs_data={"KORD": {24: [_make_member()]}},
        ecmwf_data={"KORD": {}},
        asos_history=asos,
        reference_date=reference_date,
    )

    row = df.iloc[0]
    assert row["obs_minus_model_lag1"] == pytest.approx(3.0)
    assert row["obs_minus_model_lag2"] == pytest.approx(-1.5)
    assert row["obs_minus_model_lag3"] == pytest.approx(0.5)
