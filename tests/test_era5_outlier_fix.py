import datetime

import numpy as np
import pandas as pd
import pytest

from scripts.era5_outlier_fix import (
    clip_gefs_outliers,
    compute_station_floors,
    recompute_lag_residuals,
)


def _base_row(**overrides) -> dict:
    row = {
        "station": "KORD",
        "date": datetime.date(2024, 1, 1),
        "lead_hour": 24,
        "actual_tmax": 40.0,
        "gefs_tmax_mean": 30.0,
        "ecmwf_tmax": 35.0,
        "ecmwf_gefs_tmax_delta": 5.0,
        "gefs_tmax_climo_anomaly": -5.0,
        "obs_minus_model_lag1": float("nan"),
        "obs_minus_model_lag2": float("nan"),
        "obs_minus_model_lag3": float("nan"),
        "obs_minus_model_roll_mean": float("nan"),
        "obs_minus_model_roll_std": float("nan"),
    }
    row.update(overrides)
    return row


# ── compute_station_floors ────────────────────────────────────────────────────


def test_compute_station_floors_returns_one_floor_per_station():
    df = pd.DataFrame([
        _base_row(station="KORD", gefs_tmax_mean=float(v))
        for v in range(100)
    ] + [
        _base_row(station="KSEA", gefs_tmax_mean=float(v) * 0.5)
        for v in range(100)
    ])
    floors = compute_station_floors(df)
    assert set(floors.keys()) == {"KORD", "KSEA"}


def test_compute_station_floors_is_1st_percentile():
    values = list(range(200))  # uniform 0..199
    df = pd.DataFrame([_base_row(gefs_tmax_mean=float(v)) for v in values])
    floors = compute_station_floors(df)
    expected = np.percentile(values, 1)
    assert floors["KORD"] == pytest.approx(expected, abs=0.01)


def test_compute_station_floors_is_per_station_not_global():
    # KSEA has much colder ERA5 values — its floor should be lower than KORD's
    kord_rows = [_base_row(station="KORD", gefs_tmax_mean=float(v)) for v in range(50, 100)]
    ksea_rows = [_base_row(station="KSEA", gefs_tmax_mean=float(v) - 40) for v in range(50, 100)]
    df = pd.DataFrame(kord_rows + ksea_rows)
    floors = compute_station_floors(df)
    assert floors["KSEA"] < floors["KORD"]


# ── clip_gefs_outliers ────────────────────────────────────────────────────────


def test_clip_gefs_outliers_raises_below_floor_to_floor():
    floor = 5.0
    df = pd.DataFrame([_base_row(gefs_tmax_mean=-16.0)])
    result = clip_gefs_outliers(df, floors={"KORD": floor})
    assert result.iloc[0]["gefs_tmax_mean"] == pytest.approx(floor)


def test_clip_gefs_outliers_leaves_values_above_floor_unchanged():
    df = pd.DataFrame([_base_row(gefs_tmax_mean=28.0)])
    result = clip_gefs_outliers(df, floors={"KORD": 5.0})
    assert result.iloc[0]["gefs_tmax_mean"] == pytest.approx(28.0)


def test_clip_gefs_outliers_recomputes_ecmwf_delta():
    # ecmwf_tmax=35, gefs clipped from -16 to 5 → delta = |35-5| = 30
    df = pd.DataFrame([_base_row(gefs_tmax_mean=-16.0, ecmwf_tmax=35.0)])
    result = clip_gefs_outliers(df, floors={"KORD": 5.0})
    assert result.iloc[0]["ecmwf_gefs_tmax_delta"] == pytest.approx(abs(35.0 - 5.0))


def test_clip_gefs_outliers_ecmwf_delta_unchanged_when_no_clip():
    df = pd.DataFrame([_base_row(gefs_tmax_mean=30.0, ecmwf_tmax=35.0, ecmwf_gefs_tmax_delta=5.0)])
    result = clip_gefs_outliers(df, floors={"KORD": 5.0})
    assert result.iloc[0]["ecmwf_gefs_tmax_delta"] == pytest.approx(5.0)


def test_clip_gefs_outliers_preserves_row_count():
    df = pd.DataFrame([_base_row(gefs_tmax_mean=float(v)) for v in [-20, -5, 10, 30, 50]])
    result = clip_gefs_outliers(df, floors={"KORD": 0.0})
    assert len(result) == len(df)


def test_clip_gefs_outliers_only_clips_affected_station():
    rows = [
        _base_row(station="KORD", gefs_tmax_mean=-16.0),
        _base_row(station="KSEA", gefs_tmax_mean=-16.0),
    ]
    df = pd.DataFrame(rows)
    result = clip_gefs_outliers(df, floors={"KORD": 5.0, "KSEA": -20.0})
    assert result[result["station"] == "KORD"].iloc[0]["gefs_tmax_mean"] == pytest.approx(5.0)
    assert result[result["station"] == "KSEA"].iloc[0]["gefs_tmax_mean"] == pytest.approx(-16.0)


# ── recompute_lag_residuals ───────────────────────────────────────────────────


def _make_sequence(station: str, dates: list, gefs_values: list, actuals: list, lead_hour: int = 24) -> pd.DataFrame:
    """Build a mini features DataFrame for a single station and lead_hour."""
    rows = []
    for d, g, a in zip(dates, gefs_values, actuals):
        row = _base_row(station=station, date=d, lead_hour=lead_hour,
                        gefs_tmax_mean=g, actual_tmax=a)
        rows.append(row)
    return pd.DataFrame(rows)


def test_recompute_lag_residuals_sets_lag1_from_previous_day():
    dates = [datetime.date(2024, 1, d) for d in range(1, 5)]
    gefs = [50.0, 48.0, 46.0, 44.0]
    actuals = [55.0, 53.0, 51.0, 49.0]
    df = _make_sequence("KORD", dates, gefs, actuals)
    result = recompute_lag_residuals(df)
    # lag1 for 2024-01-02 = actual[01-01] - gefs[01-01] = 55-50 = 5.0
    row = result[(result["date"] == datetime.date(2024, 1, 2)) & (result["lead_hour"] == 24)].iloc[0]
    assert row["obs_minus_model_lag1"] == pytest.approx(5.0)


def test_recompute_lag_residuals_uses_clipped_gefs_for_lag():
    # If gefs on day 1 was clipped from -16 to 5, lag1 on day 2 = actual[d1] - 5
    dates = [datetime.date(2024, 1, d) for d in range(1, 3)]
    df = _make_sequence("KSEA", dates, [5.0, 40.0], [47.0, 47.0])
    result = recompute_lag_residuals(df)
    row = result[(result["date"] == datetime.date(2024, 1, 2)) & (result["lead_hour"] == 24)].iloc[0]
    assert row["obs_minus_model_lag1"] == pytest.approx(47.0 - 5.0)


def test_recompute_lag_residuals_lag1_nan_at_start_of_history():
    dates = [datetime.date(2024, 1, d) for d in range(1, 3)]
    df = _make_sequence("KORD", dates, [50.0, 48.0], [55.0, 53.0])
    result = recompute_lag_residuals(df)
    row = result[result["date"] == datetime.date(2024, 1, 1)].iloc[0]
    assert np.isnan(row["obs_minus_model_lag1"])


def test_recompute_lag_residuals_sets_lag2_and_lag3():
    dates = [datetime.date(2024, 1, d) for d in range(1, 6)]
    gefs = [50.0, 48.0, 46.0, 44.0, 42.0]
    actuals = [55.0, 53.0, 51.0, 49.0, 47.0]
    df = _make_sequence("KORD", dates, gefs, actuals)
    result = recompute_lag_residuals(df)
    row = result[result["date"] == datetime.date(2024, 1, 4)].iloc[0]
    assert row["obs_minus_model_lag1"] == pytest.approx(51.0 - 46.0)  # day3 residual
    assert row["obs_minus_model_lag2"] == pytest.approx(53.0 - 48.0)  # day2 residual
    assert row["obs_minus_model_lag3"] == pytest.approx(55.0 - 50.0)  # day1 residual


def test_recompute_lag_residuals_lags_apply_to_all_lead_hours():
    # The lag for a given valid_date should be the same for all lead_hours
    base_date = datetime.date(2024, 1, 1)
    next_date = datetime.date(2024, 1, 2)
    rows = []
    for lh in [24, 48, 72, 96, 120, 168]:
        rows.append(_base_row(station="KORD", date=base_date, lead_hour=lh,
                              gefs_tmax_mean=50.0, actual_tmax=55.0))
        rows.append(_base_row(station="KORD", date=next_date, lead_hour=lh,
                              gefs_tmax_mean=48.0, actual_tmax=53.0))
    df = pd.DataFrame(rows)
    result = recompute_lag_residuals(df)
    next_day_rows = result[result["date"] == next_date]
    lag1_vals = next_day_rows["obs_minus_model_lag1"].unique()
    assert len(lag1_vals) == 1
    assert lag1_vals[0] == pytest.approx(5.0)  # 55 - 50


def test_recompute_lag_residuals_does_not_mix_stations():
    dates = [datetime.date(2024, 1, d) for d in range(1, 3)]
    kord_df = _make_sequence("KORD", dates, [50.0, 48.0], [60.0, 58.0])
    ksea_df = _make_sequence("KSEA", dates, [10.0, 8.0], [45.0, 43.0])
    df = pd.concat([kord_df, ksea_df], ignore_index=True)
    result = recompute_lag_residuals(df)
    kord_row = result[(result["station"] == "KORD") & (result["date"] == datetime.date(2024, 1, 2))].iloc[0]
    ksea_row = result[(result["station"] == "KSEA") & (result["date"] == datetime.date(2024, 1, 2))].iloc[0]
    assert kord_row["obs_minus_model_lag1"] == pytest.approx(60.0 - 50.0)
    assert ksea_row["obs_minus_model_lag1"] == pytest.approx(45.0 - 10.0)
