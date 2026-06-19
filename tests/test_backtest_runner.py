from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtest.report import FoldResult
from backtest.runner import BacktestRunner, RESIDUAL_SIGMA_FLOOR


def _make_settings():
    return SimpleNamespace(MIN_EDGE_CENTS=5, STATIONS=["KNYC"])


def _make_runner(**kwargs):
    defaults = dict(
        settings=_make_settings(),
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 1),
    )
    defaults.update(kwargs)
    with patch.object(BacktestRunner, "_load_kalshi_prices", return_value=pd.DataFrame()):
        return BacktestRunner(**defaults)


def _fake_fold_result(fold_month: date) -> FoldResult:
    return FoldResult(
        fold_month=fold_month,
        crps=0.0, mae=0.0, brier_score=0.0,
        reliability_slope=0.0, simulated_pnl_usd=0.0,
        num_simulated_trades=0, edge_above_threshold_pct=0.0,
    )


# ── Expanding/anchored training window ───────────────────────────────────────

def test_run_uses_expanding_anchored_training_window():
    """Each fold's training window should start at the fixed start_date
    (anchored) and grow as test_month advances, rather than rolling forward
    with a fixed-size window — this matches the production models, which are
    trained on all available history rather than a rolling N-year slice."""
    runner = _make_runner(
        start_date=date(2020, 1, 1),
        end_date=date(2020, 4, 1),
        train_window_years=0,
    )

    with patch.object(runner, "_run_fold", side_effect=lambda *a: _fake_fold_result(a[2])) as mock_run_fold:
        runner.run()

    train_starts = [call.args[0] for call in mock_run_fold.call_args_list]
    train_ends = [call.args[1] for call in mock_run_fold.call_args_list]
    test_months = [call.args[2] for call in mock_run_fold.call_args_list]

    assert test_months == [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1), date(2020, 4, 1)]
    # Anchored: every fold's training window starts at the same date
    assert all(ts == date(2020, 1, 1) for ts in train_starts)
    # Expanding: training window end grows monotonically with each fold
    assert train_ends == sorted(train_ends)
    assert train_ends[-1] > train_ends[0]


def _make_features(n: int = 60, start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="D").date
    return pd.DataFrame({
        "date": dates,
        "station": "KNYC",
        "lead_hour": 24,
        "gefs_tmax_mean": rng.normal(72, 5, n),
        "gefs_tmax_std": rng.uniform(1, 4, n),
        "actual_tmax": rng.normal(72, 4, n),
    })


def _make_kalshi_prices(station: str, date_: date, threshold: float, mid: float) -> pd.DataFrame:
    return pd.DataFrame([{
        "station": station,
        "date": date_,
        "market_type": "above",
        "threshold": threshold,
        "d1_mid": mid,
    }])


# ── Gap 14: empty fold raises ValueError ─────────────────────────────────────

def test_run_fold_raises_when_train_data_is_empty():
    runner = _make_runner()
    train_start = date(2020, 1, 1)
    train_end = date(2022, 12, 31)
    test_month = date(2023, 1, 1)

    populated = _make_features(60, "2023-01-01")

    def fake_load(start, end):
        if end < test_month:
            return pd.DataFrame()
        return populated

    with patch.object(runner, "_load_historical_features", side_effect=fake_load):
        with pytest.raises(ValueError, match="train"):
            runner._run_fold(train_start, train_end, test_month)


def test_run_fold_raises_when_test_data_is_empty():
    runner = _make_runner()
    train_start = date(2020, 1, 1)
    train_end = date(2022, 12, 31)
    test_month = date(2023, 1, 1)

    populated = _make_features(60, "2020-01-01")

    def fake_load(start, end):
        if start >= test_month:
            return pd.DataFrame()
        return populated

    with patch.object(runner, "_load_historical_features", side_effect=fake_load):
        with pytest.raises(ValueError, match="test"):
            runner._run_fold(train_start, train_end, test_month)


# ── Gap 13: NaN predictions raise ValueError ─────────────────────────────────

def test_validate_predictions_raises_on_nan_mu():
    with pytest.raises(ValueError, match="NaN"):
        BacktestRunner._validate_predictions(
            mu=np.array([72.0, float("nan"), 70.0]),
            sigma=np.array([2.0, 2.0, 2.0]),
            observations=np.array([71.0, 73.0, 69.0]),
            fold_month=date(2023, 1, 1),
        )


def test_validate_predictions_raises_on_nan_observations():
    with pytest.raises(ValueError, match="NaN"):
        BacktestRunner._validate_predictions(
            mu=np.array([72.0, 71.0, 70.0]),
            sigma=np.array([2.0, 2.0, 2.0]),
            observations=np.array([71.0, float("nan"), 69.0]),
            fold_month=date(2023, 1, 1),
        )


def test_validate_predictions_passes_when_all_finite():
    BacktestRunner._validate_predictions(
        mu=np.array([72.0, 71.0, 70.0]),
        sigma=np.array([2.0, 2.0, 2.0]),
        observations=np.array([71.0, 73.0, 69.0]),
        fold_month=date(2023, 1, 1),
    )


# ── Gap 15: threshold distance guard in _get_market_mid ──────────────────────

def test_get_market_mid_returns_none_when_threshold_too_far():
    runner = _make_runner()
    runner._kalshi_prices = _make_kalshi_prices("KNYC", date(2023, 6, 1), threshold=70.0, mid=0.55)

    result = runner._get_market_mid("KNYC", date(2023, 6, 1), threshold=76.0)

    assert result is None


def test_get_market_mid_returns_mid_when_threshold_within_tolerance():
    runner = _make_runner()
    runner._kalshi_prices = _make_kalshi_prices("KNYC", date(2023, 6, 1), threshold=70.0, mid=0.55)

    result = runner._get_market_mid("KNYC", date(2023, 6, 1), threshold=72.0)

    assert result == pytest.approx(0.55)


def test_get_market_mid_returns_none_at_exact_boundary():
    runner = _make_runner()
    runner._kalshi_prices = _make_kalshi_prices("KNYC", date(2023, 6, 1), threshold=70.0, mid=0.55)

    result = runner._get_market_mid("KNYC", date(2023, 6, 1), threshold=75.1)

    assert result is None


# ── Per-station model training ───────────────────────────────────────────────

def test_run_fold_calls_fit_qrf_and_blend_once_per_station():
    """_run_fold must train a separate NGBoost+QRF per station, mirroring the
    per-station loop in initial_train.py so backtest metrics reflect production."""
    runner = _make_runner()
    train_start = date(2020, 1, 1)
    train_end = date(2022, 12, 31)
    test_month = date(2023, 1, 1)

    rng = np.random.default_rng(5)
    n_per = 120
    stations = ["KNYC", "KLAX"]

    def _make_station_df(station, start, n):
        return pd.DataFrame({
            "date": pd.date_range(start, periods=n, freq="D").date,
            "station": station,
            "lead_hour": 24,
            "actual_tmax": rng.normal(72, 4, n),
            "ecmwf_tmax": rng.normal(70, 3, n),
            "ecmwf_diurnal_range": np.full(n, 15.0),
            "gefs_tmax_mean": rng.normal(71, 3, n),
        })

    train_df = pd.concat([
        _make_station_df("KNYC", "2020-01-01", n_per),
        _make_station_df("KLAX", "2020-01-01", n_per),
    ], ignore_index=True)
    test_df = pd.concat([
        _make_station_df("KNYC", "2023-01-01", 10),
        _make_station_df("KLAX", "2023-01-01", 10),
    ], ignore_index=True)

    call_count = {"n": 0}
    original_fit_qrf = runner._fit_qrf_and_blend

    def counting_fit(X_train, y_train, X_test, **kw):
        call_count["n"] += 1
        return original_fit_qrf(X_train, y_train, X_test, **kw)

    def fake_load(start, end):
        if end < test_month:
            return train_df
        return test_df

    with patch.object(runner, "_load_historical_features", side_effect=fake_load), \
         patch.object(runner, "_fit_qrf_and_blend", side_effect=counting_fit):
        try:
            runner._run_fold(train_start, train_end, test_month)
        except Exception:
            pass

    assert call_count["n"] == len(stations), (
        f"Expected _fit_qrf_and_blend called {len(stations)} times (once per station); "
        f"got {call_count['n']}"
    )


def test_run_fold_skips_station_with_insufficient_train_data():
    """Stations with < 100 training rows must be excluded without raising."""
    runner = _make_runner()
    train_start = date(2020, 1, 1)
    train_end = date(2022, 12, 31)
    test_month = date(2023, 1, 1)

    rng = np.random.default_rng(7)

    def _df(station, start, n, dr=15.0):
        return pd.DataFrame({
            "date": pd.date_range(start, periods=n, freq="D").date,
            "station": station,
            "lead_hour": 24,
            "actual_tmax": rng.normal(72, 4, n),
            "ecmwf_tmax": rng.normal(70, 3, n),
            "ecmwf_diurnal_range": np.full(n, dr),
            "gefs_tmax_mean": rng.normal(71, 3, n),
        })

    train_df = pd.concat([
        _df("KNYC", "2020-01-01", 120),   # enough data
        _df("KLAX", "2020-01-01", 50),    # too few → should be skipped
    ], ignore_index=True)
    test_df = pd.concat([
        _df("KNYC", "2023-01-01", 10),
        _df("KLAX", "2023-01-01", 10),
    ], ignore_index=True)

    def fake_load(start, end):
        if end < test_month:
            return train_df
        return test_df

    with patch.object(runner, "_load_historical_features", side_effect=fake_load):
        try:
            result = runner._run_fold(train_start, train_end, test_month)
            assert result is not None
        except Exception as e:
            pytest.fail(f"_run_fold raised unexpectedly: {e}")


# ── Per-lead-bucket model training ───────────────────────────────────────────

def test_run_fold_calls_fit_qrf_and_blend_once_per_station_lead_bucket():
    """_run_fold must train a separate NGBoost+QRF per (station, lead_bucket).
    With 1 station and 3 lead buckets (D1-2, D3-4, D5-7), expect 3 _fit_qrf_and_blend calls."""
    runner = _make_runner()
    train_start = date(2020, 1, 1)
    train_end = date(2022, 12, 31)
    test_month = date(2023, 1, 1)

    rng = np.random.default_rng(99)
    n_per_bucket = 120  # >= 100 minimum per (station, lead_bucket)

    # One station, three lead buckets (24h, 72h, 120h)
    def _rows(lead_hour, n, start):
        return pd.DataFrame({
            "date": pd.date_range(start, periods=n, freq="D").date,
            "station": "KNYC",
            "lead_hour": lead_hour,
            "actual_tmax": rng.normal(72, 4, n),
            "ecmwf_tmax": rng.normal(70, 3, n),
            "ecmwf_diurnal_range": np.full(n, 15.0),
            "gefs_tmax_mean": rng.normal(71, 3, n),
        })

    train_df = pd.concat([
        _rows(24, n_per_bucket, "2020-01-01"),
        _rows(72, n_per_bucket, "2020-01-01"),
        _rows(120, n_per_bucket, "2020-01-01"),
    ], ignore_index=True)
    test_df = pd.concat([
        _rows(24, 10, "2023-01-01"),
        _rows(72, 10, "2023-01-01"),
        _rows(120, 10, "2023-01-01"),
    ], ignore_index=True)

    call_count = {"n": 0}
    original_fit_qrf = runner._fit_qrf_and_blend

    def counting_fit(X_train, y_train, X_test, **kw):
        call_count["n"] += 1
        return original_fit_qrf(X_train, y_train, X_test, **kw)

    def fake_load(start, end):
        if end < test_month:
            return train_df
        return test_df

    with patch.object(runner, "_load_historical_features", side_effect=fake_load), \
         patch.object(runner, "_fit_qrf_and_blend", side_effect=counting_fit):
        try:
            runner._run_fold(train_start, train_end, test_month)
        except Exception:
            pass

    # 1 station × 3 lead buckets = 3 model fits
    assert call_count["n"] == 3, (
        f"Expected 3 _fit_qrf_and_blend calls (1 station × 3 lead buckets); "
        f"got {call_count['n']}"
    )


# ── Zero ecmwf_diurnal_range rows excluded from training ─────────────────────

def test_run_fold_excludes_zero_diurnal_range_from_training():
    """Rows where ecmwf_diurnal_range=0 are ERA5 artefacts (tmax=tmin=t2m);
    they must be filtered from training to prevent the model learning a spurious
    zero-range signal that won't appear at inference time.
    Uses 300 total rows so 150 non-zero rows survive both the diurnal-range
    filter and the per-station 100-row minimum."""
    runner = _make_runner()
    train_start = date(2020, 1, 1)
    train_end = date(2022, 12, 31)
    test_month = date(2023, 1, 1)

    rng = np.random.default_rng(42)
    n_train = 300

    # First half has diurnal_range=0 (ERA5 artefacts); second half is valid
    train_df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n_train, freq="D").date,
        "station": "KNYC",
        "lead_hour": 24,
        "actual_tmax": rng.normal(72, 4, n_train),
        "ecmwf_tmax": rng.normal(70, 3, n_train),
        "ecmwf_diurnal_range": np.where(np.arange(n_train) < 150, 0.0, 15.0),
        "gefs_tmax_mean": rng.normal(71, 3, n_train),
    })
    test_df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=10, freq="D").date,
        "station": "KNYC",
        "lead_hour": 24,
        "actual_tmax": rng.normal(72, 4, 10),
        "ecmwf_tmax": rng.normal(70, 3, 10),
        "ecmwf_diurnal_range": np.full(10, 15.0),
        "gefs_tmax_mean": rng.normal(71, 3, 10),
    })

    captured_X_trains = []

    original_fit_qrf = runner._fit_qrf_and_blend

    def capturing_fit(X_train, y_train, X_test, **kw):
        captured_X_trains.append(X_train.copy())
        return original_fit_qrf(X_train, y_train, X_test, **kw)

    def fake_load(start, end):
        if end < test_month:
            return train_df
        return test_df

    with patch.object(runner, "_load_historical_features", side_effect=fake_load), \
         patch.object(runner, "_fit_qrf_and_blend", side_effect=capturing_fit):
        try:
            runner._run_fold(train_start, train_end, test_month)
        except Exception:
            pass

    assert captured_X_trains, "Expected _fit_qrf_and_blend to be called"
    X_used = captured_X_trains[0]
    assert len(X_used) <= 150, (
        f"Zero diurnal_range rows should be excluded; got {len(X_used)} rows, expected ≤150"
    )


# ── Forecast noise includes ecmwf_diurnal_range ───────────────────────────────

def test_add_forecast_noise_perturbs_ecmwf_diurnal_range():
    """ecmwf_diurnal_range is ECMWF-derived; it must receive noise like other ECMWF features."""
    runner = _make_runner()
    rng = np.random.default_rng(1)
    n = 20
    dates = pd.date_range("2023-06-01", periods=n, freq="D").date
    meta = pd.DataFrame({
        "date": dates,
        "station": "KNYC",
        "lead_hour": 24,
    })
    avail_cols = ["gefs_tmax_mean", "ecmwf_diurnal_range"]
    X = pd.DataFrame({
        "gefs_tmax_mean": rng.normal(72, 3, n),
        "ecmwf_diurnal_range": np.full(n, 15.0),
    })

    X_noisy = runner._add_forecast_noise(X, meta, avail_cols)

    assert not np.allclose(X_noisy["ecmwf_diurnal_range"].values, 15.0), (
        "ecmwf_diurnal_range should be perturbed by forecast noise"
    )


# ── Residual sigma floor ──────────────────────────────────────────────────────

def test_residual_sigma_floor_constant_is_at_least_two_degrees():
    """Floor prevents overconfident Kelly sizing after residual reframing shrinks sigma."""
    assert RESIDUAL_SIGMA_FLOOR >= 2.0


def test_residual_sigma_floor_matches_inference_floor():
    """Backtest and inference must use the same floor to keep calibration consistent."""
    from strategies.ensemble_strategy import RESIDUAL_SIGMA_FLOOR as INFERENCE_FLOOR
    assert RESIDUAL_SIGMA_FLOOR == INFERENCE_FLOOR
