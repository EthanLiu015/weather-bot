from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtest.runner import BacktestRunner


def _make_settings():
    return SimpleNamespace(MIN_EDGE_CENTS=5, STATIONS=["KNYC"])


def _make_runner():
    with patch.object(BacktestRunner, "_load_kalshi_prices", return_value=pd.DataFrame()):
        return BacktestRunner(
            settings=_make_settings(),
            start_date=date(2020, 1, 1),
            end_date=date(2021, 1, 1),
        )


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
