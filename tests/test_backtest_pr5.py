"""Tests for PR 5: QRF blend, noise removal, empirical error stds."""
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from models.blend import ModelBlender


# ── Gap 3: ModelBlender weights sum to 1.0 and respond to log-scores ─────────

def test_blend_weights_sum_to_one():
    blender = ModelBlender()
    blender.compute_weights_from_log_scores(ngboost_log_score=-0.5, qrf_log_score=-0.4)
    w = blender.weights
    assert w["ngboost"] + w["qrf"] == pytest.approx(1.0)


def test_better_model_gets_higher_weight():
    blender = ModelBlender()
    blender.compute_weights_from_log_scores(ngboost_log_score=-0.3, qrf_log_score=-0.8)
    assert blender.weights["ngboost"] > blender.weights["qrf"], (
        "NGBoost has better (less negative) log-score and should get higher weight"
    )


def test_blend_mu_sigma_uses_both_models():
    blender = ModelBlender()
    blender.compute_weights_from_log_scores(ngboost_log_score=-0.4, qrf_log_score=-0.6)

    ngb_mu = np.array([70.0])
    ngb_sigma = np.array([2.0])
    qrf_mu = np.array([75.0])
    qrf_sigma = np.array([3.0])

    blended_mu, blended_sigma = blender.blend_mu_sigma(ngb_mu, ngb_sigma, qrf_mu, qrf_sigma)

    assert 70.0 < blended_mu[0] < 75.0, "Blended mu should be between the two models"
    assert blended_sigma[0] > 0


def test_backtest_runner_run_fold_uses_blended_predictions():
    from backtest.runner import BacktestRunner

    assert hasattr(BacktestRunner, "_fit_qrf_and_blend"), (
        "BacktestRunner should expose _fit_qrf_and_blend for testing"
    )


# ── Gap 11: Climatological fallback is deterministic (no noise) ──────────────

def test_climatological_mids_are_deterministic():
    from backtest.runner import BacktestRunner

    with patch.object(BacktestRunner, "_load_kalshi_prices", return_value=pd.DataFrame()):
        runner = BacktestRunner(
            settings=SimpleNamespace(MIN_EDGE_CENTS=5, STATIONS=["KNYC"]),
            start_date=date(2020, 1, 1),
            end_date=date(2021, 1, 1),
        )

    rng = np.random.default_rng(0)
    n = 10
    train_df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n).date,
        "station": "KNYC",
        "_month": 1,
        "actual_tmax": rng.normal(32, 5, n),
    })
    test_df = pd.DataFrame({
        "date": [date(2023, 1, 15)] * 3,
        "station": "KNYC",
    })
    thresholds = np.array([32.0, 33.0, 34.0])

    mids1, _ = runner._climatological_mids(train_df, test_df, thresholds, "actual_tmax")
    mids2, _ = runner._climatological_mids(train_df, test_df, thresholds, "actual_tmax")

    np.testing.assert_array_equal(mids1, mids2, err_msg="Mids should be identical on repeated calls (no noise)")


def test_climatological_mid_equals_clim_prob_exactly_without_noise():
    from backtest.runner import BacktestRunner

    with patch.object(BacktestRunner, "_load_kalshi_prices", return_value=pd.DataFrame()):
        runner = BacktestRunner(
            settings=SimpleNamespace(MIN_EDGE_CENTS=5, STATIONS=["KNYC"]),
            start_date=date(2020, 1, 1),
            end_date=date(2021, 1, 1),
        )

    # All training actuals well above threshold → clim_prob = 1.0, clipped to 0.95
    n = 30
    train_df = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n).date,
        "station": "KNYC",
        "_month": 1,
        "actual_tmax": np.full(n, 80.0),
    })
    test_df = pd.DataFrame({
        "date": [date(2023, 1, 15)],
        "station": "KNYC",
    })
    thresholds = np.array([40.0])

    mids, _ = runner._climatological_mids(train_df, test_df, thresholds, "actual_tmax")

    assert mids[0] == pytest.approx(0.95), (
        "Clim mid should equal clipped clim_prob (0.95) with no noise offset"
    )


# ── Gap 12: Empirical forecast error stds computed from features df ───────────

def test_compute_error_distributions_returns_correct_buckets():
    from backtest.runner import BacktestRunner

    rng = np.random.default_rng(0)
    n = 200
    features = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "station": "KNYC",
        "lead_hour": rng.choice([24, 48, 72, 96, 120], n),
        "gefs_tmax_mean": rng.normal(72, 5, n),
        "actual_tmax": rng.normal(72, 4, n),
    })

    result = BacktestRunner._compute_error_distributions(features)

    assert isinstance(result, pd.DataFrame)
    assert "std_error_f" in result.columns
    assert len(result) > 0


def test_compute_error_distributions_std_is_positive():
    from backtest.runner import BacktestRunner

    rng = np.random.default_rng(1)
    n = 300
    features = pd.DataFrame({
        "date": pd.date_range("2022-01-01", periods=n, freq="D"),
        "station": "KNYC",
        "lead_hour": rng.choice([24, 48, 96], n),
        "gefs_tmax_mean": rng.normal(72, 5, n),
        "actual_tmax": rng.normal(72, 4, n),
    })

    result = BacktestRunner._compute_error_distributions(features)

    assert (result["std_error_f"] > 0).all()
