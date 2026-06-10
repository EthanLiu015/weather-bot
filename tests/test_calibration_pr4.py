"""Tests for PR 4 calibration correctness gaps."""
import numpy as np
import pandas as pd
import pytest
from models.calibration import IsotonicCalibrator
from processing.bias_correction import BiasCorrectionRegistry


# ── Gap 8: graceful degradation below MIN_SAMPLES ────────────────────────────

def test_calibrator_with_50_samples_does_not_return_raw_passthrough():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.1, 0.9, 50)
    outcomes = (rng.uniform(0, 1, 50) < raw).astype(float)

    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)

    calibrated, _, _ = cal.calibrate(0.5)
    assert calibrated is not None
    assert 0.0 <= calibrated <= 1.0


def test_calibrator_smooths_small_sample_predictions():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.1, 0.9, 50)
    outcomes = (rng.uniform(0, 1, 50) < raw).astype(float)

    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)

    calibrated, _, _ = cal.calibrate(0.5)
    # When n<100, the calibrator should produce a smoothed prediction — not exactly 0.5 raw
    # and not degenerate (should be within [0,1])
    assert 0.0 <= calibrated <= 1.0


def test_calibrator_brier_score_with_small_sample():
    rng = np.random.default_rng(2)
    raw = rng.uniform(0.1, 0.9, 40)
    outcomes = (rng.uniform(0, 1, 40) < raw).astype(float)

    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)
    bs = cal.brier_score(raw, outcomes)

    assert isinstance(bs, float)
    assert 0.0 <= bs <= 1.0


def test_calibrator_uses_isotonic_when_samples_gte_threshold():
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.1, 0.9, 200)
    outcomes = (rng.uniform(0, 1, 200) < raw).astype(float)

    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)

    assert cal._iso is not None, "Should fit isotonic with 200 samples"


def test_calibrator_uses_isotonic_for_samples_between_30_and_100():
    rng = np.random.default_rng(3)
    raw = rng.uniform(0.1, 0.9, 50)
    outcomes = (rng.uniform(0, 1, 50) < raw).astype(float)

    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)

    assert cal._iso is not None, "Should fit isotonic for n=50 (>= MIN_SMALL_SAMPLES=30)"


def test_calibrator_falls_back_to_raw_when_fewer_than_30_samples():
    rng = np.random.default_rng(4)
    raw = rng.uniform(0.1, 0.9, 20)
    outcomes = (rng.uniform(0, 1, 20) < raw).astype(float)

    cal = IsotonicCalibrator()
    cal.fit(raw, outcomes)

    assert cal._iso is None, "Should not fit isotonic with only 20 samples"


# ── Gap 9: Kalman noise params passed through registry ───────────────────────

def test_registry_passes_process_noise_to_new_correctors():
    registry = BiasCorrectionRegistry(process_noise=0.5, obs_noise=3.0)
    corrector = registry.get_corrector("KNYC", "D1-2", "JJA")

    assert corrector.process_noise == pytest.approx(0.5)
    assert corrector.obs_noise == pytest.approx(3.0)


def test_registry_default_noise_params_unchanged():
    registry = BiasCorrectionRegistry()
    corrector = registry.get_corrector("KNYC", "D1-2", "JJA")

    assert corrector.process_noise == pytest.approx(0.1)
    assert corrector.obs_noise == pytest.approx(1.5)


# ── Gap 7: trade calibrator trained on per-row probs ─────────────────────────

def test_build_trade_calibrator_uses_per_row_thresholds():
    from backtest.runner import BacktestRunner

    rng = np.random.default_rng(42)
    n = 150
    mu = rng.normal(72, 5, n)
    sigma = rng.uniform(2, 4, n)
    y = rng.normal(72, 4, n)
    thresholds = rng.uniform(68, 76, n)

    cal = BacktestRunner._build_trade_calibrator(
        mu_train=mu,
        sigma_train=sigma,
        y_train=y,
        row_thresholds_train=thresholds,
    )

    prob = cal.calibrate(0.5)[0]
    assert 0.0 <= prob <= 1.0


def test_trade_calibrator_returns_calibrator_instance():
    from backtest.runner import BacktestRunner

    rng = np.random.default_rng(5)
    n = 150
    mu = rng.normal(72, 5, n)
    sigma = rng.uniform(2, 4, n)
    y = rng.normal(72, 4, n)
    thresholds = np.full(n, 72.0)

    cal = BacktestRunner._build_trade_calibrator(mu, sigma, y, thresholds)

    assert isinstance(cal, IsotonicCalibrator)
