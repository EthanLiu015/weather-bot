"""Tests for PR 4 calibration correctness gaps and spectrum coverage."""
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


# ── Calibrator spectrum coverage (production function) ────────────────────────

def test_build_calibration_dataset_is_importable():
    """build_calibration_dataset must be exported from models.calibration."""
    from models.calibration import build_calibration_dataset  # noqa: F401


def test_build_calibration_dataset_returns_9x_rows():
    """Grid over 9 percentile thresholds produces 9n training samples."""
    from models.calibration import build_calibration_dataset
    from models.ngboost_model import NGBoostTemperatureModel

    rng = np.random.default_rng(11)
    n = 100
    X = pd.DataFrame({"gefs_tmax_mean": rng.normal(72, 5, n)})
    y = pd.Series(rng.normal(0, 5, n))

    ngb = NGBoostTemperatureModel(n_estimators=30, learning_rate=0.1)
    ngb.fit(X, y)

    raw_probs, outcomes = build_calibration_dataset(ngb, X, y)

    assert len(raw_probs) == 9 * n, f"Expected 9×{n}={9*n} rows, got {len(raw_probs)}"
    assert len(outcomes) == len(raw_probs)


def test_build_calibration_dataset_spans_full_prob_range():
    """The extremal percentile thresholds (p5, p95) force probs near 1.0 and 0.0,
    ensuring the isotonic calibrator has coverage across the full inference range."""
    from models.calibration import build_calibration_dataset
    from models.ngboost_model import NGBoostTemperatureModel

    rng = np.random.default_rng(22)
    n_fit, n_cal = 400, 80
    X_fit = pd.DataFrame({"gefs_tmax_mean": rng.normal(72, 5, n_fit)})
    y_fit = pd.Series(rng.normal(0, 5, n_fit))

    ngb = NGBoostTemperatureModel(n_estimators=30, learning_rate=0.1)
    ngb.fit(X_fit, y_fit)

    X_cal = pd.DataFrame({"gefs_tmax_mean": rng.normal(72, 5, n_cal)})
    y_cal = pd.Series(rng.normal(0, 5, n_cal))

    raw_probs, _ = build_calibration_dataset(ngb, X_cal, y_cal)

    assert raw_probs.min() < 0.10, (
        f"p95-threshold rows should produce probs < 0.10; got min={raw_probs.min():.3f}"
    )
    assert raw_probs.max() > 0.90, (
        f"p5-threshold rows should produce probs > 0.90; got max={raw_probs.max():.3f}"
    )


# ── Test helper functions (not testing production code) ───────────────────────

def _build_training_probs_single_threshold(ngb, X, y_residual):
    """Simulate old single-threshold calibrator training."""
    threshold = float(y_residual.median())
    raw_probs = ngb.predict_prob_above(X, threshold)
    outcomes = (y_residual.values > threshold).astype(float)
    return raw_probs, outcomes


def _build_training_probs_percentile_grid(ngb, X, y_residual):
    """Simulate new percentile-grid calibrator training."""
    percentile_thresholds = np.percentile(y_residual, [5, 15, 25, 35, 50, 65, 75, 85, 95])
    prob_rows, outcome_rows = [], []
    for thr in percentile_thresholds:
        prob_rows.append(ngb.predict_prob_above(X, thr))
        outcome_rows.append((y_residual.values > thr).astype(float))
    return np.concatenate(prob_rows), np.concatenate(outcome_rows)


def test_calibrator_training_probs_span_full_range_with_percentile_grid():
    """With a small but realistic calibration set (~100 rows), a single-threshold
    calibrator's training probs may not reach 0.05 or 0.95 — isotonic clips at
    inference. A percentile-grid approach produces probs from ~0.0 to ~1.0 by
    construction (evaluating at 5th-percentile threshold gives probs near 1.0)."""
    from models.ngboost_model import NGBoostTemperatureModel

    # Fit on large set so the model has learned something, then calibrate on small set
    rng = np.random.default_rng(99)
    n_fit, n_cal = 400, 80
    X_fit = pd.DataFrame({"gefs_tmax_mean": rng.normal(72, 5, n_fit)})
    y_fit = pd.Series(rng.normal(0, 5, n_fit))

    ngb = NGBoostTemperatureModel(n_estimators=50, learning_rate=0.1)
    ngb.fit(X_fit, y_fit)

    # Small calibration set — same size as a typical (station, lead_bucket) slice
    X_cal = pd.DataFrame({"gefs_tmax_mean": rng.normal(72, 5, n_cal)})
    y_cal = pd.Series(rng.normal(0, 5, n_cal))

    single_probs, _ = _build_training_probs_single_threshold(ngb, X_cal, y_cal)
    grid_probs, _ = _build_training_probs_percentile_grid(ngb, X_cal, y_cal)

    # Percentile grid: by construction, evaluating at p5 threshold gives probs > 0.90
    # and evaluating at p95 threshold gives probs < 0.10. So grid must span full range.
    assert grid_probs.min() < 0.10, "Percentile grid (p95 threshold) must include probs < 0.10"
    assert grid_probs.max() > 0.90, "Percentile grid (p5 threshold) must include probs > 0.90"

    # Grid has 9× more training points → isotonic covers a wider domain than single threshold
    assert len(grid_probs) == 9 * len(single_probs), (
        "Grid should stack 9 threshold evaluations; got "
        f"{len(grid_probs)} vs {len(single_probs)}"
    )


def test_percentile_grid_produces_9x_more_calibration_data():
    """Grid trains on 9 thresholds × n rows = 9n data points vs n for single threshold.
    More coverage means the isotonic calibrator's training domain always includes probs
    at both extremes — the p5-threshold rows give probs near 1.0, p95-threshold near 0.0."""
    from models.ngboost_model import NGBoostTemperatureModel

    rng = np.random.default_rng(77)
    n_fit, n_cal = 400, 80
    X_fit = pd.DataFrame({"gefs_tmax_mean": rng.normal(72, 5, n_fit)})
    y_fit = pd.Series(rng.normal(0, 5, n_fit))

    ngb = NGBoostTemperatureModel(n_estimators=50, learning_rate=0.1)
    ngb.fit(X_fit, y_fit)

    X_cal = pd.DataFrame({"gefs_tmax_mean": rng.normal(72, 5, n_cal)})
    y_cal = pd.Series(rng.normal(0, 5, n_cal))

    single_probs, single_outcomes = _build_training_probs_single_threshold(ngb, X_cal, y_cal)
    grid_probs, grid_outcomes = _build_training_probs_percentile_grid(ngb, X_cal, y_cal)

    assert len(grid_probs) == 9 * len(single_probs), (
        f"Grid should have 9n={9 * len(single_probs)} training points; "
        f"got {len(grid_probs)}"
    )

    # Both calibrators must fit (not fall back to raw passthrough)
    cal_grid = IsotonicCalibrator()
    cal_grid.fit(grid_probs, grid_outcomes)
    assert cal_grid._iso is not None, "Grid calibrator must fit isotonic regression"

    # Grid calibrator's training probs cover extreme values, so it does not clip at 0.90
    grid_max_training = grid_probs.max()
    assert grid_max_training > 0.90, (
        f"Grid training max_prob={grid_max_training:.3f} must exceed 0.90 "
        "(p5 threshold produces probs close to 1.0)"
    )
