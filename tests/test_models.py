import numpy as np
import pandas as pd
import pytest
from models.ngboost_model import NGBoostTemperatureModel
from models.calibration import IsotonicCalibrator


def _make_data(n: int = 200) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({
        "gefs_tmax_mean": rng.normal(72, 5, n),
        "gefs_tmax_std": rng.uniform(1, 4, n),
        "lead_time_hours": rng.integers(24, 120, n).astype(float),
    })
    y = pd.Series(X["gefs_tmax_mean"] + rng.normal(0, 2, n), name="actual_tmax")
    return X, y


def test_ngboost_sigma_positive():
    X, y = _make_data()
    model = NGBoostTemperatureModel(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    mu, sigma = model.predict_distribution(X)
    assert (sigma > 0).all(), "All sigma values must be positive"


def test_ngboost_prob_above_in_unit_interval():
    X, y = _make_data()
    model = NGBoostTemperatureModel(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    probs = model.predict_prob_above(X, threshold=72.0)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_ngboost_full_cdf_columns():
    X, y = _make_data()
    model = NGBoostTemperatureModel(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    thresholds = [65.0, 70.0, 75.0, 80.0]
    cdf_df = model.predict_full_cdf(X, thresholds)
    assert len(cdf_df.columns) == len(thresholds)


def test_calibration_ci_covers_true_rate():
    rng = np.random.default_rng(42)
    raw_probs = rng.uniform(0.1, 0.9, 300)
    outcomes = (rng.uniform(0, 1, 300) < raw_probs).astype(float)

    cal = IsotonicCalibrator()
    cal.fit(raw_probs, outcomes)

    in_ci = 0
    n_test = 50
    test_probs = rng.uniform(0.2, 0.8, n_test)
    true_rates = test_probs

    for p, true_p in zip(test_probs, true_rates):
        _, ci_lo, ci_hi = cal.calibrate(p)
        if ci_lo <= true_p <= ci_hi:
            in_ci += 1

    coverage = in_ci / n_test
    assert coverage >= 0.5, f"CI coverage {coverage:.2f} too low"


def test_calibration_brier_score_is_float():
    rng = np.random.default_rng(1)
    raw = rng.uniform(0, 1, 200)
    out = (rng.uniform(0, 1, 200) < raw).astype(float)
    cal = IsotonicCalibrator()
    cal.fit(raw, out)
    bs = cal.brier_score(raw, out)
    assert isinstance(bs, float)
    assert 0.0 <= bs <= 1.0
