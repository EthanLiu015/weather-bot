"""Tests for live prediction path: residual correction and spread inflation."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from unittest.mock import MagicMock

from strategies.ensemble_strategy import EnsembleStrategy, RESIDUAL_SIGMA_FLOOR


def _make_X(gefs_std: float = 3.0, gefs_range: float = 10.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "gefs_tmax_mean": 72.0,
        "gefs_tmax_std": gefs_std,
        "gefs_tmax_range": gefs_range,
        "gefs_tmax_p10": 68.0,
        "gefs_tmax_p90": 78.0,
        "lead_time_hours": 24.0,
    }])


def _make_ngboost(mu: float = 72.0, sigma: float = 3.0, prob: float = 0.4):
    m = MagicMock()
    m.predict_distribution.return_value = (np.array([mu]), np.array([sigma]))
    m.predict_prob_above.return_value = np.array([prob])
    return m


def _make_residual(correction: float = 2.0):
    m = MagicMock()
    m.predict.return_value = np.array([correction])
    return m


# ── Residual correction applied to mu ────────────────────────────────────────

def test_compute_fair_value_applies_residual_correction_to_mu():
    ngb = _make_ngboost(mu=70.0, sigma=3.0)
    residual = _make_residual(correction=2.0)
    threshold = 72.0
    X = _make_X()

    result = EnsembleStrategy._compute_fair_value(
        X=X,
        ngboost_model=ngb,
        qrf_model=None,
        residual_model=residual,
        blender=None,
        calibrator=None,
        threshold=threshold,
    )

    # With mu=70+2=72, sigma=3, threshold=72 → P(T>72) ≈ 0.5
    assert result["raw_prob"] == pytest.approx(0.5, abs=0.02)


def test_compute_fair_value_skips_residual_correction_when_model_is_none():
    ngb = _make_ngboost(mu=70.0, sigma=3.0)
    threshold = 72.0
    X = _make_X()

    result_without = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb, qrf_model=None,
        residual_model=None, blender=None, calibrator=None, threshold=threshold,
    )
    result_with = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb, qrf_model=None,
        residual_model=_make_residual(2.0), blender=None, calibrator=None, threshold=threshold,
    )

    assert result_without["raw_prob"] != pytest.approx(result_with["raw_prob"], abs=0.01)


# ── Spread inflation widens sigma when ensemble is tightly clustered ──────────

def test_compute_fair_value_inflates_sigma_when_ensemble_clustered():
    ngb_tight = _make_ngboost(mu=72.0, sigma=2.0)
    ngb_spread = _make_ngboost(mu=72.0, sigma=2.0)
    threshold = 76.0

    X_tight = _make_X(gefs_std=0.05, gefs_range=5.0)   # high agreement → inflate
    X_spread = _make_X(gefs_std=3.0, gefs_range=10.0)  # low agreement → no inflate

    r_tight = EnsembleStrategy._compute_fair_value(
        X=X_tight, ngboost_model=ngb_tight, qrf_model=None,
        residual_model=None, blender=None, calibrator=None, threshold=threshold,
    )
    r_spread = EnsembleStrategy._compute_fair_value(
        X=X_spread, ngboost_model=ngb_spread, qrf_model=None,
        residual_model=None, blender=None, calibrator=None, threshold=threshold,
    )

    # Wider sigma → probability closer to 0.5 for threshold above mu
    # tight ensemble inflated → prob closer to 0.5 than spread ensemble
    assert r_tight["raw_prob"] > r_spread["raw_prob"], (
        "Inflated sigma for tight ensemble should push prob toward 0.5 (above threshold)"
    )


# ── Calibrator applied when provided ─────────────────────────────────────────

def test_compute_fair_value_applies_calibrator():
    ngb = _make_ngboost(mu=72.0, sigma=3.0)
    X = _make_X()

    calibrator = MagicMock()
    calibrator.calibrate.return_value = (0.65, 0.60, 0.70)

    result = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb, qrf_model=None,
        residual_model=None, blender=None,
        calibrator=calibrator, threshold=72.0,
    )

    assert result["cal_prob"] == pytest.approx(0.65)
    assert result["ci_width"] == pytest.approx(0.10, abs=0.01)


def test_compute_fair_value_returns_raw_prob_when_no_calibrator():
    ngb = _make_ngboost(mu=72.0, sigma=3.0)
    X = _make_X()

    result = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb, qrf_model=None,
        residual_model=None, blender=None,
        calibrator=None, threshold=72.0,
    )

    assert result["cal_prob"] == pytest.approx(result["raw_prob"], abs=1e-6)


# ── ecmwf_offset shifts residual predictions to absolute temperature scale ────

def test_ecmwf_offset_shifts_probability_to_absolute_scale():
    """NGBoost predicts correction ~0; ecmwf_offset converts it to absolute scale.

    If NGBoost predicts residual mu=5.0 and ecmwf_offset=80.0 → absolute mu=85.0.
    P(tmax > 85 | mu=85, sigma=3) ≈ 0.5, matching a zero-offset model at threshold=5.
    """
    ngb_residual = _make_ngboost(mu=5.0, sigma=3.0)   # correction model
    ngb_absolute = _make_ngboost(mu=85.0, sigma=3.0)  # reference absolute model
    threshold = 85.0
    ecmwf_tmax = 80.0
    X = _make_X()

    result_with_offset = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb_residual, qrf_model=None,
        residual_model=None, blender=None, calibrator=None,
        threshold=threshold, ecmwf_offset=ecmwf_tmax,
    )
    result_absolute = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb_absolute, qrf_model=None,
        residual_model=None, blender=None, calibrator=None,
        threshold=threshold, ecmwf_offset=0.0,
    )

    assert result_with_offset["raw_prob"] == pytest.approx(result_absolute["raw_prob"], abs=0.001)


def test_compute_fair_value_enforces_sigma_floor():
    """Residual models can produce unrealistically tight sigma; floor prevents it."""
    ngb_tight = _make_ngboost(mu=0.0, sigma=0.1)   # sigma << floor
    ngb_floor = _make_ngboost(mu=0.0, sigma=RESIDUAL_SIGMA_FLOOR)
    X = _make_X()
    threshold = 2.0

    result_tight = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb_tight, qrf_model=None,
        residual_model=None, blender=None, calibrator=None, threshold=threshold,
    )
    result_floor = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb_floor, qrf_model=None,
        residual_model=None, blender=None, calibrator=None, threshold=threshold,
    )

    assert result_tight["raw_prob"] == pytest.approx(result_floor["raw_prob"], abs=0.01), (
        "sigma=0.1 should be clipped to RESIDUAL_SIGMA_FLOOR before probability computation"
    )


def test_residual_sigma_floor_is_at_least_two_degrees():
    assert RESIDUAL_SIGMA_FLOOR >= 2.0, "Floor must be ≥ 2°F to prevent overconfident Kelly sizing"


def test_ecmwf_offset_zero_preserves_existing_behaviour():
    """ecmwf_offset=0.0 must not change the output for backward compatibility."""
    ngb = _make_ngboost(mu=72.0, sigma=3.0)
    X = _make_X()

    result_default = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb, qrf_model=None,
        residual_model=None, blender=None, calibrator=None, threshold=72.0,
    )
    result_explicit_zero = EnsembleStrategy._compute_fair_value(
        X=X, ngboost_model=ngb, qrf_model=None,
        residual_model=None, blender=None, calibrator=None,
        threshold=72.0, ecmwf_offset=0.0,
    )

    assert result_default["raw_prob"] == pytest.approx(result_explicit_zero["raw_prob"], abs=1e-6)
