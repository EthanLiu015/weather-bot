"""Tests for the production blend-weight pipeline:
- NGBoost exposes a CRPS metric comparable to QRF's log_score (CRPS)
- ModelBlender weights can be persisted and reloaded
- _fit_qrf_and_blend passes scores to ModelBlender with the correct sign
  (compute_weights_from_log_scores treats higher = better, but CRPS is
  lower = better, so callers must negate before passing it in)
"""
import numpy as np
import pandas as pd
import pytest

from models.blend import ModelBlender, windowed_mean_score
from models.ngboost_model import NGBoostTemperatureModel
from models.qrf_model import QRFTemperatureModel


def _make_xy(n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "gefs_tmax_mean": rng.normal(72, 5, n),
        "gefs_tmax_std": rng.uniform(1, 4, n),
        "lead_time_hours": rng.choice([24, 48, 72], n).astype(float),
    })
    y = pd.Series(X["gefs_tmax_mean"] + rng.normal(0, 1, n), name="actual_tmax")
    return X, y


# ── NGBoost CRPS, comparable to QRF.log_score (CRPS) ─────────────────────────

def test_ngboost_crps_returns_nonnegative_float():
    X, y = _make_xy()
    model = NGBoostTemperatureModel(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)
    crps = model.crps(X, y)
    assert isinstance(crps, float)
    assert crps >= 0.0


# ── ModelBlender persistence ─────────────────────────────────────────────────

def test_model_blender_save_load_roundtrip(tmp_path):
    blender = ModelBlender()
    blender.compute_weights_from_log_scores(ngboost_log_score=-0.3, qrf_log_score=-0.8)

    path = str(tmp_path / "blender.pkl")
    blender.save(path)
    loaded = ModelBlender.load(path)

    assert loaded.weights == pytest.approx(blender.weights)


# ── _fit_qrf_and_blend sign convention ───────────────────────────────────────

def test_fit_qrf_and_blend_favors_model_with_lower_crps():
    from backtest.runner import BacktestRunner

    rng = np.random.default_rng(1)
    n = 300
    # y is essentially a constant (100) plus tiny noise -> QRF can fit it almost
    # perfectly, giving QRF a very low (good) CRPS on the held-out validation slice.
    X_train = pd.DataFrame({
        "x": np.full(n, 1.0) + rng.normal(0, 0.01, n),
    })
    y_train = pd.Series(np.full(n, 100.0) + rng.normal(0, 0.01, n))
    X_test = pd.DataFrame({"x": [1.0]})

    # Deliberately bad NGBoost prediction, far from the truth, with a high
    # (bad) CRPS-style log score.
    ngb_mu = np.array([50.0])
    ngb_sigma = np.array([1.0])
    ngb_log_score = 50.0  # large CRPS == bad

    blended_mu, _ = BacktestRunner._fit_qrf_and_blend(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        ngb_mu=ngb_mu,
        ngb_sigma=ngb_sigma,
        ngb_log_score=ngb_log_score,
        n_estimators=50,
    )

    # QRF (CRPS near 0, much better than NGBoost's 50) should dominate the
    # blend, pulling the result toward ~100 rather than NGBoost's 50.
    assert blended_mu[0] > 75.0, (
        f"Blend should favor the lower-CRPS (QRF) model, got blended_mu={blended_mu[0]}"
    )


# ── windowed_mean_score: multi-window CRPS averaging for blend weights ──────

def test_windowed_mean_score_with_one_window_equals_global_mean():
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    assert windowed_mean_score(scores, n_windows=1) == pytest.approx(scores.mean())


def test_windowed_mean_score_weights_each_window_equally_regardless_of_size():
    # 7 scores split into 3 contiguous windows -> sizes [3, 2, 2]:
    # [0, 0, 0] mean=0, [0, 0] mean=0, [10, 10] mean=10
    scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0])
    global_mean = float(scores.mean())

    windowed = windowed_mean_score(scores, n_windows=3)

    assert windowed == pytest.approx((0.0 + 0.0 + 10.0) / 3)
    assert windowed != pytest.approx(global_mean)


# ── n_windows support on NGBoost.crps / QRF.log_score ────────────────────────

def test_ngboost_crps_default_matches_single_window():
    X, y = _make_xy()
    model = NGBoostTemperatureModel(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)

    assert model.crps(X, y, n_windows=1) == pytest.approx(model.crps(X, y))


def test_ngboost_crps_with_multiple_windows_returns_nonnegative_float():
    X, y = _make_xy()
    model = NGBoostTemperatureModel(n_estimators=50, learning_rate=0.1)
    model.fit(X, y)

    crps = model.crps(X, y, n_windows=3)

    assert isinstance(crps, float)
    assert crps >= 0.0


def test_qrf_log_score_default_matches_single_window():
    X, y = _make_xy()
    model = QRFTemperatureModel(n_estimators=50, min_samples_leaf=5)
    model.fit(X, y)

    assert model.log_score(X, y, n_windows=1) == pytest.approx(model.log_score(X, y))


def test_qrf_log_score_with_multiple_windows_returns_nonnegative_float():
    X, y = _make_xy()
    model = QRFTemperatureModel(n_estimators=50, min_samples_leaf=5)
    model.fit(X, y)

    score = model.log_score(X, y, n_windows=3)

    assert isinstance(score, float)
    assert score >= 0.0
