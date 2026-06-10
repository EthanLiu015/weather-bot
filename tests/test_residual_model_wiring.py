"""Tests that ResidualModel correction is applied in the backtest fold."""
import numpy as np
import pandas as pd
import pytest
from models.residual_model import ResidualModel


def test_residual_model_reduces_systematic_bias():
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame({
        "gefs_tmax_mean": rng.normal(72, 5, n),
        "gefs_tmax_std": rng.uniform(1, 4, n),
        "lead_time_hours": rng.choice([24, 48, 72], n).astype(float),
    })
    y = pd.Series(X["gefs_tmax_mean"] + rng.normal(0, 1, n), name="actual_tmax")

    split = 200
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Intentionally biased baseline: constant prediction (training mean)
    biased_mu_train = np.full(split, float(y_train.mean()))
    biased_mu_test = np.full(n - split, float(y_train.mean()))

    # Residuals are highly correlated with gefs_tmax_mean, which ResidualModel can learn
    residuals_train = y_train.values - biased_mu_train
    res_model = ResidualModel(station="KNYC")
    res_model.fit(X_train, pd.Series(residuals_train))
    corrected_mu_test = biased_mu_test + res_model.predict(X_test)

    mae_before = float(np.mean(np.abs(biased_mu_test - y_test.values)))
    mae_after = float(np.mean(np.abs(corrected_mu_test - y_test.values)))

    assert mae_after < mae_before, (
        f"Residual correction should reduce MAE over constant baseline; "
        f"before={mae_before:.2f}, after={mae_after:.2f}"
    )


def test_residual_model_predict_shape_matches_input():
    rng = np.random.default_rng(7)
    n = 100
    X = pd.DataFrame({
        "gefs_tmax_mean": rng.normal(72, 5, n),
        "gefs_tmax_std": rng.uniform(1, 4, n),
        "lead_time_hours": rng.choice([24, 48, 72], n).astype(float),
    })
    y = pd.Series(rng.normal(0, 2, n))
    model = ResidualModel(station="KORD")
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)


def test_residual_model_predict_selects_training_columns_from_wider_frame():
    """The live feature matrix has many more columns (and a different order)
    than the subset the residual model was trained on. predict() must select
    and reorder to match training, rather than passing the full frame to
    LightGBM (which raises on a feature-count mismatch)."""
    rng = np.random.default_rng(3)
    n = 100
    X_train = pd.DataFrame({
        "obs_minus_model_lag1": rng.normal(0, 1, n),
        "lead_time_hours": rng.choice([24, 48, 72], n).astype(float),
        "month_sin": rng.uniform(-1, 1, n),
    })
    y = pd.Series(rng.normal(0, 1, n))
    model = ResidualModel(station="KORD")
    model.fit(X_train, y)

    # Live frame: extra columns, different order
    X_live = X_train.copy()
    X_live["gefs_tmax_mean"] = rng.normal(72, 5, n)
    X_live = X_live[["gefs_tmax_mean", "month_sin", "lead_time_hours", "obs_minus_model_lag1"]]

    preds_live = model.predict(X_live)
    preds_train = model.predict(X_train)
    np.testing.assert_allclose(preds_live, preds_train)


def test_residual_model_save_load_preserves_column_selection():
    rng = np.random.default_rng(11)
    n = 100
    X_train = pd.DataFrame({
        "obs_minus_model_lag1": rng.normal(0, 1, n),
        "lead_time_hours": rng.choice([24, 48, 72], n).astype(float),
    })
    y = pd.Series(rng.normal(0, 1, n))
    model = ResidualModel(station="KORD")
    model.fit(X_train, y)

    path = "/tmp/test_residual_model_pr7.pkl"
    model.save(path)
    loaded = ResidualModel.load(path)

    X_live = X_train.copy()
    X_live["extra"] = rng.normal(0, 1, n)
    X_live = X_live[["extra", "lead_time_hours", "obs_minus_model_lag1"]]

    preds = loaded.predict(X_live)
    assert preds.shape == (len(X_live),)
