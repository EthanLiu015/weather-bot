import numpy as np
import pandas as pd
import properscoring as ps


def compute_crps(forecasts: np.ndarray, observations: np.ndarray) -> float:
    if len(forecasts.shape) == 1:
        forecasts = forecasts.reshape(-1, 1)
    scores = ps.crps_ensemble(observations, forecasts)
    return float(np.mean(scores))


def compute_mae(predictions: np.ndarray, observations: np.ndarray) -> float:
    return float(np.mean(np.abs(predictions - observations)))


def compute_brier_score(prob_forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((prob_forecasts - outcomes) ** 2))


def compute_reliability_slope(
    prob_forecasts: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_freqs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob_forecasts >= lo) & (prob_forecasts < hi)
        if mask.sum() > 0:
            bin_centers.append(prob_forecasts[mask].mean())
            bin_freqs.append(outcomes[mask].mean())
    if len(bin_centers) < 2:
        return float("nan")
    x = np.array(bin_centers).reshape(-1, 1)
    y = np.array(bin_freqs)
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression().fit(x, y)
    return float(reg.coef_[0])


def track_a_metrics(
    prob_forecasts: np.ndarray,
    mu_forecasts: np.ndarray,
    sigma_forecasts: np.ndarray,
    observations: np.ndarray,
    outcomes: np.ndarray,
) -> dict:
    return {
        "crps": compute_crps(mu_forecasts, observations),
        "mae": compute_mae(mu_forecasts, observations),
        "brier_score": compute_brier_score(prob_forecasts, outcomes),
        "reliability_slope": compute_reliability_slope(prob_forecasts, outcomes),
        "sharpness": float(np.mean(sigma_forecasts)),
    }
