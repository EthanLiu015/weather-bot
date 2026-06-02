import numpy as np
import pandas as pd
import properscoring as ps


def compute_crps(forecasts: np.ndarray, observations: np.ndarray) -> float:
    # Drop any rows where either value is NaN before scoring
    mask = np.isfinite(observations) & np.all(np.isfinite(forecasts.reshape(len(observations), -1)), axis=1)
    if mask.sum() < 2:
        return float("nan")
    obs_clean = observations[mask]
    fc_clean = forecasts[mask] if forecasts.ndim == 1 else forecasts[mask]
    if fc_clean.ndim == 1:
        fc_clean = fc_clean.reshape(-1, 1)
    scores = ps.crps_ensemble(obs_clean, fc_clean)
    return float(np.nanmean(scores))


def compute_mae(predictions: np.ndarray, observations: np.ndarray) -> float:
    mask = np.isfinite(predictions) & np.isfinite(observations)
    if mask.sum() < 2:
        return float("nan")
    return float(np.mean(np.abs(predictions[mask] - observations[mask])))


def compute_brier_score(prob_forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    mask = np.isfinite(prob_forecasts) & np.isfinite(outcomes)
    if mask.sum() < 2:
        return float("nan")
    return float(np.mean((prob_forecasts[mask] - outcomes[mask]) ** 2))


def compute_reliability_slope(
    prob_forecasts: np.ndarray,
    outcomes: np.ndarray,
    n_bins: int = 10,
) -> float:
    mask = np.isfinite(prob_forecasts) & np.isfinite(outcomes)
    if mask.sum() < 10:
        return float("nan")
    prob_forecasts = prob_forecasts[mask]
    outcomes = outcomes[mask]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_freqs = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (prob_forecasts >= lo) & (prob_forecasts < hi)
        if m.sum() > 0:
            bin_centers.append(prob_forecasts[m].mean())
            bin_freqs.append(outcomes[m].mean())
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
        "sharpness": float(np.nanmean(sigma_forecasts)),
    }
