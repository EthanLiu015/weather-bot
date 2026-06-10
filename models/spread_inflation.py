import numpy as np


def compute_ensemble_agreement_from_stats(std: float, range_: float) -> float:
    if range_ == 0 or np.isnan(std) or np.isnan(range_):
        return 1.0
    normalized_spread = std / (range_ + 1e-6)
    return float(np.clip(1.0 - normalized_spread, 0.0, 1.0))


def apply_spread_inflation_from_stats(
    mu: np.ndarray,
    sigma: np.ndarray,
    std_arr: np.ndarray,
    range_arr: np.ndarray,
    threshold: float = 0.95,
    inflation_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    inflated_sigma = sigma.copy()
    for i in range(len(sigma)):
        agreement = compute_ensemble_agreement_from_stats(float(std_arr[i]), float(range_arr[i]))
        if agreement > threshold:
            inflated_sigma[i] = sigma[i] * inflation_factor
    return mu, inflated_sigma


def compute_ensemble_agreement(member_temps: list[float]) -> float:
    if len(member_temps) < 2:
        return 1.0
    std = np.std(member_temps)
    max_range = np.ptp(member_temps)
    if max_range == 0:
        return 1.0
    # Agreement = 1 - normalized spread; high agreement → low spread
    normalized_spread = std / (max_range + 1e-6)
    agreement = 1.0 - normalized_spread
    return float(np.clip(agreement, 0.0, 1.0))


def inflate_sigma(
    sigma: float,
    ensemble_agreement: float,
    threshold: float = 0.95,
    inflation_factor: float = 1.5,
) -> float:
    if ensemble_agreement > threshold:
        return sigma * inflation_factor
    return sigma


def apply_spread_inflation(
    mu: np.ndarray,
    sigma: np.ndarray,
    member_temps_per_row: list[list[float]],
    threshold: float = 0.95,
    inflation_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    inflated_sigma = sigma.copy()
    for i, member_temps in enumerate(member_temps_per_row):
        agreement = compute_ensemble_agreement(member_temps)
        inflated_sigma[i] = inflate_sigma(sigma[i], agreement, threshold, inflation_factor)
    return mu, inflated_sigma
