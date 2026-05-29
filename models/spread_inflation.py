import numpy as np


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
