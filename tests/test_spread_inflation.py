import numpy as np
import pytest
from models.spread_inflation import (
    apply_spread_inflation_from_stats,
    compute_ensemble_agreement_from_stats,
)


def test_agreement_is_low_when_ensemble_is_spread_out():
    agreement = compute_ensemble_agreement_from_stats(std=3.0, range_=10.0)
    assert agreement < 0.95


def test_agreement_is_high_when_ensemble_is_tightly_clustered():
    agreement = compute_ensemble_agreement_from_stats(std=0.1, range_=10.0)
    assert agreement > 0.95


def test_sigma_inflated_when_ensemble_tightly_clustered():
    sigma = np.array([2.0, 3.0])
    std_arr = np.array([0.1, 0.1])
    range_arr = np.array([10.0, 10.0])
    _, inflated = apply_spread_inflation_from_stats(
        mu=np.array([70.0, 71.0]),
        sigma=sigma,
        std_arr=std_arr,
        range_arr=range_arr,
        threshold=0.95,
        inflation_factor=1.5,
    )
    assert inflated[0] == pytest.approx(2.0 * 1.5)
    assert inflated[1] == pytest.approx(3.0 * 1.5)


def test_sigma_unchanged_when_ensemble_is_spread_out():
    sigma = np.array([2.0])
    std_arr = np.array([3.0])
    range_arr = np.array([10.0])
    _, inflated = apply_spread_inflation_from_stats(
        mu=np.array([70.0]),
        sigma=sigma,
        std_arr=std_arr,
        range_arr=range_arr,
        threshold=0.95,
        inflation_factor=1.5,
    )
    assert inflated[0] == pytest.approx(2.0)


def test_sigma_never_decreased_by_inflation():
    sigma = np.array([2.0, 4.0])
    std_arr = np.array([0.05, 3.0])
    range_arr = np.array([5.0, 10.0])
    _, inflated = apply_spread_inflation_from_stats(
        mu=np.array([70.0, 71.0]),
        sigma=sigma,
        std_arr=std_arr,
        range_arr=range_arr,
    )
    assert (inflated >= sigma).all(), "Inflation must never reduce sigma"


def test_returns_copy_not_in_place():
    sigma = np.array([2.0])
    _, inflated = apply_spread_inflation_from_stats(
        mu=np.array([70.0]),
        sigma=sigma,
        std_arr=np.array([0.05]),
        range_arr=np.array([5.0]),
    )
    assert inflated is not sigma
