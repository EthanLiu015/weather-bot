import pytest
from trading.kelly import compute_size

HORIZON_MULTIPLIERS = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.2}


def test_zero_contracts_when_no_edge():
    contracts = compute_size(
        fair_value=0.50,
        market_price=0.50,
        ci_width=0.05,
        horizon_days=1,
        kelly_fraction=0.25,
        max_exposure_usd=200.0,
        horizon_multipliers=HORIZON_MULTIPLIERS,
        strategy_lock=False,
    )
    assert contracts == 0


def test_positive_contracts_with_clear_edge():
    contracts = compute_size(
        fair_value=0.70,
        market_price=0.50,
        ci_width=0.05,
        horizon_days=1,
        kelly_fraction=0.25,
        max_exposure_usd=200.0,
        horizon_multipliers=HORIZON_MULTIPLIERS,
        strategy_lock=False,
    )
    assert contracts > 0


def test_contracts_respect_max_exposure_cap():
    contracts = compute_size(
        fair_value=0.95,
        market_price=0.50,
        ci_width=0.0,
        horizon_days=1,
        kelly_fraction=1.0,
        max_exposure_usd=100.0,
        horizon_multipliers=HORIZON_MULTIPLIERS,
        strategy_lock=False,
    )
    max_allowed = int(100.0 / 0.50)
    assert contracts <= max_allowed


def test_strategy_lock_halves_size():
    size_normal = compute_size(
        fair_value=0.70,
        market_price=0.50,
        ci_width=0.05,
        horizon_days=1,
        kelly_fraction=0.25,
        max_exposure_usd=200.0,
        horizon_multipliers=HORIZON_MULTIPLIERS,
        strategy_lock=False,
    )
    size_locked = compute_size(
        fair_value=0.70,
        market_price=0.50,
        ci_width=0.05,
        horizon_days=1,
        kelly_fraction=0.25,
        max_exposure_usd=200.0,
        horizon_multipliers=HORIZON_MULTIPLIERS,
        strategy_lock=True,
    )
    assert size_locked <= size_normal


def test_wide_ci_reduces_size():
    size_narrow = compute_size(
        fair_value=0.70,
        market_price=0.50,
        ci_width=0.02,
        horizon_days=1,
        kelly_fraction=0.25,
        max_exposure_usd=200.0,
        horizon_multipliers=HORIZON_MULTIPLIERS,
        strategy_lock=False,
    )
    size_wide = compute_size(
        fair_value=0.70,
        market_price=0.50,
        ci_width=0.30,
        horizon_days=1,
        kelly_fraction=0.25,
        max_exposure_usd=200.0,
        horizon_multipliers=HORIZON_MULTIPLIERS,
        strategy_lock=False,
    )
    assert size_wide <= size_narrow


def test_zero_contracts_on_adverse_edge():
    contracts = compute_size(
        fair_value=0.20,
        market_price=0.80,
        ci_width=0.05,
        horizon_days=1,
        kelly_fraction=0.25,
        max_exposure_usd=200.0,
        horizon_multipliers=HORIZON_MULTIPLIERS,
        strategy_lock=False,
    )
    assert contracts == 0
