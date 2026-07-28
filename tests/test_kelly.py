import numpy as np
import pytest
from trading.kelly import compute_size
from backtest.track_b import simulate_pnl

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


# ── Fee-adjusted Kelly ────────────────────────────────────────────────────────

def test_kelly_size_is_smaller_with_fee_than_without():
    """The Kalshi fee reduces net win payoff, so the fee-adjusted Kelly size must
    be strictly smaller than the naive no-fee Kelly size for the same edge."""
    p, price = 0.65, 0.50
    fee_adjusted = compute_size(
        fair_value=p,
        market_price=price,
        ci_width=0.0,
        horizon_days=1,
        kelly_fraction=1.0,
        max_exposure_usd=200.0,
        horizon_multipliers={1: 1.0},
        strategy_lock=False,
    )
    # Naive no-fee Kelly contracts for the same inputs: b = (1-price)/price.
    b = (1.0 - price) / price
    kelly_pct = (b * p - (1.0 - p)) / b
    no_fee = int((kelly_pct * 200.0) / price)

    assert fee_adjusted > 0
    assert fee_adjusted < no_fee


# ── Kelly-sized simulate_pnl ──────────────────────────────────────────────────

def test_simulate_pnl_accepts_contract_sizes_array():
    """simulate_pnl must accept an optional per-row contract_sizes array."""
    n = 10
    model_probs = np.full(n, 0.65)
    market_mids = np.full(n, 0.50)
    outcomes = np.ones(n)
    contract_sizes = np.full(n, 2.0)

    result = simulate_pnl(
        model_probs=model_probs,
        market_mids=market_mids,
        outcomes=outcomes,
        min_edge=0.10,
        contract_sizes=contract_sizes,
    )

    assert isinstance(result, dict)
    assert result["num_simulated_trades"] == n


def test_simulate_pnl_contract_sizes_scales_pnl_proportionally():
    """Doubling all contract_sizes must double total P&L (linearity)."""
    n = 10
    model_probs = np.full(n, 0.70)
    market_mids = np.full(n, 0.50)
    outcomes = np.ones(n)

    result_1x = simulate_pnl(model_probs, market_mids, outcomes, min_edge=0.10,
                              contract_sizes=np.ones(n))
    result_2x = simulate_pnl(model_probs, market_mids, outcomes, min_edge=0.10,
                              contract_sizes=np.full(n, 2.0))

    assert result_2x["simulated_pnl_usd"] == pytest.approx(
        2.0 * result_1x["simulated_pnl_usd"], rel=1e-6
    )


def test_simulate_pnl_unit_contract_sizes_matches_default():
    """contract_sizes=np.ones(n) must give the same result as default contract_usd=1.0."""
    n = 10
    model_probs = np.array([0.70, 0.30, 0.65, 0.35, 0.75, 0.25, 0.60, 0.40, 0.72, 0.28])
    market_mids = np.full(n, 0.50)
    outcomes = (np.arange(n) % 2).astype(float)

    result_default = simulate_pnl(model_probs, market_mids, outcomes, min_edge=0.05)
    result_ones = simulate_pnl(model_probs, market_mids, outcomes, min_edge=0.05,
                               contract_sizes=np.ones(n))

    assert result_default["simulated_pnl_usd"] == pytest.approx(
        result_ones["simulated_pnl_usd"], rel=1e-6
    )
    assert result_default["num_simulated_trades"] == result_ones["num_simulated_trades"]


def test_kelly_returns_zero_when_fee_makes_bet_unprofitable():
    """A thin edge (fair 0.905 vs price 0.90) is smaller than the Kalshi fee eats,
    so fee-adjusted Kelly is non-positive → compute_size must return 0."""
    contracts = compute_size(
        fair_value=0.905,
        market_price=0.90,
        ci_width=0.0,
        horizon_days=1,
        kelly_fraction=1.0,
        max_exposure_usd=200.0,
        horizon_multipliers={1: 1.0},
        strategy_lock=False,
    )
    assert contracts == 0, (
        "A sub-fee edge should net zero contracts after the fee-adjusted Kelly"
    )
