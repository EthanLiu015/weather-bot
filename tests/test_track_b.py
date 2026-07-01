"""Tests for the Kalshi fee model.

Real Kalshi trading fee is `coef * C * p * (1-p)` per order (taker coef 0.07,
maker coef 0.0175), capped at $0.035/contract — symmetric in p (same fee for a
YES at p or a NO at 1-p). The old model booked `0.05 * size * p`, which is
asymmetric and wrong; these pin the corrected formula.
"""
import pytest

from backtest.track_b import kalshi_fee, TAKER_FEE_COEF, MAKER_FEE_COEF


def test_taker_fee_at_fifty_cents_is_max_1_75_cents():
    # 0.07 * 1 * 0.5 * 0.5 = 0.0175
    assert kalshi_fee(size=1.0, price=0.5) == pytest.approx(0.0175)


def test_maker_fee_is_a_quarter_of_taker():
    taker = kalshi_fee(size=1.0, price=0.5, fee_coef=TAKER_FEE_COEF)
    maker = kalshi_fee(size=1.0, price=0.5, fee_coef=MAKER_FEE_COEF)
    assert maker == pytest.approx(taker / 4.0)


def test_fee_is_symmetric_in_price():
    # A YES at 0.2 and a NO at 0.2 (=YES at 0.8) pay the same fee.
    assert kalshi_fee(1.0, 0.2) == pytest.approx(kalshi_fee(1.0, 0.8))


def test_fee_scales_linearly_with_size():
    assert kalshi_fee(10.0, 0.3) == pytest.approx(10.0 * kalshi_fee(1.0, 0.3))


def test_fee_capped_at_035_per_contract():
    # A hypothetical high coef would exceed the cap at p=0.5; cap binds.
    assert kalshi_fee(1.0, 0.5, fee_coef=0.2) == pytest.approx(0.035)


def test_general_taker_formula_never_hits_cap():
    # 0.07 formula peaks at 0.0175 < 0.035, so the cap never binds for it.
    assert kalshi_fee(1.0, 0.5, fee_coef=TAKER_FEE_COEF) < 0.035
