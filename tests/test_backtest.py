"""Tests for the round-trip MM backtest."""
import numpy as np
import pytest

from bot.research.backtest import backtest_ticker
from bot.research.mm_edge import maker_fee_cents


def test_clean_round_trip_captures_spread_minus_fees():
    # Quote 40/42 (empty queues). Bid filled for 10 (buy@40), then ask filled for
    # 10 (sell@42): flat, captured the 2c spread on 10 contracts minus maker fees.
    events = [("q", 40, 0, 42, 0), ("x", 40, "no", 10), ("x", 42, "yes", 10)]
    pnl, traded, fees = backtest_ticker(events, q=10, max_inv=50, phi=0.0)
    f = 10 * maker_fee_cents(np.array([40.0]))[0] + 10 * maker_fee_cents(np.array([42.0]))[0]
    assert traded == 20
    assert pnl == pytest.approx(20.0 - f)        # +20c gross (10 x 2c) - fees
    assert fees == pytest.approx(f)


def test_forced_flatten_on_inventory_limit():
    # One-sided fills push inventory past the limit -> forced cross-flatten -> flat,
    # and we just lose fees (bought and flattened at the same price).
    events = [("q", 40, 0, 42, 0), ("x", 40, "no", 10), ("x", 40, "no", 10)]
    pnl, traded, fees = backtest_ticker(events, q=10, max_inv=15, phi=0.0)
    assert traded == 20      # 20 bought
    assert pnl < 0           # no favourable move -> fees-only loss
    assert fees > 0
