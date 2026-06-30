"""Tests for the exit-cost net formulas."""
import numpy as np
import pandas as pd
import pytest

from bot.research.exit_cost import exit_nets, taker_fee_cents
from bot.research.mm_edge import maker_fee_cents


def test_taker_fee_is_four_times_maker():
    p = np.array([40.0])
    assert taker_fee_cents(p)[0] == pytest.approx(4 * maker_fee_cents(p)[0])


def test_exit_scenarios_for_a_long_fill():
    # Long yes (taker bought NO), entered @40c. At +5s: bid 41, ask 43, mid 42.
    t = pd.DataFrame({"sign": [1.0], "price_c": [40.0],
                      "bid_5": [41.0], "ask_5": [43.0], "mid_5": [42.0], "count": [1.0]})
    m, p, a, r = exit_nets(t, 5)
    f40, f43 = maker_fee_cents(np.array([40.0]))[0], maker_fee_cents(np.array([43.0]))[0]
    tk41 = taker_fee_cents(np.array([41.0]))[0]
    assert m[0] == pytest.approx(2.0 - f40)                  # mark at mid
    assert p[0] == pytest.approx(3.0 - f40 - f43)            # sell at ask (capture spread)
    assert a[0] == pytest.approx(1.0 - f40 - tk41)           # cross: sell at bid + taker fee
    assert r[0] == pytest.approx(p[0])                       # mid>0 -> passive


def test_realistic_uses_aggressive_when_move_is_adverse():
    # Long yes @40, but price fell: bid 36, ask 38, mid 37 -> adverse -> forced cross.
    t = pd.DataFrame({"sign": [1.0], "price_c": [40.0],
                      "bid_5": [36.0], "ask_5": [38.0], "mid_5": [37.0], "count": [1.0]})
    m, p, a, r = exit_nets(t, 5)
    assert m[0] < 0 and r[0] == pytest.approx(a[0])          # adverse -> aggressive
