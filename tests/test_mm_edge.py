"""Tests for the market-making edge core formula."""
import numpy as np
import pandas as pd
import pytest

from bot.research.mm_edge import net_maker_cents, maker_fee_cents, enrich


def test_maker_short_profits_when_mid_falls():
    # taker bought YES -> maker SOLD yes (sign=-1) at 60c; mid falls to 58 -> +2c.
    net = net_maker_cents(np.array([60.0]), np.array([-1.0]), np.array([58.0]), np.array([0.0]))
    assert net[0] == 2.0


def test_maker_long_profits_when_mid_rises():
    # taker bought NO -> maker BOUGHT yes (sign=+1) at 40c; mid rises to 42 -> +2c.
    net = net_maker_cents(np.array([40.0]), np.array([1.0]), np.array([42.0]), np.array([0.0]))
    assert net[0] == 2.0


def test_adverse_move_against_maker_is_a_loss():
    # maker sold yes at 60, mid rises to 63 -> -3c (adverse selection).
    net = net_maker_cents(np.array([60.0]), np.array([-1.0]), np.array([63.0]), np.array([0.0]))
    assert net[0] == -3.0


def test_fee_reduces_net():
    net = net_maker_cents(np.array([60.0]), np.array([-1.0]), np.array([58.0]), np.array([0.5]))
    assert net[0] == 1.5


def test_maker_fee_is_max_at_mid_and_small_in_tails():
    f50 = maker_fee_cents(np.array([50.0]))[0]
    f10 = maker_fee_cents(np.array([10.0]))[0]
    assert f50 == np.float64(100 * 0.0175 * 0.25)   # 0.4375c at P=0.5
    assert f10 < f50                                 # cheaper in the tails


def test_enrich_aligns_mid_before_and_after_per_horizon():
    # book mid 41 at t=0, 45 at t=10; a YES-taker fill at t=5 @ 0.42 (maker short).
    book = pd.DataFrame({"ts": [0.0, 10.0], "ticker": ["X", "X"],
                         "yes_bid": [40.0, 44.0], "yes_ask": [42.0, 46.0]})
    trades = pd.DataFrame({"ts": [5.0], "ticker": ["X"], "yes_price": [0.42],
                           "count": [1.0], "taker_side": ["yes"]})
    t = enrich(book, trades, horizons=[0, 10])
    fee = 100 * 0.0175 * 0.42 * 0.58
    assert t["mid_before"].iloc[0] == 41.0 and t["spr_before"].iloc[0] == 2.0
    # Δ=0: unwind at mid 41 -> captured +1c half-spread (maker sold @42)
    assert t["net_0"].iloc[0] == pytest.approx(1.0 - fee)
    # Δ=10: mid rose to 45 -> adverse -3c
    assert t["net_10"].iloc[0] == pytest.approx(-3.0 - fee)
