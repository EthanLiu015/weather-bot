"""Tests for the local Kalshi order-book replica.

Kalshi posts `yes` and `no` resting-bid sides; a NO bid at n cents is a YES offer
at (100-n) cents. So best yes_bid = max(yes), best yes_ask = 100 - max(no).
"""
import pytest

from bot.marketdata.orderbook import OrderBook, to_cents


def test_to_cents_normalises_dollar_strings():
    assert to_cents("0.0800") == 8
    assert to_cents("0.5400") == 54
    assert to_cents(0.99) == 99


def test_snapshot_sets_best_bid_ask_and_spread():
    ob = OrderBook("KXHIGH-X")
    ob.apply_snapshot(
        yes_levels=[("0.40", "100"), ("0.38", "50")],
        no_levels=[("0.55", "20"), ("0.50", "80")],
        seq=2,
    )
    assert ob.yes_bid == 40            # max yes
    assert ob.yes_ask == 45            # 100 - max(no=55)
    assert ob.spread == 5
    assert ob.top()["yes_bid_sz"] == 100
    assert ob.top()["yes_ask_sz"] == 20  # size resting at no=55


def test_delta_adds_and_improves_top():
    ob = OrderBook("X")
    ob.apply_snapshot([("0.40", "100")], [("0.55", "20")], seq=1)
    ob.apply_delta("0.42", "30", "yes", seq=2)
    assert ob.yes_bid == 42
    assert ob.top()["yes_bid_sz"] == 30


def test_delta_removes_level_when_quantity_hits_zero():
    ob = OrderBook("X")
    ob.apply_snapshot([("0.40", "100")], [("0.55", "20")], seq=1)
    ob.apply_delta("0.40", "-100", "yes", seq=2)
    assert ob.yes_bid is None
    assert 40 not in ob.yes


def test_floating_point_dust_quantity_removes_level_not_phantom():
    # Real-world bug: summing deltas lands on ~1e-17 instead of 0, leaving a
    # phantom best level that crosses the book. The dusted level must be removed.
    ob = OrderBook("X")
    ob.apply_snapshot([("0.40", "0.1")], [("0.55", "20")], seq=1)
    ob.apply_delta("0.40", "0.2", "yes", seq=2)   # 0.1 + 0.2 = 0.30000000000000004
    ob.apply_delta("0.40", "-0.3", "yes", seq=3)  # -> 5.55e-17 dust, not exactly 0
    assert 40 not in ob.yes          # dust removed
    assert ob.yes_bid is None        # not a phantom best bid


def test_delta_tracks_last_seq():
    # seq is global-per-channel (gap detection is the logger's job, not the
    # book's) — the book just records the last seq it applied.
    ob = OrderBook("X")
    ob.apply_snapshot([("0.40", "100")], [("0.55", "20")], seq=5)
    ob.apply_delta("0.41", "10", "yes", seq=9)
    assert ob.seq == 9


def test_empty_book_has_no_quotes():
    ob = OrderBook("X")
    assert ob.yes_bid is None and ob.yes_ask is None and ob.spread is None


def test_resnapshot_rebuilds_book():
    ob = OrderBook("X")
    ob.apply_snapshot([("0.40", "100")], [("0.55", "20")], seq=5)
    ob.apply_snapshot([("0.42", "100")], [("0.55", "20")], seq=20)
    assert ob.yes_bid == 42 and ob.seq == 20
