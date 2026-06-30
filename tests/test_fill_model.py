"""Tests for the FIFO queue fill simulator."""
import pytest

from bot.research.fill_model import simulate_side


def test_back_of_queue_fills_only_after_queue_clears():
    # Depth 5 ahead; maker rests q=2. Two trades of 4 at the level.
    # t1: eats 4 of the 5 ahead (qa->1), no fill. t2: eats remaining 1 ahead,
    # fills our 2, we re-join behind depth 5, leftover 1 just trims that queue.
    events = [("q", 42, 5), ("x", 42, 4, {0: 1.0}), ("x", 42, 4, {0: 1.0})]
    filled, net, touch = simulate_side(events, q=2, horizons=[0])
    assert touch == 8
    assert filled == 2
    assert net[0] == pytest.approx(2.0)


def test_empty_queue_captures_the_whole_trade():
    # No depth ahead -> we're first; a size-3 trade fills us (re-quoting).
    events = [("q", 42, 0), ("x", 42, 3, {0: 1.0})]
    filled, net, touch = simulate_side(events, q=2, horizons=[0])
    assert touch == 3 and filled == 3


def test_price_move_makes_us_miss_the_old_level():
    # We join at 42 behind depth 100; price jumps to 43 before anything trades at
    # 42, so we re-join at 43 (depth 0) and capture the trade there.
    events = [("q", 42, 100), ("q", 43, 0), ("x", 43, 2, {0: 1.0})]
    filled, _, touch = simulate_side(events, q=5, horizons=[0])
    assert touch == 2 and filled == 2


def test_credits_real_net_per_filled_contract():
    # Zero depth ahead, q=3: we fill 3, instantly re-quote, the trade's leftover 1
    # fills 1 more -> the whole 4-lot, credited at the trade's per-contract net.
    events = [("q", 42, 0), ("x", 42, 4, {0: 0.5, 5: -0.3})]
    filled, net, _ = simulate_side(events, q=3, horizons=[0, 5])
    assert filled == 4
    assert net[0] == pytest.approx(2.0) and net[5] == pytest.approx(-1.2)
