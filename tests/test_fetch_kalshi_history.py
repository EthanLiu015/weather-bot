"""Tests for the Kalshi historical price fetch — D-1 mid derivation.

Regression guard for the empty-book bug: settled markets return
previous_yes_bid=0 / previous_yes_ask=1, whose midpoint (0+1)/2 = 0.5 is a
fabricated price, not a tradeable one. The fetch must NOT emit it.
"""
import math

import pytest

from scripts.fetch_kalshi_history import _compute_d1_mid


def test_compute_d1_mid_rejects_empty_book():
    """bid=0, ask=1 is an empty/collapsed book — its 0.5 midpoint is fabricated."""
    assert math.isnan(_compute_d1_mid(0.0, 1.0))


def test_compute_d1_mid_genuine_two_sided_book():
    assert _compute_d1_mid(0.30, 0.34) == pytest.approx(0.32)


def test_compute_d1_mid_balanced_book_allowed():
    """A genuine tight book that happens to mid at 0.5 is still a real price."""
    assert _compute_d1_mid(0.48, 0.52) == pytest.approx(0.50)


def test_compute_d1_mid_none_inputs():
    assert math.isnan(_compute_d1_mid(None, None))
    assert math.isnan(_compute_d1_mid(0.45, None))


def test_compute_d1_mid_degenerate_books():
    assert math.isnan(_compute_d1_mid(0.0, 0.0))   # no ask
    assert math.isnan(_compute_d1_mid(1.0, 1.0))   # no bid
    assert math.isnan(_compute_d1_mid(0.6, 0.4))   # crossed book
