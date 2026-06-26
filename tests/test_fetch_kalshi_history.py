"""Tests for the Kalshi historical price fetch — D-1 mid derivation.

Regression guard for the empty-book bug: settled markets return
previous_yes_bid=0 / previous_yes_ask=1, whose midpoint (0+1)/2 = 0.5 is a
fabricated price, not a tradeable one. The fetch must NOT emit it.
"""
import math

import pytest

from scripts.fetch_kalshi_history import (
    _compute_d1_mid,
    _decision_price_from_candles,
    _strike_fields,
)


def _candle(ts, close):
    return {"end_period_ts": ts, "price": {"close_dollars": close}}


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


# ── Decision-time price from candlesticks (no look-ahead) ────────────────────

def test_decision_price_uses_last_candle_at_or_before_cutoff():
    candles = [_candle(100, "0.30"), _candle(200, "0.40"), _candle(300, "0.55")]
    assert _decision_price_from_candles(candles, cutoff_ts=250) == pytest.approx(0.40)


def test_decision_price_ignores_candles_after_cutoff():
    """No look-ahead: a candle priced after the decision cutoff (closer to the
    known outcome) must never be used."""
    candles = [_candle(100, "0.30"), _candle(500, "0.99")]
    assert _decision_price_from_candles(candles, cutoff_ts=250) == pytest.approx(0.30)


def test_decision_price_none_when_no_candle_before_cutoff():
    assert _decision_price_from_candles([_candle(500, "0.40")], cutoff_ts=250) is None


def test_decision_price_skips_degenerate_settled_prices():
    """close at exactly 0 or 1 is a settled/degenerate value, not a live mid."""
    candles = [_candle(100, "0.00"), _candle(200, "1.00")]
    assert _decision_price_from_candles(candles, cutoff_ts=250) is None


def test_decision_price_handles_missing_price_field():
    candles = [{"end_period_ts": 100}, _candle(200, "0.42")]
    assert _decision_price_from_candles(candles, cutoff_ts=250) == pytest.approx(0.42)


def test_decision_price_empty_list():
    assert _decision_price_from_candles([], cutoff_ts=250) is None


# ── Strike-field extraction (real bracket structure) ─────────────────────────
# Kalshi temperature markets are mutually-exclusive brackets, NOT above/below.
# strike_type ∈ {greater, less, between} with floor_strike / cap_strike.

def test_strike_fields_greater():
    m = {"strike_type": "greater", "floor_strike": 84}
    assert _strike_fields(m) == {"strike_type": "greater", "floor_strike": 84.0, "cap_strike": None}


def test_strike_fields_less():
    m = {"strike_type": "less", "cap_strike": 77}
    assert _strike_fields(m) == {"strike_type": "less", "floor_strike": None, "cap_strike": 77.0}


def test_strike_fields_between():
    m = {"strike_type": "between", "floor_strike": 83, "cap_strike": 84}
    assert _strike_fields(m) == {"strike_type": "between", "floor_strike": 83.0, "cap_strike": 84.0}


def test_strike_fields_missing_returns_none_type():
    assert _strike_fields({})["strike_type"] is None
