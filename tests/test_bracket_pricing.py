"""Tests for the shared bracket-pricing helpers.

Kalshi temperature markets are mutually-exclusive °F brackets; bracket_yes_prob
converts a calibrated P(high > x) into a bracket's YES probability. prob_above(x)
is P(high > x); integer highs decide at x+0.5 (continuity correction).
"""
import pytest

from strategies.bracket_pricing import bracket_yes_prob, bracket_primary_threshold


def test_greater_bracket_uses_floor_plus_half():
    seen = {}
    def pa(x):
        seen["x"] = x
        return 0.7
    assert bracket_yes_prob(pa, "greater", 84, None) == pytest.approx(0.7)
    assert seen["x"] == pytest.approx(84.5)


def test_less_bracket_is_complement_at_cap_minus_half():
    seen = {}
    def pa(x):
        seen["x"] = x
        return 0.7  # P(high > 76.5)
    assert bracket_yes_prob(pa, "less", None, 77) == pytest.approx(0.3)
    assert seen["x"] == pytest.approx(76.5)


def test_between_bracket_is_cdf_difference():
    probs = {82.5: 0.8, 84.5: 0.3}
    assert bracket_yes_prob(lambda x: probs[x], "between", 83, 84) == pytest.approx(0.5)


def test_between_bracket_clamped_non_negative():
    assert bracket_yes_prob(lambda x: 0.5 if x < 0 else 0.50001, "between", 0, 1) == 0.0


def test_bracket_returns_none_when_prob_unavailable():
    assert bracket_yes_prob(lambda x: None, "greater", 84, None) is None


def test_unknown_strike_type_raises():
    with pytest.raises(ValueError):
        bracket_yes_prob(lambda x: 0.5, "sideways", 1, 2)


# ── primary threshold (used to derive a single ci_width per bracket) ──────────

def test_primary_threshold_per_strike_type():
    assert bracket_primary_threshold("greater", 84, None) == pytest.approx(84.5)
    assert bracket_primary_threshold("between", 83, 84) == pytest.approx(82.5)
    assert bracket_primary_threshold("less", None, 77) == pytest.approx(76.5)


def test_primary_threshold_unknown_raises():
    with pytest.raises(ValueError):
        bracket_primary_threshold("sideways", 1, 2)
