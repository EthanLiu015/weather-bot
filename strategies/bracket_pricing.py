"""Pricing Kalshi temperature BRACKET markets from a P(high > x) function.

Kalshi temperature markets are mutually-exclusive °F brackets (strike_type
greater/less/between with floor_strike/cap_strike), NOT above/below contracts.
This module converts a model's calibrated P(high > x) into the YES probability
for any bracket. It lives in its own module so both the production strategy
(strategies.ensemble_strategy) and the eval harness (backtest.real_market_eval)
can share one implementation without a circular import.
"""
from __future__ import annotations

from typing import Callable, Optional

# NWS daily highs are reported in whole °F, so a market boundary "high > 84"
# means "high ≥ 85"; the decision boundary on the continuous forecast sits at
# 84.5. These ±0.5 continuity corrections convert integer strike rules into
# thresholds for the continuous predictive distribution.
_HALF = 0.5


def bracket_yes_prob(
    prob_above: Callable[[float], Optional[float]],
    strike_type: str,
    floor_strike: Optional[float],
    cap_strike: Optional[float],
) -> Optional[float]:
    """Model's YES probability for a real Kalshi temperature bracket.

    Kalshi temperature markets are mutually-exclusive brackets:
      * greater (>F):       YES = P(high > F)        = prob_above(F + 0.5)
      * less   (<C):        YES = P(high < C)        = 1 - prob_above(C - 0.5)
      * between [F, C]:      YES = P(F ≤ high ≤ C)    = prob_above(F-0.5) - prob_above(C+0.5)

    `prob_above(x)` is the model's calibrated P(high > x); returns None when
    unpriceable (propagated as None so the market is skipped).
    """
    if strike_type == "greater":
        return prob_above(float(floor_strike) + _HALF)
    if strike_type == "less":
        p = prob_above(float(cap_strike) - _HALF)
        return None if p is None else 1.0 - p
    if strike_type == "between":
        lo = prob_above(float(floor_strike) - _HALF)
        hi = prob_above(float(cap_strike) + _HALF)
        if lo is None or hi is None:
            return None
        return max(0.0, lo - hi)
    raise ValueError(f"Unknown strike_type: {strike_type!r}")


def bracket_primary_threshold(
    strike_type: str, floor_strike: Optional[float], cap_strike: Optional[float]
) -> float:
    """The boundary threshold whose predictive distribution best characterises a
    bracket's uncertainty — used to derive a single ci_width for risk gating.

    greater/between anchor on the floor (F+0.5 / F-0.5); less on the cap (C-0.5).
    """
    if strike_type == "greater":
        return float(floor_strike) + _HALF
    if strike_type == "between":
        return float(floor_strike) - _HALF
    if strike_type == "less":
        return float(cap_strike) - _HALF
    raise ValueError(f"Unknown strike_type: {strike_type!r}")
