"""Tests for the obs-conditioned intraday edge core.

The load-bearing logic is prob_final_above: turning 'the max can only rise' into
P(final > x). Everything else (bracket pricing, P&L) is already tested elsewhere.
"""
import numpy as np
import pandas as pd

from research.intraday_edge import (
    prob_final_above,
    residuals_by_offset,
    residuals_by_station_offset,
)


def test_threshold_already_met_is_certain():
    # run_max 80, threshold 78 → already exceeded, and R ≥ 0, so P = 1.
    R = np.array([0.0, 1.0, 2.0, 3.0])
    assert prob_final_above(R, run_max=80.0, x=78.0) == 1.0


def test_far_above_reach_is_impossible():
    # Need +10 more but the largest observed rise was +3 → P = 0.
    R = np.array([0.0, 1.0, 2.0, 3.0])
    assert prob_final_above(R, run_max=80.0, x=90.0) == 0.0


def test_partial_reach_is_the_empirical_tail_fraction():
    # run_max 80, x 82 → need R > 2. Of [0,1,2,3,4], two values (3,4) exceed 2.
    R = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert prob_final_above(R, run_max=80.0, x=82.0) == 2 / 5


def test_residuals_by_offset_uses_final_minus_run_max_per_station_day():
    # One station-day, two offsets: run_max 78 @16h then 80 @22h. final=80.
    # R@16h = 2, R@22h = 0.
    df = pd.DataFrame({
        "station": ["KX", "KX"],
        "date": pd.to_datetime(["2026-05-01", "2026-05-01"]),
        "offset_h": [16, 22],
        "run_max": [78.0, 80.0],
    })
    r = residuals_by_offset(df)
    assert r[16].tolist() == [2.0]
    assert r[22].tolist() == [0.0]


def test_residuals_by_station_offset_keys_on_station_and_offset():
    # Two stations, same offset, different final-vs-run_max rises.
    # KA: run_max 78 @16h, final 82 → R=4.  KB: run_max 90 @16h, final 91 → R=1.
    df = pd.DataFrame({
        "station": ["KA", "KA", "KB", "KB"],
        "date": pd.to_datetime(["2026-05-01"] * 4),
        "offset_h": [16, 22, 16, 22],
        "run_max": [78.0, 82.0, 90.0, 91.0],
    })
    r = residuals_by_station_offset(df)
    assert r[("KA", 16)].tolist() == [4.0]
    assert r[("KB", 16)].tolist() == [1.0]
    assert r[("KA", 22)].tolist() == [0.0]
