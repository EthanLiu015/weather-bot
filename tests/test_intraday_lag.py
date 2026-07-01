"""Behavioural tests for the intraday market-efficiency diagnostic.

The question this tool answers: does the Kalshi book converge to the outcome as
the day progresses, or does it stay mispriced while markets are still live? A
persistent gap between the market's Brier and 0 (with markets still unresolved)
is the opening the edge-gated intraday thesis needs.
"""
import numpy as np
import pandas as pd

from research.intraday_lag import snapshot_efficiency


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_perfectly_priced_snapshot_has_zero_brier_and_full_resolution():
    # Prices that exactly match the binary outcome: an efficient, resolved book.
    df = _frame([
        {"p+0": 1.0, "settlement": 1.0},
        {"p+0": 0.0, "settlement": 0.0},
        {"p+0": 1.0, "settlement": 1.0},
    ])
    stats = snapshot_efficiency(df, snapshots=["p+0"])["p+0"]
    assert stats["n"] == 3
    assert stats["brier"] == 0.0
    assert stats["resolved_frac"] == 1.0


def test_coinflip_prices_score_quarter_brier_and_zero_resolution():
    # A book stuck at 0.5 on decided outcomes: maximally uninformative.
    df = _frame([
        {"p+0": 0.5, "settlement": 1.0},
        {"p+0": 0.5, "settlement": 0.0},
    ])
    stats = snapshot_efficiency(df, snapshots=["p+0"])["p+0"]
    assert stats["brier"] == 0.25
    assert stats["resolved_frac"] == 0.0
    assert stats["mean_price"] == 0.5


def test_nulls_are_excluded_per_snapshot():
    # p-12 is entirely missing (as in the real data); it must report n=0, not crash.
    df = _frame([
        {"p-12": None, "p+0": 0.2, "settlement": 0.0},
        {"p-12": None, "p+0": 0.9, "settlement": 1.0},
    ])
    out = snapshot_efficiency(df, snapshots=["p-12", "p+0"])
    assert out["p-12"]["n"] == 0
    assert np.isnan(out["p-12"]["brier"])
    assert out["p+0"]["n"] == 2


def test_resolved_fraction_uses_distance_to_certainty():
    # min(p, 1-p) < 0.1 counts as resolved. 0.05 -> resolved; 0.30 -> not.
    df = _frame([
        {"p+0": 0.05, "settlement": 0.0},
        {"p+0": 0.30, "settlement": 0.0},
    ])
    stats = snapshot_efficiency(df, snapshots=["p+0"])["p+0"]
    assert stats["resolved_frac"] == 0.5
