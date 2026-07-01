"""Test the pure ensemble-feature helper for the multi-model edge study."""
import numpy as np
import pandas as pd

from research.multimodel_edge import ensemble_stats


def _mm():
    # one station-day at 24h lead: aifs 80, ecmwf 86, icon 84 (°F).
    return pd.DataFrame({
        "station": ["KX"] * 3,
        "date": ["2026-05-01"] * 3,
        "lead_hour": [24, 24, 24],
        "model": ["aifs", "ecmwf", "icon"],
        "tmax_f": [80.0, 86.0, 84.0],
    })


def test_ensemble_mean_std_and_range():
    e = ensemble_stats(_mm()).iloc[0]
    assert e.ens_mean == np.mean([80, 86, 84])
    assert e.ens_min == 80.0 and e.ens_max == 86.0
    assert e.n_models == 3


def test_aifs_minus_physics_gap_is_signed():
    # aifs 80 vs mean(ecmwf 86, icon 84)=85 → gap = -5 (AIFS colder).
    e = ensemble_stats(_mm()).iloc[0]
    assert e.aifs == 80.0
    assert e.aifs_minus_phys == 80.0 - 85.0


def test_only_selected_lead_used():
    mm = _mm()
    other = mm.assign(lead_hour=48, tmax_f=mm.tmax_f + 20)
    e = ensemble_stats(pd.concat([mm, other], ignore_index=True), lead_hour=24)
    assert len(e) == 1
    assert e.iloc[0].ens_mean == np.mean([80, 86, 84])
