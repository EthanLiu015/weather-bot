"""Tests for grouping live METAR observations into daily Tmax history,
used to compute obs_minus_model lag residuals for live inference."""
import datetime as dt
from datetime import timezone

import numpy as np
import pytest

from ingestion.asos import compute_daily_tmax_history


def _obs(observation_time: dt.datetime, temp_f: float) -> dict:
    return {"observation_time": observation_time, "temp_f": temp_f}


# Reference "now": 2026-06-10 12:00 UTC == 2026-06-10 07:00 America/Chicago (CDT, UTC-5)
_NOW = dt.datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc)
_TZ = "America/Chicago"


def test_groups_observations_by_local_date_and_takes_max_temp():
    obs_list = [
        _obs(dt.datetime(2026, 6, 9, 18, 0, tzinfo=timezone.utc), 80.0),  # Jun 9, 13:00 CDT
        _obs(dt.datetime(2026, 6, 9, 20, 0, tzinfo=timezone.utc), 85.0),  # Jun 9, 15:00 CDT
    ]

    history = compute_daily_tmax_history(obs_list, _TZ, days=3, reference_time=_NOW)

    assert history[dt.date(2026, 6, 9)] == pytest.approx(85.0)


def test_excludes_todays_incomplete_date():
    obs_list = [
        _obs(dt.datetime(2026, 6, 10, 11, 0, tzinfo=timezone.utc), 70.0),  # Jun 10, 06:00 CDT (today)
    ]

    history = compute_daily_tmax_history(obs_list, _TZ, days=3, reference_time=_NOW)

    assert dt.date(2026, 6, 10) not in history


def test_excludes_dates_older_than_lookback_window():
    obs_list = [
        _obs(dt.datetime(2026, 6, 6, 18, 0, tzinfo=timezone.utc), 60.0),  # Jun 6 — 4 days back
    ]

    history = compute_daily_tmax_history(obs_list, _TZ, days=3, reference_time=_NOW)

    assert dt.date(2026, 6, 6) not in history


def test_empty_obs_list_returns_empty_history():
    assert compute_daily_tmax_history([], _TZ, days=3, reference_time=_NOW) == {}


def test_skips_observations_missing_temp_or_time():
    obs_list = [
        {"observation_time": None, "temp_f": 80.0},
        {"observation_time": dt.datetime(2026, 6, 9, 18, 0, tzinfo=timezone.utc), "temp_f": None},
    ]

    history = compute_daily_tmax_history(obs_list, _TZ, days=3, reference_time=_NOW)

    assert history == {}
