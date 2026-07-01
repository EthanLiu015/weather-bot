"""Tests for the Open-Meteo Previous Runs multi-model fetcher.

The load-bearing pure logic is turning the hourly `temperature_2m_previous_dayN`
arrays (one forecast per valid hour, issued N days earlier) into a per-local-date
daily maximum — the N*24h-lead daily-high forecast that the bot decides on. The
network call itself is a thin wrapper and not unit-tested.
"""
import numpy as np
import pandas as pd

from ingestion.openmeteo import daily_max_by_date, parse_previous_runs, MODEL_SLUGS


def test_daily_max_groups_by_local_date_and_ignores_none():
    times = [
        "2026-05-01T00:00", "2026-05-01T14:00", "2026-05-01T23:00",
        "2026-05-02T10:00", "2026-05-02T15:00",
    ]
    vals = [10.0, 21.5, 18.0, None, 19.0]
    out = daily_max_by_date(times, vals)
    assert out["2026-05-01"] == 21.5
    assert out["2026-05-02"] == 19.0  # None skipped


def test_daily_max_all_none_date_is_absent():
    out = daily_max_by_date(["2026-05-01T00:00", "2026-05-01T12:00"], [None, None])
    assert "2026-05-01" not in out


def test_parse_previous_runs_extracts_model_lead_daily_max():
    resp = {
        "hourly": {
            "time": ["2026-05-01T14:00", "2026-05-01T20:00", "2026-05-02T14:00"],
            "temperature_2m_previous_day1_gfs_seamless": [20.0, 22.0, 25.0],
            "temperature_2m_previous_day3_gfs_seamless": [19.0, 19.5, 24.0],
        }
    }
    df = parse_previous_runs(resp, station="KLAX", model="gfs", leads=(1, 3))
    # long form: one row per (date, lead)
    row = df[(df.date == "2026-05-01") & (df.lead_hour == 24)].iloc[0]
    assert row.station == "KLAX"
    assert row.model == "gfs"
    assert row.tmax_c == 22.0
    assert df[(df.date == "2026-05-01") & (df.lead_hour == 72)].iloc[0].tmax_c == 19.5
    assert df[(df.date == "2026-05-02") & (df.lead_hour == 24)].iloc[0].tmax_c == 25.0


def test_parse_skips_missing_model_lead_column():
    # graphcast day3 not archived → column absent → no rows for that lead, no crash.
    resp = {"hourly": {"time": ["2026-05-01T14:00"],
                       "temperature_2m_previous_day1_gfs_graphcast025": [21.0]}}
    df = parse_previous_runs(resp, station="KLAX", model="graphcast", leads=(1, 3))
    assert set(df.lead_hour.unique()) == {24}


def test_model_slugs_cover_the_five_targets():
    assert set(MODEL_SLUGS) == {"aifs", "graphcast", "gfs", "icon", "ecmwf"}
