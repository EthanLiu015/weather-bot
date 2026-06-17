import datetime
import math

import pandas as pd
import pytest

from scripts.backfill_ecmwf import (
    _apply_ecmwf_backfill,
    _ecmwf_s3_url,
    _parse_ecmwf_s3_index,
    _steps_for_lead_day,
    _wb2_flat_idx,
    _wb2_lat_idx,
    _wb2_lead_indices_for_day,
    _wb2_lon_idx,
    _wb2_time_idx,
)

# ── WB2 grid index helpers ────────────────────────────────────────────────────


def test_wb2_lat_idx_for_kord():
    assert _wb2_lat_idx(41.9786) == 528


def test_wb2_lon_idx_for_kord_negative_longitude():
    assert _wb2_lon_idx(-87.9047) == 1088


def test_wb2_flat_idx_for_kord():
    assert _wb2_flat_idx(41.9786, -87.9047) == 528 * 1440 + 1088


def test_wb2_lat_idx_equator():
    assert _wb2_lat_idx(0.0) == 360


def test_wb2_lon_idx_positive_longitude():
    assert _wb2_lon_idx(272.0) == 1088


# ── WB2 lead indices ─────────────────────────────────────────────────────────


def test_wb2_lead_indices_for_day_1():
    assert _wb2_lead_indices_for_day(1) == [4, 5, 6, 7]


def test_wb2_lead_indices_for_day_7():
    assert _wb2_lead_indices_for_day(7) == [28, 29, 30, 31]


def test_wb2_lead_indices_for_day_are_consecutive():
    for lead_day in range(1, 8):
        indices = _wb2_lead_indices_for_day(lead_day)
        assert len(indices) == 4
        assert indices == list(range(indices[0], indices[0] + 4))


# ── WB2 time index ───────────────────────────────────────────────────────────


def test_wb2_time_idx_known_date():
    # 2021-05-27 00z is 47352 hours after 2016-01-01, index = 47352/12 = 3946
    assert _wb2_time_idx("20210527") == 3946


def test_wb2_time_idx_epoch():
    assert _wb2_time_idx("20160101") == 0


# ── S3 step ranges ───────────────────────────────────────────────────────────


def test_steps_for_lead_day_1_returns_8_3h_steps():
    steps = _steps_for_lead_day(1)
    assert steps == [24, 27, 30, 33, 36, 39, 42, 45]


def test_steps_for_lead_day_5_returns_8_3h_steps():
    steps = _steps_for_lead_day(5)
    assert steps == [120, 123, 126, 129, 132, 135, 138, 141]


def test_steps_for_lead_day_6_includes_boundary_step_and_6h_steps():
    # After step 144 the archive switches to 6h resolution
    steps = _steps_for_lead_day(6)
    assert steps == [144, 150, 156, 162]


def test_steps_for_lead_day_7_returns_4_6h_steps():
    steps = _steps_for_lead_day(7)
    assert steps == [168, 174, 180, 186]


def test_steps_for_lead_day_grouping_matches_step_div_24():
    for lead_day in range(1, 8):
        for s in _steps_for_lead_day(lead_day):
            assert s // 24 == lead_day, f"step {s} has s//24={s//24}, expected {lead_day}"


# ── S3 URL construction ──────────────────────────────────────────────────────


def test_ecmwf_s3_url_pre_cutover_uses_0p4_beta():
    url = _ecmwf_s3_url("20230601", 24)
    assert "0p4-beta/oper" in url
    assert "20230601000000-24h-oper-fc.grib2" in url


def test_ecmwf_s3_url_post_cutover_uses_ifs_0p25():
    url = _ecmwf_s3_url("20240601", 24)
    assert "ifs/0p25/oper" in url
    assert "20240601000000-24h-oper-fc.grib2" in url


def test_ecmwf_s3_url_cutover_date_uses_ifs_0p25():
    url = _ecmwf_s3_url("20240301", 48)
    assert "ifs/0p25/oper" in url


# ── S3 index parsing ─────────────────────────────────────────────────────────

_SAMPLE_S3_INDEX = """\
{"param":"u","levtype":"pl","level":1000,"_offset":0,"_length":100000}
{"param":"2t","levtype":"sfc","_offset":640000,"_length":640000}
{"param":"tp","levtype":"sfc","_offset":1280000,"_length":100000}
"""


def test_parse_ecmwf_s3_index_extracts_2t_offset_and_length():
    result = _parse_ecmwf_s3_index(_SAMPLE_S3_INDEX)
    assert result["2t"] == (640000, 640000)


def test_parse_ecmwf_s3_index_returns_first_occurrence_per_param():
    # When a param appears multiple times, first occurrence wins
    index_text = """\
{"param":"2t","_offset":100,"_length":200}
{"param":"2t","_offset":999,"_length":999}
"""
    result = _parse_ecmwf_s3_index(index_text)
    assert result["2t"] == (100, 200)


def test_parse_ecmwf_s3_index_handles_multiple_params():
    result = _parse_ecmwf_s3_index(_SAMPLE_S3_INDEX)
    assert "u" in result
    assert "tp" in result
    assert result["u"] == (0, 100000)


def test_parse_ecmwf_s3_index_ignores_blank_lines():
    index_text = '\n{"param":"2t","_offset":1,"_length":2}\n\n'
    result = _parse_ecmwf_s3_index(index_text)
    assert result["2t"] == (1, 2)


# ── _apply_ecmwf_backfill ────────────────────────────────────────────────────


def _features_row(**overrides) -> dict:
    row = {
        "date": pd.Timestamp("2024-01-01"),
        "station": "KORD",
        "lead_hour": 24,
        "gefs_tmax_mean": 30.0,
        "ecmwf_tmax": 0.0,
        "ecmwf_tmin": 0.0,
        "ecmwf_gefs_tmax_delta": 0.0,
    }
    row.update(overrides)
    return row


def _backfill_row(**overrides) -> dict:
    row = {
        "date": pd.Timestamp("2024-01-01"),
        "station": "KORD",
        "lead_hour": 24,
        "ecmwf_tmax": 72.5,
        "ecmwf_tmin": 55.0,
    }
    row.update(overrides)
    return row


def test_apply_ecmwf_backfill_overwrites_tmax_tmin():
    features_df = pd.DataFrame([_features_row()])
    backfill_df = pd.DataFrame([_backfill_row(ecmwf_tmax=72.5, ecmwf_tmin=55.0)])

    result = _apply_ecmwf_backfill(features_df, backfill_df)

    assert result.iloc[0]["ecmwf_tmax"] == pytest.approx(72.5)
    assert result.iloc[0]["ecmwf_tmin"] == pytest.approx(55.0)


def test_apply_ecmwf_backfill_recomputes_delta():
    features_df = pd.DataFrame([_features_row(gefs_tmax_mean=68.0)])
    backfill_df = pd.DataFrame([_backfill_row(ecmwf_tmax=72.5)])

    result = _apply_ecmwf_backfill(features_df, backfill_df)

    assert result.iloc[0]["ecmwf_gefs_tmax_delta"] == pytest.approx(abs(72.5 - 68.0))


def test_apply_ecmwf_backfill_leaves_unmatched_rows_unchanged():
    features_df = pd.DataFrame([
        _features_row(date=pd.Timestamp("2024-01-01"), ecmwf_tmax=0.0),
        _features_row(date=pd.Timestamp("2024-01-02"), ecmwf_tmax=0.0),
    ])
    backfill_df = pd.DataFrame([_backfill_row(date=pd.Timestamp("2024-01-01"), ecmwf_tmax=72.5)])

    result = _apply_ecmwf_backfill(features_df, backfill_df)

    row2 = result[result["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert row2["ecmwf_tmax"] == pytest.approx(0.0)
    assert row2["ecmwf_gefs_tmax_delta"] == pytest.approx(0.0)


def test_apply_ecmwf_backfill_handles_object_dtype_date():
    features_df = pd.DataFrame([_features_row(date=datetime.date(2024, 1, 1))])
    backfill_df = pd.DataFrame([_backfill_row(date=pd.Timestamp("2024-01-01"), ecmwf_tmax=72.5)])

    result = _apply_ecmwf_backfill(features_df, backfill_df)

    assert result.iloc[0]["date"] == datetime.date(2024, 1, 1)
    assert result.iloc[0]["ecmwf_tmax"] == pytest.approx(72.5)


def test_apply_ecmwf_backfill_empty_backfill_returns_copy():
    features_df = pd.DataFrame([_features_row(ecmwf_tmax=55.0)])
    backfill_df = pd.DataFrame(columns=["date", "station", "lead_hour", "ecmwf_tmax", "ecmwf_tmin"])

    result = _apply_ecmwf_backfill(features_df, backfill_df)

    assert result.iloc[0]["ecmwf_tmax"] == pytest.approx(55.0)
    assert set(result.columns) == set(features_df.columns)


def test_apply_ecmwf_backfill_preserves_row_count():
    features_df = pd.DataFrame([_features_row(date=pd.Timestamp(f"2024-01-0{i}")) for i in range(1, 6)])
    backfill_df = pd.DataFrame([_backfill_row(date=pd.Timestamp("2024-01-01"))])

    result = _apply_ecmwf_backfill(features_df, backfill_df)

    assert len(result) == len(features_df)
