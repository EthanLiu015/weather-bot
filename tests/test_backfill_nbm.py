import pandas as pd
import pytest

import numpy as np

from scripts.backfill_nbm import (
    NBM_BACKFILL_COLS,
    _apply_nbm_backfill,
    _byte_range,
    _field_byte_range,
    _find_message_index,
    _is_pop12_line,
    null_partial_nbm_rows,
)

# Minimal idx excerpts in the real `.idx` line format: "N:offset:d=...:description"
TMAX_IDX = """
1:0:d=2026061400:CAPE:surface:24 hour fcst:
2:1500000:d=2026061400:TMAX:2 m above ground:12-24 hour max fcst:
3:3000000:d=2026061400:TMAX:2 m above ground:12-24 hour max fcst:ens std dev
4:4600000:d=2026061400:MIXHT:entire atmosphere:24 hour fcst:
5:6000000:d=2026061400:APCP:surface:12-24 hour acc fcst:prob >0.254:prob fcst 255/255
6:7000000:d=2026061400:PWTHER:surface - reserved:24 hour fcst:
""".strip().splitlines()

TMIN_IDX = """
1:0:d=2026061400:CAPE:surface:12 hour fcst:
2:1400000:d=2026061400:TMIN:2 m above ground:0-12 hour min fcst:
3:2900000:d=2026061400:TMIN:2 m above ground:0-12 hour min fcst:ens std dev
4:4300000:d=2026061400:MIXHT:entire atmosphere:12 hour fcst:
""".strip().splitlines()

# Real .idx excerpts (blend.20210527/00, f024 and f012) used to verify
# _field_byte_range against the actual NBM archive line formats.
TMAX_F024_IDX = """
49:53840544:d=2021052700:MAXREF:1000 m above ground:23-24 hour max fcst:
50:55339378:d=2021052700:TMAX:2 m above ground:12-24 hour max fcst:
51:56845276:d=2021052700:TMAX:2 m above ground:12-24 hour max fcst:ens std dev
52:58459412:d=2021052700:MIXHT:entire atmosphere (considered as a single layer):24 hour fcst:
56:63509902:d=2021052700:APCP:surface:23-24 hour acc fcst:prob >0.254
57:64360147:d=2021052700:APCP:surface:18-24 hour acc fcst:prob >0.254
58:65264539:d=2021052700:APCP:surface:12-24 hour acc fcst:prob >0.254
59:66235510:d=2021052700:var discipline=0 center=7 local_table=1 parmcat=19 parm=237:surface:24 hour fcst:
""".strip().splitlines()

TMIN_F012_IDX = """
37:46215266:d=2021052700:MAXREF:1000 m above ground:11-12 hour max fcst:
38:47736478:d=2021052700:TMIN:2 m above ground:0-12 hour min fcst:
39:49083113:d=2021052700:TMIN:2 m above ground:0-12 hour min fcst:ens std dev
40:51017570:d=2021052700:MIXHT:entire atmosphere (considered as a single layer):12 hour fcst:
""".strip().splitlines()


def test_find_message_index_matches_tmax_mean_not_stddev():
    idx = _find_message_index(TMAX_IDX, lambda line: "TMAX:2 m above ground" in line and "ens std dev" not in line)

    assert idx == 2


def test_find_message_index_matches_pop12_line():
    idx = _find_message_index(TMAX_IDX, _is_pop12_line)

    assert idx == 5


def test_find_message_index_returns_none_when_no_match():
    idx = _find_message_index(TMIN_IDX, _is_pop12_line)

    assert idx is None


def test_byte_range_covers_mean_and_stddev_messages():
    rng = _byte_range(TMAX_IDX, start_index=2, span=2)

    assert rng == (1500000, 4599999)


def test_byte_range_for_single_message():
    rng = _byte_range(TMIN_IDX, start_index=2, span=2)

    assert rng == (1400000, 4299999)


def test_byte_range_last_message_extends_past_idx():
    rng = _byte_range(TMAX_IDX, start_index=6, span=1)

    assert rng[0] == 7000000
    assert rng[1] > 7000000


def test_field_byte_range_tmax_covers_mean_and_stddev():
    rng = _field_byte_range(TMAX_F024_IDX, "tmax")

    assert rng == (55339378, 58459411)


def test_field_byte_range_pop12_selects_12hr_window_not_1hr_or_6hr():
    rng = _field_byte_range(TMAX_F024_IDX, "pop12")

    assert rng == (65264539, 66235509)


def test_field_byte_range_tmin_covers_mean_message_only():
    rng = _field_byte_range(TMIN_F012_IDX, "tmin")

    assert rng == (47736478, 49083112)


def test_field_byte_range_returns_none_when_field_absent():
    rng = _field_byte_range(TMIN_F012_IDX, "pop12")

    assert rng is None


def _features_row(**overrides) -> dict:
    row = {
        "date": pd.Timestamp("2024-01-01"),
        "station": "KORD",
        "lead_hour": 24,
        "gefs_tmax_mean": 30.0,
        "nbm_t10": 0.0, "nbm_t25": 0.0, "nbm_t50": 0.0,
        "nbm_t75": 0.0, "nbm_t90": 0.0, "nbm_tmax": 0.0,
        "nbm_tmin": 0.0, "nbm_pop12": 0.0, "nbm_spread": 0.0,
        "nbm_gefs_delta": 0.0,
    }
    row.update(overrides)
    return row


def test_apply_nbm_backfill_overwrites_real_values_and_recomputes_delta():
    features_df = pd.DataFrame([
        _features_row(date=pd.Timestamp("2024-01-01"), gefs_tmax_mean=30.0),
        _features_row(date=pd.Timestamp("2024-01-02"), gefs_tmax_mean=32.0),
    ])
    backfill_df = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-01"), "station": "KORD", "lead_hour": 24,
        "t10": 28.0, "t25": 29.0, "t50": 31.0, "t75": 33.0, "t90": 34.0,
        "tmax": 31.0, "tmin": 20.0, "pop12": 10.0, "spread": 2.0,
    }])

    result = _apply_nbm_backfill(features_df, backfill_df)

    backfilled = result[result["date"] == pd.Timestamp("2024-01-01")].iloc[0]
    assert backfilled["nbm_t50"] == pytest.approx(31.0)
    assert backfilled["nbm_tmin"] == pytest.approx(20.0)
    assert backfilled["nbm_pop12"] == pytest.approx(10.0)
    assert backfilled["nbm_gefs_delta"] == pytest.approx(abs(31.0 - 30.0))

    untouched = result[result["date"] == pd.Timestamp("2024-01-02")].iloc[0]
    assert untouched["nbm_t50"] == 0.0
    assert untouched["nbm_gefs_delta"] == 0.0


def test_apply_nbm_backfill_handles_object_dtype_date_column():
    # features.parquet stores `date` as object dtype (datetime.date objects),
    # while the backfill CSV's `date` parses to datetime64 — merging must not
    # raise a dtype-mismatch error.
    import datetime

    features_df = pd.DataFrame([_features_row(date=datetime.date(2024, 1, 1), gefs_tmax_mean=30.0)])
    backfill_df = pd.DataFrame([{
        "date": pd.Timestamp("2024-01-01"), "station": "KORD", "lead_hour": 24,
        "t10": 28.0, "t25": 29.0, "t50": 31.0, "t75": 33.0, "t90": 34.0,
        "tmax": 31.0, "tmin": 20.0, "pop12": 10.0, "spread": 2.0,
    }])

    result = _apply_nbm_backfill(features_df, backfill_df)

    assert result.iloc[0]["date"] == datetime.date(2024, 1, 1)
    assert result.iloc[0]["nbm_t50"] == pytest.approx(31.0)


def test_apply_nbm_backfill_preserves_row_count_and_columns():
    features_df = pd.DataFrame([_features_row()])
    backfill_df = pd.DataFrame(columns=["date", "station", "lead_hour", *NBM_BACKFILL_COLS])

    result = _apply_nbm_backfill(features_df, backfill_df)

    assert len(result) == len(features_df)
    assert set(features_df.columns) == set(result.columns)


# ── null_partial_nbm_rows ─────────────────────────────────────────────────────

_NBM_PERCENTILE_COLS = ["nbm_t10", "nbm_t25", "nbm_t75", "nbm_t90"]
_NBM_AUX_COLS = ["nbm_tmin", "nbm_pop12", "nbm_spread"]


def _partial_nbm_row(**overrides) -> dict:
    """Row where nbm_t50 is valid but all other percentiles are 0 (partial fill)."""
    row = {
        "station": "KORD",
        "date": "2024-05-19",
        "lead_hour": 96,
        "gefs_tmax_mean": 70.0,
        "nbm_t10": 0.0,
        "nbm_t25": 0.0,
        "nbm_t50": 75.0,
        "nbm_t75": 0.0,
        "nbm_t90": 0.0,
        "nbm_tmin": 0.0,
        "nbm_pop12": 0.0,
        "nbm_spread": 0.0,
        "nbm_gefs_delta": abs(75.0 - 70.0),
    }
    row.update(overrides)
    return row


def test_null_partial_nbm_rows_nulls_percentile_cols_when_t50_valid_but_rest_zero():
    df = pd.DataFrame([_partial_nbm_row()])
    result = null_partial_nbm_rows(df)
    for col in _NBM_PERCENTILE_COLS + _NBM_AUX_COLS:
        assert np.isnan(result.iloc[0][col]), f"Expected NaN for {col}"


def test_null_partial_nbm_rows_preserves_t50_when_nulling():
    df = pd.DataFrame([_partial_nbm_row(nbm_t50=93.0)])
    result = null_partial_nbm_rows(df)
    assert result.iloc[0]["nbm_t50"] == pytest.approx(93.0)


def test_null_partial_nbm_rows_leaves_fully_valid_rows_unchanged():
    valid_row = _partial_nbm_row(nbm_t10=65.0, nbm_t25=69.0, nbm_t75=78.0, nbm_t90=82.0,
                                  nbm_tmin=60.0, nbm_pop12=0.1, nbm_spread=17.0)
    df = pd.DataFrame([valid_row])
    result = null_partial_nbm_rows(df)
    assert result.iloc[0]["nbm_t10"] == pytest.approx(65.0)
    assert result.iloc[0]["nbm_spread"] == pytest.approx(17.0)


def test_null_partial_nbm_rows_leaves_fully_missing_rows_unchanged():
    # All zeros including t50 means the entire NBM slot was missing — not a partial fill
    fully_missing = _partial_nbm_row(nbm_t50=0.0)
    df = pd.DataFrame([fully_missing])
    result = null_partial_nbm_rows(df)
    assert result.iloc[0]["nbm_t10"] == pytest.approx(0.0)
    assert result.iloc[0]["nbm_t50"] == pytest.approx(0.0)


def test_null_partial_nbm_rows_preserves_nbm_gefs_delta():
    df = pd.DataFrame([_partial_nbm_row(nbm_t50=75.0, gefs_tmax_mean=70.0, nbm_gefs_delta=5.0)])
    result = null_partial_nbm_rows(df)
    assert result.iloc[0]["nbm_gefs_delta"] == pytest.approx(5.0)


def test_null_partial_nbm_rows_preserves_row_count():
    rows = [
        _partial_nbm_row(nbm_t50=75.0),   # partial — will be nulled
        _partial_nbm_row(nbm_t10=65.0, nbm_t25=69.0, nbm_t75=78.0, nbm_t90=82.0,
                         nbm_tmin=60.0, nbm_pop12=0.1, nbm_spread=17.0),  # valid
        _partial_nbm_row(nbm_t50=0.0),    # fully missing
    ]
    df = pd.DataFrame(rows)
    result = null_partial_nbm_rows(df)
    assert len(result) == 3


def test_null_partial_nbm_rows_handles_18_station_batch():
    partial_rows = [_partial_nbm_row(station=f"K{i:03d}") for i in range(18)]
    df = pd.DataFrame(partial_rows)
    result = null_partial_nbm_rows(df)
    assert (result["nbm_t10"].isna()).sum() == 18
    assert (result["nbm_t50"] == 75.0).sum() == 18
