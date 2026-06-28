import math
from pathlib import Path

import pytest

from ingestion.nbm import (
    _TMAX_FILTER,
    _TMIN_FILTER,
    _POP12_FILTER,
    _derive_nbm_percentiles,
    _grib_values_at_indices,
    _grid_nearest_indices,
    _kelvin_to_f,
    _nbm_lead_fhours,
    _parse_nbm_files_for_stations,
)

FIXTURES = Path(__file__).parent / "fixtures"
TMAX_BLOCK = str(FIXTURES / "nbm_tmax_block.grib2")
TMIN_BLOCK = str(FIXTURES / "nbm_tmin_block.grib2")
TMAX_POP_BLOCK = str(FIXTURES / "nbm_tmax_pop_block.grib2")
POP_MULTI_BLOCK = str(FIXTURES / "nbm_pop_multi_block.grib2")
TRUNCATED_BLOCK = str(FIXTURES / "nbm_truncated_block.grib2")

# Synthetic fixtures are 3x3 grids centred on this point.
FIXTURE_LAT = 19.2518
FIXTURE_LON = 233.7476
POINTS = {"FIXTURE": (FIXTURE_LAT, FIXTURE_LON)}


def test_derive_nbm_percentiles_from_mean_and_spread():
    percentiles = _derive_nbm_percentiles(t50=50.0, spread=10.0)

    assert percentiles["t10"] == pytest.approx(50.0 - 1.2815515655446004 * 10.0)
    assert percentiles["t25"] == pytest.approx(50.0 - 0.6744897501960817 * 10.0)
    assert percentiles["t75"] == pytest.approx(50.0 + 0.6744897501960817 * 10.0)
    assert percentiles["t90"] == pytest.approx(50.0 + 1.2815515655446004 * 10.0)


def test_derive_nbm_percentiles_nan_when_inputs_missing():
    percentiles = _derive_nbm_percentiles(t50=float("nan"), spread=10.0)

    assert all(math.isnan(v) for v in percentiles.values())


def test_grid_nearest_indices_returns_center_index_and_signature():
    indices, signature = _grid_nearest_indices(TMAX_BLOCK, POINTS)

    assert indices == {"FIXTURE": 4}
    assert signature[:2] == (3, 3)  # Ni, Nj


def test_unflip_alternating_rows_reverses_odd_rows():
    # Boustrophedon storage: row0 left→right, row1 stored reversed, row2 normal.
    # raw = [row0 | row1-reversed | row2] for a 3-wide, 3-tall grid.
    from ingestion.nbm import _unflip_alternating_rows
    raw = [1, 2, 3,  6, 5, 4,  7, 8, 9]
    out = _unflip_alternating_rows(raw, ni=3, nj=3).tolist()
    assert out == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_grib_values_at_indices_extracts_tmax_mean():
    indices, signature = _grid_nearest_indices(TMAX_BLOCK, POINTS)

    values = _grib_values_at_indices(TMAX_BLOCK, _TMAX_FILTER, indices, signature, derived_forecast=None)

    assert values["FIXTURE"] == pytest.approx(295.5)


def test_grib_values_at_indices_extracts_tmax_stddev():
    indices, signature = _grid_nearest_indices(TMAX_BLOCK, POINTS)

    values = _grib_values_at_indices(TMAX_BLOCK, _TMAX_FILTER, indices, signature, derived_forecast=2)

    assert values["FIXTURE"] == pytest.approx(1.5302734375)


def test_grib_values_at_indices_extracts_tmin_mean():
    indices, signature = _grid_nearest_indices(TMIN_BLOCK, POINTS)

    values = _grib_values_at_indices(TMIN_BLOCK, _TMIN_FILTER, indices, signature, derived_forecast=None)

    assert values["FIXTURE"] == pytest.approx(288.0)


def test_grib_values_at_indices_extracts_pop12_not_pop1_or_pop6():
    # Real NBM files carry separate APCP "prob >0.254" messages for the 1hr,
    # 6hr, and 12hr accumulation windows, sharing all keys except
    # lengthOfTimeRange. _POP12_FILTER must select the 12hr one specifically.
    indices, signature = _grid_nearest_indices(POP_MULTI_BLOCK, POINTS)

    values = _grib_values_at_indices(POP_MULTI_BLOCK, _POP12_FILTER, indices, signature, derived_forecast=None)

    assert values["FIXTURE"] == pytest.approx(16.0)


def test_grib_values_at_indices_returns_nan_when_no_message_matches():
    indices, signature = _grid_nearest_indices(TMAX_BLOCK, POINTS)

    values = _grib_values_at_indices(TMAX_BLOCK, _TMIN_FILTER, indices, signature, derived_forecast=None)

    assert math.isnan(values["FIXTURE"])


def test_grib_values_at_indices_returns_nan_for_truncated_file():
    # A range fetch that fails mid-transfer can leave a partial/corrupted
    # GRIB2 file on disk. eccodes raises GribInternalError subclasses
    # (e.g. PrematureEndOfFileError, WrongLengthError) for these — they must
    # not propagate and crash the caller.
    indices, signature = _grid_nearest_indices(TMAX_BLOCK, POINTS)

    values = _grib_values_at_indices(TRUNCATED_BLOCK, _TMAX_FILTER, indices, signature, derived_forecast=None)

    assert math.isnan(values["FIXTURE"])


def test_grib_values_at_indices_returns_nan_for_missing_file():
    indices, signature = _grid_nearest_indices(TMAX_BLOCK, POINTS)

    values = _grib_values_at_indices("does/not/exist.grib2", _TMAX_FILTER, indices, signature, derived_forecast=None)

    assert math.isnan(values["FIXTURE"])


def test_grib_values_at_indices_returns_nan_on_grid_signature_mismatch():
    indices, _ = _grid_nearest_indices(TMAX_BLOCK, POINTS)
    mismatched_signature = (99, 99, 0.0, 0.0, "lambert")

    values = _grib_values_at_indices(TMAX_BLOCK, _TMAX_FILTER, indices, mismatched_signature, derived_forecast=None)

    assert math.isnan(values["FIXTURE"])


def test_parse_nbm_files_for_stations_returns_real_values_for_all_keys():
    indices, signature = _grid_nearest_indices(TMAX_POP_BLOCK, POINTS)

    parsed = _parse_nbm_files_for_stations(TMAX_POP_BLOCK, TMIN_BLOCK, indices, signature)["FIXTURE"]

    expected_t50 = _kelvin_to_f(295.5)
    expected_spread = 1.5302734375 * 9 / 5
    expected_tmin = _kelvin_to_f(288.0)

    assert parsed["t50"] == pytest.approx(expected_t50)
    assert parsed["tmax"] == pytest.approx(expected_t50)
    assert parsed["spread"] == pytest.approx(expected_spread)
    assert parsed["tmin"] == pytest.approx(expected_tmin)
    assert parsed["pop12"] == pytest.approx(16.0)
    assert parsed["t10"] == pytest.approx(expected_t50 - 1.2815515655446004 * expected_spread)
    assert parsed["t90"] == pytest.approx(expected_t50 + 1.2815515655446004 * expected_spread)


def test_parse_nbm_files_for_stations_tmin_nan_when_no_tmin_path():
    indices, signature = _grid_nearest_indices(TMAX_POP_BLOCK, POINTS)

    parsed = _parse_nbm_files_for_stations(TMAX_POP_BLOCK, None, indices, signature)["FIXTURE"]

    assert math.isnan(parsed["tmin"])
    assert parsed["t50"] == pytest.approx(_kelvin_to_f(295.5))


# fhours are shifted +24 from the raw window math so the TMAX window covers the
# VERIFICATION day (init day + lead_hour), not the init day — see _nbm_lead_fhours.

def test_nbm_lead_fhours_00z_cycle():
    # 00z: verification-day TMAX windows are "36-48hr", "60-72hr", ...
    assert _nbm_lead_fhours(cycle=0, n_days=5) == [
        (24, 48, 36), (48, 72, 60), (72, 96, 84), (96, 120, 108), (120, 144, 132),
    ]


def test_nbm_lead_fhours_12z_cycle_reverses_tmax_and_tmin_files():
    # 12z: TMAX windows land on the odd 12h block; still advanced one day (+24).
    assert _nbm_lead_fhours(cycle=12, n_days=5) == [
        (24, 36, 48), (48, 60, 72), (72, 84, 96), (96, 108, 120), (120, 132, 144),
    ]


def test_nbm_lead_fhours_06z_cycle():
    assert _nbm_lead_fhours(cycle=6, n_days=5) == [
        (18, 42, 30), (42, 66, 54), (66, 90, 78), (90, 114, 102), (114, 138, 126),
    ]


def test_nbm_lead_fhours_18z_cycle():
    assert _nbm_lead_fhours(cycle=18, n_days=5) == [
        (18, 30, 42), (42, 54, 66), (66, 78, 90), (90, 102, 114), (114, 126, 138),
    ]
