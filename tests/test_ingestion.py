import pytest
from ingestion.asos import parse_metar_string, compute_running_tmax, estimate_remaining_hours
from ingestion.qc import validate_metar_obs, qc_metar_list


def test_parse_metar_temp_and_dewpoint():
    raw = "KORD 281552Z 27012KT 10SM FEW050 24/12 A2992 RMK AO2"
    obs = parse_metar_string(raw)
    assert obs["temp_c"] == pytest.approx(24.0)
    assert obs["dewpoint_c"] == pytest.approx(12.0)
    assert obs["temp_f"] == pytest.approx(24 * 9 / 5 + 32)


def test_parse_metar_negative_temp():
    raw = "KORD 010000Z 00000KT 10SM CLR M05/M10 A2992"
    obs = parse_metar_string(raw)
    assert obs["temp_c"] == pytest.approx(-5.0)
    assert obs["dewpoint_c"] == pytest.approx(-10.0)


def test_parse_metar_fog_detection():
    raw = "KJFK 010000Z 00005KT 1/4SM FG OVC002 10/09 A2990"
    obs = parse_metar_string(raw)
    assert obs["fog"] is True


def test_compute_running_tmax():
    obs = [
        {"temp_f": 72.5},
        {"temp_f": 75.0},
        {"temp_f": 68.0},
        {"temp_f": 70.3},
    ]
    assert compute_running_tmax(obs) == pytest.approx(75.0)


def test_compute_running_tmax_empty():
    import math
    result = compute_running_tmax([])
    assert math.isnan(result)


def test_estimate_remaining_hours_returns_nonnegative():
    hours = estimate_remaining_hours(12, "America/Chicago")
    assert hours >= 0


def test_qc_rejects_out_of_range_temp():
    obs = {"temp_f": 200.0, "dewpoint_f": 60.0, "wind_speed_kt": 10}
    valid, issues = validate_metar_obs(obs)
    assert not valid
    assert any("temp_f" in i for i in issues)


def test_qc_accepts_valid_obs():
    obs = {"temp_f": 72.0, "dewpoint_f": 55.0, "wind_speed_kt": 10}
    valid, issues = validate_metar_obs(obs)
    assert valid
    assert issues == []


def test_qc_metar_list_filters_bad():
    obs_list = [
        {"temp_f": 72.0, "dewpoint_f": 55.0, "wind_speed_kt": 5},
        {"temp_f": 999.0, "dewpoint_f": 55.0, "wind_speed_kt": 5},
        {"temp_f": 68.0, "dewpoint_f": 50.0, "wind_speed_kt": 8},
    ]
    cleaned = qc_metar_list(obs_list)
    assert len(cleaned) == 2
