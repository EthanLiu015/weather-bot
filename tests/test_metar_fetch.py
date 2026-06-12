"""Tests for extracting METAR/SPECI report lines from aviationweather.gov.

The legacy `cgi-bin/data/metar.php` endpoint (lines starting directly with
the station id) now returns 410 Gone. The replacement `/api/data/metar`
endpoint prefixes each line with a report type ("METAR "/"SPECI ") before
the station id, so fetch_metar's `line.startswith(station)` filter silently
matched nothing and every live ASOS fetch returned zero observations.
"""
from ingestion.asos import _extract_metar_lines


def test_extract_metar_lines_strips_metar_prefix():
    raw = "METAR KORD 111451Z 23011G15KT 10SM FEW150 BKN250 24/19 A2979 RMK AO2"

    lines = _extract_metar_lines(raw, "KORD")

    assert lines == ["KORD 111451Z 23011G15KT 10SM FEW150 BKN250 24/19 A2979 RMK AO2"]


def test_extract_metar_lines_strips_speci_prefix():
    raw = "SPECI KORD 111347Z 24009KT 10SM SCT012 SCT250 22/19 A2977 RMK AO2"

    lines = _extract_metar_lines(raw, "KORD")

    assert lines == ["KORD 111347Z 24009KT 10SM SCT012 SCT250 22/19 A2977 RMK AO2"]


def test_extract_metar_lines_handles_multiple_lines_and_filters_station():
    raw = "\n".join([
        "METAR KORD 111451Z 23011G15KT 10SM FEW150 BKN250 24/19 A2979 RMK AO2",
        "SPECI KORD 111347Z 24009KT 10SM SCT012 SCT250 22/19 A2977 RMK AO2",
        "METAR KMDW 111451Z 22008KT 10SM FEW040 25/18 A2978 RMK AO2",
    ])

    lines = _extract_metar_lines(raw, "KORD")

    assert len(lines) == 2
    assert all(ln.startswith("KORD") for ln in lines)


def test_extract_metar_lines_handles_legacy_format_without_report_type():
    raw = "KORD 111451Z 23011G15KT 10SM FEW150 BKN250 24/19 A2979 RMK AO2"

    lines = _extract_metar_lines(raw, "KORD")

    assert lines == [raw]


def test_extract_metar_lines_handles_empty_input():
    assert _extract_metar_lines("", "KORD") == []


def test_extract_metar_lines_skips_blank_lines():
    raw = "\n\nMETAR KORD 111451Z 23011G15KT 10SM FEW150 BKN250 24/19 A2979 RMK AO2\n\n"

    lines = _extract_metar_lines(raw, "KORD")

    assert lines == ["KORD 111451Z 23011G15KT 10SM FEW150 BKN250 24/19 A2979 RMK AO2"]
