"""Tests for official NWS daily-max ingestion (IEM ASOS daily summary).

The model must train on the SAME temperature Kalshi settles on — the official
NWS daily max at the exact settlement station — not the hourly-METAR max we used
before (which disagreed with settlement ~12% of the time). See
plans/nws-settlement-source.md and [[kalshi-bracket-markets]].
"""

import pandas as pd

from ingestion.nws_daily import _parse_iem_daily, SETTLEMENT_STATION


_SAMPLE = (
    "station,day,max_temp_f,min_temp_f,precip_in\n"
    "MDW,2026-05-25,84.0,58.0,0.0\n"
    "MDW,2026-05-26,83.0,62.0,0.0\n"
    "MDW,2026-05-27,82.0,61.0,0.0\n"
)


def test_parse_iem_daily_returns_day_to_max():
    out = _parse_iem_daily(_SAMPLE)
    assert out == {"2026-05-25": 84.0, "2026-05-26": 83.0, "2026-05-27": 82.0}


def test_parse_iem_daily_skips_missing_max():
    csv = (
        "station,day,max_temp_f,min_temp_f\n"
        "XXX,2026-05-25,None,40.0\n"
        "XXX,2026-05-26,,41.0\n"
        "XXX,2026-05-27,70.0,42.0\n"
    )
    out = _parse_iem_daily(csv)
    assert out == {"2026-05-27": 70.0}


def test_parse_iem_daily_empty_returns_empty():
    assert _parse_iem_daily("station,day,max_temp_f\n") == {}


# ── Settlement-station map: must cover every station the bot trades ───────────

def test_settlement_map_covers_all_registry_stations():
    from config.stations import ALL_ICAO
    missing = [s for s in ALL_ICAO if s not in SETTLEMENT_STATION]
    assert missing == []


def test_settlement_map_uses_correct_chicago_and_ny_stations():
    # Kalshi settles Chicago on Midway (MDW) and NY on Central Park (NYC),
    # NOT our default O'Hare (KORD) / LaGuardia (KLGA) airports.
    assert SETTLEMENT_STATION["KORD"] == ("IL_ASOS", "MDW")
    assert SETTLEMENT_STATION["KLGA"] == ("NY_ASOS", "NYC")


def test_official_daily_tmax_series_is_date_indexed(monkeypatch):
    import ingestion.nws_daily as nd
    from datetime import date

    monkeypatch.setattr(
        nd, "fetch_official_daily_tmax_for_icao",
        lambda icao, s, e: {"2026-05-26": 83.0, "2026-05-25": 84.0},
    )
    out = nd.official_daily_tmax_series("KORD", date(2026, 5, 25), date(2026, 5, 26))
    assert list(out.index) == [pd.Timestamp("2026-05-25"), pd.Timestamp("2026-05-26")]
    assert out.loc[pd.Timestamp("2026-05-25")] == 84.0


def test_official_daily_tmax_series_empty(monkeypatch):
    import ingestion.nws_daily as nd
    from datetime import date

    monkeypatch.setattr(nd, "fetch_official_daily_tmax_for_icao", lambda icao, s, e: {})
    assert nd.official_daily_tmax_series("KORD", date(2026, 5, 25), date(2026, 5, 26)).empty
