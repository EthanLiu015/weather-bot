"""Behavioural tests for the 1-minute ASOS fetcher's pure parsing/aggregation.

The 1-minute feed is the freshest running-max signal we can trade on. The network
call is thin; the logic worth pinning is (a) turning IEM's CSV — with its 'M'
missing markers — into clean numeric obs, and (b) collapsing those obs into the
running (cumulative) max that the intraday model conditions on.
"""
import pandas as pd

from ingestion.asos_1min import parse_1min_csv, running_max, icao_to_1min_code


SAMPLE = """station,station_name,valid(UTC),tmpf
MSP,MINNEAPOLIS,2026-06-01 18:00,76
MSP,MINNEAPOLIS,2026-06-01 18:01,M
MSP,MINNEAPOLIS,2026-06-01 18:02,78
MSP,MINNEAPOLIS,2026-06-01 18:03,77
"""


def test_parse_coerces_types_and_drops_missing():
    df = parse_1min_csv(SAMPLE)
    # The 'M' row is dropped; three real obs remain.
    assert list(df["tmpf"]) == [76.0, 78.0, 77.0]
    assert str(df["valid_utc"].dtype).startswith("datetime64")
    assert df["valid_utc"].iloc[0] == pd.Timestamp("2026-06-01 18:00")


def test_parse_empty_body_returns_empty_frame():
    df = parse_1min_csv("station,station_name,valid(UTC),tmpf\n")
    assert len(df) == 0
    assert list(df.columns) == ["valid_utc", "tmpf"]


def test_running_max_is_cumulative_in_time_order():
    df = parse_1min_csv(SAMPLE)
    rm = running_max(df)
    # 76 -> (78 dropped-from-cummax? no) -> cummax over 76,78,77 = 76,78,78
    assert list(rm["run_max"]) == [76.0, 78.0, 78.0]


def test_running_max_sorts_out_of_order_input():
    unsorted = pd.DataFrame({
        "valid_utc": pd.to_datetime(["2026-06-01 18:03", "2026-06-01 18:00"]),
        "tmpf": [77.0, 76.0],
    })
    rm = running_max(unsorted)
    # After sorting by time: 76 then 77 -> cummax 76, 77
    assert list(rm["valid_utc"]) == [pd.Timestamp("2026-06-01 18:00"), pd.Timestamp("2026-06-01 18:03")]
    assert list(rm["run_max"]) == [76.0, 77.0]


def test_icao_to_1min_code_strips_leading_k():
    assert icao_to_1min_code("KMSP") == "MSP"
    assert icao_to_1min_code("KLGA") == "LGA"
    # Already 3-char: unchanged.
    assert icao_to_1min_code("MSP") == "MSP"
