"""Tests for persisting/retrieving the GEFS D+1 Tmax forecast used to compute
live obs_minus_model_lag1/2/3 residuals on subsequent cycles."""
import datetime as dt

import pytest

from db.session import init_db
from db.forecast_log import record_forecast_tmax, get_forecast_tmax


@pytest.fixture
def db(tmp_path):
    init_db(f"sqlite:///{tmp_path}/test.db")


def test_record_and_retrieve_forecast_tmax(db):
    target = dt.date(2026, 6, 9)
    record_forecast_tmax("KORD", target, 75.0)

    result = get_forecast_tmax("KORD", [target])

    assert result == {target: pytest.approx(75.0)}


def test_record_forecast_tmax_does_not_overwrite_existing_entry(db):
    target = dt.date(2026, 6, 9)
    record_forecast_tmax("KORD", target, 75.0)
    record_forecast_tmax("KORD", target, 999.0)

    result = get_forecast_tmax("KORD", [target])

    assert result[target] == pytest.approx(75.0)


def test_get_forecast_tmax_returns_empty_for_unknown_dates(db):
    result = get_forecast_tmax("KORD", [dt.date(2026, 1, 1)])

    assert result == {}


def test_get_forecast_tmax_filters_by_station(db):
    target = dt.date(2026, 6, 9)
    record_forecast_tmax("KORD", target, 75.0)
    record_forecast_tmax("KLAX", target, 80.0)

    result = get_forecast_tmax("KORD", [target])

    assert result == {target: pytest.approx(75.0)}
