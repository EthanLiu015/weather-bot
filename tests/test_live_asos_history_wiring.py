"""Tests wiring live obs_minus_model_lag1/2/3 history into EnsembleStrategy.

run_cycle previously hardcoded asos_history=pd.DataFrame(), so the model's
single most important feature group (~38% combined NGBoost importance) was
always 0 in production. These tests cover the two pieces that build a real
asos_history each cycle:
  - _record_tomorrows_forecast: persist today's GEFS D+1 Tmax forecast
  - _build_live_asos_history: diff persisted forecasts against observed Tmax
"""
import datetime as dt
from unittest.mock import patch

import pandas as pd
import pytest

from db.session import init_db
from db.forecast_log import record_forecast_tmax, get_forecast_tmax
from strategies.ensemble_strategy import EnsembleStrategy


@pytest.fixture
def db(tmp_path):
    init_db(f"sqlite:///{tmp_path}/test.db")


# ── _record_tomorrows_forecast ───────────────────────────────────────────────

def test_record_tomorrows_forecast_persists_d1_gefs_tmax_mean(db):
    feature_df = pd.DataFrame([
        {"station": "KORD", "lead_hour": 24, "gefs_tmax_mean": 75.0},
        {"station": "KORD", "lead_hour": 48, "gefs_tmax_mean": 78.0},
    ])
    today = dt.date(2026, 6, 10)

    EnsembleStrategy._record_tomorrows_forecast(feature_df, today=today)

    result = get_forecast_tmax("KORD", [dt.date(2026, 6, 11)])
    assert result == {dt.date(2026, 6, 11): pytest.approx(75.0)}


def test_record_tomorrows_forecast_does_not_overwrite_existing_entry(db):
    today = dt.date(2026, 6, 10)
    record_forecast_tmax("KORD", dt.date(2026, 6, 11), 70.0)

    feature_df = pd.DataFrame([{"station": "KORD", "lead_hour": 24, "gefs_tmax_mean": 999.0}])
    EnsembleStrategy._record_tomorrows_forecast(feature_df, today=today)

    result = get_forecast_tmax("KORD", [dt.date(2026, 6, 11)])
    assert result[dt.date(2026, 6, 11)] == pytest.approx(70.0)


def test_record_tomorrows_forecast_skips_stations_without_d1_row(db):
    feature_df = pd.DataFrame([{"station": "KORD", "lead_hour": 48, "gefs_tmax_mean": 78.0}])
    today = dt.date(2026, 6, 10)

    EnsembleStrategy._record_tomorrows_forecast(feature_df, today=today)

    assert get_forecast_tmax("KORD", [dt.date(2026, 6, 11)]) == {}


# ── _build_live_asos_history ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_build_live_asos_history_combines_actual_and_persisted_forecast(db):
    record_forecast_tmax("KORD", dt.date(2026, 6, 9), 72.0)

    async def fake_fetch(station, days=4):
        return {dt.date(2026, 6, 9): 75.0}

    with patch("strategies.ensemble_strategy.fetch_daily_tmax_history", side_effect=fake_fetch):
        history = await EnsembleStrategy._build_live_asos_history(["KORD"])

    assert history.loc[pd.Timestamp(dt.date(2026, 6, 9)), "KORD"] == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_build_live_asos_history_empty_when_no_forecast_recorded_yet(db):
    async def fake_fetch(station, days=4):
        return {dt.date(2026, 6, 9): 75.0}

    with patch("strategies.ensemble_strategy.fetch_daily_tmax_history", side_effect=fake_fetch):
        history = await EnsembleStrategy._build_live_asos_history(["KORD"])

    assert history.empty
