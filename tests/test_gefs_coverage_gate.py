"""Tests for per-lead-bucket GEFS coverage gating and self-healing retry.

When NOAA hasn't yet published the longer-lead GEFS forecast hours (D3-D7),
run_cycle should be able to detect this per lead bucket so it can skip
quoting off empty/zero-filled features for those tickers. Separately, the
strategy tracks overall coverage so the scheduler can keep retrying ingestion
(cheap, since fetch_latest_gefs_run only re-downloads missing files) until
coverage recovers, instead of waiting for the next 6-hourly GEFS cycle.
"""
import pytest

from ingestion.gefs import FORECAST_HOURS
from strategies.ensemble_strategy import EnsembleStrategy


def _make_gefs_raw(stations: list[str], available_hours: list[int]) -> dict:
    return {
        s: {
            lh: ([{"member": f"m{i}"} for i in range(31)] if lh in available_hours else [])
            for lh in FORECAST_HOURS
        }
        for s in stations
    }


# ── Per-bucket coverage computation ──────────────────────────────────────────

def test_compute_coverage_by_bucket_full_coverage_is_one_everywhere():
    stations = ["KORD", "KJFK"]
    gefs_raw = _make_gefs_raw(stations, FORECAST_HOURS)

    coverage = EnsembleStrategy._compute_coverage_by_bucket(gefs_raw, stations)

    assert coverage["D1-2"] == pytest.approx(1.0)
    assert coverage["D3-4"] == pytest.approx(1.0)
    assert coverage["D5-7"] == pytest.approx(1.0)


def test_compute_coverage_by_bucket_isolates_missing_d5_7():
    stations = ["KORD", "KJFK"]
    # Only D1-2 (6,12,24,48) and D3-4 (72,96) hours published; D5-7 (120,168) missing
    gefs_raw = _make_gefs_raw(stations, [6, 12, 24, 48, 72, 96])

    coverage = EnsembleStrategy._compute_coverage_by_bucket(gefs_raw, stations)

    assert coverage["D1-2"] == pytest.approx(1.0)
    assert coverage["D3-4"] == pytest.approx(1.0)
    assert coverage["D5-7"] == pytest.approx(0.0)


def test_compute_coverage_by_bucket_no_data_is_zero_everywhere():
    stations = ["KORD"]
    gefs_raw = _make_gefs_raw(stations, [])

    coverage = EnsembleStrategy._compute_coverage_by_bucket(gefs_raw, stations)

    assert coverage == {"D1-2": 0.0, "D3-4": 0.0, "D5-7": 0.0}


# ── Self-healing retry trigger ───────────────────────────────────────────────

def _make_strategy() -> EnsembleStrategy:
    return EnsembleStrategy(shared_state=None, model_registry={}, kalshi_client=None, settings=None)


def test_needs_retry_defaults_to_false_before_any_cycle_runs():
    strategy = _make_strategy()

    assert strategy.needs_retry() is False


def test_needs_retry_true_when_last_coverage_below_threshold():
    strategy = _make_strategy()
    strategy.last_gefs_coverage_pct = 0.37

    assert strategy.needs_retry() is True


def test_needs_retry_false_when_last_coverage_high():
    strategy = _make_strategy()
    strategy.last_gefs_coverage_pct = 0.95

    assert strategy.needs_retry() is False
