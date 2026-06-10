"""Tests for PR 6: tmin/below market support and multi-day lead evaluation."""
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtest.runner import BacktestRunner


def _make_runner():
    with patch.object(BacktestRunner, "_load_kalshi_prices", return_value=pd.DataFrame()):
        return BacktestRunner(
            settings=SimpleNamespace(MIN_EDGE_CENTS=5, STATIONS=["KNYC"]),
            start_date=date(2020, 1, 1),
            end_date=date(2021, 1, 1),
        )


# ── Gap 4: tmin / "below" market probability computation ─────────────────────

def test_compute_trade_prob_below_flips_direction():
    prob_below = BacktestRunner._compute_trade_prob(
        mu=np.array([70.0]),
        sigma=np.array([2.0]),
        threshold=np.array([70.0]),
        market_type="below",
    )
    prob_above = BacktestRunner._compute_trade_prob(
        mu=np.array([70.0]),
        sigma=np.array([2.0]),
        threshold=np.array([70.0]),
        market_type="above",
    )
    assert prob_below[0] + prob_above[0] == pytest.approx(1.0, abs=1e-6)


def test_compute_trade_prob_above_uses_cdf_complement():
    from scipy.stats import norm
    mu = np.array([72.0])
    sigma = np.array([3.0])
    threshold = np.array([75.0])

    prob = BacktestRunner._compute_trade_prob(mu, sigma, threshold, market_type="above")

    expected = 1.0 - norm.cdf(75.0, loc=72.0, scale=3.0)
    assert prob[0] == pytest.approx(expected, abs=1e-6)


def test_compute_trade_prob_below_uses_cdf_directly():
    from scipy.stats import norm
    mu = np.array([72.0])
    sigma = np.array([3.0])
    threshold = np.array([70.0])

    prob = BacktestRunner._compute_trade_prob(mu, sigma, threshold, market_type="below")

    expected = norm.cdf(70.0, loc=72.0, scale=3.0)
    assert prob[0] == pytest.approx(expected, abs=1e-6)


def test_get_market_mid_returns_none_for_below_market_when_type_is_above():
    runner = _make_runner()
    runner._kalshi_prices = pd.DataFrame([{
        "station": "KNYC",
        "date": date(2023, 6, 1),
        "market_type": "below",
        "threshold": 70.0,
        "d1_mid": 0.45,
    }])
    result = runner._get_market_mid("KNYC", date(2023, 6, 1), threshold=70.0, market_type="above")
    assert result is None


def test_get_market_mid_returns_mid_for_correct_market_type():
    runner = _make_runner()
    runner._kalshi_prices = pd.DataFrame([{
        "station": "KNYC",
        "date": date(2023, 6, 1),
        "market_type": "below",
        "threshold": 70.0,
        "d1_mid": 0.45,
    }])
    result = runner._get_market_mid("KNYC", date(2023, 6, 1), threshold=70.0, market_type="below")
    assert result == pytest.approx(0.45)


# ── Gap 10: multi-day lead hour evaluation ───────────────────────────────────

def test_filter_by_lead_hours_returns_all_specified_leads():
    test_df = pd.DataFrame({
        "date": [date(2023, 6, 1)] * 5,
        "station": "KNYC",
        "lead_hour": [24, 48, 72, 96, 168],
    })
    mu = np.array([70.0, 71.0, 72.0, 73.0, 74.0])
    sigma = np.array([2.0] * 5)

    filtered_df, filtered_mu, filtered_sigma = BacktestRunner._filter_by_lead_hours(
        test_df=test_df, mu=mu, sigma=sigma, lead_hours=[24, 72, 168]
    )

    assert list(filtered_df["lead_hour"]) == [24, 72, 168]
    assert len(filtered_mu) == 3


def test_filter_by_lead_hours_none_returns_all():
    test_df = pd.DataFrame({
        "date": [date(2023, 6, 1)] * 3,
        "station": "KNYC",
        "lead_hour": [24, 48, 72],
    })
    mu = np.array([70.0, 71.0, 72.0])
    sigma = np.array([2.0] * 3)

    filtered_df, filtered_mu, filtered_sigma = BacktestRunner._filter_by_lead_hours(
        test_df=test_df, mu=mu, sigma=sigma, lead_hours=None
    )

    assert len(filtered_df) == 3
    assert len(filtered_mu) == 3


def test_filter_by_lead_hours_no_lead_hour_column_returns_all():
    test_df = pd.DataFrame({"date": [date(2023, 6, 1)] * 2, "station": "KNYC"})
    mu = np.array([70.0, 71.0])
    sigma = np.array([2.0, 2.0])

    filtered_df, filtered_mu, filtered_sigma = BacktestRunner._filter_by_lead_hours(
        test_df=test_df, mu=mu, sigma=sigma, lead_hours=[24]
    )

    assert len(filtered_df) == 2
