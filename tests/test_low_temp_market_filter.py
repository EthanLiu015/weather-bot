"""Tests excluding KXLOWT* (low-temperature) markets from active processing.

The NGBoost/QRF ensemble is trained and calibrated on Tmax only, so applying
it to KXLOWT* thresholds produces meaningless fair values. Until a dedicated
low-temperature model exists, the bot should not query, seed, or quote
KXLOWT* series at all.
"""
import pytest
from unittest.mock import AsyncMock

from config.series import is_low_temp_series
from strategies.ensemble_strategy import EnsembleStrategy


# ── is_low_temp_series (pure) ────────────────────────────────────────────────

@pytest.mark.parametrize("series", [
    "KXLOWTCHI", "KXLOWTNYC", "KXLOWTLAX", "KXLOWTHOU", "KXLOWTOKC",
])
def test_is_low_temp_series_true_for_lowt_prefixes(series):
    assert is_low_temp_series(series) is True


@pytest.mark.parametrize("series", [
    "KXHIGHCHI", "KXHIGHNY", "KXHIGHNY0", "KXDENHIGH", "KXHOUHIGH",
    "KXHIGHOU", "KXHIGHTATL", "KXHIGHTOKC",
])
def test_is_low_temp_series_false_for_high_temp_prefixes(series):
    assert is_low_temp_series(series) is False


# ── fetch_active_temperature_tickers excludes KXLOWT* ────────────────────────

def _make_strategy(client) -> EnsembleStrategy:
    return EnsembleStrategy(shared_state=None, model_registry={}, kalshi_client=client, settings=None)


@pytest.mark.asyncio
async def test_fetch_active_tickers_never_queries_low_temp_series():
    queried_series = []

    async def fake_request(method, path, params=None, **kwargs):
        queried_series.append(params["series_ticker"])
        return {"markets": []}

    client = AsyncMock()
    client._request = AsyncMock(side_effect=fake_request)
    strategy = _make_strategy(client)

    await strategy.fetch_active_temperature_tickers()

    assert "KXHIGHCHI" in queried_series
    assert not any(is_low_temp_series(s) for s in queried_series)


@pytest.mark.asyncio
async def test_fetch_active_tickers_returns_market_dicts_with_strike_fields():
    # fetch must surface the bracket structure (strike_type/floor/cap) so the
    # bot can price each ticker as the real Kalshi bracket it is.
    async def fake_request(method, path, params=None, **kwargs):
        if params["series_ticker"] == "KXHIGHCHI":
            return {"markets": [{
                "ticker": "KXHIGHCHI-26JUN11-T81", "strike_type": "greater",
                "floor_strike": 81, "cap_strike": None,
            }]}
        return {"markets": []}

    client = AsyncMock()
    client._request = AsyncMock(side_effect=fake_request)
    strategy = _make_strategy(client)

    markets = await strategy.fetch_active_temperature_tickers()

    assert len(markets) == 1
    assert markets[0]["ticker"] == "KXHIGHCHI-26JUN11-T81"
    assert markets[0]["strike_type"] == "greater"
    assert markets[0]["floor_strike"] == 81.0


@pytest.mark.asyncio
async def test_fetch_active_tickers_skips_markets_without_strike_type():
    async def fake_request(method, path, params=None, **kwargs):
        if params["series_ticker"] == "KXHIGHCHI":
            return {"markets": [{"ticker": "KXHIGHCHI-26JUN11-T81"}]}  # no strike_type
        return {"markets": []}

    client = AsyncMock()
    client._request = AsyncMock(side_effect=fake_request)
    strategy = _make_strategy(client)

    markets = await strategy.fetch_active_temperature_tickers()

    assert markets == []
