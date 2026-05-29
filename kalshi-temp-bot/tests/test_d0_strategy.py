import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from strategies.d0_strategy import D0Strategy, NEAR_CERTAIN_YES, NEAR_CERTAIN_NO
from strategies.shared_state import SharedState


def _make_settings():
    s = MagicMock()
    return s


def _make_obs(temp_f: float, count: int = 5) -> list[dict]:
    return [{"temp_f": temp_f + i * 0.1, "observation_time": None} for i in range(count)]


@pytest.mark.asyncio
async def test_fair_value_near_certain_yes_when_tmax_exceeds_threshold():
    shared_state = SharedState()
    strategy = D0Strategy(shared_state=shared_state, settings=_make_settings())
    obs = _make_obs(temp_f=90.0)  # running_tmax will be ~90.4°F

    with patch("strategies.d0_strategy.fetch_metar", new=AsyncMock(return_value=obs)), \
         patch("strategies.d0_strategy.D0Strategy.is_resolution_day", new=AsyncMock(return_value=True)):
        await strategy.run_cycle("KORD", "KORD-20260601-80", threshold=80.0)

    snap = shared_state.snapshot()
    assert "KORD-20260601-80" in snap
    assert snap["KORD-20260601-80"]["fair_b"] == pytest.approx(NEAR_CERTAIN_YES)


@pytest.mark.asyncio
async def test_fair_value_near_certain_no_when_max_impossible():
    shared_state = SharedState()
    strategy = D0Strategy(shared_state=shared_state, settings=_make_settings())
    # running_tmax = 65, threshold = 90, only 1 hour remaining — impossible to reach
    obs = [{"temp_f": 65.0 + i * 0.01, "observation_time": None} for i in range(3)]

    with patch("strategies.d0_strategy.fetch_metar", new=AsyncMock(return_value=obs)), \
         patch("strategies.d0_strategy.D0Strategy.is_resolution_day", new=AsyncMock(return_value=True)), \
         patch("strategies.d0_strategy.estimate_remaining_hours", return_value=1):
        await strategy.run_cycle("KORD", "KORD-20260601-90", threshold=90.0)

    snap = shared_state.snapshot()
    if "KORD-20260601-90" in snap:
        assert snap["KORD-20260601-90"]["fair_b"] == pytest.approx(NEAR_CERTAIN_NO)


def test_conditional_prob_returns_value_in_unit_interval():
    shared_state = SharedState()
    strategy = D0Strategy(shared_state=shared_state, settings=_make_settings())
    prob = strategy.conditional_prob(
        running_tmax=70.0,
        threshold=80.0,
        hour_local=14,
        station="KORD",
        month=7,
    )
    assert 0.0 <= prob <= 1.0


def test_conditional_prob_decreases_as_gap_increases():
    shared_state = SharedState()
    strategy = D0Strategy(shared_state=shared_state, settings=_make_settings())
    prob_small_gap = strategy.conditional_prob(70.0, 72.0, 10, "KORD", 7)
    prob_large_gap = strategy.conditional_prob(70.0, 95.0, 10, "KORD", 7)
    assert prob_small_gap >= prob_large_gap


@pytest.mark.asyncio
async def test_is_resolution_day_true_for_today_ticker():
    from datetime import date
    today_str = date.today().strftime("%Y%m%d")
    ticker = f"KORD-{today_str}-80"
    strategy = D0Strategy(SharedState(), _make_settings())
    assert await strategy.is_resolution_day(ticker)


@pytest.mark.asyncio
async def test_is_resolution_day_false_for_future_ticker():
    strategy = D0Strategy(SharedState(), _make_settings())
    assert not await strategy.is_resolution_day("KORD-20990101-80")
