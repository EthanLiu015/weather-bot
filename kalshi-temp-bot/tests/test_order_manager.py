import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from trading.order_manager import OrderManager
from strategies.shared_state import SharedState


def _make_settings():
    s = MagicMock()
    s.MIN_EDGE_CENTS = 4.0
    s.MAX_CI_WIDTH = 0.12
    s.KELLY_FRACTION = 0.25
    s.MAX_EXPOSURE_PER_TICKER_USD = 200.0
    s.HORIZON_MULTIPLIERS = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.3, 5: 0.2}
    s.BOT_ACTIVE = True
    return s


async def test_never_submits_market_orders():
    client = MagicMock()
    client.create_order = AsyncMock(return_value={"order_id": "abc123"})
    shared_state = SharedState(min_edge=0.04, max_ci_width=0.12)
    shared_state.update_fair_a("TEST-TICKER", 0.75, 0.05, 1)
    shared_state.update_market("TEST-TICKER", 0.55, 0.57)

    risk = MagicMock()
    risk.can_trade.return_value = (True, "")
    settings = _make_settings()

    om = OrderManager(client, shared_state, risk, settings)

    with patch("trading.order_manager.get_session") as mock_session:
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        await om.process_tick()

    if client.create_order.called:
        call_kwargs = client.create_order.call_args.kwargs
        assert call_kwargs.get("order_type") == "limit", "Must only submit limit orders"


async def test_cancels_stale_limit():
    client = MagicMock()
    client.create_order = AsyncMock(return_value={"order_id": "new123"})
    client.cancel_order = AsyncMock(return_value={})
    shared_state = SharedState(min_edge=0.04, max_ci_width=0.12)
    shared_state.update_fair_a("STALE-TICKER", 0.75, 0.05, 1)
    shared_state.update_market("STALE-TICKER", 0.55, 0.57)

    risk = MagicMock()
    risk.can_trade.return_value = (True, "")
    settings = _make_settings()

    om = OrderManager(client, shared_state, risk, settings)
    # Inject stale resting order at different price
    om._resting_orders["STALE-TICKER"] = {
        "order_id": "old123",
        "price": 60,  # actual fair is ~75, drift = 15 > 1 cent
        "side": "yes",
        "count": 1,
    }

    with patch("trading.order_manager.get_session") as mock_session:
        mock_ctx = MagicMock()
        mock_session.return_value.__enter__ = MagicMock(return_value=mock_ctx)
        mock_session.return_value.__exit__ = MagicMock(return_value=False)
        await om.process_tick()

    client.cancel_order.assert_awaited_once_with("old123")


async def test_no_trade_when_edge_below_minimum():
    client = MagicMock()
    client.create_order = AsyncMock()
    shared_state = SharedState(min_edge=0.04, max_ci_width=0.12)
    # Set fair close to mid — less than 4¢ edge
    shared_state.update_fair_a("LOW-EDGE", 0.51, 0.05, 1)
    shared_state.update_market("LOW-EDGE", 0.50, 0.52)

    risk = MagicMock()
    risk.can_trade.return_value = (True, "")
    settings = _make_settings()
    om = OrderManager(client, shared_state, risk, settings)

    await om.process_tick()
    client.create_order.assert_not_awaited()


async def test_no_trade_when_ci_too_wide():
    client = MagicMock()
    client.create_order = AsyncMock()
    shared_state = SharedState(min_edge=0.04, max_ci_width=0.12)
    # Wide CI
    shared_state.update_fair_a("WIDE-CI", 0.75, 0.20, 1)
    shared_state.update_market("WIDE-CI", 0.50, 0.52)

    risk = MagicMock()
    risk.can_trade.return_value = (True, "")
    settings = _make_settings()
    om = OrderManager(client, shared_state, risk, settings)

    await om.process_tick()
    client.create_order.assert_not_awaited()
