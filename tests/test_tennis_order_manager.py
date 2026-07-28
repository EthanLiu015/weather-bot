import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from trading.tennis_order_manager import TennisOrderManager


def _make_settings(enabled=True, size=1, hold_seconds=30, max_concurrent=5):
    s = MagicMock()
    s.TENNIS_ENABLED = enabled
    s.TENNIS_CONTRACT_SIZE = size
    s.TENNIS_HOLD_SECONDS = hold_seconds
    s.TENNIS_MAX_CONCURRENT_POSITIONS = max_concurrent
    return s


def _make_manager(open_positions=None, can_trade=(True, ""), settings=None):
    client = AsyncMock()
    risk = MagicMock()
    risk.can_trade.return_value = can_trade
    positions = MagicMock()
    positions.get_all_positions.return_value = open_positions or []
    settings = settings or _make_settings()
    mgr = TennisOrderManager(kalshi_client=client, risk_controls=risk,
                              position_tracker=positions, settings=settings)
    return mgr, client, risk, positions, settings


@pytest.fixture(autouse=True)
def _no_real_db():
    with patch("trading.tennis_order_manager.get_session", MagicMock()):
        yield


@pytest.mark.asyncio
async def test_yes_side_signal_is_skipped_not_traded():
    mgr, client, risk, positions, _ = _make_manager()
    await mgr.on_signal("TICKER1", "yes", datetime.utcnow())
    client.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_risk_gate_rejection_blocks_entry():
    mgr, client, risk, positions, _ = _make_manager(can_trade=(False, "Kill switch active"))
    await mgr.on_signal("TICKER1", "no", datetime.utcnow())
    client.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_concurrent_position_cap_blocks_entry():
    settings = _make_settings(max_concurrent=2)
    open_positions = [
        {"ticker": "A", "net_contracts": -1},
        {"ticker": "B", "net_contracts": -1},
    ]
    mgr, client, risk, positions, _ = _make_manager(open_positions=open_positions, settings=settings)
    await mgr.on_signal("TICKER1", "no", datetime.utcnow())
    client.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_flat_positions_do_not_count_against_cap():
    settings = _make_settings(max_concurrent=1)
    open_positions = [{"ticker": "A", "net_contracts": 0}]  # closed, shouldn't count
    mgr, client, risk, positions, _ = _make_manager(open_positions=open_positions, settings=settings)
    client.get_market.return_value = {"status": "open", "no_ask": 40}
    await mgr.on_signal("TICKER1", "no", datetime.utcnow())
    client.create_order.assert_awaited()


@pytest.mark.asyncio
async def test_existing_position_on_same_ticker_blocks_reentry():
    open_positions = [{"ticker": "TICKER1", "net_contracts": -1}]
    mgr, client, risk, positions, _ = _make_manager(open_positions=open_positions)
    await mgr.on_signal("TICKER1", "no", datetime.utcnow())
    client.create_order.assert_not_called()


@pytest.mark.asyncio
async def test_normal_entry_buys_no_at_current_ask():
    mgr, client, risk, positions, settings = _make_manager()
    client.get_market.return_value = {"status": "open", "no_ask": 40}
    client.create_order.return_value = {"order_id": "PAPER-1", "paper": True}
    await mgr.on_signal("TICKER1", "no", datetime.utcnow())
    client.create_order.assert_awaited_once_with(
        ticker="TICKER1", side="no", price=40, count=settings.TENNIS_CONTRACT_SIZE)


@pytest.mark.asyncio
async def test_exit_nets_correct_pnl_minus_two_fee_legs():
    mgr, client, risk, positions, settings = _make_manager()
    # entry: bought no @ $0.40 (ask=40c), 1 contract
    from backtest.track_b import kalshi_fee, TAKER_FEE_COEF
    entry_price = 0.40
    exit_yes_ask = 0.30  # market moved favorably for the no-holder
    client.get_market.return_value = {"status": "open", "yes_ask": 30}
    client.create_order.return_value = {"order_id": "PAPER-2", "paper": True}

    await mgr._exit_position("TICKER1", entry_price=entry_price, size=1)

    exit_price = 1 - exit_yes_ask
    entry_fee = kalshi_fee(1, entry_price, TAKER_FEE_COEF)
    exit_fee = kalshi_fee(1, exit_price, TAKER_FEE_COEF)
    expected_pnl = (exit_price - entry_price) - entry_fee - exit_fee

    positions.record_realized_pnl.assert_called_once()
    args, kwargs = positions.record_realized_pnl.call_args
    called_ticker, called_pnl = args[0], args[1]
    called_fee = kwargs.get("fee", args[2] if len(args) > 2 else None)
    assert called_ticker == "TICKER1"
    assert called_pnl == pytest.approx(expected_pnl)
    assert called_fee == pytest.approx(entry_fee + exit_fee)
    client.create_order.assert_awaited_once_with(
        ticker="TICKER1", side="yes", price=30, count=1)


@pytest.mark.asyncio
async def test_exit_against_settled_market_takes_settlement_credit_not_close_order():
    mgr, client, risk, positions, settings = _make_manager()
    from backtest.track_b import kalshi_fee, TAKER_FEE_COEF
    entry_price = 0.40
    client.get_market.return_value = {"status": "settled", "result": "no"}  # we held no, it won

    await mgr._exit_position("TICKER1", entry_price=entry_price, size=1)

    client.create_order.assert_not_called()  # never attempt to close a settled market
    entry_fee = kalshi_fee(1, entry_price, TAKER_FEE_COEF)
    expected_pnl = (1.0 - entry_price) - entry_fee  # full payout, only entry fee was ever paid
    positions.record_realized_pnl.assert_called_once()
    args, kwargs = positions.record_realized_pnl.call_args
    assert args[1] == pytest.approx(expected_pnl)


@pytest.mark.asyncio
async def test_exit_against_settled_market_losing_side_pays_full_entry():
    mgr, client, risk, positions, settings = _make_manager()
    from backtest.track_b import kalshi_fee, TAKER_FEE_COEF
    entry_price = 0.40
    client.get_market.return_value = {"status": "settled", "result": "yes"}  # we held no, it lost

    await mgr._exit_position("TICKER1", entry_price=entry_price, size=1)

    entry_fee = kalshi_fee(1, entry_price, TAKER_FEE_COEF)
    expected_pnl = (0.0 - entry_price) - entry_fee
    args, kwargs = positions.record_realized_pnl.call_args
    assert args[1] == pytest.approx(expected_pnl)
