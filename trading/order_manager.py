import logging
from datetime import datetime
import asyncio

from trading.kelly import compute_size
from db.models import Order, Position
from db.session import get_session

logger = logging.getLogger(__name__)

STALE_PRICE_DRIFT_CENTS = 1
CONTRACT_PRICE_SCALE = 100  # Kalshi prices are in cents (0–100)


class OrderManager:
    def __init__(self, kalshi_client, shared_state, risk_controls, settings) -> None:
        self._client = kalshi_client
        self._state = shared_state
        self._risk = risk_controls
        self._settings = settings
        self._resting_orders: dict[str, dict] = {}

    async def process_tick(self) -> None:
        tickers = self._state.get_all_tickers()
        tasks = [self._process_ticker(ticker) for ticker in tickers]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _process_ticker(self, ticker: str) -> None:
        allowed, reason = self._risk.can_trade(ticker)
        if not allowed:
            logger.debug("Skipping %s: %s", ticker, reason)
            return

        state_snap = self._state.snapshot().get(ticker)
        if state_snap is None:
            return

        blended = state_snap.get("blended_fair")
        if blended is None:
            return

        market_mid = state_snap["market_mid"]
        ci_width = state_snap["ci_width"]
        horizon_days = state_snap["horizon_days"]
        strategy_lock = state_snap["strategy_lock"]

        edge = abs(blended - market_mid)
        fee_rate = 0.0005
        min_edge = self._settings.MIN_EDGE_CENTS / 100.0 + fee_rate

        if edge < min_edge:
            return

        if ci_width > self._settings.MAX_CI_WIDTH:
            return

        contracts = compute_size(
            fair_value=blended,
            market_price=market_mid,
            ci_width=ci_width,
            horizon_days=horizon_days,
            kelly_fraction=self._settings.KELLY_FRACTION,
            max_exposure_usd=self._settings.MAX_EXPOSURE_PER_TICKER_USD,
            horizon_multipliers=self._settings.HORIZON_MULTIPLIERS,
            strategy_lock=strategy_lock,
        )

        if contracts <= 0:
            return

        side = "yes" if blended > market_mid else "no"
        limit_price_cents = int(blended * CONTRACT_PRICE_SCALE)
        limit_price_cents = max(1, min(99, limit_price_cents))

        resting = self._resting_orders.get(ticker)
        if resting:
            resting_price = resting.get("price", 0)
            if abs(resting_price - limit_price_cents) > STALE_PRICE_DRIFT_CENTS:
                await self._cancel_resting(ticker)
            else:
                return

        try:
            order = await self._client.create_order(
                ticker=ticker,
                side=side,
                price=limit_price_cents,
                count=contracts,
                order_type="limit",
            )
            kalshi_order_id = order.get("order_id") or order.get("id", "unknown")
            self._resting_orders[ticker] = {
                "order_id": kalshi_order_id,
                "price": limit_price_cents,
                "side": side,
                "count": contracts,
            }
            with get_session() as db:
                db.add(Order(
                    ticker=ticker,
                    kalshi_order_id=kalshi_order_id,
                    side=side,
                    price=limit_price_cents,
                    size=contracts,
                    status="pending",
                    strategy="A" if state_snap.get("fair_b") is None else "B",
                    submitted_at=datetime.utcnow(),
                ))
            logger.info("Limit order posted: %s %s @ %d¢ × %d", ticker, side, limit_price_cents, contracts)
        except Exception as exc:
            logger.error("Order submission failed for %s: %s", ticker, exc)
            self._risk.record_rejected_order(ticker)

    async def _cancel_resting(self, ticker: str) -> None:
        resting = self._resting_orders.get(ticker)
        if not resting:
            return
        order_id = resting["order_id"]
        try:
            await self._client.cancel_order(order_id)
            del self._resting_orders[ticker]
            with get_session() as db:
                order = db.query(Order).filter(Order.kalshi_order_id == order_id).first()
                if order:
                    order.status = "cancelled"
            logger.info("Cancelled resting order %s for %s", order_id, ticker)
        except Exception as exc:
            logger.warning("Cancel failed for %s: %s", order_id, exc)

    async def cancel_all(self) -> None:
        tickers = list(self._resting_orders.keys())
        for ticker in tickers:
            await self._cancel_resting(ticker)
        logger.info("All resting orders cancelled")

    async def sync_fills(self) -> None:
        try:
            fills = await self._client.get_fills()
            for fill in fills:
                ticker = fill.get("ticker")
                kalshi_order_id = fill.get("order_id")
                count = int(fill.get("count", 0))
                side = fill.get("side", "yes")
                fill_price = float(fill.get("yes_price", fill.get("price", 0))) / 100.0
                contracts_delta = count if side == "yes" else -count

                self._state.update_fill(ticker, contracts_delta, fill_price)

                with get_session() as db:
                    order = db.query(Order).filter(Order.kalshi_order_id == kalshi_order_id).first()
                    if order and order.status != "filled":
                        order.status = "filled"
                        order.filled_at = datetime.utcnow()
                        order.fill_price = fill_price

                    pos = db.query(Position).filter(Position.ticker == ticker).first()
                    if pos is None:
                        pos = Position(ticker=ticker, net_contracts=0, avg_entry_price=fill_price)
                        db.add(pos)
                    old_contracts = pos.net_contracts
                    pos.net_contracts += contracts_delta
                    if pos.net_contracts != 0:
                        pos.avg_entry_price = (
                            (abs(old_contracts) * pos.avg_entry_price + abs(contracts_delta) * fill_price)
                            / abs(pos.net_contracts)
                        )
                    pos.last_updated = datetime.utcnow()

                if ticker in self._resting_orders:
                    del self._resting_orders[ticker]

        except Exception as exc:
            logger.error("sync_fills error: %s", exc)
