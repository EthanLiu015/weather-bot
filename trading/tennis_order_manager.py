"""Order execution for the validated tennis order-flow momentum signal
(plans/tennis-mm-next-steps.md). Deliberately narrow, matching only what's
actually validated:

  - "no" side only — a trade printing at the bid (bearish pressure), buying
    "no" right after. The "yes" side was found weak/inconsistent per-market
    and is intentionally not traded; signals for it are logged, not acted
    on, so the decision is visible rather than silently swallowed.
  - fixed contract size (`settings.TENNIS_CONTRACT_SIZE`) — order size
    beyond 1 contract is still unvalidated (see plans doc, "Size/depth
    cost" section), so this does not use Kelly sizing.
  - fixed hold time (`settings.TENNIS_HOLD_SECONDS`) then close by buying
    the opposite side at the current ask — the same mechanic
    `scripts/tennis_taker_pnl.py` already prices exits with (closing a
    Kalshi position means buying the complementary side; `create_order`
    always sends action="buy"), so live P&L is directly comparable to the
    backtest's numbers, not a different methodology.

Unlike the weather `OrderManager` (deleted, see plan doc) there's no
fair-value blend, so no Kelly sizing and no SharedState — this reacts to a
single point-in-time signal (from `live/tennis_signal_bot.py`) and runs a
single entry-then-exit lifecycle per position, never layering.
"""
from __future__ import annotations

import logging
from datetime import datetime

from backtest.track_b import TAKER_FEE_COEF, kalshi_fee
from db.models import Order, Position
from db.session import get_session

logger = logging.getLogger(__name__)


class TennisOrderManager:
    def __init__(self, kalshi_client, risk_controls, position_tracker, settings) -> None:
        self._client = kalshi_client
        self._risk = risk_controls
        self._positions = position_tracker
        self._settings = settings

    async def on_signal(self, ticker: str, side: str, ts: datetime) -> None:
        if side != "no":
            logger.info("skipping %s-side signal for %s (unvalidated, not traded)", side, ticker)
            return

        allowed, reason = self._risk.can_trade(ticker)
        if not allowed:
            logger.debug("skipping %s: %s", ticker, reason)
            return

        open_positions = [p for p in self._positions.get_all_positions() if p["net_contracts"] != 0]
        if any(p["ticker"] == ticker for p in open_positions):
            logger.debug("skipping %s: already holding a position", ticker)
            return
        if len(open_positions) >= self._settings.TENNIS_MAX_CONCURRENT_POSITIONS:
            logger.debug("skipping %s: at max concurrent positions (%d)",
                         ticker, self._settings.TENNIS_MAX_CONCURRENT_POSITIONS)
            return

        market = await self._client.get_market(ticker)
        if market.get("status") != "open":
            logger.debug("skipping %s: not open (%s)", ticker, market.get("status"))
            return
        no_ask_cents = market["no_ask"]
        size = self._settings.TENNIS_CONTRACT_SIZE

        order = await self._client.create_order(ticker=ticker, side="no", price=no_ask_cents, count=size)
        entry_price = no_ask_cents / 100.0
        self._record_entry(ticker, order, entry_price, size)

        import asyncio
        asyncio.create_task(self._delayed_exit(ticker, entry_price, size))

    async def _delayed_exit(self, ticker: str, entry_price: float, size: int) -> None:
        import asyncio
        await asyncio.sleep(self._settings.TENNIS_HOLD_SECONDS)
        await self._exit_position(ticker, entry_price, size)

    async def _exit_position(self, ticker: str, entry_price: float, size: int) -> None:
        entry_fee = kalshi_fee(size, entry_price, TAKER_FEE_COEF)
        market = await self._client.get_market(ticker)

        if market.get("status") != "open":
            # Settled mid-hold (walkover/retirement, or just a fast match) —
            # a close order against a closed market would be rejected. Take
            # the settlement payout instead: full $1/contract if "no" won,
            # $0 if "yes" won. Only the entry fee was ever actually paid.
            payout = 1.0 if market.get("result") == "no" else 0.0
            pnl = (payout - entry_price) - entry_fee
            self._positions.record_realized_pnl(ticker, pnl, entry_fee)
            self._flatten_position(ticker)
            logger.info("[SETTLED] %s resolved %s mid-hold; pnl=%.4f", ticker, market.get("result"), pnl)
            return

        yes_ask_cents = market["yes_ask"]
        await self._client.create_order(ticker=ticker, side="yes", price=yes_ask_cents, count=size)
        exit_price = 1 - (yes_ask_cents / 100.0)
        exit_fee = kalshi_fee(size, exit_price, TAKER_FEE_COEF)
        pnl = (exit_price - entry_price) - entry_fee - exit_fee
        self._positions.record_realized_pnl(ticker, pnl, entry_fee + exit_fee)
        self._flatten_position(ticker)
        logger.info("[EXIT] %s no@%.2f -> yes_ask@%.2f pnl=%.4f", ticker, entry_price, yes_ask_cents / 100.0, pnl)

    def _record_entry(self, ticker: str, order: dict, entry_price: float, size: int) -> None:
        order_id = order.get("order_id") or order.get("id", "unknown")
        with get_session() as db:
            db.add(Order(
                ticker=ticker, kalshi_order_id=order_id, side="no",
                price=int(round(entry_price * 100)), size=size,
                status="filled", strategy="tn",
                submitted_at=datetime.utcnow(), filled_at=datetime.utcnow(),
                fill_price=entry_price,
            ))
            pos = db.query(Position).filter(Position.ticker == ticker).first()
            if pos is None:
                pos = Position(ticker=ticker, net_contracts=0, avg_entry_price=0.0)
                db.add(pos)
            pos.net_contracts = -size
            pos.avg_entry_price = entry_price
            pos.last_updated = datetime.utcnow()

    def _flatten_position(self, ticker: str) -> None:
        with get_session() as db:
            pos = db.query(Position).filter(Position.ticker == ticker).first()
            if pos is not None:
                pos.net_contracts = 0
                pos.last_updated = datetime.utcnow()
