"""Live trigger for the validated tennis order-flow momentum signal
(plans/tennis-mm-next-steps.md). A trading-only twin of
`live/tennis_recorder.py`'s WS loop — same `Book` replay, same connection
handling (imported, not reimplemented), but no parquet buffering/flush.
Instead of writing rows to disk, it runs the exact sign-detection
comparison already validated offline in `scripts/tennis_momentum_signal.py`
(a trade's price matched against the current best yes/no bid within
tolerance) and calls `TennisOrderManager.on_signal` directly, in-process,
so reaction time is sub-second — latency was validated safe up to 5s
(`plans/tennis-mm-next-steps.md`), an in-process call is far under that.

Deliberately a **separate process from `tennis_recorder.py`**, with its own
WebSocket connection: a bug in this (new, less-proven) trading code must
never be able to destabilize the capture daemon the size/depth-cost
re-test still depends on.

Exits immediately if `settings.TENNIS_ENABLED` is false, so it's safe to
have installed/cronned without being live.

    PYTHONPATH=. python live/tennis_signal_bot.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal

import httpx
import pandas as pd
import websockets

from config.settings import get_settings
from db.session import get_session_factory, init_db
from live.tennis_recorder import (
    Book, WS_URL, RECONNECT_BACKOFF_S, MARKET_REFRESH_S,
    SUBSCRIBE_CHUNK, PERIODIC_RESYNC_S,
    fetch_open_tickers, ws_headers, _interruptible_sleep,
)
from risk.risk_controls import RiskControls
from trading.kalshi_client import KalshiClient
from trading.position_tracker import PositionTracker
from trading.tennis_order_manager import TennisOrderManager

logger = logging.getLogger(__name__)

PIDFILE = "data/capture/tennis_signal_bot.pid"


class SignalBot:
    def __init__(self, order_manager: TennisOrderManager) -> None:
        self._order_manager = order_manager
        self.books: dict[str, Book] = {}
        self.series_of: dict[str, str] = {}
        self.prev_volume: dict[str, float] = {}
        self.orderbook_sids: list[int] = []
        self.ticker_sids: list[int] = []
        self.sid_ticker_count: dict[int, int] = {}
        self.next_cmd_id = 1
        self.ws: websockets.ClientConnection | None = None
        self.needs_resync = False
        self._pending_sid_target: dict[int, tuple] = {}

    async def send(self, cmd: dict) -> int:
        cmd_id = self.next_cmd_id
        self.next_cmd_id += 1
        cmd["id"] = cmd_id
        await self.ws.send(json.dumps(cmd))
        return cmd_id

    async def subscribe_new(self, tickers: list[str]) -> None:
        for i in range(0, len(tickers), SUBSCRIBE_CHUNK):
            chunk = tickers[i:i + SUBSCRIBE_CHUNK]
            await self._add_to_channel(chunk, "orderbook_delta", self.orderbook_sids)
            await self._add_to_channel(chunk, "ticker", self.ticker_sids)

    async def _add_to_channel(self, chunk: list[str], channel: str, sids: list[int]) -> None:
        target = next((s for s in sids if self.sid_ticker_count.get(s, 0) < SUBSCRIBE_CHUNK), None)
        if target is not None:
            await self.send({"cmd": "update_subscription",
                             "params": {"sids": [target], "market_tickers": chunk, "action": "add_markets"}})
            self.sid_ticker_count[target] = self.sid_ticker_count.get(target, 0) + len(chunk)
        else:
            cmd_id = await self.send({"cmd": "subscribe",
                                      "params": {"channels": [channel], "market_tickers": chunk}})
            self._pending_sid_target[cmd_id] = (channel, sids, len(chunk))

    def handle_message(self, raw: str) -> None:
        m = json.loads(raw)
        t = m.get("type")
        if t == "subscribed":
            channel, sid = m["msg"]["channel"], m["msg"]["sid"]
            pending = self._pending_sid_target.pop(m.get("id"), None)
            if pending:
                _, sids, count = pending
                sids.append(sid)
                self.sid_ticker_count[sid] = count
            elif channel == "orderbook_delta":
                self.orderbook_sids.append(sid)
            else:
                self.ticker_sids.append(sid)
            return
        if t == "error":
            logger.warning("ws error: %s", m.get("msg"))
            return
        msg = m.get("msg", {})
        ticker = msg.get("market_ticker")
        if ticker is None:
            return
        if t == "orderbook_snapshot":
            book = self.books.setdefault(ticker, Book())
            book.apply_snapshot(msg)
        elif t == "orderbook_delta":
            book = self.books.setdefault(ticker, Book())
            book.apply_delta(msg["side"], msg["price_dollars"], float(msg["delta_fp"]))
        elif t == "ticker":
            self._on_ticker(ticker, msg)

    def _on_ticker(self, ticker: str, msg: dict) -> None:
        """Port of the sign-detection logic validated offline in
        scripts/tennis_momentum_signal.py: a new trade print (volume_fp
        increased) whose price matches the current best yes-bid is bearish
        ("no" side, validated); matching the current best yes-ask (=1-best
        no-bid) is bullish ("yes" side, logged but not traded)."""
        volume = msg.get("volume_fp")
        price = msg.get("price_dollars")
        if volume is None or price is None:
            return
        prev = self.prev_volume.get(ticker)
        self.prev_volume[ticker] = float(volume)
        if prev is None:
            return
        trade_size = float(volume) - prev
        if trade_size <= 0:
            return
        book = self.books.get(ticker)
        if book is None:
            return
        tp = round(float(price), 2)
        bp_yes = book.best_yes_bid()
        bp_no_price = max((float(p) for p in book.no), default=None)
        sign = 0
        if bp_yes is not None and abs(tp - bp_yes) < 1e-6:
            sign = -1
        elif bp_no_price is not None and abs(tp - round(1 - bp_no_price, 2)) < 1e-6:
            sign = 1
        if sign == 0:
            return
        side = "no" if sign == -1 else "yes"
        asyncio.create_task(self._order_manager.on_signal(ticker, side, pd.Timestamp.utcnow()))


async def ws_loop(bot: SignalBot, rest: httpx.Client, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            async with websockets.connect(WS_URL, additional_headers=ws_headers(), open_timeout=10) as ws:
                bot.ws = ws
                bot.orderbook_sids, bot.ticker_sids = [], []
                bot.sid_ticker_count = {}
                bot.needs_resync = False
                tickers = list(bot.series_of) or list(await asyncio.to_thread(fetch_open_tickers, rest))
                bot.series_of.update({t: bot.series_of.get(t, t.split("-")[0]) for t in tickers})
                logger.info("signal bot ws connected; subscribing %d tennis markets", len(tickers))
                await bot.subscribe_new(tickers)
                async for raw in ws:
                    if stop_event.is_set() or bot.needs_resync:
                        break
                    bot.handle_message(raw)
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.warning("signal bot ws disconnected (%s); reconnecting in %ds", e, RECONNECT_BACKOFF_S)
        bot.ws = None
        if not stop_event.is_set():
            await _interruptible_sleep(RECONNECT_BACKOFF_S, stop_event)


async def market_discovery_loop(bot: SignalBot, rest: httpx.Client, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        open_tickers = await asyncio.to_thread(fetch_open_tickers, rest)
        known = set(bot.series_of)
        new = [t for t in open_tickers if t not in known]
        closed = [t for t in known if t not in open_tickers]
        bot.series_of.update(open_tickers)
        if new and bot.ws is not None:
            await bot.subscribe_new(new)
        for t in closed:
            bot.books.pop(t, None)
            bot.series_of.pop(t, None)
            bot.prev_volume.pop(t, None)
        await _interruptible_sleep(MARKET_REFRESH_S, stop_event)


async def periodic_resync_loop(bot: SignalBot, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await _interruptible_sleep(PERIODIC_RESYNC_S, stop_event)
        if not stop_event.is_set() and bot.ws is not None:
            bot.needs_resync = True


async def main_async() -> None:
    settings = get_settings()
    if not settings.TENNIS_ENABLED:
        logger.info("TENNIS_ENABLED is false — signal bot not starting")
        return

    init_db(settings.DB_URL)
    kalshi_client = KalshiClient(
        api_key=settings.KALSHI_API_KEY, private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
        base_url=settings.KALSHI_BASE_URL, paper_trading=settings.PAPER_TRADING,
    )
    risk_controls = RiskControls(settings=settings, db_session_factory=get_session_factory())
    position_tracker = PositionTracker(db_session_factory=get_session_factory())
    order_manager = TennisOrderManager(
        kalshi_client=kalshi_client, risk_controls=risk_controls,
        position_tracker=position_tracker, settings=settings,
    )
    bot = SignalBot(order_manager)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    with httpx.Client() as rest:
        await asyncio.gather(
            ws_loop(bot, rest, stop_event),
            market_discovery_loop(bot, rest, stop_event),
            periodic_resync_loop(bot, stop_event),
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    if not settings.TENNIS_ENABLED:
        logger.info("TENNIS_ENABLED is false — exiting")
        return
    os.makedirs(os.path.dirname(PIDFILE), exist_ok=True)
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))
    try:
        asyncio.run(main_async())
    finally:
        if os.path.exists(PIDFILE):
            os.remove(PIDFILE)
    logger.info("tennis signal bot stopped")


if __name__ == "__main__":
    main()
