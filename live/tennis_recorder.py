"""Live tennis tick recorder v2 — WebSocket push, full order-book depth.

Replaces the v1 1s-REST-poll top-of-book recorder. Two gaps that blocked a
market-making feasibility read: (1) latency floor was the poll period plus a
CloudFront-edge-cache-bypass hack, not real push; (2) only top-of-book was
ever stored, so there was no way to tell whether size existed to fill at.

This version holds one Kalshi WebSocket connection (`orderbook_delta` +
`ticker` channels) across all open tennis match markets. `orderbook_delta`
gives the full resting-order ladder (snapshot once per market on subscribe,
then signed size deltas per price level — reconstruct the ladder by replay).
`ticker` gives push updates of best bid/ask, last price, volume, OI — same
fields the old top-of-book schema had, now event-driven instead of polled.

A market's book only ever lists resting BUY orders for each side; the
opposite ask is derived (yes_ask = 1 - best_no_bid). Kalshi computes
yes_bid_dollars/yes_ask_dollars the same way, so `best_yes_bid`/`best_yes_ask`
here match the old REST-derived fields.

REST is still used, on a timer, purely for market *discovery* (which tickers
exist to subscribe to) and event metadata (who's playing) — WS has no
"new tennis market opened" push we subscribe to here.

Output (unchanged path/partitioning, `scripts/tennis_compact.py` still
applies unmodified):
    data/capture/tennis_ticks/date=YYYY-MM-DD/part-HHMMSS.parquet
One row per snapshot/delta/ticker event, `type` column distinguishes them:
  - "snapshot": yes_book_json/no_book_json hold the full ladder as
    [[price_dollars, size], ...]; emitted once per market (subscribe, or
    resync after a detected sequence gap).
  - "delta": side/price/delta_fp hold the single price-level change.
  - "ticker": yes_bid/yes_ask/last_price/volume_fp/open_interest_fp, the
    old top-of-book columns, now push-driven.
All rows carry best_yes_bid/best_yes_ask recomputed from the live in-memory
book, so "what was the top of book at time T" never needs a ladder replay.

Safety: refuses to record when free disk < MIN_FREE_GB. Stop with SIGTERM or
by deleting the pidfile. The cron watchdog restarts it if it dies. A detected
orderbook sequence gap (dropped WS message) forces a full reconnect + resync
rather than silently drifting out of sync with the real book.

    PYTHONPATH=. python live/tennis_recorder.py
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import shutil
import signal
import time
from urllib.parse import urlparse

import httpx
import pandas as pd
import websockets
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from config.settings import get_settings

logger = logging.getLogger(__name__)

REST_BASE = "https://api.elections.kalshi.com/trade-api/v2"
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"
WS_SIGN_PATH = urlparse(WS_URL).path
SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH",
          "KXWTACHALLENGERMATCH", "KXCHALLENGERMATCH"]
TICKS_DIR = "data/capture/tennis_ticks"
EVENTS_OUT = "data/capture/tennis_events.parquet"
PIDFILE = "data/capture/tennis_recorder.pid"
FLUSH_S = 60
MARKET_REFRESH_S = 300
EVENTS_REFRESH_S = 3600
MIN_FREE_GB = 2.0
RECONNECT_BACKOFF_S = 5
SUBSCRIBE_CHUNK = 200
ROW_COLUMNS = ["ts", "series", "ticker", "type", "yes_bid", "yes_ask",
               "last_price", "volume_fp", "open_interest_fp", "side",
               "price", "delta_fp", "yes_book_json", "no_book_json",
               "seq", "sid"]
PERIODIC_RESYNC_S = 900

def disk_ok() -> bool:
    return shutil.disk_usage(".").free / 1e9 >= MIN_FREE_GB


def ws_headers() -> dict[str, str]:
    """Kalshi signs timestamp + method + full-path-from-API-root with
    RSA-PSS/SHA256 (see trading/kalshi_client.py for the REST-side fix this
    mirrors — PKCS1v15 or a path missing the /trade-api/v2 prefix both 401)."""
    s = get_settings()
    pem = open(s.KALSHI_PRIVATE_KEY_PATH, "rb").read()
    private_key = serialization.load_pem_private_key(pem, password=None)
    ts_ms = str(int(time.time() * 1000))
    msg = ts_ms + "GET" + WS_SIGN_PATH
    signature = private_key.sign(
        msg.encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": s.KALSHI_API_KEY,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


class Book:
    """In-memory resting-order ladder for one market. yes/no are dicts of
    price_dollars(str) -> size(float); each is the resting-buy side only."""

    __slots__ = ("yes", "no")

    def __init__(self) -> None:
        self.yes: dict[str, float] = {}
        self.no: dict[str, float] = {}

    def apply_snapshot(self, msg: dict) -> None:
        self.yes = {p: float(s) for p, s in msg.get("yes_dollars_fp", [])}
        self.no = {p: float(s) for p, s in msg.get("no_dollars_fp", [])}

    def apply_delta(self, side: str, price: str, delta: float) -> None:
        book = self.yes if side == "yes" else self.no
        new_size = book.get(price, 0.0) + delta
        if new_size <= 0:
            book.pop(price, None)
        else:
            book[price] = new_size

    def best_yes_bid(self) -> float | None:
        return max((float(p) for p in self.yes), default=None)

    def best_yes_ask(self) -> float | None:
        best_no = max((float(p) for p in self.no), default=None)
        return None if best_no is None else round(1 - best_no, 2)

    def ladder_json(self) -> tuple[str, str]:
        yes_sorted = sorted(self.yes.items(), key=lambda kv: -float(kv[0]))
        no_sorted = sorted(self.no.items(), key=lambda kv: -float(kv[0]))
        return json.dumps(yes_sorted), json.dumps(no_sorted)


class Recorder:
    def __init__(self) -> None:
        self.books: dict[str, Book] = {}
        self.series_of: dict[str, str] = {}
        self.last_ticker_row: dict[str, dict] = {}
        self.last_seq: dict[int, int] = {}
        self.orderbook_sids: list[int] = []
        self.ticker_sids: list[int] = []
        self.sid_ticker_count: dict[int, int] = {}
        self.buffer: list[dict] = []
        self.next_cmd_id = 1
        self.ws: websockets.ClientConnection | None = None
        self.needs_resync = False
        self._pending_sid_target: dict[int, tuple] = {}

    def base_row(self, series: str, ticker: str) -> dict:
        return {c: None for c in ROW_COLUMNS} | {
            "ts": pd.Timestamp.utcnow().isoformat(), "series": series, "ticker": ticker,
        }

    def emit_snapshot(self, series: str, ticker: str, book: Book, seq: int | None, sid: int | None) -> None:
        yes_json, no_json = book.ladder_json()
        row = self.base_row(series, ticker)
        row.update(type="snapshot", yes_bid=book.best_yes_bid(), yes_ask=book.best_yes_ask(),
                    yes_book_json=yes_json, no_book_json=no_json, seq=seq, sid=sid)
        self.buffer.append(row)

    def emit_delta(self, series: str, ticker: str, book: Book, side: str, price: str, delta: float,
                    seq: int | None, sid: int | None) -> None:
        row = self.base_row(series, ticker)
        row.update(type="delta", yes_bid=book.best_yes_bid(), yes_ask=book.best_yes_ask(),
                    side=side, price=float(price), delta_fp=delta, seq=seq, sid=sid)
        self.buffer.append(row)

    def emit_ticker(self, series: str, ticker: str, msg: dict, seq: int | None, sid: int | None) -> None:
        row = self.base_row(series, ticker)
        row.update(type="ticker",
                    yes_bid=_f(msg.get("yes_bid_dollars")), yes_ask=_f(msg.get("yes_ask_dollars")),
                    last_price=_f(msg.get("price_dollars")),
                    volume_fp=_f(msg.get("volume_fp")), open_interest_fp=_f(msg.get("open_interest_fp")),
                    seq=seq, sid=sid)
        self.buffer.append(row)

    def flush(self) -> None:
        if not self.buffer:
            return
        now = pd.Timestamp.utcnow()
        day_dir = f"{TICKS_DIR}/date={now.strftime('%Y-%m-%d')}"
        os.makedirs(day_dir, exist_ok=True)
        path = f"{day_dir}/part-{now.strftime('%H%M%S')}.parquet"
        pd.DataFrame(self.buffer, columns=ROW_COLUMNS).to_parquet(path)
        logger.info("flushed %d rows -> %s", len(self.buffer), path)
        self.buffer = []

    async def send(self, cmd: dict) -> int:
        """market_discovery_loop and ws_loop run concurrently; the ws can go
        None or closed between market_discovery_loop's `rec.ws is not None`
        check and this call landing (no await in between at the call site,
        but subscribe_new/unsubscribe loop multiple sends per call, and
        ws_loop can reconnect mid-loop). An uncaught exception here used to
        propagate through asyncio.gather and kill the whole process (recorder
        died silently for 8h on 2026-07-28 before the cron watchdog fix
        landed). Swallow and force a resync instead — ws_loop already
        resubscribes everything from rec.series_of on reconnect."""
        cmd_id = self.next_cmd_id
        self.next_cmd_id += 1
        cmd["id"] = cmd_id
        if self.ws is None:
            self.needs_resync = True
            return cmd_id
        try:
            await self.ws.send(json.dumps(cmd))
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.warning("send failed (%s); forcing resync", e)
            self.needs_resync = True
        return cmd_id

    async def subscribe_new(self, tickers: list[str]) -> None:
        """Add tickers to the running WS subscription, chunked and attached
        to the smallest existing sid so growth doesn't build one unbounded
        subscription (Kalshi enforces an undocumented per-subscription cap)."""
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

    async def unsubscribe(self, tickers: list[str]) -> None:
        if not tickers:
            return
        for sid in self.orderbook_sids + self.ticker_sids:
            await self.send({"cmd": "update_subscription",
                             "params": {"sids": [sid], "market_tickers": tickers, "action": "delete_markets"}})
        for t in tickers:
            self.books.pop(t, None)
            self.series_of.pop(t, None)

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
        series = self.series_of.get(ticker, ticker.split("-")[0])
        seq = m.get("seq")
        sid = m.get("sid")
        if seq is not None and sid is not None:
            prev = self.last_seq.get(sid)
            if prev is not None and seq != prev + 1 and t != "orderbook_snapshot":
                logger.warning("seq gap on sid %d (%s -> %s); forcing resync", sid, prev, seq)
                self.needs_resync = True
            self.last_seq[sid] = seq
        if t == "orderbook_snapshot":
            book = self.books.setdefault(ticker, Book())
            book.apply_snapshot(msg)
            self.emit_snapshot(series, ticker, book, seq, sid)
        elif t == "orderbook_delta":
            book = self.books.setdefault(ticker, Book())
            book.apply_delta(msg["side"], msg["price_dollars"], float(msg["delta_fp"]))
            self.emit_delta(series, ticker, book, msg["side"], msg["price_dollars"], float(msg["delta_fp"]), seq, sid)
        elif t == "ticker":
            self.emit_ticker(series, ticker, msg, seq, sid)


def _f(v) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def fetch_open_tickers(client: httpx.Client) -> dict[str, str]:
    """ticker -> series for every currently open tennis market."""
    out: dict[str, str] = {}
    for series in SERIES:
        try:
            r = client.get(f"{REST_BASE}/markets",
                           params={"series_ticker": series, "status": "open", "limit": 200}, timeout=10.0)
            for m in r.json().get("markets", []):
                out[m["ticker"]] = series
        except httpx.HTTPError as e:
            logger.warning("market discovery failed for %s: %s", series, e)
    return out


def refresh_events(client: httpx.Client) -> None:
    rows = []
    for series in SERIES:
        try:
            r = client.get(f"{REST_BASE}/markets",
                           params={"series_ticker": series, "status": "open", "limit": 200}, timeout=10.0)
            for m in r.json().get("markets", []):
                rows.append({"series": series, "ticker": m["ticker"],
                             "event_ticker": m.get("event_ticker"),
                             "title": m.get("title") or "", "yes_sub_title": m.get("yes_sub_title") or "",
                             "open_time": m.get("open_time"), "close_time": m.get("close_time"),
                             "seen_at": pd.Timestamp.utcnow().isoformat()})
        except httpx.HTTPError as e:
            logger.warning("events refresh failed for %s: %s", series, e)
    if not rows:
        return
    new = pd.DataFrame(rows)
    try:
        merged = pd.concat([pd.read_parquet(EVENTS_OUT), new]).drop_duplicates("ticker", keep="last")
    except FileNotFoundError:
        merged = new
    merged.to_parquet(EVENTS_OUT)
    logger.info("events metadata: %d markets known", len(merged))


async def _interruptible_sleep(seconds: float, stop_event: asyncio.Event) -> None:
    """Plain asyncio.sleep(N) only lets a loop notice shutdown every N
    seconds — with N=300 that's a 5-minute-late SIGTERM response. Race the
    sleep against the stop event instead so shutdown is immediate."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


async def market_discovery_loop(rec: Recorder, rest: httpx.Client, stop_event: asyncio.Event) -> None:
    last_events = 0.0
    while not stop_event.is_set():
        open_tickers = await asyncio.to_thread(fetch_open_tickers, rest)
        known = set(rec.series_of)
        new = [t for t in open_tickers if t not in known]
        closed = [t for t in known if t not in open_tickers]
        rec.series_of.update(open_tickers)
        if new and rec.ws is not None:
            logger.info("discovered %d new tennis markets", len(new))
            await rec.subscribe_new(new)
        if closed and rec.ws is not None:
            logger.info("%d tennis markets closed, unsubscribing", len(closed))
            await rec.unsubscribe(closed)
        if time.monotonic() - last_events >= EVENTS_REFRESH_S:
            await asyncio.to_thread(refresh_events, rest)
            last_events = time.monotonic()
        await _interruptible_sleep(MARKET_REFRESH_S, stop_event)


async def periodic_resync_loop(rec: Recorder, stop_event: asyncio.Event) -> None:
    """Force a full reconnect+resubscribe on a timer, independent of seq-gap
    detection. Offline replay of snapshot+delta rows has no way to detect a
    dropped/misordered WS message after the fact (seq/sid weren't persisted
    until this row was added, and even with them, silent gaps the live
    process's own gap check misses are possible) — the only mitigation is
    bounding how long any single resync interval's drift can compound before
    a fresh orderbook_snapshot resets it. See plans/tennis-mm-next-steps.md
    (\"Size/depth cost\" section) for the drift this was found to cause."""
    while not stop_event.is_set():
        await _interruptible_sleep(PERIODIC_RESYNC_S, stop_event)
        if not stop_event.is_set() and rec.ws is not None:
            logger.info("periodic resync (%ds elapsed)", PERIODIC_RESYNC_S)
            rec.needs_resync = True


async def flush_loop(rec: Recorder, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        await _interruptible_sleep(FLUSH_S, stop_event)
        if not disk_ok():
            logger.error("free disk < %.1f GB — dropping buffer", MIN_FREE_GB)
            rec.buffer = []
            continue
        rec.flush()


async def ws_loop(rec: Recorder, rest: httpx.Client, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            async with websockets.connect(WS_URL, additional_headers=ws_headers(), open_timeout=10) as ws:
                rec.ws = ws
                rec.orderbook_sids, rec.ticker_sids = [], []
                rec.sid_ticker_count = {}
                rec.last_seq = {}
                rec.needs_resync = False
                tickers = list(rec.series_of) or list(await asyncio.to_thread(fetch_open_tickers, rest))
                rec.series_of.update({t: rec.series_of.get(t, t.split("-")[0]) for t in tickers})
                logger.info("ws connected; subscribing %d tennis markets", len(tickers))
                await rec.subscribe_new(tickers)
                async for raw in ws:
                    if stop_event.is_set() or rec.needs_resync:
                        break
                    rec.handle_message(raw)
        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            logger.warning("ws disconnected (%s); reconnecting in %ds", e, RECONNECT_BACKOFF_S)
        rec.ws = None
        if not stop_event.is_set():
            await _interruptible_sleep(RECONNECT_BACKOFF_S, stop_event)


async def main_async() -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)
    rec = Recorder()
    with httpx.Client() as rest:
        await asyncio.gather(
            ws_loop(rec, rest, stop_event),
            market_discovery_loop(rec, rest, stop_event),
            flush_loop(rec, stop_event),
            periodic_resync_loop(rec, stop_event),
        )
    rec.flush()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    os.makedirs(os.path.dirname(PIDFILE), exist_ok=True)
    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))
    try:
        asyncio.run(main_async())
    finally:
        if os.path.exists(PIDFILE):
            os.remove(PIDFILE)
    logger.info("tennis recorder stopped")


if __name__ == "__main__":
    main()
