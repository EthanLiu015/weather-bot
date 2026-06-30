"""Live Kalshi order-book + trade depth logger (websocket).

Connects to the Kalshi websocket, subscribes to `orderbook_delta` (+ `trade`) for
the active temperature markets, maintains a local OrderBook per market, and logs
top-of-book CHANGES and trades to timestamped parquet shards under
data/marketdata/. This is the dataset the trades-only probe could not provide:
real quoted spreads, depth, and fills over time — the inputs to estimating
market-making profitability (realized spread vs adverse selection vs fill rate).

Run:
  PYTHONPATH=. python -m bot.marketdata.depth_logger --smoke       # verify feed/schema
  PYTHONPATH=. python -m bot.marketdata.depth_logger --hours 12    # collect

The wss endpoint / sign-path are the one thing the docs are vague on; --smoke
prints raw frames so they can be confirmed/adjusted against the live feed.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import time
from pathlib import Path

import pandas as pd
import websockets
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding

from bot.config.series import is_low_temp_series
from bot.config.settings import get_settings
from bot.marketdata.orderbook import OrderBook
from bot.research.fetch_kalshi_history import (
    LIVE_API_KEY,
    LIVE_BASE_URL,
    ReadOnlyKalshiClient,
    SERIES_STATION,
)

logger = logging.getLogger(__name__)

# Kalshi exposes depth only live. The websocket is on a SEPARATE host from REST
# (REST: external-api.kalshi.com; WS: external-api-ws.kalshi.com), same signed
# path. The signature is ts + "GET" + WS_SIGN_PATH, identical to our REST signing.
WS_URL = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_SIGN_PATH = "/trade-api/ws/v2"
CHANNELS = ["orderbook_delta", "trade"]
OUT_DIR = Path("data/marketdata")
FLUSH_SECONDS = 30


def _ws_auth_headers(private_key, api_key: str, sign_path: str) -> dict[str, str]:
    """Signed websocket handshake headers. Kalshi's WS requires RSA-PSS (the REST
    client's PKCS1v15 is accepted for REST but rejected here with 401). Message is
    timestamp_ms + "GET" + sign_path."""
    ts = str(int(time.time() * 1000))
    sig = private_key.sign(
        (ts + "GET" + sign_path).encode(),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": api_key,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        "KALSHI-ACCESS-TIMESTAMP": ts,
    }


async def fetch_open_temp_tickers(client: ReadOnlyKalshiClient) -> list[str]:
    """Open (tradeable) high-temp market tickers across the known series."""
    tickers: list[str] = []
    for series in SERIES_STATION:
        if is_low_temp_series(series):
            continue
        try:
            data = await client._get(
                "/markets", params={"series_ticker": series, "status": "open", "limit": 1000}
            )
            tickers.extend(m["ticker"] for m in data.get("markets", []) if m.get("ticker"))
        except Exception as exc:
            logger.debug("market list failed for %s: %s", series, exc)
    return tickers


class _Buffers:
    def __init__(self) -> None:
        self.book: list[dict] = []
        self.trades: list[dict] = []
        self.last_top: dict[str, tuple] = {}

    def record_top(self, ticker: str, ob: OrderBook, ts: float) -> None:
        top = ob.top()
        key = (top["yes_bid"], top["yes_ask"], top["yes_bid_sz"], top["yes_ask_sz"])
        if self.last_top.get(ticker) == key:
            return  # only log genuine top-of-book changes
        self.last_top[ticker] = key
        self.book.append({"ts": ts, "ticker": ticker, **top})

    def record_trade(self, d: dict, ts: float) -> None:
        self.trades.append({
            "ts": ts,
            "ticker": d.get("market_ticker"),
            "yes_price": d.get("yes_price_dollars") or d.get("yes_price"),
            "count": d.get("count_fp") or d.get("count"),
            "taker_side": d.get("taker_side"),
        })


def _event_ts(d: dict) -> float:
    """Server event time (epoch seconds) from ts_ms, falling back to local clock."""
    ms = d.get("ts_ms")
    return ms / 1000.0 if ms else time.time()


def _dispatch(raw: str, books: dict[str, OrderBook], buf: _Buffers) -> None:
    m = json.loads(raw)
    typ = m.get("type")
    d = m.get("msg", {})
    seq = m.get("seq", d.get("seq"))
    if typ == "orderbook_snapshot":
        tk = d["market_ticker"]
        ob = books.setdefault(tk, OrderBook(tk))
        ob.apply_snapshot(d.get("yes_dollars_fp") or d.get("yes") or [],
                          d.get("no_dollars_fp") or d.get("no") or [], seq=seq)
        buf.record_top(tk, ob, _event_ts(d))
    elif typ == "orderbook_delta":
        tk = d["market_ticker"]
        ob = books.get(tk)
        if ob is None:
            return
        ob.apply_delta(d.get("price_dollars", d.get("price")),
                       d.get("delta_fp", d.get("delta")), d["side"], seq=seq)
        buf.record_top(tk, ob, _event_ts(d))
    elif typ == "trade":
        buf.record_trade(d, _event_ts(d))
    elif typ in ("subscribed", "ok"):
        logger.info("WS %s", m)
    elif typ == "error":
        logger.error("WS error: %s", m)


def _flush(buf: _Buffers) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for name, rows in (("book", buf.book), ("trades", buf.trades)):
        if not rows:
            continue
        d = OUT_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(d / f"{name}_{stamp}.parquet", index=False)
        logger.info("flushed %d %s rows", len(rows), name)
    buf.book.clear()
    buf.trades.clear()


async def _subscribe(ws, tickers: list[str]) -> None:
    await ws.send(json.dumps({
        "id": 1, "cmd": "subscribe",
        "params": {"channels": CHANNELS, "market_tickers": tickers},
    }))


async def run(hours: float, smoke: bool = False) -> None:
    settings = get_settings()
    client = ReadOnlyKalshiClient(
        api_key=LIVE_API_KEY,
        private_key_path=settings.KALSHI_PRIVATE_KEY_PATH,
        base_url=LIVE_BASE_URL,
    )
    tickers = await fetch_open_temp_tickers(client)
    if smoke:
        tickers = tickers[:3]
    logger.info("Subscribing to %d tickers on %s", len(tickers), WS_URL)
    if not tickers:
        logger.error("No open temperature markets to subscribe to.")
        return

    headers = _ws_auth_headers(client._private_key, LIVE_API_KEY, WS_SIGN_PATH)
    # Smoke mode is a bounded connectivity/schema check — never loop on silence.
    end_time = time.time() + (25 if smoke else hours * 3600)
    books: dict[str, OrderBook] = {}
    buf = _Buffers()
    last_flush = time.time()

    while time.time() < end_time:
        try:
            async with websockets.connect(WS_URL, additional_headers=headers, max_size=2**22) as ws:
                await _subscribe(ws, tickers)
                printed = 0
                while time.time() < end_time:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60.0)
                    if smoke:
                        print(raw[:400])
                        printed += 1
                        if printed >= 20:
                            return
                        continue
                    _dispatch(raw, books, buf)
                    if time.time() - last_flush >= FLUSH_SECONDS:
                        _flush(buf)
                        last_flush = time.time()
        except asyncio.TimeoutError:
            logger.warning("WS idle 60s; reconnecting")
        except Exception as exc:
            logger.warning("WS connection error: %s; reconnecting in 3s", exc)
            await asyncio.sleep(3)
        finally:
            if not smoke:
                _flush(buf)
        # refresh headers (timestamped) + ticker list on reconnect
        headers = _ws_auth_headers(client._private_key, LIVE_API_KEY, WS_SIGN_PATH)
        tickers = await fetch_open_temp_tickers(client) or tickers


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--smoke", action="store_true", help="print raw frames and exit")
    args = ap.parse_args()
    asyncio.run(run(args.hours, smoke=args.smoke))


if __name__ == "__main__":
    main()
