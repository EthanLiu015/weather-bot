import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from pathlib import Path

import httpx
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

logger = logging.getLogger(__name__)

_TOKEN_BUCKET_CAPACITY = 10.0
_TOKEN_REFILL_RATE = 10.0  # tokens per second


class _TokenBucket:
    def __init__(self, capacity: float = _TOKEN_BUCKET_CAPACITY, rate: float = _TOKEN_REFILL_RATE) -> None:
        self._capacity = capacity
        self._rate = rate
        self._tokens = capacity
        self._last_refill = time.monotonic()

    async def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            await asyncio.sleep(0.05)


class KalshiClient:
    def __init__(self, api_key: str, private_key_path: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._bucket = _TokenBucket()
        pem = Path(private_key_path).read_bytes()
        self._private_key = serialization.load_pem_private_key(pem, password=None)

    def _sign_request(self, method: str, path: str, body: str = "") -> dict[str, str]:
        ts_ms = str(int(time.time() * 1000))
        msg = ts_ms + method.upper() + path + body
        signature = self._private_key.sign(
            msg.encode(),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        sig_b64 = base64.b64encode(signature).decode()
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts_ms,
            "KALSHI-ACCESS-SIGNATURE": sig_b64,
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        path: str,
        params: dict | None = None,
        body: dict | None = None,
        retries: int = 5,
    ) -> dict:
        await self._bucket.acquire()
        url = self._base_url + path
        body_str = json.dumps(body) if body else ""
        headers = self._sign_request(method, path, body_str)

        for attempt in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.request(
                        method,
                        url,
                        headers=headers,
                        params=params,
                        content=body_str.encode() if body_str else None,
                        timeout=30.0,
                    )
                if resp.status_code == 429:
                    wait = 2**attempt
                    logger.warning("Rate limited; retrying in %ds", wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                if attempt == retries - 1:
                    logger.error("Request %s %s failed: %s", method, path, exc)
                    raise
                await asyncio.sleep(2**attempt)
            except Exception as exc:
                if attempt == retries - 1:
                    raise
                logger.warning("Request error (attempt %d): %s", attempt + 1, exc)
                await asyncio.sleep(2**attempt)
        raise RuntimeError(f"All {retries} attempts failed for {method} {path}")

    async def get_markets(self, status: str = "open", category: str = "temperature") -> list[dict]:
        data = await self._request("GET", "/markets", params={"status": status, "category": category})
        return data.get("markets", [])

    async def get_market(self, ticker: str) -> dict:
        data = await self._request("GET", f"/markets/{ticker}")
        return data.get("market", data)

    async def get_orderbook(self, ticker: str) -> dict:
        data = await self._request("GET", f"/markets/{ticker}/orderbook")
        return data

    async def get_candlesticks(self, ticker: str, period_interval: int = 1440) -> list[dict]:
        data = await self._request("GET", f"/series/{ticker}/markets/{ticker}/candlesticks",
                                   params={"period_interval": period_interval})
        return data.get("candlesticks", [])

    async def get_historical_markets(self, **kwargs) -> list[dict]:
        data = await self._request("GET", "/markets", params={**kwargs, "status": "settled"})
        return data.get("markets", [])

    async def create_order(
        self,
        ticker: str,
        side: str,
        price: int,
        count: int,
        order_type: str = "limit",
    ) -> dict:
        body = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "type": order_type,
            "yes_price": price if side == "yes" else 100 - price,
            "no_price": 100 - price if side == "yes" else price,
            "count": count,
        }
        data = await self._request("POST", "/portfolio/orders", body=body)
        logger.info("Order created: %s %s @ %d × %d", ticker, side, price, count)
        return data.get("order", data)

    async def cancel_order(self, order_id: str) -> dict:
        data = await self._request("DELETE", f"/portfolio/orders/{order_id}")
        logger.info("Order cancelled: %s", order_id)
        return data

    async def get_fills(self, ticker: str | None = None) -> list[dict]:
        params = {}
        if ticker:
            params["ticker"] = ticker
        data = await self._request("GET", "/portfolio/fills", params=params)
        return data.get("fills", [])

    async def get_positions(self) -> list[dict]:
        data = await self._request("GET", "/portfolio/positions")
        return data.get("market_positions", [])

    async def get_balance(self) -> dict:
        data = await self._request("GET", "/portfolio/balance")
        return data
