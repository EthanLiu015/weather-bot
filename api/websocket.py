import asyncio
import json
import logging
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class WebSocketBroadcaster:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
        logger.info("WebSocket client connected; total=%d", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.discard(ws)
        logger.info("WebSocket client disconnected; total=%d", len(self._connections))

    async def broadcast(self, payload: dict) -> None:
        if not self._connections:
            return
        # Normalise markets to always be an array with ticker included
        markets_raw = payload.get("markets", {})
        if isinstance(markets_raw, dict):
            markets = [{"ticker": k, **v} for k, v in markets_raw.items()]
        else:
            markets = markets_raw

        message = json.dumps({
            "type": "state_update",
            "timestamp": datetime.utcnow().isoformat(),
            "markets": markets,
            "alerts": payload.get("alerts", []),
        })
        dead = set()
        async with self._lock:
            connections = set(self._connections)
        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self._connections -= dead


def _build_message(shared_state, position_tracker, settings) -> str:
    """Build a consistent WebSocket message with markets as an array."""
    snap = shared_state.snapshot()
    markets = [{"ticker": k, **v} for k, v in snap.items()]
    return json.dumps({
        "type": "state_update",
        "timestamp": datetime.utcnow().isoformat(),
        "markets": markets,
        "positions": position_tracker.get_all_positions(),
        "pnl": {"series": position_tracker.total_pnl_series()},
        "bot_active": settings.BOT_ACTIVE,
        "paper_trading": settings.PAPER_TRADING,
        "alerts": shared_state.get_alerts(),
    })


async def websocket_endpoint(ws: WebSocket, app_state) -> None:
    broadcaster: WebSocketBroadcaster = app_state.ws_broadcaster
    shared_state = app_state.shared_state
    position_tracker = app_state.position_tracker
    settings = app_state.settings

    await broadcaster.connect(ws)
    try:
        # Send full snapshot immediately on connect
        await ws.send_text(_build_message(shared_state, position_tracker, settings))

        # Push updates every 10 seconds
        while True:
            await asyncio.sleep(10)
            await ws.send_text(_build_message(shared_state, position_tracker, settings))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        await broadcaster.disconnect(ws)
