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
        message = json.dumps({
            "type": "state_update",
            "timestamp": datetime.utcnow().isoformat(),
            **payload,
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


async def websocket_endpoint(ws: WebSocket, app_state) -> None:
    broadcaster: WebSocketBroadcaster = app_state.ws_broadcaster
    shared_state = app_state.shared_state
    position_tracker = app_state.position_tracker
    settings = app_state.settings

    await broadcaster.connect(ws)
    try:
        # Send initial snapshot on connect
        snap = shared_state.snapshot()
        await ws.send_text(json.dumps({
            "type": "state_update",
            "timestamp": datetime.utcnow().isoformat(),
            "markets": list(snap.values()),
            "positions": position_tracker.get_all_positions(),
            "pnl": position_tracker.total_pnl_series(),
            "bot_active": settings.BOT_ACTIVE,
        }))

        # Push updates every 10 seconds
        while True:
            await asyncio.sleep(10)
            snap = shared_state.snapshot()
            await ws.send_text(json.dumps({
                "type": "state_update",
                "timestamp": datetime.utcnow().isoformat(),
                "markets": [{"ticker": k, **v} for k, v in snap.items()],
                "positions": position_tracker.get_all_positions(),
                "pnl": {"series": position_tracker.total_pnl_series()},
                "bot_active": settings.BOT_ACTIVE,
            }))
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        await broadcaster.disconnect(ws)
