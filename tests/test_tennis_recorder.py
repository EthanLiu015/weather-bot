"""Regression coverage for live/tennis_recorder.py Recorder.send().

2026-07-28: an uncaught ConnectionClosedError from ws.send() (raised from
market_discovery_loop's unsubscribe(), racing ws_loop's reconnect) propagated
through asyncio.gather and killed the whole recorder process silently for 8h.
send() must swallow ws failures and set needs_resync instead of raising.
"""
import websockets.exceptions

from live.tennis_recorder import Recorder


class _RaisingWs:
    async def send(self, _payload: str) -> None:
        raise websockets.exceptions.ConnectionClosedError(None, None)


class _OkWs:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


async def test_send_sets_needs_resync_and_does_not_raise_when_ws_closed() -> None:
    rec = Recorder()
    rec.ws = _RaisingWs()

    cmd_id = await rec.send({"cmd": "update_subscription", "params": {}})

    assert cmd_id == 1
    assert rec.needs_resync is True


async def test_send_sets_needs_resync_and_does_not_raise_when_ws_is_none() -> None:
    rec = Recorder()
    rec.ws = None

    cmd_id = await rec.send({"cmd": "subscribe", "params": {}})

    assert cmd_id == 1
    assert rec.needs_resync is True


async def test_send_delivers_payload_and_leaves_resync_false_on_success() -> None:
    rec = Recorder()
    rec.ws = _OkWs()

    await rec.send({"cmd": "subscribe", "params": {"channels": ["ticker"]}})

    assert rec.needs_resync is False
    assert len(rec.ws.sent) == 1
