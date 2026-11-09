"""An app socket's connect snapshot is judged by the scopes it holds when it is sent.

``_serve_ws`` resolves an app's scopes at connect, then waits for the shared slot-list
pass. An operator who disables the app meanwhile must not see the stale scopes decide
what the snapshot carries (``effective_allowed_events``).
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from chat_test_helpers import _make_state

from kiro_crew.dashboard import ws as ws_mod
from kiro_crew.dashboard import ws_event_scope
from kiro_crew.dashboard.state import SlotOrigin

APP = "mochi-pet"
SCOPES = frozenset({"slots:all", "yolo"})


@pytest.fixture(autouse=True)
def _clean_declared_cache():
    ws_event_scope._declared_cache.clear()
    yield
    ws_event_scope._declared_cache.clear()


async def _connect_snapshot(tmp_path, monkeypatch, *, revoke: bool) -> dict:
    state = _make_state(tmp_path)
    foreign = SimpleNamespace(_app="other-app", _origin=SlotOrigin.USER, key="foreign-1")
    state._slots = {"foreign-1": foreign}
    state._yolo = True

    async def _enabled_at_connect(app: str):
        ws_event_scope._declared_cache[app] = (time.monotonic(), True, SCOPES)
        return True, SCOPES

    async def _shared_pass_that_outlives_the_grant(st, **_kw):
        # The operator disables the app while the connect waits for the pass.
        if revoke:
            ws_event_scope._declared_cache[APP] = (time.monotonic(), False, frozenset())
        return [{"key": "foreign-1", "title": "someone else's session"}]

    async def _to_thread(fn, *args):
        return await _enabled_at_connect(*args)

    monkeypatch.setattr(ws_mod.asyncio, "to_thread", _to_thread)
    monkeypatch.setattr(ws_mod, "shared_slot_list", _shared_pass_that_outlives_the_grant)

    @web.middleware
    async def _as_the_app(request, handler):
        request["app"] = APP
        return await handler(request)

    app = web.Application(middlewares=[_as_the_app])
    app["state"] = state
    app["allowed_origins"] = set()
    app.router.add_get("/api/ws", ws_mod.api_ws)
    async with TestClient(TestServer(app)) as client:
        origin = str(client.make_url("/")).rstrip("/")
        app["allowed_origins"].add(origin)
        client.session.headers["Origin"] = origin
        sock = await client.ws_connect("/api/ws")
        frame = json.loads(
            (await sock.receive(timeout=30)).data  # flake-ok: the bound only caps a hang
        )
        await sock.close()

    return frame


@pytest.mark.asyncio
async def test_a_revocation_during_the_shared_pass_narrows_the_snapshot(
    tmp_path, monkeypatch
) -> None:
    frame = await _connect_snapshot(tmp_path, monkeypatch, revoke=True)
    assert frame["type"] == "slots"
    assert frame["data"] == [], "a disabled app was sent another session's metadata"
    assert "yolo" not in frame, "a disabled app was sent the operator's security posture"


@pytest.mark.asyncio
async def test_a_still_enabled_app_keeps_its_scoped_snapshot(tmp_path, monkeypatch) -> None:
    frame = await _connect_snapshot(tmp_path, monkeypatch, revoke=False)
    assert [r["key"] for r in frame["data"]] == ["foreign-1"]
    assert frame["yolo"] is True
