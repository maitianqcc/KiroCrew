"""The relay viewer token route does not wait on the default executor.

Each test fills the event loop's default executor with a job that blocks until the test
releases it, then mints a Crew token or calls the control plane. Before
``relay_viewer_executor`` the settings, credential and sign-in reads and the DNS lookup
queued behind that job, so the call hung.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
from collections.abc import Coroutine, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, make_mocked_request

from kiro_crew.auth.store import KasToken
from kiro_crew.dashboard.handlers import relay_viewer as rv
from kiro_crew.member_memory_auth import LocalOwnerProvenance
from kiro_crew.relay_viewer import cps

SECRET = "local-secret-value"
MINT_DEADLINE_SECS = 10
EP = "a" * 24
CREW = cps.CrewEndpoint(
    endpoint_id=EP,
    crew_url=f"https://{EP}.relay.example.test",
    name="Build box",
    crew_type="PERSONAL",
    presence="CONNECTED",
    relay_connection_status="CONNECTED",
)
_T = TypeVar("_T")


def _token(access: str, expires_in: timedelta) -> KasToken:
    return KasToken(
        access_token=access,
        expires_at=datetime.now(timezone.utc) + expires_in,
        provider="BuilderId",
        identity="social",
    )


class _Store:
    """The vault: its stored token is expired, and a peer has already refreshed it."""

    def resolve(self) -> KasToken:
        return _token("stale-sign-in", timedelta(0))

    def load(self, _identity: str) -> KasToken:
        return _token("fresh-sign-in", timedelta(hours=1))


class _StoreRefreshedUnderLock(_Store):
    """A peer's refresh lands only while this process holds the refresh lock."""

    def __init__(self, lock_dir: Path) -> None:
        self._lock_dir = lock_dir
        self._loads = 0

    def lock_path(self, identity: str) -> Path:
        return self._lock_dir / f"{identity}.lock"

    def load(self, identity: str) -> KasToken:
        self._loads += 1
        return super().load(identity) if self._loads > 1 else self.resolve()


@pytest.fixture
def viewer(monkeypatch) -> SimpleNamespace:
    """The viewer on, CPS faked, the host-process gate passing."""
    tunnel = SimpleNamespace(
        relay_viewer_enabled=True,
        relay_viewer_cps_url="https://cps.example.test",
        relay_viewer_credential="",
        relay_viewer_relay_domain="relay.example.test",
    )
    state = SimpleNamespace(tunnel=tunnel, bearers=[])

    async def _list(url, bearer, *, crew_domain, token_type=""):
        state.bearers.append(bearer)
        return (CREW,)

    async def _mint(url, bearer, endpoint_id, *, token_type=""):
        return cps.CrewAccessToken("access-token-value", 2000.0, 240.0, 1700.0)

    monkeypatch.setattr(rv, "_load_tunnel", lambda: tunnel)
    monkeypatch.setattr(cps, "list_crew_endpoints", _list)
    monkeypatch.setattr(cps, "create_crew_access_token", _mint)
    monkeypatch.setattr("kiro_crew.auth.bridge.default_token_store", _Store)
    monkeypatch.setattr("kiro_crew.auth.bridge.vault_identity_fingerprint", lambda: "fp")
    monkeypatch.setattr(
        "kiro_crew.relay_client.lease.read_lease_credential", lambda _ref: "test-credential"
    )
    monkeypatch.setattr(
        "kiro_crew.member_memory_auth.local_owner_bootstrap_provenance",
        lambda _request: LocalOwnerProvenance(),
    )
    monkeypatch.setattr("kiro_crew.sel.sel", lambda: MagicMock())
    return state


def _mint_request() -> web.Request:
    sock = MagicMock()
    sock.family = socket.AF_INET
    transport = MagicMock()
    transport.get_extra_info.side_effect = lambda key, default=None: {
        "socket": sock,
        "peername": ("127.0.0.1", 43123),
    }.get(key, default)
    app = web.Application()
    app["local_secret"] = SECRET
    request = make_mocked_request(
        "POST",
        "/api/relay-viewer/token",
        headers={"X-Local-Secret": SECRET, "Content-Type": "application/json"},
        app=app,
        transport=transport,
    )
    # A mocked request has no body stream; aiohttp answers read() from this cache.
    request._read_bytes = json.dumps({"endpointId": EP}).encode()
    return request


@contextlib.contextmanager
def _default_executor_held() -> Iterator[None]:
    """Make the event loop's default executor one thread, and hold that thread."""
    pool = ThreadPoolExecutor(max_workers=1)
    asyncio.get_running_loop().set_default_executor(pool)
    release = threading.Event()
    started = threading.Event()

    def _hold() -> None:
        started.set()
        release.wait()

    try:
        pool.submit(_hold)
        assert started.wait(MINT_DEADLINE_SECS)
        yield
    finally:
        release.set()


async def _finished_within_deadline(call: Coroutine[Any, Any, _T]) -> _T:
    task = asyncio.ensure_future(call)
    done, _pending = await asyncio.wait({task}, timeout=MINT_DEADLINE_SECS)
    if task not in done:
        task.cancel()
    assert task in done, "the call is still queued behind the default executor"
    return task.result()


@pytest.mark.asyncio
async def test_mints_with_the_vault_sign_in_while_the_default_executor_is_full(
    viewer,
) -> None:
    with _default_executor_held():
        response = await _finished_within_deadline(rv.api_relay_viewer_token(_mint_request()))

    assert response.status == 200
    assert viewer.bearers == ["fresh-sign-in"]


@pytest.mark.asyncio
async def test_mints_through_the_refresh_lock_while_the_default_executor_is_full(
    viewer, monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "kiro_crew.auth.bridge.default_token_store", lambda: _StoreRefreshedUnderLock(tmp_path)
    )

    with _default_executor_held():
        response = await _finished_within_deadline(rv.api_relay_viewer_token(_mint_request()))

    assert response.status == 200
    assert viewer.bearers == ["fresh-sign-in"]


@pytest.mark.asyncio
async def test_mints_with_the_test_credential_while_the_default_executor_is_full(
    viewer,
) -> None:
    viewer.tunnel.relay_viewer_credential = "~/relay-viewer-credential"

    with _default_executor_held():
        response = await _finished_within_deadline(rv.api_relay_viewer_token(_mint_request()))

    assert response.status == 200
    assert viewer.bearers == ["test-credential"]


@pytest.mark.asyncio
async def test_a_control_plane_call_resolves_its_host_while_the_default_executor_is_full(
    monkeypatch,
) -> None:
    async def _answer(_request: web.Request) -> web.Response:
        return web.json_response({"crewEndpoints": []})

    app = web.Application()
    app.router.add_post("/", _answer)
    monkeypatch.setattr(cps, "env_proxy_for", lambda _url: None)

    async with TestServer(app, host="127.0.0.1") as server:
        url = f"http://localhost:{server.port}/"
        with _default_executor_held():
            answer = await _finished_within_deadline(
                cps._post(url, "bearer-value", cps.LIST_TARGET, {}, "")
            )

    assert answer.status == 200
