"""Connected computers: the default-off feature preview and its live teardown.

* fresh install -- no preview entry means off on both machines.
* crafted request while off -- every gate raises the typed refusal; only an
  exact ``True`` turns it on.
* disabled during live execution -- the config watcher closes every connector
  socket, without touching files.
* while it stays off -- nothing is scheduled and shutdown closes nothing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from kiro_crew.connected_computers import broker as broker_mod
from kiro_crew.connected_computers import disable, flag
from kiro_crew.testing.wait import FALLBACK_TIMEOUT_SECS

pytestmark = pytest.mark.asyncio


def _cfg(previews: Any) -> Any:
    return SimpleNamespace(dashboard=SimpleNamespace(feature_previews=previews))


class _Change:
    def __init__(self, old: Any, new: Any, touched: bool = True) -> None:
        self.old = old
        self.new = new
        self._touched = touched

    def touched(self, *_paths: str) -> bool:
        return self._touched


class _FakeBroker:
    def __init__(self) -> None:
        self.closed: list[str] = []

    async def shutdown_all(self, why: str) -> int:
        self.closed.append(why)
        return 1


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(disable, "_TASKS", set())
    b = _FakeBroker()
    monkeypatch.setattr(broker_mod, "get_broker", lambda: b)
    return b


class TestFreshInstallIsOff:
    @pytest.mark.parametrize("previews", [None, {}, {"other": True}])
    def test_flag_off_on_fresh_install(self, previews):
        cfg = _cfg(previews)
        assert flag.preview_on(cfg) is False
        assert flag.local_enabled(cfg) is False
        assert flag.remote_enabled(cfg) is False

    @pytest.mark.parametrize("value", ["true", 1, "1", "yes", [True]])
    def test_only_an_exact_true_turns_it_on(self, value):
        assert flag.preview_on(_cfg({flag.PREVIEW_NAME: value})) is False

    def test_exact_true_turns_the_local_side_on(self):
        assert flag.local_enabled(_cfg({flag.PREVIEW_NAME: True})) is True


class TestCraftedRequestWhileOff:
    def test_require_remote_refuses_with_typed_reason(self):
        with pytest.raises(flag.ConnectedComputersDisabled) as err:
            flag.require_remote(_cfg({}))
        assert err.value.code == "connected_computers_disabled"
        assert "turned off" in str(err.value)

    def test_require_local_refuses_with_typed_reason(self):
        with pytest.raises(flag.ConnectedComputersDisabled):
            flag.require_local(_cfg({flag.PREVIEW_NAME: False}))

    def test_remote_needs_the_existing_developer_gate_too(self, monkeypatch):
        import kiro_crew.session_environment as se

        monkeypatch.setattr(se, "session_environments_enabled", lambda: False)
        assert flag.remote_enabled(_cfg({flag.PREVIEW_NAME: True})) is False

    def test_unreadable_config_reads_as_off(self, monkeypatch):
        monkeypatch.setattr(flag, "_config", lambda cfg: None)
        assert flag.preview_on() is False


async def _drain() -> None:
    await asyncio.wait_for(asyncio.gather(*list(disable._TASKS)), FALLBACK_TIMEOUT_SECS)


async def test_disabled_during_live_execution_stops_sockets(fakes):
    b = fakes
    disable.apply(_Change(_cfg({flag.PREVIEW_NAME: True}), _cfg({})))
    await _drain()
    assert len(b.closed) == 1


async def test_unrelated_config_change_does_nothing(fakes):
    b = fakes
    disable.apply(_Change(_cfg({}), _cfg({flag.PREVIEW_NAME: True}), touched=False))
    assert disable._TASKS == set()
    assert b.closed == []


@pytest.mark.parametrize("value", [False, True])
async def test_unchanged_value_does_nothing(fakes, value):
    b = fakes
    previews = {flag.PREVIEW_NAME: value}
    disable.apply(_Change(_cfg(previews), _cfg(dict(previews))))
    assert disable._TASKS == set()
    assert b.closed == []


async def test_another_preview_toggled_while_off_schedules_nothing(fakes):
    """Flag-off parity: a gateway that never turned this on does no new work."""
    b = fakes
    disable.apply(_Change(_cfg({"crew_mode": False}), _cfg({"crew_mode": True})))
    assert disable._TASKS == set()
    assert b.closed == []


async def test_turning_it_on_starts_nothing_on_the_remote(fakes):
    b = fakes
    disable.apply(_Change(_cfg({}), _cfg({flag.PREVIEW_NAME: True})))
    assert disable._TASKS == set()
    assert b.closed == []


async def test_shutdown_with_no_broker_closes_nothing(monkeypatch, fakes):
    b = fakes
    monkeypatch.setattr(broker_mod, "_BROKER", None)
    await asyncio.wait_for(disable.shutdown(), FALLBACK_TIMEOUT_SECS)
    assert b.closed == []


class TestRemoteRefusalNamesTheMissingGate:
    """A remote gateway with the preview on but no ``KIROCREW_DEV_MODE`` says so."""

    def test_flag_preview_off_keeps_the_settings_reason(self, monkeypatch):
        import kiro_crew.session_environment as se

        monkeypatch.setattr(se, "session_environments_enabled", lambda: True)
        assert flag.remote_refusal(_cfg({})) == flag.DISABLED_REASON

    def test_flag_developer_gate_off_names_the_env_var(self, monkeypatch):
        import kiro_crew.session_environment as se

        monkeypatch.setattr(se, "session_environments_enabled", lambda: False)
        reason = flag.remote_refusal(_cfg({flag.PREVIEW_NAME: True}))
        assert reason == flag.DEVELOPER_GATE_REASON
        assert se.DEVELOPER_FLAG in reason
        with pytest.raises(flag.ConnectedComputersDisabled) as err:
            flag.require_remote(_cfg({flag.PREVIEW_NAME: True}))
        assert str(err.value) == flag.DEVELOPER_GATE_REASON
        assert err.value.code == "connected_computers_disabled"

    def test_flag_both_on_has_no_refusal(self, monkeypatch):
        import kiro_crew.session_environment as se

        monkeypatch.setattr(se, "session_environments_enabled", lambda: True)
        assert flag.remote_refusal(_cfg({flag.PREVIEW_NAME: True})) is None
        flag.require_remote(_cfg({flag.PREVIEW_NAME: True}))


class TestComputerServersMeetTheGateOnlyWhileOn:
    """Flag-off parity: a person's own ``computer-*`` server keeps its old answer."""

    @pytest.fixture
    def snapshot(self, monkeypatch):
        from kiro_crew.config import live

        def use(previews: Any) -> None:
            monkeypatch.setattr(live, "snapshot", lambda: _cfg(previews))

        return use

    def test_off_keeps_a_computer_named_server_auto_approved(self, snapshot):
        from kiro_crew.platform.governance import may_skip_gate

        snapshot({})
        assert may_skip_gate("@computer-notes", None) is True
        assert may_skip_gate("@computer-notes/write_note", None) is True

    def test_on_sends_every_computer_call_to_the_gate(self, snapshot):
        from kiro_crew.platform.governance import may_skip_gate

        snapshot({flag.PREVIEW_NAME: True})
        assert may_skip_gate("@computer-notes", None) is False
        assert may_skip_gate("@computer-notes/write_note", None) is False

    def test_no_snapshot_keeps_the_gate(self, monkeypatch):
        from kiro_crew.config import live
        from kiro_crew.platform.governance import may_skip_gate

        monkeypatch.setattr(live, "snapshot", lambda: None)
        assert may_skip_gate("@computer-notes", None) is False

    @pytest.mark.parametrize("previews", [{}, {flag.PREVIEW_NAME: True}])
    def test_workspace_meets_the_gate_either_way(self, snapshot, previews):
        from kiro_crew.platform.governance import may_skip_gate

        snapshot(previews)
        assert may_skip_gate("@workspace", None) is False
