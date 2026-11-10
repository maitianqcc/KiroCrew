"""A cron or channel claim re-projects a stale KAS batch.

``spec_hooks.reproject_unattended_claim`` runs after the turn's Stop snapshot,
opens the replay gap before its first reset, and leaves it for the caller to
close after its release. A batch still stale after two resets keeps the
stale-batch fallback instead of refusing the turn. The shared channel pipeline is
driven end to end through the channel-turn fakes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from channel_turn_fakes import FakeSessions, RecordingRenderer, ScriptedProvider, answer
from channel_turns_helpers import _answer, _turns

from kiro_crew.acp.types import STOP_REASON_CANCELLED
from kiro_crew.agent_sdk import spec_hooks
from kiro_crew.messaging.dispatch import Verdict


def _stale_for(*providers: Any):
    """A ``hook_projection_stale`` that answers True for *providers* only."""

    async def _stale(provider: Any, _agent: str) -> bool:
        return any(provider is p for p in providers)

    return _stale


class _Sessions:
    """The parts of the session manager the helper touches, as one ledger."""

    def __init__(self, *, on_reset: Any = None, reset_raises: bool = False) -> None:
        self.ledger: list[str] = []
        self.gap_open = False
        self.claims = 0
        self.on_reset = on_reset
        self.reset_raises = reset_raises

    def open_replay_gap(self, key: str) -> None:
        self.gap_open = True
        self.ledger.append("open_replay_gap")

    def close_replay_gap(self, key: str) -> None:
        self.gap_open = False
        self.ledger.append("close_replay_gap")

    async def reset(self, key: str) -> None:
        self.ledger.append("reset")
        if self.reset_raises:
            raise RuntimeError("reset failed")
        if self.on_reset is not None:
            self.on_reset(self)

    def adopt_turn(self, key: str) -> None:
        self.ledger.append("adopt_turn")


def _run(sessions: _Sessions, first: Any, *successors: Any) -> spec_hooks.UnattendedClaim:
    queue = list(successors)

    async def _claim() -> tuple[Any, bool, bool]:
        sessions.claims += 1
        sessions.ledger.append("claim")
        return queue.pop(0) if len(queue) > 1 else queue[0], True, False

    return asyncio.run(
        spec_hooks.reproject_unattended_claim(sessions, "k", "agent", (first, False, False), _claim)
    )


class TestTheHelper:
    def test_a_fresh_batch_is_left_alone(self, monkeypatch) -> None:
        old = object()
        monkeypatch.setattr(spec_hooks, "hook_projection_stale", _stale_for())
        sessions = _Sessions()

        got = _run(sessions, old, object())

        assert got == spec_hooks.UnattendedClaim((old, False, False), False, False)
        assert sessions.ledger == []

    def test_a_stale_batch_is_reset_and_claimed_again_under_the_gap(self, monkeypatch) -> None:
        old, new = object(), object()
        monkeypatch.setattr(spec_hooks, "hook_projection_stale", _stale_for(old))
        sessions = _Sessions()

        got = _run(sessions, old, new)

        assert got.claimed == (new, True, False)
        assert got.reprojected is True and got.still_stale is False
        # Opened before the reset; left open for the caller to close after release.
        assert sessions.ledger == ["open_replay_gap", "reset", "claim", "adopt_turn"]
        assert sessions.gap_open is True

    def test_still_stale_after_two_resets_falls_back_without_refusing(self, monkeypatch) -> None:
        old, new = object(), object()
        monkeypatch.setattr(spec_hooks, "hook_projection_stale", _stale_for(old, new))
        sessions = _Sessions()

        got = _run(sessions, old, new)

        assert got.claimed[0] is new and got.reprojected is True and got.still_stale is True
        assert sessions.ledger.count("reset") == spec_hooks._MAX_REPROJECTIONS
        assert sessions.ledger.count("open_replay_gap") == 1

    def test_a_failed_reset_closes_the_gap_it_opened(self, monkeypatch) -> None:
        old = object()
        monkeypatch.setattr(spec_hooks, "hook_projection_stale", _stale_for(old))
        sessions = _Sessions(reset_raises=True)

        with pytest.raises(RuntimeError):
            _run(sessions, old, object())

        assert sessions.ledger == ["open_replay_gap", "reset", "close_replay_gap"]
        assert sessions.gap_open is False

    def test_the_dashboard_claim_still_refuses_a_batch_it_cannot_cure(self, monkeypatch) -> None:
        old, new = object(), object()
        monkeypatch.setattr(spec_hooks, "hook_projection_stale", _stale_for(old, new))
        sessions = _Sessions()

        async def _claim() -> tuple[Any, bool, bool]:
            return new, True, False

        with pytest.raises(spec_hooks.StaleProjectionError):
            asyncio.run(
                spec_hooks.reproject_claimed_session(
                    sessions, "k", "agent", (old, False, False), _claim
                )
            )
        assert "open_replay_gap" not in sessions.ledger


class TestTheSharedChannelPipeline:
    def test_a_stale_claim_runs_on_the_re_projected_session(self, monkeypatch) -> None:
        old = ScriptedProvider(answer("must not run"), name="old")
        new = ScriptedProvider(answer("the reply"), name="new")
        monkeypatch.setattr(spec_hooks, "hook_projection_stale", _stale_for(old))
        sessions = FakeSessions(old, new)

        outcome = _answer(_turns(sessions))

        assert outcome.verdict is Verdict.ANSWERED
        assert old.prompts == [] and new.prompts == ["hi"]
        names = sessions.names()
        assert names.count("get_or_create") == 2 and names.count("release") == 1
        assert names.index("open_replay_gap") < names.index("reset")
        assert names.index("release") < names.index("close_replay_gap")
        assert sessions.gap_open is False

    def test_a_stop_inside_the_re_projection_ends_the_turn_unprompted(self, monkeypatch) -> None:
        old = ScriptedProvider(answer("must not run"), name="old")
        new = ScriptedProvider(answer("must not run"), name="new")
        monkeypatch.setattr(spec_hooks, "hook_projection_stale", _stale_for(old))

        def _stop_in_gap(sessions: FakeSessions) -> None:
            # Recorded only because the gap is open while the key has no session.
            if sessions.gap_open:
                sessions.stop_gen += 1

        sessions = FakeSessions(old, new, on_reset=_stop_in_gap)
        renderer = RecordingRenderer()

        outcome = _answer(_turns(sessions), renderer=renderer)

        assert outcome.verdict is Verdict.ABANDONED
        assert old.prompts == [] and new.prompts == []
        assert [e.stop_reason for e in renderer.done] == [STOP_REASON_CANCELLED]
        assert sessions.count("release") == 1 and sessions.gap_open is False

    def test_a_fresh_claim_opens_no_gap(self, monkeypatch) -> None:
        monkeypatch.setattr(spec_hooks, "hook_projection_stale", _stale_for())
        sessions = FakeSessions(ScriptedProvider(answer()))

        _answer(_turns(sessions))

        assert sessions.count("reset") == 0 and sessions.count("get_or_create") == 1
        assert "open_replay_gap" not in sessions.names()
