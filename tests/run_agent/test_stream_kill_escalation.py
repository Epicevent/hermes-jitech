"""Kill-escalation tests for ``interruptible_streaming_api_call``.

A stale-stream kill only *shuts down sockets it can find*.  When the abort
primitive cannot reach the worker's sockets (oc17: GeminiNativeClient kept
its httpx client at ``_http`` where ``_iter_pool_sockets`` didn't look), the
worker stayed blocked while the poll loop repeated ineffective kills every
stale-timeout — a 19.5-minute hang ended only by a user interrupt.

Escalation pins the missing guarantee: the watchdog must not depend on the
worker cooperating.  After each stale-kill the poll loop waits a grace
window (``HERMES_STREAM_KILL_GRACE_SECONDS``); if the worker is still
pinned to the same connection with no progress, the turn is failed from
the poll loop itself — exactly like the interrupt path — and the abandoned
daemon worker self-cleans via the ``stream_abandoned`` flag if it ever
wakes.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest

from agent.request_dispatch import (
    RequestDispatchHandoff,
    snapshot_allowed_provider_routes,
)


def _stream_test_client():
    from openai import OpenAI

    client = OpenAI(api_key="fixture-key", base_url="https://example.com/v1")
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=MagicMock())
    )
    return client


# ── Helpers (mirrors test_partial_stream_finish_reason.py) ────────────────

def _make_stream_chunk(content=None, finish_reason=None):
    delta = SimpleNamespace(
        content=content, tool_calls=None,
        reasoning_content=None, reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model=None, usage=None)


def _make_agent():
    from run_agent import AIAgent
    agent = AIAgent(
        api_key="test-key",
        base_url="https://example.com/v1",
        model="test/model",
        provider="openrouter",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.api_mode = "chat_completions"
    agent._interrupt_requested = False
    return agent


def _pin_timeouts(monkeypatch):
    """Make provider-config lookups deterministic (env vars win)."""
    monkeypatch.setattr(
        "agent.chat_completion_helpers.get_provider_stale_timeout",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "agent.chat_completion_helpers.get_provider_request_timeout",
        lambda *a, **k: None,
    )


class _BlockingStream:
    """Blocks in ``__next__`` until released.

    Simulates a worker pinned in a socket read.  When released it raises
    ``exc`` if given (an *effective* kill surfacing as a network error),
    otherwise ends the stream cleanly.
    """

    def __init__(self, release: threading.Event, exc: BaseException | None = None):
        self._release = release
        self._exc = exc

    def __iter__(self):
        return self

    def __next__(self):
        released = self._release.wait(timeout=30)
        if released and self._exc is not None:
            raise self._exc
        raise StopIteration


# ── Tests ──────────────────────────────────────────────────────────────────

class TestKillEscalation:

    @patch("run_agent.AIAgent._replace_primary_openai_client", return_value=True)
    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_ineffective_kill_escalates_and_fails_turn(
        self, mock_create, mock_close, mock_abort, _mock_replace, monkeypatch
    ):
        """The oc17 defect scenario: the abort is a no-op (patched to do
        nothing), the worker never unwinds.  Pre-fix this hung forever;
        with escalation the turn must fail within stale + grace, not 19.5
        minutes."""
        release = threading.Event()
        mock_client = _stream_test_client()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: _BlockingStream(release)
        )
        mock_create.return_value = mock_client

        # This test exercises watchdog/dispatch arbitration, not provider-leaf
        # capability.  Production OpenAI leaves intentionally fail closed for
        # retrieval evidence until an atomic serialized-request adapter exists.
        def require_synthetic_atomic_test_boundary(client):
            if client is mock_client:
                return "tests.fixture.AtomicSerializedRequestAdapter"
            raise AssertionError("unexpected provider client in watchdog fixture")

        monkeypatch.setattr(
            "agent.chat_completion_helpers.require_authoritative_leaf_adapter",
            require_synthetic_atomic_test_boundary,
        )

        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.5")
        monkeypatch.setenv("HERMES_STREAM_KILL_GRACE_SECONDS", "0.4")
        _pin_timeouts(monkeypatch)

        agent = _make_agent()
        statuses = []
        monkeypatch.setattr(agent, "_buffer_status", lambda m: statuses.append(m))
        outcome_attempts = []

        def fail_outcome(status, digest, error_category):
            outcome_attempts.append((status, digest, error_category))
            raise OSError("outcome fsync failed")

        handoff = RequestDispatchHandoff(
            lambda _binding, _kwargs: None,
            interrupted=lambda: bool(agent._interrupt_requested),
            interrupted_message="stale stream abandoned before dispatch",
            max_attempts=1,
            callback_accepts_attempt_binding=True,
            outcome_callback=fail_outcome,
            configured_provider=str(agent.provider or "unknown"),
            configured_model=agent.model,
            allowed_provider_routes=snapshot_allowed_provider_routes(agent),
        )
        handoff.bind_provider_call_identity(
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
        )
        abandon_calls = []
        original_abandon = handoff.abandon

        def observed_abandon(**kwargs) -> bool:
            result = original_abandon(**kwargs)
            abandon_calls.append(result)
            return result

        handoff.abandon = observed_abandon  # type: ignore[method-assign]

        start = time.time()
        try:
            with pytest.raises(TimeoutError, match="stream abandoned"):
                agent._interruptible_streaming_api_call(
                    {},
                    on_request_dispatch=handoff,
                )
        finally:
            release.set()  # let the abandoned daemon worker exit
        elapsed = time.time() - start

        assert elapsed < 10, (
            f"escalation must bound the hang to ~stale+grace, took {elapsed:.1f}s"
        )
        # Stale kill + escalation last-ditch abort — both reached the
        # (ineffective) abort primitive.
        assert mock_abort.call_count >= 2
        assert any("abandon" in s.lower() for s in statuses), (
            "user-visible status must say the stalled stream was abandoned"
        )
        assert handoff.state == "dispatch_owned"
        assert handoff.sdk_entry_intent_committed is True
        assert handoff.future_attempts_closed is True
        assert handoff.terminal_outcome_status == "unknown"
        assert isinstance(handoff.outcome_persistence_error, OSError)
        assert len(outcome_attempts) == 1
        assert outcome_attempts[0][0] == "unknown"
        assert outcome_attempts[0][2] == "WatchdogTimeout"
        assert abandon_calls == [False], (
            "terminal escalation must close future attempts through the shared handoff"
        )

        # Zombie safety: once the worker wakes it must self-clean — the
        # owner-thread close of the request client eventually runs.
        deadline = time.time() + 2.0
        while time.time() < deadline and mock_close.call_count == 0:
            time.sleep(0.05)
        assert mock_close.call_count >= 1, (
            "abandoned worker must release its request client when it wakes"
        )

    @patch("run_agent.AIAgent._replace_primary_openai_client", return_value=True)
    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_effective_kill_recovers_via_retry_without_escalation(
        self, mock_create, _mock_close, mock_abort, _mock_replace, monkeypatch
    ):
        """When the kill actually unblocks the worker (the normal case,
        and the gemini case after the ``_iter_pool_sockets`` fix), the
        inner retry must recover on a fresh connection and escalation
        must stay silent."""
        release = threading.Event()
        # Effective kill: the abort unblocks the pinned read, which then
        # surfaces as a ReadTimeout the retry loop classifies as transient.
        mock_abort.side_effect = lambda client, **kw: release.set()

        streams = [
            _BlockingStream(release, exc=httpx.ReadTimeout("simulated kill")),
            iter([_make_stream_chunk(content="ok", finish_reason="stop")]),
        ]
        mock_client = _stream_test_client()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: streams.pop(0)
        )
        mock_create.return_value = mock_client

        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.5")
        monkeypatch.setenv("HERMES_STREAM_KILL_GRACE_SECONDS", "5")
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
        _pin_timeouts(monkeypatch)

        agent = _make_agent()
        statuses = []
        monkeypatch.setattr(agent, "_buffer_status", lambda m: statuses.append(m))

        start = time.time()
        response = agent._interruptible_streaming_api_call({})
        elapsed = time.time() - start

        assert response.choices[0].message.content == "ok"
        assert elapsed < 10
        assert not any("abandon" in s.lower() for s in statuses), (
            "escalation must not fire when the worker unwound after the kill"
        )

    @patch("run_agent.AIAgent._replace_primary_openai_client", return_value=True)
    @patch("run_agent.AIAgent._abort_request_openai_client")
    @patch("run_agent.AIAgent._close_request_openai_client")
    @patch("run_agent.AIAgent._create_request_openai_client")
    def test_grace_zero_disables_escalation(
        self, mock_create, _mock_close, mock_abort, _mock_replace, monkeypatch
    ):
        """``HERMES_STREAM_KILL_GRACE_SECONDS=0`` opts out: the poll loop
        keeps the legacy kill-and-wait behavior.  Verified here by the
        worker being released by the *second* kill and the call still
        completing (no TimeoutError, no abandon status)."""
        release = threading.Event()
        kills = {"n": 0}

        def _abort(client, **kw):
            kills["n"] += 1
            if kills["n"] >= 2:  # second stale kill finally lands
                release.set()

        mock_abort.side_effect = _abort

        streams = [
            _BlockingStream(release, exc=httpx.ReadTimeout("simulated kill")),
            iter([_make_stream_chunk(content="ok", finish_reason="stop")]),
        ]
        mock_client = _stream_test_client()
        mock_client.chat.completions.create.side_effect = (
            lambda *a, **kw: streams.pop(0)
        )
        mock_create.return_value = mock_client

        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "0.4")
        monkeypatch.setenv("HERMES_STREAM_KILL_GRACE_SECONDS", "0")
        monkeypatch.setenv("HERMES_STREAM_RETRIES", "2")
        _pin_timeouts(monkeypatch)

        agent = _make_agent()
        statuses = []
        monkeypatch.setattr(agent, "_buffer_status", lambda m: statuses.append(m))

        response = agent._interruptible_streaming_api_call({})

        assert response.choices[0].message.content == "ok"
        assert kills["n"] >= 2
        assert not any("abandon" in s.lower() for s in statuses)
