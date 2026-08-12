"""Regression guard for the dashboard's agent-bound KWRAG bridge."""

from __future__ import annotations

import json
from types import SimpleNamespace


def test_kwrag_search_uses_agent_bound_retrieval_seam(monkeypatch):
    from agent import agent_runtime_helpers
    from plugins.kwrag_slot.tools import _handle_search_with_agent
    from tools.registry import registry

    prepared = SimpleNamespace(
        result_receipt_digest="sha256:" + "a" * 64,
        result_receipt_status="written",
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.terminal.prepare_approved_retrieval",
        lambda request: prepared,
    )
    monkeypatch.setattr(
        registry,
        "get_agent_handler",
        lambda name: _handle_search_with_agent if name == "kwrag_search" else None,
    )

    class FakeAgent:
        _kwrag_pending_retrieval = None
        _memory_manager = None

    result = agent_runtime_helpers.invoke_tool(
        FakeAgent(),
        "kwrag_search",
        {"query": "current corpus"},
        "task-1",
        pre_tool_block_checked=True,
    )

    assert json.loads(result) == {
        "status": "verified",
        "result_receipt_digest": prepared.result_receipt_digest,
        "result_receipt_status": "written",
    }


def test_kwrag_search_is_suppressed_during_evidence_handoff(monkeypatch):
    from agent import agent_runtime_helpers
    from plugins.kwrag_slot.tools import _handle_search_with_agent
    from tools.registry import registry

    def unexpected_prepare(_request):
        raise AssertionError("retrieval must not re-enter during provider handoff")

    monkeypatch.setattr(
        "plugins.kwrag_slot.terminal.prepare_approved_retrieval",
        unexpected_prepare,
    )
    monkeypatch.setattr(
        registry,
        "get_agent_handler",
        lambda name: _handle_search_with_agent if name == "kwrag_search" else None,
    )

    class FakeAgent:
        _kwrag_pending_retrieval = None
        _retrieval_evidence_handoff_active = True
        _memory_manager = None

    result = agent_runtime_helpers.invoke_tool(
        FakeAgent(),
        "kwrag_search",
        {"query": "current corpus"},
        "task-1",
        pre_tool_block_checked=True,
    )

    payload = json.loads(result)
    assert "unavailable during evidence handoff" in payload["error"]
