"""Regression guard for the dashboard's agent-bound KWRAG bridge."""

from __future__ import annotations

import json
from types import SimpleNamespace


def test_kwrag_search_uses_agent_bound_retrieval_seam(monkeypatch):
    from agent import agent_runtime_helpers

    prepared = SimpleNamespace(
        result_receipt_digest="sha256:" + "a" * 64,
        result_receipt_status="written",
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.terminal.prepare_approved_retrieval",
        lambda request: prepared,
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
