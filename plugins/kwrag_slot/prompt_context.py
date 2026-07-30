"""Explicit, ephemeral Hermes prompt consumption of verified slot evidence.

The caller decides whether and when retrieval runs.  This module does not
select a backend, derive a query, register a model tool, or run automatically.
It only carries already-verified raw evidence through Hermes' existing
API-only user-message path while preserving the clean user message in history.
The serialized result-and-score representation is private and provisional;
it is not a public prompt contract or an invocation-policy decision.
"""

from __future__ import annotations

import hashlib
from typing import Any

from plugins.kwrag_slot.consumer import (
    HermesSlotRetrievalError,
    HermesSlotRetrievalResult,
)
from plugins.kwrag_slot.manifest import canonical_json_bytes


def _prompt_context(result: HermesSlotRetrievalResult) -> str:
    payload = {
        "schema_version": "hermes-kwrag-evidence-context-v1",
        "index_manifest": result.result_receipt["index_manifest"],
        "operation_receipt_digest": result.result_receipt["operation_receipt_digest"],
        "result_digest": result.result_receipt["result_digest"],
        "result_receipt_digest": result.result_receipt_digest,
        "results": list(result.results),
    }
    raw = canonical_json_bytes(payload).decode("utf-8")
    return "<kwrag_slot_evidence>\n" + raw + "\n</kwrag_slot_evidence>"


def run_conversation_with_approved_retrieval(
    agent: Any,
    user_message: str,
    result: HermesSlotRetrievalResult,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one Hermes turn with caller-approved evidence, default-off elsewhere."""

    if not isinstance(user_message, str) or not user_message:
        raise HermesSlotRetrievalError("user message is invalid")
    if "persist_user_message" in kwargs or "ephemeral_user_context" in kwargs:
        raise HermesSlotRetrievalError("user-message projection is owned by the retrieval seam")
    if getattr(agent, "api_mode", None) == "codex_app_server":
        raise HermesSlotRetrievalError(
            "persistent codex app-server sessions cannot consume retrieval evidence"
        )
    session_id = str(getattr(agent, "session_id", "") or "")
    if not session_id:
        raise HermesSlotRetrievalError("Hermes session identity is unavailable")
    context = _prompt_context(result)
    session_binding = {
        "consumer_family": "hermes",
        "session_id": session_id,
    }
    session_digest = "sha256:" + hashlib.sha256(canonical_json_bytes(session_binding)).hexdigest()
    context_digest = "sha256:" + hashlib.sha256(context.encode("utf-8")).hexdigest()
    result.record_prompt_consumption(
        session_binding_digest=session_digest,
        prompt_context_digest=context_digest,
    )
    outcome = agent.run_conversation(
        user_message,
        ephemeral_user_context=context,
        **kwargs,
    )
    if not isinstance(outcome, dict):
        raise HermesSlotRetrievalError("Hermes conversation result is invalid")
    return outcome
