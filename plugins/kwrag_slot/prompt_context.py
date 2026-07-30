"""Explicit, ephemeral Hermes prompt consumption of verified slot evidence.

The caller decides whether and when retrieval runs.  This module does not
select a backend, derive a query, register a model tool, or run automatically.
It only carries already-verified raw evidence through Hermes' existing
API-only user-message path while preserving the clean user message in history.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

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


@dataclass(frozen=True)
class HermesSlotConversationRun:
    """Content-free retrieval disposition plus the normal Hermes outcome."""

    conversation: dict[str, Any]
    retrieval_status: str
    attestation: dict[str, Any] | None


def _run_clean_turn(
    agent: Any,
    user_message: str,
    *,
    retrieval_status: str,
    attestation: dict[str, Any] | None = None,
    **kwargs: Any,
) -> HermesSlotConversationRun:
    outcome = agent.run_conversation(user_message, **kwargs)
    if not isinstance(outcome, dict):
        raise HermesSlotRetrievalError("Hermes conversation result is invalid")
    return HermesSlotConversationRun(outcome, retrieval_status, attestation)


def run_conversation_after_explicit_retrieval(
    agent: Any,
    user_message: str,
    retrieve: Callable[[], HermesSlotRetrievalResult],
    **kwargs: Any,
) -> HermesSlotConversationRun:
    """Run one caller-authorized attempt and fail open to a clean Hermes turn.

    This function does not decide whether retrieval should run, construct a
    query, select a backend, retry, or fall back to another backend. The caller
    supplies one already-authorized attempt. Only built-in timeout and the
    strict KWRAG/Hermes verification boundary are converted to a clean turn;
    unrelated programming failures still propagate.
    """

    if not isinstance(user_message, str) or not user_message:
        raise HermesSlotRetrievalError("user message is invalid")
    if "persist_user_message" in kwargs or "ephemeral_user_context" in kwargs:
        raise HermesSlotRetrievalError("user-message projection is owned by the retrieval seam")
    try:
        result = retrieve()
    except TimeoutError:
        return _run_clean_turn(
            agent,
            user_message,
            retrieval_status="timeout",
            **kwargs,
        )
    except HermesSlotRetrievalError:
        return _run_clean_turn(
            agent,
            user_message,
            retrieval_status="unavailable",
            **kwargs,
        )
    except Exception as exc:
        try:
            from kwrag.slot_consumer import SlotConsumptionError
        except ImportError:
            raise
        if not isinstance(exc, SlotConsumptionError):
            raise
        return _run_clean_turn(
            agent,
            user_message,
            retrieval_status="verification_failed",
            **kwargs,
        )

    if not isinstance(result, HermesSlotRetrievalResult):
        return _run_clean_turn(
            agent,
            user_message,
            retrieval_status="verification_failed",
            **kwargs,
        )
    if result.result_receipt.get("result_status") == "zero_hits":
        return _run_clean_turn(
            agent,
            user_message,
            retrieval_status="zero_hits",
            attestation=result.content_free_attestation(),
            **kwargs,
        )
    if result.result_receipt.get("result_status") != "hits":
        return _run_clean_turn(
            agent,
            user_message,
            retrieval_status="verification_failed",
            **kwargs,
        )

    conversation = run_conversation_with_approved_retrieval(
        agent,
        user_message,
        result,
        **kwargs,
    )
    return HermesSlotConversationRun(
        conversation,
        "consumed",
        result.content_free_attestation(),
    )
