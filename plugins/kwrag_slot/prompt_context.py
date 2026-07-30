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

from agent.request_dispatch import (
    FinalProviderBindingUnsupported,
    require_retrieval_evidence_dispatch_capability,
)
from plugins.kwrag_slot.consumer import (
    HermesSlotRetrievalError,
    HermesSlotRetrievalResult,
)
from plugins.kwrag_slot.manifest import canonical_json_bytes


def _prompt_context(result: HermesSlotRetrievalResult) -> str:
    verified_results, verified_receipt = result._verified_evidence()
    if verified_receipt.get("result_status") != "hits" or not verified_results:
        raise HermesSlotRetrievalError("retrieval evidence has no verified hits to consume")
    payload = {
        "schema_version": "hermes-kwrag-evidence-context-v1",
        "index_manifest": verified_receipt["index_manifest"],
        "operation_receipt_digest": verified_receipt["operation_receipt_digest"],
        "result_digest": verified_receipt["result_digest"],
        "result_receipt_digest": result.result_receipt_digest,
        "results": list(verified_results),
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
    if (
        "persist_user_message" in kwargs
        or "ephemeral_user_context" in kwargs
        or "ephemeral_user_context_on_request" in kwargs
        or "ephemeral_user_context_on_outcome" in kwargs
    ):
        raise HermesSlotRetrievalError("user-message projection is owned by the retrieval seam")
    if getattr(agent, "api_mode", None) == "codex_app_server":
        raise HermesSlotRetrievalError(
            "persistent codex app-server sessions cannot consume retrieval evidence"
        )
    initial_session_id = str(getattr(agent, "session_id", "") or "")
    if not initial_session_id:
        raise HermesSlotRetrievalError("Hermes session identity is unavailable")
    try:
        require_retrieval_evidence_dispatch_capability(agent)
    except FinalProviderBindingUnsupported as exc:
        raise HermesSlotRetrievalError(
            "Hermes retrieval evidence dispatch is unavailable before projection"
        ) from exc
    context = _prompt_context(result)
    context_digest = "sha256:" + hashlib.sha256(context.encode("utf-8")).hexdigest()

    def _commit_consumption_at_first_request(
        provider_attempt_binding: dict[str, Any],
    ) -> None:
        session_id = str(getattr(agent, "session_id", "") or "")
        if not session_id:
            raise HermesSlotRetrievalError("Hermes session identity is unavailable at dispatch")
        session_binding = {
            "consumer_family": "hermes",
            "session_id": session_id,
        }
        session_digest = "sha256:" + hashlib.sha256(
            canonical_json_bytes(session_binding)
        ).hexdigest()
        result.record_prompt_consumption(
            session_binding_digest=session_digest,
            prompt_context_digest=context_digest,
            provider_attempt_binding=provider_attempt_binding,
        )

    def _record_first_request_outcome(
        transport_outcome_status: str,
        provider_attempt_binding_digest: str,
        error_category: str | None,
    ) -> None:
        result.record_provider_attempt_outcome(
            provider_attempt_binding_digest=provider_attempt_binding_digest,
            transport_outcome_status=transport_outcome_status,
            error_category=error_category,
        )

    outcome = agent.run_conversation(
        user_message,
        ephemeral_user_context=context,
        ephemeral_user_context_on_request=_commit_consumption_at_first_request,
        ephemeral_user_context_on_outcome=_record_first_request_outcome,
        **kwargs,
    )
    if result.consumption_receipt_status != "written":
        raise HermesSlotRetrievalError(
            "Hermes conversation completed before retrieval evidence dispatch"
        )
    if not isinstance(outcome, dict):
        raise HermesSlotRetrievalError("Hermes conversation result is invalid")
    if (
        outcome.get("completed") is True
        and result.provider_attempt_outcome_status != "written"
    ):
        raise HermesSlotRetrievalError(
            "Hermes conversation completed without a durable provider attempt outcome"
        )
    return outcome
