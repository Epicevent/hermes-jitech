"""Caller-explicit Kakao retrieval join for one Hermes terminal turn.

This module is deliberately a thin adapter.  The fixed KWRAG producer owns
source/index validation and its operation/producer receipts; the Hermes
consumer owns the result/consumption/provider handoff receipts; and the API
caller owns whether this adapter is invoked at all.  There is no automatic
search, query generation, provider selection, or fallback here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from hermes_constants import get_hermes_home
from plugins.kwrag_slot.consumer import (
    FileConsumptionReceiptSink,
    HermesSlotRetrievalBinding,
    HermesSlotRetrievalConsumer,
    HermesSlotRetrievalError,
    HermesSlotRetrievalResult,
)
from plugins.kwrag_slot.manifest import canonical_json_bytes, load_component_manifest
from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}$")
_MAX_QUERY_CHARACTERS = 4_000
_MAX_RESULTS = 10
_MAX_RESULT_CHARACTERS = 20_000
_REQUEST_FIELDS = {
    "schema_version",
    "query",
    "request_id",
    "operation_id",
    "run_id",
    "attempt",
    "max_results",
    "corpus",
    "source_generation",
}


class KakaoTerminalRetrievalError(HermesSlotRetrievalError):
    """The explicit Kakao terminal request cannot be admitted safely."""


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise KakaoTerminalRetrievalError(f"{label} is invalid")
    return value


def _generation(value: Any, label: str = "source generation") -> str:
    if (
        not isinstance(value, str)
        or not _GENERATION.fullmatch(value)
        or value.lower() in {"unknown", "latest", "current"}
    ):
        raise KakaoTerminalRetrievalError(f"{label} is invalid")
    return value


def _query(value: Any) -> str:
    # Validate the caller's bytes before any strip/coercion.  The producer's
    # canonical request validator then applies the same invariant again.
    if not isinstance(value, str) or not 1 <= len(value) <= _MAX_QUERY_CHARACTERS:
        raise KakaoTerminalRetrievalError("query must be 1-4000 characters")
    if value != value.strip():
        raise KakaoTerminalRetrievalError("query must not have surrounding whitespace")
    return value


def validate_explicit_request(value: Mapping[str, Any]) -> dict[str, str]:
    """Validate the API's explicit request without creating a query policy."""

    if not isinstance(value, Mapping) or set(value) != {
        "query",
        "corpus",
        "expected_source_generation",
        "expected_index_manifest",
    }:
        raise KakaoTerminalRetrievalError("kwrag request fields are invalid")
    query = _query(value["query"])
    corpus = value["corpus"]
    if not isinstance(corpus, str) or not corpus or corpus != corpus.strip():
        raise KakaoTerminalRetrievalError("corpus is invalid")
    return {
        "query": query,
        "corpus": corpus,
        "expected_source_generation": _generation(value["expected_source_generation"]),
        "expected_index_manifest": _digest(
            value["expected_index_manifest"], "index manifest"
        ),
    }


def _producer_binding_path() -> Path:
    raw = os.environ.get(
        "JITECH_KWRAG_FIXED_PRODUCER_BINDING",
        "/run/kwrag/fixed-producer-binding.json",
    )
    path = Path(raw)
    if not path.is_absolute() or "\\" in raw or "." in path.parts or ".." in path.parts:
        raise KakaoTerminalRetrievalError("fixed producer binding path is invalid")
    return path


def _receipt_root() -> Path:
    value = os.environ.get("JITECH_KWRAG_RECEIPT_ROOT")
    root = Path(value) if value else Path(get_hermes_home()) / "kwrag-p1-attachment"
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise KakaoTerminalRetrievalError("Hermes KWRAG receipt root is unavailable")
    return root


@dataclass(frozen=True)
class _FixedProducerRuntime:
    request: Mapping[str, Any]
    response: Mapping[str, Any]
    operation_receipt: Mapping[str, Any]

    def search_exchange(self, request: Any) -> Any:
        if dict(request) != dict(self.request):
            raise KakaoTerminalRetrievalError("Kakao request changed after producer verification")
        return SimpleNamespace(
            response=dict(self.response),
            operation_receipt=dict(self.operation_receipt),
        )


def _fixed_producer_exchange(request: Mapping[str, Any]) -> tuple[_FixedProducerRuntime, str]:
    try:
        from kwrag import fixed_producer
    except ImportError as exc:
        raise KakaoTerminalRetrievalError("generation-aware fixed producer is unavailable") from exc

    load_binding = getattr(fixed_producer, "load_fixed_producer_binding", None)
    execute = getattr(fixed_producer, "execute_fixed_producer", None)
    verify = getattr(fixed_producer, "verify_fixed_producer_output", None)
    if not all(callable(item) for item in (load_binding, execute, verify)):
        raise KakaoTerminalRetrievalError("fixed producer ABI is incomplete")
    binding_path = _producer_binding_path()
    try:
        binding = load_binding(binding_path)
    except Exception as exc:
        raise KakaoTerminalRetrievalError("fixed producer binding is unavailable") from exc
    if getattr(binding, "enabled", None) is not True:
        raise KakaoTerminalRetrievalError("caller-explicit Kakao retrieval is disabled")
    request_fields = dict(request)
    expected_source_generation = request_fields["expected_source_generation"]
    expected_index_manifest = request_fields.pop("expected_index_manifest")
    if getattr(binding, "index_manifest_digest", None) != expected_index_manifest:
        raise KakaoTerminalRetrievalError("index manifest drifted before retrieval")
    producer_request = request_fields
    try:
        execution = execute(
            canonical_json_bytes(producer_request), binding_path=binding_path
        )
        output = json.loads(execution.output_bytes.decode("utf-8"))
        verify(
            execution.output_bytes,
            execution.producer_receipt,
            request=producer_request,
            expected_binding_digest=execution.binding_digest,
        )
    except Exception as exc:
        raise KakaoTerminalRetrievalError("Kakao source exchange was not verified") from exc
    consumable = output.get("consumable")
    linkage = output.get("linkage")
    operation_receipt = dict(execution.operation_receipt)
    if not isinstance(consumable, Mapping) or not isinstance(linkage, Mapping):
        raise KakaoTerminalRetrievalError("Kakao producer output is invalid")
    if consumable.get("source_generation") != expected_source_generation:
        raise KakaoTerminalRetrievalError("producer source generation is not bound")
    if consumable.get("index_manifest") != expected_index_manifest:
        raise KakaoTerminalRetrievalError("producer index manifest is not bound")
    receipt_digest = "sha256:" + hashlib.sha256(
        canonical_json_bytes(operation_receipt)
    ).hexdigest()
    if linkage.get("operation_receipt_digest") != receipt_digest:
        raise KakaoTerminalRetrievalError("producer operation receipt is not bound")
    response = {
        "schema_version": "kwrag-slot-search-response-v1",
        "request_id": consumable.get("request_id"),
        "operation_id": consumable.get("operation_id"),
        "run_id": consumable.get("run_id"),
        "attempt": consumable.get("attempt"),
        "authorization_basis": "slot_mounted_storage",
        "source_generation": expected_source_generation,
        "index_manifest": consumable.get("index_manifest"),
        "pipeline_fingerprint": consumable.get("pipeline_fingerprint"),
        "result_digest": linkage.get("result_digest"),
        "result_status": consumable.get("result_status"),
        "operation_receipt": {"status": "written", "digest": receipt_digest},
        "results": consumable.get("results"),
        "duration_ms": operation_receipt.get("duration_ms", 0),
    }
    return _FixedProducerRuntime(producer_request, response, operation_receipt), execution.binding_digest


def prepare_approved_retrieval(
    request: Mapping[str, Any],
) -> tuple[HermesSlotRetrievalResult, bytes]:
    """Run one explicit source-pinned exchange and return verified evidence."""

    validated = validate_explicit_request(request)
    producer_request: dict[str, Any] = {
        "schema_version": "kwrag-slot-search-request-v1",
        "query": validated["query"],
        "request_id": str(uuid.uuid4()),
        "operation_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "attempt": 1,
        "max_results": _MAX_RESULTS,
        "corpus": validated["corpus"],
        "expected_source_generation": validated["expected_source_generation"],
        "expected_index_manifest": validated["expected_index_manifest"],
    }
    runtime, runtime_binding_digest = _fixed_producer_exchange(producer_request)
    manifest = load_component_manifest()
    root = _receipt_root()
    binding = HermesSlotRetrievalBinding.from_mapping(
        {
            "schema_version": "hermes-kwrag-slot-binding-v2",
            "enabled": True,
            "component_digest": manifest["component_wheel"]["sha256"],
            "runtime_binding_digest": runtime_binding_digest,
            "expected_index_manifest": validated["expected_index_manifest"],
            "expected_pipeline_fingerprint": runtime.response["pipeline_fingerprint"],
            "expected_source_generation": validated["expected_source_generation"],
            "max_result_characters": _MAX_RESULT_CHARACTERS,
        }
    )
    result = HermesSlotRetrievalConsumer(
        binding,
        runtime,
        FileConsumptionReceiptSink(root / "result-receipts.jsonl"),
    ).search(runtime.request)
    context = {
        "schema_version": "hermes-kwrag-current-turn-context-v1",
        "source_generation": validated["expected_source_generation"],
        "index_manifest": validated["expected_index_manifest"],
        "operation_receipt_digest": result.result_receipt["operation_receipt_digest"],
        "result_receipt_digest": result.result_receipt_digest,
        "result_digest": result.result_receipt["result_digest"],
        "result_count": result.result_receipt["result_count"],
    }
    context_bytes = canonical_json_bytes(context)
    if len(context_bytes) > 4096:
        raise KakaoTerminalRetrievalError("current-turn context is not bounded")
    return result, context_bytes


def dispatch_current_terminal_turn(
    agent: Any,
    user_message: Any,
    *,
    kwrag_current_turn_context: bytes,
    approved_retrieval: HermesSlotRetrievalResult,
    **conversation_kwargs: Any,
) -> Any:
    """Dispatch exactly one current turn through the existing evidence seam."""

    try:
        context = json.loads(kwrag_current_turn_context.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise KakaoTerminalRetrievalError("current-turn context is invalid") from exc
    if set(context) != {
        "schema_version",
        "source_generation",
        "index_manifest",
        "operation_receipt_digest",
        "result_receipt_digest",
        "result_digest",
        "result_count",
    } or context.get("schema_version") != "hermes-kwrag-current-turn-context-v1":
        raise KakaoTerminalRetrievalError("current-turn context fields are invalid")
    receipt = approved_retrieval.result_receipt
    for field, key in (
        ("source_generation", "source_generation"),
        ("index_manifest", "index_manifest"),
        ("operation_receipt_digest", "operation_receipt_digest"),
        ("result_receipt_digest", None),
        ("result_digest", "result_digest"),
        ("result_count", "result_count"),
    ):
        expected = approved_retrieval.result_receipt_digest if key is None else receipt[key]
        if context[field] != expected:
            raise KakaoTerminalRetrievalError("current-turn context is not bound to verified evidence")
    return run_conversation_with_approved_retrieval(
        agent,
        user_message,
        approved_retrieval,
        **conversation_kwargs,
    )
