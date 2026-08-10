"""Thin caller adapter from one explicit Hermes turn to KWRAG.

KWRAG owns the mounted source, search implementation, and operation receipt.
Hermes owns only request-shape adaptation, result consumption, bounded current-
turn context, and the existing provider handoff.  No ops command, approval,
capsule, generated binding, or caller-supplied generation participates here.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
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


_MAX_QUERY_CHARACTERS = 4_000
_MAX_RESULTS = 10
_MAX_RESULT_CHARACTERS = 20_000
_DEFAULT_KAKAO_PACKAGE_ROOT = Path("/workspace/nas_docs/kw/package")


class KakaoTerminalRetrievalError(HermesSlotRetrievalError):
    """The explicit Kakao terminal request cannot be served."""


def _query(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= _MAX_QUERY_CHARACTERS:
        raise KakaoTerminalRetrievalError("query must be 1-4000 characters")
    if value != value.strip():
        raise KakaoTerminalRetrievalError(
            "query must not have surrounding whitespace"
        )
    return value


def validate_explicit_request(value: Mapping[str, Any]) -> dict[str, str]:
    """Accept only the caller decision that actually belongs to Hermes."""

    if not isinstance(value, Mapping) or set(value) != {"query", "corpus"}:
        raise KakaoTerminalRetrievalError("kwrag request fields are invalid")
    query = _query(value["query"])
    corpus = value["corpus"]
    if corpus != "kakao":
        raise KakaoTerminalRetrievalError("corpus must be kakao")
    return {"query": query, "corpus": corpus}


def _source_package_root(value: Path | None) -> Path:
    root = _DEFAULT_KAKAO_PACKAGE_ROOT if value is None else Path(value)
    if not root.is_absolute() or (os.name == "posix" and "\\" in str(root)) or any(
        part in {".", ".."} for part in root.parts
    ):
        raise KakaoTerminalRetrievalError("Kakao source package path is invalid")
    return root


def _receipt_root() -> Path:
    root = Path(get_hermes_home()) / "kwrag"
    if not root.is_absolute() or root.is_symlink():
        raise KakaoTerminalRetrievalError("Hermes KWRAG receipt root is unavailable")
    try:
        root.mkdir(mode=0o700, parents=False, exist_ok=True)
        if os.name == "posix" and (root.stat().st_mode & 0o777) != 0o700:
            raise KakaoTerminalRetrievalError(
                "Hermes KWRAG receipt root permissions are invalid"
            )
    except OSError as exc:
        raise KakaoTerminalRetrievalError(
            "Hermes KWRAG receipt root is unavailable"
        ) from exc
    if not root.is_dir() or root.is_symlink():
        raise KakaoTerminalRetrievalError("Hermes KWRAG receipt root is unavailable")
    return root


def prepare_approved_retrieval(
    request: Mapping[str, Any],
    *,
    package_root: Path | None = None,
) -> tuple[HermesSlotRetrievalResult, bytes]:
    """Run one explicit product-native search and return verified evidence."""

    validated = validate_explicit_request(request)
    producer_request = {
        "schema_version": "kwrag-slot-search-request-v1",
        "query": validated["query"],
        "request_id": str(uuid.uuid4()),
        "operation_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "attempt": 1,
        "max_results": _MAX_RESULTS,
    }
    try:
        from kwrag.product_runtime import open_kakao_product_runtime
    except ImportError as exc:
        raise KakaoTerminalRetrievalError(
            "product-native KWRAG runtime is unavailable"
        ) from exc

    root = _receipt_root()
    try:
        with open_kakao_product_runtime(
            package_root=_source_package_root(package_root),
            receipt_path=root / "operation-receipts.jsonl",
        ) as runtime:
            identity = runtime.identity
            manifest = load_component_manifest()
            binding = HermesSlotRetrievalBinding.from_mapping(
                {
                    "schema_version": "hermes-kwrag-slot-binding-v1",
                    "enabled": True,
                    "component_digest": manifest["component_wheel"]["sha256"],
                    "runtime_binding_digest": identity.digest,
                    "expected_index_manifest": identity.index_manifest,
                    "expected_pipeline_fingerprint": identity.pipeline_fingerprint,
                    "max_result_characters": _MAX_RESULT_CHARACTERS,
                }
            )
            result = HermesSlotRetrievalConsumer(
                binding,
                runtime,
                FileConsumptionReceiptSink(root / "result-receipts.jsonl"),
            ).search(producer_request)
    except KakaoTerminalRetrievalError:
        raise
    except Exception as exc:
        raise KakaoTerminalRetrievalError(
            "Kakao product retrieval was not verified"
        ) from exc

    context = {
        "schema_version": "hermes-kwrag-current-turn-context-v1",
        "index_manifest": result.result_receipt["index_manifest"],
        "runtime_binding_digest": result.result_receipt["runtime_binding_digest"],
        "operation_receipt_digest": result.result_receipt[
            "operation_receipt_digest"
        ],
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
        "index_manifest",
        "runtime_binding_digest",
        "operation_receipt_digest",
        "result_receipt_digest",
        "result_digest",
        "result_count",
    } or context.get("schema_version") != "hermes-kwrag-current-turn-context-v1":
        raise KakaoTerminalRetrievalError("current-turn context fields are invalid")
    receipt = approved_retrieval.result_receipt
    for field, key in (
        ("index_manifest", "index_manifest"),
        ("runtime_binding_digest", "runtime_binding_digest"),
        ("operation_receipt_digest", "operation_receipt_digest"),
        ("result_receipt_digest", None),
        ("result_digest", "result_digest"),
        ("result_count", "result_count"),
    ):
        expected = (
            approved_retrieval.result_receipt_digest
            if key is None
            else receipt[key]
        )
        if context[field] != expected:
            raise KakaoTerminalRetrievalError(
                "current-turn context is not bound to verified evidence"
            )
    return run_conversation_with_approved_retrieval(
        agent,
        user_message,
        approved_retrieval,
        **conversation_kwargs,
    )
