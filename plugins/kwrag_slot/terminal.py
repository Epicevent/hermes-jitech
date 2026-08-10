"""Thin caller adapter from one explicit Hermes turn to product-native RAG.

KWRAG owns the live mounted Kakao source, disposable Workspace index, KURE
query embedding, vector search, BGE reranking, and operation receipt. Hermes
owns only the explicit query/corpus request, verified result consumption, and
the existing provider handoff. No ops command, capsule, frozen generation, or
caller-supplied index identity participates in this path.
"""

from __future__ import annotations

import os
import re
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
from plugins.kwrag_slot.manifest import load_component_manifest
from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval


_MAX_QUERY_CHARACTERS = 4_000
_MAX_RESULTS = 10
_MAX_RESULT_CHARACTERS = 20_000
_DEFAULT_KAKAO_PACKAGE_ROOT = Path("/workspace/nas_docs/kw/package")
_DEFAULT_WORKSPACE_INDEX_ROOT = Path("/workspace/.kwrag/dense")
_SOCKET_ENV = "JITECH_KWRAG_SHARED_GPU_SOCKET"
_SLOT_ENV = "JITECH_KWRAG_SLOT_NAMESPACE"
_SLOT_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


class KakaoTerminalRetrievalError(HermesSlotRetrievalError):
    """The explicit Kakao terminal request cannot be served."""


def _query(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= _MAX_QUERY_CHARACTERS:
        raise KakaoTerminalRetrievalError("query must be 1-4000 characters")
    if value != value.strip():
        raise KakaoTerminalRetrievalError("query must not have surrounding whitespace")
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


def _absolute_path(
    value: Path | None, *, env: str, default: Path | None, label: str
) -> Path:
    raw = value if value is not None else os.environ.get(env)
    path = default if raw is None else Path(raw)
    if (
        path is None
        or not path.is_absolute()
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise KakaoTerminalRetrievalError(f"{label} is unavailable")
    return path


def _slot_namespace(value: str | None) -> str:
    slot = value if value is not None else os.environ.get(_SLOT_ENV)
    if not isinstance(slot, str) or _SLOT_RE.fullmatch(slot) is None:
        raise KakaoTerminalRetrievalError("KWRAG slot namespace is unavailable")
    return slot


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


def _runtime_paths(
    *,
    package_root: Path | None,
    workspace_root: Path | None,
    socket_path: Path | None,
    slot_namespace: str | None,
) -> tuple[Path, Path, Path, str, Path]:
    root = _receipt_root()
    return (
        _absolute_path(
            package_root,
            env="JITECH_KWRAG_KAKAO_PACKAGE_ROOT",
            default=_DEFAULT_KAKAO_PACKAGE_ROOT,
            label="Kakao source package",
        ),
        _absolute_path(
            workspace_root,
            env="JITECH_KWRAG_WORKSPACE_INDEX_ROOT",
            default=_DEFAULT_WORKSPACE_INDEX_ROOT,
            label="KWRAG Workspace index root",
        ),
        _absolute_path(
            socket_path,
            env=_SOCKET_ENV,
            default=None,
            label="KWRAG shared GPU socket",
        ),
        _slot_namespace(slot_namespace),
        root,
    )


def prepare_approved_retrieval(
    request: Mapping[str, Any],
    *,
    package_root: Path | None = None,
    workspace_root: Path | None = None,
    socket_path: Path | None = None,
    slot_namespace: str | None = None,
) -> HermesSlotRetrievalResult:
    """Run one explicit dense/vector/rerank search and verify its result."""

    validated = validate_explicit_request(request)
    producer_request = {
        "schema_version": "kwrag-slot-search-request-v1",
        "query": validated["query"],
        "request_id": str(uuid.uuid4()),
        "operation_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "attempt": 1,
        "max_results": _MAX_RESULTS,
        "corpus": None,
    }
    try:
        from kwrag.product_runtime import open_kakao_product_runtime
    except ImportError as exc:
        raise KakaoTerminalRetrievalError(
            "product-native KWRAG runtime is unavailable"
        ) from exc

    package, workspace, socket, slot, root = _runtime_paths(
        package_root=package_root,
        workspace_root=workspace_root,
        socket_path=socket_path,
        slot_namespace=slot_namespace,
    )
    try:
        with open_kakao_product_runtime(
            package_root=package,
            workspace_root=workspace,
            slot_namespace=slot,
            socket_path=socket,
            receipt_path=root / "operation-receipts.jsonl",
            gpu_receipt_path=root / "gpu-receipts.jsonl",
        ) as runtime:
            identity = runtime.identity
            manifest = load_component_manifest()
            binding = HermesSlotRetrievalBinding.from_mapping({
                "schema_version": "hermes-kwrag-slot-binding-v1",
                "enabled": True,
                "component_digest": manifest["component_wheel"]["sha256"],
                "runtime_binding_digest": identity.digest,
                "expected_index_manifest": identity.index_manifest,
                "expected_pipeline_fingerprint": identity.pipeline_fingerprint,
                "max_result_characters": _MAX_RESULT_CHARACTERS,
            })
            return HermesSlotRetrievalConsumer(
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


def rebuild_kakao_index(
    *,
    package_root: Path | None = None,
    workspace_root: Path | None = None,
    socket_path: Path | None = None,
    slot_namespace: str | None = None,
) -> dict[str, Any]:
    """Explicitly rebuild the slot's disposable Workspace index."""

    try:
        from kwrag.product_runtime import rebuild_kakao_product_index
    except ImportError as exc:
        raise KakaoTerminalRetrievalError(
            "product-native KWRAG runtime is unavailable"
        ) from exc
    package, workspace, socket, slot, root = _runtime_paths(
        package_root=package_root,
        workspace_root=workspace_root,
        socket_path=socket_path,
        slot_namespace=slot_namespace,
    )
    try:
        return rebuild_kakao_product_index(
            package_root=package,
            workspace_root=workspace,
            slot_namespace=slot,
            socket_path=socket,
            gpu_receipt_path=root / "gpu-receipts.jsonl",
        )
    except Exception as exc:
        raise KakaoTerminalRetrievalError("Kakao index rebuild failed") from exc


def dispatch_current_terminal_turn(
    agent: Any,
    user_message: Any,
    *,
    approved_retrieval: HermesSlotRetrievalResult,
    **conversation_kwargs: Any,
) -> Any:
    """Dispatch one verified result through the existing consumption seam."""

    return run_conversation_with_approved_retrieval(
        agent,
        user_message,
        approved_retrieval,
        **conversation_kwargs,
    )
