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
from typing import Any, Mapping, Sequence

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
_MAX_ROUTED_ROOMS = 5
_MAX_RESULT_CHARACTERS = 20_000
_DEFAULT_SOURCE_ROOT = Path("/workspace/nas_docs")
_DEFAULT_WORKSPACE_INDEX_ROOT = Path("/workspace/.kwrag")
_SOCKET_ENV = "JITECH_KWRAG_SHARED_GPU_SOCKET"
_SLOT_ENV = "JITECH_KWRAG_SLOT_NAMESPACE"
_SLOT_RE = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")


class KakaoTerminalRetrievalError(HermesSlotRetrievalError):
    """An explicit slot-local RAG request cannot be served."""


def _query(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= _MAX_QUERY_CHARACTERS:
        raise KakaoTerminalRetrievalError("query must be 1-4000 characters")
    if value != value.strip():
        raise KakaoTerminalRetrievalError("query must not have surrounding whitespace")
    return value


def _room_selectors(value: Any) -> list[dict[str, str]] | None:
    if value is None:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not 1 <= len(value) <= 64
    ):
        raise KakaoTerminalRetrievalError("room scope is invalid")
    normalized: list[dict[str, str]] = []
    for room in value:
        if isinstance(room, Mapping):
            if set(room) != {"source", "roomId"}:
                raise KakaoTerminalRetrievalError("room scope is invalid")
            source = room.get("source")
            if not isinstance(source, str) or not source.strip():
                raise KakaoTerminalRetrievalError("room scope is invalid")
            room_id = room.get("roomId")
            normalized_source = source.strip().lower()
        else:
            room_id = room
            normalized_source = "kakao"
        if not isinstance(room_id, str) or room_id != room_id.strip() or not room_id:
            raise KakaoTerminalRetrievalError("room scope is invalid")
        normalized.append({"source": normalized_source, "roomId": room_id})
    identities = [(room["source"], room["roomId"]) for room in normalized]
    if len(set(identities)) != len(identities):
        raise KakaoTerminalRetrievalError("room scope contains duplicates")
    return normalized


def validate_explicit_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate optional user scope without making room selection mandatory.

    ``corpus`` is retained as a compatibility alias for older callers.  It is
    a source selector, not a room id; new callers should use ``scope``.
    """

    if not isinstance(value, Mapping):
        raise KakaoTerminalRetrievalError("kwrag request fields are invalid")
    if set(value) - {"query", "corpus", "scope", "sources", "rooms"}:
        raise KakaoTerminalRetrievalError("kwrag request fields are invalid")
    if "query" not in value:
        raise KakaoTerminalRetrievalError("query is required")
    query = _query(value["query"])
    scope_supplied = "scope" in value
    scope = value.get("scope")
    if scope_supplied and not isinstance(scope, Mapping):
        raise KakaoTerminalRetrievalError("retrieval scope is invalid")
    if scope is not None and set(scope) - {"sources", "rooms"}:
        raise KakaoTerminalRetrievalError("retrieval scope fields are invalid")
    if scope is not None and any(
        field in scope and scope[field] is None for field in ("sources", "rooms")
    ):
        raise KakaoTerminalRetrievalError("retrieval scope field is invalid")
    source_values = (
        scope.get("sources") if scope is not None else value.get("sources")
    )
    room_values = scope.get("rooms") if scope is not None else value.get("rooms")
    rooms = _room_selectors(room_values)
    room_sources = (
        list(dict.fromkeys(room["source"] for room in rooms))
        if rooms is not None
        else None
    )
    legacy_corpus = value.get("corpus")
    if legacy_corpus is not None:
        if (
            not isinstance(legacy_corpus, str)
            or not legacy_corpus.strip()
            or len(legacy_corpus.strip()) > 64
        ):
            raise KakaoTerminalRetrievalError("corpus source is invalid")
        legacy_corpus = legacy_corpus.strip().lower()
        if source_values is not None and source_values != [legacy_corpus]:
            raise KakaoTerminalRetrievalError("source scope conflicts with corpus")
        source_values = [legacy_corpus]
    if source_values is None:
        # Omitted source means the runtime-visible prepared source set.  The
        # current compatibility alias below still narrows legacy `corpus=kakao`.
        sources = None
    elif (
        not isinstance(source_values, Sequence)
        or isinstance(source_values, (str, bytes))
        or not 1 <= len(source_values) <= 16
        or any(not isinstance(source, str) or not source.strip() for source in source_values)
        or len(set(source_values)) != len(source_values)
    ):
        raise KakaoTerminalRetrievalError("source scope is invalid")
    else:
        sources = [source.strip().lower() for source in source_values]
    if sources is not None and not sources:
        raise KakaoTerminalRetrievalError("source scope is empty")
    if room_sources:
        if sources is None:
            sources = room_sources
        elif any(source not in sources for source in room_sources):
            raise KakaoTerminalRetrievalError("room source conflicts with source scope")
    if scope is not None and sources is None and rooms is None:
        raise KakaoTerminalRetrievalError("retrieval scope is empty")
    return {"query": query, "sources": sources, "rooms": rooms}


def _available_rooms(runtime: Any) -> list[str]:
    candidates = [getattr(runtime, "available_rooms", None)]
    scope = getattr(getattr(runtime, "application", None), "scope", None)
    if scope is not None:
        candidates.append(getattr(scope, "available_rooms", None))
    # ProductRuntime keeps the mounted scope private while the compatibility
    # API is still settling. This read-only fallback does not select a source
    # or bypass the mount; it only exposes the rooms the opened runtime already
    # validated.
    private_runtime = getattr(runtime, "_runtime", None)
    private_scope = getattr(private_runtime, "scope", None)
    if private_scope is not None:
        candidates.append(getattr(private_scope, "available_rooms", None))
    for candidate in candidates:
        if candidate is None:
            continue
        rooms = list(candidate)
        if rooms and all(isinstance(room, str) and room for room in rooms):
            return sorted(set(rooms))
    raise KakaoTerminalRetrievalError("mounted Kakao room catalog is unavailable")


def _route_rooms(
    query: str,
    requested_rooms: list[str] | None,
    runtime: Any,
) -> tuple[list[str], str]:
    # The product runtime exposes one encoded catalog for every mounted
    # source. Kakao room ids are the only unqualified entries; other sources
    # use ``source/room`` and must never be relabelled as Kakao selectors.
    available = [room for room in _available_rooms(runtime) if "/" not in room]
    if not available:
        raise KakaoTerminalRetrievalError("mounted Kakao room catalog is unavailable")
    available_set = set(available)
    if requested_rooms is not None:
        if set(requested_rooms) - available_set:
            raise KakaoTerminalRetrievalError("requested Kakao room is outside the mount")
        return requested_rooms, "explicit_room_scope"

    mentioned = [
        room
        for room in available
        if re.search(rf"(?<![A-Za-z0-9_-]){re.escape(room)}(?![A-Za-z0-9_-])", query, re.I)
    ]
    if len(mentioned) > 1:
        raise KakaoTerminalRetrievalError(
            "Kakao room mention is ambiguous; choose one room"
        )
    if mentioned:
        return mentioned, "internal_room_id_hint"

    # A runtime may provide a measured atlas router.  The adapter treats it as
    # an optional capability and still verifies its result against the live
    # mounted room catalog.
    route = getattr(runtime, "route_rooms", None)
    if callable(route):
        routed = list(route(query, limit=3))
        if routed and len(routed) <= _MAX_ROUTED_ROOMS and all(
            isinstance(room, str) and room in available_set for room in routed
        ):
            return list(dict.fromkeys(routed)), "internal_room_router"

    # Current 0.5 runtime has no room-atlas API yet.  Searching the complete
    # mounted room set is the safe compatibility behavior until that API is
    # shipped; it never guesses a single room.
    return available, "all_mounted_rooms"


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
            env="JITECH_KWRAG_SOURCE_ROOT",
            default=_DEFAULT_SOURCE_ROOT,
            label="slot source mount",
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
    producer_request: dict[str, Any] = {
        "schema_version": "kwrag-product-cli-request-v1",
        "operation": "search",
        "query": validated["query"],
        "request_id": str(uuid.uuid4()),
        "operation_id": str(uuid.uuid4()),
        "run_id": str(uuid.uuid4()),
        "attempt": 1,
        "max_results": _MAX_RESULTS,
    }
    try:
        from kwrag import product_runtime
    except ImportError as exc:
        raise KakaoTerminalRetrievalError(
            "product-native KWRAG runtime is unavailable"
        ) from exc
    open_runtime = getattr(product_runtime, "open_product_runtime", None)
    if not callable(open_runtime):
        raise KakaoTerminalRetrievalError(
            "product-native KWRAG search API is unavailable"
        )
    status_reader = getattr(product_runtime, "index_status", None)
    # The production helper owns its default mount.  Tests and embedding
    # callers may provide an explicit runtime root, in which case the
    # context-free status function can inspect `/workspace/nas_docs` before
    # the supplied root is opened. A status failure is therefore advisory
    # until the explicitly bound runtime is opened; a reported unbuilt or
    # invalid state remains a hard failure.
    explicit_runtime_paths = any(
        value is not None
        for value in (package_root, workspace_root, socket_path, slot_namespace)
    )
    if callable(status_reader):
        status_scope: dict[str, Any] | None = None
        if validated["sources"] is not None or validated["rooms"] is not None:
            status_scope = {}
            if validated["sources"] is not None:
                status_scope["sources"] = list(validated["sources"])
            if validated["rooms"] is not None:
                status_scope["rooms"] = [
                    {"source": room["source"], "room_id": room["roomId"]}
                    for room in validated["rooms"]
                ]
        try:
            status = (
                status_reader()
                if status_scope is None
                else status_reader(scope=status_scope)
            )
        except Exception as exc:
            if not explicit_runtime_paths:
                raise KakaoTerminalRetrievalError("rag_backend_unavailable") from exc
            status = None
        if isinstance(status, Mapping):
            status_name = status.get("status")
            if status_name == "unbuilt":
                raise KakaoTerminalRetrievalError("index_required")
            if status_name in {"unavailable", "invalid"}:
                raise KakaoTerminalRetrievalError("rag_backend_unavailable")

    package, workspace, socket, slot, root = _runtime_paths(
        package_root=package_root,
        workspace_root=workspace_root,
        socket_path=socket_path,
        slot_namespace=slot_namespace,
    )
    try:
        runtime_receipt_root = workspace / "receipts"
        runtime_kwargs = {
            "workspace_root": workspace,
            "slot_namespace": slot,
            "socket_path": socket,
            "receipt_path": runtime_receipt_root / "operation-receipts.jsonl",
            "gpu_receipt_path": runtime_receipt_root / "gpu-receipts.jsonl",
        }
        # ProductRuntime owns source discovery below the mounted product root.
        # Passing the legacy Kakao package leaf here would hide every other
        # mounted adapter (for example Groupware) from federated search.
        runtime_kwargs["source_root"] = package
        with open_runtime(**runtime_kwargs) as runtime:
            requested_sources = validated["sources"]
            requested_rooms = validated["rooms"]
            if requested_sources is not None:
                producer_request["scope"] = {"sources": list(requested_sources)}
            if requested_rooms is not None:
                # Validate the Kakao subset against its mounted room catalog,
                # while preserving every source-qualified selector exactly.
                kakao_rooms = [
                    room["roomId"]
                    for room in requested_rooms
                    if room["source"] == "kakao"
                ]
                if kakao_rooms:
                    _route_rooms(validated["query"], kakao_rooms, runtime)
                producer_request["scope"]["rooms"] = [
                    {"source": room["source"], "room_id": room["roomId"]}
                    for room in requested_rooms
                ]
                route_strategy = "explicit_room_scope"
            elif requested_sources == ["kakao"]:
                rooms, route_strategy = _route_rooms(validated["query"], None, runtime)
                producer_request["scope"]["rooms"] = [
                    {"source": "kakao", "room_id": room} for room in rooms
                ]
            elif requested_sources is None:
                # No caller scope means federated search across the exact
                # source adapters opened by the mounted product runtime. Do
                # not send an empty scope: the product contract defines an
                # omitted scope as all mounted sources.
                route_strategy = "all_mounted_sources"
            else:
                route_strategy = "source_wide_scope"
            identity = runtime.identity
            active_index_id = getattr(identity, "active_index_id", None)
            index_manifest = getattr(identity, "index_manifest", active_index_id)
            if not isinstance(index_manifest, str) or not re.fullmatch(
                r"sha256:[0-9a-f]{64}", index_manifest
            ):
                index_manifest = None
            manifest = load_component_manifest()
            binding = HermesSlotRetrievalBinding.from_mapping({
                "schema_version": "hermes-kwrag-slot-binding-v1",
                "enabled": True,
                "component_digest": manifest["component_wheel"]["sha256"],
                "runtime_binding_digest": identity.digest,
                # Bind the release currently opened by the slot runtime.  The
                # browser request carries only query/corpus; no generation or
                # manifest pin is accepted from the caller.
                "current_index_manifest": index_manifest,
                "current_pipeline_fingerprint": identity.pipeline_fingerprint,
                "max_result_characters": _MAX_RESULT_CHARACTERS,
            })
            return HermesSlotRetrievalConsumer(
                binding,
                runtime,
                FileConsumptionReceiptSink(root / "result-receipts.jsonl"),
            ).search(producer_request, routing_strategy=route_strategy)
    except KakaoTerminalRetrievalError:
        raise
    except Exception as exc:
        raise KakaoTerminalRetrievalError(
            "product-native RAG retrieval was not verified"
        ) from exc


def _validate_index_scope(value: Any, label: str) -> Any:
    if value is None:
        return None
    if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (str, bytes)):
        raise KakaoTerminalRetrievalError(f"{label} is invalid")
    return value


def _native_index_scope(value: Any) -> dict[str, list[str]] | None:
    """Map the product-facing scope to the current source API.

    The current server package indexes sources, while room narrowing belongs
    to the subsequent search request.  Do not pass the richer UI mapping to a
    narrower runtime API or silently broaden an explicitly selected source.
    """

    if value is None:
        return None
    if isinstance(value, Mapping):
        sources = value.get("sources")
        rooms = value.get("rooms")
        if sources is None and rooms is not None:
            sources = [room.get("source") for room in rooms]
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            raise KakaoTerminalRetrievalError("index scope sources are invalid")
        normalized = list(sources)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = list(value)
    else:
        raise KakaoTerminalRetrievalError("index scope is invalid")
    if not normalized or any(not isinstance(source, str) for source in normalized):
        raise KakaoTerminalRetrievalError("index scope sources are invalid")
    normalized = [source.strip().lower() for source in normalized]
    if len(set(normalized)) != len(normalized):
        raise KakaoTerminalRetrievalError("index scope sources contain duplicates")
    return {"sources": normalized}


def build_index(
    *,
    scope: Mapping[str, Any] | None = None,
    exclude: Sequence[Mapping[str, Any]] | None = None,
    rebuild: bool = False,
    source_root: Path | None = None,
    workspace_root: Path | None = None,
    socket_path: Path | None = None,
    slot_namespace: str | None = None,
) -> dict[str, Any]:
    """Build the disposable slot index through the product-native API."""

    if not isinstance(rebuild, bool):
        raise KakaoTerminalRetrievalError("rebuild must be boolean")
    scope = _native_index_scope(_validate_index_scope(scope, "index scope"))
    exclude = _validate_index_scope(exclude, "index exclusions")
    if exclude:
        raise KakaoTerminalRetrievalError(
            "index exclusions are not supported by the current product runtime"
        )
    try:
        from kwrag import product_runtime
    except ImportError as exc:
        raise KakaoTerminalRetrievalError(
            "product-native KWRAG runtime is unavailable"
        ) from exc

    builder = getattr(product_runtime, "build_index", None)
    if not callable(builder):
        raise KakaoTerminalRetrievalError(
            "product-native index build API is unavailable"
        )
    try:
        # The product-native server API owns the runtime mount, Workspace,
        # slot identity and GPU socket. Hermes supplies only bounded scope.
        return builder(scope=scope, rebuild=rebuild)
    except Exception as exc:
        raise KakaoTerminalRetrievalError("KWRAG index build failed") from exc


def index_status(
    *,
    source_root: Path | None = None,
    workspace_root: Path | None = None,
    slot_namespace: str | None = None,
) -> dict[str, Any]:
    """Return content-free status for the slot's disposable active index."""

    try:
        from kwrag import product_runtime
    except ImportError:
        product_runtime = None
    status_fn = getattr(product_runtime, "index_status", None) if product_runtime else None
    if not callable(status_fn):
        raise KakaoTerminalRetrievalError(
            "product-native index status API is unavailable"
        )
    try:
        return status_fn()
    except Exception as exc:
        raise KakaoTerminalRetrievalError("KWRAG index status failed") from exc


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
