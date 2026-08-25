"""Bounded model tools for slot-local KWRAG and Kakao evidence.

The KWRAG index tools do not expose a shell, ops command, generation pin, or
unverified search content.  The weekly period tool is the deliberate exception:
it returns every record in a fixed period while keeping package path, identity,
membership scope, SQL, batching, and reconciliation outside model control.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from plugins.kwrag_slot.terminal import (
    KakaoTerminalRetrievalError,
    build_index,
    index_status,
)
from plugins.kwrag_slot.period_records import execute_period_records
from tools.registry import tool_error, tool_result


INDEX_BUILD_SCHEMA = {
    "name": "kwrag_index_build",
    "description": (
        "Build or refresh the disposable RAG index from the source currently "
        "visible to this Hermes slot. Use only when the user explicitly asks "
        "to index or refresh the corpus."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "scope": {
                "type": "object",
                "description": "Optional source and room narrowing.",
                "properties": {
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "rooms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "roomId": {"type": "string"},
                            },
                            "required": ["source", "roomId"],
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
            "exclude": {
                "type": "array",
                "description": "Optional source-relative exclusions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "pattern": {"type": "string"},
                    },
                    "required": ["source", "pattern"],
                    "additionalProperties": False,
                },
            },
            "rebuild": {
                "type": "boolean",
                "description": "Rebuild the disposable index even when one is active.",
            },
        },
        "additionalProperties": False,
    },
}

INDEX_STATUS_SCHEMA = {
    "name": "kwrag_index_status",
    "description": (
        "Report whether this slot has an active disposable RAG index. "
        "Returns only content-free status and counts."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
}

SEARCH_SCHEMA = {
    "name": "kwrag_search",
    "description": (
        "Search the active slot-local RAG index for the user's current question. "
        "Use this only when the user asks for an answer grounded in the indexed "
        "corpus; the Hermes turn will consume the verified result before the provider."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's original question, unchanged.",
            },
            "scope": {
                "type": "object",
                "description": "Optional source/room narrowing selected by the caller.",
                "properties": {
                    "sources": {"type": "array", "items": {"type": "string"}},
                    "rooms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "roomId": {"type": "string"},
                            },
                            "required": ["source", "roomId"],
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

PERIOD_RECORDS_SCHEMA = {
    "name": "jitech_kakaowork_period_records",
    "description": (
        "Enumerate every authorized KakaoWork record in a fixed weekly period. "
        "Call manifest first, read every batch page, then reconcile coverage."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["manifest", "read_batch", "reconcile"],
            },
            "period": {
                "type": "string",
                "enum": ["rolling_7d", "previous_calendar_week"],
                "description": "Required only for manifest.",
            },
            "snapshot_ref": {
                "type": "string",
                "description": "Opaque workflow reference returned by manifest.",
            },
            "batch_id": {
                "type": "string",
                "description": "A manifest batch identifier, required for read_batch.",
            },
            "cursor": {
                "type": "string",
                "description": "Opaque next_cursor from the preceding read_batch page.",
            },
            "coverage": {
                "type": "array",
                "description": "One final-page coverage digest for every manifest batch.",
                "items": {
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "coverage_digest": {"type": "string"},
                    },
                    "required": ["batch_id", "coverage_digest"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["operation"],
        "additionalProperties": False,
    },
}


_INDEX_REQUEST_MARKERS = (
    "index",
    "indexing",
    "reindex",
    "re-index",
    "refresh the index",
    "build the index",
    "인덱스",
    "인덱싱",
    "재인덱스",
    "인덱스를 만들어",
    "인덱스를 갱신",
    "인덱스 갱신",
)
_INDEX_REQUEST_ACTIONS = (
    "build",
    "refresh",
    "rebuild",
    "create",
    "make",
    "update",
    "run",
    "해",
    "만들",
    "갱신",
    "생성",
)
_INDEX_DIRECTIVE_RE = re.compile(
    r"^\s*(?:re-?index|index)\s+(?:this|the|current|mounted|visible|my|our)\b"
)
_INDEX_STATUS_WORDS = (
    "status",
    "상태",
    "available",
    "있어",
    "몇 개",
    "how many",
)


def explicit_index_build_args(user_message: Any) -> dict[str, Any] | None:
    """Return build arguments only for a clear user indexing instruction."""

    if not isinstance(user_message, str):
        return None
    text = re.sub(r"\s+", " ", user_message.strip().lower())
    if not text or not any(marker in text for marker in _INDEX_REQUEST_MARKERS):
        return None
    if any(status_word in text for status_word in _INDEX_STATUS_WORDS):
        return None
    if not (
        any(action in text for action in _INDEX_REQUEST_ACTIONS)
        or _INDEX_DIRECTIVE_RE.search(text)
    ):
        return None
    source = None
    for candidate in ("groupware", "kakao", "whatsapp", "files", "file"):
        if re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", text):
            source = "files" if candidate == "file" else candidate
            break
    args: dict[str, Any] = {"rebuild": True}
    if source:
        args["scope"] = {"sources": [source]}
    return args


def execute_explicit_index_build(
    agent: Any,
    user_message: Any,
    messages: list[dict[str, Any]],
    effective_task_id: str,
) -> bool:
    """Execute a clear indexing instruction through the product tool handler."""

    args = explicit_index_build_args(user_message)
    if args is None:
        return False
    call_id = f"kwrag-index-{uuid.uuid4().hex}"
    progress_callback = getattr(agent, "tool_progress_callback", None)
    start_callback = getattr(agent, "tool_start_callback", None)
    complete_callback = getattr(agent, "tool_complete_callback", None)
    if progress_callback:
        try:
            progress_callback(
                "tool.started", "kwrag_index_build", "building the mounted RAG index", args
            )
        except Exception:
            pass
    if start_callback:
        try:
            start_callback(call_id, "kwrag_index_build", args)
        except Exception:
            pass
    try:
        result = agent._invoke_tool(
            "kwrag_index_build", args, effective_task_id, call_id, messages
        )
    except Exception as exc:  # pragma: no cover - final product boundary
        result = tool_error(f"KWRAG index build failed: {type(exc).__name__}")
    result = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    try:
        result_payload = json.loads(result)
    except (TypeError, ValueError):
        result_payload = None
    is_error = isinstance(result_payload, dict) and bool(result_payload.get("error"))
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "kwrag_index_build",
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "name": "kwrag_index_build",
            "tool_call_id": call_id,
            "content": result,
        }
    )
    if progress_callback:
        try:
            progress_callback(
                "tool.completed", "kwrag_index_build", None, None,
                duration=0.0, is_error=is_error, result=result,
            )
        except Exception:
            pass
    if complete_callback:
        try:
            complete_callback(call_id, "kwrag_index_build", args, result)
        except Exception:
            pass
    return True


def _check_kwrag_available() -> bool:
    """Keep the tools visible; the handler reports a precise unavailable state."""

    try:
        import kwrag.product_runtime  # noqa: F401

        return True
    except Exception:
        return False


def _handle_index_build(args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        result = build_index(
            scope=args.get("scope"),
            exclude=args.get("exclude"),
            rebuild=args.get("rebuild", False),
        )
        return tool_result(result)
    except KakaoTerminalRetrievalError as exc:
        return tool_error(str(exc))
    except Exception as exc:  # pragma: no cover - final product boundary
        return tool_error(f"KWRAG index build failed: {type(exc).__name__}")


def _handle_index_status(_args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        return tool_result(index_status())
    except KakaoTerminalRetrievalError as exc:
        return tool_error(str(exc))
    except Exception as exc:  # pragma: no cover - final product boundary
        return tool_error(f"KWRAG index status failed: {type(exc).__name__}")


def _handle_search_without_agent(_args: dict[str, Any], **_kwargs: Any) -> str:
    """Reject registry-only calls; an active agent owns provider consumption."""

    return tool_error("kwrag_search requires an active Hermes conversation")


def _handle_search_with_agent(agent, args: dict[str, Any], **_kwargs: Any) -> str:
    """Retain verified search on the active agent for provider consumption."""

    if getattr(agent, "_retrieval_evidence_handoff_active", False):
        return tool_error("agent-bound retrieval tools are unavailable during evidence handoff")
    if getattr(agent, "_kwrag_pending_retrieval", None) is not None:
        return tool_error("KWRAG search is already pending")
    request = {"query": args.get("query")}
    if "scope" in args:
        request["scope"] = args["scope"]
    try:
        from plugins.kwrag_slot.terminal import prepare_approved_retrieval

        prepared = prepare_approved_retrieval(request)
    except Exception as exc:
        return tool_error(f"KWRAG search failed: {type(exc).__name__}")
    agent._kwrag_pending_retrieval = prepared
    return tool_result(
        {
            "status": "verified",
            "result_receipt_digest": prepared.result_receipt_digest,
            "result_receipt_status": prepared.result_receipt_status,
        }
    )


def _handle_period_records(args: dict[str, Any], **_kwargs: Any) -> str:
    return tool_result(execute_period_records(args))


def register(ctx) -> None:
    ctx.register_tool(
        name="kwrag_index_build",
        toolset="kwrag",
        schema=INDEX_BUILD_SCHEMA,
        handler=_handle_index_build,
        check_fn=_check_kwrag_available,
        emoji="IDX",
    )
    ctx.register_tool(
        name="kwrag_index_status",
        toolset="kwrag",
        schema=INDEX_STATUS_SCHEMA,
        handler=_handle_index_status,
        check_fn=_check_kwrag_available,
        emoji="STS",
    )
    ctx.register_tool(
        name="kwrag_search",
        toolset="kwrag",
        schema=SEARCH_SCHEMA,
        handler=_handle_search_without_agent,
        check_fn=_check_kwrag_available,
        emoji="RAG",
        agent_handler=_handle_search_with_agent,
    )
    ctx.register_tool(
        name="jitech_kakaowork_period_records",
        toolset="kwrag",
        schema=PERIOD_RECORDS_SCHEMA,
        handler=_handle_period_records,
        check_fn=lambda: True,
        emoji="KW",
    )
