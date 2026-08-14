"""Explicit model tools for the slot-local KWRAG index.

These tools are deliberately small product adapters.  They do not expose a
shell, an ops command, a generation pin, or raw corpus content to the model.
The KWRAG package owns the actual source walk and index publication; Hermes
only supplies the caller's optional scope and returns content-free status.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from plugins.kwrag_slot.terminal import (
    HermesTerminalRetrievalError,
    build_index,
    index_status,
)
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
    "index",
    "run",
    "해",
    "만들",
    "갱신",
    "생성",
)
_INDEX_STATUS_WORDS = (
    "status",
    "상태",
    "available",
    "있어",
    "몇 개",
    "how many",
)
_SOURCE_HINTS = (
    ("groupware", "groupware"),
    ("그룹웨어", "groupware"),
    ("kakao", "kakao"),
    ("카카오워크", "kakao"),
    ("카카오", "kakao"),
    ("whatsapp", "whatsapp"),
    ("왓츠앱", "whatsapp"),
    ("files", "files"),
    ("file", "files"),
    ("파일", "files"),
)


def explicit_index_build_args(user_message: Any) -> dict[str, Any] | None:
    """Return build arguments only for a clear user indexing instruction.

    This is intentionally a small lexical bridge for the product action that
    the user explicitly requested.  It must not infer retrieval for ordinary
    questions or decide whether a turn should use RAG.  A source word is an
    optional hard scope; otherwise the mounted source set is used by KWRAG.
    """

    if not isinstance(user_message, str):
        return None
    text = re.sub(r"\s+", " ", user_message.strip().lower())
    if not text or not any(marker in text for marker in _INDEX_REQUEST_MARKERS):
        return None
    if any(status_word in text for status_word in _INDEX_STATUS_WORDS):
        return None
    if not any(action in text for action in _INDEX_REQUEST_ACTIONS):
        return None

    source = None
    for candidate, source_name in _SOURCE_HINTS:
        matched = (
            candidate in text
            if not candidate.isascii()
            else bool(
                re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", text)
            )
        )
        if matched:
            source = source_name
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
    """Execute a clear indexing instruction through the product tool handler.

    The synthetic assistant/tool messages make the action observable in the
    same conversation turn while preserving the existing provider loop.  No
    shell or generic execute-code fallback is introduced.
    """

    args = explicit_index_build_args(user_message)
    if args is None:
        return False
    call_id = f"kwrag-index-{uuid.uuid4().hex}"
    if getattr(agent, "tool_progress_callback", None):
        try:
            agent.tool_progress_callback(
                "tool.started", "kwrag_index_build", "building the mounted RAG index", args
            )
        except Exception:
            pass
    try:
        result = agent._invoke_tool(
            "kwrag_index_build", args, effective_task_id, call_id, messages
        )
    except Exception as exc:  # pragma: no cover - final product boundary
        result = tool_error(f"KWRAG index build failed: {type(exc).__name__}")
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
    if getattr(agent, "tool_progress_callback", None):
        try:
            agent.tool_progress_callback(
                "tool.completed", "kwrag_index_build", None, None,
                duration=0.0, is_error=False, result=result,
            )
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
    except HermesTerminalRetrievalError as exc:
        return tool_error(str(exc))
    except Exception as exc:  # pragma: no cover - final product boundary
        return tool_error(f"KWRAG index build failed: {type(exc).__name__}")


def _handle_index_status(_args: dict[str, Any], **_kwargs: Any) -> str:
    try:
        return tool_result(index_status())
    except HermesTerminalRetrievalError as exc:
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
