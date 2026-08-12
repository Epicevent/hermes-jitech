"""Explicit model tools for the slot-local KWRAG index.

These tools are deliberately small product adapters.  They do not expose a
shell, an ops command, a generation pin, or raw corpus content to the model.
The KWRAG package owns the actual source walk and index publication; Hermes
only supplies the caller's optional scope and returns content-free status.
"""

from __future__ import annotations

from typing import Any

from plugins.kwrag_slot.terminal import (
    KakaoTerminalRetrievalError,
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
