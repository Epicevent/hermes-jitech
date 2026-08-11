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
