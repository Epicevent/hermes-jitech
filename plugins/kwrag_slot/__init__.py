"""Slot-bounded Kakao evidence and KWRAG integration for Hermes.

The plugin registers fixed index build/status, verified search, and exhaustive
weekly Kakao record operations plus the legacy CLI/status surface. Its only turn
hook handles a clear user-issued index instruction through the generic host hook
surface; it registers no shell tool, automatic retrieval policy, or operations-tool
admission contract.
"""

from __future__ import annotations

from plugins.kwrag_slot.cli import kwrag_slot_command, register_cli
from plugins.kwrag_slot.tools import (
    execute_explicit_index_build,
    register as register_tools,
)


def _before_user_turn(*, agent, user_message, messages, effective_task_id, **_kwargs):
    return execute_explicit_index_build(
        agent, user_message, messages, effective_task_id
    )


def register(ctx) -> None:
    register_tools(ctx)
    ctx.register_hook("pre_user_turn", _before_user_turn)
    ctx.register_cli_command(
        name="kwrag-slot",
        help="Inspect or explicitly rebuild the embedded KWRAG product component",
        setup_fn=register_cli,
        handler_fn=kwrag_slot_command,
        description=(
            "Build identity and explicit Workspace indexing for the embedded "
            "KWRAG component. Model tools include fixed weekly Kakao period "
            "enumeration and remain bounded to the slot's mounted membership; "
            "they expose no shell or runtime admission policy."
        ),
    )
