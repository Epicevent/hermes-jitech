"""Caller-explicit, in-process KWRAG product integration for Hermes.

The plugin registers two bounded model tools for explicit index build/status
requests plus the legacy CLI/status surface. It registers no shell tool,
prompt hook, lifecycle hook, automatic retrieval policy, or operations-tool
admission contract.
"""

from __future__ import annotations

from plugins.kwrag_slot.cli import kwrag_slot_command, register_cli
from plugins.kwrag_slot.tools import register as register_tools


def register(ctx) -> None:
    register_tools(ctx)
    ctx.register_cli_command(
        name="kwrag-slot",
        help="Inspect or explicitly rebuild the embedded KWRAG product component",
        setup_fn=register_cli,
        handler_fn=kwrag_slot_command,
        description=(
            "Build identity and explicit Workspace indexing for the embedded "
            "KWRAG component. Model tools remain bounded to explicit index "
            "build/status and expose no shell, hook, or runtime admission "
            "policy."
        ),
    )
