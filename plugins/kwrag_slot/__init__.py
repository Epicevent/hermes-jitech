"""Caller-explicit, in-process KWRAG product integration for Hermes.

The plugin registers build-status observability only. It intentionally
registers no model tool, prompt hook, lifecycle hook, automatic retrieval
policy, or operations-tool admission contract.
"""

from __future__ import annotations

from plugins.kwrag_slot.cli import kwrag_slot_command, register_cli


def register(ctx) -> None:
    ctx.register_cli_command(
        name="kwrag-slot",
        help="Inspect the embedded KWRAG product component",
        setup_fn=register_cli,
        handler_fn=kwrag_slot_command,
        description=(
            "Build identity for the embedded in-process KWRAG component. "
            "This command exposes no model tool, hook, or runtime admission policy."
        ),
    )
