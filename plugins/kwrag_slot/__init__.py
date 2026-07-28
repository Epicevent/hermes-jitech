"""Default-off, in-process KWRAG product integration for Hermes.

The plugin registers an operator CLI only.  It intentionally registers no
model tool, prompt hook, lifecycle hook, or automatic retrieval policy.
"""

from __future__ import annotations

from plugins.kwrag_slot.cli import kwrag_slot_command, register_cli


def register(ctx) -> None:
    ctx.register_cli_command(
        name="kwrag-slot",
        help="Inspect the embedded default-off KWRAG slot consumer",
        setup_fn=register_cli,
        handler_fn=kwrag_slot_command,
        description=(
            "Content-free status for the embedded, in-process KWRAG component. "
            "This command does not run retrieval or expose a model tool."
        ),
    )
