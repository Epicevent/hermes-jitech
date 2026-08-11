"""The Hermes agent can explicitly ask its slot to build and inspect an index."""

from __future__ import annotations

import json

from plugins.kwrag_slot import tools


def test_index_tools_have_fixed_product_surface() -> None:
    assert tools.INDEX_BUILD_SCHEMA["name"] == "kwrag_index_build"
    assert tools.INDEX_STATUS_SCHEMA["name"] == "kwrag_index_status"
    assert tools.INDEX_BUILD_SCHEMA["parameters"]["additionalProperties"] is False
    assert "query" not in tools.INDEX_BUILD_SCHEMA["parameters"]["properties"]


def test_index_build_tool_delegates_explicit_scope_without_raw_content(monkeypatch) -> None:
    seen = {}

    def build_index(**kwargs):
        seen.update(kwargs)
        return {
            "status": "active",
            "build_id": "build-1",
            "indexed_source_count": 1,
            "skipped_source_count": 0,
        }

    monkeypatch.setattr(tools, "build_index", build_index)
    encoded = tools._handle_index_build(
        {
            "scope": {"sources": ["kakao"]},
            "exclude": [],
            "rebuild": True,
        }
    )
    payload = json.loads(encoded)
    assert payload["status"] == "active"
    assert seen == {
        "scope": {"sources": ["kakao"]},
        "exclude": [],
        "rebuild": True,
    }
    assert "query" not in payload


def test_index_status_tool_returns_content_free_state(monkeypatch) -> None:
    monkeypatch.setattr(
        tools,
        "index_status",
        lambda: {"status": "unavailable", "active_build_id": None},
    )
    payload = json.loads(tools._handle_index_status({}))
    assert payload == {"status": "unavailable", "active_build_id": None}


def test_plugin_registers_agent_tools_and_cli() -> None:
    from plugins.kwrag_slot import register

    class Context:
        def __init__(self):
            self.tools = []
            self.commands = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_cli_command(self, **kwargs):
            self.commands.append(kwargs)

    context = Context()
    register(context)
    assert {item["name"] for item in context.tools} == {
        "kwrag_index_build",
        "kwrag_index_status",
    }
    assert len(context.commands) == 1


def test_dashboard_api_toolset_exposes_only_explicit_index_controls() -> None:
    from toolsets import resolve_toolset

    tools_for_dashboard = set(resolve_toolset("hermes-api-server"))
    assert {"kwrag_index_build", "kwrag_index_status"} <= tools_for_dashboard
