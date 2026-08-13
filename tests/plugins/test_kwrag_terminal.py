"""Tests for the thin caller-explicit Kakao -> Hermes product seam."""

from __future__ import annotations

from contextlib import closing
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import socketserver
import sys
import types
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from plugins.kwrag_slot import terminal
from plugins.kwrag_slot.manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
WHEEL = ROOT / "vendor" / "kwrag" / "kwrag_product_service-0.5.0-py3-none-any.whl"


def _request() -> dict[str, object]:
    return {"query": "who owns the slot?", "corpus": "kakao"}


def test_explicit_request_owns_only_query_and_product_corpus() -> None:
    assert terminal.validate_explicit_request(_request()) == {
        "query": _request()["query"],
        "sources": ["kakao"],
        "rooms": None,
    }
    invalid = (
        {"query": "question", "corpus": ""},
        {**_request(), "expected_source_generation": "sha256:" + "1" * 64},
        {**_request(), "expected_index_manifest": "sha256:" + "2" * 64},
    )
    for value in invalid:
        with pytest.raises(terminal.KakaoTerminalRetrievalError):
            terminal.validate_explicit_request(value)
    assert terminal.validate_explicit_request({"query": "all sources"}) == {
        "query": "all sources",
        "sources": None,
        "rooms": None,
    }
    assert terminal.validate_explicit_request(
        {"query": "one source", "scope": {"sources": ["kakao"]}}
    ) == {
        "query": "one source",
        "sources": ["kakao"],
        "rooms": None,
    }
    assert terminal.validate_explicit_request(
        {"query": "one source", "scope": {"sources": ["groupware"]}}
    ) == {
        "query": "one source",
        "sources": ["groupware"],
        "rooms": None,
    }


def test_query_is_not_trimmed_or_generated() -> None:
    for query in (" question", "question ", "", 7):
        with pytest.raises(terminal.KakaoTerminalRetrievalError):
            terminal.validate_explicit_request({"query": query, "corpus": "kakao"})
    with pytest.raises(terminal.KakaoTerminalRetrievalError, match="query is required"):
        terminal.validate_explicit_request({"corpus": "kakao"})


def _install_runtime_module(monkeypatch, **members) -> None:
    runtime_module = types.ModuleType("kwrag.product_runtime")
    for name, value in members.items():
        setattr(runtime_module, name, value)
    package_module = types.ModuleType("kwrag")
    package_module.__path__ = []
    monkeypatch.setitem(sys.modules, "kwrag", package_module)
    monkeypatch.setitem(sys.modules, "kwrag.product_runtime", runtime_module)


def test_prepare_uses_dense_product_runtime_without_ops_or_generation_contracts(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    package_root = tmp_path / "mounted-package"
    workspace_root = tmp_path / "workspace-index"
    socket_path = tmp_path / "gpu.sock"
    kakao_package = package_root / "kw" / "package"
    kakao_package.mkdir(parents=True)
    (kakao_package / "membership.json").write_text("{}", encoding="utf-8")
    captured: dict[str, object] = {}
    identity = SimpleNamespace(
        digest="sha256:" + "6" * 64,
        index_manifest="sha256:" + "2" * 64,
        pipeline_fingerprint="sha256:" + "7" * 64,
    )

    class _Runtime:
        def __init__(self) -> None:
            self.identity = identity
            self.application = SimpleNamespace(
                scope=SimpleNamespace(
                    available_rooms=["room-a", "room-b", "groupware/docs"]
                )
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def open_runtime(**kwargs):
        captured["runtime"] = kwargs
        return _Runtime()

    def index_status(**kwargs):
        captured.setdefault("status_calls", []).append(kwargs)
        return {"status": "active"}

    _install_runtime_module(
        monkeypatch,
        open_product_runtime=open_runtime,
        index_status=index_status,
    )
    monkeypatch.setattr(terminal, "get_hermes_home", lambda: home)
    monkeypatch.setattr(
        terminal,
        "load_component_manifest",
        lambda: {"component_wheel": {"sha256": "sha256:" + "8" * 64}},
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.consumer.load_component_manifest",
        lambda: {"component_wheel": {"sha256": "sha256:" + "8" * 64}},
    )
    result = SimpleNamespace(result_receipt={})

    class _Consumer:
        def __init__(self, binding, runtime, sink):
            captured["binding"] = binding
            captured["consumer_runtime"] = runtime
            captured["sink"] = sink

        def search(self, request, **kwargs):
            captured["producer_request"] = request
            captured["routing_strategy"] = kwargs.get("routing_strategy")
            return result

    monkeypatch.setattr(terminal, "HermesSlotRetrievalConsumer", _Consumer)
    monkeypatch.setattr(terminal, "FileConsumptionReceiptSink", lambda path: path)

    prepared = terminal.prepare_approved_retrieval(
        _request(),
        package_root=package_root,
        workspace_root=workspace_root,
        socket_path=socket_path,
        slot_namespace="oc20",
    )

    assert prepared is result
    assert captured["runtime"] == {
        "source_root": package_root,
        "workspace_root": workspace_root,
        "slot_namespace": "oc20",
        "socket_path": socket_path,
        "receipt_path": workspace_root / "receipts" / "operation-receipts.jsonl",
        "gpu_receipt_path": workspace_root / "receipts" / "gpu-receipts.jsonl",
    }
    producer_request = captured["producer_request"]
    assert set(producer_request) == {
        "schema_version",
        "operation",
        "query",
        "request_id",
        "operation_id",
        "run_id",
        "attempt",
        "max_results",
        "scope",
    }
    assert producer_request["query"] == _request()["query"]
    assert producer_request["schema_version"] == "kwrag-product-cli-request-v1"
    assert producer_request["operation"] == "search"
    assert producer_request["scope"] == {"sources": ["kakao"], "rooms": [
        {"source": "kakao", "room_id": "room-a"},
        {"source": "kakao", "room_id": "room-b"},
    ]}
    assert captured["routing_strategy"] == "all_mounted_rooms"
    assert "expected_source_generation" not in producer_request
    assert "expected_index_manifest" not in producer_request
    binding = captured["binding"]
    assert binding.current_index_manifest == identity.index_manifest
    assert binding.current_pipeline_fingerprint == identity.pipeline_fingerprint
    assert "binding_path" not in captured["runtime"]

    terminal.prepare_approved_retrieval(
        {"query": "search every mounted source"},
        package_root=package_root,
        workspace_root=workspace_root,
        socket_path=socket_path,
        slot_namespace="oc20",
    )
    assert "scope" not in captured["producer_request"]
    assert captured["routing_strategy"] == "all_mounted_sources"

    terminal.prepare_approved_retrieval(
        {
            "query": "find the groupware note",
            "scope": {
                "rooms": [{"source": "groupware", "roomId": "document-set-a"}]
            },
        },
        package_root=package_root,
        workspace_root=workspace_root,
        socket_path=socket_path,
        slot_namespace="oc20",
    )
    assert captured["producer_request"]["scope"] == {
        "sources": ["groupware"],
        "rooms": [{"source": "groupware", "room_id": "document-set-a"}],
    }
    assert captured["routing_strategy"] == "explicit_room_scope"
    assert captured["status_calls"] == [
        {"scope": {"sources": ["kakao"]}},
        {},
        {
            "scope": {
                "sources": ["groupware"],
                "rooms": [
                    {"source": "groupware", "room_id": "document-set-a"}
                ],
            }
        },
    ]


def test_embedded_wheel_registers_product_sources() -> None:
    with ZipFile(WHEEL) as archive:
        contract = archive.read("kwrag/product_contract.py").decode("utf-8")
        mounted_sources = archive.read("kwrag/product_sources.py").decode("utf-8")
    assert 'source="kakao"' in contract
    assert 'source="groupware"' in contract
    assert '"groupware": _GROUPWARE_ADAPTER' in contract
    assert '"kakao": _KAKAO_ADAPTER' in contract
    assert 'if source_name == "groupware"' in mounted_sources
    assert "return LiveGroupwareSource(" in mounted_sources


def test_embedded_contract_uses_omitted_scope_for_all_mounted_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(WHEEL))
    contract = importlib.import_module("kwrag.product_contract")

    normalized = contract.normalize_product_search_request(
        {
            "schema_version": "kwrag-product-cli-request-v1",
            "operation": "search",
            "query": "search every mounted source",
        }
    )

    assert normalized.scope.sources == ("groupware", "kakao")
    assert normalized.scope.rooms is None
    with pytest.raises(contract.ProductContractError, match="scope_invalid"):
        contract.normalize_product_search_request(
            {
                "schema_version": "kwrag-product-cli-request-v1",
                "operation": "search",
                "query": "search every mounted source",
                "scope": {},
            }
        )


def test_routing_scope_observation_includes_nested_source_scope() -> None:
    from plugins.kwrag_slot.consumer import _routing_scope_observation

    assert _routing_scope_observation(
        {
            "scope": {
                "sources": ["groupware"],
                "rooms": [
                    {"source": "groupware", "room_id": "document-set-a"}
                ],
            }
        }
    ) == {
        "sources": ["groupware"],
        "rooms": [{"source": "groupware", "room_id": "document-set-a"}],
    }


def test_unique_room_id_in_question_becomes_hard_scope() -> None:
    runtime = SimpleNamespace(
        application=SimpleNamespace(
            scope=SimpleNamespace(available_rooms=["room-a", "room-b"])
        )
    )


def test_explicit_single_room_is_passed_as_the_runtime_corpus() -> None:
    runtime = SimpleNamespace(
        application=SimpleNamespace(
            scope=SimpleNamespace(available_rooms=["room-a", "room-b"])
        )
    )
    assert terminal._route_rooms("ordinary question", ["room-b"], runtime) == (
        ["room-b"],
        "explicit_room_scope",
    )


def test_structured_room_source_is_retained_for_capability_validation() -> None:
    request = {
        "query": "find the groupware note",
        "scope": {"rooms": [{"source": "groupware", "roomId": "room-a"}]},
    }
    assert terminal.validate_explicit_request(request) == {
        "query": request["query"],
        "sources": ["groupware"],
        "rooms": [{"source": "groupware", "roomId": "room-a"}],
    }
    with pytest.raises(terminal.KakaoTerminalRetrievalError, match="conflicts"):
        terminal.validate_explicit_request(
            {
                "query": "q",
                "scope": {
                    "sources": ["kakao"],
                    "rooms": [{"source": "groupware", "roomId": "room-a"}],
                },
            }
        )


def test_mixed_source_rooms_keep_each_source_during_forwarding(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    captured: dict[str, object] = {}
    identity = SimpleNamespace(
        digest="sha256:" + "6" * 64,
        index_manifest="sha256:" + "2" * 64,
        pipeline_fingerprint="sha256:" + "7" * 64,
    )

    class _Runtime:
        def __init__(self) -> None:
            self.identity = identity
            self.application = SimpleNamespace(
                scope=SimpleNamespace(available_rooms=["kakao-room"])
            )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    _install_runtime_module(monkeypatch, open_product_runtime=lambda **_kwargs: _Runtime())
    monkeypatch.setattr(terminal, "get_hermes_home", lambda: home)
    monkeypatch.setattr(
        terminal,
        "load_component_manifest",
        lambda: {"component_wheel": {"sha256": "sha256:" + "8" * 64}},
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.consumer.load_component_manifest",
        lambda: {"component_wheel": {"sha256": "sha256:" + "8" * 64}},
    )

    class _Consumer:
        def __init__(self, *_args):
            pass

        def search(self, request, **kwargs):
            captured["request"] = request
            captured["strategy"] = kwargs["routing_strategy"]
            return SimpleNamespace()

    monkeypatch.setattr(terminal, "HermesSlotRetrievalConsumer", _Consumer)
    monkeypatch.setattr(terminal, "FileConsumptionReceiptSink", lambda path: path)

    terminal.prepare_approved_retrieval(
        {
            "query": "compare sources",
            "scope": {
                "sources": ["kakao", "groupware"],
                "rooms": [
                    {"source": "groupware", "roomId": "document-set-a"},
                    {"source": "kakao", "roomId": "kakao-room"},
                ],
            },
        },
        package_root=tmp_path / "mounted-package",
        workspace_root=tmp_path / "workspace-index",
        socket_path=tmp_path / "gpu.sock",
        slot_namespace="oc20",
    )

    assert captured["request"]["scope"] == {
        "sources": ["kakao", "groupware"],
        "rooms": [
            {"source": "groupware", "room_id": "document-set-a"},
            {"source": "kakao", "room_id": "kakao-room"},
        ],
    }
    assert captured["strategy"] == "explicit_room_scope"


def test_ambiguous_room_id_mentions_fail_closed_before_search() -> None:
    runtime = SimpleNamespace(
        application=SimpleNamespace(
            scope=SimpleNamespace(available_rooms=["room-a", "room-b"])
        )
    )
    with pytest.raises(terminal.KakaoTerminalRetrievalError, match="ambiguous"):
        terminal._route_rooms("compare room-a with room-b", None, runtime)
    assert terminal._route_rooms("what happened in room-a?", None, runtime) == (
        ["room-a"],
        "internal_room_id_hint",
    )


def test_agent_index_build_uses_product_native_scope_api(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    captured = {}

    def build_index(**kwargs):
        captured.update(kwargs)
        return {
            "status": "active",
            "build_id": "build-1",
            "indexed_source_count": 2,
            "skipped_source_count": 0,
        }

    _install_runtime_module(monkeypatch, build_index=build_index)
    monkeypatch.setattr(terminal, "get_hermes_home", lambda: home)
    result = terminal.build_index(
        scope={"sources": ["kakao"]},
        rebuild=True,
        source_root=tmp_path / "nas_docs",
        workspace_root=tmp_path / "workspace" / ".kwrag",
        socket_path=tmp_path / "gpu.sock",
        slot_namespace="oc20",
    )
    assert result["status"] == "active"
    assert captured == {
        "scope": {"sources": ["kakao"]},
        "rebuild": True,
    }
    with pytest.raises(terminal.KakaoTerminalRetrievalError, match="exclusions"):
        terminal.build_index(
            scope={"sources": ["kakao"]},
            exclude=[{"source": "kakao", "pattern": "tmp/*"}],
        )


def test_missing_active_index_fails_before_runtime_open(tmp_path, monkeypatch) -> None:
    opened = False
    status_scopes = []

    def open_runtime(**_kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("runtime must not open without an active index")

    def index_status(*, scope=None):
        status_scopes.append(scope)
        return {"status": "unbuilt"}

    _install_runtime_module(
        monkeypatch,
        index_status=index_status,
        open_product_runtime=open_runtime,
    )
    with pytest.raises(terminal.KakaoTerminalRetrievalError, match="index_required"):
        terminal.prepare_approved_retrieval(
            _request(),
            package_root=tmp_path / "nas_docs",
            workspace_root=tmp_path / "workspace",
            socket_path=tmp_path / "gpu.sock",
            slot_namespace="oc20",
        )
    assert opened is False
    assert status_scopes == [{"sources": ["kakao"]}]


def test_dispatch_uses_existing_consumption_and_provider_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace()
    called: dict[str, object] = {}
    monkeypatch.setattr(
        terminal,
        "run_conversation_with_approved_retrieval",
        lambda *args, **kwargs: (
            called.update(args=args, kwargs=kwargs) or {"completed": True}
        ),
    )

    assert terminal.dispatch_current_terminal_turn(
        object(),
        "answer with the verified hits",
        approved_retrieval=result,
        task_id="session-1",
    ) == {"completed": True}
    assert called["args"][2] is result
    assert called["kwargs"]["task_id"] == "session-1"


@pytest.mark.skipif(
    not hasattr(socketserver, "UnixStreamServer"),
    reason="embedded GPU transport integration requires POSIX UnixStreamServer",
)
def test_real_embedded_dense_runtime_reaches_verified_hermes_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Join the embedded wheel to Hermes without claiming a live GPU/provider."""

    installed_component = tmp_path / "installed-component"
    with ZipFile(WHEEL) as archive:
        archive.extractall(installed_component)
    monkeypatch.syspath_prepend(str(installed_component))
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == "kwrag" or name.startswith("kwrag."):
            sys.modules.pop(name, None)

    from kwrag.product_runtime import open_product_runtime
    from kwrag.shared_gpu_core import CallerIdentity, SharedGpuCore, SharedGpuCoreConfig
    from kwrag.shared_gpu_models import NonProductionDeterministicBackend
    from kwrag.shared_gpu_transport import (
        ContentFreeGpuJournal,
        InProcessSharedGpuClient,
    )
    from kwrag.workspace_dense import rebuild_workspace_dense_index

    package = tmp_path / "nas_docs" / "kw" / "package"
    workspace_root = tmp_path / "workspace" / ".kwrag"
    workspace = workspace_root / "dense"
    home = tmp_path / "home"
    package.mkdir(parents=True)
    workspace.mkdir(parents=True)
    home.mkdir()
    (package / "membership.json").write_text(
        json.dumps({"user_id": "7519030", "conversation_ids": ["room-a"]}),
        encoding="utf-8",
    )
    (package / "profile.json").write_text(
        json.dumps({"user_id": "7519030"}), encoding="utf-8"
    )
    with closing(sqlite3.connect(package / "messages.sqlite")) as connection:
        connection.execute(
            "CREATE TABLE messages("
            "conversation_id TEXT NOT NULL,message_id TEXT NOT NULL,"
            "user_id TEXT,user_name TEXT,sent_time INTEGER,plain_text TEXT)"
        )
        connection.execute(
            "INSERT INTO messages VALUES(?,?,?,?,?,?)",
            (
                "room-a",
                "message-1",
                "user-1",
                "owner",
                1,
                "the slot owner is atelier",
            ),
        )
        connection.commit()

    core = SharedGpuCore(
        NonProductionDeterministicBackend(8),
        SharedGpuCoreConfig(allow_nonproduction_backend=True),
    )
    core.start()
    client = InProcessSharedGpuClient(
        core=core,
        caller=CallerIdentity(
            slot="oc20",
            principal=f"uid:{getattr(os, 'geteuid', lambda: 0)()}",
            peer_uid=getattr(os, "geteuid", lambda: 0)(),
            peer_gid=getattr(os, "getegid", lambda: 0)(),
        ),
        journal=ContentFreeGpuJournal(tmp_path / "gpu.jsonl"),
    )

    class _PortableReceiptSink:
        def __init__(self, path: Path):
            self.path = path

        def preflight_before_retrieval(self) -> None:
            return None

        def write(self, receipt: dict) -> str:
            raw = canonical_json_bytes(receipt)
            self.path.write_bytes(raw + b"\n")
            return "sha256:" + hashlib.sha256(raw).hexdigest()

    try:
        monkeypatch.setattr("kwrag.live_kakao._require_readonly", lambda _path: None)
        rebuild_workspace_dense_index(
            package_root=package,
        workspace_root=workspace,
            slot_namespace="oc20",
            gpu=client,
            allow_nonproduction=True,
        )

        def _open_test_runtime(**kwargs):
            return open_product_runtime(
                **kwargs,
                _allow_nonproduction_release=True,
            )

        monkeypatch.setattr(
            "kwrag.product_runtime._gpu_client", lambda **_kwargs: client
        )
        monkeypatch.setattr(
            "kwrag.product_runtime.open_product_runtime",
            _open_test_runtime,
        )
        monkeypatch.setattr(terminal, "get_hermes_home", lambda: home)
        monkeypatch.setattr(
            terminal, "FileConsumptionReceiptSink", _PortableReceiptSink
        )
        prepared = terminal.prepare_approved_retrieval(
            {"query": "who owns the slot?", "corpus": "kakao"},
            package_root=package,
            workspace_root=workspace_root,
            socket_path=tmp_path / "unused.sock",
            slot_namespace="oc20",
        )
    finally:
        core.shutdown(grace_seconds=1)

    assert prepared.result_receipt["result_status"] == "hits"
    assert prepared.result_receipt["result_count"] == 1
    assert prepared.results[0]["snippet"] == "the slot owner is atelier"
    operation_receipt = json.loads(
        (workspace_root / "receipts" / "operation-receipts.jsonl").read_text(
            encoding="utf-8"
        )
    )
    assert [
        stage["stage_id"] for stage in operation_receipt["pipeline_evidence"]["stages"]
    ] == ["query_embedding", "dense_index_search", "candidate_rerank"]
    assert "source_generation" not in json.dumps(operation_receipt)
