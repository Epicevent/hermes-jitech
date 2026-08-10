"""Tests for the thin caller-explicit Kakao -> Hermes product seam."""

from __future__ import annotations

from contextlib import closing
import hashlib
import importlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import types
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from plugins.kwrag_slot import terminal
from plugins.kwrag_slot.manifest import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[2]
WHEEL = ROOT / "vendor" / "kwrag" / "kwrag_product_service-0.5.0-py3-none-any.whl"


def _request() -> dict[str, str]:
    return {"query": "who owns the slot?", "corpus": "kakao"}


def test_explicit_request_owns_only_query_and_product_corpus() -> None:
    assert terminal.validate_explicit_request(_request()) == _request()
    invalid = (
        {"query": "question"},
        {"query": "question", "corpus": "groupware"},
        {**_request(), "expected_source_generation": "sha256:" + "1" * 64},
        {**_request(), "expected_index_manifest": "sha256:" + "2" * 64},
    )
    for value in invalid:
        with pytest.raises(terminal.KakaoTerminalRetrievalError):
            terminal.validate_explicit_request(value)


def test_query_is_not_trimmed_or_generated() -> None:
    for query in (" question", "question ", "", 7):
        with pytest.raises(terminal.KakaoTerminalRetrievalError):
            terminal.validate_explicit_request({"query": query, "corpus": "kakao"})


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
    captured: dict[str, object] = {}
    identity = SimpleNamespace(
        digest="sha256:" + "6" * 64,
        index_manifest="sha256:" + "2" * 64,
        pipeline_fingerprint="sha256:" + "7" * 64,
    )

    class _Runtime:
        def __init__(self) -> None:
            self.identity = identity

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    def open_runtime(**kwargs):
        captured["runtime"] = kwargs
        return _Runtime()

    _install_runtime_module(
        monkeypatch,
        open_kakao_product_runtime=open_runtime,
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

        def search(self, request):
            captured["producer_request"] = request
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
        "package_root": package_root,
        "workspace_root": workspace_root,
        "slot_namespace": "oc20",
        "socket_path": socket_path,
        "receipt_path": home / "kwrag" / "operation-receipts.jsonl",
        "gpu_receipt_path": home / "kwrag" / "gpu-receipts.jsonl",
    }
    producer_request = captured["producer_request"]
    assert set(producer_request) == {
        "schema_version",
        "query",
        "request_id",
        "operation_id",
        "run_id",
        "attempt",
        "max_results",
        "corpus",
    }
    assert producer_request["query"] == _request()["query"]
    assert producer_request["corpus"] is None
    assert "expected_source_generation" not in producer_request
    assert "expected_index_manifest" not in producer_request
    assert "binding_path" not in captured["runtime"]


def test_explicit_index_rebuild_uses_the_same_product_paths(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    captured = {}

    def rebuild(**kwargs):
        captured.update(kwargs)
        return {"activation": {"status": "active"}}

    _install_runtime_module(
        monkeypatch,
        rebuild_kakao_product_index=rebuild,
    )
    monkeypatch.setattr(terminal, "get_hermes_home", lambda: home)
    result = terminal.rebuild_kakao_index(
        package_root=tmp_path / "package",
        workspace_root=tmp_path / "index",
        socket_path=tmp_path / "gpu.sock",
        slot_namespace="oc20",
    )
    assert result["activation"]["status"] == "active"
    assert captured == {
        "package_root": tmp_path / "package",
        "workspace_root": tmp_path / "index",
        "slot_namespace": "oc20",
        "socket_path": tmp_path / "gpu.sock",
        "gpu_receipt_path": home / "kwrag" / "gpu-receipts.jsonl",
    }


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

    from kwrag.product_runtime import open_kakao_product_runtime
    from kwrag.shared_gpu_core import CallerIdentity, SharedGpuCore, SharedGpuCoreConfig
    from kwrag.shared_gpu_models import NonProductionDeterministicBackend
    from kwrag.shared_gpu_transport import (
        ContentFreeGpuJournal,
        InProcessSharedGpuClient,
    )
    from kwrag.workspace_dense import rebuild_workspace_dense_index

    package = tmp_path / "nas_docs" / "kw" / "package"
    workspace = tmp_path / "workspace" / ".kwrag" / "dense"
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
            return open_kakao_product_runtime(
                **kwargs,
                _allow_nonproduction_release=True,
            )

        monkeypatch.setattr(
            "kwrag.product_runtime._gpu_client", lambda **_kwargs: client
        )
        monkeypatch.setattr(
            "kwrag.product_runtime.open_kakao_product_runtime",
            _open_test_runtime,
        )
        monkeypatch.setattr(terminal, "get_hermes_home", lambda: home)
        monkeypatch.setattr(
            terminal, "FileConsumptionReceiptSink", _PortableReceiptSink
        )
        prepared = terminal.prepare_approved_retrieval(
            {"query": "who owns the slot?", "corpus": "kakao"},
            package_root=package,
            workspace_root=workspace,
            socket_path=tmp_path / "unused.sock",
            slot_namespace="oc20",
        )
    finally:
        core.shutdown(grace_seconds=1)

    assert prepared.result_receipt["result_status"] == "hits"
    assert prepared.result_receipt["result_count"] == 1
    assert prepared.results[0]["snippet"] == "the slot owner is atelier"
    operation_receipt = json.loads(
        (home / "kwrag" / "operation-receipts.jsonl").read_text(encoding="utf-8")
    )
    assert [
        stage["stage_id"] for stage in operation_receipt["pipeline_evidence"]["stages"]
    ] == ["query_embedding", "dense_index_search", "candidate_rerank"]
    assert "source_generation" not in json.dumps(operation_receipt)
