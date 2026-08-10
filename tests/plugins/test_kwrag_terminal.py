"""Tests for the thin caller-explicit Kakao -> Hermes product seam."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace

import pytest

from plugins.kwrag_slot import terminal


def _request() -> dict[str, str]:
    return {"query": "who owns the slot?", "corpus": "kakao"}


def _receipt() -> dict[str, object]:
    return {
        "index_manifest": "sha256:" + "2" * 64,
        "runtime_binding_digest": "sha256:" + "6" * 64,
        "operation_receipt_digest": "sha256:" + "3" * 64,
        "result_digest": "sha256:" + "4" * 64,
        "result_count": 1,
    }


def _context(result: object) -> bytes:
    receipt = result.result_receipt
    return terminal.canonical_json_bytes(
        {
            "schema_version": "hermes-kwrag-current-turn-context-v1",
            "index_manifest": receipt["index_manifest"],
            "runtime_binding_digest": receipt["runtime_binding_digest"],
            "operation_receipt_digest": receipt["operation_receipt_digest"],
            "result_receipt_digest": result.result_receipt_digest,
            "result_digest": receipt["result_digest"],
            "result_count": receipt["result_count"],
        }
    )


def test_explicit_request_owns_only_query_and_product_corpus() -> None:
    assert terminal.validate_explicit_request(_request()) == _request()
    invalid = (
        {"query": "question"},
        {"query": "question", "corpus": "groupware"},
        {
            **_request(),
            "expected_source_generation": "sha256:" + "1" * 64,
        },
        {**_request(), "expected_index_manifest": "sha256:" + "2" * 64},
    )
    for value in invalid:
        with pytest.raises(terminal.KakaoTerminalRetrievalError):
            terminal.validate_explicit_request(value)


def test_query_is_not_trimmed_or_generated() -> None:
    for query in (" question", "question ", "", 7):
        with pytest.raises(terminal.KakaoTerminalRetrievalError):
            terminal.validate_explicit_request({"query": query, "corpus": "kakao"})


def test_prepare_uses_product_runtime_without_ops_binding_or_generation_pins(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    package_root = tmp_path / "mounted-package"
    package_root.mkdir()
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

    runtime_module = types.ModuleType("kwrag.product_runtime")
    runtime_module.open_kakao_product_runtime = open_runtime
    package_module = types.ModuleType("kwrag")
    package_module.__path__ = []
    monkeypatch.setitem(sys.modules, "kwrag", package_module)
    monkeypatch.setitem(sys.modules, "kwrag.product_runtime", runtime_module)
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

    result = SimpleNamespace(
        result_receipt=_receipt(),
        result_receipt_digest="sha256:" + "5" * 64,
    )

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

    prepared, context = terminal.prepare_approved_retrieval(
        _request(),
        package_root=package_root,
    )

    assert prepared is result
    assert captured["runtime"] == {
        "package_root": package_root,
        "receipt_path": home / "kwrag" / "operation-receipts.jsonl",
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
    }
    assert producer_request["query"] == _request()["query"]
    assert "expected_source_generation" not in producer_request
    assert "expected_index_manifest" not in producer_request
    assert "binding_path" not in captured["runtime"]
    assert json.loads(context) == json.loads(_context(result))


def test_dispatch_validates_context_before_existing_consumption_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = SimpleNamespace(
        result_receipt=_receipt(),
        result_receipt_digest="sha256:" + "5" * 64,
    )
    called: dict[str, object] = {}
    monkeypatch.setattr(
        terminal,
        "run_conversation_with_approved_retrieval",
        lambda *args, **kwargs: called.update(args=args, kwargs=kwargs)
        or {"completed": True},
    )

    assert terminal.dispatch_current_terminal_turn(
        object(),
        "answer with the verified hits",
        kwrag_current_turn_context=_context(result),
        approved_retrieval=result,
        task_id="session-1",
    ) == {"completed": True}
    assert called["args"][2] is result
    assert called["kwargs"]["task_id"] == "session-1"


def test_dispatch_rejects_context_not_bound_to_verified_result() -> None:
    result = SimpleNamespace(
        result_receipt=_receipt(),
        result_receipt_digest="sha256:" + "5" * 64,
    )
    context = json.loads(_context(result))
    context["runtime_binding_digest"] = "sha256:" + "9" * 64
    with pytest.raises(terminal.KakaoTerminalRetrievalError):
        terminal.dispatch_current_terminal_turn(
            object(),
            "question",
            kwrag_current_turn_context=terminal.canonical_json_bytes(context),
            approved_retrieval=result,
        )
