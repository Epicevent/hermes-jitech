"""Tests for the caller-explicit Kakao -> Hermes terminal seam."""

import json
import sys
import types
from types import SimpleNamespace

import pytest

from plugins.kwrag_slot import terminal


def _pins() -> dict[str, str]:
    return {
        "query": "who owns the slot?",
        "corpus": "kakao",
        "expected_source_generation": "sha256:" + "1" * 64,
        "expected_index_manifest": "sha256:" + "2" * 64,
    }


def _runtime_handoff() -> dict[str, object]:
    digest = "sha256:" + "a" * 64
    return {
        "schema_version": "kwrag-dense-runtime-handoff-v1",
        "status": "active",
        "slot_namespace": "oc20",
        "release_id": "d-" + "b" * 64,
        "release_relative": "oc20/releases/d-" + "b" * 64,
        "read_view_relative": "views/current",
        "read_only_required": True,
        "source_generation": digest,
        "source_snapshot_sha256": digest,
        "source_database_sha256": digest,
        "source_membership_sha256": digest,
        "source_profile_sha256": digest,
        "index_manifest_sha256": digest,
        "pipeline_fingerprint": digest,
        "embedding_fingerprint": digest,
        "receipt_digests": {
            "embedding_operation": digest,
            "activation": digest,
            "operation": None,
            "result": None,
            "consumption": None,
            "provider": None,
        },
        "raw_content_present": False,
    }


def test_explicit_request_requires_query_and_two_generation_pins() -> None:
    assert terminal.validate_explicit_request(_pins())["corpus"] == "kakao"
    for field in ("query", "expected_source_generation", "expected_index_manifest"):
        value = _pins()
        value.pop(field)
        with pytest.raises(terminal.KakaoTerminalRetrievalError):
            terminal.validate_explicit_request(value)


def test_explicit_request_can_use_runtime_owned_pins() -> None:
    value = terminal.validate_explicit_request(
        {"query": "who owns the slot?", "corpus": "kakao"},
        require_pins=False,
    )
    assert "expected_source_generation" not in value
    assert "expected_index_manifest" not in value


def test_runtime_handoff_binds_source_index_and_receipt_lineage() -> None:
    value, digest = terminal._validate_runtime_handoff(_runtime_handoff())
    assert value["source_generation"] == value["index_manifest_sha256"]
    assert digest.startswith("sha256:")


def test_fixed_producer_execute_receives_both_runtime_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handoff = _runtime_handoff()
    generation = handoff["source_generation"]
    index_manifest = handoff["index_manifest_sha256"]
    pipeline = handoff["pipeline_fingerprint"]
    binding = SimpleNamespace(enabled=True, index_manifest_digest=index_manifest)
    captured: dict[str, object] = {}
    operation_receipt = {
        "source_generation": generation,
        "index_manifest": index_manifest,
        "pipeline_fingerprint": pipeline,
        "read_only_required": True,
        "duration_ms": 7,
    }
    operation_receipt_digest = "sha256:" + terminal.hashlib.sha256(
        terminal.canonical_json_bytes(operation_receipt)
    ).hexdigest()

    def execute(payload: bytes, *, binding_path: object) -> SimpleNamespace:
        captured["request"] = json.loads(payload.decode("utf-8"))
        captured["binding_path"] = binding_path
        request = captured["request"]
        output = {
            "consumable": {
                "request_id": request["request_id"],
                "operation_id": request["operation_id"],
                "run_id": request["run_id"],
                "attempt": request["attempt"],
                "source_generation": generation,
                "index_manifest": index_manifest,
                "pipeline_fingerprint": pipeline,
                "result_digest": "sha256:" + "4" * 64,
                "result_status": "zero_hits",
                "results": [],
            },
            "linkage": {"operation_receipt_digest": operation_receipt_digest},
        }
        return SimpleNamespace(
            output_bytes=terminal.canonical_json_bytes(output),
            producer_receipt={},
            binding_digest="sha256:" + "5" * 64,
            operation_receipt=operation_receipt,
        )

    fixed_producer = SimpleNamespace(
        load_fixed_producer_binding=lambda path: binding,
        load_runtime_handoff=lambda path: handoff,
        execute_fixed_producer=execute,
        verify_fixed_producer_output=lambda *args, **kwargs: None,
    )
    kwrag = types.ModuleType("kwrag")
    kwrag.fixed_producer = fixed_producer
    monkeypatch.setitem(sys.modules, "kwrag", kwrag)
    monkeypatch.setattr(
        terminal, "_producer_binding_path", lambda: terminal.Path("/run/binding.json")
    )

    terminal._fixed_producer_exchange(
        {
            "schema_version": "kwrag-slot-search-request-v1",
            "query": "who owns the slot?",
            "request_id": "request-1",
            "operation_id": "operation-1",
            "run_id": "run-1",
            "attempt": 1,
            "max_results": 10,
            "corpus": "kakao",
        }
    )

    request = captured["request"]
    assert request["expected_source_generation"] == generation
    assert request["expected_index_manifest"] == index_manifest


@pytest.mark.parametrize(
    "field, replacement",
    [
        ("read_only_required", False),
        ("source_database_sha256", "not-a-digest"),
        ("receipt_digests", {"embedding_operation": None}),
    ],
)
def test_runtime_handoff_rejects_drift_or_incomplete_receipts(field, replacement) -> None:
    value = _runtime_handoff()
    value[field] = replacement
    with pytest.raises(terminal.KakaoTerminalRetrievalError):
        terminal._validate_runtime_handoff(value)


def test_query_is_not_trimmed_or_generated() -> None:
    value = _pins()
    value["query"] = " who owns the slot?"
    with pytest.raises(terminal.KakaoTerminalRetrievalError):
        terminal.validate_explicit_request(value)


def test_dispatch_validates_content_free_context_before_existing_consumption_seam(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = {
        "source_generation": _pins()["expected_source_generation"],
        "index_manifest": _pins()["expected_index_manifest"],
        "runtime_binding_digest": "sha256:" + "6" * 64,
        "operation_receipt_digest": "sha256:" + "3" * 64,
        "result_digest": "sha256:" + "4" * 64,
        "result_count": 1,
    }
    result = SimpleNamespace(
        result_receipt=receipt,
        result_receipt_digest="sha256:" + "5" * 64,
    )
    context = terminal.canonical_json_bytes(
        {
            "schema_version": "hermes-kwrag-current-turn-context-v1",
            "source_generation": receipt["source_generation"],
            "index_manifest": receipt["index_manifest"],
            "runtime_binding_digest": "sha256:" + "6" * 64,
            "operation_receipt_digest": receipt["operation_receipt_digest"],
            "result_receipt_digest": result.result_receipt_digest,
            "result_digest": receipt["result_digest"],
            "result_count": receipt["result_count"],
        }
    )
    called = {}
    monkeypatch.setattr(
        terminal,
        "run_conversation_with_approved_retrieval",
        lambda *args, **kwargs: called.update(args=args, kwargs=kwargs) or {"completed": True},
    )
    assert terminal.dispatch_current_terminal_turn(
        object(),
        "answer with the verified hits",
        kwrag_current_turn_context=context,
        approved_retrieval=result,
        task_id="session-1",
    ) == {"completed": True}
    assert called["kwargs"] == {"task_id": "session-1"}


def test_dispatch_rejects_context_from_different_generation() -> None:
    receipt = {
        "source_generation": _pins()["expected_source_generation"],
        "index_manifest": _pins()["expected_index_manifest"],
        "runtime_binding_digest": "sha256:" + "6" * 64,
        "operation_receipt_digest": "sha256:" + "3" * 64,
        "result_digest": "sha256:" + "4" * 64,
        "result_count": 1,
    }
    result = SimpleNamespace(
        result_receipt=receipt,
        result_receipt_digest="sha256:" + "5" * 64,
    )
    context = terminal.canonical_json_bytes(
        {
            "schema_version": "hermes-kwrag-current-turn-context-v1",
            "source_generation": "sha256:" + "9" * 64,
            "index_manifest": receipt["index_manifest"],
            "runtime_binding_digest": "sha256:" + "6" * 64,
            "operation_receipt_digest": receipt["operation_receipt_digest"],
            "result_receipt_digest": result.result_receipt_digest,
            "result_digest": receipt["result_digest"],
            "result_count": receipt["result_count"],
        }
    )
    with pytest.raises(terminal.KakaoTerminalRetrievalError):
        terminal.dispatch_current_terminal_turn(
            object(),
            "question",
            kwrag_current_turn_context=context,
            approved_retrieval=result,
        )


def test_dispatch_rejects_context_from_different_runtime_handoff() -> None:
    receipt = {
        "source_generation": _pins()["expected_source_generation"],
        "index_manifest": _pins()["expected_index_manifest"],
        "runtime_binding_digest": "sha256:" + "6" * 64,
        "operation_receipt_digest": "sha256:" + "3" * 64,
        "result_digest": "sha256:" + "4" * 64,
        "result_count": 1,
    }
    result = SimpleNamespace(
        result_receipt=receipt,
        result_receipt_digest="sha256:" + "5" * 64,
    )
    context = terminal.canonical_json_bytes(
        {
            "schema_version": "hermes-kwrag-current-turn-context-v1",
            "source_generation": receipt["source_generation"],
            "index_manifest": receipt["index_manifest"],
            "runtime_binding_digest": "sha256:" + "7" * 64,
            "operation_receipt_digest": receipt["operation_receipt_digest"],
            "result_receipt_digest": result.result_receipt_digest,
            "result_digest": receipt["result_digest"],
            "result_count": receipt["result_count"],
        }
    )
    with pytest.raises(terminal.KakaoTerminalRetrievalError):
        terminal.dispatch_current_terminal_turn(
            object(),
            "question",
            kwrag_current_turn_context=context,
            approved_retrieval=result,
        )
