"""Tests for the caller-explicit Kakao -> Hermes terminal seam."""

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


def test_explicit_request_requires_query_and_two_generation_pins() -> None:
    assert terminal.validate_explicit_request(_pins())["corpus"] == "kakao"
    for field in ("query", "expected_source_generation", "expected_index_manifest"):
        value = _pins()
        value.pop(field)
        with pytest.raises(terminal.KakaoTerminalRetrievalError):
            terminal.validate_explicit_request(value)


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
