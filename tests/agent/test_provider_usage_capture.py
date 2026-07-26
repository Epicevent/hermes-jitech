import asyncio

import pytest

from agent.provider_usage_capture import (
    ProviderAttemptSeries,
    bind_provider_usage_context,
    capture_provider_call,
    capture_provider_call_async,
)
from hermes_state import SessionDB


def _receipts(db_path):
    db = SessionDB(db_path)
    try:
        return db.export_provider_usage_receipts()["receipts"]
    finally:
        db.close()


def test_contextless_call_records_nullable_correlation_and_exact_usage(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home",
        lambda: tmp_path,
    )
    response = {
        "id": "resp_123",
        "model": "gpt-5.4-2026-06-01",
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
        "choices": [{"finish_reason": "stop"}],
    }

    assert capture_provider_call(
        lambda: response,
        provider="openai",
        model="gpt-5.4",
    ) is response

    [receipt] = _receipts(tmp_path / "state.db")
    assert receipt["runId"] is None
    assert receipt["turnId"] is None
    assert receipt["requestId"] is None
    assert receipt["sessionId"] is None
    assert receipt["trigger"] == "unknown"
    assert receipt["actual"] == {
        "provider": "openai",
        "model": "gpt-5.4-2026-06-01",
        "responseId": "resp_123",
        "evidenceSource": "response.model",
    }
    assert receipt["usage"]["inputTotal"] == 11
    assert receipt["usage"]["outputCandidates"] == 7
    assert receipt["usage"]["providerReportedTotal"] == 18


def test_retry_and_fallback_have_one_receipt_per_physical_attempt(tmp_path):
    db_path = tmp_path / "state.db"
    with bind_provider_usage_context(
        session_id="session-1",
        run_id="run-1",
        turn_id="turn-1",
        request_id="request-1",
        trigger="user",
        configured_provider="openai",
        configured_model="primary-model",
        db_path=db_path,
    ):
        series = ProviderAttemptSeries()
        with pytest.raises(RuntimeError, match="first failed"):
            capture_provider_call(
                lambda: (_ for _ in ()).throw(RuntimeError("first failed")),
                provider="openai",
                model="primary-model",
                series=series,
            )
        capture_provider_call(
            lambda: {"model": "primary-model", "usage": {"input_tokens": 3}},
            provider="openai",
            model="primary-model",
            series=series,
        )
        capture_provider_call(
            lambda: {"model": "fallback-model", "usage": {"input_tokens": 4}},
            provider="anthropic",
            model="fallback-model",
            series=series,
        )
        capture_provider_call(
            lambda: {"model": "fallback-model", "usage": {"input_tokens": 5}},
            provider="anthropic",
            model="fallback-model",
            series=series,
        )

    receipts = _receipts(db_path)
    assert [item["status"] for item in receipts] == [
        "failed",
        "succeeded",
        "succeeded",
        "succeeded",
    ]
    assert [item["attempt"] for item in receipts] == [1, 2, 3, 4]
    assert len({item["callId"] for item in receipts}) == 4
    assert receipts[0]["retryOf"] is None
    assert receipts[1]["retryOf"] == receipts[0]["callId"]
    assert receipts[1]["fallbackParent"] is None
    assert receipts[2]["retryOf"] == receipts[1]["callId"]
    assert receipts[2]["fallbackParent"] == receipts[1]["callId"]
    assert receipts[3]["retryOf"] == receipts[2]["callId"]
    assert receipts[3]["fallbackParent"] == receipts[1]["callId"]
    assert [item["fallbackIndex"] for item in receipts] == [0, 0, 1, 1]
    assert {item["runId"] for item in receipts} == {"run-1"}
    assert {item["turnId"] for item in receipts} == {"turn-1"}


def test_missing_per_call_usage_stays_unavailable_instead_of_zero(tmp_path):
    db_path = tmp_path / "state.db"
    with bind_provider_usage_context(
        session_id=None,
        run_id=None,
        turn_id=None,
        trigger="unknown",
        db_path=db_path,
    ):
        capture_provider_call(
            lambda: {"id": "image_1"},
            provider="openai",
            model="gpt-image-2",
        )

    [receipt] = _receipts(db_path)
    assert receipt["status"] == "succeeded"
    assert receipt["usageCoverage"] == "unavailable"
    assert all(value is None for value in receipt["usage"].values())


def test_bedrock_camel_case_usage_is_preserved_per_call(tmp_path):
    db_path = tmp_path / "state.db"
    with bind_provider_usage_context(
        session_id=None,
        run_id=None,
        turn_id=None,
        db_path=db_path,
    ):
        capture_provider_call(
            lambda: {
                "model": "amazon.nova-pro-v1:0",
                "provider_usage": {
                    "inputTokens": 12,
                    "outputTokens": 5,
                    "totalTokens": 17,
                    "cacheReadInputTokens": 4,
                    "cacheWriteInputTokens": 2,
                },
            },
            provider="bedrock",
            model="amazon.nova-pro-v1:0",
        )

    [receipt] = _receipts(db_path)
    assert receipt["usage"]["inputTotal"] == 12
    assert receipt["usage"]["inputNonCached"] == 8
    assert receipt["usage"]["cacheRead"] == 4
    assert receipt["usage"]["cacheWrite"] == 2
    assert receipt["usage"]["outputCandidates"] == 5
    assert receipt["usage"]["providerReportedTotal"] == 17


def test_observation_transform_failure_does_not_break_provider_result(
    tmp_path, caplog
):
    db_path = tmp_path / "state.db"
    response = {"id": "raw-response"}
    with bind_provider_usage_context(
        session_id=None,
        run_id=None,
        turn_id=None,
        db_path=db_path,
    ):
        result = capture_provider_call(
            lambda: response,
            provider="xai",
            model="xai-tts",
            response_transform=lambda _response: 1 / 0,
        )

    assert result is response
    assert "Provider usage response observation failed" in caplog.text
    assert _receipts(db_path)[0]["status"] == "succeeded"


def test_async_cancelled_attempt_is_recorded_and_reraised(tmp_path):
    db_path = tmp_path / "state.db"

    async def _cancel():
        raise asyncio.CancelledError

    async def _run():
        with bind_provider_usage_context(
            session_id=None,
            run_id="run-async",
            turn_id="turn-async",
            db_path=db_path,
        ):
            with pytest.raises(asyncio.CancelledError):
                await capture_provider_call_async(
                    _cancel,
                    provider="openai",
                    model="gpt-5.4",
                )

    asyncio.run(_run())
    [receipt] = _receipts(db_path)
    assert receipt["status"] == "cancelled"
    assert receipt["errorCategory"] == "CancelledError"
