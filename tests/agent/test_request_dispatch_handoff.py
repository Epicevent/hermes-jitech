from __future__ import annotations

import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.request_dispatch import (
    DispatchAttemptsClosed,
    FinalProviderBindingUnsupported,
    ProviderAttemptLedgerRequired,
    RequestDispatchHandoff,
    coerce_request_dispatch_handoff,
    provider_endpoint_identity,
    require_authoritative_leaf_adapter,
    require_retrieval_evidence_dispatch_capability,
    snapshot_allowed_provider_routes,
)


def _handoff(callback=None, *, interrupted=lambda: False):
    return RequestDispatchHandoff(
        callback,
        interrupted=interrupted,
        interrupted_message="request abandoned before dispatch",
    )


def _route(
    *,
    fallback_index: int = 0,
    provider: str = "fixture-provider",
    model: str = "fixture-model",
    api_mode: str = "chat_completions",
    endpoint_identity: str = "https://fixture.example/v1",
) -> dict:
    return {
        "fallbackIndex": fallback_index,
        "provider": provider,
        "model": model,
        "apiMode": api_mode,
        "endpointIdentity": endpoint_identity,
    }


def test_product_revision_has_no_retrieval_evidence_dispatch_adapter() -> None:
    with pytest.raises(
        FinalProviderBindingUnsupported,
        match="no production atomic serialized-request adapter",
    ):
        require_retrieval_evidence_dispatch_capability(SimpleNamespace())


def test_abandon_wins_before_commit() -> None:
    receipt_called = threading.Event()
    sdk_called = threading.Event()
    handoff = _handoff(receipt_called.set)

    assert handoff.abandon() is True
    with pytest.raises(DispatchAttemptsClosed) as exc_info:
        handoff.commit_and_claim_dispatch(sdk_called.set)
    assert exc_info.value.cause == "cancelled"

    assert handoff.state == "abandoned"
    assert handoff.sdk_entry_intent_committed is False
    assert receipt_called.is_set() is False
    assert sdk_called.is_set() is False


def test_dispatch_owned_wins_and_losing_abandon_waits_for_entry_announcement() -> None:
    receipt_started = threading.Event()
    release_receipt = threading.Event()
    sdk_entered = threading.Event()
    worker_done = threading.Event()
    abandon_started = threading.Event()
    abandon_done = threading.Event()
    abandon_result: list[bool] = []

    def receipt_callback() -> None:
        receipt_started.set()
        assert release_receipt.wait(timeout=5)

    handoff = _handoff(receipt_callback)

    def worker() -> None:
        handoff.commit_and_claim_dispatch(sdk_entered.set)
        worker_done.set()

    def canceller() -> None:
        abandon_started.set()
        abandon_result.append(handoff.abandon())
        abandon_done.set()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    assert receipt_started.wait(timeout=5)
    cancel_thread = threading.Thread(target=canceller, daemon=True)
    cancel_thread.start()
    assert abandon_started.wait(timeout=5)
    assert abandon_done.is_set() is False

    release_receipt.set()
    assert sdk_entered.wait(timeout=5)
    assert abandon_done.wait(timeout=5)
    assert worker_done.wait(timeout=5)
    worker_thread.join(timeout=5)
    cancel_thread.join(timeout=5)

    assert abandon_result == [False]
    assert handoff.state == "dispatch_owned"
    assert handoff.sdk_entry_intent_committed is True


def test_sdk_exception_after_entry_attempt_preserves_dispatch_owned() -> None:
    receipt_called = threading.Event()
    sdk_entered = threading.Event()
    handoff = _handoff(receipt_called.set)

    def failed_sdk_call():
        sdk_entered.set()
        raise RuntimeError("provider failed after SDK entry")

    with pytest.raises(RuntimeError, match="after SDK entry"):
        handoff.commit_and_claim_dispatch(failed_sdk_call)

    assert receipt_called.is_set() is True
    assert sdk_entered.is_set() is True
    assert handoff.state == "dispatch_owned"
    assert handoff.sdk_entry_intent_committed is True
    assert handoff.abandon() is False


def test_outcome_sink_failure_latches_one_winner_and_blocks_late_rewrite() -> None:
    attempts: list[tuple[str, str, str | None]] = []

    def failing_outcome(status: str, digest: str, error: str | None) -> None:
        attempts.append((status, digest, error))
        raise OSError("outcome fsync failed")

    handoff = RequestDispatchHandoff(
        lambda _binding, _kwargs: None,
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        outcome_callback=failing_outcome,
        configured_provider="fixture-provider",
        configured_model="fixture-model",
        allowed_provider_routes=(_route(),),
    )
    response = _bound_dispatch(
        handoff,
        {"model": "fixture-model", "messages": []},
        lambda _kwargs: "ok",
    )

    assert response == "ok"
    assert handoff.terminal_outcome_status == "response_observed"
    assert isinstance(handoff.outcome_persistence_error, OSError)
    handoff.record_terminal_outcome("sdk_exception", "LateFailure")
    assert [item[0] for item in attempts] == ["response_observed"]


def test_sdk_exception_identity_survives_outcome_sink_failure() -> None:
    attempts: list[tuple[str, str, str | None]] = []

    def failing_outcome(status: str, digest: str, error: str | None) -> None:
        attempts.append((status, digest, error))
        raise OSError("outcome fsync failed")

    handoff = RequestDispatchHandoff(
        lambda _binding, _kwargs: None,
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        outcome_callback=failing_outcome,
        configured_provider="fixture-provider",
        configured_model="fixture-model",
        allowed_provider_routes=(_route(),),
    )

    def failed_sdk(_kwargs):
        raise RuntimeError("provider transport exploded")

    with pytest.raises(RuntimeError, match="provider transport exploded"):
        _bound_dispatch(
            handoff,
            {"model": "fixture-model", "messages": []},
            failed_sdk,
        )

    assert handoff.terminal_outcome_status == "sdk_exception"
    assert isinstance(handoff.outcome_persistence_error, OSError)
    assert [item[0] for item in attempts] == ["sdk_exception"]


def test_receipt_failure_is_terminal_and_never_enters_sdk() -> None:
    sdk_called = threading.Event()

    def failed_receipt() -> None:
        raise RuntimeError("receipt sink failed")

    handoff = _handoff(failed_receipt)
    with pytest.raises(RuntimeError, match="receipt sink failed"):
        handoff.commit_and_claim_dispatch(sdk_called.set)

    assert handoff.state == "failed"
    assert handoff.sdk_entry_intent_committed is False
    assert sdk_called.is_set() is False
    assert handoff.abandon() is False


def test_nested_provider_wrappers_reuse_the_same_attempt_handoff() -> None:
    handoff = _handoff()

    nested = coerce_request_dispatch_handoff(
        handoff,
        interrupted=lambda: True,
        interrupted_message="must not replace shared handoff",
    )

    assert nested is handoff


def test_terminal_close_wins_between_retry_check_and_retry_claim() -> None:
    first_sdk_calls = 0
    retry_sdk_calls = 0
    retry_reached_barrier = threading.Event()
    release_retry_claim = threading.Event()
    retry_done = threading.Event()
    retry_errors: list[BaseException] = []
    handoff = _handoff()

    def first_sdk() -> None:
        nonlocal first_sdk_calls
        first_sdk_calls += 1

    handoff.commit_and_claim_dispatch(first_sdk)

    def retry_sdk() -> None:
        nonlocal retry_sdk_calls
        retry_sdk_calls += 1

    def retry_worker() -> None:
        try:
            retry_reached_barrier.set()
            assert release_retry_claim.wait(timeout=5)
            handoff.commit_and_claim_dispatch(retry_sdk)
        except BaseException as exc:
            retry_errors.append(exc)
        finally:
            retry_done.set()

    thread = threading.Thread(target=retry_worker, daemon=True)
    thread.start()
    assert retry_reached_barrier.wait(timeout=5)
    assert handoff.abandon() is False
    assert handoff.future_attempts_closed is True
    release_retry_claim.set()
    assert retry_done.wait(timeout=5)
    thread.join(timeout=5)

    assert first_sdk_calls == 1
    assert retry_sdk_calls == 0
    assert len(retry_errors) == 1
    assert isinstance(retry_errors[0], DispatchAttemptsClosed)
    assert handoff.attempt_claim_count == 1


def test_normal_retry_claims_a_distinct_provider_attempt() -> None:
    calls: list[str] = []
    handoff = _handoff()

    def first_sdk() -> None:
        calls.append("first")
        raise RuntimeError("transient provider failure")

    with pytest.raises(RuntimeError, match="transient provider failure"):
        handoff.commit_and_claim_dispatch(first_sdk)

    handoff.commit_and_claim_dispatch(lambda: calls.append("retry"))

    assert calls == ["first", "retry"]
    assert handoff.state == "dispatch_owned"
    assert handoff.future_attempts_closed is False
    assert handoff.attempt_claim_count == 2


def _bound_dispatch(handoff, request_kwargs, invoke):
    if (
        handoff.requires_exact_provider_attempt_binding
        and handoff.provider_call_id is None
    ):
        handoff.bind_provider_call_identity(str(uuid.uuid4()))
    return handoff.commit_and_claim_dispatch(
        invoke,
        provider="fixture-provider",
        api_mode="chat_completions",
        model=str(request_kwargs.get("model") or "fixture-model"),
        sdk_method="chat.completions.create",
        leaf_adapter="tests.fixture.FakeClient",
        endpoint_identity="https://fixture.example/v1",
        fallback_index=0,
        request_kwargs=request_kwargs,
    )


def test_exactly_one_evidence_attempt_requires_a_distinct_retry_ledger() -> None:
    bindings: list[dict] = []
    sdk_calls: list[dict] = []
    outcomes: list[tuple[str, str, str | None]] = []
    handoff = RequestDispatchHandoff(
        lambda binding, _kwargs: bindings.append(dict(binding)),
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        outcome_callback=lambda status, digest, error: outcomes.append(
            (status, digest, error)
        ),
        configured_provider="fixture-provider",
        configured_model="fixture-model",
        allowed_provider_routes=(_route(),),
    )
    request = {"model": "fixture-model", "messages": [{"role": "user", "content": "x"}]}

    _bound_dispatch(handoff, request, lambda kwargs: sdk_calls.append(kwargs))
    with pytest.raises(ProviderAttemptLedgerRequired):
        _bound_dispatch(handoff, request, lambda kwargs: sdk_calls.append(kwargs))

    assert len(bindings) == 1
    assert bindings[0]["providerAttemptId"] == 1
    assert bindings[0]["configuredProvider"] == "fixture-provider"
    assert bindings[0]["configuredModel"] == "fixture-model"
    assert bindings[0]["provider"] == "fixture-provider"
    assert bindings[0]["leafAdapter"] == "tests.fixture.FakeClient"
    assert bindings[0]["fallbackIndex"] == 0
    assert len(sdk_calls) == 1
    assert handoff.attempt_claim_count == 1
    assert outcomes == [
        (
            "response_observed",
            bindings[0]["providerAttemptBindingDigest"],
            None,
        )
    ]


def test_bound_attempt_records_actual_fallback_route_and_leaf() -> None:
    bindings: list[dict] = []
    handoff = RequestDispatchHandoff(
        lambda binding, _kwargs: bindings.append(dict(binding)),
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        configured_provider="primary-provider",
        configured_model="primary-model",
        allowed_provider_routes=(
            _route(provider="primary-provider", model="primary-model"),
            _route(
                fallback_index=1,
                provider="fallback-provider",
                model="fallback-model",
                endpoint_identity="https://fallback.example/v1",
            ),
        ),
    )
    handoff.bind_provider_call_identity(
        "77777777-7777-4777-8777-777777777777"
    )
    handoff.commit_and_claim_dispatch(
        lambda _kwargs: "ok",
        provider="fallback-provider",
        api_mode="chat_completions",
        model="fallback-model",
        sdk_method="chat.completions.create",
        leaf_adapter="openai.OpenAI",
        endpoint_identity="https://fallback.example/v1",
        fallback_index=1,
        request_kwargs={"model": "fallback-model", "messages": []},
    )

    assert len(bindings) == 1
    binding = bindings[0]
    assert binding["configuredProvider"] == "primary-provider"
    assert binding["configuredModel"] == "primary-model"
    assert binding["provider"] == "fallback-provider"
    assert binding["model"] == "fallback-model"
    assert binding["fallbackIndex"] == 1
    assert binding["leafAdapter"] == "openai.OpenAI"
    assert binding["endpointIdentity"] == "https://fallback.example/v1"
    assert binding["configuredRouteChainDigest"].startswith("sha256:")


def test_same_request_bytes_use_distinct_fresh_provider_call_bindings() -> None:
    bindings: list[dict] = []
    request = {"model": "fixture-model", "messages": [{"role": "user", "content": "x"}]}

    def new_handoff(call_id: str) -> RequestDispatchHandoff:
        handoff = RequestDispatchHandoff(
            lambda binding, _kwargs: bindings.append(dict(binding)),
            interrupted=lambda: False,
            interrupted_message="request abandoned before dispatch",
            max_attempts=1,
            callback_accepts_attempt_binding=True,
            configured_provider="fixture-provider",
            configured_model="fixture-model",
            allowed_provider_routes=(_route(),),
        )
        handoff.bind_provider_call_identity(call_id)
        return handoff

    first_id = "11111111-1111-4111-8111-111111111111"
    second_id = "22222222-2222-4222-8222-222222222222"
    _bound_dispatch(new_handoff(first_id), request, lambda _kwargs: "first")
    _bound_dispatch(new_handoff(second_id), request, lambda _kwargs: "second")

    assert [binding["providerCallId"] for binding in bindings] == [
        first_id,
        second_id,
    ]
    assert bindings[0]["finalRequestKwargsDigest"] == bindings[1][
        "finalRequestKwargsDigest"
    ]
    assert bindings[0]["providerAttemptBindingDigest"] != bindings[1][
        "providerAttemptBindingDigest"
    ]


def test_provider_call_identity_cannot_be_reused_or_swapped_after_commit() -> None:
    handoff = RequestDispatchHandoff(
        lambda _binding, _kwargs: None,
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        configured_provider="fixture-provider",
        configured_model="fixture-model",
        allowed_provider_routes=(_route(),),
    )
    first_id = "33333333-3333-4333-8333-333333333333"
    handoff.bind_provider_call_identity(first_id)
    with pytest.raises(RuntimeError, match="cannot be reused"):
        handoff.bind_provider_call_identity(first_id)

    final_id = "44444444-4444-4444-8444-444444444444"
    handoff.bind_provider_call_identity(final_id)
    _bound_dispatch(
        handoff,
        {"model": "fixture-model", "messages": []},
        lambda _kwargs: "ok",
    )
    with pytest.raises(RuntimeError, match="already dispatch-owned"):
        handoff.bind_provider_call_identity(
            "55555555-5555-4555-8555-555555555555"
        )


def test_anthropic_bedrock_leaf_without_atomic_boundary_is_rejected() -> None:
    expected_type = type(
        "AnthropicBedrock",
        (),
        {"__module__": "anthropic.lib.bedrock._client"},
    )
    with pytest.raises(FinalProviderBindingUnsupported, match="atomic final serialized"):
        require_authoritative_leaf_adapter(expected_type())


@pytest.mark.parametrize(
    "override",
    [
        {"provider": "mutated-provider"},
        {"model": "mutated-model"},
        {"api_mode": "codex_responses"},
        {"endpoint_identity": "https://other.example/v1"},
        {"fallback_index": 2},
    ],
)
def test_nonconfigured_final_route_fails_before_receipt_or_sdk(override) -> None:
    receipt_calls = 0
    sdk_calls = 0

    def receipt(_binding, _kwargs) -> None:
        nonlocal receipt_calls
        receipt_calls += 1

    def sdk(_kwargs) -> None:
        nonlocal sdk_calls
        sdk_calls += 1

    handoff = RequestDispatchHandoff(
        receipt,
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        configured_provider="primary-provider",
        configured_model="primary-model",
        allowed_provider_routes=(
            _route(provider="primary-provider", model="primary-model"),
            _route(
                fallback_index=1,
                provider="fallback-provider",
                model="fallback-model",
                endpoint_identity="https://fallback.example/v1",
            ),
        ),
    )
    handoff.bind_provider_call_identity(
        "88888888-8888-4888-8888-888888888888"
    )
    call = {
        "provider": "fallback-provider",
        "api_mode": "chat_completions",
        "model": "fallback-model",
        "sdk_method": "chat.completions.create",
        "leaf_adapter": "openai.OpenAI",
        "endpoint_identity": "https://fallback.example/v1",
        "fallback_index": 1,
        "request_kwargs": {"model": "fallback-model", "messages": []},
    }
    call.update(override)

    with pytest.raises(FinalProviderBindingUnsupported):
        handoff.commit_and_claim_dispatch(sdk, **call)

    assert receipt_calls == 0
    assert sdk_calls == 0
    assert handoff.sdk_entry_intent_committed is False


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("agent.gemini_native_adapter", "GeminiNativeClient"),
        ("agent.gemini_cloudcode_adapter", "GeminiCloudCodeClient"),
        ("agent.copilot_acp_client", "CopilotACPClient"),
    ],
)
def test_unsupported_provider_facade_has_no_authoritative_leaf_binding(
    module_name: str,
    class_name: str,
) -> None:
    client_type = type(class_name, (), {"__module__": module_name})
    with pytest.raises(FinalProviderBindingUnsupported):
        require_authoritative_leaf_adapter(client_type())


@pytest.mark.parametrize(
    "client",
    [
        type("UnknownFacade", (), {})(),
        type(
            "OpenAIProxy",
            (type("OpenAI", (), {"__module__": "openai"}),),
            {"__module__": "tests.fixture"},
        )(),
    ],
)
def test_unknown_or_subclass_leaf_is_not_positive_capability(client) -> None:
    with pytest.raises(FinalProviderBindingUnsupported):
        require_authoritative_leaf_adapter(client)


def test_exact_openai_leaf_without_atomic_boundary_is_rejected() -> None:
    from openai import OpenAI

    client = OpenAI(api_key="fixture-key")
    with pytest.raises(FinalProviderBindingUnsupported, match="atomic final serialized"):
        require_authoritative_leaf_adapter(client)


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("openai", "OpenAI"),
        ("openai.lib.azure", "AzureOpenAI"),
        ("anthropic", "Anthropic"),
        ("anthropic.lib.bedrock._client", "AnthropicBedrock"),
        ("botocore.client", "BedrockRuntime"),
    ],
)
def test_known_dynamic_sdk_leaves_are_not_atomic_retrieval_adapters(
    module_name: str,
    class_name: str,
) -> None:
    client_type = type(class_name, (), {"__module__": module_name})
    with pytest.raises(FinalProviderBindingUnsupported, match="atomic final serialized"):
        require_authoritative_leaf_adapter(client_type())


def test_codex_final_callsite_rejects_unbound_facade_before_receipt_or_sdk() -> None:
    from agent.codex_runtime import run_codex_stream

    receipt_calls = 0

    def receipt(_binding, _kwargs) -> None:
        nonlocal receipt_calls
        receipt_calls += 1

    handoff = RequestDispatchHandoff(
        receipt,
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        configured_provider="openai-codex",
        configured_model="gpt-test",
        allowed_provider_routes=(
            _route(
                provider="openai-codex",
                model="gpt-test",
                api_mode="codex_responses",
                endpoint_identity="https://api.openai.com/v1",
            ),
        ),
    )
    client_type = type(
        "GeminiNativeClient",
        (),
        {"__module__": "agent.gemini_native_adapter"},
    )
    client = client_type()
    client.responses = SimpleNamespace(create=MagicMock())
    agent = SimpleNamespace(
        _provider_usage_outer_attempt_tracking=True,
        _interrupt_requested=False,
        provider="openai-codex",
        api_mode="codex_responses",
        model="gpt-test",
        base_url="https://api.openai.com/v1",
        _fallback_index=0,
        _fire_stream_delta=lambda _text: None,
        _fire_reasoning_delta=lambda _text: None,
        _touch_activity=lambda _text: None,
    )

    with pytest.raises(FinalProviderBindingUnsupported):
        run_codex_stream(
            agent,
            {"model": "gpt-test", "input": []},
            client=client,
            on_request_dispatch=handoff,
        )

    assert receipt_calls == 0
    client.responses.create.assert_not_called()
    assert handoff.sdk_entry_intent_committed is False


def test_final_request_digest_binds_all_ordinary_request_values() -> None:
    captured: list[dict] = []

    def digest_for(request: dict) -> str:
        route_model = str(request.get("model") or "fixture-model")
        handoff = RequestDispatchHandoff(
            lambda binding, _kwargs: captured.append(dict(binding)),
            interrupted=lambda: False,
            interrupted_message="request abandoned before dispatch",
            callback_accepts_attempt_binding=True,
            configured_provider="fixture-provider",
            configured_model="fixture-model",
            allowed_provider_routes=(_route(model=route_model),),
        )
        _bound_dispatch(handoff, request, lambda _kwargs: None)
        return captured[-1]["finalRequestKwargsDigest"]

    base = {
        "model": "fixture-model",
        "max_tokens": 1,
        "stream": False,
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "login",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "password": {"type": "string"},
                        "token": {"type": "string"},
                        "secret": {"type": "string"},
                    },
                },
            },
        }],
    }
    digests = {
        digest_for(base),
        digest_for({**base, "max_tokens": 2}),
        digest_for({**base, "model": "other-model"}),
        digest_for({**base, "stream": True}),
        digest_for({**base, "tools": []}),
    }
    assert len(digests) == 5


@pytest.mark.parametrize(
    "forbidden",
    [
        {"api_key": "do-not-bind"},
        {"authorization": "Bearer do-not-bind"},
        {"extra_headers": {"Authorization": "Bearer do-not-bind"}},
    ],
)
def test_transport_secret_in_final_kwargs_blocks_receipt_and_sdk(forbidden) -> None:
    receipt_calls = 0
    sdk_calls = 0

    def receipt(_binding, _kwargs) -> None:
        nonlocal receipt_calls
        receipt_calls += 1

    def sdk(_kwargs) -> None:
        nonlocal sdk_calls
        sdk_calls += 1

    handoff = RequestDispatchHandoff(
        receipt,
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        callback_accepts_attempt_binding=True,
        configured_provider="fixture-provider",
        configured_model="fixture-model",
        allowed_provider_routes=(_route(),),
    )
    request = {"model": "fixture-model", "messages": [], **forbidden}
    with pytest.raises(ValueError, match="forbidden secret field"):
        _bound_dispatch(handoff, request, sdk)

    assert receipt_calls == 0
    assert sdk_calls == 0


def test_receipt_callback_gets_an_isolated_request_projection() -> None:
    callback_kwargs: list[dict] = []
    sdk_kwargs: list[dict] = []

    def mutate_callback_copy(_binding, bound_kwargs) -> None:
        callback_kwargs.append(bound_kwargs)
        bound_kwargs["max_tokens"] = 2

    handoff = RequestDispatchHandoff(
        mutate_callback_copy,
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        callback_accepts_attempt_binding=True,
        configured_provider="fixture-provider",
        configured_model="fixture-model",
        allowed_provider_routes=(_route(),),
    )
    _bound_dispatch(
        handoff,
        {"model": "fixture-model", "messages": [], "max_tokens": 1},
        lambda kwargs: sdk_kwargs.append(kwargs),
    )

    assert callback_kwargs[0]["max_tokens"] == 2
    assert sdk_kwargs[0]["max_tokens"] == 1
    assert handoff.attempt_claim_count == 1


def test_request_mutation_before_commit_blocks_receipt_attempt_and_sdk() -> None:
    receipt_calls = 0
    sdk_calls = 0

    class MutatingProjection:
        def __init__(self) -> None:
            self.calls = 0

        def as_dict(self):
            self.calls += 1
            return {"revision": self.calls}

    def receipt(_binding, _kwargs) -> None:
        nonlocal receipt_calls
        receipt_calls += 1

    def sdk(_kwargs) -> None:
        nonlocal sdk_calls
        sdk_calls += 1

    handoff = RequestDispatchHandoff(
        receipt,
        interrupted=lambda: False,
        interrupted_message="request abandoned before dispatch",
        callback_accepts_attempt_binding=True,
        configured_provider="fixture-provider",
        configured_model="fixture-model",
        allowed_provider_routes=(_route(),),
    )
    with pytest.raises(RuntimeError, match="mutated before attempt commitment"):
        _bound_dispatch(
            handoff,
            {
                "model": "fixture-model",
                "messages": [],
                "provider_option": MutatingProjection(),
            },
            sdk,
        )
    assert receipt_calls == 0
    assert sdk_calls == 0
    assert handoff.attempt_claim_count == 0
    assert handoff.sdk_entry_intent_committed is False


def test_malformed_live_endpoint_cannot_fall_back_to_configured_identity() -> None:
    client = SimpleNamespace(base_url="not-a-valid-provider-endpoint")
    with pytest.raises(FinalProviderBindingUnsupported, match="scheme"):
        provider_endpoint_identity(
            client,
            provider="openrouter",
            configured_base_url="https://openrouter.ai/api/v1",
        )


def test_ipv6_endpoint_identity_preserves_bracketed_authority() -> None:
    from agent.request_dispatch import canonical_endpoint_identity

    assert canonical_endpoint_identity(
        "http://[::1]:8080/v1/",
        provider="fixture",
    ) == "http://[::1]:8080/v1"
    assert canonical_endpoint_identity(
        "https://[2001:db8::1]/v1",
        provider="fixture",
    ) == "https://[2001:db8::1]/v1"
    assert canonical_endpoint_identity(
        "http://[::1]:8080/v1",
        provider="fixture",
    ) != canonical_endpoint_identity(
        "http://[::1]:8081/v1",
        provider="fixture",
    )


def test_bedrock_leaf_without_atomic_boundary_is_rejected_before_sdk() -> None:

    agent = SimpleNamespace(
        provider="bedrock",
        model="anthropic.claude-test-v1:0",
        api_mode="bedrock_converse",
        base_url="",
        _bedrock_region="eu-west-1",
        _fallback_chain=[],
    )
    client_type = type(
        "BedrockRuntime",
        (),
        {"__module__": "botocore.client"},
    )
    client = client_type()
    client.meta = SimpleNamespace(
        endpoint_url="https://bedrock-runtime.eu-west-1.amazonaws.com",
    )
    sdk_call = MagicMock()
    with pytest.raises(FinalProviderBindingUnsupported, match="atomic final serialized"):
        require_authoritative_leaf_adapter(client)
    sdk_call.assert_not_called()
    assert provider_endpoint_identity(client, provider="bedrock") == (
        "https://bedrock-runtime.eu-west-1.amazonaws.com"
    )
    with pytest.raises(
        FinalProviderBindingUnsupported,
        match="does not identify the final SDK endpoint",
    ):
        snapshot_allowed_provider_routes(agent)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://bedrock-runtime.cn-north-1.amazonaws.com.cn",
        "https://bedrock-runtime-fips.us-gov-west-1.amazonaws.com",
        "https://private-bedrock.example.internal",
    ],
)
def test_bedrock_live_endpoint_is_measured_but_never_inferred_from_region(
    endpoint: str,
) -> None:
    client = SimpleNamespace(
        meta=SimpleNamespace(endpoint_url=endpoint),
    )
    assert provider_endpoint_identity(client, provider="bedrock") == endpoint

    agent = SimpleNamespace(
        provider="bedrock",
        model="anthropic.claude-test-v1:0",
        api_mode="bedrock_converse",
        base_url="",
        _bedrock_region="cn-north-1",
        _fallback_chain=[],
    )
    with pytest.raises(
        FinalProviderBindingUnsupported,
        match="does not identify the final SDK endpoint",
    ):
        snapshot_allowed_provider_routes(agent)
