"""Hermes product boundary tests for the embedded KWRAG component."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import importlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.kwrag_slot import register
from plugins.kwrag_slot.cli import _status
from plugins.kwrag_slot.manifest import (
    canonical_json_bytes,
    load_component_manifest,
    load_resource_profile,
)


ROOT = Path(__file__).resolve().parents[2]
WHEEL = ROOT / "vendor" / "kwrag" / "kwrag_product_service-0.1.0-py3-none-any.whl"
STATUS_FIXTURES = ROOT / "tests" / "fixtures" / "kwrag_slot"


@pytest.fixture(autouse=True)
def _embedded_component_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(WHEEL))
    importlib.invalidate_caches()
    for name in list(sys.modules):
        if name == "kwrag" or name.startswith("kwrag."):
            sys.modules.pop(name, None)


_SupportedOpenAILeaf = type("OpenAI", (), {"__module__": "openai"})
_SupportedAnthropicLeaf = type("Anthropic", (), {"__module__": "anthropic"})


def _supported_openai_client(
    *,
    base_url: str = "https://openrouter.ai/api/v1",
):
    client = _SupportedOpenAILeaf()
    client.base_url = base_url
    client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=MagicMock())
    )
    client.responses = SimpleNamespace(
        create=MagicMock(),
        stream=MagicMock(),
    )
    return client


def _supported_anthropic_client(
    *,
    base_url: str = "https://api.anthropic.com",
):
    client = _SupportedAnthropicLeaf()
    client.base_url = base_url
    client.messages = SimpleNamespace(
        create=MagicMock(),
        stream=MagicMock(),
    )
    client.close = MagicMock()
    return client


def _test_receipt_sink(path: Path):
    """Use the real POSIX sink; emulate only insert-once on Windows tests."""

    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalError,
    )

    sink = FileConsumptionReceiptSink(path)
    if os.name == "posix":
        return sink
    outcomes: dict[str, bytes] = {}

    def write_once(identity: str, receipt: dict) -> str:
        raw = canonical_json_bytes(receipt)
        existing = outcomes.get(identity)
        if existing is not None and existing != raw:
            raise HermesSlotRetrievalError(
                "provider attempt outcome identity collision"
            )
        outcomes[identity] = raw
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    sink.write_once = write_once  # type: ignore[method-assign]
    return sink


def _request() -> dict:
    return {
        "schema_version": "kwrag-slot-search-request-v1",
        "query": "fixture query",
        "request_id": "request-fixture-1",
        "operation_id": "operation-fixture-1",
        "run_id": "run-fixture-1",
        "attempt": 1,
        "max_results": 2,
        "corpus": "alpha",
    }


def _exchange(results: list[dict] | None = None) -> SimpleNamespace:
    from kwrag.jsonutil import canonical_json_bytes as kwrag_canonical_json_bytes

    if results is None:
        results = [{
            "id": "alpha:segment:7",
            "corpus": "alpha",
            "path": "corpus/123/seg/7",
            "title": "",
            "snippet": "fixture result",
            "score": 0.75,
            "source_ids": ["message-1"],
        }]
    result_status = "hits" if results else "zero_hits"
    result_digest = "sha256:" + hashlib.sha256(kwrag_canonical_json_bytes(results)).hexdigest()
    receipt = {
        "schema_version": "kwrag-slot-search-operation-receipt-v1",
        "recorded_at": "2026-07-28T00:00:00Z",
        "operation_id": "operation-fixture-1",
        "operation_id_source": "client",
        "request_id": "request-fixture-1",
        "run_id": "run-fixture-1",
        "attempt": 1,
        "authorization_basis": "slot_mounted_storage",
        "corpora": ["alpha"],
        "query_chars": 13,
        "requested_max_results": 2,
        "index_manifest": "sha256:" + "a" * 64,
        "pipeline_fingerprint": "sha256:" + "b" * 64,
        "execution_status": "completed",
        "result_status": result_status,
        "duration_ms": 12,
        "result_count": len(results),
        "result_digest": result_digest,
        "pipeline_evidence": {
            "status": "available",
            "schema_version": "kwrag-slot-pipeline-evidence-v1",
            "backend_id": "fixture-backend-v1",
            "stages": [{
                "stage_id": "index_lookup",
                "execution_scope": "slot_local",
                "call_count": 1,
                "input_count": 1,
                "output_count": 1,
                "model": None,
                "revision": None,
            }],
            "candidate_count": len(results),
            "returned_count": len(results),
            "corpus_count": 1,
            "data_boundary": {
                "bytes_sent_outside_slot": 0,
                "external_persistence": "not_applicable",
                "persistence_receipt_digest": None,
            },
        },
        "provider_billing": {
            "status": "not_applicable",
            "amount": None,
            "currency": None,
            "reason": "fully_slot_local_pipeline_has_no_external_provider_receipt",
        },
        "full_economic_cost": {
            "status": "unavailable",
            "amount": None,
            "currency": None,
            "reason": "hardware_energy_depreciation_and_operator_cost_not_measured",
        },
        "error_code": None,
    }
    receipt_digest = "sha256:" + hashlib.sha256(kwrag_canonical_json_bytes(receipt)).hexdigest()
    response = {
        "schema_version": "kwrag-slot-search-response-v1",
        "request_id": "request-fixture-1",
        "operation_id": "operation-fixture-1",
        "run_id": "run-fixture-1",
        "attempt": 1,
        "authorization_basis": "slot_mounted_storage",
        "index_manifest": "sha256:" + "a" * 64,
        "pipeline_fingerprint": "sha256:" + "b" * 64,
        "result_digest": result_digest,
        "result_status": result_status,
        "operation_receipt": {"status": "written", "digest": receipt_digest},
        "results": results,
        "duration_ms": 12,
    }
    return SimpleNamespace(response=response, operation_receipt=receipt)


def _fixture_result_character_budget() -> int:
    results = _exchange().response["results"]
    return len(canonical_json_bytes(results).decode("utf-8"))


def _provider_attempt_binding(**overrides) -> dict:
    binding = {
        "schema": "jitech-provider-sdk-request-attempt-binding/v1",
        "providerAttemptId": 1,
        "configuredProvider": "openrouter",
        "configuredModel": "test/model",
        "provider": "openrouter",
        "apiMode": "chat_completions",
        "model": "test/model",
        "sdkMethod": "chat.completions.create",
        "leafAdapter": "tests.fixture.FakeClient",
        "endpointIdentity": "https://openrouter.ai/api/v1",
        "fallbackIndex": 0,
        "configuredRouteChainDigest": "sha256:" + "e" * 64,
        "finalRequestKwargsDigest": "sha256:" + "d" * 64,
    }
    binding.update(overrides)
    binding["providerAttemptBindingDigest"] = "sha256:" + hashlib.sha256(
        canonical_json_bytes(binding)
    ).hexdigest()
    return binding


def _commit_fake_ephemeral_request(kwargs: dict) -> None:
    binding = _provider_attempt_binding()
    kwargs.pop("ephemeral_user_context_on_request")(binding)
    kwargs.pop("ephemeral_user_context_on_outcome")(
        "response_observed",
        binding["providerAttemptBindingDigest"],
        None,
    )


def _prepared_hits(
    tmp_path: Path,
    filename: str,
    *,
    results: list[dict] | None = None,
):
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
    )

    exchange = _exchange(results)
    result_character_budget = len(
        canonical_json_bytes(exchange.response["results"]).decode("utf-8")
    )
    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": load_component_manifest()["component_wheel"]["sha256"],
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": result_character_budget,
    })

    class Runtime:
        def search_exchange(self, _request):
            return exchange

    receipt_path = tmp_path / filename
    prepared = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        _test_receipt_sink(receipt_path),
    ).search(_request())
    return prepared, receipt_path


@pytest.mark.parametrize(
    "owned_key,owned_value",
    [
        ("persist_user_message", False),
        ("ephemeral_user_context", "caller context"),
        ("ephemeral_user_context_on_request", lambda _binding: None),
        (
            "ephemeral_user_context_on_outcome",
            lambda _status, _binding_digest, _error: None,
        ),
    ],
)
def test_retrieval_seam_rejects_caller_owned_projection_overrides(
    tmp_path: Path,
    owned_key: str,
    owned_value,
) -> None:
    from plugins.kwrag_slot.consumer import HermesSlotRetrievalError
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        f"owned-{owned_key}.jsonl",
    )

    class Agent:
        session_id = "owned-projection-session"

        def run_conversation(self, *_args, **_kwargs):
            raise AssertionError("owned projection override must fail before dispatch")

    with pytest.raises(
        HermesSlotRetrievalError,
        match="user-message projection is owned by the retrieval seam",
    ):
        run_conversation_with_approved_retrieval(
            Agent(),
            "authorized retrieval turn",
            prepared,
            **{owned_key: owned_value},
        )

    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"


def _actual_chat_completions_agent(session_id: str):
    from run_agent import AIAgent

    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            provider="openrouter",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.session_id = session_id
    agent.client = _supported_openai_client()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent._api_max_retries = 1
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _actual_anthropic_agent(session_id: str, *, streaming: bool):
    from run_agent import AIAgent

    old_client = _supported_anthropic_client()
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            return_value=old_client,
        ),
    ):
        agent = AIAgent(
            api_key="sk-ant-oat01-stale-token",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-20250514",
            provider="anthropic",
            api_mode="anthropic_messages",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.session_id = session_id
    agent._anthropic_client = old_client
    agent._anthropic_api_key = "sk-ant-oat01-stale-token"
    agent._anthropic_base_url = "https://api.anthropic.com"
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = not streaming
    agent._api_max_retries = 1
    agent.compression_enabled = False
    agent.save_trajectories = False
    if streaming:
        agent.stream_delta_callback = lambda _text: None
    return agent, old_client


def test_plugin_registers_only_operator_cli() -> None:
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="kwrag_slot"), manager)
    register(ctx)
    assert set(manager._cli_commands) == {"kwrag-slot"}
    assert manager._plugin_tool_names == set()
    assert manager._hooks == {}


def test_bundled_status_cli_is_available_without_enabling_retrieval() -> None:
    plugin_dir = ROOT / "plugins" / "kwrag_slot"
    manifest = PluginManager()._parse_manifest(
        plugin_dir / "plugin.yaml",
        plugin_dir,
        source="bundled",
        prefix="",
    )
    assert manifest is not None
    assert manifest.kind == "backend"


def test_embedded_wheel_and_disabled_status_bind_exact_component(monkeypatch) -> None:
    manifest = load_component_manifest()
    resource = load_resource_profile()
    payload = WHEEL.read_bytes()
    assert len(payload) == manifest["component_wheel"]["bytes"]
    assert "sha256:" + hashlib.sha256(payload).hexdigest() == manifest["component_wheel"]["sha256"]
    monkeypatch.setenv("JITECH_RETRIEVAL_ENABLED", "false")
    monkeypatch.setenv("JITECH_RETRIEVAL_COMPONENT_DIGEST", manifest["component_wheel"]["sha256"])
    monkeypatch.setenv("JITECH_RETRIEVAL_BINDING_DIGEST", "sha256:" + "d" * 64)
    monkeypatch.setenv("JITECH_RETRIEVAL_RESOURCE_PROFILE_DIGEST", resource["profileDigest"])
    monkeypatch.setenv("HERMES_WORKSPACE_DIR", "/workspace")
    monkeypatch.setattr("plugins.kwrag_slot.cli.os.statvfs", lambda _path: SimpleNamespace(f_flag=1), raising=False)
    status = _status()
    assert set(status) == {
        "bindingDigest",
        "componentDigest",
        "consumerHealth",
        "consumptionReceiptDigest",
        "gpuAccessStatus",
        "hostPortCount",
        "linkageStatus",
        "mountReadOnly",
        "operationReceiptDigest",
        "resourceProfileDigest",
        "resourceStatus",
        "resultReceiptDigest",
        "revocationStatus",
        "schema",
    }
    assert status["schema"] == "jitech-embedded-retrieval-status/v1"
    assert status["consumerHealth"] == "disabled"
    assert status["componentDigest"] == manifest["component_wheel"]["sha256"]
    assert status["bindingDigest"] == "sha256:" + "d" * 64
    assert status["resourceProfileDigest"] == resource["profileDigest"]
    assert status["hostPortCount"] == 0
    assert status["mountReadOnly"] is True
    assert status["resourceStatus"] == "unavailable"
    assert status["gpuAccessStatus"] == "none"
    assert status["linkageStatus"] == "not_applicable"
    assert status["revocationStatus"] == "complete"
    assert status["operationReceiptDigest"] is None
    assert status["resultReceiptDigest"] is None
    assert status["consumptionReceiptDigest"] is None


def test_status_fails_closed_before_enabled_product_invocation(monkeypatch) -> None:
    manifest = load_component_manifest()
    resource = load_resource_profile()
    monkeypatch.setenv("JITECH_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("JITECH_RETRIEVAL_COMPONENT_DIGEST", manifest["component_wheel"]["sha256"])
    monkeypatch.setenv("JITECH_RETRIEVAL_BINDING_DIGEST", "sha256:" + "d" * 64)
    monkeypatch.setenv("JITECH_RETRIEVAL_RESOURCE_PROFILE_DIGEST", resource["profileDigest"])
    monkeypatch.setenv("HERMES_WORKSPACE_DIR", "/workspace")
    monkeypatch.setattr("plugins.kwrag_slot.cli.os.statvfs", lambda _path: SimpleNamespace(f_flag=1), raising=False)
    with pytest.raises(RuntimeError, match="approved product invocation"):
        _status()


def test_docker_labels_match_embedded_retrieval_ops_contract() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    manifest = load_component_manifest()
    resource = load_resource_profile()
    expected = {
        "schema": "jitech-embedded-retrieval/v1",
        "component-digest": manifest["component_wheel"]["sha256"],
        "contract-digest": manifest["contract_collection_digest"],
        "component-manifest-digest": manifest["manifest_digest"],
        "source-archive-digest": manifest["component_source_archive"]["sha256"],
        "source-revision": manifest["component_source_revision"],
        "transport": "in_process",
        "default-enabled": "false",
        "host-port-count": "0",
        "nas-read-only": "true",
    }
    prefix = "com.epicevent.agent-runtime.retrieval."
    for suffix, value in expected.items():
        assert f'{prefix}{suffix}="{value}"' in dockerfile
    resource_json = json.dumps(resource, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert f"{prefix}resource.json='{resource_json}'" in dockerfile
    assert f"{prefix}verify-command.json='[\"hermes\",\"kwrag-slot\",\"status\",\"--json\"]'" in dockerfile


def test_enabled_and_disabled_status_fixtures_are_canonical_and_content_free() -> None:
    expected_keys = {
        "bindingDigest",
        "componentDigest",
        "consumerHealth",
        "consumptionReceiptDigest",
        "gpuAccessStatus",
        "hostPortCount",
        "linkageStatus",
        "mountReadOnly",
        "operationReceiptDigest",
        "resourceProfileDigest",
        "resourceStatus",
        "resultReceiptDigest",
        "revocationStatus",
        "schema",
    }
    for name in ("status-disabled.json", "status-enabled.json"):
        raw = (STATUS_FIXTURES / name).read_bytes()
        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        value = json.loads(raw.decode("utf-8"))
        assert set(value) == expected_keys
        assert raw == canonical_json_bytes(value) + b"\n"
        assert value["schema"] == "jitech-embedded-retrieval-status/v1"
        assert "query" not in value and "results" not in value and "prompt" not in value


def test_explicit_consumer_binds_operation_result_and_consumption_receipts(tmp_path: Path) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
    )
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    component_digest = load_component_manifest()["component_wheel"]["sha256"]
    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": component_digest,
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": _fixture_result_character_budget(),
    })

    class Runtime:
        def search_exchange(self, request):
            assert request == _request()
            return _exchange()

    receipt_path = tmp_path / "hermes-consumption.jsonl"
    prepared = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        _test_receipt_sink(receipt_path),
    ).search(_request())
    assert prepared.results[0]["snippet"] == "fixture result"
    receipt = prepared.result_receipt
    assert receipt["component_digest"] == component_digest
    assert receipt["operation_receipt_digest"] == _exchange().response["operation_receipt"]["digest"]
    assert receipt["result_digest"] == _exchange().response["result_digest"]
    assert receipt["adapter_status"] == "verified_by_product_adapter"
    assert receipt["result_status"] == "hits"
    assert receipt["result_characters"] == _fixture_result_character_budget()
    assert "query" not in receipt and "results" not in receipt and "snippet" not in receipt
    assert prepared.result_receipt_digest == (
        "sha256:" + hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    )
    assert prepared.result_receipt_status == "written"
    assert prepared.consumption_receipt_status == "pending"

    class Agent:
        session_id = "session-fixture-1"

        def run_conversation(self, message, **kwargs):
            self.user_message = message
            self.ephemeral_context = kwargs.pop("ephemeral_user_context")
            _commit_fake_ephemeral_request(kwargs)
            self.kwargs = kwargs
            return {"completed": True, "final_response": "fixture answer"}

    agent = Agent()
    outcome = run_conversation_with_approved_retrieval(
        agent,
        "What happened?",
        prepared,
        task_id="task-fixture-1",
    )
    assert outcome["completed"] is True
    assert agent.user_message == "What happened?"
    assert agent.kwargs == {"task_id": "task-fixture-1"}
    assert agent.ephemeral_context.startswith("<kwrag_slot_evidence>\n")
    assert '"snippet":"fixture result"' in agent.ephemeral_context
    assert '"index_manifest":"sha256:' + "a" * 64 + '"' in agent.ephemeral_context
    assert prepared.consumption_receipt_status == "written"
    consumption = prepared.consumption_receipt
    assert consumption is not None
    assert consumption["result_receipt_digest"] == prepared.result_receipt_digest
    assert consumption["index_manifest"] == "sha256:" + "a" * 64
    assert consumption["consumption_status"] == "evidence_dispatch_handoff_committed"
    assert consumption["evidence_projection_status"] == "verified_hits"
    assert consumption["dispatch_handoff_status"] == (
        "evidence_dispatch_handoff_committed"
    )
    assert consumption["transport_outcome_status"] == "unknown"
    assert consumption["provider_attestation_status"] == "unavailable"
    assert consumption["billing_status"] == "unavailable"
    assert "query" not in consumption and "results" not in consumption and "snippet" not in consumption
    assert prepared.consumption_receipt_digest == (
        "sha256:" + hashlib.sha256(canonical_json_bytes(consumption)).hexdigest()
    )
    assert receipt_path.read_bytes() == (
        canonical_json_bytes(receipt) + b"\n" + canonical_json_bytes(consumption) + b"\n"
    )
    assert prepared.content_free_attestation() == {
        "schema": "jitech-hermes-kwrag-consumption-attestation/v1",
        "componentDigest": component_digest,
        "runtimeBindingDigest": "sha256:" + "c" * 64,
        "indexManifestDigest": "sha256:" + "a" * 64,
        "resultStatus": "hits",
        "operationReceiptDigest": receipt["operation_receipt_digest"],
        "resultReceiptDigest": prepared.result_receipt_digest,
            "consumptionReceiptDigest": prepared.consumption_receipt_digest,
            "providerAttemptId": 1,
            "providerAttemptBindingDigest": consumption[
                "provider_attempt_binding_digest"
            ],
            "providerAttemptOutcomeReceiptDigest": (
                prepared.provider_attempt_outcome_receipt_digest
            ),
            "evidenceProjectionStatus": "verified_hits",
            "dispatchHandoffStatus": "evidence_dispatch_handoff_committed",
            "transportOutcomeStatus": "response_observed",
        "providerAttestationStatus": "unavailable",
        "billingStatus": "unavailable",
    }


@pytest.mark.skipif(os.name != "posix", reason="POSIX insert-once contract")
def test_provider_outcome_sink_is_insert_once_and_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalError,
    )

    sink = FileConsumptionReceiptSink(tmp_path / "insert-once.jsonl")
    sink.write({"schema_version": "fixture-ledger-anchor-v1"})
    identity = "sha256:" + "d" * 64
    receipt = {
        "schema_version": "hermes-kwrag-provider-attempt-outcome-receipt-v1",
        "provider_attempt_binding_digest": identity,
        "transport_outcome_status": "response_observed",
    }
    expected = "sha256:" + hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()

    assert sink.write_once(identity, receipt) == expected
    assert sink.write_once(identity, receipt) == expected
    with pytest.raises(HermesSlotRetrievalError, match="identity collision"):
        sink.write_once(
            identity,
            {**receipt, "transport_outcome_status": "sdk_exception"},
        )

    outcome_path = (
        tmp_path
        / "insert-once.jsonl.outcomes"
        / ("d" * 64 + ".json")
    )
    assert outcome_path.read_bytes() == canonical_json_bytes(receipt) + b"\n"


@pytest.mark.skipif(os.name == "posix", reason="non-POSIX fail-closed contract")
def test_provider_outcome_sink_fails_closed_off_posix(tmp_path: Path) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalError,
    )

    sink = FileConsumptionReceiptSink(tmp_path / "non-posix.jsonl")
    sink.write({"schema_version": "fixture-ledger-anchor-v1"})
    with pytest.raises(HermesSlotRetrievalError, match="requires the POSIX"):
        sink.write_once(
            "sha256:" + "0" * 64,
            {"schema_version": "fixture-outcome-v1", "status": "unknown"},
        )


@pytest.mark.skipif(os.name != "posix", reason="POSIX no-follow contract")
def test_provider_outcome_sink_rejects_parent_and_final_symlinks(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalError,
    )

    identity = "sha256:" + "1" * 64
    receipt = {"schema_version": "fixture-outcome-v1", "status": "unknown"}
    trusted_parent = tmp_path / "trusted-parent"
    trusted_parent.mkdir(mode=0o700)
    trusted_parent.chmod(0o700)
    redirected_sink = FileConsumptionReceiptSink(trusted_parent / "sink.jsonl")
    redirected_sink.write({"schema_version": "fixture-ledger-anchor-v1"})
    detached_parent = tmp_path / "detached-trusted-parent"
    trusted_parent.rename(detached_parent)
    redirected_parent = tmp_path / "redirected-parent"
    redirected_parent.mkdir(mode=0o700)
    os.symlink(redirected_parent, trusted_parent, target_is_directory=True)
    with pytest.raises(HermesSlotRetrievalError, match="persisted safely"):
        redirected_sink.write_once(identity, receipt)
    assert list(redirected_parent.iterdir()) == []

    receipt_path = tmp_path / "secure-outcomes.jsonl"
    outcome_root = tmp_path / "secure-outcomes.jsonl.outcomes"
    redirected = tmp_path / "redirected"
    redirected.mkdir(mode=0o700)
    os.symlink(redirected, outcome_root, target_is_directory=True)
    sink = FileConsumptionReceiptSink(receipt_path)
    sink.write({"schema_version": "fixture-ledger-anchor-v1"})

    with pytest.raises(HermesSlotRetrievalError, match="persisted safely"):
        sink.write_once(identity, receipt)
    assert list(redirected.iterdir()) == []

    outcome_root.unlink()
    outcome_root.mkdir(mode=0o700)
    outcome_root.chmod(0o700)
    target = tmp_path / "symlink-target.json"
    target.write_bytes(canonical_json_bytes(receipt) + b"\n")
    target.chmod(0o600)
    os.symlink(target, outcome_root / ("1" * 64 + ".json"))

    with pytest.raises(HermesSlotRetrievalError, match="persisted safely"):
        sink.write_once(identity, receipt)


@pytest.mark.skipif(os.name != "posix", reason="POSIX single-link contract")
def test_provider_outcome_sink_rejects_hardlinked_existing_receipt(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalError,
    )

    identity = "sha256:" + "2" * 64
    receipt = {"schema_version": "fixture-outcome-v1", "status": "unknown"}
    outcome_root = tmp_path / "hardlink.jsonl.outcomes"
    outcome_root.mkdir(mode=0o700)
    outcome_root.chmod(0o700)
    source = tmp_path / "hardlink-source.json"
    source.write_bytes(canonical_json_bytes(receipt) + b"\n")
    source.chmod(0o600)
    os.link(source, outcome_root / ("2" * 64 + ".json"))

    sink = FileConsumptionReceiptSink(tmp_path / "hardlink.jsonl")
    sink.write({"schema_version": "fixture-ledger-anchor-v1"})
    with pytest.raises(HermesSlotRetrievalError, match="persisted safely"):
        sink.write_once(identity, receipt)


@pytest.mark.skipif(os.name != "posix", reason="POSIX directory fsync contract")
def test_provider_outcome_sink_fsyncs_new_directory_and_file_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plugins.kwrag_slot.consumer import FileConsumptionReceiptSink

    fsynced_inodes: list[int] = []
    real_fsync = os.fsync

    def observed_fsync(descriptor: int) -> None:
        fsynced_inodes.append(os.fstat(descriptor).st_ino)
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observed_fsync)
    sink = FileConsumptionReceiptSink(tmp_path / "durable.jsonl")
    sink.write({"schema_version": "fixture-ledger-anchor-v1"})
    sink.write_once(
        "sha256:" + "3" * 64,
        {"schema_version": "fixture-outcome-v1", "status": "unknown"},
    )

    outcome_root = tmp_path / "durable.jsonl.outcomes"
    assert tmp_path.stat().st_ino in fsynced_inodes
    assert outcome_root.stat().st_ino in fsynced_inodes


@pytest.mark.skipif(os.name != "posix", reason="POSIX parent boundary contract")
def test_provider_outcome_sink_requires_preexisting_safe_receipt_parent(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalError,
    )

    missing_sink = FileConsumptionReceiptSink(
        tmp_path / "missing-parent" / "receipts.jsonl"
    )
    with pytest.raises(HermesSlotRetrievalError, match="approved POSIX ledger"):
        missing_sink.write({"schema_version": "fixture-ledger-anchor-v1"})
    assert not (tmp_path / "missing-parent").exists()

    unsafe_parent = tmp_path / "unsafe-parent"
    unsafe_parent.mkdir(mode=0o755)
    unsafe_parent.chmod(0o755)
    unsafe_sink = FileConsumptionReceiptSink(unsafe_parent / "receipts.jsonl")
    with pytest.raises(HermesSlotRetrievalError, match="approved POSIX ledger"):
        unsafe_sink.write({"schema_version": "fixture-ledger-anchor-v1"})
    assert not (unsafe_parent / "receipts.jsonl").exists()


def test_consumer_budgets_complete_canonical_results_with_catch_and_valid_controls(
    tmp_path: Path,
) -> None:
    from kwrag.slot_consumer import SlotConsumptionError
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
    )

    component_digest = load_component_manifest()["component_wheel"]["sha256"]
    full_payload_characters = _fixture_result_character_budget()
    assert full_payload_characters > len("fixture result")
    binding_fields = {
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": component_digest,
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
    }

    class Runtime:
        def search_exchange(self, _request):
            return _exchange()

    too_small = HermesSlotRetrievalBinding.from_mapping({
        **binding_fields,
        "max_result_characters": full_payload_characters - 1,
    })
    with pytest.raises(SlotConsumptionError, match="result character budget is exceeded"):
        HermesSlotRetrievalConsumer(
            too_small,
            Runtime(),
            _test_receipt_sink(tmp_path / "too-small.jsonl"),
        ).search(_request())

    exact = HermesSlotRetrievalBinding.from_mapping({
        **binding_fields,
        "max_result_characters": full_payload_characters,
    })
    prepared = HermesSlotRetrievalConsumer(
        exact,
        Runtime(),
        _test_receipt_sink(tmp_path / "exact.jsonl"),
    ).search(_request())
    assert prepared.result_receipt["result_characters"] == full_payload_characters


def test_disabled_binding_has_no_runtime_or_residual_slot_identity() -> None:
    from plugins.kwrag_slot.consumer import (
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
        HermesSlotRetrievalError,
    )

    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": False,
        "component_digest": load_component_manifest()["component_wheel"]["sha256"],
        "runtime_binding_digest": None,
        "expected_index_manifest": None,
        "expected_pipeline_fingerprint": None,
        "max_result_characters": 0,
    })
    consumer = HermesSlotRetrievalConsumer(binding, None, None)
    with pytest.raises(HermesSlotRetrievalError, match="disabled"):
        consumer.search(_request())


def test_enabled_binding_fails_closed_on_component_or_receipt_drift() -> None:
    from plugins.kwrag_slot.consumer import (
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
        HermesSlotRetrievalError,
    )
    from kwrag.slot_consumer import SlotConsumptionError

    component_digest = load_component_manifest()["component_wheel"]["sha256"]
    base = {
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": component_digest,
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": _fixture_result_character_budget(),
    }
    with pytest.raises(HermesSlotRetrievalError, match="embedded component"):
        HermesSlotRetrievalBinding.from_mapping({**base, "component_digest": "sha256:" + "d" * 64})

    binding = HermesSlotRetrievalBinding.from_mapping(base)

    class DriftedRuntime:
        def search_exchange(self, _request):
            exchange = _exchange()
            exchange.operation_receipt["attempt"] = 2
            return exchange

    with pytest.raises(SlotConsumptionError, match="attempt binding"):
        HermesSlotRetrievalConsumer(binding, DriftedRuntime(), object()).search(_request())


def test_consumption_receipt_sink_digest_mismatch_fails_closed() -> None:
    from plugins.kwrag_slot.consumer import (
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
        HermesSlotRetrievalError,
    )

    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": load_component_manifest()["component_wheel"]["sha256"],
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": _fixture_result_character_budget(),
    })

    class Runtime:
        def search_exchange(self, _request):
            return _exchange()

    class DriftedSink:
        def write(self, _receipt):
            return "sha256:" + "d" * 64

    with pytest.raises(HermesSlotRetrievalError, match="digest is not bound"):
        HermesSlotRetrievalConsumer(binding, Runtime(), DriftedSink()).search(_request())


def test_prompt_consumption_is_single_use_and_requires_session_identity(tmp_path: Path) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
        HermesSlotRetrievalError,
    )
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": load_component_manifest()["component_wheel"]["sha256"],
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": _fixture_result_character_budget(),
    })

    class Runtime:
        def search_exchange(self, _request):
            return _exchange()

    prepared = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        _test_receipt_sink(tmp_path / "receipts.jsonl"),
    ).search(_request())

    class MissingSessionAgent:
        session_id = None

        def run_conversation(self, *_args, **_kwargs):
            raise AssertionError("must not dispatch")

    with pytest.raises(HermesSlotRetrievalError, match="session identity"):
        run_conversation_with_approved_retrieval(MissingSessionAgent(), "question", prepared)
    assert prepared.consumption_receipt_status == "pending"

    class Agent:
        session_id = "session-fixture-2"

        def run_conversation(self, *_args, **kwargs):
            _commit_fake_ephemeral_request(kwargs)
            return {"completed": True}

    run_conversation_with_approved_retrieval(Agent(), "question", prepared)
    with pytest.raises(HermesSlotRetrievalError, match="already consumed"):
        run_conversation_with_approved_retrieval(Agent(), "question", prepared)


def test_codex_app_server_is_rejected_before_consumption_receipt(tmp_path: Path) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
        HermesSlotRetrievalError,
    )
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    receipt_path = tmp_path / "codex-app-server-receipts.jsonl"
    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": load_component_manifest()["component_wheel"]["sha256"],
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": _fixture_result_character_budget(),
    })

    class Runtime:
        def search_exchange(self, _request):
            return _exchange()

    prepared = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        _test_receipt_sink(receipt_path),
    ).search(_request())

    class PersistentAgent:
        api_mode = "codex_app_server"
        session_id = "persistent-thread-session"

        def run_conversation(self, *_args, **_kwargs):
            raise AssertionError("raw evidence must not reach the persistent transport")

    with pytest.raises(HermesSlotRetrievalError, match="persistent codex app-server"):
        run_conversation_with_approved_retrieval(
            PersistentAgent(),
            "authorized retrieval turn",
            prepared,
        )

    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


def test_pre_dispatch_failure_does_not_commit_complete_consumption(tmp_path: Path) -> None:
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    prepared, receipt_path = _prepared_hits(tmp_path, "pre-dispatch-failure.jsonl")

    class FailingAgent:
        session_id = "pre-dispatch-failure-session"

        def run_conversation(self, *_args, **_kwargs):
            raise RuntimeError("failed before first provider request")

    with pytest.raises(RuntimeError, match="before first provider request"):
        run_conversation_with_approved_retrieval(
            FailingAgent(),
            "authorized retrieval turn",
            prepared,
        )

    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"
    attestation = prepared.content_free_attestation()
    assert attestation["dispatchHandoffStatus"] == "not_committed"
    assert attestation["transportOutcomeStatus"] == "not_attempted"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


@pytest.mark.parametrize(
    ("module_name", "class_name", "provider", "base_url"),
    [
        (
            "agent.gemini_native_adapter",
            "GeminiNativeClient",
            "google",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
        (
            "agent.gemini_cloudcode_adapter",
            "GeminiCloudCodeClient",
            "google-gemini-cli",
            "cloudcode-pa://google",
        ),
        (
            "agent.copilot_acp_client",
            "CopilotACPClient",
            "copilot-acp",
            "acp://copilot",
        ),
    ],
)
def test_provider_facades_without_leaf_binding_are_rejected_before_dispatch(
    tmp_path: Path,
    module_name: str,
    class_name: str,
    provider: str,
    base_url: str,
) -> None:
    from plugins.kwrag_slot.consumer import HermesSlotRetrievalError
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, receipt_path = _prepared_hits(
        tmp_path,
        f"unsupported-leaf-{class_name}.jsonl",
    )
    client_type = type(class_name, (), {"__module__": module_name})
    leaf_client = client_type()
    leaf_client.chat = SimpleNamespace(
        completions=SimpleNamespace(create=MagicMock())
    )
    agent = _actual_chat_completions_agent("unsupported-leaf-session")
    agent.provider = provider
    agent.base_url = base_url
    agent.model = "test/model"
    provider_ledger = MagicMock()
    agent._session_db = provider_ledger
    agent._session_db_created = True

    with (
        patch.object(agent, "_create_request_openai_client", return_value=leaf_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch.object(agent, "_try_activate_fallback", return_value=False),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        pytest.raises(
            HermesSlotRetrievalError,
            match="completed before retrieval evidence dispatch",
        ),
    ):
        run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    leaf_client.chat.completions.create.assert_not_called()
    provider_ledger.record_provider_call.assert_not_called()
    assert prepared.consumption_receipt_status == "pending"
    assert prepared.provider_attempt_outcome_status == "pending"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


def test_supported_configured_fallback_binds_only_the_final_evidence_attempt(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "supported-precommit-fallback.jsonl",
    )
    agent = _actual_chat_completions_agent("supported-precommit-fallback-session")
    configured_provider = agent.provider
    configured_model = agent.model
    agent._fallback_chain = [{
        "provider": "fallback-provider",
        "model": "fallback-model",
        "base_url": "https://fallback.example/v1",
        "api_mode": "chat_completions",
    }]
    provider_ledger = MagicMock()
    provider_ledger.record_provider_call.return_value = {
        "ledgerSeq": 1,
        "receiptDigest": "sha256:" + "f" * 64,
    }
    agent._session_db = provider_ledger
    agent._session_db_created = True
    request_client = _supported_openai_client(
        base_url="https://fallback.example/v1"
    )
    request_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="fallback answer",
                tool_calls=None,
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
            ),
            finish_reason="stop",
        )],
        model="fallback-model",
        usage=None,
    )
    fallback_calls = 0

    def activate_configured_fallback(*_args, **_kwargs) -> bool:
        nonlocal fallback_calls
        fallback_calls += 1
        if fallback_calls > 1:
            return False
        agent.provider = "fallback-provider"
        agent.model = "fallback-model"
        agent.api_mode = "chat_completions"
        agent.base_url = "https://fallback.example/v1"
        agent.client = request_client
        agent._fallback_index = 1
        return True

    with (
        patch.object(
            agent,
            "_create_request_openai_client",
            side_effect=[
                RuntimeError("primary client failed before leaf entry"),
                request_client,
            ],
        ) as create_client,
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch.object(
            agent,
            "_try_activate_fallback",
            side_effect=activate_configured_fallback,
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    assert create_client.call_count == 2
    request_client.chat.completions.create.assert_called_once()
    assert fallback_calls == 1
    assert outcome["completed"] is True
    receipt = prepared.consumption_receipt
    assert receipt is not None
    assert receipt["provider_attempt_id"] == 1
    assert receipt["configured_provider"] == configured_provider
    assert receipt["configured_model"] == configured_model
    assert receipt["provider"] == "fallback-provider"
    assert receipt["model"] == "fallback-model"
    assert receipt["fallback_index"] == 1
    assert receipt["leaf_adapter"] == "openai.OpenAI"
    assert receipt["endpoint_identity"] == "https://fallback.example/v1"
    assert receipt["configured_route_chain_digest"].startswith("sha256:")
    assert prepared.provider_attempt_outcome_status == "written"
    provider_ledger.record_provider_call.assert_called_once()
    recorded_call = provider_ledger.record_provider_call.call_args.kwargs
    assert recorded_call["requested_provider"] == "fallback-provider"
    assert recorded_call["requested_model"] == "fallback-model"
    assert recorded_call["attempt"] == 1
    assert recorded_call["fallback_index"] == 1
    assert recorded_call["api_call_index"] == 1
    assert recorded_call["retry_of"] is None
    assert recorded_call["fallback_parent"] is None


def test_committed_evidence_sdk_exception_cannot_fallback_or_dispatch_again(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "committed-sdk-exception-no-fallback.jsonl",
    )
    agent = _actual_chat_completions_agent("committed-sdk-exception-session")
    request_client = _supported_openai_client()
    request_client.chat.completions.create.side_effect = RuntimeError(
        "provider SDK failed after dispatch commitment"
    )

    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_try_recover_primary_transport") as recover,
        patch.object(agent, "_try_activate_fallback") as fallback,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    request_client.chat.completions.create.assert_called_once()
    recover.assert_not_called()
    fallback.assert_not_called()
    assert outcome["completed"] is False
    assert outcome["failed"] is True
    assert outcome["api_calls"] == 1
    assert "RuntimeError" in outcome["error"]
    assert prepared.consumption_receipt_status == "written"
    assert prepared.provider_attempt_outcome_status == "written"
    assert prepared.content_free_attestation()["transportOutcomeStatus"] == (
        "sdk_exception"
    )


@pytest.mark.parametrize(
    "response_kind",
    ["empty", "incomplete_scratchpad", "invalid_tool"],
)
def test_unaccepted_first_evidence_response_cannot_start_a_clean_second_request(
    tmp_path: Path,
    response_kind: str,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        f"unaccepted-first-{response_kind}.jsonl",
    )
    agent = _actual_chat_completions_agent(
        f"unaccepted-first-{response_kind}-session"
    )
    request_client = _supported_openai_client()
    if response_kind == "invalid_tool":
        tool_calls = [SimpleNamespace(
            id="call-invalid",
            type="function",
            function=SimpleNamespace(name="missing_tool", arguments="{}"),
        )]
        content = ""
        finish_reason = "tool_calls"
    elif response_kind == "incomplete_scratchpad":
        tool_calls = None
        content = "<REASONING_SCRATCHPAD>unfinished"
        finish_reason = "stop"
    else:
        tool_calls = None
        content = ""
        finish_reason = "stop"
    request_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content=content,
                tool_calls=tool_calls,
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
            ),
            finish_reason=finish_reason,
        )],
        model="fixture-model",
        usage=None,
    )

    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_try_recover_primary_transport") as recover,
        patch.object(agent, "_try_activate_fallback") as fallback,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    request_client.chat.completions.create.assert_called_once()
    recover.assert_not_called()
    fallback.assert_not_called()
    assert outcome["completed"] is False
    assert outcome["failed"] is True
    assert outcome["api_calls"] == 1
    assert "clean follow-up request is forbidden" in outcome["error"]
    assert prepared.consumption_receipt_status == "written"
    assert prepared.provider_attempt_outcome_status == "written"


def test_tool_response_local_failure_does_not_authorize_clean_follow_up(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "tool-response-local-failure.jsonl",
    )
    agent = _actual_chat_completions_agent("tool-response-local-failure-session")
    agent.valid_tool_names = {"fixture_tool"}
    request_client = _supported_openai_client()
    request_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="",
                tool_calls=[SimpleNamespace(
                    id="call-valid",
                    type="function",
                    function=SimpleNamespace(name="fixture_tool", arguments="{}"),
                )],
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
            ),
            finish_reason="tool_calls",
        )],
        model="fixture-model",
        usage=None,
    )

    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(
            agent,
            "_cap_delegate_task_calls",
            side_effect=RuntimeError("local tool response processing failed"),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    request_client.chat.completions.create.assert_called_once()
    assert outcome["completed"] is False
    assert outcome["failed"] is True
    assert outcome["api_calls"] == 1
    assert "clean follow-up request is forbidden" in outcome["error"]


def test_final_text_local_failure_does_not_authorize_clean_follow_up(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "final-text-local-failure.jsonl",
    )
    agent = _actual_chat_completions_agent("final-text-local-failure-session")
    request_client = _supported_openai_client()
    request_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="valid visible text",
                tool_calls=None,
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
            ),
            finish_reason="stop",
        )],
        model="fixture-model",
        usage=None,
    )

    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(
            agent,
            "_build_assistant_message",
            side_effect=RuntimeError("final message construction failed"),
        ),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    request_client.chat.completions.create.assert_called_once()
    assert outcome["completed"] is False
    assert outcome["failed"] is True
    assert outcome["api_calls"] == 1
    assert "clean follow-up request is forbidden" in outcome["error"]


def test_unaccepted_response_cannot_bypass_guard_via_max_iteration_summary(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "max-iteration-summary-guard.jsonl",
    )
    agent = _actual_chat_completions_agent("max-iteration-summary-guard-session")
    agent.max_iterations = 1
    request_client = _supported_openai_client()
    request_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="",
                tool_calls=None,
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
            ),
            finish_reason="stop",
        )],
        model="fixture-model",
        usage=None,
    )

    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_handle_max_iterations") as summarize,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    request_client.chat.completions.create.assert_called_once()
    summarize.assert_not_called()
    assert outcome["completed"] is False
    assert outcome["failed"] is True
    assert outcome["api_calls"] == 1
    assert "clean follow-up request is forbidden" in outcome["error"]


def test_accepted_tool_response_allows_one_clean_tool_result_follow_up(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "accepted-tool-follow-up.jsonl",
    )
    agent = _actual_chat_completions_agent("accepted-tool-follow-up-session")
    agent.valid_tool_names = {"fixture_tool"}
    request_client = _supported_openai_client()
    observed_requests: list[dict] = []
    tool_call = SimpleNamespace(
        id="call-valid",
        type="function",
        function=SimpleNamespace(name="fixture_tool", arguments="{}"),
    )
    responses = [
        SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[tool_call],
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                ),
                finish_reason="tool_calls",
            )],
            model="fixture-model",
            usage=None,
        ),
        SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(
                    content="final answer",
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                    reasoning_details=None,
                ),
                finish_reason="stop",
            )],
            model="fixture-model",
            usage=None,
        ),
    ]

    def sdk_create(**kwargs):
        observed_requests.append(kwargs)
        return responses.pop(0)

    def execute_tool_calls(message, messages, *_args):
        messages.append({
            "role": "tool",
            "name": message.tool_calls[0].function.name,
            "tool_call_id": message.tool_calls[0].id,
            "content": "fixture tool result",
        })

    request_client.chat.completions.create.side_effect = sdk_create
    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_execute_tool_calls", side_effect=execute_tool_calls),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    assert outcome["completed"] is True
    assert outcome["final_response"] == "final answer"
    assert len(observed_requests) == 2
    first_user = next(
        item for item in reversed(observed_requests[0]["messages"])
        if item.get("role") == "user"
    )
    second_user = next(
        item for item in reversed(observed_requests[1]["messages"])
        if item.get("role") == "user"
    )
    assert "<kwrag_slot_evidence>" in first_user["content"]
    assert "<kwrag_slot_evidence>" not in second_user["content"]
    assert prepared.provider_attempt_outcome_status == "written"


def test_completed_conversation_without_terminal_outcome_fails_closed(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.consumer import HermesSlotRetrievalError
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "completed-without-terminal-outcome.jsonl",
    )

    class MissingOutcomeAgent:
        session_id = "missing-outcome-session"

        def run_conversation(self, _message, **kwargs):
            binding = _provider_attempt_binding()
            kwargs["ephemeral_user_context_on_request"](binding)
            return {"completed": True, "final_response": "unverified success"}

    with pytest.raises(
        HermesSlotRetrievalError,
        match="without a durable provider attempt outcome",
    ):
        run_conversation_with_approved_retrieval(
            MissingOutcomeAgent(),
            "authorized retrieval turn",
            prepared,
        )

    assert prepared.consumption_receipt_status == "written"
    assert prepared.provider_attempt_outcome_status == "pending"
    assert prepared.content_free_attestation()["transportOutcomeStatus"] == "unknown"


def test_actual_aiagent_client_creation_failure_stays_not_consumed(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.consumer import HermesSlotRetrievalError
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    prepared, receipt_path = _prepared_hits(tmp_path, "client-creation-failure.jsonl")
    agent = _actual_chat_completions_agent("client-creation-failure-session")

    with (
        patch.object(
            agent,
            "_create_request_openai_client",
            side_effect=RuntimeError("request client creation failed"),
        ) as create_client,
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch.object(agent, "_try_activate_fallback", return_value=False),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        pytest.raises(
            HermesSlotRetrievalError,
            match="completed before retrieval evidence dispatch",
        ),
    ):
        run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    create_client.assert_called_once()
    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"
    attestation = prepared.content_free_attestation()
    assert attestation["dispatchHandoffStatus"] == "not_committed"
    assert attestation["transportOutcomeStatus"] == "not_attempted"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


def test_actual_aiagent_commits_exactly_once_at_sdk_dispatch_handoff(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    prepared, receipt_path = _prepared_hits(tmp_path, "actual-sdk-dispatch.jsonl")
    agent = _actual_chat_completions_agent("actual-sdk-dispatch-session")
    request_client = _supported_openai_client()
    dispatch_observations: list[str] = []

    def sdk_create(**_kwargs):
        dispatch_observations.append(prepared.consumption_receipt_status)
        message = SimpleNamespace(
            content="answer",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="fixture-model",
            usage=None,
        )

    request_client.chat.completions.create.side_effect = sdk_create
    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    assert outcome["final_response"] == "answer"
    assert dispatch_observations == ["written"]
    request_client.chat.completions.create.assert_called_once()
    assert prepared.consumption_receipt_status == "written"
    lines = receipt_path.read_bytes().splitlines()
    assert len(lines) == 2
    assert lines[0] == canonical_json_bytes(prepared.result_receipt)
    assert lines[1] == canonical_json_bytes(prepared.consumption_receipt)
    assert prepared.provider_attempt_outcome_status == "written"
    assert prepared.content_free_attestation()["transportOutcomeStatus"] == (
        "response_observed"
    )


def test_successful_response_with_outcome_sink_failure_is_not_completed(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "outcome-write-failure-after-response.jsonl",
    )
    agent = _actual_chat_completions_agent("outcome-write-failure-session")
    request_client = _supported_openai_client()
    request_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                content="answer that must not be reported complete",
                tool_calls=None,
                reasoning=None,
                reasoning_content=None,
                reasoning_details=None,
            ),
            finish_reason="stop",
        )],
        model="fixture-model",
        usage=None,
    )
    sink = prepared._receipt_sink
    assert sink is not None
    sink.write_once = MagicMock(side_effect=OSError("outcome fsync failed"))

    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    request_client.chat.completions.create.assert_called_once()
    sink.write_once.assert_called_once()
    assert outcome["completed"] is False
    assert outcome["failed"] is True
    assert outcome["error"] == (
        "retrieval provider-attempt outcome persistence failed"
    )
    assert prepared.consumption_receipt_status == "written"
    assert prepared.provider_attempt_outcome_status == "pending"
    assert prepared.content_free_attestation()["transportOutcomeStatus"] == "unknown"


def test_sdk_exception_identity_survives_product_outcome_sink_failure(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import (
        run_conversation_with_approved_retrieval,
    )

    prepared, _receipt_path = _prepared_hits(
        tmp_path,
        "outcome-write-failure-after-sdk-error.jsonl",
    )
    agent = _actual_chat_completions_agent("sdk-outcome-write-failure-session")
    request_client = _supported_openai_client()
    request_client.chat.completions.create.side_effect = RuntimeError(
        "provider transport exploded"
    )
    sink = prepared._receipt_sink
    assert sink is not None
    sink.write_once = MagicMock(side_effect=OSError("outcome fsync failed"))

    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_try_recover_primary_transport") as recover,
        patch.object(agent, "_try_activate_fallback") as fallback,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    request_client.chat.completions.create.assert_called_once()
    sink.write_once.assert_called_once()
    recover.assert_not_called()
    fallback.assert_not_called()
    assert outcome["completed"] is False
    assert outcome["failed"] is True
    assert "RuntimeError" in outcome["error"]
    assert prepared.consumption_receipt_status == "written"
    assert prepared.provider_attempt_outcome_status == "pending"
    assert prepared.content_free_attestation()["transportOutcomeStatus"] == "unknown"


def test_delayed_stream_client_cannot_dispatch_after_interrupt(
    tmp_path: Path,
) -> None:
    prepared, receipt_path = _prepared_hits(tmp_path, "delayed-client-interrupt.jsonl")
    agent = _actual_chat_completions_agent("delayed-client-interrupt-session")
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda _text: None
    request_client = _supported_openai_client()
    client_creation_started = threading.Event()
    client_creation_finished = threading.Event()
    release_client = threading.Event()

    def delayed_client(**_kwargs):
        client_creation_started.set()
        assert release_client.wait(timeout=5)
        client_creation_finished.set()
        return request_client

    def commit_consumption() -> None:
        prepared.record_prompt_consumption(
            session_binding_digest="sha256:" + "e" * 64,
            prompt_context_digest="sha256:" + "f" * 64,
            provider_attempt_binding=_provider_attempt_binding(),
        )

    errors: list[BaseException] = []
    caller_done = threading.Event()

    def call_provider() -> None:
        try:
            agent._interruptible_streaming_api_call(
                {"model": agent.model, "messages": []},
                on_request_dispatch=commit_consumption,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            caller_done.set()

    with (
        patch.object(agent, "_create_request_openai_client", side_effect=delayed_client),
        patch.object(agent, "_close_request_openai_client"),
    ):
        caller = threading.Thread(target=call_provider, daemon=True)
        caller.start()
        assert client_creation_started.wait(timeout=5)
        agent._interrupt_requested = True
        assert caller_done.wait(timeout=5)
        request_client.chat.completions.create.assert_not_called()
        release_client.set()
        assert client_creation_finished.wait(timeout=2)
        caller.join(timeout=5)
    agent._interrupt_requested = False

    assert len(errors) == 1
    assert isinstance(errors[0], InterruptedError)
    request_client.chat.completions.create.assert_not_called()
    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


def test_dispatch_commitment_wins_interrupt_during_receipt_write(
    tmp_path: Path,
) -> None:
    from agent.request_dispatch import (
        RequestDispatchHandoff,
        snapshot_allowed_provider_routes,
    )

    prepared, receipt_path = _prepared_hits(
        tmp_path,
        "blocking-receipt-write.jsonl",
    )
    agent = _actual_chat_completions_agent("blocking-receipt-write-session")
    agent._disable_streaming = False
    agent.stream_delta_callback = lambda _text: None
    request_client = _supported_openai_client()
    receipt_write_started = threading.Event()
    release_receipt_write = threading.Event()
    sdk_entry = threading.Event()
    abandon_entered = threading.Event()
    caller_done = threading.Event()
    errors: list[BaseException] = []

    sink = prepared._receipt_sink
    assert sink is not None
    sink.write_once = MagicMock(side_effect=OSError("outcome fsync failed"))

    def blocking_receipt_write(binding, _final_kwargs) -> None:
        receipt_write_started.set()
        assert release_receipt_write.wait(timeout=5)
        prepared.record_prompt_consumption(
            session_binding_digest="sha256:" + "e" * 64,
            prompt_context_digest="sha256:" + "f" * 64,
            provider_attempt_binding=binding,
        )

    def persist_outcome(status, binding_digest, error_category) -> None:
        prepared.record_provider_attempt_outcome(
            provider_attempt_binding_digest=binding_digest,
            transport_outcome_status=status,
            error_category=error_category,
        )

    handoff = RequestDispatchHandoff(
        blocking_receipt_write,
        interrupted=lambda: bool(agent._interrupt_requested),
        interrupted_message="test request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        outcome_callback=persist_outcome,
        configured_provider="openrouter",
        configured_model="test/model",
        allowed_provider_routes=snapshot_allowed_provider_routes(agent),
    )
    original_abandon = handoff.abandon

    def observed_abandon(**kwargs) -> bool:
        abandon_entered.set()
        return original_abandon(**kwargs)

    handoff.abandon = observed_abandon  # type: ignore[method-assign]

    def sdk_create(**_kwargs):
        sdk_entry.set()
        return MagicMock()

    request_client.chat.completions.create.side_effect = sdk_create

    def call_provider() -> None:
        try:
            agent._interruptible_streaming_api_call(
                {"model": agent.model, "messages": []},
                on_request_dispatch=handoff,
            )
        except BaseException as exc:
            errors.append(exc)
        finally:
            caller_done.set()

    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client") as close_client,
    ):
        caller = threading.Thread(target=call_provider, daemon=True)
        caller.start()
        assert receipt_write_started.wait(timeout=5)
        agent._interrupt_requested = True
        assert abandon_entered.wait(timeout=5)
        assert caller_done.is_set() is False
        release_receipt_write.set()
        assert caller_done.wait(timeout=5)
        assert handoff.sdk_entry_intent_committed is True
        assert sdk_entry.wait(timeout=5)
        caller.join(timeout=5)
        deadline = time.time() + 2.0
        while time.time() < deadline and close_client.call_count == 0:
            time.sleep(0.02)
        close_client.assert_called()
    agent._interrupt_requested = False

    assert len(errors) == 1
    assert isinstance(errors[0], InterruptedError)
    assert handoff.state == "dispatch_owned"
    assert handoff.sdk_entry_intent_committed is True
    assert handoff.future_attempts_closed is True
    assert handoff.terminal_outcome_status == "interrupted"
    assert isinstance(handoff.outcome_persistence_error, OSError)
    request_client.chat.completions.create.assert_called_once()
    sink.write_once.assert_called_once()
    assert prepared.consumption_receipt_status == "written"
    assert prepared.provider_attempt_outcome_status == "pending"
    attestation = prepared.content_free_attestation()
    assert attestation["dispatchHandoffStatus"] == "evidence_dispatch_handoff_committed"
    assert attestation["transportOutcomeStatus"] == "unknown"
    assert len(receipt_path.read_bytes().splitlines()) == 2


def test_sdk_exception_after_dispatch_commit_keeps_receipt_complete(
    tmp_path: Path,
) -> None:
    prepared, receipt_path = _prepared_hits(
        tmp_path,
        "sdk-error-after-dispatch.jsonl",
    )
    agent = _actual_chat_completions_agent("sdk-error-after-dispatch-session")
    request_client = _supported_openai_client()
    network_started = threading.Event()

    def commit_consumption() -> None:
        prepared.record_prompt_consumption(
            session_binding_digest="sha256:" + "e" * 64,
            prompt_context_digest="sha256:" + "f" * 64,
            provider_attempt_binding=_provider_attempt_binding(),
        )

    def sdk_error(**_kwargs):
        network_started.set()
        raise RuntimeError("provider failed after network start")

    request_client.chat.completions.create.side_effect = sdk_error
    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        pytest.raises(RuntimeError, match="after network start"),
    ):
        agent._interruptible_api_call(
            {"model": agent.model, "messages": []},
            on_request_dispatch=commit_consumption,
        )

    assert network_started.is_set()
    request_client.chat.completions.create.assert_called_once()
    assert prepared.consumption_receipt_status == "written"
    attestation = prepared.content_free_attestation()
    assert attestation["dispatchHandoffStatus"] == "evidence_dispatch_handoff_committed"
    assert attestation["transportOutcomeStatus"] == "unknown"
    assert len(receipt_path.read_bytes().splitlines()) == 2


@pytest.mark.parametrize("streaming", [False, True], ids=["nonstream", "stream"])
def test_anthropic_refresh_build_failure_stays_not_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    from plugins.kwrag_slot.consumer import HermesSlotRetrievalError
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    prepared, receipt_path = _prepared_hits(
        tmp_path,
        f"anthropic-refresh-failure-{streaming}.jsonl",
    )
    agent, old_client = _actual_anthropic_agent(
        f"anthropic-refresh-failure-{streaming}",
        streaming=streaming,
    )
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")

    with (
        patch(
            "agent.anthropic_adapter.resolve_anthropic_token",
            return_value="sk-ant-oat01-fresh-token",
        ),
        patch(
            "agent.anthropic_adapter.build_anthropic_client",
            side_effect=RuntimeError("replacement construction failed"),
        ),
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch.object(agent, "_try_activate_fallback", return_value=False),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        pytest.raises(
            HermesSlotRetrievalError,
            match="completed before retrieval evidence dispatch",
        ),
    ):
        run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    old_client.close.assert_not_called()
    old_client.messages.create.assert_not_called()
    old_client.messages.stream.assert_not_called()
    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


def _anthropic_bound_handoff(agent, receipt_bindings: list[dict]):
    from agent.request_dispatch import (
        RequestDispatchHandoff,
        snapshot_allowed_provider_routes,
    )

    return RequestDispatchHandoff(
        lambda binding, _kwargs: receipt_bindings.append(dict(binding)),
        interrupted=lambda: bool(agent._interrupt_requested),
        interrupted_message="Anthropic request abandoned before dispatch",
        max_attempts=1,
        callback_accepts_attempt_binding=True,
        configured_provider=agent.provider,
        configured_model=agent.model,
        allowed_provider_routes=snapshot_allowed_provider_routes(agent),
    )


def test_anthropic_nonstream_uses_the_exact_leaf_validated_before_commit() -> None:
    agent, original_client = _actual_anthropic_agent(
        "anthropic-nonstream-leaf-snapshot",
        streaming=False,
    )
    replacement_client = _supported_anthropic_client()
    response = SimpleNamespace(id="msg-test", model=agent.model, usage=None)
    original_client.messages.create.return_value = response
    receipts: list[dict] = []
    handoff = _anthropic_bound_handoff(agent, receipts)

    def replace_shared_client(client, *, provider, configured_base_url=None):
        assert client is original_client
        agent._anthropic_client = replacement_client
        return "https://api.anthropic.com"

    with (
        patch.object(agent, "_try_refresh_anthropic_client_credentials"),
        patch(
            "agent.request_dispatch.provider_endpoint_identity",
            side_effect=replace_shared_client,
        ),
    ):
        observed = agent._anthropic_messages_create(
            {"model": agent.model, "messages": [], "max_tokens": 16},
            on_request_dispatch=handoff,
        )

    assert observed is response
    original_client.messages.create.assert_called_once()
    replacement_client.messages.create.assert_not_called()
    assert receipts[0]["leafAdapter"] == "anthropic.Anthropic"


def test_anthropic_stream_uses_the_exact_leaf_validated_before_commit() -> None:
    agent, original_client = _actual_anthropic_agent(
        "anthropic-stream-leaf-snapshot",
        streaming=True,
    )
    replacement_client = _supported_anthropic_client()
    final_message = SimpleNamespace(
        id="msg-stream-test",
        model=agent.model,
        content=[],
        stop_reason="end_turn",
        usage=None,
    )

    class StreamContext:
        response = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(())

        def get_final_message(self):
            return final_message

    original_client.messages.stream.return_value = StreamContext()
    receipts: list[dict] = []
    handoff = _anthropic_bound_handoff(agent, receipts)
    agent._provider_usage_outer_attempt_tracking = True

    def replace_shared_client(_handoff, _agent, client, *, base_url=None):
        assert client is original_client
        agent._anthropic_client = replacement_client
        return "https://api.anthropic.com"

    with (
        patch.object(agent, "_try_refresh_anthropic_client_credentials"),
        patch(
            "agent.chat_completion_helpers._dispatch_endpoint_identity",
            side_effect=replace_shared_client,
        ),
    ):
        observed = agent._interruptible_streaming_api_call(
            {"model": agent.model, "messages": [], "max_tokens": 16},
            on_request_dispatch=handoff,
        )

    assert observed is final_message
    original_client.messages.stream.assert_called_once()
    replacement_client.messages.stream.assert_not_called()
    assert receipts[0]["leafAdapter"] == "anthropic.Anthropic"


@pytest.mark.parametrize("force_ascii", [True, False], ids=["catch", "valid"])
def test_final_request_must_preserve_exact_non_ascii_evidence(
    tmp_path: Path,
    force_ascii: bool,
) -> None:
    from plugins.kwrag_slot.consumer import HermesSlotRetrievalError
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    unicode_results = json.loads(json.dumps(_exchange().response["results"]))
    unicode_results[0]["snippet"] = "승인된 한글 증거"
    prepared, receipt_path = _prepared_hits(
        tmp_path,
        f"non-ascii-evidence-{force_ascii}.jsonl",
        results=unicode_results,
    )
    agent = _actual_chat_completions_agent(f"non-ascii-evidence-{force_ascii}")
    agent._force_ascii_payload = force_ascii
    request_client = _supported_openai_client()
    observed_requests: list[dict] = []

    def sdk_create(**kwargs):
        observed_requests.append(kwargs)
        assert prepared.consumption_receipt_status == "written"
        message = SimpleNamespace(
            content="answer",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="fixture-model",
            usage=None,
        )

    request_client.chat.completions.create.side_effect = sdk_create
    context = (
        pytest.raises(
            HermesSlotRetrievalError,
            match="completed before retrieval evidence dispatch",
        )
        if force_ascii
        else nullcontext()
    )
    with (
        patch.object(agent, "_create_request_openai_client", return_value=request_client),
        patch.object(agent, "_close_request_openai_client"),
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch.object(agent, "_try_activate_fallback", return_value=False),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        context,
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "authorized retrieval turn",
            prepared,
        )

    if force_ascii:
        request_client.chat.completions.create.assert_not_called()
        assert observed_requests == []
        assert prepared.consumption_receipt is None
        assert prepared.consumption_receipt_status == "pending"
        assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"
    else:
        assert outcome["final_response"] == "answer"
        assert len(observed_requests) == 1
        request_messages = observed_requests[0]["messages"]
        assert any(
            "승인된 한글 증거" in str(item.get("content"))
            for item in request_messages
            if isinstance(item, dict) and item.get("role") == "user"
        )
        assert prepared.consumption_receipt_status == "written"


@pytest.mark.parametrize("mutation", ["results", "result_receipt"])
def test_prompt_assembly_rejects_mutated_verified_evidence(
    tmp_path: Path,
    mutation: str,
) -> None:
    from plugins.kwrag_slot.consumer import HermesSlotRetrievalError
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    prepared, receipt_path = _prepared_hits(tmp_path, f"mutated-{mutation}.jsonl")
    original_receipt_bytes = canonical_json_bytes(prepared.result_receipt)
    if mutation == "results":
        prepared.results[0]["snippet"] = "tampered after verification"
        expected_error = "results were mutated"
    else:
        prepared.result_receipt["index_manifest"] = "sha256:" + "d" * 64
        expected_error = "result receipt was mutated"

    class Agent:
        session_id = "mutation-session"

        def run_conversation(self, *_args, **_kwargs):
            raise AssertionError("mutated evidence must not dispatch")

    with pytest.raises(HermesSlotRetrievalError, match=expected_error):
        run_conversation_with_approved_retrieval(
            Agent(),
            "authorized retrieval turn",
            prepared,
        )

    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"
    assert receipt_path.read_bytes() == original_receipt_bytes + b"\n"


def test_consecutive_user_repair_fails_before_consumption_or_dispatch(tmp_path: Path) -> None:
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval
    from run_agent import AIAgent

    prepared, receipt_path = _prepared_hits(tmp_path, "repair-merge-failure.jsonl")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            provider="openrouter",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.session_id = "repair-merge-session"
    agent.client = _supported_openai_client()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = False
    agent.save_trajectories = False

    with (
        patch.object(agent, "_interruptible_api_call") as provider_call,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        pytest.raises(RuntimeError, match="message repair did not preserve exactly one"),
    ):
        run_conversation_with_approved_retrieval(
            agent,
            "current authorized question",
            prepared,
            conversation_history=[{"role": "user", "content": "unclosed prior user"}],
        )

    provider_call.assert_not_called()
    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"
    attestation = prepared.content_free_attestation()
    assert attestation["dispatchHandoffStatus"] == "not_committed"
    assert attestation["transportOutcomeStatus"] == "not_attempted"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


def test_approved_evidence_reaches_actual_aiagent_request_but_not_returned_history(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
    )
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval
    from run_agent import AIAgent

    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": load_component_manifest()["component_wheel"]["sha256"],
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": _fixture_result_character_budget(),
    })

    class Runtime:
        def search_exchange(self, _request):
            return _exchange()

    prepared = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        _test_receipt_sink(tmp_path / "actual-agent-receipts.jsonl"),
    ).search(_request())
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            provider="openrouter",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.session_id = "actual-agent-session"
    agent.client = _supported_openai_client()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = False
    agent.save_trajectories = False
    captured: dict[str, object] = {}
    projected: dict[str, list[dict]] = {}

    def snapshot(name, messages, *_args, **_kwargs):
        projected[name] = json.loads(json.dumps(messages))

    def respond(api_kwargs, **call_kwargs):
        call_kwargs["on_request_dispatch"].commit_and_claim_dispatch(
            lambda bound_kwargs: captured.update(bound_kwargs),
            provider=agent.provider,
            api_mode=agent.api_mode,
            model=api_kwargs["model"],
            sdk_method="chat.completions.create",
            leaf_adapter="tests.fixture.FakeClient",
            endpoint_identity="https://openrouter.ai/api/v1",
            fallback_index=0,
            request_kwargs=api_kwargs,
        )
        message = SimpleNamespace(
            content="answer",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="fixture-model",
            usage=None,
        )

    with (
        patch.object(agent, "_interruptible_api_call", side_effect=respond),
        patch.object(agent, "_save_session_log", side_effect=lambda messages: snapshot("log", messages)),
        patch.object(
            agent,
            "_flush_messages_to_session_db",
            side_effect=lambda messages, history=None: snapshot("db", messages, history),
        ),
        patch.object(
            agent,
            "_save_trajectory",
            side_effect=lambda messages, *_args: snapshot("trajectory", messages),
        ),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "What happened?",
            prepared,
            conversation_history=[
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "tool", "tool_call_id": "stray-tool", "content": "orphan"},
            ],
        )

    request_messages = captured["messages"]
    assert isinstance(request_messages, list)
    user_request = next(
        item for item in reversed(request_messages) if item.get("role") == "user"
    )
    assert "<kwrag_slot_evidence>" in user_request["content"]
    assert '"snippet":"fixture result"' in user_request["content"]
    returned_user = next(
        item for item in reversed(outcome["messages"]) if item.get("role") == "user"
    )
    assert returned_user["content"] == "What happened?"
    assert "fixture result" not in returned_user["content"]
    assert set(projected) == {"db", "log", "trajectory"}
    for messages in projected.values():
        persisted_user = next(
            item for item in reversed(messages) if item.get("role") == "user"
        )
        assert persisted_user["content"] == "What happened?"
        assert "kwrag_slot_evidence" not in persisted_user["content"]
    assert outcome["final_response"] == "answer"
    assert prepared.consumption_receipt_status == "written"


def test_approved_evidence_rebinds_to_current_turn_after_preflight_compression(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
    )
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval
    from run_agent import AIAgent

    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": load_component_manifest()["component_wheel"]["sha256"],
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": _fixture_result_character_budget(),
    })

    class Runtime:
        def search_exchange(self, _request):
            return _exchange()

    prepared = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        _test_receipt_sink(tmp_path / "compressed-turn-receipts.jsonl"),
    ).search(_request())
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            provider="openrouter",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.session_id = "compressed-turn-session"
    agent.client = _supported_openai_client()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent.compression_enabled = True
    agent.context_compressor.protect_first_n = 0
    agent.context_compressor.protect_last_n = 0
    agent.save_trajectories = False
    captured: dict[str, object] = {}
    compression_observed: dict[str, object] = {}

    def compress_with_replacement(messages, _system_message, **_kwargs):
        anchored = [
            message
            for message in messages
            if isinstance(message, dict) and message.get("_hermes_current_turn_anchor")
        ]
        assert len(anchored) == 1
        compression_observed["old_index"] = messages.index(anchored[0])
        return [
            {"role": "assistant", "content": "compressed prior history"},
            anchored[0].copy(),
        ], "You are helpful after compression."

    def respond(api_kwargs, **call_kwargs):
        call_kwargs["on_request_dispatch"].commit_and_claim_dispatch(
            lambda bound_kwargs: captured.update(bound_kwargs),
            provider=agent.provider,
            api_mode=agent.api_mode,
            model=api_kwargs["model"],
            sdk_method="chat.completions.create",
            leaf_adapter="tests.fixture.FakeClient",
            endpoint_identity="https://openrouter.ai/api/v1",
            fallback_index=0,
            request_kwargs=api_kwargs,
        )
        message = SimpleNamespace(
            content="answer after compression",
            tool_calls=None,
            reasoning=None,
            reasoning_content=None,
            reasoning_details=None,
        )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            model="fixture-model",
            usage=None,
        )

    history = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
    ]
    with (
        patch.object(agent.context_compressor, "should_compress", return_value=True),
        patch.object(agent, "_compress_context", side_effect=compress_with_replacement),
        patch.object(agent, "_interruptible_api_call", side_effect=respond),
        patch.object(agent, "_save_session_log"),
        patch.object(agent, "_flush_messages_to_session_db"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        outcome = run_conversation_with_approved_retrieval(
            agent,
            "current authorized question",
            prepared,
            conversation_history=history,
        )

    assert compression_observed["old_index"] == 2
    request_messages = captured["messages"]
    current_request = next(
        message
        for message in reversed(request_messages)
        if message.get("role") == "user"
    )
    assert current_request["content"].startswith("current authorized question")
    assert "<kwrag_slot_evidence>" in current_request["content"]
    assert all("_hermes_current_turn_anchor" not in message for message in request_messages)
    returned_current = next(
        message
        for message in reversed(outcome["messages"])
        if message.get("role") == "user"
    )
    assert returned_current["content"] == "current authorized question"
    assert "kwrag_slot_evidence" not in returned_current["content"]
    assert all("_hermes_current_turn_anchor" not in message for message in outcome["messages"])
    assert prepared.consumption_receipt_status == "written"


def test_zero_hits_are_not_consumed_or_dispatched(tmp_path: Path) -> None:
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
        HermesSlotRetrievalError,
    )
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": load_component_manifest()["component_wheel"]["sha256"],
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": _fixture_result_character_budget(),
    })

    class Runtime:
        def search_exchange(self, _request):
            return _exchange([])

    prepared = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        _test_receipt_sink(tmp_path / "zero-hit-receipts.jsonl"),
    ).search(_request())
    assert prepared.result_receipt["result_status"] == "zero_hits"

    class Agent:
        session_id = "zero-hit-session"

        def __init__(self):
            self.calls = []

        def run_conversation(self, message, **kwargs):
            self.calls.append((message, kwargs))
            return {"completed": True, "messages": [{"role": "user", "content": message}]}

    must_not_dispatch = Agent()
    with pytest.raises(HermesSlotRetrievalError, match="no verified hits"):
        run_conversation_with_approved_retrieval(
            must_not_dispatch,
            "clean question",
            prepared,
        )
    assert must_not_dispatch.calls == []

    assert prepared.consumption_receipt is None
    assert prepared.consumption_receipt_status == "pending"
