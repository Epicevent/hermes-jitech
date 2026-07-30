"""Hermes product boundary tests for the embedded KWRAG component."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
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


def _prepared_hits(tmp_path: Path, filename: str):
    from plugins.kwrag_slot.consumer import (
        FileConsumptionReceiptSink,
        HermesSlotRetrievalBinding,
        HermesSlotRetrievalConsumer,
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

    receipt_path = tmp_path / filename
    prepared = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        FileConsumptionReceiptSink(receipt_path),
    ).search(_request())
    return prepared, receipt_path


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
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent._disable_streaming = True
    agent._api_max_retries = 1
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


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
        FileConsumptionReceiptSink(receipt_path),
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
            kwargs.pop("ephemeral_user_context_on_request")()
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
    assert consumption["consumption_status"] == "assembled_into_ephemeral_user_context"
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
        "linkageStatus": "complete",
    }


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
            FileConsumptionReceiptSink(tmp_path / "too-small.jsonl"),
        ).search(_request())

    exact = HermesSlotRetrievalBinding.from_mapping({
        **binding_fields,
        "max_result_characters": full_payload_characters,
    })
    prepared = HermesSlotRetrievalConsumer(
        exact,
        Runtime(),
        FileConsumptionReceiptSink(tmp_path / "exact.jsonl"),
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
        FileConsumptionReceiptSink(tmp_path / "receipts.jsonl"),
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
            kwargs.pop("ephemeral_user_context_on_request")()
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
        FileConsumptionReceiptSink(receipt_path),
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
    assert prepared.content_free_attestation()["linkageStatus"] == "not_consumed"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


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
    assert prepared.content_free_attestation()["linkageStatus"] == "not_consumed"
    assert receipt_path.read_bytes() == canonical_json_bytes(prepared.result_receipt) + b"\n"


def test_actual_aiagent_commits_exactly_once_immediately_before_sdk_call(
    tmp_path: Path,
) -> None:
    from plugins.kwrag_slot.prompt_context import run_conversation_with_approved_retrieval

    prepared, receipt_path = _prepared_hits(tmp_path, "actual-sdk-dispatch.jsonl")
    agent = _actual_chat_completions_agent("actual-sdk-dispatch-session")
    request_client = MagicMock()
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
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
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
    assert prepared.content_free_attestation()["linkageStatus"] == "not_consumed"
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
        FileConsumptionReceiptSink(tmp_path / "actual-agent-receipts.jsonl"),
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
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    captured: dict[str, object] = {}
    projected: dict[str, list[dict]] = {}

    def snapshot(name, messages, *_args, **_kwargs):
        projected[name] = json.loads(json.dumps(messages))

    def respond(api_kwargs, **call_kwargs):
        call_kwargs["on_request_dispatch"]()
        captured.update(api_kwargs)
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
        FileConsumptionReceiptSink(tmp_path / "compressed-turn-receipts.jsonl"),
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
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
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
        call_kwargs["on_request_dispatch"]()
        captured.update(api_kwargs)
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
        FileConsumptionReceiptSink(tmp_path / "zero-hit-receipts.jsonl"),
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
