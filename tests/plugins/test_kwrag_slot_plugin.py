"""Hermes product boundary tests for the embedded KWRAG component."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _exchange() -> SimpleNamespace:
    from kwrag.jsonutil import canonical_json_bytes as kwrag_canonical_json_bytes

    results = [{
        "id": "alpha:segment:7",
        "corpus": "alpha",
        "path": "corpus/123/seg/7",
        "title": "",
        "snippet": "fixture result",
        "score": 0.75,
        "source_ids": ["message-1"],
    }]
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
        "result_status": "hits",
        "duration_ms": 12,
        "result_count": 1,
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
            "candidate_count": 1,
            "returned_count": 1,
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
        "result_status": "hits",
        "operation_receipt": {"status": "written", "digest": receipt_digest},
        "results": results,
        "duration_ms": 12,
    }
    return SimpleNamespace(response=response, operation_receipt=receipt)


def test_plugin_registers_only_operator_cli() -> None:
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="kwrag_slot"), manager)
    register(ctx)
    assert set(manager._cli_commands) == {"kwrag-slot"}
    assert manager._plugin_tool_names == set()
    assert manager._hooks == {}


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
    monkeypatch.setenv("OPENCLAW_NAS_CONTAINER_PATH", "/workspace/nas_docs")
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
    monkeypatch.setenv("OPENCLAW_NAS_CONTAINER_PATH", "/workspace/nas_docs")
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

    component_digest = load_component_manifest()["component_wheel"]["sha256"]
    binding = HermesSlotRetrievalBinding.from_mapping({
        "schema_version": "hermes-kwrag-slot-binding-v1",
        "enabled": True,
        "component_digest": component_digest,
        "runtime_binding_digest": "sha256:" + "c" * 64,
        "expected_index_manifest": "sha256:" + "a" * 64,
        "expected_pipeline_fingerprint": "sha256:" + "b" * 64,
        "max_result_characters": 100,
    })

    class Runtime:
        def search_exchange(self, request):
            assert request == _request()
            return _exchange()

    receipt_path = tmp_path / "hermes-consumption.jsonl"
    consumed = HermesSlotRetrievalConsumer(
        binding,
        Runtime(),
        FileConsumptionReceiptSink(receipt_path),
    ).search(_request())
    assert consumed.results[0]["snippet"] == "fixture result"
    receipt = consumed.consumption_receipt
    assert receipt["component_digest"] == component_digest
    assert receipt["operation_receipt_digest"] == _exchange().response["operation_receipt"]["digest"]
    assert receipt["result_digest"] == _exchange().response["result_digest"]
    assert receipt["consumption_status"] == "accepted_by_product_adapter"
    assert "query" not in receipt and "results" not in receipt and "snippet" not in receipt
    assert consumed.consumption_receipt_digest == (
        "sha256:" + hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
    )
    assert consumed.consumption_receipt_status == "written"
    assert receipt_path.read_bytes() == canonical_json_bytes(receipt) + b"\n"


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
        "max_result_characters": 100,
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
        "max_result_characters": 100,
    })

    class Runtime:
        def search_exchange(self, _request):
            return _exchange()

    class DriftedSink:
        def write(self, _receipt):
            return "sha256:" + "d" * 64

    with pytest.raises(HermesSlotRetrievalError, match="digest is not bound"):
        HermesSlotRetrievalConsumer(binding, Runtime(), DriftedSink()).search(_request())
