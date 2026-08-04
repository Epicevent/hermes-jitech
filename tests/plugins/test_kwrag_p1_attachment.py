from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.kwrag_slot.manifest import (
    canonical_json_bytes,
    load_component_manifest,
    load_resource_profile,
)
from plugins.kwrag_slot.p1_attachment import (
    P1_IDENTITY,
    P1_IDENTITY_DIGEST,
    P1_PIPELINE_FINGERPRINT,
    HermesP1AttachmentError,
    enabled_p1_status,
    run_p1_attachment_probe,
)


ROOT = Path(__file__).parents[2]
KWRAG_WHEEL = ROOT / "vendor" / "kwrag" / "kwrag_product_service-0.1.0-py3-none-any.whl"
P1_WHEEL = ROOT / "vendor" / "kwrag_p1" / "kwrag_p1_attachment-0.1.2-py3-none-any.whl"
P1_COMPONENT_WHEEL_DIGEST = (
    "sha256:d1ddb673a6dff6518b1be7222f40215a2051136d32703825b4df6e7630eebcd7"
)
P1_COMPONENT_MANIFEST_DIGEST = (
    "sha256:2e104a98a0abf53e696a8a32625f7e81a32ffac1692bcd2da3d8606269adb12c"
)
P1_FACTORY_SOURCE_DIGEST = (
    "sha256:104276b46fa427d741fcf63db87b70d9a6d8a2ad32e63c4a43e87692041ed43e"
)
KWRAG_SOURCE_COMMIT = "49c10212ff12433941cfbe43d95013d1d2f0aebe"
KWRAG_WHEEL_DIGEST = (
    "sha256:f8dd900d0d00775853ee95dfbf15960c9ea7de2711ea5635fe229b06a550fa6f"
)
POSIX_RUNTIME = pytest.mark.skipif(
    os.name != "posix", reason="Linux slot runtime is authoritative"
)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write(path: Path, value: object) -> str:
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    return _digest(raw)


@pytest.fixture(autouse=True)
def _components(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(P1_WHEEL))
    monkeypatch.syspath_prepend(str(KWRAG_WHEEL))
    importlib.invalidate_caches()
    for name in tuple(sys.modules):
        if name == "kwrag" or name.startswith(("kwrag.", "kwrag_p1_attachment")):
            sys.modules.pop(name, None)


def _database(path: Path) -> None:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE turns(turn_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    db.execute("CREATE TABLE turn_mids(turn_id INTEGER NOT NULL, mid TEXT NOT NULL)")
    db.execute(
        "CREATE VIRTUAL TABLE turns_fts USING fts5(turn_id UNINDEXED, text, tokenize='trigram')"
    )
    db.execute(
        "INSERT INTO turns VALUES (1, 'parcel marker alpha arrived through the blue gate')"
    )
    db.execute("INSERT INTO turn_mids VALUES (1, 'marker-positive')")
    db.execute(
        "INSERT INTO turns_fts VALUES (1, 'parcel marker alpha arrived through the blue gate')"
    )
    db.commit()
    db.close()


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    corpus: str = "alpha",
) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    mount = tmp_path / "workspace" / "nas_docs"
    index = mount / "index"
    index.mkdir(parents=True)
    database = index / "room.meta.sqlite"
    _database(database)
    database_digest = _digest(database.read_bytes())
    snapshot = _digest(b"snapshot")
    manifest = {
        "version": 1,
        "release_id": "fixture",
        "corpus_snapshot": snapshot,
        "embedding_fingerprint": _digest(b"fts"),
        "rooms": {
            corpus: {
                "conversation_id": corpus,
                "files": [{"path": "room.meta.sqlite", "sha256": database_digest}],
            }
        },
    }
    manifest_digest = _write(index / "manifest.json", manifest)
    state_root = home / "kwrag-p1-attachment"
    state_root.mkdir(mode=0o700)
    runtime = {
        "schema_version": "kwrag-slot-runtime-binding-v1",
        "mount_root": mount.as_posix(),
        "index_manifest_relative": "index/manifest.json",
        "index_manifest_digest": manifest_digest,
        "receipt_path": (state_root / "operation-receipts.jsonl").as_posix(),
        "pipeline_fingerprint": P1_PIPELINE_FINGERPRINT,
        "max_concurrent": 1,
    }
    runtime_path = state_root / "runtime-binding.json"
    runtime_digest = _write(runtime_path, runtime)
    component = load_component_manifest()
    resource = load_resource_profile()
    attachment_data = {
        "databaseSha256": database_digest,
        "indexManifestDigest": manifest_digest,
        "sourceSnapshotDigest": snapshot,
        "readOnlyAuthorityReceiptDigest": _digest(b"authority"),
        "slotRuntimeBindingDigest": runtime_digest,
    }
    binding = {
        "schema": "agent-runtime-retrieval-binding/v2",
        "proofMode": "attachment_only",
        "enabled": enabled,
        "family": "hermes",
        "instanceId": "oc20-fixture",
        "runtimeProfileDigest": _digest(b"runtime-profile"),
        "containerNasRoot": "/workspace/nas_docs",
        "transport": "in_process",
        "hostPortCount": 0,
        "mountReadOnly": True,
        "componentDigest": component["component_wheel"]["sha256"],
        "contractDigest": component["contract_collection_digest"],
        "resourceProfileDigest": resource["profileDigest"],
        "p1Identity": dict(P1_IDENTITY),
        "attachmentData": attachment_data if enabled else None,
    }
    binding_path = state_root / "binding-v2.json"
    binding_digest = _write(binding_path, binding)
    container_digest, cgroup_digest = _digest(b"container"), _digest(b"cgroup")
    observation = {
        "schema": "agent-runtime-retrieval-headroom/v1",
        "status": "within_required_headroom",
        "observedAt": datetime.now(timezone.utc).isoformat(),
        "ttlSeconds": 300,
        "targetInstanceId": "oc20-fixture",
        "containerIdentityDigest": container_digest,
        "cgroupIdentityDigest": cgroup_digest,
        "profileDigest": resource["profileDigest"],
        "requiredCpuMillicores": resource["cpuReservationMillicores"],
        "requiredMemoryBytes": resource["memoryReservationBytes"],
        "requiredPids": resource["pidsReservation"],
        "containerCpuUsedMillicores": 25,
        "containerMemoryUsedBytes": 64 * 1024 * 1024,
        "containerPidsUsed": 8,
        "hostCpuAvailableMillicores": 2000,
        "hostMemoryAvailableBytes": 4 * 1024**3,
        "hostPidsAvailable": 1024,
    }
    observation["observationDigest"] = _digest(canonical_json_bytes(observation))
    observation_path = state_root / "resource-observation.json"
    _write(observation_path, observation)
    request_path = tmp_path / "request.json"
    _write(
        request_path,
        {
            "schema_version": "kwrag-slot-search-request-v1",
            "query": "parcel marker alpha",
            "request_id": "request-1",
            "operation_id": "operation-1",
            "run_id": "run-1",
            "attempt": 1,
            "max_results": 5,
            "corpus": corpus,
        },
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment.get_hermes_home", lambda: home
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment._workspace_mount", lambda: mount
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment._runtime_identity",
        lambda: (container_digest, cgroup_digest),
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment._OPS_OBSERVATION_UID",
        getattr(os, "geteuid", lambda: 0)(),
    )
    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment.os.statvfs",
        lambda _path: SimpleNamespace(f_flag=getattr(os, "ST_RDONLY", 1)),
        raising=False,
    )
    monkeypatch.setattr("kwrag.slot_mount._require_readonly_mount", lambda _path: None)
    monkeypatch.setenv("JITECH_RETRIEVAL_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv(
        "JITECH_RETRIEVAL_COMPONENT_DIGEST", component["component_wheel"]["sha256"]
    )
    monkeypatch.setenv("JITECH_RETRIEVAL_BINDING_DIGEST", binding_digest)
    monkeypatch.setenv(
        "JITECH_RETRIEVAL_RESOURCE_PROFILE_DIGEST", resource["profileDigest"]
    )
    return {
        "home": home,
        "mount": mount,
        "database": database,
        "state": state_root,
        "runtime": runtime_path,
        "binding": binding_path,
        "resource": observation_path,
        "request": request_path,
        "binding_value": binding,
    }


def _probe(fixture: dict[str, object]) -> dict[str, object]:
    return run_p1_attachment_probe(
        runtime_binding_path=fixture["runtime"],
        p1_binding_path=fixture["binding"],
        resource_observation_path=fixture["resource"],
        request_path=fixture["request"],
        state_root=fixture["state"],
    )


def test_component_wheel_is_reproducible_exact_research_source_with_streaming_adapter() -> (
    None
):
    assert _digest(P1_WHEEL.read_bytes()) == P1_COMPONENT_WHEEL_DIGEST
    manifest = json.loads(
        (ROOT / "vendor" / "kwrag_p1" / "component-manifest.json").read_text()
    )
    assert (
        _digest((ROOT / "vendor" / "kwrag_p1" / "component-manifest.json").read_bytes())
        == P1_COMPONENT_MANIFEST_DIGEST
    )
    assert manifest["researchFactorySourceSha256"] == P1_FACTORY_SOURCE_DIGEST
    assert manifest["databaseHashMode"] == "streaming"
    assert manifest["wholeDatabaseRead"] is False
    with zipfile.ZipFile(P1_WHEEL) as wheel:
        assert (
            _digest(wheel.read("kwrag_p1_attachment/adapter.py"))
            == manifest["adapterSourceSha256"]
        )
        assert (
            _digest(wheel.read("kwrag_p1_attachment/factory.py"))
            == P1_FACTORY_SOURCE_DIGEST
        )


@POSIX_RUNTIME
def test_resource_observation_requires_read_only_ops_authority(tmp_path: Path) -> None:
    from kwrag_p1_attachment import load_authoritative_canonical_mapping

    path = tmp_path / "resource.json"
    path.write_bytes(b"{}")
    path.chmod(0o660)
    with pytest.raises(ValueError, match="file identity"):
        load_authoritative_canonical_mapping(
            path, "resource observation", required_owner_uid=os.geteuid()
        )
    path.chmod(0o640)
    with pytest.raises(ValueError, match="file identity"):
        load_authoritative_canonical_mapping(
            path, "resource observation", required_owner_uid=os.geteuid() + 1
        )
    assert (
        load_authoritative_canonical_mapping(
            path, "resource observation", required_owner_uid=os.geteuid()
        )[0]
        == {}
    )


def test_adapter_budgets_complete_mapped_result_payload() -> None:
    from kwrag.jsonutil import canonical_json_bytes
    from kwrag.operation import map_results
    from kwrag_p1_attachment import KwragFtsPipeline

    raw = [
        {
            "unitId": f"turn:{index}",
            "text": str(index) * 7_900,
            "score": float(index),
            "sourceIds": [f"message-{index}"],
        }
        for index in range(1, 4)
    ]
    pipeline = KwragFtsPipeline.__new__(KwragFtsPipeline)
    pipeline.room = "alpha"
    pipeline.pipeline = SimpleNamespace(search=lambda _query: raw)
    response = pipeline.search("marker", ["alpha"], 10)
    mapped = map_results(response["hits"], 10, {"alpha"})
    assert len(response["hits"]) == 2
    assert len(canonical_json_bytes(mapped).decode("utf-8")) <= 20_000


def test_component_streams_database_digest_without_path_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "fixture.sqlite"
    _database(database)
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (
            (_ for _ in ()).throw(AssertionError("whole DB read"))
            if self == database
            else original(self)
        ),
    )
    from kwrag_p1_attachment.adapter import _open_verified

    pipeline = _open_verified(
        database,
        {
            "databaseSha256": _digest(original(database)),
            "readOnlyAuthorityReceiptDigest": _digest(b"authority"),
            "sourceSnapshotDigest": _digest(b"snapshot"),
        },
        database.stat().st_size,
    )
    try:
        assert pipeline.search("parcel marker")[0]["sourceIds"] == ["marker-positive"]
    finally:
        pipeline.close()


def test_component_stable_readers_normalize_missing_input(tmp_path: Path) -> None:
    from kwrag_p1_attachment import load_canonical_mapping, load_receipt

    missing = (tmp_path / "missing.json").resolve()
    with pytest.raises(ValueError, match="unavailable"):
        load_canonical_mapping(missing, "mapping")
    with pytest.raises(ValueError, match="unavailable"):
        load_receipt(missing, None, "receipt")


def test_component_selects_latest_canonical_receipt(tmp_path: Path) -> None:
    from kwrag_p1_attachment import load_receipt

    first, second = {"attempt": 1}, {"attempt": 2}
    first_raw, second_raw = canonical_json_bytes(first), canonical_json_bytes(second)
    ledger = (tmp_path / "receipts.jsonl").resolve()
    ledger.write_bytes(first_raw + b"\n" + second_raw + b"\n")
    assert load_receipt(ledger, None, "receipt") == second
    assert load_receipt(ledger, _digest(first_raw), "receipt") == first
    ledger.write_bytes(first_raw + b"\n" + first_raw + b"\n")
    assert load_receipt(ledger, _digest(first_raw), "receipt") == first
    ledger.write_bytes(first_raw + b"\n" + b'{"attempt": 2}' + b"\n")
    with pytest.raises(ValueError, match="ledger is invalid"):
        load_receipt(ledger, None, "receipt")


@pytest.mark.parametrize(
    "field,replacement,error",
    [
        ("observedAt", datetime.now().isoformat(), "time is invalid"),
        ("ttlSeconds", "300", "time is invalid"),
        ("containerPidsUsed", True, "measurements are invalid"),
    ],
)
def test_component_rejects_untyped_or_unzoned_resource_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    error: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    observation = json.loads(fixture["resource"].read_text())
    observation[field] = replacement
    observation.pop("observationDigest")
    observation["observationDigest"] = _digest(canonical_json_bytes(observation))
    resource = load_resource_profile()
    from kwrag_p1_attachment import validate_resource_observation

    with pytest.raises(ValueError, match=error):
        validate_resource_observation(
            observation,
            profile_digest=resource["profileDigest"],
            instance_id="oc20-fixture",
            container_identity_digest=_digest(b"container"),
            cgroup_identity_digest=_digest(b"cgroup"),
            cpu_reservation=resource["cpuReservationMillicores"],
            memory_reservation=resource["memoryReservationBytes"],
            pids_reservation=resource["pidsReservation"],
        )


@pytest.mark.parametrize(
    "field,replacement,error",
    [
        ("hostPortCount", False, "approved Hermes P1 profile"),
        ("hostPortCount", 0.0, "approved Hermes P1 profile"),
        ("instanceId", 20, "target identity is invalid"),
    ],
)
def test_component_rejects_inexact_binding_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    error: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    binding = json.loads(json.dumps(fixture["binding_value"]))
    binding[field] = replacement
    digest = _digest(canonical_json_bytes(binding))
    component = load_component_manifest()
    resource = load_resource_profile()
    from kwrag_p1_attachment import validate_attachment_binding

    with pytest.raises(ValueError, match=error):
        validate_attachment_binding(
            binding,
            digest=digest,
            expected_digest=digest,
            component_digest=component["component_wheel"]["sha256"],
            contract_digest=component["contract_collection_digest"],
            resource_profile_digest=resource["profileDigest"],
            p1_identity=P1_IDENTITY,
        )


@POSIX_RUNTIME
def test_actual_fts_probe_and_restart_status_are_content_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    before = fixture["database"].read_bytes()
    proof = _probe(fixture)
    assert proof["resultStatus"] == "hits" and proof["resultCount"] == 1
    assert proof["consumptionStatus"] == "not_consumed"
    assert proof["providerDispatchAttempted"] is False
    assert fixture["database"].read_bytes() == before
    assert "parcel marker" not in json.dumps(proof)
    status = enabled_p1_status(state_root=fixture["state"])
    assert status["attachmentHealth"] == "healthy"
    assert status["consumptionStatus"] == "not_consumed"
    assert status["p1IdentityDigest"] == P1_IDENTITY_DIGEST
    assert set(status) == set(
        json.loads(
            (
                ROOT
                / "tests"
                / "fixtures"
                / "jitech-embedded-retrieval-attachment-status-v1.valid.json"
            ).read_text()
        )
    )


@POSIX_RUNTIME
def test_repeated_successful_probe_projects_latest_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    first = _probe(fixture)
    request = json.loads(fixture["request"].read_text())
    request.update(
        request_id="request-positive-2",
        operation_id="operation-positive-2",
        run_id="run-positive-2",
    )
    _write(fixture["request"], request)
    second = _probe(fixture)
    assert second["consumptionReceiptDigest"] != first["consumptionReceiptDigest"]
    status = enabled_p1_status(state_root=fixture["state"])
    assert status["consumptionReceiptDigest"] == second["consumptionReceiptDigest"]
    assert status["attachmentHealth"] == "healthy"


@POSIX_RUNTIME
def test_zero_hit_is_linked_without_provider_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    request = json.loads(fixture["request"].read_text())
    request["query"] = "quartz zephyr"
    _write(fixture["request"], request)
    proof = _probe(fixture)
    assert proof["resultStatus"] == "zero_hits" and proof["resultCount"] == 0
    assert proof["providerDispatchAttempted"] is False


@pytest.mark.parametrize(
    "field,replacement,error",
    [
        (
            "observedAt",
            (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "stale, foreign, or mismatched",
        ),
        ("observedAt", datetime.now().isoformat(), "time is invalid"),
        ("ttlSeconds", "300", "time is invalid"),
        ("containerPidsUsed", True, "measurements are invalid"),
        ("targetInstanceId", "oc19", "stale, foreign, or mismatched"),
        (
            "containerIdentityDigest",
            "sha256:" + "9" * 64,
            "stale, foreign, or mismatched",
        ),
        (
            "cgroupIdentityDigest",
            "sha256:" + "9" * 64,
            "stale, foreign, or mismatched",
        ),
    ],
)
@POSIX_RUNTIME
def test_stale_or_foreign_resource_observation_fails_before_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    error: str,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    observation = json.loads(fixture["resource"].read_text())
    observation[field] = replacement
    observation.pop("observationDigest")
    observation["observationDigest"] = _digest(canonical_json_bytes(observation))
    _write(fixture["resource"], observation)
    with pytest.raises(ValueError, match=error):
        _probe(fixture)
    assert not list(fixture["state"].glob("*-receipts.jsonl"))


@POSIX_RUNTIME
def test_low_memory_and_database_tamper_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    observation = json.loads(fixture["resource"].read_text())
    observation["containerMemoryUsedBytes"] = observation["requiredMemoryBytes"] - 1
    observation.pop("observationDigest")
    observation["observationDigest"] = _digest(canonical_json_bytes(observation))
    _write(fixture["resource"], observation)
    with pytest.raises(Exception, match="memory headroom"):
        _probe(fixture)

    fixture = _fixture(tmp_path / "tamper", monkeypatch)
    fixture["database"].write_bytes(fixture["database"].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="mounted index manifest verification failed"):
        _probe(fixture)


@POSIX_RUNTIME
def test_restart_status_rejects_database_binding_and_receipt_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _probe(fixture)
    fixture["database"].write_bytes(fixture["database"].read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="mounted index manifest verification failed"):
        enabled_p1_status(state_root=fixture["state"])

    fixture = _fixture(tmp_path / "receipt", monkeypatch)
    _probe(fixture)
    binding = json.loads(fixture["binding"].read_text())
    binding["instanceId"] = "oc19"
    _write(fixture["binding"], binding)
    with pytest.raises(ValueError, match="fields or digest are invalid"):
        enabled_p1_status(state_root=fixture["state"])


@POSIX_RUNTIME
def test_semantically_corrupt_consumption_receipt_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _probe(fixture)
    ledger = fixture["state"] / "attachment-receipts.jsonl"
    receipt = json.loads(ledger.read_text())
    receipt["consumer_family"] = "openclaw"
    raw = canonical_json_bytes(receipt)
    ledger.write_bytes(raw + b"\n")
    with pytest.raises(ValueError, match="linkage"):
        enabled_p1_status(state_root=fixture["state"])


@pytest.mark.parametrize(
    "target,field,replacement",
    [
        ("operation", "schema_version", "kwrag-slot-search-operation-receipt-v0"),
        ("operation", "authorization_basis", "unbound"),
        ("operation", "pipeline_backend", "different-backend"),
        ("operation", "pipeline_scope", "external"),
        ("operation", "pipeline_stage_id", "other-stage"),
        ("operation", "pipeline_model", "local-model"),
        ("operation", "pipeline_revision", "sha256:" + "9" * 64),
        ("operation", "pipeline_call_count", 0),
        ("operation", "pipeline_candidate_count", 11),
        ("operation", "pipeline_candidate_count", False),
        ("operation", "provider_billing", "charged"),
        ("operation", "corpora", ["x" * 257]),
        ("operation", "extra_field", "unexpected"),
        ("result", "schema_version", "hermes-kwrag-result-receipt-v0"),
        ("result", "consumer_family", "openclaw"),
        ("result", "adapter_status", "unverified"),
        ("result", "extra_field", "unexpected"),
        ("result", "result_characters", False),
        ("result", "result_characters", 20_001),
    ],
)
@POSIX_RUNTIME
def test_self_consistent_semantic_receipt_corruption_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    replacement: object,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _probe(fixture)
    root = fixture["state"]
    operation_path = root / "operation-receipts.jsonl"
    result_path = root / "result-receipts.jsonl"
    consumption_path = root / "attachment-receipts.jsonl"
    operation = json.loads(operation_path.read_text())
    result = json.loads(result_path.read_text())
    consumption = json.loads(consumption_path.read_text())
    if target == "operation":
        if field == "pipeline_backend":
            operation["pipeline_evidence"]["backend_id"] = replacement
        elif field == "pipeline_scope":
            operation["pipeline_evidence"]["stages"][0]["execution_scope"] = replacement
        elif field == "pipeline_candidate_count":
            operation["pipeline_evidence"]["candidate_count"] = replacement
        elif field.startswith("pipeline_"):
            stage_field = field.removeprefix("pipeline_")
            operation["pipeline_evidence"]["stages"][0][stage_field] = replacement
        elif field == "provider_billing":
            operation["provider_billing"]["status"] = replacement
        elif field == "extra_field":
            operation["unexpected"] = replacement
        else:
            operation[field] = replacement
        operation_raw = canonical_json_bytes(operation)
        operation_path.write_bytes(operation_raw + b"\n")
        operation_digest = _digest(operation_raw)
        result["operation_receipt_digest"] = operation_digest
        consumption["operation_receipt_digest"] = operation_digest
    else:
        result[field] = replacement
    result_raw = canonical_json_bytes(result)
    result_path.write_bytes(result_raw + b"\n")
    consumption["result_receipt_digest"] = _digest(result_raw)
    consumption_path.write_bytes(canonical_json_bytes(consumption) + b"\n")
    with pytest.raises(ValueError, match="linkage"):
        enabled_p1_status(state_root=root)


@POSIX_RUNTIME
def test_restart_status_accepts_the_core_maximum_corpus_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, corpus="x" * 256)
    _probe(fixture)
    status = enabled_p1_status(state_root=fixture["state"])
    assert status["attachmentHealth"] == "healthy"
    assert status["linkageStatus"] == "complete"


@POSIX_RUNTIME
def test_result_count_cannot_exceed_requested_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _probe(fixture)
    root = fixture["state"]
    operation_path = root / "operation-receipts.jsonl"
    result_path = root / "result-receipts.jsonl"
    consumption_path = root / "attachment-receipts.jsonl"
    operation = json.loads(operation_path.read_text())
    result = json.loads(result_path.read_text())
    consumption = json.loads(consumption_path.read_text())
    operation["requested_max_results"] = 1
    operation["pipeline_evidence"]["candidate_count"] = 2
    operation["pipeline_evidence"]["returned_count"] = 2
    operation["pipeline_evidence"]["stages"][0]["output_count"] = 2
    for receipt in (operation, result, consumption):
        receipt["result_count"] = 2
    operation_raw = canonical_json_bytes(operation)
    operation_path.write_bytes(operation_raw + b"\n")
    operation_digest = _digest(operation_raw)
    result["operation_receipt_digest"] = operation_digest
    consumption["operation_receipt_digest"] = operation_digest
    result_raw = canonical_json_bytes(result)
    result_path.write_bytes(result_raw + b"\n")
    consumption["result_receipt_digest"] = _digest(result_raw)
    consumption_path.write_bytes(canonical_json_bytes(consumption) + b"\n")
    with pytest.raises(ValueError, match="linkage"):
        enabled_p1_status(state_root=root)


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("result_status", "invented"),
        ("result_count", -1),
        ("result_count", False),
        ("result_digest", "not-a-digest"),
    ],
)
@POSIX_RUNTIME
def test_self_consistent_invalid_result_semantics_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _probe(fixture)
    root = fixture["state"]
    operation_path = root / "operation-receipts.jsonl"
    result_path = root / "result-receipts.jsonl"
    consumption_path = root / "attachment-receipts.jsonl"
    operation = json.loads(operation_path.read_text())
    result = json.loads(result_path.read_text())
    consumption = json.loads(consumption_path.read_text())
    for receipt in (operation, result, consumption):
        receipt[field] = replacement
    operation_raw = canonical_json_bytes(operation)
    operation_path.write_bytes(operation_raw + b"\n")
    operation_digest = _digest(operation_raw)
    result["operation_receipt_digest"] = operation_digest
    consumption["operation_receipt_digest"] = operation_digest
    result_raw = canonical_json_bytes(result)
    result_path.write_bytes(result_raw + b"\n")
    consumption["result_receipt_digest"] = _digest(result_raw)
    consumption_path.write_bytes(canonical_json_bytes(consumption) + b"\n")
    with pytest.raises(ValueError, match="linkage"):
        enabled_p1_status(state_root=root)


@pytest.mark.parametrize("target", ["operation", "result"])
@pytest.mark.parametrize(
    "field,replacement",
    [("result_count", True), ("result_count", 1.0), ("attempt", True)],
)
@POSIX_RUNTIME
def test_each_linked_receipt_requires_exact_identity_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    replacement: object,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _probe(fixture)
    root = fixture["state"]
    operation_path = root / "operation-receipts.jsonl"
    result_path = root / "result-receipts.jsonl"
    consumption_path = root / "attachment-receipts.jsonl"
    operation = json.loads(operation_path.read_text())
    result = json.loads(result_path.read_text())
    consumption = json.loads(consumption_path.read_text())
    receipt = operation if target == "operation" else result
    receipt[field] = replacement
    operation_raw = canonical_json_bytes(operation)
    operation_path.write_bytes(operation_raw + b"\n")
    operation_digest = _digest(operation_raw)
    result["operation_receipt_digest"] = operation_digest
    consumption["operation_receipt_digest"] = operation_digest
    result_raw = canonical_json_bytes(result)
    result_path.write_bytes(result_raw + b"\n")
    consumption["result_receipt_digest"] = _digest(result_raw)
    consumption_path.write_bytes(canonical_json_bytes(consumption) + b"\n")
    with pytest.raises(ValueError, match="linkage"):
        enabled_p1_status(state_root=root)


@POSIX_RUNTIME
def test_identical_replayed_receipts_preserve_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    _probe(fixture)
    for name in (
        "operation-receipts.jsonl",
        "result-receipts.jsonl",
        "attachment-receipts.jsonl",
    ):
        ledger = fixture["state"] / name
        raw = ledger.read_bytes()
        ledger.write_bytes(raw + raw)
    status = enabled_p1_status(state_root=fixture["state"])
    assert status["attachmentHealth"] == "healthy"
    assert status["linkageStatus"] == "complete"


@POSIX_RUNTIME
def test_disabled_v2_status_requires_exact_binding_and_read_only_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, enabled=False)
    _write(fixture["state"] / "binding-v2.json", fixture["binding_value"])
    status = enabled_p1_status(state_root=fixture["state"])
    assert status["enabled"] is False and status["mountReadOnly"] is True
    assert status["p1IdentityDigest"] == P1_IDENTITY_DIGEST
    assert status["attachmentDataDigest"] is None

    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment.os.statvfs",
        lambda _path: SimpleNamespace(f_flag=0),
    )
    with pytest.raises(HermesP1AttachmentError, match="not read-only"):
        enabled_p1_status(state_root=fixture["state"])

    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment.os.statvfs",
        lambda _path: SimpleNamespace(f_flag=getattr(os, "ST_RDONLY", 1)),
    )
    invalid = json.loads(json.dumps(fixture["binding_value"]))
    invalid["componentDigest"] = None
    monkeypatch.setenv(
        "JITECH_RETRIEVAL_BINDING_DIGEST",
        _write(fixture["state"] / "binding-v2.json", invalid),
    )
    with pytest.raises(ValueError, match="approved Hermes P1 profile"):
        enabled_p1_status(state_root=fixture["state"])

    invalid = json.loads(json.dumps(fixture["binding_value"]))
    invalid["schema"] = "agent-runtime-retrieval-binding/v1"
    monkeypatch.setenv(
        "JITECH_RETRIEVAL_BINDING_DIGEST",
        _write(fixture["state"] / "binding-v2.json", invalid),
    )
    with pytest.raises(ValueError, match="approved Hermes P1 profile"):
        enabled_p1_status(state_root=fixture["state"])

    (fixture["state"] / "binding-v2.json").unlink()
    with pytest.raises(ValueError, match="unavailable"):
        enabled_p1_status(state_root=fixture["state"])


@POSIX_RUNTIME
def test_disabled_probe_has_no_backend_or_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch, enabled=False)
    with pytest.raises(HermesP1AttachmentError, match="disabled"):
        _probe(fixture)
    assert not list(fixture["state"].glob("*-receipts.jsonl"))


def test_image_labels_and_status_fixture_bind_exact_product_contract() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    for value in (P1_COMPONENT_WHEEL_DIGEST, P1_COMPONENT_MANIFEST_DIGEST):
        assert value in dockerfile
    manifest = json.loads(
        (ROOT / "vendor/kwrag_p1/component-manifest.json").read_text()
    )
    assert manifest["researchFactorySourceSha256"] == P1_FACTORY_SOURCE_DIGEST
    assert manifest["pipelineFingerprint"] == P1_PIPELINE_FINGERPRINT
    assert manifest["databaseHashMode"] == "streaming"
    assert manifest["wholeDatabaseRead"] is False
    assert (
        'com.epicevent.hermes.kwrag.p1.verify-command.json=\'["hermes","kwrag-slot","p1-attachment-status","--json"]\''
        in dockerfile
    )
    fixture_path = (
        ROOT
        / "tests"
        / "fixtures"
        / "jitech-embedded-retrieval-attachment-status-v1.valid.json"
    )
    fixture = json.loads(fixture_path.read_text())
    assert fixture_path.read_bytes() == canonical_json_bytes(fixture) + b"\n"
    assert fixture["schema"] == "jitech-embedded-retrieval-attachment-status/v1"
    assert fixture["proofMode"] == "attachment_only"
    assert fixture["consumptionStatus"] == "not_consumed"
    assert KWRAG_SOURCE_COMMIT == load_component_manifest()["component_source_revision"]
    assert KWRAG_WHEEL_DIGEST == load_component_manifest()["component_wheel"]["sha256"]
