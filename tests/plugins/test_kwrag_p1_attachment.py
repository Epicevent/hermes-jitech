from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from plugins.kwrag_slot.manifest import canonical_json_bytes, load_component_manifest, load_resource_profile
from plugins.kwrag_slot.p1_attachment import (
    KWRAG_SOURCE_COMMIT,
    KWRAG_WHEEL_DIGEST,
    P1_FACTORY_SOURCE_DIGEST,
    P1_IDENTITY,
    P1_IDENTITY_DIGEST,
    P1_PIPELINE_FINGERPRINT,
    HermesP1AttachmentError,
    enabled_p1_status,
    run_p1_attachment_probe,
)


ROOT = Path(__file__).parents[2]
WHEEL = ROOT / "vendor" / "kwrag" / "kwrag_product_service-0.1.0-py3-none-any.whl"
POSIX_RUNTIME = pytest.mark.skipif(
    os.name != "posix",
    reason="KWRAG slot runtime requires canonical POSIX paths; Linux CI is authoritative",
)


@pytest.fixture(autouse=True)
def _embedded_component_on_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(WHEEL))
    importlib.invalidate_caches()
    for name in tuple(sys.modules):
        if name == "kwrag" or name.startswith("kwrag."):
            sys.modules.pop(name, None)


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _write_canonical(path: Path, value: object) -> str:
    raw = canonical_json_bytes(value)
    path.write_bytes(raw)
    return _digest(raw)


def _make_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE turns(turn_id INTEGER PRIMARY KEY, text TEXT NOT NULL)")
    connection.execute("CREATE TABLE turn_mids(turn_id INTEGER NOT NULL, mid TEXT NOT NULL)")
    connection.execute(
        "CREATE VIRTUAL TABLE turns_fts USING fts5(turn_id UNINDEXED, text, tokenize='trigram')"
    )
    rows = [
        (1, "parcel marker alpha arrived through the blue loading gate", "marker-positive"),
        (2, "sibling room evidence must remain outside the selected room", "sibling-negative"),
    ]
    for turn_id, text, source_id in rows:
        connection.execute("INSERT INTO turns(turn_id,text) VALUES (?,?)", (turn_id, text))
        connection.execute("INSERT INTO turn_mids(turn_id,mid) VALUES (?,?)", (turn_id, source_id))
        connection.execute("INSERT INTO turns_fts(turn_id,text) VALUES (?,?)", (turn_id, text))
    connection.commit()
    connection.close()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path | str]:
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    mount = tmp_path / "workspace" / "nas_docs"
    index = mount / "index"
    index.mkdir(parents=True)
    database = index / "room.meta.sqlite"
    _make_database(database)
    database_digest = _digest(database.read_bytes())
    source_snapshot_digest = _digest(b"hermes-p1-two-record-snapshot")
    manifest = {
        "version": 1,
        "release_id": "hermes-p1-fixture-v1",
        "corpus_snapshot": source_snapshot_digest,
        "embedding_fingerprint": _digest(b"model-free-sqlite-fts5"),
        "rooms": {
            "alpha": {
                "conversation_id": "conversation-alpha",
                "files": [{"path": "room.meta.sqlite", "sha256": database_digest}],
            }
        },
    }
    manifest_path = index / "manifest.json"
    manifest_digest = _write_canonical(manifest_path, manifest)
    state_root = home / "kwrag-p1-attachment"
    runtime = {
        "schema_version": "kwrag-slot-runtime-binding-v1",
        "mount_root": mount.as_posix(),
        "index_manifest_relative": "index/manifest.json",
        "index_manifest_digest": manifest_digest,
        "receipt_path": (state_root / "operation-receipts.jsonl").as_posix(),
        "pipeline_fingerprint": P1_PIPELINE_FINGERPRINT,
        "max_concurrent": 1,
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_digest = _write_canonical(runtime_path, runtime)
    resource = load_resource_profile()
    resource_observation = {
        "containerCpuUsedMillicores": 25,
        "containerMemoryUsedBytes": 64 * 1024 * 1024,
        "containerPidsUsed": 8,
        "hostCpuAvailableMillicores": 2_000,
        "hostMemoryAvailableBytes": 4 * 1024 * 1024 * 1024,
        "hostPidsAvailable": 1_024,
        "profileDigest": resource["profileDigest"],
        "requiredCpuMillicores": resource["cpuReservationMillicores"],
        "requiredMemoryBytes": resource["memoryReservationBytes"],
        "requiredPids": resource["pidsReservation"],
        "schema": "agent-runtime-retrieval-headroom/v1",
        "status": "within_required_headroom",
    }
    resource_observation["observationDigest"] = _digest(
        canonical_json_bytes(resource_observation)
    )
    resource_observation_path = tmp_path / "resource-observation.json"
    _write_canonical(resource_observation_path, resource_observation)
    p1 = {
        "schema_version": "hermes-kwrag-p1-attachment-binding-v1",
        "slot_instance_id": "oc20-fixture",
        "mount_authority_digest": _digest(b"mount-authority"),
        "slot_runtime_binding_digest": runtime_digest,
        "resource_observation_digest": resource_observation["observationDigest"],
        "room": "alpha",
        "database_relative": "room.meta.sqlite",
        "database_sha256": database_digest,
        "source_snapshot_digest": source_snapshot_digest,
        "authority_receipt_digest": _digest(b"read-only-authority-receipt"),
        "p1_identity": dict(P1_IDENTITY),
        "max_result_characters": 20_000,
        "provider_dispatch_required": False,
    }
    p1_path = tmp_path / "p1.json"
    _write_canonical(p1_path, p1)
    request = {
        "schema_version": "kwrag-slot-search-request-v1",
        "query": "parcel marker alpha",
        "request_id": "request-positive-1",
        "operation_id": "operation-positive-1",
        "run_id": "run-positive-1",
        "attempt": 1,
        "max_results": 5,
        "corpus": "alpha",
    }
    request_path = tmp_path / "request.json"
    _write_canonical(request_path, request)

    manifest_info = load_component_manifest()
    monkeypatch.setenv("JITECH_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv(
        "JITECH_RETRIEVAL_COMPONENT_DIGEST",
        manifest_info["component_wheel"]["sha256"],
    )
    monkeypatch.setenv("JITECH_RETRIEVAL_BINDING_DIGEST", _digest(b"ops-binding"))
    monkeypatch.setenv(
        "JITECH_RETRIEVAL_RESOURCE_PROFILE_DIGEST", resource["profileDigest"]
    )
    monkeypatch.setenv("HERMES_WORKSPACE_DIR", (tmp_path / "workspace").as_posix())
    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment.get_hermes_home", lambda: home
    )
    monkeypatch.setattr("kwrag.slot_mount._require_readonly_mount", lambda _path: None)
    monkeypatch.setattr(
        "plugins.kwrag_slot.p1_attachment.os.statvfs",
        lambda _path: SimpleNamespace(f_flag=getattr(os, "ST_RDONLY", 1)),
        raising=False,
    )
    return {
        "home": home,
        "mount": mount,
        "database": database,
        "state_root": state_root,
        "runtime_path": runtime_path,
        "p1_path": p1_path,
        "resource_observation_path": resource_observation_path,
        "request_path": request_path,
        "ops_binding_digest": _digest(b"ops-binding"),
    }


def test_vendored_factory_is_the_exact_research_source_modulo_packaging_lf() -> None:
    from plugins.kwrag_slot import p1_attachment_fts_factory

    raw = Path(p1_attachment_fts_factory.__file__).read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert _digest(raw + b"\n") == P1_FACTORY_SOURCE_DIGEST


@POSIX_RUNTIME
def test_caller_explicit_probe_runs_actual_fts_and_survives_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    before_database = Path(fixture["database"]).read_bytes()
    proof = run_p1_attachment_probe(
        runtime_binding_path=Path(fixture["runtime_path"]),
        p1_binding_path=Path(fixture["p1_path"]),
        resource_observation_path=Path(fixture["resource_observation_path"]),
        request_path=Path(fixture["request_path"]),
        state_root=Path(fixture["state_root"]),
    )
    assert proof["schema"] == "jitech-hermes-kwrag-p1-attachment-proof/v1"
    assert proof["resultStatus"] == "hits"
    assert proof["resultCount"] == 1
    assert proof["consumptionStatus"] == "not_consumed"
    assert proof["providerDispatchRequired"] is False
    assert proof["providerDispatchAttempted"] is False
    assert proof["p1Identity"] == P1_IDENTITY
    assert Path(fixture["database"]).read_bytes() == before_database
    serialized = json.dumps(proof, ensure_ascii=False)
    assert "parcel marker" not in serialized
    assert "marker-positive" not in serialized

    # A fresh status read uses only the durable content-free ledgers.
    status = enabled_p1_status(state_root=Path(fixture["state_root"]))
    assert status["consumerHealth"] == "healthy"
    assert status["linkageStatus"] == "complete"
    assert status["schema"] == "jitech-embedded-retrieval-attachment-status/v1"
    assert status["proofMode"] == "attachment_only"
    assert status["bindingDigest"] == fixture["ops_binding_digest"]
    assert status["consumptionReceiptDigest"] == proof["consumptionReceiptDigest"]
    assert status["consumptionStatus"] == "not_consumed"
    assert status["resourceStatus"] == "within_declared_reservation"
    assert status["p1IdentityDigest"] == P1_IDENTITY_DIGEST
    assert status["operationReceiptDigest"] == proof["operationReceiptDigest"]
    assert status["resultReceiptDigest"] == proof["resultReceiptDigest"]
    fixture_status = json.loads(
        (ROOT / "tests" / "fixtures" / "jitech-embedded-retrieval-attachment-status-v1.valid.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(status) == set(fixture_status)


@pytest.mark.parametrize(
    "active_name,field,replacement",
    [
        ("active-runtime-binding.json", "max_concurrent", 2),
        (
            "active-p1-binding.json",
            "database_sha256",
            "sha256:" + "0" * 64,
        ),
        (
            "active-resource-observation.json",
            "hostMemoryAvailableBytes",
            536_870_911,
        ),
    ],
)
@POSIX_RUNTIME
def test_enabled_status_fails_closed_on_active_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    active_name: str,
    field: str,
    replacement: object,
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    run_p1_attachment_probe(
        runtime_binding_path=Path(fixture["runtime_path"]),
        p1_binding_path=Path(fixture["p1_path"]),
        resource_observation_path=Path(fixture["resource_observation_path"]),
        request_path=Path(fixture["request_path"]),
        state_root=Path(fixture["state_root"]),
    )
    assert enabled_p1_status(state_root=Path(fixture["state_root"]))["consumerHealth"] == "healthy"
    active_path = Path(fixture["state_root"]) / active_name
    changed = json.loads(active_path.read_text(encoding="utf-8"))
    changed[field] = replacement
    _write_canonical(active_path, changed)
    with pytest.raises(HermesP1AttachmentError, match="drift"):
        enabled_p1_status(state_root=Path(fixture["state_root"]))


@POSIX_RUNTIME
def test_zero_hit_is_linked_without_provider_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    request = {
        "schema_version": "kwrag-slot-search-request-v1",
        "query": "nonexistent marker zulu",
        "request_id": "request-zero-1",
        "operation_id": "operation-zero-1",
        "run_id": "run-zero-1",
        "attempt": 1,
        "max_results": 5,
        "corpus": "alpha",
    }
    _write_canonical(Path(fixture["request_path"]), request)
    proof = run_p1_attachment_probe(
        runtime_binding_path=Path(fixture["runtime_path"]),
        p1_binding_path=Path(fixture["p1_path"]),
        resource_observation_path=Path(fixture["resource_observation_path"]),
        request_path=Path(fixture["request_path"]),
        state_root=Path(fixture["state_root"]),
    )
    assert proof["resultStatus"] == "zero_hits"
    assert proof["resultCount"] == 0
    assert proof["providerDispatchAttempted"] is False


@POSIX_RUNTIME
def test_sibling_room_and_tampered_database_fail_without_attachment_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    request_path = Path(fixture["request_path"])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["corpus"] = "sibling"
    _write_canonical(request_path, request)
    with pytest.raises(Exception, match="outside the slot-mounted index"):
        run_p1_attachment_probe(
            runtime_binding_path=Path(fixture["runtime_path"]),
            p1_binding_path=Path(fixture["p1_path"]),
            resource_observation_path=Path(fixture["resource_observation_path"]),
            request_path=request_path,
            state_root=Path(fixture["state_root"]),
        )
    assert not (Path(fixture["state_root"]) / "attachment-receipts.jsonl").exists()

    p1_path = Path(fixture["p1_path"])
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    p1["database_sha256"] = "sha256:" + "0" * 64
    _write_canonical(p1_path, p1)
    request["corpus"] = "alpha"
    _write_canonical(request_path, request)
    with pytest.raises(Exception, match="database digest mismatch"):
        run_p1_attachment_probe(
            runtime_binding_path=Path(fixture["runtime_path"]),
            p1_binding_path=p1_path,
            resource_observation_path=Path(fixture["resource_observation_path"]),
            request_path=request_path,
            state_root=Path(fixture["state_root"]),
        )
    assert not (Path(fixture["state_root"]) / "attachment-receipts.jsonl").exists()


@POSIX_RUNTIME
def test_disabled_probe_stops_before_runtime_or_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("JITECH_RETRIEVAL_ENABLED", "false")
    with pytest.raises(HermesP1AttachmentError, match="disabled"):
        run_p1_attachment_probe(
            runtime_binding_path=Path(fixture["runtime_path"]),
            p1_binding_path=Path(fixture["p1_path"]),
            resource_observation_path=Path(fixture["resource_observation_path"]),
            request_path=Path(fixture["request_path"]),
            state_root=Path(fixture["state_root"]),
        )
    assert not Path(fixture["state_root"]).exists()


@POSIX_RUNTIME
def test_budget_and_resource_claims_are_exactly_bound_before_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    p1_path = Path(fixture["p1_path"])
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    p1["max_result_characters"] = 20_001
    _write_canonical(p1_path, p1)
    with pytest.raises(HermesP1AttachmentError, match="selected factory"):
        run_p1_attachment_probe(
            runtime_binding_path=Path(fixture["runtime_path"]),
            p1_binding_path=p1_path,
            resource_observation_path=Path(fixture["resource_observation_path"]),
            request_path=Path(fixture["request_path"]),
            state_root=Path(fixture["state_root"]),
        )
    assert not Path(fixture["state_root"]).exists()

    resource_root = tmp_path / "resource-catch"
    resource_root.mkdir()
    fixture = _fixture(resource_root, monkeypatch)
    observation_path = Path(fixture["resource_observation_path"])
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    observation["containerMemoryUsedBytes"] = observation["requiredMemoryBytes"] + 1
    body = dict(observation)
    body.pop("observationDigest")
    observation["observationDigest"] = _digest(canonical_json_bytes(body))
    _write_canonical(observation_path, observation)
    p1_path = Path(fixture["p1_path"])
    p1 = json.loads(p1_path.read_text(encoding="utf-8"))
    p1["resource_observation_digest"] = observation["observationDigest"]
    _write_canonical(p1_path, p1)
    with pytest.raises(HermesP1AttachmentError, match="exceeds"):
        run_p1_attachment_probe(
            runtime_binding_path=Path(fixture["runtime_path"]),
            p1_binding_path=p1_path,
            resource_observation_path=observation_path,
            request_path=Path(fixture["request_path"]),
            state_root=Path(fixture["state_root"]),
        )
    assert not Path(fixture["state_root"]).exists()


@POSIX_RUNTIME
def test_enabled_status_uses_persisted_attachment_and_exact_mount_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    run_p1_attachment_probe(
        runtime_binding_path=Path(fixture["runtime_path"]),
        p1_binding_path=Path(fixture["p1_path"]),
        resource_observation_path=Path(fixture["resource_observation_path"]),
        request_path=Path(fixture["request_path"]),
        state_root=Path(fixture["state_root"]),
    )
    assert enabled_p1_status(state_root=Path(fixture["state_root"]))["consumerHealth"] == "healthy"


@POSIX_RUNTIME
def test_enabled_status_fails_closed_after_database_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, monkeypatch)
    run_p1_attachment_probe(
        runtime_binding_path=Path(fixture["runtime_path"]),
        p1_binding_path=Path(fixture["p1_path"]),
        resource_observation_path=Path(fixture["resource_observation_path"]),
        request_path=Path(fixture["request_path"]),
        state_root=Path(fixture["state_root"]),
    )
    database = Path(fixture["database"])
    database.write_bytes(database.read_bytes() + b"tamper")
    with pytest.raises(HermesP1AttachmentError, match="database digest mismatch"):
        enabled_p1_status(state_root=Path(fixture["state_root"]))


def test_image_labels_bind_p1_candidate_without_selecting_product_policy() -> None:
    dockerfile = (Path(__file__).parents[2] / "Dockerfile").read_text(encoding="utf-8")
    expected = {
        "status": P1_IDENTITY["status"],
        "backend-id": P1_IDENTITY["backendId"],
        "factory-source-digest": P1_FACTORY_SOURCE_DIGEST,
        "pipeline-factory-digest": P1_IDENTITY["pipelineFactoryDigest"],
        "pipeline-fingerprint": P1_IDENTITY["pipelineFingerprint"],
        "research-decision-digest": P1_IDENTITY["researchDecisionDigest"],
        "default-enabled": "false",
        "caller-explicit": "true",
        "provider-dispatch-required": "false",
        "status-schema": "jitech-embedded-retrieval-attachment-status/v1",
        "verify-command.json": '["hermes","kwrag-slot","p1-attachment-status","--json"]',
    }
    for suffix, value in expected.items():
        if suffix == "verify-command.json":
            assert f"com.epicevent.hermes.kwrag.p1.{suffix}='{value}'" in dockerfile
        else:
            assert f'com.epicevent.hermes.kwrag.p1.{suffix}="{value}"' in dockerfile
    assert KWRAG_SOURCE_COMMIT == load_component_manifest()["component_source_revision"]
    assert KWRAG_WHEEL_DIGEST == load_component_manifest()["component_wheel"]["sha256"]


def test_attachment_status_fixture_is_canonical_strict_and_content_free() -> None:
    path = ROOT / "tests" / "fixtures" / "jitech-embedded-retrieval-attachment-status-v1.valid.json"
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    fixture = json.loads(raw)
    assert raw == canonical_json_bytes(fixture) + b"\n"
    assert set(fixture) == {
        "schema",
        "proofMode",
        "enabled",
        "componentDigest",
        "bindingDigest",
        "resourceProfileDigest",
        "p1IdentityDigest",
        "attachmentDataDigest",
        "hostPortCount",
        "mountReadOnly",
        "attachmentHealth",
        "resourceStatus",
        "gpuAccessStatus",
        "operationReceiptDigest",
        "resultReceiptDigest",
        "consumptionReceiptDigest",
        "consumptionStatus",
        "linkageStatus",
        "revocationStatus",
    }
    assert fixture["schema"] == "jitech-embedded-retrieval-attachment-status/v1"
    assert fixture["proofMode"] == "attachment_only"
    assert fixture["consumptionStatus"] == "not_consumed"
    assert fixture["p1IdentityDigest"] == P1_IDENTITY_DIGEST
    assert fixture["resourceStatus"] == "within_declared_reservation"
    serialized = raw.decode("utf-8")
    for forbidden in ("query", "resultText", "nasName", "credential", "providerResponse"):
        assert forbidden not in serialized
