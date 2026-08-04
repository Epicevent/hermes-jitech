"""Caller-explicit SQLite FTS attachment proof for the embedded KWRAG seam.

This module is a canary adapter, not an invocation policy.  It never chooses a
query, exposes a model tool, assembles a prompt, or calls a provider.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from hermes_constants import get_hermes_home
from plugins.kwrag_slot.consumer import (
    FileConsumptionReceiptSink,
    HermesSlotRetrievalBinding,
    HermesSlotRetrievalConsumer,
    HermesSlotRetrievalError,
)
from plugins.kwrag_slot.manifest import (
    canonical_json_bytes,
    load_component_manifest,
    load_resource_profile,
)
from plugins.kwrag_slot.p1_attachment_fts_factory import (
    BACKEND_ID,
    FtsAttachmentPipeline,
    TEXT_CHARACTER_MAXIMUM,
)


KWRAG_SOURCE_COMMIT = "49c10212ff12433941cfbe43d95013d1d2f0aebe"
KWRAG_WHEEL_DIGEST = (
    "sha256:f8dd900d0d00775853ee95dfbf15960c9ea7de2711ea5635fe229b06a550fa6f"
)
P1_FACTORY_SOURCE_DIGEST = (
    "sha256:104276b46fa427d741fcf63db87b70d9a6d8a2ad32e63c4a43e87692041ed43e"
)
P1_FACTORY_DIGEST = (
    "sha256:0dbe54f5a8bc56a6c821e181a0dc6cfda85d25be8cea6a01235cb5e347782f0e"
)
P1_PIPELINE_FINGERPRINT = (
    "sha256:53e14752cc9d147dfb4129e00234d1c7fb9f6558df00da7c03189db8da8e4606"
)
P1_RESEARCH_DECISION_DIGEST = (
    "sha256:81e6f4d83e6cde6a9c83a9aa435c65354a1122dded735bf607462c3497e9b25d"
)
P1_IDENTITY = {
    "status": "research_selected_p1_attachment_probe_candidate",
    "pipelineFactoryDigest": P1_FACTORY_DIGEST,
    "backendId": BACKEND_ID,
    "pipelineFingerprint": P1_PIPELINE_FINGERPRINT,
    "researchDecisionDigest": P1_RESEARCH_DECISION_DIGEST,
}

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,191}$")
_BINDING_FIELDS = {
    "schema_version",
    "slot_instance_id",
    "mount_authority_digest",
    "slot_runtime_binding_digest",
    "resource_observation_digest",
    "room",
    "database_relative",
    "database_sha256",
    "source_snapshot_digest",
    "authority_receipt_digest",
    "p1_identity",
    "max_result_characters",
    "provider_dispatch_required",
}
_ATTACHMENT_FIELDS = {
    "schema_version",
    "consumer_family",
    "attachment_status",
    "consumption_status",
    "provider_dispatch_required",
    "provider_dispatch_attempted",
    "slot_instance_id",
    "ops_binding_digest",
    "slot_runtime_binding_digest",
    "p1_binding_digest",
    "mount_authority_digest",
    "authority_receipt_digest",
    "resource_observation_digest",
    "component_source_revision",
    "component_wheel_digest",
    "p1_identity",
    "request_id",
    "operation_id",
    "run_id",
    "attempt",
    "index_manifest_digest",
    "database_digest",
    "source_snapshot_digest",
    "result_status",
    "result_count",
    "result_digest",
    "operation_receipt_digest",
    "result_receipt_digest",
}
_ACTIVE_SET_FIELDS = {
    "schema",
    "runtime_binding_digest",
    "p1_binding_digest",
    "resource_observation_digest",
    "attachment_receipt_digest",
}
_RESOURCE_OBSERVATION_FIELDS = {
    "containerCpuUsedMillicores",
    "containerMemoryUsedBytes",
    "containerPidsUsed",
    "hostCpuAvailableMillicores",
    "hostMemoryAvailableBytes",
    "hostPidsAvailable",
    "profileDigest",
    "requiredCpuMillicores",
    "requiredMemoryBytes",
    "requiredPids",
    "schema",
    "status",
    "observationDigest",
}


class HermesP1AttachmentError(HermesSlotRetrievalError):
    """The caller-explicit P1 attachment contract could not be verified."""


def _digest(value: object, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise HermesP1AttachmentError(f"{field} is not a canonical SHA-256 digest")
    return text


def _identifier(value: object, field: str, *, maximum: int = 192) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise HermesP1AttachmentError(f"{field} is invalid")
    if maximum == 192 and not _IDENTIFIER.fullmatch(value):
        raise HermesP1AttachmentError(f"{field} is invalid")
    return value


def _relative(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise HermesP1AttachmentError(f"{field} is invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise HermesP1AttachmentError(f"{field} is not a safe relative POSIX path")
    return value


def _canonical_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


P1_IDENTITY_DIGEST = _canonical_digest(P1_IDENTITY)


def _read_canonical_mapping(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    source = Path(path)
    if not source.is_absolute() or source.is_symlink():
        raise HermesP1AttachmentError(f"{label} must be an absolute regular file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise HermesP1AttachmentError(f"{label} could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise HermesP1AttachmentError(f"{label} must be a single-link regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 16 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > 64 * 1024:
                raise HermesP1AttachmentError(f"{label} is too large")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise HermesP1AttachmentError(f"{label} changed while loading")
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HermesP1AttachmentError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise HermesP1AttachmentError(f"{label} is not canonical JSON")
    return value, "sha256:" + hashlib.sha256(payload).hexdigest()


def _assert_vendored_factory() -> None:
    from plugins.kwrag_slot import p1_attachment_fts_factory as factory_module

    payload = Path(factory_module.__file__).read_bytes()
    # apply_patch preserves one terminal LF while the research source has a
    # second empty terminal line.  Normalize only that packaging difference.
    if payload.endswith(b"\n") and not payload.endswith(b"\n\n"):
        payload += b"\n"
    if "sha256:" + hashlib.sha256(payload).hexdigest() != P1_FACTORY_SOURCE_DIGEST:
        raise HermesP1AttachmentError("vendored P1 factory source identity is invalid")


@dataclass(frozen=True)
class P1AttachmentBinding:
    slot_instance_id: str
    mount_authority_digest: str
    slot_runtime_binding_digest: str
    resource_observation_digest: str
    room: str
    database_relative: str
    database_sha256: str
    source_snapshot_digest: str
    authority_receipt_digest: str
    max_result_characters: int
    digest: str

    @classmethod
    def from_mapping(cls, raw: Any) -> "P1AttachmentBinding":
        if not isinstance(raw, Mapping) or set(raw) != _BINDING_FIELDS:
            raise HermesP1AttachmentError("P1 attachment binding fields are invalid")
        if raw.get("schema_version") != "hermes-kwrag-p1-attachment-binding-v1":
            raise HermesP1AttachmentError("P1 attachment binding schema is invalid")
        if raw.get("provider_dispatch_required") is not False:
            raise HermesP1AttachmentError(
                "P1 attachment proof must not require provider dispatch"
            )
        if raw.get("p1_identity") != P1_IDENTITY:
            raise HermesP1AttachmentError("P1 research identity does not match")
        max_characters = raw.get("max_result_characters")
        if (
            isinstance(max_characters, bool)
            or not isinstance(max_characters, int)
            or max_characters != TEXT_CHARACTER_MAXIMUM
        ):
            raise HermesP1AttachmentError(
                "P1 result character budget does not match the selected factory"
            )
        value = dict(raw)
        return cls(
            slot_instance_id=_identifier(value["slot_instance_id"], "slot instance"),
            mount_authority_digest=_digest(
                value["mount_authority_digest"], "mount authority digest"
            ),
            slot_runtime_binding_digest=_digest(
                value["slot_runtime_binding_digest"], "slot runtime binding digest"
            ),
            resource_observation_digest=_digest(
                value["resource_observation_digest"], "resource observation digest"
            ),
            room=_identifier(value["room"], "bound room", maximum=256),
            database_relative=_relative(value["database_relative"], "database path"),
            database_sha256=_digest(value["database_sha256"], "database digest"),
            source_snapshot_digest=_digest(
                value["source_snapshot_digest"], "source snapshot digest"
            ),
            authority_receipt_digest=_digest(
                value["authority_receipt_digest"], "authority receipt digest"
            ),
            max_result_characters=max_characters,
            digest=_canonical_digest(value),
        )


class P1AttachmentSearchPipeline:
    """Adapt the exact research FTS factory to KWRAG's backend-neutral ABI."""

    def __init__(self, scope: Any, binding: P1AttachmentBinding):
        _assert_vendored_factory()
        if scope.manifest.corpus_snapshot != binding.source_snapshot_digest:
            raise HermesP1AttachmentError("mounted source snapshot does not match P1 binding")
        room_entry = scope.manifest.raw.get("rooms", {}).get(binding.room)
        if not isinstance(room_entry, Mapping):
            raise HermesP1AttachmentError("bound room is absent from the mounted manifest")
        files = room_entry.get("files")
        if not isinstance(files, list):
            raise HermesP1AttachmentError("bound room manifest files are invalid")
        matching = [
            item
            for item in files
            if isinstance(item, Mapping)
            and item.get("path") == binding.database_relative
            and item.get("sha256") == binding.database_sha256
        ]
        if len(matching) != 1:
            raise HermesP1AttachmentError(
                "bound FTS database is not uniquely named by the mounted manifest"
            )
        try:
            from kwrag.slot_mount import resolve_mounted_path

            index_relative = PurePosixPath(scope.index_root.relative_to(scope.mount_root).as_posix())
            database_path = resolve_mounted_path(
                scope.mount_root,
                index_relative / PurePosixPath(binding.database_relative),
                kind="file",
            )
        except (ImportError, ValueError) as exc:
            raise HermesP1AttachmentError("bound FTS database is outside the slot scope") from exc
        self._room = binding.room
        self._pipeline = FtsAttachmentPipeline(
            database_path,
            {
                "databaseSha256": binding.database_sha256,
                "authority": {
                    "mode": "slot_local_read_only_nas",
                    "readOnlyObserved": True,
                    "receiptDigest": binding.authority_receipt_digest,
                },
                "sourceSnapshotDigest": binding.source_snapshot_digest,
            },
        )

    def close(self) -> None:
        self._pipeline.close()

    def search(self, query: str, rooms: list[str], k: int) -> dict[str, Any]:
        if rooms != [self._room]:
            raise HermesP1AttachmentError("P1 search escaped its one bound room")
        raw = self._pipeline.search(query)
        hits = [
            {
                "room": self._room,
                "level": "turn",
                "seg_id": item["unitId"],
                "text": item["text"],
                "score": item["score"],
                "message_ids": item["sourceIds"],
            }
            for item in raw[:k]
        ]
        return {
            "hits": hits,
            "operation_evidence": {
                "schema_version": "kwrag-slot-pipeline-evidence-v1",
                "backend_id": BACKEND_ID,
                "stages": [
                    {
                        "stage_id": "slot_local_fts5",
                        "execution_scope": "slot_local",
                        "call_count": 1,
                        "input_count": 1,
                        "output_count": len(hits),
                        "model": None,
                        "revision": P1_FACTORY_DIGEST,
                    }
                ],
                "candidate_count": len(raw),
                "data_boundary": {
                    "bytes_sent_outside_slot": 0,
                    "external_persistence": "not_applicable",
                    "persistence_receipt_digest": None,
                },
            },
        }


def _state_root() -> Path:
    return Path(get_hermes_home()) / "kwrag-p1-attachment"


def _prepare_state_root(root: Path) -> Path:
    home = Path(get_hermes_home()).resolve(strict=True)
    target = Path(root)
    if not target.is_absolute():
        raise HermesP1AttachmentError("P1 receipt root must be absolute")
    try:
        target.relative_to(home)
    except ValueError as exc:
        raise HermesP1AttachmentError("P1 receipt root must be inside HERMES_HOME") from exc
    if target.exists():
        if target.is_symlink() or not target.is_dir():
            raise HermesP1AttachmentError("P1 receipt root is not a real directory")
    else:
        target.mkdir(mode=0o700, parents=False)
    if os.name == "posix":
        info = target.stat()
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise HermesP1AttachmentError("P1 receipt root owner or mode is invalid")
    return target


def _runtime_environment() -> tuple[str, str, str, Path]:
    manifest = load_component_manifest()
    resource = load_resource_profile()
    enabled = os.environ.get("JITECH_RETRIEVAL_ENABLED")
    if enabled != "true":
        raise HermesP1AttachmentError("caller-explicit P1 attachment is disabled")
    component_digest = os.environ.get("JITECH_RETRIEVAL_COMPONENT_DIGEST", "")
    if component_digest != manifest["component_wheel"]["sha256"]:
        raise HermesP1AttachmentError("runtime component digest does not match the image")
    ops_binding_digest = _digest(
        os.environ.get("JITECH_RETRIEVAL_BINDING_DIGEST"), "OPS binding digest"
    )
    resource_digest = os.environ.get("JITECH_RETRIEVAL_RESOURCE_PROFILE_DIGEST", "")
    if resource_digest != resource["profileDigest"]:
        raise HermesP1AttachmentError("runtime resource profile does not match the image")
    workspace_raw = os.environ.get("HERMES_WORKSPACE_DIR", "")
    workspace = Path(workspace_raw)
    if not (
        workspace.is_absolute()
        or (
            workspace_raw.startswith("/")
            and "\\" not in workspace_raw
            and ".." not in PurePosixPath(workspace_raw).parts
        )
    ):
        raise HermesP1AttachmentError("runtime workspace root is unavailable")
    return component_digest, ops_binding_digest, resource_digest, workspace / "nas_docs"


def _assert_mount_read_only(mount_root: Path) -> None:
    try:
        flags = os.statvfs(mount_root).f_flag
    except (AttributeError, OSError) as exc:
        raise HermesP1AttachmentError("runtime NAS mount cannot be verified") from exc
    if not flags & getattr(os, "ST_RDONLY", 1):
        raise HermesP1AttachmentError("runtime NAS mount is not read-only")


def _atomic_write_canonical(root: Path, name: str, value: object) -> str:
    """Durably replace one private, content-free active-binding artifact."""

    payload = canonical_json_bytes(value)
    target = root / name
    temporary = root / f".{name}.{uuid.uuid4().hex}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = -1
    try:
        descriptor = os.open(temporary, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short active-binding write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, target)
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        if directory_flag:
            directory = os.open(
                root,
                os.O_RDONLY | directory_flag | getattr(os, "O_CLOEXEC", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except OSError as exc:
        raise HermesP1AttachmentError("active P1 binding could not be published") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _publish_active_bindings(
    root: Path,
    *,
    runtime_binding: Mapping[str, Any],
    p1_binding: Mapping[str, Any],
    resource_observation: Mapping[str, Any],
    attachment_receipt_digest: str,
) -> None:
    runtime_digest = _atomic_write_canonical(
        root, "active-runtime-binding.json", runtime_binding
    )
    p1_digest = _atomic_write_canonical(root, "active-p1-binding.json", p1_binding)
    _atomic_write_canonical(
        root, "active-resource-observation.json", resource_observation
    )
    active_set = {
        "schema": "hermes-kwrag-p1-active-binding-set-v1",
        "runtime_binding_digest": runtime_digest,
        "p1_binding_digest": p1_digest,
        "resource_observation_digest": resource_observation["observationDigest"],
        "attachment_receipt_digest": attachment_receipt_digest,
    }
    _atomic_write_canonical(root, "active-binding-set.json", active_set)


def _validate_resource_observation(
    value: Any,
    *,
    expected_digest: str,
    expected_profile_digest: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _RESOURCE_OBSERVATION_FIELDS:
        raise HermesP1AttachmentError("resource observation fields are invalid")
    observation = dict(value)
    digest = _digest(observation.pop("observationDigest"), "resource observation digest")
    if digest != expected_digest or _canonical_digest(observation) != digest:
        raise HermesP1AttachmentError("resource observation digest is not bound")
    if (
        observation.get("schema") != "agent-runtime-retrieval-headroom/v1"
        or observation.get("status") != "within_required_headroom"
        or observation.get("profileDigest") != expected_profile_digest
    ):
        raise HermesP1AttachmentError("resource observation status is invalid")
    profile = load_resource_profile()
    required = {
        "requiredCpuMillicores": profile["cpuReservationMillicores"],
        "requiredMemoryBytes": profile["memoryReservationBytes"],
        "requiredPids": profile["pidsReservation"],
    }
    if any(observation.get(key) != expected for key, expected in required.items()):
        raise HermesP1AttachmentError("resource observation reservation is invalid")
    numeric_fields = {
        "containerCpuUsedMillicores",
        "containerMemoryUsedBytes",
        "containerPidsUsed",
        "hostCpuAvailableMillicores",
        "hostMemoryAvailableBytes",
        "hostPidsAvailable",
    }
    if any(
        isinstance(observation.get(field), bool)
        or not isinstance(observation.get(field), int)
        or observation[field] < 0
        for field in numeric_fields
    ):
        raise HermesP1AttachmentError("resource observation contains an invalid count")
    comparisons = (
        ("containerCpuUsedMillicores", "requiredCpuMillicores", False),
        ("containerMemoryUsedBytes", "requiredMemoryBytes", False),
        ("containerPidsUsed", "requiredPids", False),
        ("hostCpuAvailableMillicores", "requiredCpuMillicores", True),
        ("hostMemoryAvailableBytes", "requiredMemoryBytes", True),
        ("hostPidsAvailable", "requiredPids", True),
    )
    for observed_field, required_field, is_available in comparisons:
        observed = observation[observed_field]
        required_value = observation[required_field]
        if (is_available and observed < required_value) or (
            not is_available and observed > required_value
        ):
            raise HermesP1AttachmentError("resource observation exceeds its declared envelope")
    return dict(value)


def run_p1_attachment_probe(
    *,
    runtime_binding_path: Path,
    p1_binding_path: Path,
    resource_observation_path: Path,
    request_path: Path,
    state_root: Path | None = None,
) -> dict[str, Any]:
    """Run one explicit FTS search and return only content-free proof fields."""

    component_digest, ops_binding_digest, resource_digest, mount_root = (
        _runtime_environment()
    )
    p1_raw, p1_file_digest = _read_canonical_mapping(
        p1_binding_path, label="P1 attachment binding"
    )
    p1_binding = P1AttachmentBinding.from_mapping(p1_raw)
    resource_observation, _resource_file_digest = _read_canonical_mapping(
        resource_observation_path, label="retrieval resource observation"
    )
    _validate_resource_observation(
        resource_observation,
        expected_digest=p1_binding.resource_observation_digest,
        expected_profile_digest=resource_digest,
    )
    runtime_raw, runtime_digest = _read_canonical_mapping(
        runtime_binding_path, label="slot runtime binding"
    )
    if runtime_digest != p1_binding.slot_runtime_binding_digest:
        raise HermesP1AttachmentError("slot runtime binding digest does not match P1 binding")
    request, _request_digest = _read_canonical_mapping(
        request_path, label="slot search request"
    )
    try:
        from kwrag.slot_runtime import SlotRuntimeSpec, build_slot_runtime

        spec = SlotRuntimeSpec.from_mapping(runtime_raw)
    except (ImportError, ValueError) as exc:
        raise HermesP1AttachmentError("slot runtime binding is invalid") from exc
    receipts = _prepare_state_root(state_root or _state_root())
    expected_operation_path = receipts / "operation-receipts.jsonl"
    if spec.mount_root != mount_root:
        raise HermesP1AttachmentError("slot runtime mount does not match the container NAS root")
    if spec.receipt_path != expected_operation_path:
        raise HermesP1AttachmentError("slot operation receipt path is not product-owned")
    if (
        spec.pipeline_fingerprint != P1_PIPELINE_FINGERPRINT
        or spec.max_concurrent != 1
    ):
        raise HermesP1AttachmentError("slot runtime does not bind the P1 execution profile")
    pipeline: P1AttachmentSearchPipeline | None = None

    def pipeline_factory(scope: Any) -> P1AttachmentSearchPipeline:
        nonlocal pipeline
        pipeline = P1AttachmentSearchPipeline(scope, p1_binding)
        return pipeline

    result_sink = FileConsumptionReceiptSink(receipts / "result-receipts.jsonl")
    attachment_sink = FileConsumptionReceiptSink(receipts / "attachment-receipts.jsonl")
    result_sink.preflight_before_retrieval()
    attachment_sink.preflight_before_retrieval()
    try:
        runtime = build_slot_runtime(spec, pipeline_factory=pipeline_factory)
        product_binding = HermesSlotRetrievalBinding.from_mapping(
            {
                "schema_version": "hermes-kwrag-slot-binding-v1",
                "enabled": True,
                "component_digest": component_digest,
                "runtime_binding_digest": runtime_digest,
                "expected_index_manifest": spec.index_manifest_digest,
                "expected_pipeline_fingerprint": P1_PIPELINE_FINGERPRINT,
                "max_result_characters": p1_binding.max_result_characters,
            }
        )
        prepared = HermesSlotRetrievalConsumer(
            product_binding,
            runtime,
            result_sink,
        ).search(request)
        attestation = prepared.content_free_attestation()
        verified_receipt = prepared.result_receipt
        attachment_receipt = {
            "schema_version": "hermes-kwrag-p1-attachment-consumption-receipt-v1",
            "consumer_family": "hermes",
            "attachment_status": "verified",
            "consumption_status": "not_consumed",
            "provider_dispatch_required": False,
            "provider_dispatch_attempted": False,
            "slot_instance_id": p1_binding.slot_instance_id,
            "ops_binding_digest": ops_binding_digest,
            "slot_runtime_binding_digest": runtime_digest,
            "p1_binding_digest": p1_file_digest,
            "mount_authority_digest": p1_binding.mount_authority_digest,
            "authority_receipt_digest": p1_binding.authority_receipt_digest,
            "resource_observation_digest": p1_binding.resource_observation_digest,
            "component_source_revision": KWRAG_SOURCE_COMMIT,
            "component_wheel_digest": KWRAG_WHEEL_DIGEST,
            "p1_identity": dict(P1_IDENTITY),
            "request_id": verified_receipt["request_id"],
            "operation_id": verified_receipt["operation_id"],
            "run_id": verified_receipt["run_id"],
            "attempt": verified_receipt["attempt"],
            "index_manifest_digest": verified_receipt["index_manifest"],
            "database_digest": p1_binding.database_sha256,
            "source_snapshot_digest": p1_binding.source_snapshot_digest,
            "result_status": verified_receipt["result_status"],
            "result_count": verified_receipt["result_count"],
            "result_digest": verified_receipt["result_digest"],
            "operation_receipt_digest": verified_receipt[
                "operation_receipt_digest"
            ],
            "result_receipt_digest": prepared.result_receipt_digest,
        }
        attachment_digest = attachment_sink.write(attachment_receipt)
        _publish_active_bindings(
            receipts,
            runtime_binding=runtime_raw,
            p1_binding=p1_raw,
            resource_observation=resource_observation,
            attachment_receipt_digest=attachment_digest,
        )
        return {
            "schema": "jitech-hermes-kwrag-p1-attachment-proof/v1",
            "consumptionReceiptDigest": attachment_digest,
            "componentDigest": component_digest,
            "opsBindingDigest": ops_binding_digest,
            "slotRuntimeBindingDigest": runtime_digest,
            "p1BindingDigest": p1_file_digest,
            "indexManifestDigest": verified_receipt["index_manifest"],
            "databaseDigest": p1_binding.database_sha256,
            "sourceSnapshotDigest": p1_binding.source_snapshot_digest,
            "mountAuthorityDigest": p1_binding.mount_authority_digest,
            "authorityReceiptDigest": p1_binding.authority_receipt_digest,
            "resourceObservationDigest": p1_binding.resource_observation_digest,
            "operationReceiptDigest": attestation["operationReceiptDigest"],
            "resultReceiptDigest": attestation["resultReceiptDigest"],
            "resultStatus": attestation["resultStatus"],
            "resultCount": verified_receipt["result_count"],
            "consumptionStatus": "not_consumed",
            "providerDispatchRequired": False,
            "providerDispatchAttempted": False,
            "p1Identity": dict(P1_IDENTITY),
            "resourceProfileDigest": resource_digest,
        }
    finally:
        if pipeline is not None:
            pipeline.close()


def _load_last_attachment(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "attachment-receipts.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise HermesP1AttachmentError("P1 attachment ledger is unavailable")
    lines = path.read_bytes().splitlines()
    if not lines:
        raise HermesP1AttachmentError("P1 attachment ledger is empty")
    try:
        value = json.loads(lines[-1].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HermesP1AttachmentError("P1 attachment ledger is invalid") from exc
    if (
        not isinstance(value, dict)
        or set(value) != _ATTACHMENT_FIELDS
        or lines[-1] != canonical_json_bytes(value)
    ):
        raise HermesP1AttachmentError("P1 attachment receipt is not canonical")
    digest = "sha256:" + hashlib.sha256(lines[-1]).hexdigest()
    return value, digest


def _ledger_contains_digest(path: Path, digest: str) -> bool:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        return False
    for line in path.read_bytes().splitlines():
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        if line != canonical_json_bytes(value):
            return False
        if "sha256:" + hashlib.sha256(line).hexdigest() == digest:
            return True
    return False


def enabled_p1_status(*, state_root: Path | None = None) -> dict[str, Any]:
    component_digest, ops_binding_digest, resource_digest, mount_root = (
        _runtime_environment()
    )
    root = state_root or _state_root()
    receipt, attachment_digest = _load_last_attachment(root)
    active_set, _active_set_digest = _read_canonical_mapping(
        root / "active-binding-set.json", label="active P1 binding set"
    )
    if (
        set(active_set) != _ACTIVE_SET_FIELDS
        or active_set.get("schema") != "hermes-kwrag-p1-active-binding-set-v1"
        or active_set.get("attachment_receipt_digest") != attachment_digest
    ):
        raise HermesP1AttachmentError("active P1 binding set is invalid")
    runtime_raw, runtime_digest = _read_canonical_mapping(
        root / "active-runtime-binding.json", label="active slot runtime binding"
    )
    p1_raw, p1_digest = _read_canonical_mapping(
        root / "active-p1-binding.json", label="active P1 attachment binding"
    )
    resource_observation, _resource_file_digest = _read_canonical_mapping(
        root / "active-resource-observation.json",
        label="active retrieval resource observation",
    )
    if (
        active_set["runtime_binding_digest"] != runtime_digest
        or active_set["p1_binding_digest"] != p1_digest
        or active_set["resource_observation_digest"]
        != resource_observation.get("observationDigest")
    ):
        raise HermesP1AttachmentError("active P1 binding files have drifted")
    p1_binding = P1AttachmentBinding.from_mapping(p1_raw)
    _validate_resource_observation(
        resource_observation,
        expected_digest=p1_binding.resource_observation_digest,
        expected_profile_digest=resource_digest,
    )
    try:
        from kwrag.slot_runtime import SlotRuntimeSpec, build_slot_runtime

        spec = SlotRuntimeSpec.from_mapping(runtime_raw)
    except (ImportError, ValueError) as exc:
        raise HermesP1AttachmentError("active slot runtime binding is invalid") from exc
    if (
        runtime_digest != p1_binding.slot_runtime_binding_digest
        or spec.mount_root != mount_root
        or spec.receipt_path != root / "operation-receipts.jsonl"
        or spec.pipeline_fingerprint != P1_PIPELINE_FINGERPRINT
        or spec.max_concurrent != 1
    ):
        raise HermesP1AttachmentError("active slot runtime binding has drifted")
    _assert_mount_read_only(mount_root)
    pipeline: P1AttachmentSearchPipeline | None = None

    def pipeline_factory(scope: Any) -> P1AttachmentSearchPipeline:
        nonlocal pipeline
        pipeline = P1AttachmentSearchPipeline(scope, p1_binding)
        return pipeline

    try:
        build_slot_runtime(spec, pipeline_factory=pipeline_factory)
    except (OSError, ValueError) as exc:
        raise HermesP1AttachmentError("active P1 mount or index is invalid") from exc
    finally:
        if pipeline is not None:
            pipeline.close()
    if (
        receipt["component_wheel_digest"] != component_digest
        or receipt["ops_binding_digest"] != ops_binding_digest
        or receipt["slot_runtime_binding_digest"] != runtime_digest
        or receipt["p1_binding_digest"] != p1_digest
        or receipt["resource_observation_digest"]
        != p1_binding.resource_observation_digest
        or receipt["index_manifest_digest"] != spec.index_manifest_digest
        or receipt["database_digest"] != p1_binding.database_sha256
        or receipt["source_snapshot_digest"] != p1_binding.source_snapshot_digest
        or receipt["mount_authority_digest"] != p1_binding.mount_authority_digest
        or receipt["authority_receipt_digest"] != p1_binding.authority_receipt_digest
        or receipt["p1_identity"] != P1_IDENTITY
        or receipt["provider_dispatch_required"] is not False
        or receipt["provider_dispatch_attempted"] is not False
        or receipt["consumption_status"] != "not_consumed"
    ):
        raise HermesP1AttachmentError("P1 attachment receipt does not match this runtime")
    if not _ledger_contains_digest(
        root / "operation-receipts.jsonl", receipt["operation_receipt_digest"]
    ) or not _ledger_contains_digest(
        root / "result-receipts.jsonl", receipt["result_receipt_digest"]
    ):
        raise HermesP1AttachmentError("P1 attachment receipt linkage is incomplete")
    attachment_data = {
        "databaseSha256": p1_binding.database_sha256,
        "indexManifestDigest": spec.index_manifest_digest,
        "sourceSnapshotDigest": p1_binding.source_snapshot_digest,
        "readOnlyAuthorityReceiptDigest": p1_binding.authority_receipt_digest,
        "slotRuntimeBindingDigest": runtime_digest,
    }
    return {
        "schema": "jitech-embedded-retrieval-attachment-status/v1",
        "proofMode": "attachment_only",
        "enabled": True,
        "componentDigest": component_digest,
        "bindingDigest": ops_binding_digest,
        "resourceProfileDigest": resource_digest,
        "p1IdentityDigest": P1_IDENTITY_DIGEST,
        "attachmentDataDigest": _canonical_digest(attachment_data),
        "attachmentHealth": "healthy",
        "gpuAccessStatus": "none",
        "hostPortCount": 0,
        "linkageStatus": "complete",
        "mountReadOnly": True,
        "operationReceiptDigest": receipt["operation_receipt_digest"],
        "resourceStatus": "within_declared_reservation",
        "resultReceiptDigest": receipt["result_receipt_digest"],
        "consumptionReceiptDigest": attachment_digest,
        "consumptionStatus": "not_consumed",
        "revocationStatus": None,
    }
