from __future__ import annotations

import hashlib
import os
import re
import socket
import stat
from pathlib import Path
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


P1_PIPELINE_FINGERPRINT = (
    "sha256:53e14752cc9d147dfb4129e00234d1c7fb9f6558df00da7c03189db8da8e4606"
)
P1_IDENTITY = {
    "status": "research_selected_p1_attachment_probe_candidate",
    "pipelineFactoryDigest": "sha256:0dbe54f5a8bc56a6c821e181a0dc6cfda85d25be8cea6a01235cb5e347782f0e",
    "backendId": "slot-local-fts5-trigram-or-attachment-v1",
    "pipelineFingerprint": P1_PIPELINE_FINGERPRINT,
    "researchDecisionDigest": "sha256:81e6f4d83e6cde6a9c83a9aa435c65354a1122dded735bf607462c3497e9b25d",
}

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_OPS_OBSERVATION_UID = 0


HermesP1AttachmentError = HermesSlotRetrievalError


def _hash(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


P1_IDENTITY_DIGEST = _hash(P1_IDENTITY)


def _component():
    import kwrag_p1_attachment

    return kwrag_p1_attachment


def _digest(value: object, label: str) -> str:
    if not _SHA256.fullmatch(text := str(value or "")):
        raise HermesP1AttachmentError(f"{label} is invalid")
    return text


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    return _component().load_canonical_mapping(path, label)


def _read_resource_observation(path: Path) -> tuple[dict[str, Any], str]:
    return _component().load_authoritative_canonical_mapping(
        path, "resource observation", required_owner_uid=_OPS_OBSERVATION_UID
    )


def _runtime_identity() -> tuple[str, str]:
    cgroup = Path("/proc/self/cgroup").read_bytes()
    if not cgroup or len(cgroup) > 64 * 1024:
        raise HermesP1AttachmentError("runtime cgroup identity is invalid")
    return (
        "sha256:" + hashlib.sha256(socket.gethostname().encode()).hexdigest(),
        "sha256:" + hashlib.sha256(cgroup).hexdigest(),
    )


def _workspace_mount() -> Path:
    if os.environ.get("HERMES_WORKSPACE_DIR") != "/workspace":
        raise HermesP1AttachmentError("Hermes workspace root is invalid")
    return Path("/workspace/nas_docs")


def _environment() -> tuple[bool, str, str, str, Path]:
    manifest, resource = load_component_manifest(), load_resource_profile()
    enabled_raw = os.environ.get("JITECH_RETRIEVAL_ENABLED")
    if enabled_raw not in {"true", "false"}:
        raise HermesP1AttachmentError("retrieval enabled state is unavailable")
    component = os.environ.get("JITECH_RETRIEVAL_COMPONENT_DIGEST", "")
    if component != manifest["component_wheel"]["sha256"]:
        raise HermesP1AttachmentError("runtime component does not match the image")
    binding = _digest(
        os.environ.get("JITECH_RETRIEVAL_BINDING_DIGEST"), "binding digest"
    )
    resource_digest = os.environ.get("JITECH_RETRIEVAL_RESOURCE_PROFILE_DIGEST", "")
    if resource_digest != resource["profileDigest"]:
        raise HermesP1AttachmentError(
            "runtime resource profile does not match the image"
        )
    mount = _workspace_mount()
    try:
        read_only = bool(os.statvfs(mount).f_flag & getattr(os, "ST_RDONLY", 1))
    except (AttributeError, OSError) as exc:
        raise HermesP1AttachmentError("runtime NAS mount cannot be verified") from exc
    if not read_only:
        raise HermesP1AttachmentError("runtime NAS mount is not read-only")
    return enabled_raw == "true", component, binding, resource_digest, mount


def _validate_binding(
    value: dict[str, Any],
    digest: str,
    expected_digest: str,
    component: str,
    resource: str,
) -> None:
    manifest = load_component_manifest()
    _component().validate_attachment_binding(
        value,
        digest=digest,
        expected_digest=expected_digest,
        component_digest=component,
        contract_digest=manifest["contract_collection_digest"],
        resource_profile_digest=resource,
        p1_identity=P1_IDENTITY,
    )


def _validate_resource(value: dict[str, Any], *, profile: str, instance: str) -> int:
    container, cgroup = _runtime_identity()
    limits = load_resource_profile()
    return _component().validate_resource_observation(
        value,
        profile_digest=profile,
        instance_id=instance,
        container_identity_digest=container,
        cgroup_identity_digest=cgroup,
        cpu_reservation=limits["cpuReservationMillicores"],
        memory_reservation=limits["memoryReservationBytes"],
        pids_reservation=limits["pidsReservation"],
    )


def _root(root: Path | None) -> Path:
    target = Path(root or Path(get_hermes_home()) / "kwrag-p1-attachment")
    if target.is_symlink():
        raise HermesP1AttachmentError("attachment state root is unsafe")
    target = target.resolve(strict=True)
    home = Path(get_hermes_home()).resolve(strict=True)
    if not target.is_absolute() or not target.is_relative_to(home):
        raise HermesP1AttachmentError("attachment state root is outside HERMES_HOME")
    if not target.is_dir():
        raise HermesP1AttachmentError("attachment state root is unsafe")
    if os.name == "posix":
        info = target.stat()
        effective_uid = getattr(os, "geteuid", lambda: None)()
        if info.st_uid != effective_uid or stat.S_IMODE(info.st_mode) != 0o700:
            raise HermesP1AttachmentError("attachment state root is unsafe")
    return target


def _receipt(path: Path, digest: str | None, label: str) -> dict[str, Any]:
    return _component().load_receipt(path, digest, label)


def _runtime(
    runtime: Mapping[str, Any],
    runtime_digest: str,
    binding: Mapping[str, Any],
    mount: Path,
    root: Path,
    room: str,
    maximum_bytes: int,
):
    return _component().build_p1_runtime(
        runtime,
        runtime_digest,
        binding,
        mount,
        root / "operation-receipts.jsonl",
        room,
        maximum_bytes,
        P1_PIPELINE_FINGERPRINT,
    )


def run_p1_attachment_probe(
    *,
    runtime_binding_path: Path,
    p1_binding_path: Path,
    resource_observation_path: Path,
    request_path: Path,
    conversation_message: str | None = None,
    state_root: Path | None = None,
) -> dict[str, Any]:
    enabled, component, ops_digest, resource_digest, mount = _environment()
    if not enabled:
        raise HermesP1AttachmentError("caller-explicit P1 attachment is disabled")
    root = _root(state_root)
    if tuple(
        map(Path, (p1_binding_path, runtime_binding_path, resource_observation_path))
    ) != (
        root / "binding-v2.json",
        root / "runtime-binding.json",
        root / "resource-observation.json",
    ):
        raise HermesP1AttachmentError(
            "attachment inputs are outside the OPS state root"
        )
    binding, binding_digest = _read_json(p1_binding_path, "attachment binding")
    _validate_binding(binding, binding_digest, ops_digest, component, resource_digest)
    observation, _ = _read_resource_observation(resource_observation_path)
    memory = _validate_resource(
        observation, profile=resource_digest, instance=binding["instanceId"]
    )
    runtime_raw, runtime_digest = _read_json(
        runtime_binding_path, "slot runtime binding"
    )
    request, _ = _read_json(request_path, "slot search request")
    if binding["enabled"] is not True:
        raise HermesP1AttachmentError(
            "caller-explicit P1 attachment binding is disabled"
        )
    spec, runtime_object, pipeline = _runtime(
        runtime_raw,
        runtime_digest,
        binding,
        mount,
        root,
        str(request.get("corpus", "")),
        memory,
    )
    try:
        consumer = HermesSlotRetrievalConsumer(
            HermesSlotRetrievalBinding.from_mapping({
                "schema_version": "hermes-kwrag-slot-binding-v1",
                "enabled": True,
                "component_digest": component,
                "runtime_binding_digest": runtime_digest,
                "expected_index_manifest": spec.index_manifest_digest,
                "expected_pipeline_fingerprint": P1_PIPELINE_FINGERPRINT,
                "max_result_characters": 20_000,
            }),
            runtime_object,
            FileConsumptionReceiptSink(root / "result-receipts.jsonl"),
        )
        result = consumer.search(request)
        receipt = result.result_receipt
        consumption = {
            "schema_version": "hermes-kwrag-p1-attachment-consumption-receipt-v1",
            "consumer_family": "hermes",
            "attachment_status": "verified",
            "consumption_status": "not_consumed",
            "provider_dispatch_attempted": False,
            "instance_id": binding["instanceId"],
            "binding_digest": binding_digest,
            "resource_observation_digest": observation["observationDigest"],
            "component_digest": component,
            "p1_identity_digest": P1_IDENTITY_DIGEST,
            "attachment_data_digest": _hash(binding["attachmentData"]),
            "request_id": receipt["request_id"],
            "operation_id": receipt["operation_id"],
            "run_id": receipt["run_id"],
            "attempt": receipt["attempt"],
            "result_status": receipt["result_status"],
            "result_count": receipt["result_count"],
            "result_digest": receipt["result_digest"],
            "operation_receipt_digest": receipt["operation_receipt_digest"],
            "result_receipt_digest": result.result_receipt_digest,
        }
        consumption_digest = FileConsumptionReceiptSink(
            root / "attachment-receipts.jsonl"
        ).write(consumption)
        proof = {
            "schema": "jitech-hermes-kwrag-p1-attachment-proof/v1",
            "operationReceiptDigest": receipt["operation_receipt_digest"],
            "resultReceiptDigest": result.result_receipt_digest,
            "consumptionReceiptDigest": consumption_digest,
            "resultStatus": receipt["result_status"],
            "resultCount": receipt["result_count"],
        }
        if conversation_message is not None:
            if not conversation_message:
                raise HermesP1AttachmentError("conversation message is empty")
            from hermes_cli.oneshot import _run_agent

            outcome = _run_agent(
                conversation_message,
                toolsets=[],
                use_config_toolsets=False,
                approved_retrieval=result,
            )
            attestation = result.content_free_attestation()
            if not (
                isinstance(outcome, dict)
                and outcome.get("completed") is True
                and attestation["transportOutcomeStatus"] == "response_observed"
            ):
                raise HermesP1AttachmentError("retrieval conversation did not complete")
            proof["conversationAttestation"] = attestation
        return proof
    finally:
        pipeline.close()


def enabled_p1_status(*, state_root: Path | None = None) -> dict[str, Any]:
    enabled, component, ops_digest, resource_digest, mount = _environment()
    base = {
        "schema": "jitech-embedded-retrieval-attachment-status/v1",
        "proofMode": "attachment_only",
        "enabled": enabled,
        "componentDigest": component,
        "bindingDigest": ops_digest,
        "resourceProfileDigest": resource_digest,
        "p1IdentityDigest": P1_IDENTITY_DIGEST,
        "hostPortCount": 0,
        "mountReadOnly": True,
        "gpuAccessStatus": "none",
    }
    root = _root(state_root)
    if not enabled:
        disabled, disabled_digest = _read_json(
            root / "binding-v2.json", "disabled attachment binding"
        )
        _validate_binding(
            disabled, disabled_digest, ops_digest, component, resource_digest
        )
        if disabled["enabled"] is not False:
            raise HermesP1AttachmentError("disabled status has an enabled binding")
        return base | {
            "attachmentDataDigest": None,
            "attachmentHealth": "disabled",
            "resourceStatus": "unavailable",
            "operationReceiptDigest": None,
            "resultReceiptDigest": None,
            "consumptionReceiptDigest": None,
            "consumptionStatus": "not_applicable",
            "linkageStatus": "not_applicable",
            "revocationStatus": "complete",
        }
    binding, binding_digest = _read_json(root / "binding-v2.json", "attachment binding")
    _validate_binding(binding, binding_digest, ops_digest, component, resource_digest)
    observation, _ = _read_resource_observation(root / "resource-observation.json")
    memory = _validate_resource(
        observation, profile=resource_digest, instance=binding["instanceId"]
    )
    runtime_raw, runtime_digest = _read_json(
        root / "runtime-binding.json", "slot runtime binding"
    )
    consumption = _receipt(root / "attachment-receipts.jsonl", None, "attachment")
    consumption_digest = _hash(consumption)
    operation = _receipt(
        root / "operation-receipts.jsonl",
        consumption["operation_receipt_digest"],
        "operation",
    )
    room = operation.get("corpora", [""])[0]
    spec, _runtime_object, pipeline = _runtime(
        runtime_raw, runtime_digest, binding, mount, root, str(room), memory
    )
    try:
        result = _receipt(
            root / "result-receipts.jsonl",
            consumption["result_receipt_digest"],
            "result",
        )
    finally:
        pipeline.close()
    _component().validate_receipt_linkage(
        consumption,
        operation,
        result,
        binding=binding,
        observation_digest=observation["observationDigest"],
        component_digest=component,
        runtime_binding_digest=runtime_digest,
        index_manifest_digest=spec.index_manifest_digest,
        binding_digest=ops_digest,
        p1_identity_digest=P1_IDENTITY_DIGEST,
        attachment_data_digest=_hash(binding["attachmentData"]),
    )
    return base | {
        "attachmentDataDigest": _hash(binding["attachmentData"]),
        "attachmentHealth": "healthy",
        "resourceStatus": "within_declared_reservation",
        "operationReceiptDigest": consumption["operation_receipt_digest"],
        "resultReceiptDigest": consumption["result_receipt_digest"],
        "consumptionReceiptDigest": consumption_digest,
        "consumptionStatus": "not_consumed",
        "linkageStatus": "complete",
        "revocationStatus": None,
    }
