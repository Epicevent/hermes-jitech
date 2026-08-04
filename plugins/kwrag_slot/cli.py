"""Content-free operator status for the embedded KWRAG component."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
from pathlib import Path

from plugins.kwrag_slot.manifest import load_component_manifest, load_resource_profile
from plugins.kwrag_slot.p1_attachment import enabled_p1_status, run_p1_attachment_probe


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def register_cli(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="kwrag_slot_command", required=True)
    for name, help_text in (
        ("status", "Print content-free component status"),
        ("p1-attachment-status", "Verify the caller-explicit P1 attachment"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--json", action="store_true")
    probe = subparsers.add_parser(
        "p1-attachment-probe",
        help="Run one caller-explicit, provider-free P1 attachment probe",
    )
    for option in ("runtime-binding", "p1-binding", "resource-observation", "request"):
        probe.add_argument(f"--{option}", type=Path, required=True)
    probe.add_argument("--json", action="store_true", help="Emit canonical JSON")


def _status() -> dict[str, object]:
    manifest = load_component_manifest()
    try:
        importlib.metadata.version("kwrag-product-service")
        __import__("kwrag.slot_consumer")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("embedded KWRAG component is unavailable") from exc
    resource = load_resource_profile()
    enabled_raw = os.environ.get("JITECH_RETRIEVAL_ENABLED")
    if enabled_raw not in {"true", "false"}:
        raise RuntimeError("retrieval enabled state is unavailable")
    enabled = enabled_raw == "true"
    component_digest = os.environ.get("JITECH_RETRIEVAL_COMPONENT_DIGEST", "")
    if component_digest != manifest["component_wheel"]["sha256"]:
        raise RuntimeError("runtime component digest does not match the image")
    binding_digest = os.environ.get("JITECH_RETRIEVAL_BINDING_DIGEST", "")
    if not _SHA256.fullmatch(binding_digest):
        raise RuntimeError("runtime binding digest is unavailable")
    resource_digest = os.environ.get("JITECH_RETRIEVAL_RESOURCE_PROFILE_DIGEST", "")
    if resource_digest != resource["profileDigest"]:
        raise RuntimeError("runtime resource profile digest does not match the image")
    workspace_root = os.environ.get("HERMES_WORKSPACE_DIR", "")
    if not workspace_root.startswith("/"):
        raise RuntimeError("runtime NAS root is unavailable")
    nas_root = Path(workspace_root) / "nas_docs"
    try:
        mount_read_only = bool(
            os.statvfs(nas_root).f_flag & getattr(os, "ST_RDONLY", 1)
        )
    except (AttributeError, OSError) as exc:
        raise RuntimeError("runtime NAS mount cannot be verified") from exc
    if not mount_read_only:
        raise RuntimeError("runtime NAS mount is not read-only")
    if enabled:
        raise RuntimeError("enabled attachment status requires p1-attachment-status")
    return {
        "bindingDigest": binding_digest,
        "componentDigest": component_digest,
        "consumerHealth": "disabled",
        "consumptionReceiptDigest": None,
        "gpuAccessStatus": "none",
        "hostPortCount": 0,
        "linkageStatus": "not_applicable",
        "mountReadOnly": True,
        "operationReceiptDigest": None,
        "resourceProfileDigest": resource_digest,
        "resourceStatus": "unavailable",
        "resultReceiptDigest": None,
        "revocationStatus": "complete",
        "schema": "jitech-embedded-retrieval-status/v1",
    }


def kwrag_slot_command(args: argparse.Namespace) -> int:
    command = args.kwrag_slot_command
    if command == "p1-attachment-probe":
        result = run_p1_attachment_probe(
            runtime_binding_path=args.runtime_binding,
            p1_binding_path=args.p1_binding,
            resource_observation_path=args.resource_observation,
            request_path=args.request,
        )
    else:
        result = {"status": _status, "p1-attachment-status": enabled_p1_status}[
            command
        ]()
    if args.json:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    else:
        for key in sorted(result):
            print(f"{key}={result[key]}")
    return 0
