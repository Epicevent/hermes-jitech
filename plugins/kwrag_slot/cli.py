"""Content-free operator status for the embedded KWRAG component."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re

from plugins.kwrag_slot.manifest import load_component_manifest, load_resource_profile


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


def register_cli(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="kwrag_slot_command", required=True)
    status = subparsers.add_parser("status", help="Print content-free component status")
    status.add_argument("--json", action="store_true", help="Emit canonical JSON")


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
    nas_root = os.environ.get("OPENCLAW_NAS_CONTAINER_PATH", "")
    if not nas_root.startswith("/"):
        raise RuntimeError("runtime NAS root is unavailable")
    try:
        mount_read_only = bool(os.statvfs(nas_root).f_flag & getattr(os, "ST_RDONLY", 1))
    except (AttributeError, OSError) as exc:
        raise RuntimeError("runtime NAS mount cannot be verified") from exc
    if not mount_read_only:
        raise RuntimeError("runtime NAS mount is not read-only")
    if enabled:
        raise RuntimeError("enabled retrieval status is unavailable before an approved product invocation")
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
    status = _status()
    if args.json:
        print(json.dumps(status, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")))
    else:
        for key in sorted(status):
            print(f"{key}={status[key]}")
    return 0
