"""Content-free operator status for the embedded KWRAG component."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

from plugins.kwrag_slot.p1_attachment import (
    _environment,
    enabled_p1_status,
    run_p1_attachment_probe,
)


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
        help="Run one caller-explicit P1 retrieval probe",
    )
    for option in ("runtime-binding", "p1-binding", "resource-observation", "request"):
        probe.add_argument(f"--{option}", type=Path, required=True)
    probe.add_argument("--conversation-message-file", type=Path)
    probe.add_argument("--json", action="store_true", help="Emit canonical JSON")


def _status() -> dict[str, object]:
    try:
        importlib.metadata.version("kwrag-product-service")
        __import__("kwrag.slot_consumer")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("embedded KWRAG component is unavailable") from exc
    enabled, component_digest, binding_digest, resource_digest, _mount = _environment()
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
        message = None
        if args.conversation_message_file is not None:
            raw = args.conversation_message_file.read_bytes()
            if not 0 < len(raw) <= 16_000:
                raise RuntimeError("conversation message file size is invalid")
            message = raw.decode("utf-8")
        result = run_p1_attachment_probe(
            runtime_binding_path=args.runtime_binding,
            p1_binding_path=args.p1_binding,
            resource_observation_path=args.resource_observation,
            request_path=args.request,
            conversation_message=message,
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
