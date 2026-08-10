"""Product-owned build status for the embedded KWRAG component."""

from __future__ import annotations

import argparse
import importlib.metadata
import json

from plugins.kwrag_slot.manifest import load_component_manifest


def register_cli(parser: argparse.ArgumentParser) -> None:
    subparsers = parser.add_subparsers(dest="kwrag_slot_command", required=True)
    status = subparsers.add_parser(
        "status",
        help="Print the embedded product component identity",
    )
    status.add_argument("--json", action="store_true")


def _status() -> dict[str, object]:
    try:
        importlib.metadata.version("kwrag-product-service")
        __import__("kwrag.slot_consumer")
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        raise RuntimeError("embedded KWRAG component is unavailable") from exc
    manifest = load_component_manifest()
    return {
        "componentDigest": manifest["component_wheel"]["sha256"],
        "componentManifestDigest": manifest["manifest_digest"],
        "componentSourceRevision": manifest["component_source_revision"],
        "defaultEnabled": manifest["default_enabled"],
        "hostPortCount": 0,
        "schema": "hermes-kwrag-product-component-status/v1",
        "transport": manifest["transport"],
    }


def kwrag_slot_command(args: argparse.Namespace) -> int:
    if args.kwrag_slot_command != "status":
        raise RuntimeError("unsupported KWRAG command")
    result = _status()
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
