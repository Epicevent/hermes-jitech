"""Strict access to the build-bound embedded KWRAG component identity."""

from __future__ import annotations

import hashlib
import json
import re
from importlib import resources
from typing import Any, Mapping


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_FIELDS = {
    "schema_version",
    "component_source_revision",
    "component_source_archive",
    "component_wheel",
    "contract_collection_digest",
    "default_enabled",
    "host_port_count",
    "transport",
    "policy_boundary",
}


class ComponentManifestError(ValueError):
    """The embedded component manifest is missing or internally inconsistent."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise ComponentManifestError(f"{field} is not a canonical SHA-256 digest")
    return text


def _artifact(value: Any, *, filename_required: bool) -> dict[str, Any]:
    expected = {"bytes", "sha256"} | ({"filename"} if filename_required else set())
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ComponentManifestError("component artifact fields are invalid")
    size = value.get("bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ComponentManifestError("component artifact size is invalid")
    result = {"bytes": size, "sha256": _digest(value.get("sha256"), "artifact digest")}
    if filename_required:
        filename = value.get("filename")
        if (
            not isinstance(filename, str)
            or not filename
            or "/" in filename
            or "\\" in filename
            or filename in {".", ".."}
        ):
            raise ComponentManifestError("component wheel filename is invalid")
        result["filename"] = filename
    return result


def load_component_manifest() -> dict[str, Any]:
    payload = (
        resources
        .files("plugins.kwrag_slot")
        .joinpath("component-manifest.json")
        .read_bytes()
    )
    if (
        payload.startswith(b"\xef\xbb\xbf")
        or not payload.endswith(b"\n")
        or payload.endswith(b"\n\n")
    ):
        raise ComponentManifestError("component manifest bytes are not canonical")
    canonical_payload = payload[:-1]
    try:
        parsed = json.loads(canonical_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComponentManifestError(
            "component manifest is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, dict) or set(parsed) != _FIELDS:
        raise ComponentManifestError("component manifest fields are invalid")
    if canonical_json_bytes(parsed) != canonical_payload:
        raise ComponentManifestError("component manifest is not canonically encoded")
    if parsed.get("schema_version") != "hermes-kwrag-embedded-component-manifest-v1":
        raise ComponentManifestError("component manifest schema is invalid")
    revision = parsed.get("component_source_revision")
    if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
        raise ComponentManifestError("component source revision is invalid")
    source_archive = _artifact(
        parsed.get("component_source_archive"), filename_required=False
    )
    wheel = _artifact(parsed.get("component_wheel"), filename_required=True)
    contract_digest = _digest(
        parsed.get("contract_collection_digest"), "contract digest"
    )
    if (
        parsed.get("default_enabled") is not False
        or parsed.get("host_port_count") != 0
        or parsed.get("transport") != "in_process"
    ):
        raise ComponentManifestError("component runtime boundary is invalid")
    policy = parsed.get("policy_boundary")
    if not isinstance(policy, dict) or policy != {
        "automatic_search": False,
        "backend_selected": True,
        "invocation_policy_selected": False,
        "prompt_assembly_added": False,
    }:
        raise ComponentManifestError("component policy boundary is invalid")
    return {
        **parsed,
        "component_source_archive": source_archive,
        "component_wheel": wheel,
        "contract_collection_digest": contract_digest,
        "manifest_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }
