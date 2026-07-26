"""Exact machine-readable coverage manifest for Hermes provider-call receipts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


SCHEMA_NAME = "jitech-provider-usage-coverage/v1"
PRODUCT_FAMILY = "hermes"

_TOP_FIELDS = frozenset({
    "schema",
    "productFamily",
    "manifestDigest",
    "coverageStatus",
    "surfaces",
})
_SURFACE_FIELDS = frozenset({
    "surfaceCode",
    "observationKind",
    "meterFamily",
    "modelEvidence",
    "retryObservation",
    "usageObservation",
    "status",
    "gapCode",
})
_OBSERVATION_KINDS = frozenset({"per_call", "turn_aggregate", "request_only"})
_METER_FAMILIES = frozenset({
    "tokens",
    "image",
    "audio",
    "characters",
    "search",
    "other",
})
_MODEL_EVIDENCE = frozenset({
    "provider_response",
    "requested_only",
    "unavailable",
})
_RETRY_OBSERVATIONS = frozenset({
    "physical_attempt",
    "logical_call_only",
    "unavailable",
})
_USAGE_OBSERVATIONS = frozenset({
    "provider_reported",
    "request_observed",
    "unavailable",
})
_STATUSES = frozenset({"implemented", "partial", "gap"})


def _surface(
    surface_code: str,
    observation_kind: str,
    meter_family: str,
    model_evidence: str,
    retry_observation: str,
    usage_observation: str,
    status: str = "implemented",
    gap_code: str | None = None,
) -> dict[str, Any]:
    return {
        "surfaceCode": surface_code,
        "observationKind": observation_kind,
        "meterFamily": meter_family,
        "modelEvidence": model_evidence,
        "retryObservation": retry_observation,
        "usageObservation": usage_observation,
        "status": status,
        "gapCode": gap_code,
    }


_SURFACES = (
    _surface(
        "hermes.aux.async",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.aux.sync",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.chat.main.anthropic",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.chat.main.bedrock.nonstream",
        "per_call",
        "tokens",
        "requested_only",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.chat.main.bedrock.stream",
        "per_call",
        "tokens",
        "requested_only",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.chat.main.codex.stream",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.chat.main.gemini.cloudcode",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.chat.main.gemini.native",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.chat.main.openai.nonstream",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.chat.main.openai.stream",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.cli.goals",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.cli.kanban.decompose",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.cli.kanban.specify",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.cli.profile.describe",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.compress.async",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.compress.sync",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.image.fal.generate",
        "request_only",
        "image",
        "requested_only",
        "logical_call_only",
        "request_observed",
        "partial",
        "SDK_INTERNAL_RETRY_UNVERIFIED",
    ),
    _surface(
        "hermes.image.fal.upscale",
        "request_only",
        "image",
        "requested_only",
        "logical_call_only",
        "request_observed",
        "partial",
        "SDK_INTERNAL_RETRY_UNVERIFIED",
    ),
    _surface(
        "hermes.image.krea.submit",
        "request_only",
        "image",
        "requested_only",
        "physical_attempt",
        "request_observed",
    ),
    _surface(
        "hermes.image.openai",
        "per_call",
        "image",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.image.openai_codex",
        "per_call",
        "image",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.image.xai",
        "request_only",
        "image",
        "requested_only",
        "physical_attempt",
        "request_observed",
    ),
    _surface(
        "hermes.mini_swe",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.moa.aggregate",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.moa.reference",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.plugin.image.external",
        "request_only",
        "image",
        "unavailable",
        "unavailable",
        "unavailable",
        "gap",
        "EXTERNAL_PLUGIN_PROVIDER_UNINSTRUMENTED",
    ),
    _surface(
        "hermes.plugin.stt.external",
        "request_only",
        "audio",
        "unavailable",
        "unavailable",
        "unavailable",
        "gap",
        "EXTERNAL_PLUGIN_PROVIDER_UNINSTRUMENTED",
    ),
    _surface(
        "hermes.plugin.tts.external",
        "request_only",
        "audio",
        "unavailable",
        "unavailable",
        "unavailable",
        "gap",
        "EXTERNAL_PLUGIN_PROVIDER_UNINSTRUMENTED",
    ),
    _surface(
        "hermes.plugin.video.external",
        "request_only",
        "other",
        "unavailable",
        "unavailable",
        "unavailable",
        "gap",
        "EXTERNAL_PLUGIN_PROVIDER_UNINSTRUMENTED",
    ),
    _surface(
        "hermes.probe.gemini.tier",
        "request_only",
        "other",
        "requested_only",
        "physical_attempt",
        "request_observed",
    ),
    _surface(
        "hermes.realtime.openai.audio",
        "request_only",
        "audio",
        "requested_only",
        "unavailable",
        "unavailable",
        "gap",
        "REALTIME_USAGE_NOT_CAPTURED",
    ),
    _surface(
        "hermes.search.xai.responses",
        "per_call",
        "search",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.stt.groq",
        "request_only",
        "audio",
        "requested_only",
        "physical_attempt",
        "request_observed",
    ),
    _surface(
        "hermes.stt.mistral",
        "request_only",
        "audio",
        "requested_only",
        "logical_call_only",
        "request_observed",
        "partial",
        "SDK_INTERNAL_RETRY_UNVERIFIED",
    ),
    _surface(
        "hermes.stt.openai",
        "per_call",
        "audio",
        "provider_response",
        "physical_attempt",
        "provider_reported",
        "partial",
        "USAGE_RESPONSE_FORMAT_DEPENDENT",
    ),
    _surface(
        "hermes.stt.xai",
        "request_only",
        "audio",
        "requested_only",
        "physical_attempt",
        "unavailable",
        "partial",
        "AUDIO_DURATION_NOT_EXPORTED",
    ),
    _surface(
        "hermes.summary.anthropic",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.summary.codex",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.summary.openai_wire",
        "per_call",
        "tokens",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.tts.edge",
        "request_only",
        "characters",
        "requested_only",
        "logical_call_only",
        "request_observed",
        "partial",
        "SDK_INTERNAL_RETRY_UNVERIFIED",
    ),
    _surface(
        "hermes.tts.elevenlabs",
        "request_only",
        "characters",
        "requested_only",
        "logical_call_only",
        "request_observed",
        "partial",
        "SDK_INTERNAL_RETRY_UNVERIFIED",
    ),
    _surface(
        "hermes.tts.elevenlabs.stream",
        "request_only",
        "characters",
        "requested_only",
        "logical_call_only",
        "request_observed",
        "partial",
        "SDK_INTERNAL_RETRY_UNVERIFIED",
    ),
    _surface(
        "hermes.tts.gemini",
        "per_call",
        "audio",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
    _surface(
        "hermes.tts.minimax",
        "request_only",
        "characters",
        "requested_only",
        "physical_attempt",
        "unavailable",
        "partial",
        "CHARACTER_METER_NOT_EXPORTED",
    ),
    _surface(
        "hermes.tts.mistral",
        "request_only",
        "characters",
        "requested_only",
        "logical_call_only",
        "request_observed",
        "partial",
        "SDK_INTERNAL_RETRY_UNVERIFIED",
    ),
    _surface(
        "hermes.tts.openai",
        "request_only",
        "characters",
        "requested_only",
        "physical_attempt",
        "request_observed",
    ),
    _surface(
        "hermes.tts.xai",
        "request_only",
        "characters",
        "requested_only",
        "physical_attempt",
        "request_observed",
    ),
    _surface(
        "hermes.video.fal.generate",
        "request_only",
        "other",
        "requested_only",
        "logical_call_only",
        "request_observed",
        "partial",
        "SDK_INTERNAL_RETRY_UNVERIFIED",
    ),
    _surface(
        "hermes.web.registry.external",
        "request_only",
        "search",
        "unavailable",
        "unavailable",
        "unavailable",
        "gap",
        "WEB_PROVIDER_USAGE_UNINSTRUMENTED",
    ),
    _surface(
        "hermes.web.xai.responses",
        "per_call",
        "search",
        "provider_response",
        "physical_attempt",
        "provider_reported",
    ),
)


def canonical_manifest_bytes(manifest: dict[str, Any]) -> bytes:
    body = {key: value for key, value in manifest.items() if key != "manifestDigest"}
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def manifest_digest(manifest: dict[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()}"


def validate_provider_usage_coverage(
    manifest: dict[str, Any],
    *,
    expected_family: str = PRODUCT_FAMILY,
) -> None:
    if not isinstance(manifest, dict) or frozenset(manifest) != _TOP_FIELDS:
        raise ValueError("coverage manifest top-level fields mismatch")
    if manifest["schema"] != SCHEMA_NAME:
        raise ValueError("coverage manifest schema mismatch")
    if manifest["productFamily"] != expected_family:
        raise ValueError("coverage manifest productFamily mismatch")
    if manifest["coverageStatus"] not in {"complete", "partial"}:
        raise ValueError("coverageStatus must be complete or partial")
    surfaces = manifest["surfaces"]
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError("surfaces must be a nonempty array")

    codes: list[str] = []
    for surface in surfaces:
        if not isinstance(surface, dict) or frozenset(surface) != _SURFACE_FIELDS:
            raise ValueError("coverage surface fields mismatch")
        code = surface["surfaceCode"]
        if not isinstance(code, str) or not code:
            raise ValueError("surfaceCode must be a nonempty string")
        codes.append(code)
        enum_checks = (
            ("observationKind", _OBSERVATION_KINDS),
            ("meterFamily", _METER_FAMILIES),
            ("modelEvidence", _MODEL_EVIDENCE),
            ("retryObservation", _RETRY_OBSERVATIONS),
            ("usageObservation", _USAGE_OBSERVATIONS),
            ("status", _STATUSES),
        )
        for field, allowed in enum_checks:
            if surface[field] not in allowed:
                raise ValueError(f"invalid {field} for {code}")
        gap_code = surface["gapCode"]
        if gap_code is not None and (not isinstance(gap_code, str) or not gap_code):
            raise ValueError(f"gapCode must be nonempty or null for {code}")
        if surface["status"] == "implemented" and gap_code is not None:
            raise ValueError(f"implemented surface cannot have gapCode: {code}")
        if surface["status"] != "implemented" and gap_code is None:
            raise ValueError(f"non-implemented surface requires gapCode: {code}")

    if codes != sorted(codes) or len(codes) != len(set(codes)):
        raise ValueError("surfaceCode values must be unique and sorted")
    expected_coverage = (
        "complete"
        if all(surface["status"] == "implemented" for surface in surfaces)
        else "partial"
    )
    if manifest["coverageStatus"] != expected_coverage:
        raise ValueError(f"coverageStatus must be {expected_coverage}")
    digest = manifest["manifestDigest"]
    if not isinstance(digest, str) or digest != manifest_digest(manifest):
        raise ValueError("coverage manifest digest mismatch")


def provider_usage_coverage_manifest() -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA_NAME,
        "productFamily": PRODUCT_FAMILY,
        "manifestDigest": None,
        "coverageStatus": "partial",
        "surfaces": [dict(surface) for surface in _SURFACES],
    }
    manifest["manifestDigest"] = manifest_digest(manifest)
    validate_provider_usage_coverage(manifest)
    return manifest
