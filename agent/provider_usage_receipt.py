"""Canonical content-free provider-call usage receipts.

Hermes core owns provider-call truth. Downstream collectors may attach runtime
binding, but they must not reconstruct usage or provider/model identity.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


SCHEMA_NAME = "jitech-provider-usage-call/v1"
EXPORT_SCHEMA_NAME = "jitech-provider-usage-export/v1"

TRIGGERS = frozenset({
    "user",
    "cron",
    "heartbeat",
    "manual",
    "memory",
    "overflow",
    "unknown",
})
STATUSES = frozenset({"succeeded", "failed", "interrupted", "cancelled"})
COVERAGES = frozenset({"unavailable", "partial", "complete"})

_CALL_FIELDS = frozenset({
    "schema",
    "ledgerSeq",
    "receiptDigest",
    "callId",
    "runId",
    "turnId",
    "requestId",
    "sessionId",
    "trigger",
    "producerCoverageDigest",
    "attempt",
    "retryOf",
    "fallbackParent",
    "fallbackIndex",
    "startedAt",
    "completedAt",
    "status",
    "configured",
    "requested",
    "actual",
    "usage",
    "usageCoverage",
    "missingUsageFields",
    "receiptCoverage",
    "missingReceiptFields",
    "finishReason",
    "errorCategory",
})
_EXPORT_FIELDS = frozenset({
    "schema",
    "after",
    "nextCursor",
    "highWatermark",
    "count",
    "hasMore",
    "receipts",
    "coverageManifests",
})
_MODEL_FIELDS = frozenset({"provider", "model"})
_ACTUAL_FIELDS = frozenset({"provider", "model", "responseId", "evidenceSource"})
_USAGE_FIELD_ORDER = (
    "inputTotal",
    "inputNonCached",
    "cacheRead",
    "cacheWrite",
    "outputCandidates",
    "reasoningThinking",
    "toolUsePrompt",
    "providerReportedTotal",
    "serviceTier",
    "rawProviderUsage",
)
_USAGE_FIELDS = frozenset(_USAGE_FIELD_ORDER)
_COUNT_FIELDS = frozenset(_USAGE_FIELD_ORDER[:8])

_GEMINI_ALLOWED_USAGE_FIELDS = frozenset({
    "promptTokenCount",
    "cachedContentTokenCount",
    "candidatesTokenCount",
    "thoughtsTokenCount",
    "toolUsePromptTokenCount",
    "totalTokenCount",
    "serviceTier",
    "trafficType",
    "promptTokensDetails",
    "cacheTokensDetails",
    "candidatesTokensDetails",
    "toolUsePromptTokensDetails",
})
_GENERIC_ALLOWED_USAGE_FIELDS = frozenset({
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "prompt_tokens_details",
    "completion_tokens_details",
    "input_tokens_details",
    "output_tokens_details",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "inputTokens",
    "outputTokens",
    "totalTokens",
    "cacheReadInputTokens",
    "cacheWriteInputTokens",
    "service_tier",
})
_USAGE_DETAIL_FIELDS = frozenset({
    "modality",
    "tokenCount",
    "cached_tokens",
    "cache_creation_tokens",
    "cache_write_tokens",
    "audio_tokens",
    "reasoning_tokens",
    "accepted_prediction_tokens",
    "rejected_prediction_tokens",
})
_USAGE_DETAIL_CONTAINER_FIELDS = frozenset({
    "promptTokensDetails",
    "cacheTokensDetails",
    "candidatesTokensDetails",
    "toolUsePromptTokensDetails",
    "prompt_tokens_details",
    "completion_tokens_details",
    "input_tokens_details",
    "output_tokens_details",
})
_RAW_USAGE_ENUM_FIELDS = frozenset({"serviceTier", "trafficType", "service_tier"})
_RAW_USAGE_COUNT_FIELDS = (
    _GEMINI_ALLOWED_USAGE_FIELDS
    | _GENERIC_ALLOWED_USAGE_FIELDS
) - _RAW_USAGE_ENUM_FIELDS - _USAGE_DETAIL_CONTAINER_FIELDS
_MAX_RAW_USAGE_BYTES = 64 * 1024


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (
            value != value or value in {float("inf"), float("-inf")}
        ):
            return None
        return value
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _usage_details(value: Any) -> Any:
    """Filter nested accounting buckets so content cannot hide in them."""
    if isinstance(value, list):
        return [_usage_details(item) for item in value if isinstance(item, dict)]
    if not isinstance(value, dict):
        return None
    return {
        str(key): _json_value(item)
        for key, item in value.items()
        if key in _USAGE_DETAIL_FIELDS
    }


def _safe_usage_value(key: str, value: Any) -> Any:
    if key in _USAGE_DETAIL_CONTAINER_FIELDS:
        return _usage_details(value)
    return _json_value(value)


def allowed_provider_usage(
    provider: str,
    usage: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Return only provider accounting fields, never response content."""
    if not isinstance(usage, dict):
        return {}
    provider_name = (provider or "").strip().lower()
    allowed = (
        _GEMINI_ALLOWED_USAGE_FIELDS
        if provider_name in {"gemini", "google"}
        else _GENERIC_ALLOWED_USAGE_FIELDS
    )
    return {
        key: _safe_usage_value(key, value)
        for key, value in usage.items()
        if key in allowed
    }


def _nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    result = int(value)
    return result if result >= 0 else None


def _nested_nonnegative_int(
    value: dict[str, Any],
    containers: tuple[str, ...],
    fields: tuple[str, ...],
) -> Optional[int]:
    for container in containers:
        details = value.get(container)
        if not isinstance(details, dict):
            continue
        for field in fields:
            result = _nonnegative_int(details.get(field))
            if result is not None:
                return result
    return None


def _canonical_provider(value: Optional[str]) -> Optional[str]:
    provider = (value or "").strip().lower()
    if not provider:
        return None
    return "google" if provider == "gemini" else provider


def _rfc3339(timestamp: Optional[float]) -> Optional[str]:
    if timestamp is None:
        return None
    return (
        datetime
        .fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def canonical_receipt_bytes(receipt: dict[str, Any]) -> bytes:
    body = {
        key: value
        for key, value in receipt.items()
        if key not in {"ledgerSeq", "receiptDigest"}
    }
    return json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def receipt_digest(receipt: dict[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(canonical_receipt_bytes(receipt)).hexdigest()}"


def _require_exact_fields(
    value: Any,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch: missing={sorted(expected - actual)} "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def _validate_optional_string(value: Any, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"{label} must be a nonempty string or null")


def _validate_required_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _validate_timestamp(value: Any, label: str) -> datetime:
    _validate_required_string(value, label)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be RFC3339") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validate_sha256_digest(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or value != value.lower()
    ):
        raise ValueError(f"{label} must be sha256:<64 lowercase hex>")
    try:
        int(value[7:], 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be sha256:<64 lowercase hex>") from exc


def _validate_nonnegative_integer(value: Any, label: str, *, nullable: bool) -> None:
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        suffix = " or null" if nullable else ""
        raise ValueError(f"{label} must be a nonnegative integer{suffix}")


def _validate_string_array(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be an array of strings")


def _validate_accounting_usage(value: dict[str, Any]) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_RAW_USAGE_BYTES:
        raise ValueError(
            f"usage.rawProviderUsage exceeds {_MAX_RAW_USAGE_BYTES} bytes"
        )
    for field in _RAW_USAGE_COUNT_FIELDS:
        if field in value and value[field] is not None:
            _validate_nonnegative_integer(
                value[field],
                f"usage.rawProviderUsage.{field}",
                nullable=False,
            )
    for field in _RAW_USAGE_ENUM_FIELDS:
        if field in value and value[field] is not None:
            _validate_required_string(
                value[field],
                f"usage.rawProviderUsage.{field}",
            )
    for field in _USAGE_DETAIL_CONTAINER_FIELDS:
        if field not in value or value[field] is None:
            continue
        details = value[field]
        if not isinstance(details, (dict, list)):
            raise ValueError(
                f"usage.rawProviderUsage.{field} must be an object or array"
            )
        items = details if isinstance(details, list) else [details]
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(
                    f"usage.rawProviderUsage.{field}[{index}] must be an object"
                )
            for detail_field, detail_value in item.items():
                label = (
                    f"usage.rawProviderUsage.{field}[{index}].{detail_field}"
                )
                if detail_field == "modality":
                    _validate_required_string(detail_value, label)
                else:
                    _validate_nonnegative_integer(
                        detail_value,
                        label,
                        nullable=False,
                    )


def _missing_usage_fields(usage: dict[str, Any]) -> list[str]:
    return [field for field in _USAGE_FIELD_ORDER if usage[field] is None]


def _coverage_for_missing(missing_count: int, applicable_count: int) -> str:
    if missing_count == 0:
        return "complete"
    if missing_count == applicable_count:
        return "unavailable"
    return "partial"


def _missing_receipt_fields(
    receipt: dict[str, Any],
    missing_usage_fields: list[str],
) -> tuple[list[str], int]:
    missing: list[str] = []
    for field in ("runId", "turnId", "requestId", "sessionId"):
        if receipt[field] is None:
            missing.append(field)
    if receipt["trigger"] == "unknown":
        missing.append("trigger")

    if receipt["status"] == "succeeded":
        actual = receipt["actual"]
        for field in ("provider", "model", "responseId", "evidenceSource"):
            if actual[field] is None:
                missing.append(f"actual.{field}")
        if receipt["finishReason"] is None:
            missing.append("finishReason")
        status_evidence_count = 5
    else:
        if receipt["errorCategory"] is None:
            missing.append("errorCategory")
        status_evidence_count = 1

    missing.extend(f"usage.{field}" for field in missing_usage_fields)
    applicable_count = 4 + 1 + status_evidence_count + len(_USAGE_FIELD_ORDER)
    return missing, applicable_count


def validate_provider_usage_receipt(receipt: dict[str, Any]) -> None:
    """Fail closed unless *receipt* matches the exact shared CALL v1 wire."""
    value = _require_exact_fields(receipt, _CALL_FIELDS, "receipt")
    if value["schema"] != SCHEMA_NAME:
        raise ValueError(f"provider usage ledger schema mismatch: {value['schema']!r}")
    _validate_nonnegative_integer(value["ledgerSeq"], "ledgerSeq", nullable=False)
    digest = value["receiptDigest"]
    _validate_sha256_digest(digest, "receiptDigest")
    if digest != receipt_digest(value):
        raise ValueError("provider usage ledger digest mismatch")
    _validate_sha256_digest(
        value["producerCoverageDigest"],
        "producerCoverageDigest",
    )

    call_id = value["callId"]
    if not isinstance(call_id, str) or not call_id:
        raise ValueError("callId must be a nonempty UUID-like string")
    try:
        parsed_call_id = uuid.UUID(call_id)
    except (ValueError, AttributeError) as exc:
        raise ValueError("callId must be a nonempty UUID-like string") from exc
    if parsed_call_id.int == 0:
        raise ValueError("callId must not be the nil UUID")

    for field in (
        "runId",
        "turnId",
        "requestId",
        "sessionId",
        "retryOf",
        "fallbackParent",
    ):
        _validate_optional_string(value[field], field)
    if value["trigger"] not in TRIGGERS:
        raise ValueError(f"invalid trigger: {value['trigger']!r}")
    _validate_nonnegative_integer(value["attempt"], "attempt", nullable=False)
    if value["attempt"] < 1:
        raise ValueError("attempt must be a positive integer")
    _validate_nonnegative_integer(
        value["fallbackIndex"], "fallbackIndex", nullable=False
    )
    if value["fallbackParent"] is None and value["fallbackIndex"] != 0:
        raise ValueError("fallbackIndex must be 0 when fallbackParent is null")
    if value["fallbackParent"] is not None and value["fallbackIndex"] == 0:
        raise ValueError("fallbackIndex must be positive when fallbackParent is set")
    started_at = _validate_timestamp(value["startedAt"], "startedAt")
    completed_at = _validate_timestamp(value["completedAt"], "completedAt")
    if completed_at < started_at:
        raise ValueError("completedAt precedes startedAt")
    for field in ("finishReason", "errorCategory"):
        _validate_optional_string(value[field], field)
    if value["status"] not in STATUSES:
        raise ValueError(f"invalid status: {value['status']!r}")
    if value["status"] == "succeeded" and value["errorCategory"] is not None:
        raise ValueError("succeeded receipt must have null errorCategory")
    if value["status"] != "succeeded" and value["finishReason"] is not None:
        raise ValueError("non-succeeded receipt must have null finishReason")

    configured = _require_exact_fields(value["configured"], _MODEL_FIELDS, "configured")
    requested = _require_exact_fields(value["requested"], _MODEL_FIELDS, "requested")
    actual = _require_exact_fields(value["actual"], _ACTUAL_FIELDS, "actual")
    for label, identity in (("configured", configured), ("requested", requested)):
        for field in ("provider", "model"):
            _validate_required_string(identity[field], f"{label}.{field}")
    for field in ("provider", "model", "responseId", "evidenceSource"):
        _validate_optional_string(actual[field], f"actual.{field}")

    usage = _require_exact_fields(value["usage"], _USAGE_FIELDS, "usage")
    for field in _COUNT_FIELDS:
        _validate_nonnegative_integer(usage[field], f"usage.{field}", nullable=True)
    _validate_optional_string(usage["serviceTier"], "usage.serviceTier")
    raw_usage = usage["rawProviderUsage"]
    if raw_usage is not None:
        if not isinstance(raw_usage, dict):
            raise ValueError("usage.rawProviderUsage must be an object or null")
        provider = actual["provider"] or requested["provider"]
        if allowed_provider_usage(provider, raw_usage) != raw_usage:
            raise ValueError("usage.rawProviderUsage contains non-accounting fields")
        _validate_accounting_usage(raw_usage)

    if value["usageCoverage"] not in COVERAGES:
        raise ValueError(f"invalid usageCoverage: {value['usageCoverage']!r}")
    if value["receiptCoverage"] not in COVERAGES:
        raise ValueError(f"invalid receiptCoverage: {value['receiptCoverage']!r}")
    _validate_string_array(value["missingUsageFields"], "missingUsageFields")
    _validate_string_array(value["missingReceiptFields"], "missingReceiptFields")

    expected_missing_usage = _missing_usage_fields(usage)
    if value["missingUsageFields"] != expected_missing_usage:
        raise ValueError(
            "missingUsageFields must be the exact ordered null usage field list"
        )
    expected_usage_coverage = _coverage_for_missing(
        len(expected_missing_usage), len(_USAGE_FIELD_ORDER)
    )
    if value["usageCoverage"] != expected_usage_coverage:
        raise ValueError(
            f"usageCoverage must be {expected_usage_coverage!r} for the usage fields"
        )

    expected_missing_receipt, applicable_count = _missing_receipt_fields(
        value, expected_missing_usage
    )
    if value["missingReceiptFields"] != expected_missing_receipt:
        raise ValueError(
            "missingReceiptFields must be the exact ordered applicable null field list"
        )
    expected_receipt_coverage = _coverage_for_missing(
        len(expected_missing_receipt), applicable_count
    )
    if value["receiptCoverage"] != expected_receipt_coverage:
        raise ValueError(
            f"receiptCoverage must be {expected_receipt_coverage!r} "
            "for the applicable receipt fields"
        )


def validate_provider_usage_export(
    page: dict[str, Any],
    *,
    expected_after: Optional[int] = None,
    expected_family: str = "hermes",
) -> None:
    """Fail closed unless *page* matches the exact shared EXPORT v1 wire."""
    value = _require_exact_fields(page, _EXPORT_FIELDS, "export")
    if value["schema"] != EXPORT_SCHEMA_NAME:
        raise ValueError(f"provider usage export schema mismatch: {value['schema']!r}")
    for field in ("after", "nextCursor", "highWatermark", "count"):
        _validate_nonnegative_integer(value[field], field, nullable=False)
    if expected_after is not None and value["after"] != expected_after:
        raise ValueError(
            f"after mismatch: expected={expected_after} actual={value['after']}"
        )
    if value["highWatermark"] < value["after"]:
        raise ValueError("provider usage ledger moved backwards")
    if not isinstance(value["hasMore"], bool):
        raise ValueError("hasMore must be a boolean")
    receipts = value["receipts"]
    if not isinstance(receipts, list):
        raise ValueError("receipts must be an array")
    if value["count"] != len(receipts):
        raise ValueError("count must equal receipts.length")
    for receipt in receipts:
        validate_provider_usage_receipt(receipt)
    coverage_manifests = value["coverageManifests"]
    if not isinstance(coverage_manifests, list):
        raise ValueError("coverageManifests must be an array")
    from agent.provider_usage_coverage import validate_provider_usage_coverage

    manifest_digests: list[str] = []
    for manifest in coverage_manifests:
        validate_provider_usage_coverage(
            manifest,
            expected_family=expected_family,
        )
        manifest_digests.append(manifest["manifestDigest"])
    if manifest_digests != sorted(manifest_digests) or len(manifest_digests) != len(
        set(manifest_digests)
    ):
        raise ValueError(
            "coverageManifests must be ordered by unique ascending manifestDigest"
        )
    receipt_digests = {receipt["producerCoverageDigest"] for receipt in receipts}
    if set(manifest_digests) != receipt_digests:
        raise ValueError(
            "coverageManifests digests must exactly match receipt producerCoverageDigest values"
        )
    sequences = [receipt["ledgerSeq"] for receipt in receipts]
    if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
        raise ValueError("receipts must be ordered by unique ascending ledgerSeq")
    if any(sequence <= value["after"] for sequence in sequences):
        raise ValueError("every receipt ledgerSeq must be greater than after")
    if sequences and sequences[-1] > value["highWatermark"]:
        raise ValueError("receipt ledgerSeq must not exceed highWatermark")
    expected_next = sequences[-1] if sequences else value["after"]
    if value["nextCursor"] != expected_next:
        raise ValueError("nextCursor must equal the last receipt ledgerSeq or after")
    if value["hasMore"] and value["highWatermark"] <= value["nextCursor"]:
        raise ValueError("hasMore=true requires a row after nextCursor")
    if value["hasMore"] and not sequences:
        raise ValueError("hasMore=true requires cursor progress")
    if not value["hasMore"] and value["highWatermark"] > value["nextCursor"]:
        raise ValueError("hasMore=false conflicts with highWatermark")


def build_provider_usage_receipt(
    *,
    ledger_seq: Optional[int],
    call_id: str,
    run_id: Optional[str],
    turn_id: Optional[str],
    request_id: Optional[str],
    session_id: Optional[str],
    trigger: str,
    producer_coverage_digest: str,
    attempt: int,
    retry_of: Optional[str],
    fallback_parent: Optional[str],
    fallback_index: int,
    started_at: Optional[float],
    completed_at: Optional[float],
    status: str,
    configured_provider: str,
    configured_model: Optional[str],
    requested_provider: str,
    requested_model: Optional[str],
    actual_provider: Optional[str],
    actual_model: Optional[str],
    response_id: Optional[str],
    evidence_source: Optional[str],
    raw_usage: Optional[dict[str, Any]],
    finish_reason: Optional[str],
    error_category: Optional[str],
) -> dict[str, Any]:
    canonical_configured_provider = _canonical_provider(configured_provider)
    canonical_requested_provider = _canonical_provider(requested_provider)
    _validate_required_string(canonical_configured_provider, "configured.provider")
    _validate_required_string(configured_model, "configured.model")
    _validate_required_string(canonical_requested_provider, "requested.provider")
    _validate_required_string(requested_model, "requested.model")

    provider_for_usage = (
        _canonical_provider(actual_provider) or canonical_requested_provider
    )
    raw = allowed_provider_usage(provider_for_usage or "", raw_usage)
    is_google = provider_for_usage == "google"
    prompt = _nonnegative_int(
        raw.get("promptTokenCount")
        if is_google
        else raw.get("prompt_tokens", raw.get("input_tokens", raw.get("inputTokens")))
    )
    cached = _nonnegative_int(
        raw.get("cachedContentTokenCount")
        if is_google
        else raw.get(
            "cache_read_tokens",
            raw.get("cache_read_input_tokens", raw.get("cacheReadInputTokens")),
        )
    )
    if cached is None and not is_google:
        cached = _nested_nonnegative_int(
            raw,
            ("prompt_tokens_details", "input_tokens_details"),
            ("cached_tokens",),
        )
    cache_write = None
    if not is_google:
        cache_write = _nonnegative_int(
            raw.get(
                "cache_write_tokens",
                raw.get("cache_creation_input_tokens", raw.get("cacheWriteInputTokens")),
            )
        )
        if cache_write is None:
            cache_write = _nested_nonnegative_int(
                raw,
                ("prompt_tokens_details", "input_tokens_details"),
                ("cache_creation_tokens", "cache_write_tokens"),
            )
    candidates = _nonnegative_int(
        raw.get("candidatesTokenCount")
        if is_google
        else raw.get(
            "completion_tokens", raw.get("output_tokens", raw.get("outputTokens"))
        )
    )
    thoughts = _nonnegative_int(
        raw.get("thoughtsTokenCount") if is_google else raw.get("reasoning_tokens")
    )
    if thoughts is None and not is_google:
        thoughts = _nested_nonnegative_int(
            raw,
            ("completion_tokens_details", "output_tokens_details"),
            ("reasoning_tokens",),
        )
    tool_use = (
        _nonnegative_int(raw.get("toolUsePromptTokenCount")) if is_google else None
    )
    total = _nonnegative_int(
        raw.get("totalTokenCount")
        if is_google
        else raw.get("total_tokens", raw.get("totalTokens"))
    )
    input_non_cached = (
        prompt - cached
        if prompt is not None and cached is not None and cached <= prompt
        else None
    )
    service_tier = raw.get("serviceTier") if is_google else raw.get("service_tier")
    if not isinstance(service_tier, str):
        service_tier = None

    receipt: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "ledgerSeq": ledger_seq,
        "callId": call_id,
        "runId": run_id,
        "turnId": turn_id,
        "requestId": request_id,
        "sessionId": session_id,
        "trigger": trigger,
        "producerCoverageDigest": producer_coverage_digest,
        "attempt": attempt,
        "retryOf": retry_of,
        "fallbackParent": fallback_parent,
        "fallbackIndex": fallback_index,
        "startedAt": _rfc3339(started_at),
        "completedAt": _rfc3339(completed_at),
        "status": status,
        "configured": {
            "provider": canonical_configured_provider,
            "model": configured_model,
        },
        "requested": {
            "provider": canonical_requested_provider,
            "model": requested_model,
        },
        "actual": {
            "provider": _canonical_provider(actual_provider),
            "model": actual_model,
            "responseId": response_id,
            "evidenceSource": evidence_source,
        },
        "usage": {
            "inputTotal": prompt,
            "inputNonCached": input_non_cached,
            "cacheRead": cached,
            "cacheWrite": cache_write,
            "outputCandidates": candidates,
            "reasoningThinking": thoughts,
            "toolUsePrompt": tool_use,
            "providerReportedTotal": total,
            "serviceTier": service_tier,
            "rawProviderUsage": raw if raw else None,
        },
        "finishReason": finish_reason,
        "errorCategory": error_category,
    }
    missing_usage = _missing_usage_fields(receipt["usage"])
    receipt["usageCoverage"] = _coverage_for_missing(
        len(missing_usage), len(_USAGE_FIELD_ORDER)
    )
    receipt["missingUsageFields"] = missing_usage
    missing_receipt, applicable_count = _missing_receipt_fields(receipt, missing_usage)
    receipt["receiptCoverage"] = _coverage_for_missing(
        len(missing_receipt), applicable_count
    )
    receipt["missingReceiptFields"] = missing_receipt
    receipt["receiptDigest"] = receipt_digest(receipt)
    return receipt
