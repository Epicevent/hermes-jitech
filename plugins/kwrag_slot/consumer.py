"""Hermes-owned consumer for an explicitly invoked KWRAG slot runtime.

This module owns correlation and consumption receipts only.  The caller owns
whether retrieval runs, the backend implementation, query construction,
stopping, prompt assembly, and whether returned evidence is shown to a model.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from plugins.kwrag_slot.manifest import canonical_json_bytes, load_component_manifest


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_BINDING_FIELDS = {
    "schema_version",
    "enabled",
    "component_digest",
    "runtime_binding_digest",
    "expected_index_manifest",
    "expected_pipeline_fingerprint",
    "max_result_characters",
}


class HermesSlotRetrievalError(ValueError):
    """The explicit Hermes/KWRAG boundary could not be verified."""


class SlotRuntimeProtocol(Protocol):
    def search_exchange(self, request: Any) -> Any: ...


class ConsumptionReceiptSink(Protocol):
    def write(self, receipt: Mapping[str, Any]) -> str: ...


class FileConsumptionReceiptSink:
    """Persist canonical consumption receipts using the KWRAG POSIX writer."""

    def __init__(self, path: Path):
        if not path.is_absolute():
            raise HermesSlotRetrievalError("consumption receipt path must be absolute")
        try:
            from kwrag.operation import ReceiptWriter
        except ImportError as exc:
            raise HermesSlotRetrievalError("embedded KWRAG component is unavailable") from exc
        self._writer = ReceiptWriter(path)

    def write(self, receipt: Mapping[str, Any]) -> str:
        result = self._writer.write(dict(receipt))
        if result.status != "written":
            raise HermesSlotRetrievalError("consumption receipt was not written")
        return result.digest


def _digest(value: object, field: str) -> str:
    text = str(value or "")
    if not _SHA256.fullmatch(text):
        raise HermesSlotRetrievalError(f"{field} is not a canonical SHA-256 digest")
    return text


@dataclass(frozen=True)
class HermesSlotRetrievalBinding:
    enabled: bool
    component_digest: str
    runtime_binding_digest: str | None
    expected_index_manifest: str | None
    expected_pipeline_fingerprint: str | None
    max_result_characters: int

    @classmethod
    def from_mapping(cls, raw: Any) -> "HermesSlotRetrievalBinding":
        if not isinstance(raw, Mapping) or set(raw) != _BINDING_FIELDS:
            raise HermesSlotRetrievalError("Hermes slot retrieval binding fields are invalid")
        if raw.get("schema_version") != "hermes-kwrag-slot-binding-v1":
            raise HermesSlotRetrievalError("Hermes slot retrieval binding schema is invalid")
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            raise HermesSlotRetrievalError("Hermes slot retrieval enabled flag is invalid")
        manifest = load_component_manifest()
        component_digest = _digest(raw.get("component_digest"), "component digest")
        if component_digest != manifest["component_wheel"]["sha256"]:
            raise HermesSlotRetrievalError("Hermes binding does not name the embedded component")
        max_chars = raw.get("max_result_characters")
        if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 0 <= max_chars <= 80_000:
            raise HermesSlotRetrievalError("Hermes result character budget is invalid")
        runtime_digest = raw.get("runtime_binding_digest")
        manifest_digest = raw.get("expected_index_manifest")
        pipeline_digest = raw.get("expected_pipeline_fingerprint")
        if enabled:
            return cls(
                enabled=True,
                component_digest=component_digest,
                runtime_binding_digest=_digest(runtime_digest, "runtime binding digest"),
                expected_index_manifest=_digest(manifest_digest, "index manifest digest"),
                expected_pipeline_fingerprint=_digest(pipeline_digest, "pipeline fingerprint"),
                max_result_characters=max_chars,
            )
        if any(value is not None for value in (runtime_digest, manifest_digest, pipeline_digest)):
            raise HermesSlotRetrievalError("disabled Hermes retrieval must not retain a runtime binding")
        if max_chars != 0:
            raise HermesSlotRetrievalError("disabled Hermes retrieval must have a zero result budget")
        return cls(False, component_digest, None, None, None, 0)


@dataclass
class HermesSlotRetrievalResult:
    results: tuple[dict[str, Any], ...]
    result_receipt: dict[str, Any]
    result_receipt_digest: str
    result_receipt_status: str
    consumption_receipt: dict[str, Any] | None = None
    consumption_receipt_digest: str | None = None
    consumption_receipt_status: str = "pending"
    _receipt_sink: ConsumptionReceiptSink | None = field(repr=False, compare=False, default=None)

    def record_prompt_consumption(
        self,
        *,
        session_binding_digest: str,
        prompt_context_digest: str,
    ) -> str:
        if self.consumption_receipt_status != "pending" or self.consumption_receipt is not None:
            raise HermesSlotRetrievalError("retrieval evidence was already consumed")
        receipt = {
            "schema_version": "hermes-kwrag-consumption-receipt-v1",
            "consumer_family": "hermes",
            "consumption_status": "assembled_into_ephemeral_user_context",
            "component_digest": self.result_receipt["component_digest"],
            "runtime_binding_digest": self.result_receipt["runtime_binding_digest"],
            "session_binding_digest": _digest(session_binding_digest, "session binding digest"),
            "prompt_context_digest": _digest(prompt_context_digest, "prompt context digest"),
            "request_id": self.result_receipt["request_id"],
            "operation_id": self.result_receipt["operation_id"],
            "run_id": self.result_receipt["run_id"],
            "attempt": self.result_receipt["attempt"],
            "result_digest": self.result_receipt["result_digest"],
            "operation_receipt_digest": self.result_receipt["operation_receipt_digest"],
            "result_receipt_digest": self.result_receipt_digest,
        }
        receipt_digest = "sha256:" + hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()
        if self._receipt_sink is None:
            raise HermesSlotRetrievalError("consumption receipt sink is unavailable")
        written_digest = self._receipt_sink.write(receipt)
        if written_digest != receipt_digest:
            raise HermesSlotRetrievalError("written consumption receipt digest is not bound")
        self.consumption_receipt = receipt
        self.consumption_receipt_digest = receipt_digest
        self.consumption_receipt_status = "written"
        return receipt_digest


class HermesSlotRetrievalConsumer:
    """Execute only a caller-authorized search and bind the consumed evidence."""

    def __init__(
        self,
        binding: HermesSlotRetrievalBinding,
        runtime: SlotRuntimeProtocol | None,
        receipt_sink: ConsumptionReceiptSink | None,
    ):
        if binding.enabled != (runtime is not None) or binding.enabled != (receipt_sink is not None):
            raise HermesSlotRetrievalError("runtime and receipt sink do not match the enabled binding")
        self._binding = binding
        self._runtime = runtime
        self._receipt_sink = receipt_sink

    def search(self, request: Mapping[str, Any]) -> HermesSlotRetrievalResult:
        if not self._binding.enabled or self._runtime is None or self._receipt_sink is None:
            raise HermesSlotRetrievalError("Hermes slot retrieval is disabled")
        try:
            from kwrag.slot_consumer import verify_slot_search_exchange
        except ImportError as exc:
            raise HermesSlotRetrievalError("embedded KWRAG component is unavailable") from exc
        exchange = self._runtime.search_exchange(dict(request))
        verified = verify_slot_search_exchange(
            request,
            exchange.response,
            exchange.operation_receipt,
            expected_index_manifest=self._binding.expected_index_manifest,
            expected_pipeline_fingerprint=self._binding.expected_pipeline_fingerprint,
            max_result_characters=self._binding.max_result_characters,
        )
        receipt = {
            "schema_version": "hermes-kwrag-result-receipt-v1",
            "consumer_family": "hermes",
            "adapter_status": "verified_by_product_adapter",
            "component_digest": self._binding.component_digest,
            "runtime_binding_digest": self._binding.runtime_binding_digest,
            "request_id": verified.request_id,
            "operation_id": verified.operation_id,
            "run_id": verified.run_id,
            "attempt": verified.attempt,
            "index_manifest": verified.index_manifest,
            "pipeline_fingerprint": verified.pipeline_fingerprint,
            "result_status": verified.result_status,
            "result_digest": verified.result_digest,
            "operation_receipt_digest": verified.operation_receipt_digest,
            "result_count": verified.result_count,
            "result_characters": verified.result_characters,
        }
        receipt_bytes = canonical_json_bytes(receipt)
        receipt_digest = "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
        written_digest = self._receipt_sink.write(receipt)
        if written_digest != receipt_digest:
            raise HermesSlotRetrievalError("written result receipt digest is not bound")
        return HermesSlotRetrievalResult(
            results=tuple(verified.results()),
            result_receipt=receipt,
            result_receipt_digest=receipt_digest,
            result_receipt_status="written",
            _receipt_sink=self._receipt_sink,
        )
