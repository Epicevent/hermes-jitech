"""Runtime capture seam for authoritative provider-call usage receipts.

The caller owns request content; this module only receives provider/model
coordinates and provider-returned accounting metadata.  A context variable
correlates calls made inside a conversation turn.  Calls outside that context
still append a receipt with nullable run/turn/request/session identifiers.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import inspect
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterator, Optional, TypeVar
from urllib.parse import urlparse


logger = logging.getLogger(__name__)
_T = TypeVar("_T")


@dataclass
class _ContextState:
    call_index: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def next_call_index(self) -> int:
        with self.lock:
            self.call_index += 1
            return self.call_index


@dataclass(frozen=True)
class ProviderUsageContext:
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    turn_id: Optional[str] = None
    request_id: Optional[str] = None
    trigger: str = "unknown"
    configured_provider: Optional[str] = None
    configured_model: Optional[str] = None
    db_path: Any = None
    state: _ContextState = field(default_factory=_ContextState)


_CURRENT_CONTEXT: contextvars.ContextVar[Optional[ProviderUsageContext]] = (
    contextvars.ContextVar("provider_usage_context", default=None)
)


def current_provider_usage_context() -> Optional[ProviderUsageContext]:
    return _CURRENT_CONTEXT.get()


def normalize_provider_usage_trigger(source: Any) -> str:
    normalized = str(source or "unknown").strip().lower()
    if normalized in {"cron", "heartbeat", "manual", "memory", "overflow"}:
        return normalized
    if not normalized or normalized == "unknown":
        return "unknown"
    return "user"


@contextlib.contextmanager
def bind_provider_usage_context(
    *,
    session_id: Optional[str],
    run_id: Optional[str],
    turn_id: Optional[str],
    request_id: Optional[str] = None,
    trigger: str = "unknown",
    configured_provider: Optional[str] = None,
    configured_model: Optional[str] = None,
    db_path: Any = None,
) -> Iterator[ProviderUsageContext]:
    context = ProviderUsageContext(
        session_id=session_id,
        run_id=run_id,
        turn_id=turn_id,
        request_id=request_id,
        trigger=trigger,
        configured_provider=configured_provider,
        configured_model=configured_model,
        db_path=db_path,
    )
    token = _CURRENT_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_CONTEXT.reset(token)


@dataclass(frozen=True)
class ProviderAttempt:
    call_id: str
    request_id: Optional[str]
    api_call_index: int
    attempt: int
    retry_of: Optional[str]
    fallback_parent: Optional[str]
    fallback_index: int
    provider: str
    model: str


class ProviderAttemptSeries:
    """Correlate physical retries and provider fallbacks for one logical call."""

    def __init__(self, *, api_call_index: Optional[int] = None):
        context = current_provider_usage_context()
        self.request_id = (
            context.request_id or str(uuid.uuid4())
            if context is not None
            else None
        )
        self.api_call_index = api_call_index or (
            context.state.next_call_index() if context is not None else 1
        )
        self._attempt = 0
        self._fallback_index = 0
        self._fallback_parent: Optional[str] = None
        self._previous_call_id: Optional[str] = None
        self._previous_provider: Optional[str] = None

    def next(self, *, provider: str, model: str) -> ProviderAttempt:
        normalized_provider = (provider or "unknown").strip().lower() or "unknown"
        requested_model = (model or "unknown").strip() or "unknown"
        if (
            self._previous_provider is not None
            and normalized_provider != self._previous_provider
        ):
            self._fallback_index += 1
            self._fallback_parent = self._previous_call_id
        self._attempt += 1
        attempt = ProviderAttempt(
            call_id=str(uuid.uuid4()),
            request_id=self.request_id,
            api_call_index=self.api_call_index,
            attempt=self._attempt,
            retry_of=self._previous_call_id,
            fallback_parent=self._fallback_parent,
            fallback_index=self._fallback_index,
            provider=normalized_provider,
            model=requested_model,
        )
        self._previous_call_id = attempt.call_id
        self._previous_provider = normalized_provider
        return attempt


_USAGE_KEYS = (
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
)


def _plain_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    for method_name in ("model_dump", "to_dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                dumped = method()
            except Exception:
                dumped = None
            if inspect.isawaitable(dumped):
                close = getattr(dumped, "close", None)
                if callable(close):
                    close()
                dumped = None
            if isinstance(dumped, dict):
                return _plain_value(dumped)
    try:
        attributes = vars(value)
    except TypeError:
        return None
    return {
        str(key): _plain_value(item)
        for key, item in attributes.items()
        if not str(key).startswith("_")
    }


def provider_raw_usage(value: Any) -> Optional[dict[str, Any]]:
    plain = _plain_value(value)
    if not isinstance(plain, dict):
        return None
    return {key: plain[key] for key in _USAGE_KEYS if key in plain} or None


def response_with_provider_receipt(response: Any, receipt: Any) -> Any:
    if not isinstance(receipt, dict):
        return response
    plain = _plain_value(response)
    if not isinstance(plain, dict):
        plain = {}
    plain["providerReceipt"] = _plain_value(receipt)
    return plain


def provider_from_client(client: Any, fallback: Optional[str] = None) -> str:
    provider = (fallback or "").strip().lower()
    if provider not in {"", "auto", "custom"}:
        return provider
    base_url = str(getattr(client, "base_url", "") or "")
    host = (urlparse(base_url).hostname or "").lower()
    host_map = (
        ("openrouter.ai", "openrouter"),
        ("nousresearch.com", "nous"),
        ("anthropic.com", "anthropic"),
        ("generativelanguage.googleapis.com", "gemini"),
        ("cloudcode-pa.googleapis.com", "google-gemini-cli"),
        ("api.openai.com", "openai"),
        ("chatgpt.com", "openai-codex"),
        ("api.x.ai", "xai"),
        ("moonshot.ai", "kimi"),
        ("bigmodel.cn", "zai"),
    )
    for suffix, label in host_map:
        if host == suffix or host.endswith(f".{suffix}"):
            return label
    return provider or "unknown"


def _response_evidence(response: Any, provider: str) -> dict[str, Any]:
    plain = _plain_value(response)
    value = plain if isinstance(plain, dict) else {}
    provider_receipt = value.get("providerReceipt")
    if not isinstance(provider_receipt, dict):
        provider_receipt = None

    usage_value = value.get("usageMetadata")
    if usage_value is None:
        usage_value = value.get("provider_usage") or value.get("providerUsage")
    if usage_value is None:
        usage_value = value.get("usage")
    if usage_value is None and isinstance(value.get("metrics"), dict):
        usage_value = value.get("metrics")

    model_version = value.get("modelVersion")
    evidence_source = None
    if isinstance(model_version, str) and model_version:
        actual_model = model_version
        evidence_source = "gemini_response.modelVersion"
    else:
        actual_model = value.get("model")
        if isinstance(actual_model, str) and actual_model:
            evidence_source = "response.model"
        else:
            actual_model = None

    response_id = value.get("id") or value.get("responseId")
    response_metadata = value.get("ResponseMetadata")
    if not response_id and isinstance(response_metadata, dict):
        response_id = response_metadata.get("RequestId")

    finish_reason = value.get("stop_reason") or value.get("finishReason")
    choices = value.get("choices")
    if finish_reason is None and isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            finish_reason = first.get("finish_reason")

    if provider_receipt is not None:
        model_version = provider_receipt.get("modelVersion")
        if isinstance(model_version, str) and model_version:
            actual_model = model_version
            evidence_source = provider_receipt.get("evidenceSource")
        response_id = provider_receipt.get("responseId") or response_id
        usage_value = provider_receipt.get("usageMetadata") or usage_value
        finish_reason = provider_receipt.get("finishReason") or finish_reason

    return {
        "actual_provider": provider if actual_model is not None else None,
        "actual_model": actual_model,
        "response_id": response_id if isinstance(response_id, str) else None,
        "evidence_source": evidence_source,
        "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
        "usage": provider_raw_usage(usage_value),
    }


def _status_for_exception(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "cancelled"
    if isinstance(exc, (InterruptedError, KeyboardInterrupt)):
        return "interrupted"
    return "failed"


def _record_attempt(
    attempt: ProviderAttempt,
    *,
    started_at: float,
    completed_at: float,
    status: str,
    response: Any = None,
    error: Optional[BaseException] = None,
) -> Optional[dict[str, Any]]:
    context = current_provider_usage_context()
    configured_provider = (
        context.configured_provider if context and context.configured_provider else attempt.provider
    )
    configured_model = (
        context.configured_model if context and context.configured_model else attempt.model
    )
    evidence = (
        _response_evidence(response, attempt.provider)
        if status == "succeeded"
        else {
            "actual_provider": None,
            "actual_model": None,
            "response_id": None,
            "evidence_source": None,
            "finish_reason": None,
            "usage": None,
        }
    )

    db = None
    owns_db = False
    try:
        from hermes_cli.config import get_hermes_home
        from hermes_state import SessionDB

        db_path = context.db_path if context and context.db_path else get_hermes_home() / "state.db"
        db = SessionDB(db_path)
        owns_db = True
        return db.record_provider_call(
            context.session_id if context else None,
            call_id=attempt.call_id,
            request_id=attempt.request_id,
            api_call_index=attempt.api_call_index,
            attempt=attempt.attempt,
            fallback_index=attempt.fallback_index,
            configured_provider=configured_provider,
            configured_model=configured_model,
            requested_provider=attempt.provider,
            requested_model=attempt.model,
            actual_provider=evidence["actual_provider"],
            actual_model=evidence["actual_model"],
            response_id=evidence["response_id"],
            evidence_source=evidence["evidence_source"],
            finish_reason=evidence["finish_reason"],
            usage=evidence["usage"],
            run_id=context.run_id if context else None,
            turn_id=context.turn_id if context else None,
            retry_of=attempt.retry_of,
            fallback_parent=attempt.fallback_parent,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            trigger=context.trigger if context else "unknown",
            error_category=type(error).__name__ if error is not None else None,
        )
    except Exception:
        logger.exception(
            "Provider usage capture failed (provider=%s model=%s call=%s)",
            attempt.provider,
            attempt.model,
            attempt.call_id,
        )
        return None
    finally:
        if owns_db and db is not None:
            db.close()


def capture_provider_call(
    invoke: Callable[[], _T],
    *,
    provider: str,
    model: str,
    series: Optional[ProviderAttemptSeries] = None,
    response_transform: Optional[Callable[[_T], Any]] = None,
) -> _T:
    attempt = (series or ProviderAttemptSeries()).next(provider=provider, model=model)
    started_at = time.time()
    try:
        response = invoke()
    except BaseException as exc:
        _record_attempt(
            attempt,
            started_at=started_at,
            completed_at=time.time(),
            status=_status_for_exception(exc),
            error=exc,
        )
        raise
    try:
        observed = response_transform(response) if response_transform else response
    except Exception:
        # Receipt observation is diagnostic.  A buggy or provider-specific
        # observation adapter must never turn a successful provider response
        # into an application-visible failure.
        logger.exception(
            "Provider usage response observation failed "
            "(provider=%s model=%s call=%s)",
            attempt.provider,
            attempt.model,
            attempt.call_id,
        )
        observed = response
    _record_attempt(
        attempt,
        started_at=started_at,
        completed_at=time.time(),
        status="succeeded",
        response=observed,
    )
    return response


async def capture_provider_call_async(
    invoke: Callable[[], Awaitable[_T]],
    *,
    provider: str,
    model: str,
    series: Optional[ProviderAttemptSeries] = None,
    response_transform: Optional[Callable[[_T], Any]] = None,
) -> _T:
    attempt = (series or ProviderAttemptSeries()).next(provider=provider, model=model)
    started_at = time.time()
    try:
        response = await invoke()
    except BaseException as exc:
        _record_attempt(
            attempt,
            started_at=started_at,
            completed_at=time.time(),
            status=_status_for_exception(exc),
            error=exc,
        )
        raise
    try:
        observed = response_transform(response) if response_transform else response
    except Exception:
        logger.exception(
            "Provider usage response observation failed "
            "(provider=%s model=%s call=%s)",
            attempt.provider,
            attempt.model,
            attempt.call_id,
        )
        observed = response
    _record_attempt(
        attempt,
        started_at=started_at,
        completed_at=time.time(),
        status="succeeded",
        response=observed,
    )
    return response
