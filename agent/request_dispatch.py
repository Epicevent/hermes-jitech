"""Synchronized handoff between pre-dispatch cancellation and provider calls."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import uuid
from collections.abc import Callable
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit


class DispatchAttemptsClosed(RuntimeError):
    """A terminal owner closed the provider-attempt gate."""

    def __init__(self, cause: str) -> None:
        super().__init__(f"provider dispatch attempts closed: {cause}")
        self.cause = cause


class ProviderAttemptLedgerRequired(RuntimeError):
    """A retry would need its own durable provider-attempt record."""


class FinalProviderBindingUnsupported(RuntimeError):
    """The selected provider reshapes requests below the bound SDK seam."""


_SDK_LEAVES_WITHOUT_ATOMIC_REQUEST_BINDING = frozenset({
    "anthropic.Anthropic",
    "anthropic.lib.bedrock._client.AnthropicBedrock",
    "botocore.client.BedrockRuntime",
    "openai.OpenAI",
    "openai.lib.azure.AzureOpenAI",
})


_FORBIDDEN_SECRET_KEYS = frozenset({
    "access_token",
    "api-key",
    "api_key",
    "apikey",
    "authorization",
    "bearer_token",
    "credential",
    "credentials",
    "password",
    "proxy-authorization",
    "refresh_token",
    "secret",
    "x-api-key",
})


def _snapshot_request_value(value: Any) -> Any:
    """Copy request containers while preserving provider SDK sentinel identity."""

    if isinstance(value, Mapping):
        return {key: _snapshot_request_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_request_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_request_value(item) for item in value)
    return value


def _canonical_request_projection(
    value: Any,
    *,
    depth: int = 0,
    parent_key: str | None = None,
) -> Any:
    """Return a deterministic projection, rejecting secret-bearing kwargs."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("provider request contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        projected: dict[str, Any] = {}
        raw_keys = list(value)
        if any(not isinstance(raw_key, str) for raw_key in raw_keys):
            raise TypeError("provider request mapping keys must be strings")
        for raw_key in sorted(raw_keys):
            lower_key = raw_key.lower()
            top_level_secret = depth == 0 and lower_key in _FORBIDDEN_SECRET_KEYS
            auth_header = (
                depth == 1
                and parent_key in {"extra_headers", "headers"}
                and lower_key in {"authorization", "proxy-authorization", "x-api-key"}
            )
            if top_level_secret or auth_header:
                raise ValueError(
                    f"provider request kwargs contain forbidden secret field: {raw_key}"
                )
            projected[raw_key] = _canonical_request_projection(
                value[raw_key],
                depth=depth + 1,
                parent_key=lower_key,
            )
        return projected
    if isinstance(value, (list, tuple)):
        return [
            _canonical_request_projection(
                item,
                depth=depth + 1,
                parent_key=parent_key,
            )
            for item in value
        ]
    as_dict = getattr(value, "as_dict", None)
    if callable(as_dict):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "$value": _canonical_request_projection(
                as_dict(),
                depth=depth + 1,
                parent_key=parent_key,
            ),
        }
    state = getattr(value, "__dict__", None)
    if isinstance(state, dict):
        return {
            "$type": f"{type(value).__module__}.{type(value).__qualname__}",
            "$value": _canonical_request_projection(
                {name: item for name, item in state.items() if not name.startswith("_")},
                depth=depth + 1,
                parent_key=parent_key,
            ),
        }
    raise TypeError(
        "provider request contains a value without a canonical projection: "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def provider_leaf_adapter_identity(client: Any) -> str:
    """Return a bounded, non-secret identity for the actual SDK adapter."""

    identity = f"{type(client).__module__}.{type(client).__qualname__}"
    if not identity or len(identity) > 240:
        raise ValueError("provider leaf adapter identity is invalid")
    return identity


def require_authoritative_leaf_adapter(client: Any) -> str:
    """Require an atomic final-request adapter for retrieval evidence."""

    identity = provider_leaf_adapter_identity(client)
    if identity == "agent.gemini_native_adapter.GeminiNativeClient":
        import httpx
        from agent.gemini_native_adapter import GeminiNativeClient

        transport = getattr(client, "_http", None)
        hooks = getattr(transport, "event_hooks", {})
        if (
            type(client) is GeminiNativeClient
            and type(transport) is httpx.Client
            and type(getattr(transport, "_transport", None)) is httpx.HTTPTransport
            and getattr(getattr(transport, "build_request", None), "__func__", None)
            is httpx.Client.build_request
            and getattr(
                getattr(transport._transport, "handle_request", None), "__func__", None
            )
            is httpx.HTTPTransport.handle_request
            and not getattr(client, "_default_headers", None)
            and not any(hooks.values())
        ):
            return "agent.gemini_native_adapter.GeminiNativeAtomicHttpRequest/v1"
        raise FinalProviderBindingUnsupported(
            "Gemini retrieval evidence requires the exact unhooked atomic HTTP leaf"
        )
    if identity in _SDK_LEAVES_WITHOUT_ATOMIC_REQUEST_BINDING:
        raise FinalProviderBindingUnsupported(
            "retrieval evidence dispatch requires an atomic final serialized "
            f"request boundary; {identity} does not expose one"
        )
    raise FinalProviderBindingUnsupported(
        f"provider leaf request binding is unavailable for {identity}"
    )


def require_retrieval_evidence_dispatch_capability(agent: Any) -> str:
    """Admit only the configured native Gemini atomic request boundary."""

    from agent.gemini_native_adapter import is_native_gemini_base_url

    if (
        str(getattr(agent, "provider", "") or "") == "gemini"
        and str(getattr(agent, "api_mode", "") or "chat_completions") == "chat_completions"
        and is_native_gemini_base_url(str(getattr(agent, "base_url", "") or ""))
    ):
        return "agent.gemini_native_adapter.GeminiNativeAtomicHttpRequest/v1"
    raise FinalProviderBindingUnsupported(
        "retrieval evidence requires the configured native Gemini atomic request boundary"
    )


def canonical_endpoint_identity(value: Any, *, provider: str) -> str:
    """Return a non-secret endpoint/data-boundary identity."""

    raw = str(value or "").strip()
    provider_id = str(provider or "").strip().lower()
    if not raw:
        raise FinalProviderBindingUnsupported(
            f"provider endpoint identity is unavailable for {provider_id or 'unknown'}"
        )
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme in {"cloudcode-pa", "acp", "acp+tcp"}:
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise FinalProviderBindingUnsupported(
                "provider endpoint identity contains credentials, query, or fragment"
            )
        if not parsed.netloc:
            raise FinalProviderBindingUnsupported(
                "provider endpoint authority is missing"
            )
        return urlunsplit((scheme, parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))
    if scheme not in {"http", "https"}:
        raise FinalProviderBindingUnsupported(
            f"provider endpoint scheme is not bindable: {parsed.scheme or 'missing'}"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise FinalProviderBindingUnsupported(
            "provider endpoint identity contains credentials, query, or fragment"
        )
    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise FinalProviderBindingUnsupported("provider endpoint hostname is missing")
    port = parsed.port
    default_port = 443 if scheme == "https" else 80
    authority_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = (
        authority_host
        if port in {None, default_port}
        else f"{authority_host}:{port}"
    )
    path = (parsed.path or "").rstrip("/")
    return urlunsplit((scheme, netloc, path, "", ""))


def provider_endpoint_identity(
    client: Any,
    *,
    provider: str,
    configured_base_url: Any = None,
) -> str:
    """Measure the endpoint used by the concrete SDK leaf without secrets."""

    # A concrete endpoint reported by the final SDK client is authoritative.
    # Never hide a malformed or conflicting live endpoint by falling back to
    # the configured value: the route-chain comparison must see the actual
    # data boundary and fail closed on disagreement.
    client_base_url = getattr(client, "base_url", None)
    if (
        client_base_url is not None
        and type(client_base_url).__module__ != "unittest.mock"
        and str(client_base_url).strip()
    ):
        return canonical_endpoint_identity(client_base_url, provider=provider)
    meta = getattr(client, "meta", None)
    endpoint_url = getattr(meta, "endpoint_url", None)
    if isinstance(endpoint_url, str) and endpoint_url.strip():
        return canonical_endpoint_identity(endpoint_url, provider=provider)
    return canonical_endpoint_identity(configured_base_url, provider=provider)


def endpoint_identity_for_dispatch(
    client: Any,
    *,
    provider: str,
    configured_base_url: Any = None,
    require_exact: bool,
) -> str:
    """Require endpoint truth only for a receipt-bearing dispatch."""

    try:
        return provider_endpoint_identity(
            client,
            provider=provider,
            configured_base_url=configured_base_url,
        )
    except FinalProviderBindingUnsupported:
        if require_exact:
            raise
        return "unverified:ordinary-provider-call"


def _bedrock_endpoint_identity(region: Any) -> str:
    region_id = str(region or "").strip().lower()
    if not region_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789-"
        for character in region_id
    ):
        raise FinalProviderBindingUnsupported("configured Bedrock region is invalid")
    raise FinalProviderBindingUnsupported(
        "configured Bedrock region does not identify the final SDK endpoint; "
        "retrieval evidence requires an explicit atomic provider boundary"
    )


def _configured_route_endpoint(
    provider: str,
    explicit_base_url: Any,
    *,
    bedrock_region: Any = None,
) -> str:
    if isinstance(explicit_base_url, str) and explicit_base_url.strip():
        return canonical_endpoint_identity(explicit_base_url, provider=provider)
    if provider.strip().lower() == "bedrock":
        return _bedrock_endpoint_identity(bedrock_region or "us-east-1")
    from hermes_cli.providers import get_provider

    definition = get_provider(provider)
    if definition is None:
        raise FinalProviderBindingUnsupported(
            f"configured fallback endpoint is unresolved for {provider}"
        )
    configured = ""
    if definition.base_url_env_var:
        configured = os.getenv(definition.base_url_env_var, "").strip()
    configured = configured or definition.base_url
    return canonical_endpoint_identity(configured, provider=provider)


def _normalized_route_model(model: Any, provider: str) -> str:
    value = str(model or "").strip()
    if not value:
        raise FinalProviderBindingUnsupported("configured provider route model is missing")
    try:
        from hermes_cli.model_normalize import normalize_model_for_provider

        return str(normalize_model_for_provider(value, provider) or value)
    except Exception:
        return value


def snapshot_allowed_provider_routes(agent: Any) -> tuple[dict[str, Any], ...]:
    """Freeze the turn-start primary and configured fallback data boundaries."""

    from hermes_cli.providers import determine_api_mode, normalize_provider

    primary_provider = normalize_provider(str(getattr(agent, "provider", "") or ""))
    primary_model = _normalized_route_model(getattr(agent, "model", ""), primary_provider)
    primary_base_url = getattr(agent, "base_url", "")
    primary_endpoint = (
        _bedrock_endpoint_identity(
            getattr(agent, "_bedrock_region", None) or "us-east-1"
        )
        if primary_provider == "bedrock" and not str(primary_base_url or "").strip()
        else canonical_endpoint_identity(primary_base_url, provider=primary_provider)
    )
    routes: list[dict[str, Any]] = [{
        "fallbackIndex": 0,
        "provider": primary_provider,
        "model": primary_model,
        "apiMode": str(getattr(agent, "api_mode", "") or "chat_completions"),
        "endpointIdentity": primary_endpoint,
    }]
    for index, raw_entry in enumerate(
        list(getattr(agent, "_fallback_chain", []) or []),
        start=1,
    ):
        if not isinstance(raw_entry, Mapping):
            raise FinalProviderBindingUnsupported(
                f"configured fallback route {index} is invalid"
            )
        provider = normalize_provider(str(raw_entry.get("provider") or ""))
        if not provider:
            raise FinalProviderBindingUnsupported(
                f"configured fallback route {index} provider is missing"
            )
        model = _normalized_route_model(raw_entry.get("model"), provider)
        endpoint = _configured_route_endpoint(
            provider,
            raw_entry.get("base_url"),
            bedrock_region=(
                raw_entry.get("region")
                or getattr(agent, "_bedrock_region", None)
                or "us-east-1"
            ),
        )
        api_mode = str(raw_entry.get("api_mode") or "").strip()
        if not api_mode:
            api_mode = determine_api_mode(provider, endpoint)
            if (
                api_mode == "chat_completions"
                and callable(getattr(agent, "_provider_model_requires_responses_api", None))
                and agent._provider_model_requires_responses_api(
                    model,
                    provider=provider,
                )
            ):
                api_mode = "codex_responses"
        routes.append({
            "fallbackIndex": index,
            "provider": provider,
            "model": model,
            "apiMode": api_mode,
            "endpointIdentity": endpoint,
        })
    return tuple(routes)


class RequestDispatchHandoff:
    """Choose one terminal owner for a provider-attempt dispatch boundary.

    Request/client construction remains outside the handoff.  At the final
    provider call site, ``commit_and_claim_dispatch`` serializes the durable
    receipt callback with pre-dispatch cancellation.  The winner is either:

    * ``abandoned``: no receipt callback and no SDK-entry intent, or
    * ``dispatch_owned``: the receipt callback succeeded and the SDK entry
      intent has been irrevocably committed before a losing canceller may return.

    This is deliberately an SDK-entry-intent contract.  Physical network
    send acknowledgement requires provider transport support and is not
    inferred here.
    """

    def __init__(
        self,
        callback: Callable[[], Any] | None,
        *,
        interrupted: Callable[[], bool],
        interrupted_message: str,
        max_attempts: int | None = None,
        callback_accepts_attempt_binding: bool = False,
        outcome_callback: Callable[[str, str, str | None], Any] | None = None,
        configured_provider: str | None = None,
        configured_model: str | None = None,
        allowed_provider_routes: tuple[Mapping[str, Any], ...] | None = None,
    ) -> None:
        if max_attempts is not None and (
            isinstance(max_attempts, bool) or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer or None")
        self._callback = callback
        self._interrupted = interrupted
        self._interrupted_message = interrupted_message
        self._max_attempts = max_attempts
        self._callback_accepts_attempt_binding = callback_accepts_attempt_binding
        self._outcome_callback = outcome_callback
        self._configured_provider = configured_provider
        self._configured_model = configured_model
        self._allowed_provider_routes = tuple(
            dict(route) for route in (allowed_provider_routes or ())
        )
        self._configured_route_chain_digest = (
            "sha256:"
            + hashlib.sha256(
                _canonical_bytes(list(self._allowed_provider_routes))
            ).hexdigest()
            if self._allowed_provider_routes
            else None
        )
        self._lock = threading.Lock()
        self._sdk_entry_intent_committed = threading.Event()
        self._active_attempt_entry: threading.Event | None = None
        self._future_attempts_closed = False
        self._closure_cause: str | None = None
        self._attempt_claim_count = 0
        self._provider_call_id: str | None = None
        self._seen_provider_call_ids: set[str] = set()
        self._active_attempt_binding_digest: str | None = None
        self._terminal_outcome_status: str | None = None
        self._outcome_persistence_error: BaseException | None = None
        self._state = "pending"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def sdk_entry_intent_committed(self) -> bool:
        return self._sdk_entry_intent_committed.is_set()

    @property
    def future_attempts_closed(self) -> bool:
        with self._lock:
            return self._future_attempts_closed

    @property
    def attempt_claim_count(self) -> int:
        with self._lock:
            return self._attempt_claim_count

    @property
    def provider_call_id(self) -> str | None:
        with self._lock:
            return self._provider_call_id

    def bind_provider_call_identity(self, call_id: str) -> None:
        """Bind the fresh provider-ledger call UUID before leaf dispatch.

        Pre-commit route construction may fail and select an approved fallback,
        so a pending handoff may replace its candidate call id.  An id already
        presented to this handoff is never reusable, and dispatch ownership
        freezes the final id under the same lock as receipt commitment.
        """

        try:
            canonical = str(uuid.UUID(str(call_id)))
        except (AttributeError, ValueError) as exc:
            raise ValueError("provider call identity must be a canonical UUID") from exc
        if canonical != str(call_id):
            raise ValueError("provider call identity must be a canonical UUID")
        with self._lock:
            if self._state != "pending" or self._attempt_claim_count:
                raise RuntimeError("provider call identity is already dispatch-owned")
            if canonical in self._seen_provider_call_ids:
                raise RuntimeError("provider call identity cannot be reused")
            self._seen_provider_call_ids.add(canonical)
            self._provider_call_id = canonical

    @property
    def terminal_outcome_status(self) -> str | None:
        with self._lock:
            return self._terminal_outcome_status

    @property
    def outcome_persistence_error(self) -> BaseException | None:
        with self._lock:
            return self._outcome_persistence_error

    @property
    def requires_exact_provider_attempt_binding(self) -> bool:
        return self._callback_accepts_attempt_binding

    def _record_terminal_outcome_locked(
        self,
        status: str,
        error_category: str | None,
    ) -> None:
        if self._terminal_outcome_status is not None:
            return
        # Claim the one terminal owner before calling an external/fsyncing
        # sink.  A sink failure must not reopen the winner race or replace
        # the SDK/timeout/interrupt cause that led here.
        self._terminal_outcome_status = status
        if (
            self._outcome_callback is None
            or self._active_attempt_binding_digest is None
        ):
            return
        try:
            self._outcome_callback(
                status,
                self._active_attempt_binding_digest,
                error_category,
            )
        except BaseException as exc:
            self._outcome_persistence_error = exc

    def record_terminal_outcome(
        self,
        status: str,
        error_category: str | None = None,
    ) -> None:
        """Append the first terminal outcome for a committed attempt."""

        with self._lock:
            if self._state != "dispatch_owned":
                raise RuntimeError("provider attempt is not dispatch-owned")
            self._record_terminal_outcome_locked(status, error_category)

    @property
    def closure_cause(self) -> str | None:
        with self._lock:
            return self._closure_cause

    def abandon(self, *, cause: str = "cancelled") -> bool:
        """Return True only when pre-dispatch cancellation wins.

        If dispatch commitment is in progress, acquiring ``_lock`` waits for
        its receipt write to reach ``dispatch_owned`` or ``failed``.  A losing
        canceller then also waits for the explicit SDK-entry-intent event, so
        it cannot report cancellation complete before the worker crosses that
        boundary.
        """

        active_entry = None
        with self._lock:
            if self._state == "pending":
                self._state = "abandoned"
                self._future_attempts_closed = True
                self._closure_cause = cause
                return True
            state = self._state
            if state == "dispatch_owned":
                self._future_attempts_closed = True
                self._closure_cause = cause
                if cause == "user_interrupt":
                    self._record_terminal_outcome_locked("interrupted", None)
                elif cause == "watchdog_timeout":
                    self._record_terminal_outcome_locked(
                        "unknown",
                        "WatchdogTimeout",
                    )
                else:
                    self._record_terminal_outcome_locked("unknown", None)
                active_entry = self._active_attempt_entry
        if active_entry is not None:
            active_entry.wait()
        return False

    def commit_and_claim_dispatch(
        self,
        invoke: Callable[..., Any],
        *,
        provider: str | None = None,
        api_mode: str | None = None,
        model: str | None = None,
        sdk_method: str | None = None,
        leaf_adapter: str | None = None,
        endpoint_identity: str | None = None,
        fallback_index: int = 0,
        request_kwargs: Mapping[str, Any] | None = None,
        outcome_on_return: str | None = "response_observed",
    ) -> Any:
        """Commit receipt truth and claim one SDK-entry-intent boundary."""

        if not callable(invoke):
            raise TypeError("provider dispatch invoke must be callable")
        entry_intent = threading.Event()
        bound_kwargs: dict[str, Any] | None = None
        request_projection: Any = None
        if request_kwargs is not None:
            bound_kwargs = _snapshot_request_value(dict(request_kwargs))
            if self._callback_accepts_attempt_binding:
                if not all(
                    isinstance(item, str) and item
                    for item in (
                        provider,
                        api_mode,
                        model,
                        sdk_method,
                        leaf_adapter,
                        endpoint_identity,
                    )
                ):
                    raise ValueError(
                        "provider, api_mode, model, sdk_method, leaf_adapter, and "
                        "endpoint_identity are required for a bound provider attempt"
                    )
                if isinstance(fallback_index, bool) or fallback_index < 0:
                    raise ValueError("fallback_index must be a nonnegative integer")
                if not all(
                    isinstance(item, str) and item
                    for item in (self._configured_provider, self._configured_model)
                ):
                    raise ValueError(
                        "configured provider and model are required for a bound receipt"
                    )
                if not self._allowed_provider_routes:
                    raise ValueError(
                        "bound receipt callback requires an immutable provider route chain"
                    )
                request_projection = _canonical_request_projection(bound_kwargs)
        elif self._callback_accepts_attempt_binding:
            raise ValueError("bound receipt callback requires final provider kwargs")
        with self._lock:
            if self._state == "failed":
                raise RuntimeError("request dispatch handoff previously failed")
            if (
                self._max_attempts is not None
                and self._attempt_claim_count >= self._max_attempts
            ):
                self._future_attempts_closed = True
                self._closure_cause = "attempt_limit"
                raise ProviderAttemptLedgerRequired(
                    "provider retry requires a distinct durable attempt ledger"
                )
            if self._future_attempts_closed or self._state == "abandoned":
                raise DispatchAttemptsClosed(self._closure_cause or "closed")
            if self._interrupted():
                self._future_attempts_closed = True
                self._closure_cause = "user_interrupt"
                if self._state == "pending":
                    self._state = "abandoned"
                raise DispatchAttemptsClosed("user_interrupt")
            attempt_id = self._attempt_claim_count + 1
            attempt_binding: dict[str, Any] | None = None
            if request_projection is not None:
                # Revalidate the private SDK kwargs snapshot before any
                # durable callback.  The callback receives only a separate
                # canonical projection, never the object that will be passed
                # to the SDK, so callback code cannot mutate the bound call.
                current_projection = _canonical_request_projection(bound_kwargs)
                if current_projection != request_projection:
                    self._future_attempts_closed = True
                    self._closure_cause = "request_mutated_before_commit"
                    raise RuntimeError(
                        "final provider request mutated before attempt commitment"
                    )
                if self._callback_accepts_attempt_binding:
                    if self._provider_call_id is None:
                        raise ProviderAttemptLedgerRequired(
                            "bound receipt requires a fresh provider ledger call identity"
                        )
                    if fallback_index >= len(self._allowed_provider_routes):
                        raise FinalProviderBindingUnsupported(
                            "final provider route is outside the configured fallback chain"
                        )
                    allowed_route = self._allowed_provider_routes[fallback_index]
                    expected_route = {
                        "fallbackIndex": fallback_index,
                        "provider": provider,
                        "model": model,
                        "apiMode": api_mode,
                        "endpointIdentity": endpoint_identity,
                    }
                    if allowed_route != expected_route:
                        raise FinalProviderBindingUnsupported(
                            "final provider route does not match the configured fallback chain"
                        )
                request_projection_digest = "sha256:" + hashlib.sha256(
                    _canonical_bytes(request_projection)
                ).hexdigest()
                attempt_binding = {
                    "schema": "jitech-provider-sdk-request-attempt-binding/v1",
                    "providerAttemptId": attempt_id,
                    "providerCallId": self._provider_call_id,
                    "configuredProvider": self._configured_provider or provider,
                    "configuredModel": self._configured_model or model,
                    "provider": provider,
                    "apiMode": api_mode,
                    "model": model,
                    "sdkMethod": sdk_method,
                    "leafAdapter": leaf_adapter,
                    "endpointIdentity": endpoint_identity,
                    "fallbackIndex": fallback_index,
                    "configuredRouteChainDigest": self._configured_route_chain_digest,
                    "finalRequestKwargsDigest": request_projection_digest,
                }
                attempt_binding["providerAttemptBindingDigest"] = (
                    "sha256:"
                    + hashlib.sha256(_canonical_bytes(attempt_binding)).hexdigest()
                )
            if self._state == "pending":
                if self._callback is not None and not callable(self._callback):
                    self._state = "failed"
                    self._future_attempts_closed = True
                    raise TypeError("on_request_dispatch must be callable")
                self._state = "committing"
                try:
                    if self._callback is not None:
                        if self._callback_accepts_attempt_binding:
                            self._callback(
                                attempt_binding,
                                _snapshot_request_value(request_projection),
                            )
                        else:
                            self._callback()
                except BaseException:
                    self._state = "failed"
                    self._future_attempts_closed = True
                    raise
                self._state = "dispatch_owned"
            self._attempt_claim_count += 1
            if (
                self._max_attempts is not None
                and self._attempt_claim_count >= self._max_attempts
            ):
                self._future_attempts_closed = True
            self._active_attempt_entry = entry_intent
            if attempt_binding is not None:
                self._active_attempt_binding_digest = str(
                    attempt_binding["providerAttemptBindingDigest"]
                )

        # This is the documented SDK-entry-intent linearization point.  A
        # losing canceller waits for this per-attempt event before returning.
        self._sdk_entry_intent_committed.set()
        entry_intent.set()
        try:
            if bound_kwargs is None:
                response = invoke()
            else:
                response = invoke(bound_kwargs)
        except BaseException as exc:
            with self._lock:
                self._record_terminal_outcome_locked(
                    "sdk_exception",
                    type(exc).__name__,
                )
            raise
        else:
            if outcome_on_return is not None:
                with self._lock:
                    self._record_terminal_outcome_locked(
                        outcome_on_return,
                        None,
                    )
            return response
        finally:
            with self._lock:
                if self._active_attempt_entry is entry_intent:
                    self._active_attempt_entry = None


def coerce_request_dispatch_handoff(
    value: RequestDispatchHandoff | Callable[[], Any] | None,
    *,
    interrupted: Callable[[], bool],
    interrupted_message: str,
) -> RequestDispatchHandoff:
    """Reuse an upstream attempt gate instead of creating nested gates."""

    if isinstance(value, RequestDispatchHandoff):
        return value
    return RequestDispatchHandoff(
        value,
        interrupted=interrupted,
        interrupted_message=interrupted_message,
    )
