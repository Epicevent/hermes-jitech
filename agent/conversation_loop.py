"""The agent conversation loop ??extracted from ``run_agent.AIAgent``.

This is the biggest single chunk pulled out of ``run_agent.py``: the
roughly 3,900-line :func:`run_conversation` body that drives one user
turn through the agent (model call, tool dispatch, retries, fallbacks,
compression, post-turn hooks, background memory/skill review nudges).

The function takes the parent ``AIAgent`` instance as its first
argument (``agent``) and accesses its state via attribute lookup.
``_ra().AIAgent.run_conversation`` is now a thin forwarder.

Symbols that production code or tests patch on ``run_agent`` directly
(``handle_function_call``, ``_set_interrupt``, ``OpenAI``, ...) are
resolved through :func:`_ra` so those patches keep working.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import ssl
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from agent.anthropic_adapter import _is_oauth_token
from agent.auxiliary_client import set_runtime_main
from agent.codex_responses_adapter import _summarize_user_message_for_log
from agent.display import KawaiiSpinner
from agent.error_classifier import FailoverReason, classify_api_error
from agent.iteration_budget import IterationBudget
from agent.memory_manager import build_memory_context_block
from agent.message_sanitization import (
    _repair_tool_call_arguments,
    _sanitize_messages_non_ascii,
    _sanitize_messages_surrogates,
    _sanitize_structure_non_ascii,
    _sanitize_structure_surrogates,
    _sanitize_surrogates,
    _sanitize_tools_non_ascii,
    _strip_images_from_messages,
    _strip_non_ascii,
)
from agent.model_metadata import (
    MINIMUM_CONTEXT_LENGTH,
    estimate_messages_tokens_rough,
    estimate_request_tokens_rough,
    get_context_length_from_provider_error,
    parse_available_output_tokens_from_error,
    save_context_length,
)
from agent.nous_rate_guard import (
    clear_nous_rate_limit,
    is_genuine_nous_rate_limit,
    nous_rate_limit_remaining,
    record_nous_rate_limit,
)
from agent.process_bootstrap import _install_safe_stdio
from agent.prompt_caching import apply_anthropic_cache_control
from agent.retry_utils import jittered_backoff
from agent.request_dispatch import (
    RequestDispatchHandoff,
    snapshot_allowed_provider_routes,
)
from agent.trajectory import has_incomplete_scratchpad
from agent.usage_pricing import estimate_usage_cost, normalize_usage
from hermes_constants import display_hermes_home as _dhh_fn, PARTIAL_STREAM_STUB_ID
from hermes_logging import set_session_context
from tools.schema_sanitizer import strip_pattern_and_format
from tools.skill_provenance import set_current_write_origin
from utils import base_url_host_matches, env_var_enabled

logger = logging.getLogger(__name__)


_PROVIDER_USAGE_KEYS = (
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


def _provider_usage_trigger(agent: Any) -> str:
    """Map the observed run origin to the shared content-free trigger enum."""
    source = (
        getattr(agent, "platform", None)
        or os.environ.get("HERMES_SESSION_SOURCE")
        or "unknown"
    )
    normalized = str(source).strip().lower()
    if normalized in {"cron", "heartbeat", "manual", "memory", "overflow"}:
        return normalized
    if not normalized or normalized == "unknown":
        return "unknown"
    return "user"


def _plain_usage_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_plain_usage_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_usage_value(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return _plain_usage_value(dumped)
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
        except Exception:
            dumped = None
        if isinstance(dumped, dict):
            return _plain_usage_value(dumped)
    try:
        attributes = vars(value)
    except TypeError:
        return None
    return {
        str(key): _plain_usage_value(item)
        for key, item in attributes.items()
        if not str(key).startswith("_")
    }


def _provider_raw_usage(response_usage: Any) -> Optional[dict[str, Any]]:
    """Copy only named accounting buckets from an SDK usage object."""
    plain = _plain_usage_value(response_usage)
    if isinstance(plain, dict):
        return {key: plain[key] for key in _PROVIDER_USAGE_KEYS if key in plain} or None
    values: dict[str, Any] = {}
    for key in _PROVIDER_USAGE_KEYS:
        item = getattr(response_usage, key, None)
        if item is not None:
            values[key] = _plain_usage_value(item)
    return values or None


def _attach_usage_ledger_reference(
    receipt: dict[str, Any],
    result: dict[str, Any],
) -> None:
    exported = result.get("receipt")
    if not isinstance(exported, dict):
        raise ValueError("provider usage ledger insert returned no receipt")
    receipt["usageLedger"] = {
        "schema": exported.get("schema"),
        "status": "persisted",
        "ledgerSeq": exported.get("ledgerSeq"),
        "receiptDigest": exported.get("receiptDigest"),
        "receiptCoverage": exported.get("receiptCoverage"),
        "missingReceiptFields": list(exported.get("missingReceiptFields") or []),
        "usageCoverage": exported.get("usageCoverage"),
        "missingUsageFields": list(exported.get("missingUsageFields") or []),
    }


def _attach_unavailable_usage_ledger(
    receipt: dict[str, Any],
    exc: BaseException,
) -> None:
    receipt["usageLedger"] = {
        "schema": "jitech-provider-usage-call/v1",
        "status": "unavailable",
        "ledgerSeq": None,
        "receiptDigest": None,
        "receiptCoverage": "unavailable",
        "missingReceiptFields": ["immutableLedgerPersistence"],
        "usageCoverage": "unavailable",
        "missingUsageFields": ["immutableLedgerPersistence"],
        "failureClass": type(exc).__name__,
    }


def _record_provider_attempt(
    agent: Any,
    response: Any,
    *,
    configured_model: str,
    configured_provider: str,
    requested_model: str,
    requested_provider: str,
    run_id: str,
    turn_id: str,
    request_id: str,
    call_id: str,
    api_call_index: int,
    attempt: int,
    fallback_index: int,
    retry_of: Optional[str],
    fallback_parent: Optional[str],
    started_at: float,
    completed_at: float,
    status: str,
    finish_reason: Optional[str] = None,
    error_category: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Persist one actual provider attempt before mutable usage aggregation."""
    if not call_id:
        return None

    provider_receipt = (
        getattr(agent, "last_provider_receipt", None)
        if response is not None
        else None
    )
    provider_receipt = (
        dict(provider_receipt) if isinstance(provider_receipt, dict) else None
    )
    actual_provider = None
    actual_model = None
    response_id = None
    evidence_source = None
    raw_usage = None

    if provider_receipt is not None:
        model_version = provider_receipt.get("modelVersion")
        source = provider_receipt.get("evidenceSource")
        if (
            isinstance(model_version, str)
            and model_version
            and source == "gemini_response.modelVersion"
        ):
            actual_provider = requested_provider
            actual_model = model_version
            evidence_source = source
        candidate_response_id = provider_receipt.get("responseId")
        if isinstance(candidate_response_id, str) and candidate_response_id:
            response_id = candidate_response_id
        provider_usage = provider_receipt.get("usageMetadata")
        if isinstance(provider_usage, dict) and provider_usage:
            raw_usage = provider_usage
        candidate_finish = provider_receipt.get("finishReason")
        if isinstance(candidate_finish, str) and candidate_finish:
            finish_reason = candidate_finish

    if actual_model is None and requested_provider not in {"gemini", "google"}:
        response_model = getattr(response, "model", None)
        if isinstance(response_model, str) and response_model:
            actual_provider = requested_provider
            actual_model = response_model
            evidence_source = "response.model"
    if response_id is None:
        candidate_response_id = getattr(response, "id", None)
        if isinstance(candidate_response_id, str) and candidate_response_id:
            response_id = candidate_response_id
    if raw_usage is None:
        provider_usage = getattr(response, "provider_usage", None)
        raw_usage = _provider_raw_usage(
            provider_usage
            if provider_usage is not None
            else getattr(response, "usage", None)
        )

    receipt_projection = provider_receipt or {"provider": requested_provider}
    receipt_projection.update({
        "configuredModel": configured_model,
        "requestedModel": requested_model,
        "actualModel": actual_model,
        "runId": run_id,
        "turnId": turn_id,
        "requestId": request_id,
        "callId": call_id,
        "apiCallIndex": api_call_index,
        "attempt": attempt,
        "fallbackIndex": fallback_index,
        "retryOf": retry_of,
        "fallbackParent": fallback_parent,
        "status": status,
    })

    try:
        session_db = getattr(agent, "_session_db", None)
        if session_db is None:
            from hermes_cli.config import get_hermes_home
            from hermes_state import SessionDB

            session_db = SessionDB(get_hermes_home() / "state.db")
            agent._session_db = session_db
        session_id = getattr(agent, "session_id", None)
        if session_id and not getattr(agent, "_session_db_created", False):
            agent._ensure_db_session()
        result = session_db.record_provider_call(
            session_id,
            call_id=call_id,
            request_id=request_id,
            api_call_index=api_call_index,
            attempt=attempt,
            fallback_index=fallback_index,
            configured_provider=configured_provider,
            configured_model=configured_model,
            requested_provider=requested_provider,
            requested_model=requested_model,
            actual_provider=actual_provider,
            actual_model=actual_model,
            response_id=response_id,
            evidence_source=evidence_source,
            finish_reason=finish_reason,
            usage=raw_usage,
            run_id=run_id,
            turn_id=turn_id,
            retry_of=retry_of,
            fallback_parent=fallback_parent,
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            trigger=_provider_usage_trigger(agent),
            error_category=error_category,
        )
        _attach_usage_ledger_reference(receipt_projection, result)
        agent.last_provider_receipt = receipt_projection
        return result
    except Exception as exc:
        _attach_unavailable_usage_ledger(receipt_projection, exc)
        agent.last_provider_receipt = receipt_projection
        logger.error(
            "Provider usage ledger persistence failed (session=%s, call=%s): %s",
            getattr(agent, "session_id", None),
            call_id,
            exc,
        )
        return None


def _ollama_context_limit_error(agent: Any, request_tokens: int) -> Optional[str]:
    """Return a user-facing error when Ollama is loaded with too little context."""
    if not getattr(agent, "tools", None):
        return None

    runtime_ctx = getattr(agent, "_ollama_num_ctx", None)
    if not isinstance(runtime_ctx, int) or runtime_ctx <= 0:
        return None
    if runtime_ctx >= MINIMUM_CONTEXT_LENGTH:
        return None

    model = getattr(agent, "model", "") or "the selected model"
    base_url = getattr(agent, "base_url", "") or "unknown base URL"
    provider = getattr(agent, "provider", "") or "unknown"
    tool_count = len(getattr(agent, "tools", None) or [])

    logger.warning(
        "Ollama runtime context too small for Hermes tool use: "
        "model=%s provider=%s base_url=%s runtime_context=%d "
        "minimum_context=%d estimated_request_tokens=%d tool_count=%d "
        "session=%s",
        model,
        provider,
        base_url,
        runtime_ctx,
        MINIMUM_CONTEXT_LENGTH,
        request_tokens,
        tool_count,
        getattr(agent, "session_id", None) or "none",
    )

    return (
        f"Ollama loaded `{model}` with only {runtime_ctx:,} tokens of runtime "
        f"context, but Hermes needs at least {MINIMUM_CONTEXT_LENGTH:,} tokens "
        "for reliable tool use.\n\n"
        "Increase the Ollama context for this model and restart/reload the "
        "model before trying again. A known-good starting point is 65,536 "
        "tokens. In Hermes config, set `model.ollama_num_ctx: 65536` "
        "(and `model.context_length: 65536` if you also override the displayed "
        "model context). If you manage the model through an Ollama Modelfile, "
        "set `PARAMETER num_ctx 65536` there instead."
    )


def _ra():
    """Lazy reference to ``run_agent`` so callers can patch
    ``run_agent.handle_function_call`` / ``run_agent._set_interrupt`` /
    ``run_agent.OpenAI`` and have those patches reach this code path.
    """
    import run_agent
    return run_agent


def _nous_entitlement_message(capability: str) -> str:
    try:
        from hermes_cli.nous_account import (
            format_nous_portal_entitlement_message,
            get_nous_portal_account_info,
        )

        account_info = get_nous_portal_account_info(force_fresh=True)
        message = format_nous_portal_entitlement_message(
            account_info,
            capability=capability,
        )
        return message or ""
    except Exception:
        return ""


def _print_nous_entitlement_guidance(agent, capability: str) -> bool:
    message = _nous_entitlement_message(capability)
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   ?뮕 {line}", force=True)
    return True


def _is_nous_inference_route(provider: str, base_url: str) -> bool:
    provider = (provider or "").strip().lower()
    if provider == "nous":
        return True
    base = str(base_url or "")
    return (
        base_url_host_matches(base, "inference-api.nousresearch.com")
        or base_url_host_matches(base, "inference.nousresearch.com")
    )


def _billing_or_entitlement_message(
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
) -> str:
    if _is_nous_inference_route(provider, base_url):
        return _nous_entitlement_message(capability)

    provider_label = (provider or "").strip() or "the selected provider"
    model_label = (model or "").strip() or "the selected model"
    lines = [
        (
            f"{provider_label} reported that billing, credits, or account "
            f"entitlement is exhausted for {model_label}."
        ),
        "Add credits or update billing with that provider, then retry.",
    ]
    if base_url_host_matches(str(base_url or ""), "openrouter.ai"):
        lines.append("OpenRouter credits: https://openrouter.ai/settings/credits")
    lines.append("You can switch providers temporarily with /model <model> --provider <provider>.")
    return "\n".join(lines)


def _print_billing_or_entitlement_guidance(
    agent,
    *,
    capability: str,
    provider: str,
    base_url: str,
    model: str,
) -> bool:
    message = _billing_or_entitlement_message(
        capability=capability,
        provider=provider,
        base_url=base_url,
        model=model,
    )
    if not message:
        return False
    for line in message.splitlines():
        agent._vprint(f"{agent.log_prefix}   ?뮕 {line}", force=True)
    return True


def _try_refresh_nous_paid_entitlement_credentials(agent) -> bool:
    """Refresh Nous runtime credentials after a fresh paid-entitlement check."""
    try:
        from hermes_cli.auth import NOUS_INFERENCE_AUTH_MODE_LEGACY
        from hermes_cli.nous_account import get_nous_portal_account_info

        account_info = get_nous_portal_account_info(force_fresh=True)
        if account_info.paid_service_access is not True:
            return False
        return agent._try_refresh_nous_client_credentials(
            force=False,
            inference_auth_mode=NOUS_INFERENCE_AUTH_MODE_LEGACY,
        )
    except Exception:
        return False


def _restore_or_build_system_prompt(agent, system_message, conversation_history):
    """Restore the cached system prompt from the session DB or build it fresh.

    Mutates ``agent._cached_system_prompt`` and persists a freshly-built
    prompt back to the session DB on first build.  E…77514 tokens truncated…Control.ObjectSecurity NewSecurityDescriptorOfType(System.String, System.Security.AccessControl.AccessControlSections) System.Security.AccessControl.ObjectSecurity NewSecurityDescriptor(ItemType) System.Management.Automation.ErrorRecord CreateErrorRecord(System.String, System.String) System.String GetHelpMaml(System.String, System.String) System.Management.Automation.ProviderInfo Start(System.Management.Automation.ProviderInfo) System.Management.Automation.PSDriveInfo NewDrive(System.Management.Automation.PSDriveInfo) Void MapNetworkDrive(System.Management.Automation.PSDriveInfo) System.Management.Automation.PSDriveInfo RemoveDrive(System.Management.Automation.PSDriveInfo) System.String GetUNCForNetworkDrive(System.String) System.String GetSubstitutedPathForNetworkDosDevice(System.String) Boolean IsValidPath(System.String) Void GetItem(System.String) System.IO.FileSystemInfo GetFileSystemItem(System.String, Boolean ByRef, Boolean) Boolean ConvertPath(System.String, System.String, System.String ByRef, System.String ByRef) Void GetPathItems(System.String, Boolean, UInt32, Boolean, System.Management.Automation.ReturnContainers) Void Dir(System.IO.DirectoryInfo, Boolean, UInt32, Boolean, System.Management.Automation.ReturnContainers, InodeTracker) System.Management.Automation.FlagsExpression`1[System.IO.FileAttributes] FormatAttributeSwitchParamters() System.String Mode(System.Management.Automation.PSObject) Void RenameItem(System.String, System.String) Void NewItem(System.String, System.String, System.Object) ItemType GetItemType(System.String) Void CreateDirectory(System.String, Boolean) Boolean CreateIntermediateDirectories(System.String) Void RemoveItem(System.String, Boolean) Void RemoveDirectoryInfoItem(System.IO.DirectoryInfo, Boolean, Boolean, Boolean) Void RemoveFileSystemItem(System.IO.FileSystemInfo, Boolean) Boolean ItemExists(System.String, System.Management.Automation.ErrorRecord ByRef) Boolean DirectoryInfoHasChildItems(System.IO.DirectoryInfo) Void CopyItem(System.String, System.String, Boolean) Void CopyItemFromRemoteSession(System.String, System.String, Boolean, Boolean, System.Management.Automation.Runspaces.PSSession) Void CopyDirectoryInfoItem(System.IO.DirectoryInfo, System.String, Boolean, Boolean, System.Management.Automation.PowerShell) Void CopyFileInfoItem(System.IO.FileInfo, System.String, Boolean, System.Management.Automation.PowerShell) Void CopyDirectoryFromRemoteSession(System.String, System.String, System.String, Boolean, Boolean, System.Management.Automation.PowerShell) System.Collections.ArrayList GetRemoteSourceAlternateStreams(System.Management.Automation.PowerShell, System.String) Void RemoveFunctionsPSCopyFileFromRemoteSession(System.Management.Automation.PowerShell) System.Collections.Hashtable GetRemoteFileMetadata(System.String, System.Management.Automation.PowerShell) Void SetFileMetadata(System.String, System.IO.FileInfo, System.Management.Automation.PowerShell) Void CopyFileFromRemoteSession(System.String, System.String, System.String, Boolean, System.Management.Automation.PowerShell, Int64) Boolean PerformCopyFileFromRemoteSession(System.String, System.IO.FileInfo, System.String, Boolean, System.Management.Automation.PowerShell, Int64, Boolean, System.String) Void RemoveFunctionPSCopyFileToRemoteSession(System.Management.Automation.PowerShell) Boolean RemoteTargetSupportsAlternateStreams(System.Management.Automation.PowerShell, System.String) System.String MakeRemotePath(System.Management.Automation.PowerShell, System.String, System.String) Boolean RemoteDirectoryExist(System.Management.Automation.PowerShell, System.String) Boolean CopyFileStreamToRemoteSession(System.IO.FileInfo, System.String, System.Management.Automation.PowerShell, Boolean, System.String) System.Collections.Hashtable GetFileMetadata(System.IO.FileInfo) Void SetRemoteFileMetadata(System.IO.FileInfo, System.String, System.Management.Automation.PowerShell) Boolean PerformCopyFileToRemoteSession(System.IO.FileInfo, System.String, System.Management.Automation.PowerShell) Boolean RemoteDestinationPathIsFile(System.String, System.Management.Automation.PowerShell) System.String CreateDirectoryOnRemoteSession(System.String, Boolean, System.Management.Automation.PowerShell) System.String GetParentPath(System.String, System.String) Boolean IsUNCPath(System.String) Boolean IsUNCRoot(System.String) Boolean IsPathRoot(System.String) System.String NormalizeRelativePath(System.String, System.String) System.String NormalizeRelativePathHelper(System.String, System.String) System.String RemoveRelativeTokens(System.String) System.Collections.Generic.Stack`1[System.String] TokenizePathToStack(System.String, System.String) System.Collections.Generic.Stack`1[System.String] NormalizeThePath(System.String, System.Collections.Generic.Stack`1[System.String]) System.String GetChildName(System.String) System.String EnsureDriveIsRooted(System.String) Void MoveItem(System.String, System.String) Void MoveFileInfoItem(System.IO.FileInfo, System.String, Boolean, Boolean) Void MoveDirectoryInfoItem(System.IO.DirectoryInfo, System.String, Boolean) Void CopyAndDelete(System.IO.DirectoryInfo, System.String, Boolean) Void GetProperty(System.String, System.Collections.ObjectModel.Collection`1[System.String]) Void SetProperty(System.String, System.Management.Automation.PSObject) Void ClearProperty(System.String, System.Collections.ObjectModel.Collection`1[System.String]) System.Management.Automation.Provider.IContentReader GetContentReader(System.String) System.Object GetContentReaderDynamicParameters(System.String) System.Management.Automation.Provider.IContentWriter GetContentWriter(System.String) Void ClearContent(System.String) Void ValidateParameters(Boolean) Void GetSecurityDescriptor(System.String, System.Security.AccessControl.AccessControlSections) Void SetSecurityDescriptor(System.String, System.Security.AccessControl.ObjectSecurity) Void SetSecurityDescriptor(System.String, System.Security.AccessControl.ObjectSecurity, System.Security.AccessControl.AccessControlSections) Void <RemoveDirectoryInfoItem>g__WriteErrorHelper|43_0(System.Exception, <>c__DisplayClass43_0 ByRef) Void .ctor() Void .cctor() System.Collections.ObjectModel.Collection`1[System.Management.Automation.WildcardPattern] excludeMatcher System.Management.Automation.PSTraceSource tracer Int32 FILETRANSFERSIZE System.String ProviderName Microsoft.PowerShell.Commands.FileSystemProvider+ItemType Microsoft.PowerShell.Commands.FileSystemProvider+NativeMethods Microsoft.PowerShell.Commands.FileSystemProvider+NetResource Microsoft.PowerShell.Commands.FileSystemProvider+InodeTracker Microsoft.PowerShell.Commands.FileSystemProvider+<>c__DisplayClass43_0