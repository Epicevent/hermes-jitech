import json
import sqlite3
from copy import deepcopy
from pathlib import Path

import pytest

import agent.provider_usage_coverage as coverage_contract
from agent.provider_usage_coverage import manifest_digest
from agent.provider_usage_receipt import (
    build_provider_usage_receipt,
    receipt_digest,
    validate_provider_usage_export,
    validate_provider_usage_receipt,
)
from hermes_state import (
    ProviderCallConflictError,
    SessionDB,
    export_provider_usage_receipts_readonly,
)


def _call_id(index: int) -> str:
    return f"00000000-0000-4000-8000-{index:012d}"


def _fixture() -> dict:
    path = (
        Path(__file__).parents[1] / "fixtures" / "jitech-provider-usage-export-v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_fixture() -> dict:
    path = (
        Path(__file__).parents[1]
        / "fixtures"
        / "jitech-provider-usage-coverage-v1.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _coverage_variant(surface_code: str) -> dict:
    manifest = _coverage_fixture()
    manifest["surfaces"][0]["surfaceCode"] = surface_code
    manifest["manifestDigest"] = manifest_digest(manifest)
    coverage_contract.validate_provider_usage_coverage(manifest)
    return manifest


def _record_kwargs(index: int = 1, **overrides) -> dict:
    values = {
        "call_id": _call_id(index),
        "request_id": f"request-{index}",
        "api_call_index": index,
        "attempt": 1,
        "fallback_index": 0,
        "configured_provider": "gemini",
        "configured_model": "gemini-3.6-flash",
        "requested_provider": "gemini",
        "requested_model": "gemini-3.6-flash",
        "actual_provider": "gemini",
        "actual_model": "gemini-3.6-flash-001",
        "response_id": f"response-{index}",
        "evidence_source": "gemini_response.modelVersion",
        "finish_reason": "STOP",
        "usage": {
            "promptTokenCount": 100,
            "cachedContentTokenCount": 20,
            "candidatesTokenCount": 10,
            "thoughtsTokenCount": 4,
            "toolUsePromptTokenCount": 5,
            "totalTokenCount": 119,
            "serviceTier": "STANDARD",
        },
        "run_id": "run-1",
        "turn_id": "turn-1",
        "started_at": 1.0,
        "completed_at": 2.0,
        "status": "succeeded",
        "trigger": "user",
    }
    values.update(overrides)
    return values


def _builder_kwargs(**overrides) -> dict:
    values = {
        "ledger_seq": 1,
        "session_id": "session-1",
        "retry_of": None,
        "fallback_parent": None,
        "error_category": None,
        "producer_coverage_digest": _coverage_fixture()["manifestDigest"],
        **_record_kwargs(),
    }
    values.pop("api_call_index")
    values["raw_usage"] = values.pop("usage")
    values.update(overrides)
    return values


def _redigest(receipt: dict) -> dict:
    receipt["receiptDigest"] = receipt_digest(receipt)
    return receipt


def test_exact_shared_fixture_matches_readonly_export(tmp_path, monkeypatch):
    fixture = _fixture()
    validate_provider_usage_export(fixture)
    monkeypatch.setattr(
        coverage_contract,
        "provider_usage_coverage_manifest",
        lambda: deepcopy(fixture["coverageManifests"][0]),
    )

    db = SessionDB(tmp_path / "state.db")
    try:
        result = db.record_provider_call(
            "session-fixture-1",
            **_record_kwargs(
                101,
                request_id="request-fixture-1",
                run_id="run-fixture-1",
                turn_id="turn-fixture-1",
                response_id="response-fixture-1",
            ),
        )
        page = export_provider_usage_receipts_readonly(
            db_path=db.db_path,
            after=0,
            limit=1,
        )
    finally:
        db.close()

    assert result["receipt"] == fixture["receipts"][0]
    assert page == fixture


def test_validator_requires_valid_producer_coverage_digest():
    receipt = deepcopy(_fixture()["receipts"][0])
    receipt["producerCoverageDigest"] = "sha256:not-a-digest"
    _redigest(receipt)

    with pytest.raises(ValueError, match="producerCoverageDigest"):
        validate_provider_usage_receipt(receipt)


@pytest.mark.parametrize(
    "raw_usage",
    [
        {"prompt_tokens": -1},
        {"service_tier": ""},
        {"prompt_tokens_details": {"cached_tokens": 1.5}},
        {"prompt_tokens_details": [{"modality": None, "tokenCount": 1}]},
    ],
)
def test_raw_provider_usage_rejects_non_accounting_value_shapes(raw_usage):
    receipt = build_provider_usage_receipt(
        **_builder_kwargs(
            actual_provider="openai",
            raw_usage=raw_usage,
        )
    )

    with pytest.raises(ValueError, match="usage"):
        validate_provider_usage_receipt(receipt)


def test_export_requires_exact_sorted_unique_historical_manifests():
    fixture = _fixture()

    missing = deepcopy(fixture)
    missing["coverageManifests"] = []
    with pytest.raises(ValueError, match="exactly match"):
        validate_provider_usage_export(missing)

    duplicate = deepcopy(fixture)
    duplicate["coverageManifests"].append(
        deepcopy(duplicate["coverageManifests"][0])
    )
    with pytest.raises(ValueError, match="unique ascending"):
        validate_provider_usage_export(duplicate)

    second_manifest = _coverage_variant("fixture.second")
    second_receipt = deepcopy(fixture["receipts"][0])
    second_receipt["ledgerSeq"] = 2
    second_receipt["callId"] = _call_id(102)
    second_receipt["producerCoverageDigest"] = second_manifest["manifestDigest"]
    _redigest(second_receipt)
    unsorted = deepcopy(fixture)
    unsorted["receipts"].append(second_receipt)
    unsorted["count"] = 2
    unsorted["nextCursor"] = 2
    unsorted["highWatermark"] = 2
    unsorted["coverageManifests"].append(second_manifest)
    unsorted["coverageManifests"].sort(
        key=lambda item: item["manifestDigest"],
        reverse=True,
    )
    with pytest.raises(ValueError, match="unique ascending"):
        validate_provider_usage_export(unsorted)


def test_export_empty_page_requires_empty_manifest_set():
    empty = {
        "schema": "jitech-provider-usage-export/v1",
        "after": 0,
        "nextCursor": 0,
        "highWatermark": 0,
        "count": 0,
        "hasMore": False,
        "receipts": [],
        "coverageManifests": [],
    }
    validate_provider_usage_export(empty)

    unexpected = deepcopy(empty)
    unexpected["coverageManifests"] = [_coverage_fixture()]
    with pytest.raises(ValueError, match="exactly match"):
        validate_provider_usage_export(unexpected)


def test_export_validates_expected_product_family():
    with pytest.raises(ValueError, match="productFamily mismatch"):
        validate_provider_usage_export(_fixture(), expected_family="openclaw")


@pytest.mark.parametrize(
    ("raw_usage", "expected_missing"),
    [
        (
            {
                "promptTokenCount": 10,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 0,
                "toolUsePromptTokenCount": 0,
                "totalTokenCount": 11,
                "serviceTier": "STANDARD",
            },
            ["inputNonCached", "cacheRead", "cacheWrite"],
        ),
        (
            {
                "cachedContentTokenCount": 2,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 0,
                "toolUsePromptTokenCount": 0,
                "totalTokenCount": 11,
                "serviceTier": "STANDARD",
            },
            ["inputTotal", "inputNonCached", "cacheWrite"],
        ),
    ],
)
def test_input_non_cached_is_missing_when_either_source_is_missing(
    raw_usage,
    expected_missing,
):
    receipt = build_provider_usage_receipt(
        **_builder_kwargs(raw_usage=raw_usage),
    )

    assert receipt["usage"]["inputNonCached"] is None
    assert receipt["missingUsageFields"] == expected_missing
    assert receipt["missingReceiptFields"] == [
        f"usage.{field}" for field in expected_missing
    ]
    validate_provider_usage_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("configured_provider", ""),
        ("configured_model", None),
        ("requested_provider", "   "),
        ("requested_model", ""),
    ],
)
def test_builder_rejects_missing_configured_or_requested_identity(field, value):
    with pytest.raises(ValueError, match="must be a nonempty string"):
        build_provider_usage_receipt(**_builder_kwargs(**{field: value}))


def test_validator_recomputes_exact_usage_paths_and_order():
    fixture_receipt = _fixture()["receipts"][0]

    bogus = deepcopy(fixture_receipt)
    bogus["missingUsageFields"] = ["usage.bogus"]
    _redigest(bogus)
    with pytest.raises(ValueError, match="exact ordered null usage field list"):
        validate_provider_usage_receipt(bogus)

    omitted = deepcopy(fixture_receipt)
    omitted["usage"]["inputNonCached"] = None
    _redigest(omitted)
    with pytest.raises(ValueError, match="exact ordered null usage field list"):
        validate_provider_usage_receipt(omitted)

    reordered = deepcopy(omitted)
    reordered["missingUsageFields"] = ["cacheWrite", "inputNonCached"]
    reordered["missingReceiptFields"] = [
        "usage.cacheWrite",
        "usage.inputNonCached",
    ]
    _redigest(reordered)
    with pytest.raises(ValueError, match="exact ordered null usage field list"):
        validate_provider_usage_receipt(reordered)


def test_validator_recomputes_exact_receipt_paths_and_order():
    fixture_receipt = _fixture()["receipts"][0]

    unknown_trigger = deepcopy(fixture_receipt)
    unknown_trigger["trigger"] = "unknown"
    _redigest(unknown_trigger)
    with pytest.raises(ValueError, match="exact ordered applicable null field list"):
        validate_provider_usage_receipt(unknown_trigger)

    reordered = deepcopy(fixture_receipt)
    reordered["runId"] = None
    reordered["missingReceiptFields"] = ["usage.cacheWrite", "runId"]
    _redigest(reordered)
    with pytest.raises(ValueError, match="exact ordered applicable null field list"):
        validate_provider_usage_receipt(reordered)


@pytest.mark.parametrize("status", ["interrupted", "cancelled"])
def test_validator_requires_error_category_for_non_success_statuses(status):
    receipt = deepcopy(_fixture()["receipts"][0])
    receipt["status"] = status
    receipt["finishReason"] = None
    receipt["errorCategory"] = None
    _redigest(receipt)

    with pytest.raises(ValueError, match="exact ordered applicable null field list"):
        validate_provider_usage_receipt(receipt)


def test_status_evidence_fields_are_mutually_exclusive():
    succeeded = deepcopy(_fixture()["receipts"][0])
    succeeded["errorCategory"] = "unexpected"
    _redigest(succeeded)
    with pytest.raises(ValueError, match="null errorCategory"):
        validate_provider_usage_receipt(succeeded)

    failed = deepcopy(_fixture()["receipts"][0])
    failed["status"] = "failed"
    failed["errorCategory"] = "HTTPError"
    _redigest(failed)
    with pytest.raises(ValueError, match="null finishReason"):
        validate_provider_usage_receipt(failed)


def test_retry_timing_and_export_cursor_invariants_match_collector_contract():
    invalid_fallback = deepcopy(_fixture()["receipts"][0])
    invalid_fallback["fallbackIndex"] = 1
    _redigest(invalid_fallback)
    with pytest.raises(ValueError, match="fallbackIndex must be 0"):
        validate_provider_usage_receipt(invalid_fallback)

    reversed_time = deepcopy(_fixture()["receipts"][0])
    reversed_time["completedAt"] = "1970-01-01T00:00:00.000Z"
    _redigest(reversed_time)
    with pytest.raises(ValueError, match="precedes"):
        validate_provider_usage_receipt(reversed_time)

    no_progress = {
        "schema": "jitech-provider-usage-export/v1",
        "after": 7,
        "nextCursor": 7,
        "highWatermark": 8,
        "count": 0,
        "hasMore": True,
        "receipts": [],
        "coverageManifests": [],
    }
    with pytest.raises(ValueError, match="cursor progress"):
        validate_provider_usage_export(no_progress, expected_after=7)

    rollback = deepcopy(no_progress)
    rollback["highWatermark"] = 6
    rollback["hasMore"] = False
    with pytest.raises(ValueError, match="moved backwards"):
        validate_provider_usage_export(rollback, expected_after=7)


def test_all_applicable_null_fields_are_unavailable():
    receipt = deepcopy(_fixture()["receipts"][0])
    for field in ("runId", "turnId", "requestId", "sessionId"):
        receipt[field] = None
    receipt["trigger"] = "unknown"
    receipt["actual"] = {
        "provider": None,
        "model": None,
        "responseId": None,
        "evidenceSource": None,
    }
    receipt["finishReason"] = None
    receipt["usage"] = {field: None for field in receipt["usage"]}
    receipt["missingUsageFields"] = [
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
    ]
    receipt["usageCoverage"] = "unavailable"
    receipt["missingReceiptFields"] = [
        "runId",
        "turnId",
        "requestId",
        "sessionId",
        "trigger",
        "actual.provider",
        "actual.model",
        "actual.responseId",
        "actual.evidenceSource",
        "finishReason",
        *(f"usage.{field}" for field in receipt["missingUsageFields"]),
    ]
    receipt["receiptCoverage"] = "unavailable"
    _redigest(receipt)

    validate_provider_usage_receipt(receipt)


def test_provider_call_replay_is_idempotent_and_divergence_is_atomic(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        first = db.record_provider_call("session-1", **_record_kwargs())
        replay = db.record_provider_call("session-1", **_record_kwargs())
        changed = _record_kwargs(usage={"promptTokenCount": 101})
        statements = []
        db._conn.set_trace_callback(statements.append)
        try:
            with pytest.raises(ProviderCallConflictError):
                db.record_provider_call("session-1", **changed)
        finally:
            db._conn.set_trace_callback(None)

        calls = db.get_provider_calls("session-1")
        conflict = db._conn.execute(
            "SELECT * FROM provider_call_conflicts WHERE call_id = ?",
            (_call_id(1),),
        ).fetchone()
    finally:
        db.close()

    assert first["inserted"] is True
    assert replay["inserted"] is False
    assert len(calls) == 1
    assert calls[0]["usage"]["promptTokenCount"] == 100
    assert conflict is not None
    assert conflict["existing_digest"] != conflict["incoming_digest"]
    audit_index = next(
        index
        for index, statement in enumerate(statements)
        if "INSERT INTO provider_call_conflicts" in statement
    )
    begin_index = max(
        index
        for index, statement in enumerate(statements[:audit_index])
        if statement == "BEGIN IMMEDIATE"
    )
    commit_index = next(
        index
        for index, statement in enumerate(
            statements[audit_index + 1 :], audit_index + 1
        )
        if statement == "COMMIT"
    )
    assert begin_index < audit_index < commit_index


def test_export_preserves_creation_time_coverage_across_delayed_collection(
    tmp_path,
    monkeypatch,
):
    first_manifest = _coverage_variant("fixture.historical.first")
    second_manifest = _coverage_variant("fixture.historical.second")
    collection_time_manifest = _coverage_variant("fixture.collection.current")
    current_manifest = first_manifest
    monkeypatch.setattr(
        coverage_contract,
        "provider_usage_coverage_manifest",
        lambda: deepcopy(current_manifest),
    )

    db = SessionDB(tmp_path / "state.db")
    try:
        first = db.record_provider_call("session-1", **_record_kwargs(1))
        current_manifest = second_manifest
        second = db.record_provider_call("session-1", **_record_kwargs(2))

        # Collection happens after the producer has moved to a third manifest.
        current_manifest = collection_time_manifest
        full_page = export_provider_usage_receipts_readonly(
            db_path=db.db_path,
            after=0,
            limit=10,
        )
        first_page = db.export_provider_usage_receipts(after=0, limit=1)
        second_page = db.export_provider_usage_receipts(
            after=first_page["nextCursor"],
            limit=1,
        )
    finally:
        db.close()

    assert first["receipt"]["producerCoverageDigest"] == first_manifest[
        "manifestDigest"
    ]
    assert second["receipt"]["producerCoverageDigest"] == second_manifest[
        "manifestDigest"
    ]
    assert [
        item["manifestDigest"] for item in full_page["coverageManifests"]
    ] == sorted({first_manifest["manifestDigest"], second_manifest["manifestDigest"]})
    assert collection_time_manifest["manifestDigest"] not in {
        item["manifestDigest"] for item in full_page["coverageManifests"]
    }
    assert first_page["coverageManifests"] == [first_manifest]
    assert second_page["coverageManifests"] == [second_manifest]


def test_call_and_coverage_manifest_binding_roll_back_together(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db._conn.execute(
            "CREATE TRIGGER reject_provider_call BEFORE INSERT ON provider_calls "
            "BEGIN SELECT RAISE(ABORT, 'test call rejection'); END"
        )

        with pytest.raises(sqlite3.IntegrityError, match="test call rejection"):
            db.record_provider_call("session-1", **_record_kwargs())

        call_count = db._conn.execute(
            "SELECT COUNT(*) FROM provider_calls"
        ).fetchone()[0]
        manifest_count = db._conn.execute(
            "SELECT COUNT(*) FROM provider_usage_coverage_manifests"
        ).fetchone()[0]
    finally:
        db.close()

    assert call_count == 0
    assert manifest_count == 0


def test_stored_coverage_manifests_are_sql_immutable(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        result = db.record_provider_call("session-1", **_record_kwargs())
        digest = result["receipt"]["producerCoverageDigest"]

        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db._conn.execute(
                "UPDATE provider_usage_coverage_manifests "
                "SET manifest_json = '{}' WHERE manifest_digest = ?",
                (digest,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db._conn.execute(
                "DELETE FROM provider_usage_coverage_manifests "
                "WHERE manifest_digest = ?",
                (digest,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            db._conn.execute(
                "INSERT OR REPLACE INTO provider_usage_coverage_manifests "
                "(manifest_digest, manifest_json, created_at) VALUES (?, '{}', 0)",
                (digest,),
            )
        stored = db._conn.execute(
            "SELECT manifest_json FROM provider_usage_coverage_manifests "
            "WHERE manifest_digest = ?",
            (digest,),
        ).fetchone()
    finally:
        db.close()

    assert stored is not None


def test_readonly_export_is_monotonic_content_free_and_never_initializes(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        for index in range(1, 4):
            db.record_provider_call(
                "session-1",
                **_record_kwargs(
                    index,
                    usage={
                        "promptTokenCount": index,
                        "secretPrompt": "must-not-export",
                    },
                ),
            )
        first = export_provider_usage_receipts_readonly(
            db_path=db.db_path,
            after=0,
            limit=2,
        )
        second = export_provider_usage_receipts_readonly(
            db_path=db.db_path,
            after=first["nextCursor"],
            limit=2,
        )
    finally:
        db.close()

    assert [item["ledgerSeq"] for item in first["receipts"]] == [1, 2]
    assert [item["ledgerSeq"] for item in second["receipts"]] == [3]
    assert first["highWatermark"] == 3
    assert first["hasMore"] is True
    assert second["hasMore"] is False
    assert "must-not-export" not in str(first)

    missing = tmp_path / "must-not-be-created.db"
    with pytest.raises(sqlite3.OperationalError):
        export_provider_usage_receipts_readonly(db_path=missing)
    assert not missing.exists()


def test_export_rejects_an_incomplete_ledger_row(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.record_provider_call("session-1", **_record_kwargs())

        def corrupt(conn):
            conn.execute(
                "UPDATE provider_calls SET export_receipt_json = NULL "
                "WHERE call_id = ?",
                (_call_id(1),),
            )

        db._execute_write(corrupt)
        with pytest.raises(ValueError, match="incomplete row"):
            db.export_provider_usage_receipts()
    finally:
        db.close()


def test_export_rejects_a_divergent_manifest_binding(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        db.record_provider_call("session-1", **_record_kwargs())

        def corrupt(conn):
            conn.execute(
                "UPDATE provider_calls SET producer_coverage_digest = ? "
                "WHERE call_id = ?",
                ("sha256:" + "0" * 64, _call_id(1)),
            )

        db._execute_write(corrupt)
        with pytest.raises(ValueError, match="coverage binding mismatch"):
            db.export_provider_usage_receipts()
    finally:
        db.close()


def test_authoritative_provider_table_has_no_pricing_projection_columns(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        columns = {
            row["name"]
            for row in db._conn.execute("PRAGMA table_info(provider_calls)").fetchall()
        }
    finally:
        db.close()

    assert not any("cost" in column or "pricing" in column for column in columns)
