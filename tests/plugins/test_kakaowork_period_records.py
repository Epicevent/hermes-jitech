from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from plugins.kwrag_slot import period_records


NOW = datetime(2026, 8, 20, 3, 0, tzinfo=timezone.utc)


MESSAGE_DDL = """
CREATE TABLE messages (
  conversation_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  room_name TEXT,
  request_id TEXT,
  user_id TEXT,
  user_name TEXT,
  sent_time INTEGER NOT NULL,
  text_kind TEXT,
  plain_text TEXT,
  decrypt_status TEXT,
  PRIMARY KEY (conversation_id, message_id)
);
CREATE TABLE attachments (
  conversation_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  block_index INTEGER NOT NULL,
  block_type TEXT,
  file_name TEXT,
  mime_type TEXT,
  nas_path TEXT,
  PRIMARY KEY (conversation_id, message_id, block_index)
);
"""


def _package(root: Path, rooms: list[str] | None = None) -> Path:
    package = root / "kw" / "package"
    package.mkdir(parents=True)
    (package / "membership.json").write_text(
        json.dumps(
            {
                "schema": "kw-user-membership/1",
                "user_id": "7519030",
                "conversation_ids": rooms if rooms is not None else ["room-a"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with sqlite3.connect(package / "messages.sqlite") as connection:
        connection.executescript(MESSAGE_DDL)
    return package


def _insert_message(
    package: Path,
    *,
    room: str,
    message_id: str,
    sent_time: int,
    text: str = "message",
    decrypt_status: str = "ok",
    room_name: str | None = None,
) -> None:
    with sqlite3.connect(package / "messages.sqlite") as connection:
        connection.execute(
            "INSERT INTO messages(conversation_id,message_id,room_name,request_id,"
            "user_id,user_name,sent_time,text_kind,plain_text,decrypt_status) "
            "VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                room,
                message_id,
                room_name or room,
                f"request-{message_id}",
                "sender-1",
                "발신자",
                sent_time,
                "text",
                text,
                decrypt_status,
            ),
        )


def _coverage(manifest: dict, source_root: Path) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for batch in manifest["batches"]:
        cursor = None
        returned = 0
        while True:
            page = period_records.read_batch(
                manifest["snapshot_token"],
                batch["batch_id"],
                cursor,
                source_root=source_root,
            )
            returned += page["returned_count"]
            cursor = page["next_cursor"]
            if cursor is None:
                assert returned == batch["message_count"]
                result.append(
                    {
                        "batch_id": batch["batch_id"],
                        "coverage_digest": page["batch_coverage_digest"],
                    }
                )
                break
    return result


def test_1001_records_are_deterministically_batched_paged_and_reconciled(tmp_path: Path) -> None:
    package = _package(tmp_path)
    end = int(NOW.timestamp())
    with sqlite3.connect(package / "messages.sqlite") as connection:
        connection.executemany(
            "INSERT INTO messages VALUES(?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    "room-a",
                    f"message-{index:04d}",
                    "대용량방",
                    f"request-{index}",
                    "sender-1",
                    "발신자",
                    end - 10_000 + index,
                    "text",
                    "가",
                    "ok",
                )
                for index in range(1001)
            ],
        )
        connection.execute(
            "INSERT INTO attachments VALUES(?,?,?,?,?,?,?)",
            (
                "room-a",
                "message-0000",
                0,
                "file",
                "evidence.pdf",
                "application/pdf",
                "attachments/room-a/message-0000/evidence.pdf",
            ),
        )

    first = period_records.manifest("rolling_7d", source_root=tmp_path, now=NOW)
    second = period_records.manifest("rolling_7d", source_root=tmp_path, now=NOW)

    assert first["totals"] == {
        "rooms": 1,
        "messages": 1001,
        "attachments": 1,
        "text_characters": 1001,
        "text_utf8_bytes": 3003,
        "decrypt_failures": 0,
        "unsafe_attachment_references": 0,
    }
    assert len(first["batches"]) == 6
    assert [batch["batch_id"] for batch in first["batches"]] == [
        batch["batch_id"] for batch in second["batches"]
    ]
    assert max(batch["message_count"] for batch in first["batches"]) == 200
    coverage = _coverage(first, tmp_path)
    reconciled = period_records.reconcile(
        first["snapshot_token"], coverage, source_root=tmp_path
    )
    assert reconciled["complete"] is True
    assert reconciled["source_total_messages"] == 1001
    assert reconciled["processed_messages"] == 1001
    assert reconciled["failed_messages"] == 0
    assert reconciled["uncovered_messages"] == 0

    first_page = period_records.read_batch(
        first["snapshot_token"], first["batches"][0]["batch_id"], source_root=tmp_path
    )
    assert first_page["returned_count"] == 50
    assert "batch_coverage_digest" not in first_page
    record = first_page["records"][0]
    assert record["message_id"] == "message-0000"
    assert record["stable_message_id"].startswith("sha256:")
    assert record["attachments"][0]["nas_reference"].startswith("attachments/")


def test_membership_is_a_hard_room_filter(tmp_path: Path) -> None:
    package = _package(tmp_path, ["allowed"])
    sent = int(NOW.timestamp()) - 1_000
    _insert_message(package, room="allowed", message_id="a", sent_time=sent)
    _insert_message(package, room="forbidden", message_id="b", sent_time=sent)

    result = period_records.manifest("rolling_7d", source_root=tmp_path, now=NOW)

    assert result["connection"]["membership_room_count"] == 1
    assert result["totals"]["messages"] == 1
    page = period_records.read_batch(
        result["snapshot_token"], result["batches"][0]["batch_id"], source_root=tmp_path
    )
    assert {record["conversation_id"] for record in page["records"]} == {"allowed"}


def test_connected_empty_period_reports_freshness_and_reconciles(tmp_path: Path) -> None:
    package = _package(tmp_path)
    _insert_message(
        package,
        room="room-a",
        message_id="old",
        sent_time=int(NOW.timestamp()) - 40 * 24 * 60 * 60,
    )
    old = int(NOW.timestamp()) - 40 * 24 * 60 * 60
    os.utime(package / "messages.sqlite", (old, old))

    result = period_records.manifest("rolling_7d", source_root=tmp_path, now=NOW)

    assert result["status"] == "ready"
    assert result["connection"]["status"] == "connected"
    assert result["totals"]["messages"] == 0
    assert result["batches"] == []
    assert result["freshness"]["stale"] is True
    reconciled = period_records.reconcile(
        result["snapshot_token"], [], source_root=tmp_path
    )
    assert reconciled["complete"] is False
    assert reconciled["source_total_messages"] == 0
    assert reconciled["freshness"]["status"] == "stale"


def test_recent_database_publication_is_fresh_despite_old_message_activity(tmp_path: Path) -> None:
    package = _package(tmp_path)
    old_message = int(NOW.timestamp()) - 40 * 24 * 60 * 60
    _insert_message(
        package,
        room="room-a",
        message_id="old",
        sent_time=old_message,
    )
    recent_publication = int(NOW.timestamp()) - 60 * 60
    os.utime(
        package / "messages.sqlite",
        (recent_publication, recent_publication),
    )

    result = period_records.manifest("rolling_7d", source_root=tmp_path, now=NOW)

    assert result["totals"]["messages"] == 0
    assert result["freshness"]["status"] == "fresh"
    assert result["freshness"]["stale"] is False
    assert result["freshness"]["latest_observed_at"] == recent_publication
    assert result["freshness"]["database_lag_to_period_end_seconds"] == 60 * 60
    assert result["freshness"]["activity_lag_to_period_end_seconds"] == 40 * 24 * 60 * 60
    reconciled = period_records.reconcile(
        result["snapshot_token"], [], source_root=tmp_path
    )
    assert reconciled["complete"] is True


def test_missing_corrupt_membership_and_schema_are_structured_diagnostics(tmp_path: Path) -> None:
    missing = period_records.execute_period_records(
        {"operation": "manifest", "period": "rolling_7d"},
        source_root=tmp_path,
        now=NOW,
    )
    assert missing["status"] == "unavailable"
    assert missing["error"]["code"] == "package_missing"

    invalid_membership_root = tmp_path / "membership"
    package = _package(invalid_membership_root)
    (package / "membership.json").write_text("not-json", encoding="utf-8")
    invalid_membership = period_records.execute_period_records(
        {"operation": "manifest", "period": "rolling_7d"},
        source_root=invalid_membership_root,
        now=NOW,
    )
    assert invalid_membership["error"]["code"] == "membership_invalid"

    corrupt_root = tmp_path / "corrupt"
    package = _package(corrupt_root)
    (package / "messages.sqlite").write_bytes(b"not-a-sqlite-database")
    corrupt = period_records.execute_period_records(
        {"operation": "manifest", "period": "rolling_7d"},
        source_root=corrupt_root,
        now=NOW,
    )
    assert corrupt["error"]["code"] == "database_corrupt"

    schema_root = tmp_path / "schema"
    package = _package(schema_root)
    with sqlite3.connect(package / "messages.sqlite") as connection:
        connection.execute("DROP TABLE attachments")
    invalid_schema = period_records.execute_period_records(
        {"operation": "manifest", "period": "rolling_7d"},
        source_root=schema_root,
        now=NOW,
    )
    assert invalid_schema["error"]["code"] == "schema_invalid"


def test_snapshot_mutation_requires_a_new_manifest(tmp_path: Path) -> None:
    package = _package(tmp_path)
    sent = int(NOW.timestamp()) - 100
    _insert_message(package, room="room-a", message_id="before", sent_time=sent)
    result = period_records.manifest("rolling_7d", source_root=tmp_path, now=NOW)
    _insert_message(package, room="room-a", message_id="after", sent_time=sent + 1)

    changed = period_records.execute_period_records(
        {
            "operation": "read_batch",
            "snapshot_token": result["snapshot_token"],
            "batch_id": result["batches"][0]["batch_id"],
        },
        source_root=tmp_path,
    )

    assert changed["status"] == "error"
    assert changed["error"]["code"] == "snapshot_mismatch"
    assert changed["complete"] is False


def test_reconcile_rejects_missing_duplicate_and_wrong_coverage(tmp_path: Path) -> None:
    package = _package(tmp_path)
    sent = int(NOW.timestamp()) - 1_000
    for index in range(201):
        _insert_message(
            package,
            room="room-a",
            message_id=f"m-{index}",
            sent_time=sent + index,
        )
    result = period_records.manifest("rolling_7d", source_root=tmp_path, now=NOW)
    coverage = _coverage(result, tmp_path)

    incomplete = period_records.reconcile(
        result["snapshot_token"],
        [coverage[0], coverage[0]],
        source_root=tmp_path,
    )
    assert incomplete["complete"] is False
    assert incomplete["duplicate_batch_ids"] == [coverage[0]["batch_id"]]
    assert incomplete["missing_batch_ids"] == [coverage[1]["batch_id"]]
    assert incomplete["uncovered_messages"] == 201

    wrong = period_records.reconcile(
        result["snapshot_token"],
        [coverage[0], {**coverage[1], "coverage_digest": "sha256:" + "0" * 64}],
        source_root=tmp_path,
    )
    assert wrong["complete"] is False
    assert wrong["digest_mismatch_batch_ids"] == [coverage[1]["batch_id"]]
    assert wrong["uncovered_messages"] == 1


def test_decrypt_failure_is_counted_and_prevents_complete_claim(tmp_path: Path) -> None:
    package = _package(tmp_path)
    sent = int(NOW.timestamp()) - 100
    _insert_message(package, room="room-a", message_id="ok", sent_time=sent)
    _insert_message(
        package,
        room="room-a",
        message_id="failed",
        sent_time=sent + 1,
        decrypt_status="failed",
    )
    result = period_records.manifest("rolling_7d", source_root=tmp_path, now=NOW)
    reconciled = period_records.reconcile(
        result["snapshot_token"], _coverage(result, tmp_path), source_root=tmp_path
    )

    assert reconciled["complete"] is False
    assert reconciled["source_total_messages"] == 2
    assert reconciled["processed_messages"] == 1
    assert reconciled["failed_messages"] == 1
    assert reconciled["uncovered_messages"] == 0


def test_previous_calendar_week_uses_seoul_monday_and_exclusive_end(tmp_path: Path) -> None:
    package = _package(tmp_path)
    expected_start = int(datetime(2026, 8, 9, 15, 0, tzinfo=timezone.utc).timestamp())
    expected_end = int(datetime(2026, 8, 16, 15, 0, tzinfo=timezone.utc).timestamp())
    _insert_message(package, room="room-a", message_id="start", sent_time=expected_start)
    _insert_message(package, room="room-a", message_id="last", sent_time=expected_end - 1)
    _insert_message(package, room="room-a", message_id="end", sent_time=expected_end)

    result = period_records.manifest(
        "previous_calendar_week", source_root=tmp_path, now=NOW
    )

    assert result["period"]["start"] == expected_start
    assert result["period"]["end"] == expected_end
    assert result["period"]["end_exclusive"] is True
    assert result["totals"]["messages"] == 2


def test_model_schema_cannot_select_identity_path_sql_or_arbitrary_dates() -> None:
    from plugins.kwrag_slot.tools import PERIOD_RECORDS_SCHEMA

    properties = PERIOD_RECORDS_SCHEMA["parameters"]["properties"]
    assert set(properties) == {
        "operation",
        "period",
        "snapshot_token",
        "batch_id",
        "cursor",
        "coverage",
    }
    assert PERIOD_RECORDS_SCHEMA["parameters"]["additionalProperties"] is False
    assert properties["period"]["enum"] == ["rolling_7d", "previous_calendar_week"]
