"""Read-only, exhaustive KakaoWork period records for one Hermes slot.

The model chooses only a fixed period and an operation.  Package discovery,
membership scope, SQLite queries, batching, pagination, and reconciliation are
owned here so a prompt cannot widen the slot's mounted authorization boundary.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


_SOURCE_ROOT_ENV = "JITECH_KWRAG_SOURCE_ROOT"
_DEFAULT_SOURCE_ROOT = Path("/workspace/nas_docs")
# Korea has used UTC+09:00 without daylight-saving changes since 1988.  A
# fixed offset keeps this runtime dependency-free on minimal customer images.
_SEOUL = timezone(timedelta(hours=9), name="Asia/Seoul")
_BATCH_MAX_MESSAGES = 200
_BATCH_MAX_UTF8_BYTES = 32_768
_PAGE_MAX_MESSAGES = 50
_FRESHNESS_TOLERANCE_SECONDS = 48 * 60 * 60
_TOKEN_SECRET = os.urandom(32)
_EXPECTED_MESSAGE_COLUMNS = {
    "conversation_id",
    "message_id",
    "room_name",
    "request_id",
    "user_id",
    "user_name",
    "sent_time",
    "text_kind",
    "plain_text",
    "decrypt_status",
}
_EXPECTED_ATTACHMENT_COLUMNS = {
    "conversation_id",
    "message_id",
    "block_index",
    "block_type",
    "file_name",
    "mime_type",
    "nas_path",
}
_DECRYPT_SUCCESS = {"ok", "success", "plain", "plaintext"}


class PeriodRecordsError(ValueError):
    """A bounded period-record operation could not be completed."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _Period:
    preset: str
    start: int
    end: int


@dataclass(frozen=True)
class _Package:
    root: Path
    database: Path
    membership: Path


@dataclass(frozen=True)
class _Message:
    conversation_id: str
    message_id: str
    room_name: str
    request_id: str | None
    user_id: str | None
    user_name: str | None
    sent_time: int
    text_kind: str | None
    plain_text: str
    decrypt_status: str
    local_date: str
    stable_message_id: str
    attachments: tuple[dict[str, Any], ...]

    @property
    def text_characters(self) -> int:
        return len(self.plain_text)

    @property
    def text_utf8_bytes(self) -> int:
        return len(self.plain_text.encode("utf-8"))

    @property
    def decrypt_failed(self) -> bool:
        return self.decrypt_status.strip().lower() not in _DECRYPT_SUCCESS


@dataclass(frozen=True)
class _Batch:
    batch_id: str
    conversation_id: str
    room_name: str
    local_date: str
    messages: tuple[_Message, ...]
    coverage_digest: str

    @property
    def text_characters(self) -> int:
        return sum(message.text_characters for message in self.messages)

    @property
    def text_utf8_bytes(self) -> int:
        return sum(message.text_utf8_bytes for message in self.messages)

    @property
    def decrypt_failure_count(self) -> int:
        return sum(message.decrypt_failed for message in self.messages)


@dataclass(frozen=True)
class _Snapshot:
    period: _Period
    observed_at: int
    database_digest: str
    membership_digest: str
    membership_room_count: int
    database_modified_at: int
    max_source_sent_at: int | None
    batches: tuple[_Batch, ...]
    total_messages: int
    total_attachments: int
    total_text_characters: int
    total_text_utf8_bytes: int
    decrypt_failure_count: int
    unsafe_attachment_reference_count: int
    snapshot_id: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _unb64(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise PeriodRecordsError("invalid_token", "snapshot or cursor token is invalid")
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError) as exc:
        raise PeriodRecordsError("invalid_token", "snapshot or cursor token is invalid") from exc


def _encode_token(kind: str, payload: Mapping[str, Any]) -> str:
    envelope = _canonical_bytes({"kind": kind, "payload": dict(payload), "version": 1})
    signature = hmac.new(_TOKEN_SECRET, envelope, hashlib.sha256).digest()
    return f"{_b64(envelope)}.{_b64(signature)}"


def _decode_token(token: Any, expected_kind: str) -> dict[str, Any]:
    if not isinstance(token, str) or token.count(".") != 1:
        raise PeriodRecordsError("invalid_token", "snapshot or cursor token is invalid")
    encoded, signature = token.split(".", 1)
    envelope = _unb64(encoded)
    supplied = _unb64(signature)
    expected = hmac.new(_TOKEN_SECRET, envelope, hashlib.sha256).digest()
    if not hmac.compare_digest(supplied, expected):
        raise PeriodRecordsError("invalid_token", "snapshot or cursor token is invalid")
    try:
        parsed = json.loads(envelope.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PeriodRecordsError("invalid_token", "snapshot or cursor token is invalid") from exc
    if (
        not isinstance(parsed, dict)
        or parsed.get("version") != 1
        or parsed.get("kind") != expected_kind
        or not isinstance(parsed.get("payload"), dict)
    ):
        raise PeriodRecordsError("invalid_token", "snapshot or cursor token is invalid")
    return dict(parsed["payload"])


def _file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
    except OSError as exc:
        raise PeriodRecordsError("package_unreadable", "Kakao package is not readable") from exc
    return "sha256:" + hasher.hexdigest()


def _period(preset: Any, now: datetime | None = None) -> _Period:
    if preset not in {"rolling_7d", "previous_calendar_week"}:
        raise PeriodRecordsError("invalid_period", "period must be rolling_7d or previous_calendar_week")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    current = current.astimezone(timezone.utc)
    if preset == "rolling_7d":
        end = int(current.timestamp())
        return _Period(preset=preset, start=end - 7 * 24 * 60 * 60, end=end)
    local = current.astimezone(_SEOUL)
    this_monday = (local - timedelta(days=local.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    previous_monday = this_monday - timedelta(days=7)
    return _Period(
        preset=preset,
        start=int(previous_monday.timestamp()),
        end=int(this_monday.timestamp()),
    )


def _source_root(source_root: Path | None) -> Path:
    root = Path(source_root) if source_root is not None else Path(
        os.environ.get(_SOURCE_ROOT_ENV, _DEFAULT_SOURCE_ROOT)
    )
    if not root.is_absolute():
        raise PeriodRecordsError("package_missing", "Kakao source root is unavailable")
    return root


def _resolve_package(source_root: Path | None = None) -> _Package:
    root = _source_root(source_root)
    candidates = (root, root / "kw" / "package")
    for candidate in candidates:
        database = candidate / "messages.sqlite"
        membership = candidate / "membership.json"
        if database.is_file() and membership.is_file():
            try:
                resolved_root = root.resolve(strict=True)
                resolved_candidate = candidate.resolve(strict=True)
                database = database.resolve(strict=True)
                membership = membership.resolve(strict=True)
            except OSError as exc:
                raise PeriodRecordsError("package_unreadable", "Kakao package is not readable") from exc
            if resolved_candidate != resolved_root and resolved_root not in resolved_candidate.parents:
                raise PeriodRecordsError("package_unreadable", "Kakao package is outside the mounted source")
            return _Package(resolved_candidate, database, membership)
    raise PeriodRecordsError("package_missing", "Kakao package is not mounted")


def _membership(path: Path) -> tuple[tuple[str, ...], str]:
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PeriodRecordsError("membership_invalid", "Kakao membership is invalid") from exc
    rooms = parsed.get("conversation_ids") if isinstance(parsed, dict) else None
    if (
        not isinstance(rooms, list)
        or any(not isinstance(room, str) or not room for room in rooms)
        or len(set(rooms)) != len(rooms)
    ):
        raise PeriodRecordsError("membership_invalid", "Kakao membership is invalid")
    return tuple(sorted(rooms)), _sha256(raw)


def _connect(database: Path) -> sqlite3.Connection:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        integrity = connection.execute("PRAGMA integrity_check(1)").fetchone()[0]
        if integrity != "ok":
            raise PeriodRecordsError("database_corrupt", "Kakao database integrity check failed")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            ).fetchall()
        }
        if not {"messages", "attachments"} <= tables:
            raise PeriodRecordsError("schema_invalid", "Kakao database schema is unsupported")
        message_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        attachment_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(attachments)").fetchall()
        }
        if not _EXPECTED_MESSAGE_COLUMNS <= message_columns or not _EXPECTED_ATTACHMENT_COLUMNS <= attachment_columns:
            raise PeriodRecordsError("schema_invalid", "Kakao database schema is unsupported")
        return connection
    except PeriodRecordsError:
        if connection is not None:
            connection.close()
        raise
    except sqlite3.DatabaseError as exc:
        if connection is not None:
            connection.close()
        raise PeriodRecordsError("database_corrupt", "Kakao database is unreadable") from exc
    except OSError as exc:
        raise PeriodRecordsError("package_unreadable", "Kakao database is unreadable") from exc


def _chunks(values: Sequence[str], size: int = 800) -> Iterable[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _query_messages(
    connection: sqlite3.Connection,
    rooms: Sequence[str],
    period: _Period,
) -> list[sqlite3.Row]:
    if not rooms:
        return []
    rows: list[sqlite3.Row] = []
    for room_chunk in _chunks(rooms):
        placeholders = ",".join("?" for _ in room_chunk)
        rows.extend(
            connection.execute(
                "SELECT conversation_id,message_id,room_name,request_id,user_id,user_name,"
                "sent_time,text_kind,plain_text,decrypt_status FROM messages "
                f"WHERE conversation_id IN ({placeholders}) AND sent_time>=? AND sent_time<?",
                (*room_chunk, period.start, period.end),
            ).fetchall()
        )
    rows.sort(key=lambda row: (str(row["conversation_id"]), int(row["sent_time"]), str(row["message_id"])))
    identities = [(str(row["conversation_id"]), str(row["message_id"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise PeriodRecordsError("message_identity_invalid", "Kakao message identities are not unique")
    return rows


def _query_attachments(
    connection: sqlite3.Connection,
    rooms: Sequence[str],
    period: _Period,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = {}
    if not rooms:
        return result
    for room_chunk in _chunks(rooms):
        placeholders = ",".join("?" for _ in room_chunk)
        rows = connection.execute(
            "SELECT a.conversation_id,a.message_id,a.block_index,a.block_type,"
            "a.file_name,a.mime_type,a.nas_path FROM attachments a JOIN messages m "
            "ON m.conversation_id=a.conversation_id AND m.message_id=a.message_id "
            f"WHERE m.conversation_id IN ({placeholders}) AND m.sent_time>=? AND m.sent_time<? "
            "ORDER BY a.conversation_id,a.message_id,a.block_index",
            (*room_chunk, period.start, period.end),
        ).fetchall()
        for row in rows:
            path = row["nas_path"]
            safe_reference = _safe_attachment_reference(path)
            item = {
                "block_index": int(row["block_index"]),
                "block_type": row["block_type"],
                "file_name": row["file_name"],
                "mime_type": row["mime_type"],
                "reference_status": "available" if safe_reference is not None else "invalid",
            }
            if safe_reference is not None:
                item["nas_reference"] = safe_reference
            result.setdefault((str(row["conversation_id"]), str(row["message_id"])), []).append(item)
    return result


def _safe_attachment_reference(value: Any) -> str | None:
    if not isinstance(value, str) or not value or "\\" in value:
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _stable_message_id(conversation_id: str, message_id: str) -> str:
    return _sha256(_canonical_bytes([conversation_id, message_id]))


def _to_messages(
    rows: Sequence[sqlite3.Row],
    attachments: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
) -> list[_Message]:
    messages: list[_Message] = []
    for row in rows:
        conversation_id = str(row["conversation_id"])
        message_id = str(row["message_id"])
        sent_time = int(row["sent_time"])
        messages.append(
            _Message(
                conversation_id=conversation_id,
                message_id=message_id,
                room_name=str(row["room_name"] or conversation_id),
                request_id=None if row["request_id"] is None else str(row["request_id"]),
                user_id=None if row["user_id"] is None else str(row["user_id"]),
                user_name=None if row["user_name"] is None else str(row["user_name"]),
                sent_time=sent_time,
                text_kind=None if row["text_kind"] is None else str(row["text_kind"]),
                plain_text="" if row["plain_text"] is None else str(row["plain_text"]),
                decrypt_status="" if row["decrypt_status"] is None else str(row["decrypt_status"]),
                local_date=datetime.fromtimestamp(sent_time, _SEOUL).date().isoformat(),
                stable_message_id=_stable_message_id(conversation_id, message_id),
                attachments=tuple(attachments.get((conversation_id, message_id), ())),
            )
        )
    messages.sort(
        key=lambda message: (
            message.conversation_id,
            message.local_date,
            message.sent_time,
            message.message_id,
        )
    )
    return messages


def _coverage_digest(messages: Sequence[_Message]) -> str:
    return _sha256(_canonical_bytes([message.stable_message_id for message in messages]))


def _build_batches(messages: Sequence[_Message]) -> tuple[_Batch, ...]:
    grouped: dict[tuple[str, str], list[_Message]] = {}
    for message in messages:
        grouped.setdefault((message.conversation_id, message.local_date), []).append(message)
    batches: list[_Batch] = []
    for (conversation_id, local_date), group in grouped.items():
        current: list[_Message] = []
        current_bytes = 0
        pieces: list[list[_Message]] = []
        for message in group:
            would_exceed = current and (
                len(current) >= _BATCH_MAX_MESSAGES
                or current_bytes + message.text_utf8_bytes > _BATCH_MAX_UTF8_BYTES
            )
            if would_exceed:
                pieces.append(current)
                current = []
                current_bytes = 0
            current.append(message)
            current_bytes += message.text_utf8_bytes
        if current:
            pieces.append(current)
        for piece_index, piece in enumerate(pieces, start=1):
            identity = {
                "conversation_id": conversation_id,
                "local_date": local_date,
                "piece": piece_index,
                "first": piece[0].stable_message_id,
                "last": piece[-1].stable_message_id,
                "count": len(piece),
            }
            suffix = hashlib.sha256(_canonical_bytes(identity)).hexdigest()[:16]
            batch_id = f"batch-{len(batches) + 1:04d}-{suffix}"
            batches.append(
                _Batch(
                    batch_id=batch_id,
                    conversation_id=conversation_id,
                    room_name=piece[0].room_name,
                    local_date=local_date,
                    messages=tuple(piece),
                    coverage_digest=_coverage_digest(piece),
                )
            )
    return tuple(batches)


def _snapshot(
    period: _Period,
    *,
    source_root: Path | None = None,
    observed_at: int | None = None,
) -> _Snapshot:
    package = _resolve_package(source_root)
    observed = int(datetime.now(timezone.utc).timestamp()) if observed_at is None else int(observed_at)
    database_before = _file_digest(package.database)
    rooms, membership_before = _membership(package.membership)
    try:
        database_modified_at = int(package.database.stat().st_mtime)
    except OSError as exc:
        raise PeriodRecordsError("package_unreadable", "Kakao package is not readable") from exc
    connection = _connect(package.database)
    try:
        max_source_row = None
        for room_chunk in _chunks(rooms):
            placeholders = ",".join("?" for _ in room_chunk)
            value = connection.execute(
                f"SELECT MAX(sent_time) FROM messages WHERE conversation_id IN ({placeholders})",
                tuple(room_chunk),
            ).fetchone()[0]
            if value is not None:
                max_source_row = max(int(value), max_source_row or int(value))
        rows = _query_messages(connection, rooms, period)
        attachment_map = _query_attachments(connection, rooms, period)
    except sqlite3.DatabaseError as exc:
        raise PeriodRecordsError("database_corrupt", "Kakao database query failed") from exc
    finally:
        connection.close()
    database_after = _file_digest(package.database)
    _, membership_after = _membership(package.membership)
    if database_before != database_after or membership_before != membership_after:
        raise PeriodRecordsError("snapshot_changed", "Kakao package changed during snapshot creation")
    messages = _to_messages(rows, attachment_map)
    batches = _build_batches(messages)
    unsafe_attachments = sum(
        attachment.get("reference_status") != "available"
        for message in messages
        for attachment in message.attachments
    )
    identity = {
        "database_digest": database_before,
        "membership_digest": membership_before,
        "period": {"preset": period.preset, "start": period.start, "end": period.end},
        "batches": [
            {
                "batch_id": batch.batch_id,
                "count": len(batch.messages),
                "coverage_digest": batch.coverage_digest,
            }
            for batch in batches
        ],
    }
    return _Snapshot(
        period=period,
        observed_at=observed,
        database_digest=database_before,
        membership_digest=membership_before,
        membership_room_count=len(rooms),
        database_modified_at=database_modified_at,
        max_source_sent_at=max_source_row,
        batches=batches,
        total_messages=len(messages),
        total_attachments=sum(len(message.attachments) for message in messages),
        total_text_characters=sum(message.text_characters for message in messages),
        total_text_utf8_bytes=sum(message.text_utf8_bytes for message in messages),
        decrypt_failure_count=sum(message.decrypt_failed for message in messages),
        unsafe_attachment_reference_count=unsafe_attachments,
        snapshot_id=_sha256(_canonical_bytes(identity)),
    )


def _freshness(snapshot: _Snapshot) -> dict[str, Any]:
    observations = [snapshot.database_modified_at]
    if snapshot.max_source_sent_at is not None:
        observations.append(snapshot.max_source_sent_at)
    latest_observed = max(observations)
    publication_lag = max(0, snapshot.period.end - latest_observed)
    activity_lag = (
        None
        if snapshot.max_source_sent_at is None
        else max(0, snapshot.period.end - snapshot.max_source_sent_at)
    )
    database_lag = max(0, snapshot.period.end - snapshot.database_modified_at)
    stale = publication_lag > _FRESHNESS_TOLERANCE_SECONDS
    return {
        "status": "stale" if stale else "fresh",
        "stale": stale,
        "tolerance_seconds": _FRESHNESS_TOLERANCE_SECONDS,
        "latest_observed_at": latest_observed,
        "lag_to_period_end_seconds": publication_lag,
        "activity_lag_to_period_end_seconds": activity_lag,
        "database_lag_to_period_end_seconds": database_lag,
        "max_source_sent_at": snapshot.max_source_sent_at,
        "database_modified_at": snapshot.database_modified_at,
    }


def _snapshot_token(snapshot: _Snapshot) -> str:
    return _encode_token(
        "snapshot",
        {
            "snapshot_id": snapshot.snapshot_id,
            "period": snapshot.period.preset,
            "start": snapshot.period.start,
            "end": snapshot.period.end,
            "observed_at": snapshot.observed_at,
        },
    )


def _snapshot_from_token(token: Any, source_root: Path | None) -> tuple[_Snapshot, dict[str, Any]]:
    payload = _decode_token(token, "snapshot")
    expected_fields = {"snapshot_id", "period", "start", "end", "observed_at"}
    if set(payload) != expected_fields:
        raise PeriodRecordsError("invalid_token", "snapshot token is invalid")
    if payload["period"] not in {"rolling_7d", "previous_calendar_week"}:
        raise PeriodRecordsError("invalid_token", "snapshot token is invalid")
    if any(isinstance(payload[field], bool) or not isinstance(payload[field], int) for field in ("start", "end", "observed_at")):
        raise PeriodRecordsError("invalid_token", "snapshot token is invalid")
    period = _Period(str(payload["period"]), int(payload["start"]), int(payload["end"]))
    snapshot = _snapshot(period, source_root=source_root, observed_at=int(payload["observed_at"]))
    if snapshot.snapshot_id != payload["snapshot_id"]:
        raise PeriodRecordsError("snapshot_mismatch", "Kakao package changed; create a new manifest")
    return snapshot, payload


def manifest(
    preset: str,
    *,
    source_root: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    period = _period(preset, now)
    last_error: PeriodRecordsError | None = None
    for _attempt in range(2):
        try:
            snapshot = _snapshot(
                period,
                source_root=source_root,
                observed_at=int((now or datetime.now(timezone.utc)).timestamp()),
            )
            break
        except PeriodRecordsError as exc:
            last_error = exc
            if exc.code != "snapshot_changed":
                raise
    else:
        assert last_error is not None
        raise last_error
    return {
        "schema_version": "jitech-kakaowork-period-records-v1",
        "operation": "manifest",
        "status": "ready",
        "period": {
            "preset": period.preset,
            "start": period.start,
            "end": period.end,
            "timezone": "Asia/Seoul",
            "end_exclusive": True,
        },
        "connection": {
            "status": "connected",
            "read_only": True,
            "membership_room_count": snapshot.membership_room_count,
            "database_digest": snapshot.database_digest,
            "membership_digest": snapshot.membership_digest,
        },
        "freshness": _freshness(snapshot),
        "totals": {
            "rooms": len({batch.conversation_id for batch in snapshot.batches}),
            "messages": snapshot.total_messages,
            "attachments": snapshot.total_attachments,
            "text_characters": snapshot.total_text_characters,
            "text_utf8_bytes": snapshot.total_text_utf8_bytes,
            "decrypt_failures": snapshot.decrypt_failure_count,
            "unsafe_attachment_references": snapshot.unsafe_attachment_reference_count,
        },
        "batch_limits": {
            "messages": _BATCH_MAX_MESSAGES,
            "text_utf8_bytes": _BATCH_MAX_UTF8_BYTES,
            "page_messages": _PAGE_MAX_MESSAGES,
        },
        "batches": [
            {
                "batch_id": batch.batch_id,
                "message_count": len(batch.messages),
                "text_utf8_bytes": batch.text_utf8_bytes,
                "page_count": (len(batch.messages) + _PAGE_MAX_MESSAGES - 1) // _PAGE_MAX_MESSAGES,
            }
            for batch in snapshot.batches
        ],
        "snapshot_token": _snapshot_token(snapshot),
    }


def _message_result(message: _Message) -> dict[str, Any]:
    return {
        "conversation_id": message.conversation_id,
        "message_id": message.message_id,
        "stable_message_id": message.stable_message_id,
        "room_name": message.room_name,
        "request_id": message.request_id,
        "sender": {"user_id": message.user_id, "user_name": message.user_name},
        "sent_time": message.sent_time,
        "local_time": datetime.fromtimestamp(message.sent_time, _SEOUL).isoformat(),
        "text_kind": message.text_kind,
        "plain_text": message.plain_text,
        "decrypt_status": message.decrypt_status,
        "attachments": list(message.attachments),
    }


def read_batch(
    snapshot_token: str,
    batch_id: str,
    cursor: str | None = None,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(batch_id, str) or not batch_id:
        raise PeriodRecordsError("invalid_batch", "batch_id is required")
    snapshot, _payload = _snapshot_from_token(snapshot_token, source_root)
    batch = next((item for item in snapshot.batches if item.batch_id == batch_id), None)
    if batch is None:
        raise PeriodRecordsError("invalid_batch", "batch_id is not part of this snapshot")
    offset = 0
    if cursor is not None:
        cursor_payload = _decode_token(cursor, "cursor")
        if set(cursor_payload) != {"snapshot_id", "batch_id", "offset"}:
            raise PeriodRecordsError("invalid_cursor", "cursor is invalid")
        if cursor_payload.get("snapshot_id") != snapshot.snapshot_id or cursor_payload.get("batch_id") != batch_id:
            raise PeriodRecordsError("invalid_cursor", "cursor does not belong to this batch")
        offset = cursor_payload.get("offset")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset <= 0:
            raise PeriodRecordsError("invalid_cursor", "cursor is invalid")
    if offset >= len(batch.messages):
        raise PeriodRecordsError("invalid_cursor", "cursor is past the end of the batch")
    page = batch.messages[offset : offset + _PAGE_MAX_MESSAGES]
    next_offset = offset + len(page)
    next_cursor = None
    if next_offset < len(batch.messages):
        next_cursor = _encode_token(
            "cursor",
            {"snapshot_id": snapshot.snapshot_id, "batch_id": batch_id, "offset": next_offset},
        )
    result = {
        "schema_version": "jitech-kakaowork-period-records-v1",
        "operation": "read_batch",
        "status": "ready",
        "batch_id": batch_id,
        "cursor": cursor,
        "next_cursor": next_cursor,
        "returned_count": len(page),
        "batch_total_count": len(batch.messages),
        "records": [_message_result(message) for message in page],
    }
    if next_cursor is None:
        result["batch_coverage_digest"] = batch.coverage_digest
    return result


def reconcile(
    snapshot_token: str,
    coverage: Sequence[Mapping[str, Any]],
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(coverage, Sequence) or isinstance(coverage, (str, bytes)):
        raise PeriodRecordsError("invalid_coverage", "coverage must be a list")
    snapshot, _payload = _snapshot_from_token(snapshot_token, source_root)
    submitted: list[tuple[str, str]] = []
    for item in coverage:
        if not isinstance(item, Mapping) or set(item) != {"batch_id", "coverage_digest"}:
            raise PeriodRecordsError("invalid_coverage", "coverage entry is invalid")
        batch_id = item.get("batch_id")
        digest = item.get("coverage_digest")
        if not isinstance(batch_id, str) or not isinstance(digest, str):
            raise PeriodRecordsError("invalid_coverage", "coverage entry is invalid")
        submitted.append((batch_id, digest))
    expected = {batch.batch_id: batch for batch in snapshot.batches}
    counts: dict[str, int] = {}
    for batch_id, _digest_value in submitted:
        counts[batch_id] = counts.get(batch_id, 0) + 1
    duplicate_batch_ids = sorted(batch_id for batch_id, count in counts.items() if count > 1)
    unknown_batch_ids = sorted(batch_id for batch_id in counts if batch_id not in expected)
    supplied = {batch_id: digest for batch_id, digest in submitted if batch_id in expected}
    missing_batch_ids = sorted(batch_id for batch_id in expected if batch_id not in supplied)
    digest_mismatch_batch_ids = sorted(
        batch_id
        for batch_id, digest in supplied.items()
        if expected[batch_id].coverage_digest != digest
    )
    valid_batch_ids = {
        batch_id
        for batch_id, digest in supplied.items()
        if counts.get(batch_id) == 1 and expected[batch_id].coverage_digest == digest
    }
    covered_messages = sum(len(expected[batch_id].messages) for batch_id in valid_batch_ids)
    failed_messages = sum(expected[batch_id].decrypt_failure_count for batch_id in valid_batch_ids)
    processed_messages = covered_messages - failed_messages
    uncovered_messages = snapshot.total_messages - covered_messages
    freshness = _freshness(snapshot)
    complete = not any(
        (
            duplicate_batch_ids,
            unknown_batch_ids,
            missing_batch_ids,
            digest_mismatch_batch_ids,
            uncovered_messages,
            failed_messages,
            snapshot.unsafe_attachment_reference_count,
            freshness["stale"],
        )
    )
    return {
        "schema_version": "jitech-kakaowork-period-records-v1",
        "operation": "reconcile",
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "period": {
            "preset": snapshot.period.preset,
            "start": snapshot.period.start,
            "end": snapshot.period.end,
            "timezone": "Asia/Seoul",
            "end_exclusive": True,
        },
        "freshness": freshness,
        "source_total_messages": snapshot.total_messages,
        "covered_messages": covered_messages,
        "processed_messages": processed_messages,
        "failed_messages": failed_messages,
        "uncovered_messages": uncovered_messages,
        "source_total_attachments": snapshot.total_attachments,
        "unsafe_attachment_references": snapshot.unsafe_attachment_reference_count,
        "source_decrypt_failures": snapshot.decrypt_failure_count,
        "missing_batch_ids": missing_batch_ids,
        "duplicate_batch_ids": duplicate_batch_ids,
        "unknown_batch_ids": unknown_batch_ids,
        "digest_mismatch_batch_ids": digest_mismatch_batch_ids,
    }


def execute_period_records(
    args: Mapping[str, Any],
    *,
    source_root: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Dispatch the fixed model surface and preserve structured diagnostics."""

    try:
        if not isinstance(args, Mapping):
            raise PeriodRecordsError("invalid_request", "tool arguments must be an object")
        operation = args.get("operation")
        if operation == "manifest":
            if set(args) != {"operation", "period"}:
                raise PeriodRecordsError("invalid_request", "manifest accepts only operation and period")
            return manifest(str(args.get("period")), source_root=source_root, now=now)
        if operation == "read_batch":
            if set(args) - {"operation", "snapshot_token", "batch_id", "cursor"}:
                raise PeriodRecordsError("invalid_request", "read_batch arguments are invalid")
            return read_batch(
                args.get("snapshot_token"),
                args.get("batch_id"),
                args.get("cursor"),
                source_root=source_root,
            )
        if operation == "reconcile":
            if set(args) != {"operation", "snapshot_token", "coverage"}:
                raise PeriodRecordsError("invalid_request", "reconcile arguments are invalid")
            return reconcile(
                args.get("snapshot_token"),
                args.get("coverage"),
                source_root=source_root,
            )
        raise PeriodRecordsError("invalid_operation", "operation must be manifest, read_batch, or reconcile")
    except PeriodRecordsError as exc:
        return {
            "schema_version": "jitech-kakaowork-period-records-v1",
            "operation": args.get("operation") if isinstance(args, Mapping) else None,
            "status": "unavailable" if exc.code in {
                "package_missing",
                "package_unreadable",
                "membership_invalid",
                "database_corrupt",
                "schema_invalid",
                "message_identity_invalid",
            } else "error",
            "complete": False,
            "error": {"code": exc.code, "message": str(exc)},
            "connection": {
                "status": "unavailable",
                "read_only": True,
                "diagnostic": exc.code,
            },
        }
