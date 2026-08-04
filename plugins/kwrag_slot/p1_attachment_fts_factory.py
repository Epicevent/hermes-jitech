from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "kwrag-p1-attachment-fts-pipeline-v1"
BACKEND_ID = "slot-local-fts5-trigram-or-attachment-v1"
TERM_MAXIMUM = 12
RESULT_MAXIMUM = 10
TEXT_CHARACTER_MAXIMUM = 20_000


class ContractError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def normalize_terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = " ".join(normalized.split())
    cleaned = re.sub(r'["\'()*:^+]+', " ", normalized)
    return list(dict.fromkeys(term for term in cleaned.split() if len(term) >= 3))[
        :TERM_MAXIMUM
    ]


def fts_expression(text: str) -> str:
    terms = normalize_terms(text)
    if not terms:
        return '""'
    return " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)


def build_p1_identity(*, decision: dict[str, Any], contract_sha256: str) -> dict[str, str]:
    factory_sha256 = sha256_bytes(Path(__file__).read_bytes())
    pipeline_spec = {
        "schemaVersion": SCHEMA_VERSION,
        "backendId": BACKEND_ID,
        "termMaximum": TERM_MAXIMUM,
        "resultMaximum": RESULT_MAXIMUM,
        "textCharacterMaximum": TEXT_CHARACTER_MAXIMUM,
        "networkRequired": False,
        "modelRequired": False,
    }
    return {
        "status": "research_selected_p1_attachment_probe_candidate",
        "pipelineFactoryDigest": sha256_bytes(
            canonical_bytes(
                {
                    "implementation": "research/experiments/p1_attachment_fts_v1/factory.py",
                    "factorySourceSha256": factory_sha256,
                    "contractSha256": contract_sha256,
                }
            )
        ),
        "backendId": BACKEND_ID,
        "pipelineFingerprint": sha256_bytes(canonical_bytes(pipeline_spec)),
        "researchDecisionDigest": sha256_bytes(canonical_bytes(decision)),
    }


class FtsAttachmentPipeline:
    """Read-only SQLite FTS5 retrieval; mount authority remains caller-owned."""

    def __init__(self, database_path: Path, binding: dict[str, Any]):
        required = {"databaseSha256", "authority", "sourceSnapshotDigest"}
        if set(binding) != required:
            raise ContractError("binding fields are invalid")
        authority = binding["authority"]
        if not isinstance(authority, dict) or set(authority) != {
            "mode",
            "readOnlyObserved",
            "receiptDigest",
        }:
            raise ContractError("authority fields are invalid")
        if authority["mode"] not in {
            "offline_fixture_read_only",
            "slot_local_read_only_nas",
        }:
            raise ContractError("authority mode is invalid")
        if authority["readOnlyObserved"] is not True:
            raise ContractError("read-only authority was not observed")
        for field in ("databaseSha256", "sourceSnapshotDigest"):
            value = binding[field]
            if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
                raise ContractError(f"{field} is invalid")
        receipt = authority["receiptDigest"]
        if not isinstance(receipt, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", receipt):
            raise ContractError("authority receipt digest is invalid")

        unresolved = Path(database_path)
        if unresolved.is_symlink() or not unresolved.is_file():
            raise ContractError("database must be a regular non-symlink file")
        resolved = unresolved.resolve(strict=True)
        if sha256_bytes(resolved.read_bytes()) != binding["databaseSha256"]:
            raise ContractError("database digest mismatch")
        self._connection = sqlite3.connect(resolved.as_uri() + "?mode=ro&immutable=1", uri=True)
        tables = {
            str(row[0])
            for row in self._connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
            )
        }
        if not {"turns", "turn_mids", "turns_fts"}.issubset(tables):
            self._connection.close()
            raise ContractError("required FTS schema is absent")
        self.binding = json.loads(json.dumps(binding))

    def close(self) -> None:
        self._connection.close()

    def search(self, query: str) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ContractError("query is invalid")
        expression = fts_expression(query)
        if expression == '""':
            return []
        try:
            rows = self._connection.execute(
                "SELECT CAST(f.turn_id AS INTEGER), t.text, bm25(turns_fts) "
                "FROM turns_fts f JOIN turns t ON t.turn_id=CAST(f.turn_id AS INTEGER) "
                "WHERE turns_fts MATCH ? "
                "ORDER BY bm25(turns_fts), CAST(f.turn_id AS INTEGER) LIMIT 1000",
                (expression,),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise ContractError("FTS query failed") from exc

        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        used = 0
        for turn_id, text, score in rows:
            source_ids = tuple(
                sorted(
                    {
                        str(row[0])
                        for row in self._connection.execute(
                            "SELECT mid FROM turn_mids WHERE turn_id=? ORDER BY mid",
                            (int(turn_id),),
                        )
                    }
                )
            )
            if not source_ids or any(source_id in seen for source_id in source_ids):
                continue
            text = text or ""
            if used + len(text) > TEXT_CHARACTER_MAXIMUM:
                continue
            output.append(
                {
                    "sourceIds": list(source_ids),
                    "text": text,
                    "score": -float(score),
                    "unitId": f"turn:{int(turn_id)}",
                }
            )
            used += len(text)
            seen.update(source_ids)
            if len(output) >= RESULT_MAXIMUM or used >= TEXT_CHARACTER_MAXIMUM:
                break
        return output
