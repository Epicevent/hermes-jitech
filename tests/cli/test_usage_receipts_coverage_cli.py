import json
import os
import subprocess
import sys
from pathlib import Path


def test_coverage_json_is_read_only_and_exact(tmp_path):
    hermes_home = tmp_path / "must-not-be-created"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "usage-receipts",
            "coverage",
            "--json",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    manifest = json.loads(result.stdout)

    assert manifest["schema"] == "jitech-provider-usage-coverage/v1"
    assert manifest["coverageStatus"] == "partial"
    assert manifest["manifestDigest"].startswith("sha256:")
    assert not hermes_home.exists()


def test_export_cli_fast_path_is_readonly_and_emits_exact_page(tmp_path):
    hermes_home = tmp_path / "hermes-home"
    db_path = hermes_home / "state.db"
    from hermes_state import SessionDB

    db = SessionDB(db_path)
    try:
        db.record_provider_call(
            None,
            call_id="00000000-0000-4000-8000-000000000707",
            request_id=None,
            api_call_index=1,
            attempt=1,
            fallback_index=0,
            configured_provider="openai",
            configured_model="gpt-5.2",
            requested_provider="openai",
            requested_model="gpt-5.2",
            actual_provider=None,
            actual_model=None,
            response_id=None,
            evidence_source=None,
            finish_reason=None,
            usage=None,
            started_at=1.0,
            completed_at=2.0,
            trigger="manual",
        )
    finally:
        db.close()

    before = {path.name for path in hermes_home.iterdir()}
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "hermes_cli.main",
            "usage-receipts",
            "export",
            "--after",
            "0",
            "--limit",
            "500",
        ],
        cwd=Path(__file__).parents[2],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    page = json.loads(result.stdout)
    after = {path.name for path in hermes_home.iterdir()}

    assert page["count"] == 1
    assert len(page["coverageManifests"]) == 1
    assert after - before <= {"state.db-shm", "state.db-wal"}
    assert "logs" not in after
