from __future__ import annotations

from pathlib import Path


SKILL = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "jitech-weekly-kakaowork-summary"
    / "SKILL.md"
)


def test_skill_requires_manifest_all_pages_and_reconcile() -> None:
    text = SKILL.read_text(encoding="utf-8")

    assert "`manifest`" in text
    assert "`read_batch`" in text
    assert "`reconcile`" in text
    assert "next_cursor" in text
    assert "batch_coverage_digest" in text
    assert "complete=true" in text
    assert "stable_message_id" in text


def test_skill_never_promotes_incomplete_or_stale_results() -> None:
    text = SKILL.read_text(encoding="utf-8")

    assert 'freshness가 `stale`' in text
    assert '"전건 요약 완료"' in text
    assert "complete=false" in text
    assert "processed_messages + failed_messages + uncovered_messages" in text


def test_skill_description_meets_bundled_skill_limit() -> None:
    description = next(
        line.removeprefix("description: ")
        for line in SKILL.read_text(encoding="utf-8").splitlines()
        if line.startswith("description: ")
    )
    assert len(description) <= 60
    assert description.endswith(".")
