import copy
import json
from pathlib import Path

import pytest

from agent.provider_usage_coverage import (
    canonical_manifest_bytes,
    manifest_digest,
    provider_usage_coverage_manifest,
    validate_provider_usage_coverage,
)


FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "jitech-provider-usage-coverage-v1.json"
)


def test_checked_in_coverage_fixture_matches_exact_wire_contract():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    validate_provider_usage_coverage(fixture)
    assert fixture["manifestDigest"] == manifest_digest(fixture)


def test_product_manifest_is_sorted_unique_and_explicitly_partial():
    manifest = provider_usage_coverage_manifest()
    codes = [surface["surfaceCode"] for surface in manifest["surfaces"]]

    assert manifest["schema"] == "jitech-provider-usage-coverage/v1"
    assert manifest["productFamily"] == "hermes"
    assert manifest["coverageStatus"] == "partial"
    assert manifest["manifestDigest"] == (
        "sha256:aa1ca9904398cd80d05e871217ec5508ceede79c43fda2928d3f75b75dcca3de"
    )
    assert len(codes) == 50
    assert codes == sorted(codes)
    assert len(codes) == len(set(codes))
    assert any(surface["status"] == "partial" for surface in manifest["surfaces"])
    assert all(
        surface["observationKind"] != "turn_aggregate"
        for surface in manifest["surfaces"]
    )


def test_digest_excludes_only_the_digest_field():
    manifest = provider_usage_coverage_manifest()
    changed_digest = copy.deepcopy(manifest)
    changed_digest["manifestDigest"] = "ignored"
    assert canonical_manifest_bytes(changed_digest) == canonical_manifest_bytes(
        manifest
    )

    changed_surface = copy.deepcopy(manifest)
    changed_surface["surfaces"][0]["meterFamily"] = "other"
    assert manifest_digest(changed_surface) != manifest["manifestDigest"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update(extra=True), "top-level fields mismatch"),
        (
            lambda value: value.update(coverageStatus="complete"),
            "coverageStatus must be partial",
        ),
        (
            lambda value: value["surfaces"].reverse(),
            "unique and sorted",
        ),
        (
            lambda value: value["surfaces"][0].update(gapCode="UNEXPECTED"),
            "implemented surface cannot have gapCode",
        ),
    ],
)
def test_validator_rejects_contract_drift(mutation, match):
    manifest = provider_usage_coverage_manifest()
    mutation(manifest)
    manifest["manifestDigest"] = manifest_digest(manifest)
    with pytest.raises(ValueError, match=match):
        validate_provider_usage_coverage(manifest)
