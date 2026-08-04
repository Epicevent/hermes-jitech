"""Image-level contract for the default-off embedded KWRAG component."""

from __future__ import annotations

import json
import subprocess


COMPONENT_DIGEST = (
    "sha256:f8dd900d0d00775853ee95dfbf15960c9ea7de2711ea5635fe229b06a550fa6f"
)
MANIFEST_DIGEST = (
    "sha256:6578b61f91151d6cfa2d6a100397a409a6293c1396ce957cd7e87cf0da74e811"
)
CONTRACT_DIGEST = (
    "sha256:ccf826f0fe6f7edc36b6d5eacdee87277859d2f6dae3a4ea4cab5f51cba183db"
)
SOURCE_ARCHIVE_DIGEST = (
    "sha256:50962d7199515839ce47a454e68b0280c0e0999a9d6a63a9bbcb14aff591d2b9"
)
SOURCE_REVISION = "49c10212ff12433941cfbe43d95013d1d2f0aebe"
RESOURCE_PROFILE_DIGEST = (
    "sha256:2d4ff46a2d76e712421a9758ecb0ae1d262e2d42ea00cee888c103477e6709ed"
)
BINDING_DIGEST = "sha256:" + "d" * 64
LABEL_PREFIX = "com.epicevent.agent-runtime.retrieval."
P1_LABEL_PREFIX = "com.epicevent.hermes.kwrag.p1."
P1_COMPONENT_DIGEST = (
    "sha256:f8c90245dabfce1edf840ef308f1d0969233e6adfa383a499ecf9632dea8284d"
)
P1_COMPONENT_MANIFEST_DIGEST = (
    "sha256:9df5ac053f40265bb864aa38d8dd00b0b1f05a32841e245a45d0f52cf8697be2"
)


def _inspect_labels(image: str) -> dict[str, str]:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{json .Config.Labels}}"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    labels = json.loads(result.stdout)
    assert isinstance(labels, dict)
    return labels


def test_image_binds_embedded_component_and_default_off_status(
    built_image, tmp_path
) -> None:
    labels = _inspect_labels(built_image)
    expected = {
        "schema": "jitech-embedded-retrieval/v1",
        "component-digest": COMPONENT_DIGEST,
        "component-manifest-digest": MANIFEST_DIGEST,
        "contract-digest": CONTRACT_DIGEST,
        "source-archive-digest": SOURCE_ARCHIVE_DIGEST,
        "source-revision": SOURCE_REVISION,
        "transport": "in_process",
        "default-enabled": "false",
        "host-port-count": "0",
        "nas-read-only": "true",
        "verify-command.json": '["hermes","kwrag-slot","status","--json"]',
    }
    for suffix, value in expected.items():
        assert labels[f"{LABEL_PREFIX}{suffix}"] == value
    assert labels[f"{P1_LABEL_PREFIX}default-enabled"] == "false"
    assert labels[f"{P1_LABEL_PREFIX}caller-explicit"] == "true"
    assert labels[f"{P1_LABEL_PREFIX}component-wheel-digest"] == P1_COMPONENT_DIGEST
    assert (
        labels[f"{P1_LABEL_PREFIX}component-manifest-digest"]
        == P1_COMPONENT_MANIFEST_DIGEST
    )
    assert (
        labels[f"{P1_LABEL_PREFIX}status-schema"]
        == "jitech-embedded-retrieval-attachment-status/v1"
    )
    assert (
        labels[f"{P1_LABEL_PREFIX}verify-command.json"]
        == '["hermes","kwrag-slot","p1-attachment-status","--json"]'
    )

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
            "--tmpfs",
            "/run:rw,exec,nosuid,nodev,size=64m,mode=755",
            "--mount",
            f"type=bind,src={tmp_path},dst=/workspace/nas_docs,readonly",
            "-e",
            "HERMES_WORKSPACE_DIR=/workspace",
            "-e",
            "JITECH_RETRIEVAL_ENABLED=false",
            "-e",
            f"JITECH_RETRIEVAL_COMPONENT_DIGEST={COMPONENT_DIGEST}",
            "-e",
            f"JITECH_RETRIEVAL_BINDING_DIGEST={BINDING_DIGEST}",
            "-e",
            f"JITECH_RETRIEVAL_RESOURCE_PROFILE_DIGEST={RESOURCE_PROFILE_DIGEST}",
            "--entrypoint",
            "hermes",
            built_image,
            "kwrag-slot",
            "status",
            "--json",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status == {
        "bindingDigest": BINDING_DIGEST,
        "componentDigest": COMPONENT_DIGEST,
        "consumerHealth": "disabled",
        "consumptionReceiptDigest": None,
        "gpuAccessStatus": "none",
        "hostPortCount": 0,
        "linkageStatus": "not_applicable",
        "mountReadOnly": True,
        "operationReceiptDigest": None,
        "resourceProfileDigest": RESOURCE_PROFILE_DIGEST,
        "resourceStatus": "unavailable",
        "resultReceiptDigest": None,
        "revocationStatus": "complete",
        "schema": "jitech-embedded-retrieval-status/v1",
    }

    component = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python",
            built_image,
            "-c",
            "import kwrag_p1_attachment; print(kwrag_p1_attachment.TEXT_CHARACTER_MAXIMUM)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert component.returncode == 0, component.stderr
    assert component.stdout.strip() == "20000"
