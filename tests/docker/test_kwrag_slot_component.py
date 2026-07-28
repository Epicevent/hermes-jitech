"""Image-level contract for the default-off embedded KWRAG component."""

from __future__ import annotations

import json
import subprocess


COMPONENT_DIGEST = "sha256:7f6e4ace39c8d868e0517040be0a82742b791dd44744afdae66d54e596b25478"
MANIFEST_DIGEST = "sha256:c1e0e8ed1462db8663d8063e4e97ba4530c4f1a7bf3f24a514807eb56c19baf6"
CONTRACT_DIGEST = "sha256:ccf826f0fe6f7edc36b6d5eacdee87277859d2f6dae3a4ea4cab5f51cba183db"
SOURCE_ARCHIVE_DIGEST = "sha256:6c04a7d297410708a0300b3ab3193e047c950c924bc7edc6d4ae7ae127efb97a"
SOURCE_REVISION = "832684981b5203911d299b65d32e01816da06cf3"
RESOURCE_PROFILE_DIGEST = "sha256:2d4ff46a2d76e712421a9758ecb0ae1d262e2d42ea00cee888c103477e6709ed"
BINDING_DIGEST = "sha256:" + "d" * 64
LABEL_PREFIX = "com.epicevent.agent-runtime.retrieval."


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


def test_image_binds_embedded_component_and_default_off_status(built_image, tmp_path) -> None:
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

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--read-only",
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
            built_image,
            "hermes",
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
