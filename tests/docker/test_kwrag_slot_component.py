"""Image-level provenance for the product-native KWRAG caller."""

from __future__ import annotations

import json
import subprocess


COMPONENT_DIGEST = (
    "sha256:8b97e30d2ac6e7ed44c70fdd2a8075406e01b68be9c9068262987d769c1b98ed"
)
MANIFEST_DIGEST = (
    "sha256:ab6fd5b5e5d03bf601f0edcfcb1967dc41ad1ee184982cb1c7412994d6682643"
)
CONTRACT_DIGEST = (
    "sha256:ccf826f0fe6f7edc36b6d5eacdee87277859d2f6dae3a4ea4cab5f51cba183db"
)
SOURCE_ARCHIVE_DIGEST = (
    "sha256:d2c178439144bafe5edef9b385fc7c6c0c7070a140732eb18ef41862de6c00eb"
)
SOURCE_REVISION = "5f3a35e462ca2ef155b67f77a799b11e5f08b39a"
LABEL_PREFIX = "com.epicevent.hermes.kwrag."


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


def test_image_includes_groupware_pdf_extractor(built_image) -> None:
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/usr/bin/test",
            built_image,
            "-x",
            "/usr/bin/pdftotext",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_image_binds_product_component_without_ops_runtime_contract(
    built_image,
) -> None:
    labels = _inspect_labels(built_image)
    expected = {
        "schema": "hermes-kwrag-product-component/v1",
        "component-digest": COMPONENT_DIGEST,
        "component-manifest-digest": MANIFEST_DIGEST,
        "contract-digest": CONTRACT_DIGEST,
        "source-archive-digest": SOURCE_ARCHIVE_DIGEST,
        "source-revision": SOURCE_REVISION,
        "transport": "in_process",
        "default-enabled": "false",
        "host-port-count": "0",
        "verify-command.json": '["hermes","kwrag-slot","status","--json"]',
    }
    for suffix, value in expected.items():
        assert labels[f"{LABEL_PREFIX}{suffix}"] == value
    assert not any(
        key.startswith("com.epicevent.agent-runtime.retrieval.") for key in labels
    )
    assert not any(key.startswith("com.epicevent.hermes.kwrag.p1.") for key in labels)

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
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
    assert json.loads(result.stdout) == {
        "componentDigest": COMPONENT_DIGEST,
        "componentManifestDigest": MANIFEST_DIGEST,
        "componentSourceRevision": SOURCE_REVISION,
        "defaultEnabled": False,
        "hostPortCount": 0,
        "schema": "hermes-kwrag-product-component-status/v1",
        "transport": "in_process",
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
            "from kwrag.product_runtime import open_product_runtime; print('ok')",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert component.returncode == 0, component.stderr
    assert component.stdout.strip() == "ok"
