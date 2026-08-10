"""Image-level provenance for the product-native KWRAG caller."""

from __future__ import annotations

import json
import subprocess


COMPONENT_DIGEST = (
    "sha256:40c8516fb15a617d1e2082798782442b2ba9ba855146d688a546e36db6dd5739"
)
MANIFEST_DIGEST = (
    "sha256:10fd5fd1ca835f89d32bf3a3b7c407f11d022293cfbe32b54a1871227e0f2786"
)
CONTRACT_DIGEST = (
    "sha256:ccf826f0fe6f7edc36b6d5eacdee87277859d2f6dae3a4ea4cab5f51cba183db"
)
SOURCE_ARCHIVE_DIGEST = (
    "sha256:19c2fc2193ebb5633129f98d972e477471330cb79fa5189bb7eb777123bec5ea"
)
SOURCE_REVISION = "83afc8fa97b2191665afa152e61feae1f5e87145"
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


def test_image_binds_product_component_without_ops_runtime_contract(built_image) -> None:
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
    assert not any(key.startswith("com.epicevent.agent-runtime.retrieval.") for key in labels)
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
            "from kwrag.product_runtime import open_kakao_product_runtime; print('ok')",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert component.returncode == 0, component.stderr
    assert component.stdout.strip() == "ok"
