"""Canonical JSON and runtime identity helpers for joint-evidence run manifests."""

from __future__ import annotations

import json
import platform
from typing import Any


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _runtime_identity() -> dict[str, Any]:
    """Capture a self-reported environment identity without implying attestation."""
    from artifacts.provenance import versions

    observed = versions()
    package_names = (
        "numpy",
        "scikit-learn",
        "torch",
        "sionna",
        "sionna-rt",
        "sionna-phy",
    )
    return {
        "schema": "loslab.g3_execution_environment.v1",
        "python": observed["python"],
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "packages": {name: observed[name] for name in package_names},
        "source_revision": observed["git_sha"],
    }
