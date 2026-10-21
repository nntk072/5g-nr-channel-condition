"""Shared run-manifest envelope and completion marker helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .identity import content_identity


def run_manifest(run_id: str, config: dict[str, Any], **fields: Any) -> dict[str, Any]:
    """Build a versioned run manifest around domain-specific evidence fields."""
    return {
        "schema": "loslab.run.v2",
        "run_id": run_id,
        "state": "running",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_identity": content_identity(config),
        "config": config,
        **fields,
    }


def mark_complete(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a completed manifest without mutating the caller's mapping."""
    return {
        **manifest,
        "state": "complete",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
