"""Shared run-manifest envelope and completion marker helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .identity import content_identity
from .run_replay import transition_run_replay


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
    completed_utc = datetime.now(timezone.utc).isoformat()
    result = {
        **manifest,
        "state": "complete",
        "completed_utc": completed_utc,
    }
    if "run_replay" in manifest:
        result["run_replay"] = transition_run_replay(
            manifest["run_replay"], state="complete", changed_utc=completed_utc
        )
    return result


def mark_failed(
    manifest: dict[str, Any], *, error_type: str, error_message: str, stage: str
) -> dict[str, Any]:
    """Return a failed manifest and bind terminal failure details into replay provenance."""
    failed_utc = datetime.now(timezone.utc).isoformat()
    result = {
        **manifest,
        "state": "failed",
        "failed_utc": failed_utc,
        "failure": {
            "error_type": error_type,
            "message": error_message,
            "stage": stage,
        },
    }
    if "run_replay" in manifest:
        result["run_replay"] = transition_run_replay(
            manifest["run_replay"],
            state="failed",
            changed_utc=failed_utc,
            failure={
                "error_type": error_type,
                "message": error_message,
                "stage": stage,
            },
        )
    return result
