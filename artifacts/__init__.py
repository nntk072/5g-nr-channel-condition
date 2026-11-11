"""Shared identity, provenance, and safe artifact-writing primitives."""

from .identity import content_identity
from .layout import RunLayout
from .manifest import mark_complete, mark_failed, run_manifest
from .provenance import atomic_json, canonical_hash, package_source_sha256, sha256_file, versions
from .run_replay import (
    build_run_replay_provenance,
    transition_run_replay,
    validate_run_replay_provenance,
)

__all__ = [
    "RunLayout",
    "atomic_json",
    "canonical_hash",
    "content_identity",
    "mark_complete",
    "mark_failed",
    "package_source_sha256",
    "build_run_replay_provenance",
    "run_manifest",
    "sha256_file",
    "transition_run_replay",
    "validate_run_replay_provenance",
    "versions",
]
