"""Pair existing independently solved channel bundles under a frozen split lock.

This command reads UMCD data only. It does not invoke Sionna or run propagation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import jsonschema
from datasets.episode_store import StoredEpisodeBundle, _publish_immutable
from datasets.frequency_pair import (
    audit_paired_frequency_split_binding,
    pair_stored_frequency_bundles,
    write_paired_frequency_manifest,
)
from datasets.split_policy import read_split_policy_registry


def _bundle_ref(episode_id: str, digest: str) -> StoredEpisodeBundle:
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("bundle SHA-256 must be 64 lowercase hexadecimal characters")
    manifest_path = (
        f"manifests/episodes/{hashlib.sha256(episode_id.encode('utf-8')).hexdigest()}.json"
    )
    return StoredEpisodeBundle(episode_id, digest, manifest_path)


def _publish_audit(store_root: Path, audit: dict[str, object]) -> tuple[str, str]:
    encoded = json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    digest = str(audit["audit_sha256"])
    relative = f"manifests/paired_frequency_split_audits/{digest[:2]}/{digest}.json"
    _publish_immutable(store_root / relative, encoded)
    return digest, relative


def pair_stored_frequency_views(
    *,
    store_root: Path,
    split_registry_path: Path,
    pair_id: str,
    source_scene_id: str,
    parent_episode_id: str,
    lower_episode_id: str,
    lower_bundle_sha256: str,
    lower_view_id: str,
    upper_episode_id: str,
    upper_bundle_sha256: str,
    upper_view_id: str,
) -> dict[str, object]:
    """Validate split lineage, then publish a pair manifest and input audit."""
    lower_ref = _bundle_ref(lower_episode_id, lower_bundle_sha256)
    upper_ref = _bundle_ref(upper_episode_id, upper_bundle_sha256)
    registry = read_split_policy_registry(split_registry_path)
    pair = pair_stored_frequency_bundles(
        store_root,
        pair_id=pair_id,
        source_scene_id=source_scene_id,
        parent_episode_id=parent_episode_id,
        lower_bundle=lower_ref,
        upper_bundle=upper_ref,
        lower_view_id=lower_view_id,
        upper_view_id=upper_view_id,
    )
    audit = audit_paired_frequency_split_binding(pair, registry)
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "datasets/schema/paired-frequency-split-binding-audit.v1.schema.json"
    )
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
        audit
    )
    pair_ref = write_paired_frequency_manifest(store_root, pair)
    audit_sha256, audit_path = _publish_audit(store_root, audit)
    return {
        "status": "PASS_INPUT_SCOPED",
        "qualification_status": "NOT_QUALIFIED",
        "rt_generation_executed": False,
        "pair_sha256": pair_ref.pair_sha256,
        "pair_manifest_path": pair_ref.relative_path,
        "audit_sha256": audit_sha256,
        "audit_path": audit_path,
        "split_registry_sha256": str(audit["registry_sha256"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pair two existing independently solved complex UMCD channel bundles and audit "
            "their frozen split binding. No propagation is executed."
        )
    )
    parser.add_argument("--store", type=Path, required=True, help="UMCD episode store root")
    parser.add_argument("--split-registry", type=Path, required=True, help="frozen split lock")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--source-scene-id", required=True)
    parser.add_argument("--parent-episode-id", required=True)
    for side in ("lower", "upper"):
        parser.add_argument(f"--{side}-episode-id", required=True)
        parser.add_argument(f"--{side}-bundle-sha256", required=True)
        parser.add_argument(f"--{side}-view-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = pair_stored_frequency_views(
        store_root=args.store,
        split_registry_path=args.split_registry,
        pair_id=args.pair_id,
        source_scene_id=args.source_scene_id,
        parent_episode_id=args.parent_episode_id,
        lower_episode_id=args.lower_episode_id,
        lower_bundle_sha256=args.lower_bundle_sha256,
        lower_view_id=args.lower_view_id,
        upper_episode_id=args.upper_episode_id,
        upper_bundle_sha256=args.upper_bundle_sha256,
        upper_view_id=args.upper_view_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
