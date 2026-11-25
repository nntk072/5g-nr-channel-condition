"""Materialize a mathematical direct-path ablation from an existing UMCD bundle.

The command reuses a stored path table and writes a derived bundle to a distinct
store. It does not retrace scene geometry or model a physical blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import jsonschema
from datasets.counterfactual import remove_direct_path_component
from datasets.episode_store import (
    StoredEpisodeBundle,
    _publish_immutable,
    read_episode_bundle,
    write_episode_bundle,
)
from datasets.split_policy import read_split_policy_registry


def _stored_bundle_ref(episode_id: str, digest: str) -> StoredEpisodeBundle:
    if not episode_id or episode_id != episode_id.strip():
        raise ValueError("episode_id must be a non-empty canonical string")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("bundle SHA-256 must be 64 lowercase hexadecimal characters")
    manifest_path = (
        f"manifests/episodes/{hashlib.sha256(episode_id.encode('utf-8')).hexdigest()}.json"
    )
    return StoredEpisodeBundle(episode_id, digest, manifest_path)


def _require_separate_stores(source: Path, output: Path) -> tuple[Path, Path]:
    source_root = source.expanduser().resolve()
    output_root = output.expanduser().resolve()
    if (
        source_root == output_root
        or source_root.is_relative_to(output_root)
        or output_root.is_relative_to(source_root)
    ):
        raise ValueError("source and output stores must be separate, non-overlapping directories")
    return source_root, output_root


def materialize_direct_path_ablation(
    *,
    source_store: Path,
    output_store: Path,
    split_registry_path: Path,
    episode_id: str,
    bundle_sha256: str,
    parent_subject_id: str,
    base_child_id: str,
    ablated_child_id: str,
) -> dict[str, object]:
    """Create and replay-verify one content-addressed mathematical ablation."""
    source_root, output_root = _require_separate_stores(source_store, output_store)
    source_ref = _stored_bundle_ref(episode_id, bundle_sha256)
    source_bundle = read_episode_bundle(source_root, source_ref)
    if source_bundle.path_oracle is None:
        raise ValueError("source episode bundle has no path-oracle layer")
    if parent_subject_id != source_bundle.episode.episode_id:
        raise ValueError("split parent subject ID must equal the source episode ID")
    registry = read_split_policy_registry(split_registry_path)
    ablation = remove_direct_path_component(
        source_bundle.path_oracle,
        frequency_offset_hz=source_bundle.episode.subcarrier_offsets_hz,
        carrier_hz=source_bundle.episode.carrier_hz,
        registry=registry,
        parent_subject_id=parent_subject_id,
        base_child_id=base_child_id,
        ablated_child_id=ablated_child_id,
    )
    derived_bundle = replace(
        source_bundle,
        direct_path_ablations=(*source_bundle.direct_path_ablations, ablation),
    )
    derived_ref = write_episode_bundle(output_root, derived_bundle)
    replayed = read_episode_bundle(output_root, derived_ref)
    if len(replayed.direct_path_ablations) != len(derived_bundle.direct_path_ablations):
        raise ValueError("derived bundle replay lost a direct-path ablation layer")
    core: dict[str, object] = {
        "schema": "umcd.direct_path_ablation_materialization_report.v1",
        "status": "PASS_SOFTWARE_REPLAY",
        "qualification_status": "NOT_QUALIFIED",
        "physical_blocker_modeled": False,
        "rt_generation_executed": False,
        "source_bundle_sha256": source_ref.bundle_sha256,
        "derived_bundle_sha256": derived_ref.bundle_sha256,
        "derived_manifest_path": derived_ref.relative_path,
        "path_table_sha256": ablation.path_table_sha256,
        "split_registry_sha256": str(registry.to_dict()["registry_sha256"]),
        "parent_subject_id": parent_subject_id,
        "base_child_id": base_child_id,
        "ablated_child_id": ablated_child_id,
        "split": ablation.split,
    }
    report_sha256 = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    report = {**core, "report_sha256": report_sha256}
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "datasets/schema/direct-path-ablation-materialization-report.v1.schema.json"
    )
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(
        report
    )
    report_path = f"manifests/direct_path_ablation_reports/{report_sha256[:2]}/{report_sha256}.json"
    report_payload = json.dumps(
        report, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    _publish_immutable(output_root / report_path, report_payload)
    return {**report, "report_path": report_path}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a same-path mathematical direct-component ablation from an existing "
            "UMCD bundle into a separate store. No scene retracing or physical blocker model "
            "is used."
        )
    )
    parser.add_argument("--source-store", type=Path, required=True)
    parser.add_argument("--output-store", type=Path, required=True)
    parser.add_argument("--split-registry", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--parent-subject-id", required=True)
    parser.add_argument("--base-child-id", required=True)
    parser.add_argument("--ablated-child-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = materialize_direct_path_ablation(
        source_store=args.source_store,
        output_store=args.output_store,
        split_registry_path=args.split_registry,
        episode_id=args.episode_id,
        bundle_sha256=args.bundle_sha256,
        parent_subject_id=args.parent_subject_id,
        base_child_id=args.base_child_id,
        ablated_child_id=args.ablated_child_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
