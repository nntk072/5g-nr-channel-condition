"""Audit a frozen UMCD episode-bundle inventory without claiming qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from datasets.episode_store import (
    EpisodeBundle,
    read_episode_bundle,
)
from datasets.split_policy import (
    LeakageNamespace,
    LeakageRelation,
    SplitChildBinding,
    audit_split_policy_registry,
    network_episode_subject,
    read_split_policy_registry,
)
from evaluation.near_duplicate_audit import ChannelRecord, audit_near_duplicates
from evaluation.target_coverage import (
    ExpectedInventoryCell,
    TargetRecord,
    audit_target_coverage,
    convert_target_map,
)

_SCHEMA = "umcd.corpus_audit_inventory.v1"
_REPORT_SCHEMA = "umcd.corpus_audit_report.v1"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_report(path: Path, encoded: bytes) -> None:
    """Publish a report once, allowing only byte-identical replays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            metadata = os.lstat(path)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("existing report must be a single-link regular file")
            if path.read_bytes() != encoded:
                raise FileExistsError(f"refusing to overwrite different audit report: {path}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("inventory must be a JSON object")
    return payload


def _resolve(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise ValueError(f"{name} must be relative to the inventory file")
    resolved = (base / candidate).resolve(strict=True)
    if not resolved.is_relative_to(base.resolve()):
        raise ValueError(f"{name} escapes the inventory directory")
    return resolved


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("bundles must be a nonempty array")
    rows = [_mapping(item, "bundle inventory row") for item in value]
    ids = [_nonempty(row.get("episode_id"), "bundle.episode_id") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("bundle episode_id values must be unique")
    return rows


def _relations(value: Any, episode_id: str) -> tuple[LeakageRelation, ...]:
    if not isinstance(value, list):
        raise ValueError(f"relations for {episode_id!r} must be an array")
    result: list[LeakageRelation] = []
    for item in value:
        relation = _mapping(item, "leakage relation")
        result.append(
            LeakageRelation(
                LeakageNamespace(_nonempty(relation.get("namespace"), "relation.namespace")),
                _nonempty(relation.get("identity"), "relation.identity"),
            )
        )
    return tuple(result)


def _target_map(bundle: EpisodeBundle) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for target in bundle.targets:
        target_id = f"{target.entity_id}::{target.definition.semantic}"
        if target_id in result:
            raise ValueError(f"duplicate derived target ID {target_id!r}")
        result[target_id] = target
    return result


def _bundle_children(bundle: EpisodeBundle) -> tuple[SplitChildBinding, ...]:
    return tuple(
        SplitChildBinding(
            child_id=binding.observation.sample_id,
            parent_subject_id=bundle.episode.episode_id,
            child_kind="receiver_observation",
        )
        for binding in bundle.receiver_observations
    )


def _bundle_geometry_hash(bundle: EpisodeBundle) -> str | None:
    scene = bundle.episode
    identity = {
        "arrays": [
            {
                "array_id": array.array_id,
                "positions": array.element_positions_local_m.tolist(),
                "orientation": array.orientation_local_to_world.tolist(),
                "polarization": list(array.polarization_per_element),
                "ports": list(array.port_ids),
                "port_weights": [
                    [[float(value.real), float(value.imag)] for value in row]
                    for row in array.port_element_weights
                ],
            }
            for array in scene.arrays
        ],
        "nodes": [
            {
                "node_id": node.node_id,
                "array_id": node.array_id,
                "positions": [
                    scene.position_at(node.node_id, index)
                    for index in range(len(scene.frame_times_s))
                ],
                "orientations": [
                    scene.orientation_at(node.node_id, index).tolist()
                    for index in range(len(scene.frame_times_s))
                ],
            }
            for node in scene.nodes
        ],
        "links": [
            {
                "link_id": link.link_id,
                "tx": link.tx_node_id,
                "rx": link.rx_node_id,
                "tx_ports": list(link.tx_port_indices),
                "rx_ports": list(link.rx_port_indices),
            }
            for link in scene.links
        ],
    }
    return hashlib.sha256(_canonical_json(identity)).hexdigest()


def audit_inventory(inventory_path: Path, *, max_pair_comparisons: int = 100_000) -> dict[str, Any]:
    """Run manifest, bundle, split, target-coverage, and duplicate checks."""
    if isinstance(max_pair_comparisons, bool) or not isinstance(max_pair_comparisons, int):
        raise TypeError("max_pair_comparisons must be a positive integer")
    if max_pair_comparisons <= 0:
        raise ValueError("max_pair_comparisons must be a positive integer")
    inventory_path = inventory_path.resolve(strict=True)
    base = inventory_path.parent
    inventory = _load_json(inventory_path)
    if inventory.get("schema") != _SCHEMA or inventory.get("frozen") is not True:
        raise ValueError(f"inventory must be frozen and use schema {_SCHEMA}")
    inventory_id = _nonempty(inventory.get("inventory_id"), "inventory_id")

    legal = _mapping(inventory.get("legal_assessment"), "legal_assessment")
    legal_status = _nonempty(legal.get("status"), "legal_assessment.status")
    if legal_status not in {"pending", "approved", "rejected", "not_applicable"}:
        raise ValueError(
            "legal_assessment.status must be pending, approved, rejected, or not_applicable"
        )
    attestation = _mapping(inventory.get("external_lock_attestation"), "external_lock_attestation")
    attestation_status = _nonempty(attestation.get("status"), "external_lock_attestation.status")
    if attestation_status not in {"missing", "provided", "attested", "rejected"}:
        raise ValueError(
            "external_lock_attestation.status must be missing, provided, attested, or rejected"
        )

    store = _resolve(base, inventory.get("store"), "store")
    registry_config = _mapping(inventory.get("split_registry"), "split_registry")
    registry_path = _resolve(base, registry_config.get("path"), "split_registry.path")
    registry_hash = _nonempty(registry_config.get("sha256"), "split_registry.sha256")
    if _sha256(registry_path) != registry_hash:
        raise ValueError("split registry file hash does not match frozen inventory")
    registry = read_split_policy_registry(registry_path)

    rows = _records(inventory.get("bundles"))
    expected_manifests: dict[str, tuple[str, str]] = {}
    bundles: dict[str, EpisodeBundle] = {}
    subjects = []
    observed_children: list[SplitChildBinding] = []
    for row in rows:
        episode_id = _nonempty(row.get("episode_id"), "bundle.episode_id")
        relative = _nonempty(row.get("manifest"), f"manifest for {episode_id}")
        expected_hash = _nonempty(row.get("manifest_sha256"), f"manifest hash for {episode_id}")
        manifest_path = _resolve(store, relative, f"manifest for {episode_id}")
        if _sha256(manifest_path) != expected_hash:
            raise ValueError(f"bundle manifest hash mismatch for {episode_id!r}")
        bundle = read_episode_bundle(store, manifest_path)
        if bundle.episode.episode_id != episode_id:
            raise ValueError(f"manifest and inventory episode IDs differ for {episode_id!r}")
        bundles[episode_id] = bundle
        expected_manifests[episode_id] = (relative, expected_hash)
        subjects.append(
            network_episode_subject(
                bundle.episode,
                relations=_relations(row.get("relations", []), episode_id),
            )
        )
        observed_children.extend(_bundle_children(bundle))

    manifest_dir = store / "manifests" / "episodes"
    discovered = (
        sorted(path.name for path in manifest_dir.glob("*.json")) if manifest_dir.exists() else []
    )
    declared = sorted(Path(value[0]).name for value in expected_manifests.values())
    if discovered != declared:
        raise ValueError("frozen bundle inventory must cover exactly the episode manifest set")
    if len({item.child_id for item in observed_children}) != len(observed_children):
        raise ValueError("receiver child IDs must be unique across bundles")
    audit_split_policy_registry(registry, subjects, observed_children)

    target_records: list[TargetRecord] = []
    all_definitions: dict[str, Any] = {}
    duplicate_groups: dict[tuple[str, tuple[int, ...]], list[ChannelRecord]] = defaultdict(list)
    for episode_id in sorted(bundles):
        bundle = bundles[episode_id]
        split_id = registry.split_for_parent(episode_id)
        definitions, observations = convert_target_map(_target_map(bundle))
        for definition in definitions:
            previous = all_definitions.setdefault(definition.target_id, definition)
            if previous != definition:
                raise ValueError(f"target definition or shape changes for {definition.target_id!r}")
        if observations:
            target_records.append(
                TargetRecord(
                    record_id=episode_id,
                    split_id=split_id,
                    family_id=bundle.episode.family_id,
                    targets=observations,
                )
            )
        geometry_hash = _bundle_geometry_hash(bundle)
        for view in bundle.channel_views:
            duplicate_groups[(view.view_id, tuple(view.channel.shape))].append(
                ChannelRecord(
                    sample_id=f"{episode_id}::{view.view_id}",
                    split_id=split_id,
                    family_site_geometry_hash=geometry_hash,
                    cfr=view.channel,
                )
            )

    expected_coverage = inventory.get("expected_coverage")
    if not isinstance(expected_coverage, list) or not expected_coverage:
        raise ValueError("expected_coverage must declare nonempty split/family/target cells")
    coverage = audit_target_coverage(
        tuple(target_records),
        tuple(all_definitions.values()),
        tuple(ExpectedInventoryCell(**item) for item in expected_coverage),
    )

    threshold = inventory.get("duplicate_threshold", 0.999)
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)):
        raise ValueError("duplicate_threshold must be numeric")
    pair_count = 0
    comparison_count = 0
    for group in duplicate_groups.values():
        counts: dict[str, int] = defaultdict(int)
        for item in group:
            counts[item.split_id] += 1
        pair_count += (len(group) ** 2 - sum(count**2 for count in counts.values())) // 2
        comparison_count += len(group) * (len(group) - 1) // 2
    if comparison_count > max_pair_comparisons:
        raise ValueError("duplicate pair comparison budget exceeded")
    duplicate_reports = [
        audit_near_duplicates(tuple(group), normalized_inner_product_threshold=threshold)
        for _, group in sorted(duplicate_groups.items())
        if len(group) > 1
    ]
    candidates = [item for report in duplicate_reports for item in report.candidates]
    skipped = [item for report in duplicate_reports for item in report.skipped_comparisons]

    report_core = {
        "schema": _REPORT_SCHEMA,
        "inventory_id": inventory_id,
        "inventory_sha256": _sha256(inventory_path),
        "bundle_manifest_sha256": {
            episode_id: expected_manifests[episode_id][1] for episode_id in sorted(bundles)
        },
        "split_registry_file_sha256": registry_hash,
        "checks": {
            "bundle_store_integrity": {"status": "passed", "bundle_count": len(bundles)},
            "split_lineage": {
                "status": "passed",
                "registry_sha256": registry.to_dict()["registry_sha256"],
                "parent_count": len(subjects),
                "child_count": len(observed_children),
                "split_counts": dict(sorted(registry.independent_component_counts.items())),
            },
            "target_coverage": {
                "status": "passed",
                "cells": [asdict(cell) for cell in coverage.cells],
            },
            "near_duplicate_review": {
                "status": (
                    "review_required"
                    if candidates
                    else "no_candidates_detected"
                    if pair_count
                    else "not_run_no_cross_split_pairs"
                ),
                "threshold": float(threshold),
                "comparable_group_count": len(duplicate_reports),
                "cross_split_pair_count": pair_count,
                "total_pair_comparison_count": comparison_count,
                "candidate_count": len(candidates),
                "candidates": [asdict(item) for item in candidates],
                "skipped_comparison_count": len(skipped),
            },
        },
        "external_gates": {
            "legal_assessment": {
                "declared_status": legal_status,
                "status": "external_unverified",
                "evidence_sha256": legal.get("evidence_sha256"),
            },
            "external_lock_attestation": {
                "declared_status": attestation_status,
                "status": "external_unverified",
                "evidence_sha256": attestation.get("evidence_sha256"),
            },
        },
        "qualification": "not_claimed",
        "fixture_notice": "Passing software checks do not qualify a corpus; fixtures remain fixtures.",
    }
    return {
        **report_core,
        "report_sha256": hashlib.sha256(_canonical_json(report_core)).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path, help=f"frozen {_SCHEMA} JSON")
    parser.add_argument("--output", type=Path, required=True, help="report JSON output path")
    parser.add_argument("--max-pair-comparisons", type=int, default=100_000)
    args = parser.parse_args(argv)
    try:
        report = audit_inventory(args.inventory, max_pair_comparisons=args.max_pair_comparisons)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"corpus audit failed: {exc}", file=sys.stderr)
        try:
            inventory_path = args.inventory.resolve(strict=True)
            failed_core = {
                "schema": _REPORT_SCHEMA,
                "inventory_sha256": _sha256(inventory_path),
                "execution": {"status": "failed", "error": str(exc)},
                "qualification": "not_claimed",
            }
            report = {
                **failed_core,
                "report_sha256": hashlib.sha256(_canonical_json(failed_core)).hexdigest(),
            }
            _publish_report(
                args.output,
                (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                    "utf-8"
                ),
            )
        except (OSError, ValueError):
            pass
        return 2
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        _publish_report(args.output, encoded.encode("utf-8"))
    except (OSError, ValueError) as exc:
        print(f"report publication failed: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output} ({report['report_sha256']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
