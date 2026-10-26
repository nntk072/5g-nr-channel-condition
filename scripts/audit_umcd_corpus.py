"""Audit frozen UMCD episode and NR trajectory inventories without qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from datasets.episode_store import (
    EpisodeBundle,
    _stable_file_bytes,
    read_episode_bundle,
)
from datasets.nr import NRTrajectory, audit_nr_trajectory_target_support
from datasets.nr_store import read_nr_trajectory
from datasets.split_policy import (
    SPLIT_POLICY_SIGNATURE_NAMESPACE,
    LeakageNamespace,
    LeakageRelation,
    SplitChildBinding,
    audit_split_policy_child_roster,
    audit_split_policy_registry,
    audit_whole_scene_holdout,
    network_episode_subject,
    read_split_policy_registry,
    verify_openssh_signed_artifact,
)
from evaluation.near_duplicate_audit import ChannelRecord, audit_near_duplicates
from evaluation.target_coverage import (
    ExpectedInventoryCell,
    TargetRecord,
    audit_target_coverage,
    convert_target_map,
)

_SCHEMA = "umcd.corpus_audit_inventory.v2"
_STRATIFIED_SCHEMA = "umcd.corpus_audit_inventory.v3"
_SCENE_ROSTER_SCHEMA = "umcd.corpus_audit_inventory.v4"
_REPORT_SCHEMA_V6 = "umcd.corpus_audit_report.v6"
_REPORT_SCHEMA_V7 = "umcd.corpus_audit_report.v7"
_REPORT_SCHEMA = "umcd.corpus_audit_report.v8"
_REPORT_SCHEMA_PATHS = {
    _REPORT_SCHEMA_V6: Path(__file__).resolve().parents[1]
    / "datasets/schema/umcd-corpus-audit-report.v6.schema.json",
    _REPORT_SCHEMA_V7: Path(__file__).resolve().parents[1]
    / "datasets/schema/umcd-corpus-audit-report.v7.schema.json",
    _REPORT_SCHEMA: Path(__file__).resolve().parents[1]
    / "datasets/schema/umcd-corpus-audit-report.v8.schema.json",
}
_DUPLICATE_DISPOSITION_SCHEMA = "umcd.near_duplicate_dispositions.v2"
_DUPLICATE_DISPOSITION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "datasets/schema/umcd-near-duplicate-dispositions.v2.schema.json"
)
NEAR_DUPLICATE_DISPOSITION_SIGNATURE_NAMESPACE = "umcd-near-duplicate-dispositions-v2@github.com"
MAX_NEAR_DUPLICATE_EVIDENCE_FILE_BYTES = 64 * 1024 * 1024
MAX_NEAR_DUPLICATE_EVIDENCE_TOTAL_BYTES = 256 * 1024 * 1024
MAX_NEAR_DUPLICATE_EVIDENCE_FILES_PER_CANDIDATE = 16
CORPUS_INVENTORY_SIGNATURE_NAMESPACE = "umcd-corpus-inventory-v3@github.com"
SCENE_ROSTER_INVENTORY_SIGNATURE_NAMESPACE = "umcd-corpus-inventory-v4@github.com"
_SCHEMA_PATHS = {
    _SCHEMA: Path(__file__).resolve().parents[1]
    / "datasets/schema/umcd-corpus-audit-inventory.v2.schema.json",
    _STRATIFIED_SCHEMA: Path(__file__).resolve().parents[1]
    / "datasets/schema/umcd-corpus-audit-inventory.v3.schema.json",
    _SCENE_ROSTER_SCHEMA: Path(__file__).resolve().parents[1]
    / "datasets/schema/umcd-corpus-audit-inventory.v4.schema.json",
}
_STRATIFIED_SCHEMAS = frozenset({_STRATIFIED_SCHEMA, _SCENE_ROSTER_SCHEMA})


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_roster(candidates: list[Any]) -> tuple[list[dict[str, Any]], str]:
    roster: list[dict[str, Any]] = []
    for candidate in candidates:
        value = asdict(candidate)
        candidate_id = hashlib.sha256(_canonical_json(value)).hexdigest()
        roster.append({"candidate_id": candidate_id, **value})
    roster.sort(key=lambda item: item["candidate_id"])
    roster_digest = hashlib.sha256(_canonical_json(roster)).hexdigest()
    return roster, roster_digest


def _load_signed_near_duplicate_dispositions(
    *,
    base: Path,
    inventory_sha256: str,
    candidate_set_sha256: str,
    candidate_ids: set[str],
    disposition_path: str | Path | None,
    signature_path: str | Path | None,
    signer_id: str | None,
    trusted_allowed_signers_path: Path | None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    options = (disposition_path, signature_path, signer_id, trusted_allowed_signers_path)
    if any(value is not None for value in options) and not all(
        value is not None for value in options
    ):
        raise ValueError(
            "near-duplicate disposition, signature, signer, and trusted keyring options "
            "must be supplied together"
        )
    if disposition_path is None:
        return (
            "not_required_no_candidates" if not candidate_ids else "not_provided",
            [],
            None,
        )
    if not candidate_ids:
        raise ValueError(
            "a signed near-duplicate disposition is invalid when there are no candidates"
        )
    assert signature_path is not None
    assert signer_id is not None
    assert trusted_allowed_signers_path is not None
    for value, label in (
        (disposition_path, "near-duplicate dispositions"),
        (signature_path, "near-duplicate disposition signature"),
    ):
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError(f"{label} path must be relative to the inventory file")
        if (base / candidate).is_symlink():
            raise ValueError(f"{label} path must not be a symlink")
    disposition_file = _resolve(base, str(disposition_path), "near_duplicate_dispositions")
    signature_file = _resolve(base, str(signature_path), "near_duplicate_dispositions_signature")
    verified = verify_openssh_signed_artifact(
        disposition_file,
        signature_file,
        trusted_allowed_signers_path=trusted_allowed_signers_path,
        signer_id=signer_id,
        namespace=NEAR_DUPLICATE_DISPOSITION_SIGNATURE_NAMESPACE,
    )
    value = _parse_json_object(verified.payload, label="signed near-duplicate dispositions")
    if value.get("schema") != _DUPLICATE_DISPOSITION_SCHEMA:
        raise ValueError("unsupported signed near-duplicate disposition schema")
    expected_bytes = _canonical_json(value) + b"\n"
    if verified.payload != expected_bytes:
        raise ValueError(
            "near-duplicate dispositions must use canonical compact JSON and one newline"
        )
    import jsonschema

    schema = _load_json(_DUPLICATE_DISPOSITION_SCHEMA_PATH)
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(value))
    if errors:
        raise ValueError(
            "near-duplicate dispositions violate schema: "
            + "; ".join(error.message for error in errors[:10])
        )
    if value["reviewer_id"] != signer_id:
        raise ValueError("near-duplicate reviewer_id differs from the verified signer")
    if value["inventory_sha256"] != inventory_sha256:
        raise ValueError("near-duplicate dispositions bind a different inventory digest")
    if value["candidate_set_sha256"] != candidate_set_sha256:
        raise ValueError("near-duplicate dispositions bind a different candidate roster")
    dispositions = value["dispositions"]
    observed_ids = [row["candidate_id"] for row in dispositions]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("near-duplicate dispositions contain duplicate candidate IDs")
    if set(observed_ids) != candidate_ids:
        raise ValueError(
            "near-duplicate dispositions must cover exactly the frozen candidate roster"
        )
    canonical_dispositions = sorted(dispositions, key=lambda row: row["candidate_id"])
    evidence_total_bytes = 0
    base_root = base.resolve(strict=True)
    for disposition in canonical_dispositions:
        verified_evidence: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for evidence in disposition["evidence_files"]:
            relative = evidence["path"]
            relative_path = PurePosixPath(relative)
            if (
                relative_path.is_absolute()
                or not relative_path.parts
                or relative_path.as_posix() != relative
                or ".." in relative_path.parts
                or "\\" in relative
                or relative in {".", ".."}
            ):
                raise ValueError("near-duplicate evidence path must be canonical and inventory-relative")
            if relative in seen_paths:
                raise ValueError("near-duplicate evidence paths must be unique per candidate")
            seen_paths.add(relative)
            evidence_path = base.joinpath(*relative_path.parts)
            try:
                resolved_evidence_path = evidence_path.resolve(strict=True)
            except OSError as error:
                raise ValueError("near-duplicate evidence file is missing or unreadable") from error
            if not resolved_evidence_path.is_relative_to(base_root):
                raise ValueError("near-duplicate evidence path escapes the inventory directory")
            if resolved_evidence_path != evidence_path.absolute():
                raise ValueError("near-duplicate evidence path must not traverse symlinks")
            payload = _stable_file_bytes(
                evidence_path,
                label="near-duplicate review evidence",
                max_bytes=MAX_NEAR_DUPLICATE_EVIDENCE_FILE_BYTES,
            )
            evidence_total_bytes += len(payload)
            if evidence_total_bytes > MAX_NEAR_DUPLICATE_EVIDENCE_TOTAL_BYTES:
                raise ValueError("near-duplicate review evidence exceeds the total byte ceiling")
            digest = hashlib.sha256(payload).hexdigest()
            if digest != evidence["sha256"]:
                raise ValueError("near-duplicate evidence digest does not match its file bytes")
            verified_evidence.append({"path": relative, "sha256": digest})
        disposition["evidence_files"] = verified_evidence
    return (
        "verified",
        canonical_dispositions,
        {
            "manifest_sha256": verified.artifact_sha256,
            "signature_sha256": verified.signature_sha256,
            "allowed_signers_sha256": verified.allowed_signers_sha256,
            "signer_id": verified.signer_id,
            "signature_namespace": verified.namespace,
        },
    )


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
    return _parse_json_object(path.read_bytes(), label=str(path))


def _validate_report(report: dict[str, Any]) -> None:
    import jsonschema

    schema_path = _REPORT_SCHEMA_PATHS.get(report.get("schema"))
    if schema_path is None:
        raise ValueError("unsupported corpus audit report schema")
    schema = _load_json(schema_path)
    errors = sorted(
        jsonschema.Draft202012Validator(schema).iter_errors(report),
        key=lambda error: (tuple(str(item) for item in error.absolute_path), error.message),
    )
    if errors:
        details = "; ".join(error.message for error in errors[:10])
        raise ValueError(f"corpus audit report violates {report.get('schema')}: {details}")
    report_core = {key: value for key, value in report.items() if key != "report_sha256"}
    expected_digest = hashlib.sha256(_canonical_json(report_core)).hexdigest()
    if report.get("report_sha256") != expected_digest:
        raise ValueError("corpus audit report digest does not match its canonical payload")


def _parse_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is not strict UTF-8 JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{label} must be a JSON object")
    return decoded


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


def _episode_topology_id(bundle: EpisodeBundle) -> str:
    """Hash the explicit node/link graph and port mapping for this episode."""
    episode = bundle.episode
    arrays = {array.array_id: array for array in episode.arrays}
    nodes_by_id = {node.node_id: node for node in episode.nodes}
    identity = {
        "nodes": sorted(
            (
                node.node_id,
                node.role.value,
                node.rf_chain_count,
                arrays[node.array_id].num_ports,
            )
            for node in episode.nodes
        ),
        "links": sorted(
            (
                link.link_id,
                link.tx_node_id,
                link.rx_node_id,
                nodes_by_id[link.tx_node_id].role.value,
                nodes_by_id[link.rx_node_id].role.value,
                link.direction.value,
                link.role.value,
                link.tx_port_indices,
                link.rx_port_indices,
            )
            for link in episode.links
        ),
    }
    return "topology:sha256:" + hashlib.sha256(_canonical_json(identity)).hexdigest()


def _episode_antenna_config_id(bundle: EpisodeBundle) -> str:
    """Hash exact antenna array geometry, polarization, orientation, and port mapping."""
    arrays_by_id: dict[str, dict[str, Any]] = {}
    for array in bundle.episode.arrays:
        arrays_by_id[array.array_id] = {
            "element_positions_local_m": array.element_positions_local_m.tolist(),
            "polarization_per_element": list(array.polarization_per_element),
            "orientation_local_to_world": array.orientation_local_to_world.tolist(),
            "port_ids": list(array.port_ids),
            "port_element_weights": [
                [[float(value.real), float(value.imag)] for value in row]
                for row in array.port_element_weights
            ],
        }
    episode = bundle.episode
    identity = {
        "arrays": sorted(arrays_by_id.values(), key=_canonical_json),
        "node_assignments": sorted(
            (
                node.node_id,
                node.role.value,
                _canonical_json(arrays_by_id[node.array_id]).hex(),
            )
            for node in episode.nodes
        ),
    }
    digest = hashlib.sha256(_canonical_json(identity)).hexdigest()
    return "antenna:sha256:" + digest


def _validated_band_registry(inventory: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Validate named half-open frequency intervals declared by inventory v3."""
    bands = inventory.get("band_registry")
    if not isinstance(bands, list) or not bands:
        raise ValueError("band_registry must be a nonempty array in corpus inventory v3")
    parsed: dict[str, tuple[float, float]] = {}
    seen: set[str] = set()
    for row in bands:
        entry = _mapping(row, "band_registry entry")
        band_id = _nonempty(entry.get("band_id"), "band_registry.band_id")
        lower = entry.get("lower_hz")
        upper = entry.get("upper_hz")
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
        ):
            raise ValueError(f"band bounds for {band_id!r} must be real numbers")
        lower_hz, upper_hz = float(lower), float(upper)
        if not (math.isfinite(lower_hz) and math.isfinite(upper_hz) and 0 < lower_hz < upper_hz):
            raise ValueError(f"band bounds for {band_id!r} must be finite with 0 < lower < upper")
        if band_id in seen:
            raise ValueError(f"duplicate band_id {band_id!r}")
        seen.add(band_id)
        parsed[band_id] = (lower_hz, upper_hz)
    return parsed


def _band_for_frequency(
    carrier_hz: float,
    declared_band_id: str,
    band_registry: dict[str, tuple[float, float]],
) -> str:
    bounds = band_registry.get(declared_band_id)
    if bounds is None:
        raise ValueError(f"episode declares unknown frozen band {declared_band_id!r}")
    lower, upper = bounds
    if not lower <= carrier_hz < upper:
        raise ValueError(
            f"carrier frequency {carrier_hz!r} is outside frozen band "
            f"{declared_band_id!r} interval [{lower}, {upper})"
        )
    return declared_band_id


def _validated_target_task_bindings(inventory: dict[str, Any]) -> dict[str, str]:
    """Return the one-to-one frozen target-ID to task-ID assignment for v3."""
    raw_bindings = inventory.get("target_task_bindings")
    if not isinstance(raw_bindings, list) or not raw_bindings:
        raise ValueError("target_task_bindings must be nonempty in corpus inventory v3")
    result: dict[str, str] = {}
    for raw in raw_bindings:
        binding = _mapping(raw, "target_task_binding")
        target_id = _nonempty(binding.get("target_id"), "target_task_binding.target_id")
        task_id = _nonempty(binding.get("task_id"), "target_task_binding.task_id")
        if target_id in result:
            raise ValueError(f"duplicate target task binding for {target_id!r}")
        result[target_id] = task_id
    return result


def _audit_nr_trajectory_inventory(
    store: Path,
    inventory: dict[str, Any],
    *,
    split_by_episode: dict[str, str],
    family_by_episode: dict[str, str],
) -> dict[str, Any]:
    """Audit exact NR trajectory storage and aggregate targets by locked split/family."""
    manifest_dir = store / "manifests" / "trajectories"
    for directory in (store / "manifests", manifest_dir):
        try:
            directory_metadata = os.lstat(directory)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise ValueError("NR trajectory manifest paths must be real, non-symlink directories")
    trajectory_rows_value = inventory.get("nr_trajectory_inventory")
    expected_cells_value = inventory.get("nr_expected_coverage")
    if trajectory_rows_value is None:
        if "nr_trajectory_inventory" in inventory:
            raise ValueError("nr_trajectory_inventory must be an array when present")
        if "nr_expected_coverage" in inventory:
            raise ValueError("nr_expected_coverage requires nr_trajectory_inventory")
        if manifest_dir.exists():
            if any(manifest_dir.glob("*.json")):
                raise ValueError(
                    "NR trajectory manifests exist but are absent from the frozen inventory"
                )
        return {
            "schema": "umcd.nr_target_support_audit.v1",
            "status": "NOT_PROVIDED",
            "trajectory_count": 0,
            "cells": [],
        }

    if not isinstance(trajectory_rows_value, list):
        raise ValueError("nr_trajectory_inventory must be an array")
    if not isinstance(expected_cells_value, list) or not expected_cells_value:
        raise ValueError("nr_expected_coverage must declare split/family/target cells")
    if trajectory_rows_value and not manifest_dir.exists():
        raise ValueError("NR trajectory inventory is declared but its manifest directory is absent")

    expected_cells: list[tuple[str, str, str]] = []
    for value in expected_cells_value:
        row = _mapping(value, "NR expected-coverage row")
        if set(row) != {"split_id", "family_id", "target_id"}:
            raise ValueError(
                "NR expected-coverage rows must declare split_id, family_id, target_id"
            )
        expected_cells.append(
            (
                _nonempty(row["split_id"], "NR expected split_id"),
                _nonempty(row["family_id"], "NR expected family_id"),
                _nonempty(row["target_id"], "NR expected target_id"),
            )
        )
    valid_split_families = {
        (split_id, family_by_episode[episode_id])
        for episode_id, split_id in split_by_episode.items()
    }
    for split_id, family_id, _ in expected_cells:
        if (split_id, family_id) not in valid_split_families:
            raise ValueError(
                f"NR expected-coverage cell references an unknown split/family: "
                f"{split_id!r}/{family_id!r}"
            )

    declared_paths: set[str] = set()
    trajectory_ids: set[str] = set()
    assignments: list[tuple[str, str, NRTrajectory]] = []
    for value in trajectory_rows_value:
        row = _mapping(value, "NR trajectory inventory row")
        if set(row) != {"trajectory_id", "parent_episode_id", "manifest", "manifest_sha256"}:
            raise ValueError(
                "NR trajectory rows must declare trajectory_id, parent_episode_id, "
                "manifest, and manifest_sha256"
            )
        trajectory_id = _nonempty(row["trajectory_id"], "NR trajectory_id")
        if trajectory_id in trajectory_ids:
            raise ValueError(f"duplicate NR trajectory_id: {trajectory_id!r}")
        trajectory_ids.add(trajectory_id)
        parent_episode_id = _nonempty(row["parent_episode_id"], "NR parent_episode_id")
        if parent_episode_id not in split_by_episode or parent_episode_id not in family_by_episode:
            raise ValueError(
                f"NR trajectory {trajectory_id!r} references an unknown parent episode"
            )

        relative = _nonempty(row["manifest"], "NR trajectory manifest")
        relative_path = PurePosixPath(relative)
        expected_relative = (
            PurePosixPath("manifests/trajectories")
            / f"{hashlib.sha256(trajectory_id.encode()).hexdigest()}.json"
        )
        if (
            "\\" in relative
            or relative_path.is_absolute()
            or relative_path.as_posix() != relative
            or relative_path != expected_relative
        ):
            raise ValueError(f"NR trajectory manifest path is not canonical for {trajectory_id!r}")
        declared_paths.add(relative)
        manifest_candidate = store.joinpath(*relative_path.parts)
        manifest_metadata = os.lstat(manifest_candidate)
        if not stat.S_ISREG(manifest_metadata.st_mode) or manifest_metadata.st_nlink != 1:
            raise ValueError("NR trajectory manifest must be a single-link regular file")
        manifest_path = _resolve(store, relative, f"NR trajectory manifest for {trajectory_id}")
        expected_hash = _nonempty(row["manifest_sha256"], "NR trajectory manifest_sha256")
        if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
            raise ValueError("NR trajectory manifest_sha256 must be a lowercase SHA-256 digest")
        if _sha256(manifest_path) != expected_hash:
            raise ValueError(f"NR trajectory manifest hash mismatch for {trajectory_id!r}")
        trajectory = read_nr_trajectory(store, manifest_path)
        if trajectory.trajectory_id != trajectory_id:
            raise ValueError(f"NR trajectory manifest ID mismatch for {trajectory_id!r}")
        assignments.append(
            (
                split_by_episode[parent_episode_id],
                family_by_episode[parent_episode_id],
                trajectory,
            )
        )

    discovered = (
        sorted(path.relative_to(store).as_posix() for path in manifest_dir.glob("*.json"))
        if manifest_dir.exists()
        else []
    )
    if discovered != sorted(declared_paths):
        raise ValueError("frozen NR trajectory inventory must cover exactly the manifest set")
    return audit_nr_trajectory_target_support(tuple(assignments), tuple(expected_cells))


def audit_inventory(
    inventory_path: Path,
    *,
    max_pair_comparisons: int = 100_000,
    trusted_allowed_signers_path: Path | None = None,
    near_duplicate_dispositions: str | Path | None = None,
    near_duplicate_dispositions_signature: str | Path | None = None,
    near_duplicate_reviewer_id: str | None = None,
    trusted_near_duplicate_reviewers_path: Path | None = None,
) -> dict[str, Any]:
    """Run manifest, bundle, split, target-coverage, trajectory, and duplicate checks."""
    if isinstance(max_pair_comparisons, bool) or not isinstance(max_pair_comparisons, int):
        raise TypeError("max_pair_comparisons must be a positive integer")
    if max_pair_comparisons <= 0:
        raise ValueError("max_pair_comparisons must be a positive integer")
    inventory_path = inventory_path.resolve(strict=True)
    base = inventory_path.parent
    inventory = _load_json(inventory_path)
    import jsonschema

    verified_inventory = None
    if inventory.get("schema") in _STRATIFIED_SCHEMAS:
        inventory_attestation = _mapping(
            inventory.get("inventory_attestation"), "inventory_attestation"
        )
        inventory_signature_relative = inventory_attestation.get("signature_path")
        inventory_signer_id = inventory_attestation.get("signer_id")
        has_inventory_signature = inventory_signature_relative is not None
        has_inventory_signer = inventory_signer_id is not None
        if has_inventory_signature != has_inventory_signer:
            raise ValueError(
                "inventory_attestation signature_path and signer_id must be supplied together"
            )
        inventory_attestation_status = _nonempty(
            inventory_attestation.get("status"), "inventory_attestation.status"
        )
        if has_inventory_signature:
            if inventory_attestation_status not in {"provided", "attested"}:
                raise ValueError(
                    "inventory signature requires inventory_attestation status provided or attested"
                )
            if trusted_allowed_signers_path is None:
                raise ValueError(
                    "signed corpus inventory verification requires an independent allowed-signers file"
                )
            signature_candidate = base / _nonempty(
                inventory_signature_relative, "inventory_attestation.signature_path"
            )
            if signature_candidate.is_symlink():
                raise ValueError("corpus inventory signature path must not be a symlink")
            inventory_signature_path = _resolve(
                base,
                inventory_signature_relative,
                "inventory_attestation.signature_path",
            )
            verified_inventory = verify_openssh_signed_artifact(
                inventory_path,
                inventory_signature_path,
                trusted_allowed_signers_path=trusted_allowed_signers_path,
                signer_id=_nonempty(inventory_signer_id, "inventory_attestation.signer_id"),
                namespace=(
                    SCENE_ROSTER_INVENTORY_SIGNATURE_NAMESPACE
                    if inventory.get("schema") == _SCENE_ROSTER_SCHEMA
                    else CORPUS_INVENTORY_SIGNATURE_NAMESPACE
                ),
            )
            signed_inventory = _parse_json_object(
                verified_inventory.payload,
                label="signed corpus inventory",
            )
            if signed_inventory != inventory:
                raise ValueError(
                    "corpus inventory changed between parsing and signature verification"
                )
            inventory = signed_inventory
        elif inventory_attestation_status in {"provided", "attested"}:
            raise ValueError(
                "provided or attested inventory signature status requires signature_path and signer_id"
            )

    inventory_schema_name = inventory.get("schema")
    if not isinstance(inventory_schema_name, str):
        raise ValueError("inventory schema must be a supported string version")
    schema_path = _SCHEMA_PATHS.get(inventory_schema_name)
    if schema_path is None:
        raise ValueError(f"inventory schema must be one of {sorted(_SCHEMA_PATHS)!r}")
    inventory_schema = _load_json(schema_path)
    schema_errors = sorted(
        jsonschema.Draft202012Validator(inventory_schema).iter_errors(inventory),
        key=lambda error: (tuple(str(item) for item in error.absolute_path), error.message),
    )
    if schema_errors:
        details = "; ".join(error.message for error in schema_errors[:10])
        raise ValueError(f"corpus inventory violates {inventory_schema_name} schema: {details}")
    if inventory.get("schema") not in _SCHEMA_PATHS or inventory.get("frozen") is not True:
        raise ValueError("inventory must be frozen and use a supported corpus-audit schema")
    is_stratified_inventory = inventory_schema_name in _STRATIFIED_SCHEMAS
    has_scene_roster = inventory_schema_name == _SCENE_ROSTER_SCHEMA
    band_registry = _validated_band_registry(inventory) if is_stratified_inventory else {}
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
    registry_relative = _nonempty(registry_config.get("path"), "split_registry.path")
    registry_candidate = base / registry_relative
    if registry_candidate.is_symlink():
        raise ValueError("split registry path must not be a symlink")
    registry_path = _resolve(base, registry_relative, "split_registry.path")
    registry_hash = _nonempty(registry_config.get("sha256"), "split_registry.sha256")
    signature_relative = attestation.get("signature_path")
    signer_id = attestation.get("signer_id")
    has_signature_path = signature_relative is not None
    has_signer_id = signer_id is not None
    if has_signature_path != has_signer_id:
        raise ValueError("split-lock signature_path and signer_id must be supplied together")
    verified_split_lock = None
    if has_signature_path:
        if attestation_status not in {"provided", "attested"}:
            raise ValueError(
                "signed split-lock metadata requires external_lock_attestation status provided or attested"
            )
        if trusted_allowed_signers_path is None:
            raise ValueError(
                "signed split-lock verification requires an independent allowed-signers file"
            )
        signature_relative = _nonempty(
            signature_relative, "external_lock_attestation.signature_path"
        )
        signature_candidate = base / signature_relative
        if signature_candidate.is_symlink():
            raise ValueError("split-lock signature path must not be a symlink")
        signature_path = _resolve(
            base, signature_relative, "external_lock_attestation.signature_path"
        )
        verified_split_lock = verify_openssh_signed_artifact(
            registry_path,
            signature_path,
            trusted_allowed_signers_path=trusted_allowed_signers_path,
            signer_id=_nonempty(signer_id, "external_lock_attestation.signer_id"),
            namespace=SPLIT_POLICY_SIGNATURE_NAMESPACE,
        )
        if verified_split_lock.artifact_sha256 != registry_hash:
            raise ValueError("signed split registry file hash does not match frozen inventory")
        with tempfile.TemporaryDirectory(prefix="umcd-split-registry-audit-") as temporary:
            signed_snapshot = Path(temporary) / "split-registry.json"
            signed_snapshot.write_bytes(verified_split_lock.payload)
            registry = read_split_policy_registry(signed_snapshot)
    else:
        if _sha256(registry_path) != registry_hash:
            raise ValueError("split registry file hash does not match frozen inventory")
        registry = read_split_policy_registry(registry_path)

    if (
        trusted_allowed_signers_path is not None
        and verified_split_lock is None
        and verified_inventory is None
    ):
        raise ValueError("allowed-signers file supplied but no signed corpus artifact is declared")

    rows = _records(inventory.get("bundles"))
    expected_manifests: dict[str, tuple[str, str]] = {}
    bundles: dict[str, EpisodeBundle] = {}
    split_by_episode: dict[str, str] = {}
    family_by_episode: dict[str, str] = {}
    strata_by_episode: dict[str, dict[str, str]] = {}
    subjects = []
    observed_children: list[SplitChildBinding] = []
    manifest_dir = store / "manifests" / "episodes"
    for directory in (store / "manifests", manifest_dir):
        directory_metadata = os.lstat(directory)
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise ValueError("episode manifest directories must be real, non-symlink directories")
    for row in rows:
        episode_id = _nonempty(row.get("episode_id"), "bundle.episode_id")
        relative = _nonempty(row.get("manifest"), f"manifest for {episode_id}")
        manifest_relative_path = PurePosixPath(relative)
        if (
            "\\" in relative
            or manifest_relative_path.is_absolute()
            or manifest_relative_path.as_posix() != relative
            or any(part in {"", ".", ".."} for part in manifest_relative_path.parts)
            or manifest_relative_path.parts[:2] != ("manifests", "episodes")
            or manifest_relative_path.suffix != ".json"
        ):
            raise ValueError(
                f"episode manifest path for {episode_id!r} must be a canonical "
                "manifests/episodes/*.json path"
            )
        manifest_candidate = store.joinpath(*manifest_relative_path.parts)
        manifest_metadata = os.lstat(manifest_candidate)
        if not stat.S_ISREG(manifest_metadata.st_mode) or manifest_metadata.st_nlink != 1:
            raise ValueError("episode manifest must be a single-link regular non-symlink file")
        expected_hash = _nonempty(row.get("manifest_sha256"), f"manifest hash for {episode_id}")
        manifest_path = _resolve(store, relative, f"manifest for {episode_id}")
        if _sha256(manifest_path) != expected_hash:
            raise ValueError(f"bundle manifest hash mismatch for {episode_id!r}")
        bundle = read_episode_bundle(store, manifest_path)
        if bundle.episode.episode_id != episode_id:
            raise ValueError(f"manifest and inventory episode IDs differ for {episode_id!r}")
        bundles[episode_id] = bundle
        split_by_episode[episode_id] = registry.split_for_parent(episode_id)
        family_by_episode[episode_id] = bundle.episode.family_id
        expected_manifests[episode_id] = (relative, expected_hash)
        relations = _relations(row.get("relations", []), episode_id)
        subjects.append(network_episode_subject(bundle.episode, relations=relations))
        if is_stratified_inventory:
            scene_ids = [
                relation.identity
                for relation in relations
                if relation.namespace is LeakageNamespace.SCENE_REALIZATION
            ]
            if len(scene_ids) != 1:
                raise ValueError(
                    f"stratified inventory requires exactly one scene_realization relation "
                    f"for episode {episode_id!r}"
                )
            strata_by_episode[episode_id] = {
                "scene_id": scene_ids[0],
                "band_id": _band_for_frequency(
                    bundle.episode.carrier_hz,
                    _nonempty(row.get("band_id"), f"bundle band_id for {episode_id}"),
                    band_registry,
                ),
                "topology_id": _episode_topology_id(bundle),
                "antenna_config_id": _episode_antenna_config_id(bundle),
            }
        observed_children.extend(_bundle_children(bundle))

    discovered = (
        sorted(path.relative_to(store).as_posix() for path in manifest_dir.glob("*.json"))
        if manifest_dir.exists()
        else []
    )
    declared = sorted(value[0] for value in expected_manifests.values())
    if discovered != declared:
        raise ValueError("frozen bundle inventory must cover exactly the episode manifest set")
    if len({item.child_id for item in observed_children}) != len(observed_children):
        raise ValueError("receiver child IDs must be unique across bundles")
    audit_split_policy_registry(registry, subjects, observed_children)
    whole_scene_report = None
    if has_scene_roster:
        scene_rows = inventory.get("scene_inventory")
        if not isinstance(scene_rows, list) or not scene_rows:
            raise ValueError("scene_inventory must be a nonempty frozen roster")
        scene_subjects: dict[str, list[str]] = {}
        for row in scene_rows:
            scene_row = _mapping(row, "scene_inventory row")
            scene_id = _nonempty(scene_row.get("scene_id"), "scene_inventory.scene_id")
            parent_ids = scene_row.get("parent_episode_ids")
            if not isinstance(parent_ids, list) or not parent_ids:
                raise ValueError("scene_inventory.parent_episode_ids must be a nonempty array")
            if scene_id in scene_subjects:
                raise ValueError(f"duplicate scene_inventory scene_id: {scene_id!r}")
            scene_subjects[scene_id] = [
                _nonempty(parent_id, "scene_inventory parent_episode_id")
                for parent_id in parent_ids
            ]
        whole_scene_report = audit_whole_scene_holdout(
            registry,
            scene_subjects,
            expected_scene_ids=tuple(scene_subjects),
            expected_subject_ids=tuple(sorted(bundles)),
        )
    children_by_kind: dict[str, list[SplitChildBinding]] = defaultdict(list)
    for binding in observed_children:
        children_by_kind[binding.child_kind].append(binding)
    child_roster_reports = {
        kind: audit_split_policy_child_roster(registry, tuple(bindings), child_kind=kind)
        for kind, bindings in sorted(children_by_kind.items())
    }

    target_records: list[TargetRecord] = []
    all_definitions: dict[str, Any] = {}
    target_task_bindings = (
        _validated_target_task_bindings(inventory) if is_stratified_inventory else None
    )
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
            if target_task_bindings is None:
                target_records.append(
                    TargetRecord(
                        record_id=episode_id,
                        split_id=split_id,
                        family_id=bundle.episode.family_id,
                        targets=observations,
                        independent_group_id=registry.subject_to_component[episode_id],
                    )
                )
            else:
                grouped_observations: dict[str, list[Any]] = defaultdict(list)
                for observation in observations:
                    task_id = target_task_bindings.get(observation.target_id)
                    if task_id is None:
                        raise ValueError(
                            f"target {observation.target_id!r} has no frozen task binding"
                        )
                    grouped_observations[task_id].append(observation)
                strata = strata_by_episode[episode_id]
                for task_id, task_observations in sorted(grouped_observations.items()):
                    record_id = json.dumps(
                        [episode_id, task_id], separators=(",", ":"), ensure_ascii=False
                    )
                    target_records.append(
                        TargetRecord(
                            record_id=record_id,
                            split_id=split_id,
                            family_id=bundle.episode.family_id,
                            targets=tuple(task_observations),
                            independent_group_id=registry.subject_to_component[episode_id],
                            scene_id=strata["scene_id"],
                            band_id=strata["band_id"],
                            topology_id=strata["topology_id"],
                            antenna_config_id=strata["antenna_config_id"],
                            task_id=task_id,
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

    if target_task_bindings is not None:
        target_ids = set(all_definitions)
        bound_target_ids = set(target_task_bindings)
        if bound_target_ids != target_ids:
            raise ValueError(
                "target_task_bindings must cover exactly the observed target definitions; "
                f"missing={sorted(target_ids - bound_target_ids)!r}, "
                f"extra={sorted(bound_target_ids - target_ids)!r}"
            )
        registered_bands = set(band_registry)
        inventoried_bands = {
            _nonempty(item.get("band_id"), "expected_coverage.band_id")
            for item in inventory["expected_coverage"]
        }
        unknown_bands = inventoried_bands - registered_bands
        if unknown_bands:
            raise ValueError(
                f"expected coverage references unknown bands: {sorted(unknown_bands)!r}"
            )

    expected_coverage = inventory.get("expected_coverage")
    if not isinstance(expected_coverage, list) or not expected_coverage:
        raise ValueError("expected_coverage must declare nonempty frozen target cells")
    coverage = audit_target_coverage(
        tuple(target_records),
        tuple(all_definitions.values()),
        tuple(ExpectedInventoryCell(**item) for item in expected_coverage),
    )
    nr_target_support = _audit_nr_trajectory_inventory(
        store,
        inventory,
        split_by_episode=split_by_episode,
        family_by_episode=family_by_episode,
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
    candidate_rows, candidate_set_sha256 = _candidate_roster(candidates)
    inventory_sha256 = (
        verified_inventory.artifact_sha256
        if verified_inventory is not None
        else _sha256(inventory_path)
    )
    disposition_status, dispositions, disposition_attestation = (
        _load_signed_near_duplicate_dispositions(
            base=base,
            inventory_sha256=inventory_sha256,
            candidate_set_sha256=candidate_set_sha256,
            candidate_ids={row["candidate_id"] for row in candidate_rows},
            disposition_path=near_duplicate_dispositions,
            signature_path=near_duplicate_dispositions_signature,
            signer_id=near_duplicate_reviewer_id,
            trusted_allowed_signers_path=trusted_near_duplicate_reviewers_path,
        )
    )
    if disposition_attestation is not None:
        upstream_attestations = [
            item for item in (verified_inventory, verified_split_lock) if item is not None
        ]
        if any(
            disposition_attestation["signer_id"] == item.signer_id for item in upstream_attestations
        ):
            raise ValueError(
                "near-duplicate reviewer must be distinct from inventory and split-lock signers"
            )
        if any(
            disposition_attestation["allowed_signers_sha256"] == item.allowed_signers_sha256
            for item in upstream_attestations
        ):
            raise ValueError(
                "near-duplicate reviewer keyring must be distinct from inventory and split-lock keyrings"
            )
    if disposition_status == "verified":
        disposition_outcomes = {row["outcome"] for row in dispositions}
        duplicate_status = (
            "confirmed_cross_split_leakage"
            if "confirmed_leakage" in disposition_outcomes
            else "incomplete_comparisons"
            if skipped
            else "adjudicated_no_leakage"
        )
    else:
        duplicate_status = (
            "review_required"
            if candidates
            else "incomplete_comparisons"
            if skipped
            else "no_candidates_detected"
            if pair_count
            else "not_run_no_cross_split_pairs"
        )

    report_core = {
        "schema": _REPORT_SCHEMA,
        "inventory_id": inventory_id,
        "inventory_sha256": (inventory_sha256),
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
                "child_rosters": child_roster_reports,
                "split_counts": dict(sorted(registry.independent_component_counts.items())),
                **(
                    {"whole_scene_holdout": whole_scene_report}
                    if whole_scene_report is not None
                    else {}
                ),
            },
            "target_coverage": {
                "status": "passed",
                "cells": [asdict(cell) for cell in coverage.cells],
            },
            "nr_trajectory_target_support": nr_target_support,
            "near_duplicate_review": {
                "status": duplicate_status,
                "threshold": float(threshold),
                "comparable_group_count": len(duplicate_reports),
                "cross_split_pair_count": pair_count,
                "total_pair_comparison_count": comparison_count,
                "candidate_count": len(candidates),
                "candidates": candidate_rows,
                "skipped_comparison_count": len(skipped),
                "candidate_set_sha256": candidate_set_sha256,
                "disposition_status": disposition_status,
                "disposition_manifest_sha256": (
                    disposition_attestation["manifest_sha256"]
                    if disposition_attestation is not None
                    else None
                ),
                "disposition_signature_sha256": (
                    disposition_attestation["signature_sha256"]
                    if disposition_attestation is not None
                    else None
                ),
                "disposition_allowed_signers_sha256": (
                    disposition_attestation["allowed_signers_sha256"]
                    if disposition_attestation is not None
                    else None
                ),
                "disposition_signer_id": (
                    disposition_attestation["signer_id"]
                    if disposition_attestation is not None
                    else None
                ),
                "disposition_signature_namespace": (
                    disposition_attestation["signature_namespace"]
                    if disposition_attestation is not None
                    else None
                ),
                "dispositions": dispositions,
            },
        },
        "external_gates": {
            "inventory_attestation": {
                "declared_status": (
                    inventory.get("inventory_attestation", {}).get("status", "not_applicable")
                ),
                "status": (
                    "signature_valid_against_supplied_trust_file"
                    if verified_inventory is not None
                    else "external_unverified"
                    if is_stratified_inventory
                    else "not_applicable"
                ),
                "trust_file_custody": (
                    "not_established_by_this_audit"
                    if verified_inventory is not None
                    else "not_verified"
                    if is_stratified_inventory
                    else "not_applicable"
                ),
                "signer_id": None if verified_inventory is None else verified_inventory.signer_id,
                "signature_sha256": (
                    None if verified_inventory is None else verified_inventory.signature_sha256
                ),
                "allowed_signers_sha256": (
                    None
                    if verified_inventory is None
                    else verified_inventory.allowed_signers_sha256
                ),
                "signature_namespace": (
                    None if verified_inventory is None else verified_inventory.namespace
                ),
            },
            "legal_assessment": {
                "declared_status": legal_status,
                "status": "external_unverified",
                "evidence_sha256": legal.get("evidence_sha256"),
            },
            "external_lock_attestation": {
                "declared_status": attestation_status,
                "status": (
                    "signature_valid_against_supplied_trust_file"
                    if verified_split_lock is not None
                    else "external_unverified"
                ),
                "trust_file_custody": (
                    "not_established_by_this_audit"
                    if verified_split_lock is not None
                    else "not_verified"
                ),
                "evidence_sha256": attestation.get("evidence_sha256"),
                "signer_id": None if verified_split_lock is None else verified_split_lock.signer_id,
                "signature_sha256": (
                    None if verified_split_lock is None else verified_split_lock.signature_sha256
                ),
                "allowed_signers_sha256": (
                    None
                    if verified_split_lock is None
                    else verified_split_lock.allowed_signers_sha256
                ),
                "signature_namespace": (
                    None if verified_split_lock is None else verified_split_lock.namespace
                ),
            },
        },
        "qualification": "not_claimed",
        "fixture_notice": "Passing software checks do not qualify a corpus; fixtures remain fixtures.",
    }
    report = {
        **report_core,
        "report_sha256": hashlib.sha256(_canonical_json(report_core)).hexdigest(),
    }
    _validate_report(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inventory",
        type=Path,
        help=f"frozen corpus inventory using one of {sorted(_SCHEMA_PATHS)!r}",
    )
    parser.add_argument("--output", type=Path, required=True, help="report JSON output path")
    parser.add_argument("--max-pair-comparisons", type=int, default=100_000)
    parser.add_argument(
        "--trusted-allowed-signers",
        type=Path,
        help="independently governed OpenSSH trust file for a signed split registry",
    )
    parser.add_argument(
        "--near-duplicate-dispositions",
        help="inventory-relative signed reviewer disposition manifest for the exact candidate set",
    )
    parser.add_argument(
        "--near-duplicate-dispositions-signature",
        help="inventory-relative detached OpenSSH signature for the disposition manifest",
    )
    parser.add_argument(
        "--near-duplicate-reviewer-id",
        help="OpenSSH principal expected to sign the candidate dispositions",
    )
    parser.add_argument(
        "--trusted-near-duplicate-reviewers",
        type=Path,
        help="externally governed OpenSSH allowed_signers file for near-duplicate reviewers",
    )
    args = parser.parse_args(argv)
    try:
        report = audit_inventory(
            args.inventory,
            max_pair_comparisons=args.max_pair_comparisons,
            trusted_allowed_signers_path=args.trusted_allowed_signers,
            near_duplicate_dispositions=args.near_duplicate_dispositions,
            near_duplicate_dispositions_signature=args.near_duplicate_dispositions_signature,
            near_duplicate_reviewer_id=args.near_duplicate_reviewer_id,
            trusted_near_duplicate_reviewers_path=args.trusted_near_duplicate_reviewers,
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
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
            _validate_report(report)
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
    duplicate_status = report.get("checks", {}).get("near_duplicate_review", {}).get("status")
    if duplicate_status in {
        "review_required",
        "incomplete_comparisons",
        "confirmed_cross_split_leakage",
    }:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
