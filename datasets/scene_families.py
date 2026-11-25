"""Versioned scene-family inventory and deterministic candidate sampling.

The registry describes candidate availability and split identities. It does not
establish that a family is independent, calibrated, releasable, or qualified.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets.split_policy import LeakageNamespace, LeakageRelation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "config/scenes/scene-family-registry.v1.json"
FAMILY_IDS = (
    "S00-analytic",
    "S01-urban-canyon",
    "S02-dense-urban",
    "S03-suburban-rural",
    "S04-indoor-office",
    "S05-indoor-factory-warehouse",
    "S06-station-airport-mall",
    "S07-outdoor-to-indoor",
    "S08-tunnel-garage",
    "S09-mmwave-street",
    "S10-public-unseen-site",
    "S11-measured-campaign",
)
_SOURCE_KINDS = frozenset(
    {
        "analytic_fixture",
        "procedural_fixture",
        "odbl_osm_scene",
        "surveyed_site",
        "measured_campaign",
    }
)
_RESEARCH_STATES = frozenset({"allowed", "not_recorded", "restricted"})
_RELEASE_STATES = frozenset({"not_cleared", "review_required", "not_applicable"})


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a canonical nonempty string")
    return value


def _digest(value: Any, field: str) -> str:
    value = _canonical_text(value, field)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def validate_scene_family_registry(value: Any) -> dict[str, Any]:
    """Validate a complete S00–S11 catalogue and its evidenced candidates."""
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "taxonomy",
        "evidence",
        "candidates",
        "qualification_status",
    }:
        raise ValueError("scene-family registry has unexpected top-level fields")
    if value["schema"] != "umcd.scene_family_registry.v1":
        raise ValueError("unsupported scene-family registry schema")
    if value["qualification_status"] != "NOT_QUALIFIED":
        raise ValueError("scene-family registry cannot claim qualification")
    taxonomy = value["taxonomy"]
    if (
        not isinstance(taxonomy, list)
        or tuple(row.get("family_id") if isinstance(row, dict) else None for row in taxonomy)
        != FAMILY_IDS
    ):
        raise ValueError("taxonomy must contain the complete ordered S00–S11 family roster")
    for row in taxonomy:
        if set(row) != {"family_id", "role", "description"}:
            raise ValueError("taxonomy row fields are invalid")
        for field in ("role", "description"):
            _canonical_text(row[field], f"taxonomy.{field}")
    evidence = value["evidence"]
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("registry must bind source evidence")
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "purpose"}:
            raise ValueError("registry evidence row fields are invalid")
        _canonical_text(item["path"], "evidence.path")
        _canonical_text(item["purpose"], "evidence.purpose")
        _digest(item["sha256"], "evidence.sha256")
    evidence_by_path = {item["path"]: item["sha256"] for item in evidence}
    candidates = value["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("candidates must be an array")
    ids: set[str] = set()
    for row in candidates:
        required = {
            "candidate_id",
            "family_id",
            "scene_id",
            "independence_group_id",
            "source_kind",
            "geometry_asset_sha256",
            "source_record_path",
            "source_record_sha256",
            "license",
            "internal_research_status",
            "public_release_status",
            "qualification_status",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("scene candidate fields are invalid")
        candidate_id = _canonical_text(row["candidate_id"], "candidate_id")
        if candidate_id in ids:
            raise ValueError(f"duplicate scene candidate ID: {candidate_id}")
        ids.add(candidate_id)
        if row["family_id"] not in FAMILY_IDS:
            raise ValueError(f"unknown scene family: {row['family_id']!r}")
        for field in (
            "scene_id",
            "independence_group_id",
            "source_record_path",
            "license",
        ):
            _canonical_text(row[field], field)
        if row["source_kind"] not in _SOURCE_KINDS:
            raise ValueError(f"unsupported scene source kind: {row['source_kind']!r}")
        _digest(row["geometry_asset_sha256"], "geometry_asset_sha256")
        _digest(row["source_record_sha256"], "source_record_sha256")
        if evidence_by_path.get(row["source_record_path"]) != row["source_record_sha256"]:
            raise ValueError("candidate source record is not bound by registry evidence")
        if row["internal_research_status"] not in _RESEARCH_STATES:
            raise ValueError("invalid internal research status")
        if row["public_release_status"] not in _RELEASE_STATES:
            raise ValueError("invalid public release status")
        if row["qualification_status"] != "NOT_QUALIFIED":
            raise ValueError("scene candidates cannot claim scientific qualification")
        if row["source_kind"] == "odbl_osm_scene" and (
            row["license"] != "ODbL-1.0"
            or row["internal_research_status"] != "allowed"
            or row["public_release_status"] != "not_cleared"
        ):
            raise ValueError("ODbL scene candidate must preserve the accepted use/release policy")
    return value


def load_scene_family_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    """Load and validate the frozen catalogue, rejecting duplicate JSON keys."""

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    value = validate_scene_family_registry(
        json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    )
    return value


def scene_family_registry_sha256(registry: Mapping[str, Any]) -> str:
    """Return the canonical digest used to bind a roster or report."""
    value = dict(validate_scene_family_registry(dict(registry)))
    value["evidence"] = sorted(value["evidence"], key=lambda row: row["path"])
    value["candidates"] = sorted(value["candidates"], key=lambda row: row["candidate_id"])
    return _sha256(_canonical_json(value))


def sample_scene_candidates(
    registry: Mapping[str, Any], *, seed: int, candidates_per_family: int = 1
) -> dict[str, Any]:
    """Deterministically rank candidate independence groups within each family.

    Families without candidates and shortages remain explicit. Candidate rows
    marked ``not_recorded`` or ``restricted`` for internal use are not selected.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**128:
        raise ValueError("seed must be a nonnegative integer below 2^128")
    if (
        isinstance(candidates_per_family, bool)
        or not isinstance(candidates_per_family, int)
        or candidates_per_family <= 0
    ):
        raise ValueError("candidates_per_family must be a positive integer")
    value = validate_scene_family_registry(dict(registry))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in value["candidates"]:
        if candidate["internal_research_status"] == "allowed":
            by_family[candidate["family_id"]].append(candidate)
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for family_id in FAMILY_IDS:
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in by_family[family_id]:
            groups[candidate["independence_group_id"]].append(candidate)
        ranked_groups = sorted(
            groups,
            key=lambda group: hashlib.sha256(f"{seed}\0{family_id}\0{group}".encode()).digest(),
        )
        chosen_groups = ranked_groups[:candidates_per_family]
        for group in chosen_groups:
            candidate = min(
                groups[group],
                key=lambda item: hashlib.sha256(
                    f"{seed}\0{family_id}\0{group}\0{item['candidate_id']}".encode()
                ).digest(),
            )
            selected.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "family_id": family_id,
                    "independence_group_id": group,
                    "geometry_asset_sha256": candidate["geometry_asset_sha256"],
                    "public_release_status": candidate["public_release_status"],
                }
            )
        rows.append(
            {
                "family_id": family_id,
                "available_candidates": len(by_family[family_id]),
                "available_independence_groups": len(groups),
                "requested_independence_groups": candidates_per_family,
                "selected_independence_groups": len(chosen_groups),
                "shortfall": candidates_per_family - len(chosen_groups),
                "status": "CANDIDATES_AVAILABLE" if chosen_groups else "UNSUPPORTED_NO_CANDIDATE",
            }
        )
    core = {
        "schema": "umcd.scene_family_sample.v1",
        "registry_sha256": scene_family_registry_sha256(value),
        "implementation_source_sha256": _sha256(Path(__file__).read_bytes()),
        "cli_source_sha256": _sha256((ROOT / "scripts/sample_scene_families.py").read_bytes()),
        "sampling_algorithm": "sha256_rank_independence_group_v1",
        "seed": seed,
        "qualification_status": "NOT_QUALIFIED",
        "selected_candidates": selected,
        "family_coverage": rows,
    }
    return {**core, "sample_sha256": _sha256(_canonical_json(core))}


def scene_candidate_split_relations(
    registry: Mapping[str, Any], candidate_ids: list[str]
) -> dict[str, tuple[LeakageRelation, ...]]:
    """Map candidate IDs to canonical relations accepted by the signed splitter.

    Callers attach these relations to the corresponding channel parents before
    freezing the existing SplitPolicyRegistry. Asset and site edges therefore
    join with the canonical union-find and signed lock rather than a new splitter.
    """
    value = validate_scene_family_registry(dict(registry))
    rows = {row["candidate_id"]: row for row in value["candidates"]}
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must not contain duplicates")
    result: dict[str, tuple[LeakageRelation, ...]] = {}
    for candidate_id in candidate_ids:
        if candidate_id not in rows:
            raise ValueError(f"unknown scene candidate: {candidate_id}")
        row = rows[candidate_id]
        result[candidate_id] = (
            LeakageRelation(LeakageNamespace.ASSET_FAMILY, row["independence_group_id"]),
            LeakageRelation(LeakageNamespace.SITE, row["scene_id"]),
            LeakageRelation(
                LeakageNamespace.SCENE_REALIZATION,
                f"sha256:{row['geometry_asset_sha256']}",
            ),
        )
    return result


def scene_candidate_split_metadata(
    registry: Mapping[str, Any], *, candidate_id: str, actual_geometry_asset_sha256: str
) -> dict[str, Any]:
    """Build row metadata only when candidate identity matches actual scene bytes."""
    value = validate_scene_family_registry(dict(registry))
    candidate_id = _canonical_text(candidate_id, "candidate_id")
    actual_digest = _digest(actual_geometry_asset_sha256, "actual_geometry_asset_sha256")
    rows = {row["candidate_id"]: row for row in value["candidates"]}
    if candidate_id not in rows:
        raise ValueError(f"unknown scene candidate: {candidate_id}")
    candidate = rows[candidate_id]
    if candidate["internal_research_status"] != "allowed":
        raise ValueError("scene candidate is not admitted for internal research use")
    if actual_digest != candidate["geometry_asset_sha256"]:
        raise ValueError("scene candidate geometry digest differs from loaded scene assets")
    relations = scene_candidate_split_relations(value, [candidate_id])[candidate_id]
    return {
        "candidate_id": candidate_id,
        "family_id": candidate["family_id"],
        "registry_sha256": scene_family_registry_sha256(value),
        "geometry_asset_sha256": candidate["geometry_asset_sha256"],
        "split_leakage_relations": [relation.to_dict() for relation in relations],
        "qualification_status": "NOT_QUALIFIED",
    }
