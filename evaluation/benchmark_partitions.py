"""Candidate benchmark cohort manifests bound to the canonical signed split lock.

These are input-scoped, non-copying row indexes. Declared site, array and
external provenance are reviewed *claims*, not empirical qualification.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

from datasets.channel import ChannelDataset
from datasets.split_policy import LeakageNamespace, SplitPolicyRegistry

TEST_COHORTS = ("site_held_out", "site_array_held_out", "real_external")
BENCHMARK_COHORTS = ("train", "validation", "calibration", *TEST_COHORTS)
_EXTERNAL_FIELDS = frozenset(
    {"measurement_inventory_sha256", "measurement_protocol_sha256", "rights_review_sha256"}
)


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a canonical nonempty string")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _declared_id_map(value: Any, label: str, expected: set[str]) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} must cover exactly the frozen parent roster")
    return {
        parent: _identifier(identity, f"{label}[{parent}]") for parent, identity in value.items()
    }


def _validate_declarations(
    registry: SplitPolicyRegistry, declaration: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if not isinstance(declaration, dict) or set(declaration) != {
        "schema",
        "schema_version",
        "test_cohorts",
        "site_ids",
        "array_ids",
        "external_evidence",
    }:
        raise ValueError("benchmark cohort declaration has an invalid field roster")
    if (declaration["schema"], declaration["schema_version"]) != (
        "umcd.benchmark_cohort_declaration.v1",
        1,
    ) or type(declaration["schema_version"]) is not int:
        raise ValueError("unsupported benchmark cohort declaration")
    if set(registry.policy.split_fractions) != {"train", "validation", "calibration", "test"}:
        raise ValueError("benchmark requires the canonical four-way split policy")
    parents = set(registry.subjects)
    sites = _declared_id_map(declaration["site_ids"], "site_ids", parents)
    arrays = _declared_id_map(declaration["array_ids"], "array_ids", parents)
    cohorts = declaration["test_cohorts"]
    if not isinstance(cohorts, dict) or set(cohorts) != set(TEST_COHORTS):
        raise ValueError("test_cohorts must declare all three benchmark test cohorts")
    assigned: dict[str, str] = {}
    for cohort in TEST_COHORTS:
        ids = cohorts[cohort]
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError(f"{cohort} must be an array of parent IDs")
        if ids != sorted(set(ids)):
            raise ValueError(f"{cohort} roster must be sorted and unique")
        for parent in ids:
            if parent not in registry.subjects or registry.split_for_parent(parent) != "test":
                raise ValueError(f"{cohort} references an unknown or non-test parent: {parent}")
            if parent in assigned:
                raise ValueError(f"parent assigned to more than one cohort: {parent}")
            assigned[parent] = cohort
    test_parents = {parent for parent in parents if registry.split_for_parent(parent) == "test"}
    if set(assigned) != test_parents:
        raise ValueError("test cohort declaration must cover every frozen test parent exactly")
    for parent in parents - test_parents:
        assigned[parent] = registry.split_for_parent(parent)

    evidence = declaration["external_evidence"]
    external_parents = set(cohorts["real_external"])
    if not isinstance(evidence, dict) or set(evidence) != external_parents:
        raise ValueError("external_evidence must cover exactly the real_external parent roster")
    for parent, row in evidence.items():
        if not isinstance(row, dict) or set(row) != _EXTERNAL_FIELDS:
            raise ValueError(f"external evidence is incomplete for {parent}")
        for field, digest in row.items():
            _sha256(digest, f"external_evidence[{parent}].{field}")

    # When a site identity is present in the signed lock, the declaration must agree.
    # Missing site relations remain an authority gap, not an invented release claim.
    for parent, subject in registry.subjects.items():
        bound_sites = {
            item.identity for item in subject.relations if item.namespace == LeakageNamespace.SITE
        }
        if bound_sites and bound_sites != {sites[parent]}:
            raise ValueError(f"declared site contradicts frozen site relation: {parent}")

    by_cohort: dict[str, set[str]] = defaultdict(set)
    for parent, cohort in assigned.items():
        by_cohort[cohort].add(parent)
    for cohort in ("site_held_out", "site_array_held_out"):
        held_sites = {sites[parent] for parent in by_cohort[cohort]}
        other_sites = {sites[parent] for parent in parents if assigned[parent] != cohort}
        if held_sites & other_sites:
            raise ValueError(f"{cohort} sites are present in another cohort")
    held_arrays = {arrays[parent] for parent in by_cohort["site_array_held_out"]}
    other_arrays = {
        arrays[parent] for parent in parents if assigned[parent] != "site_array_held_out"
    }
    if held_arrays & other_arrays:
        raise ValueError("site_array_held_out arrays are present in another cohort")

    component_cohort: dict[str, str] = {}
    for parent, cohort in assigned.items():
        component = registry.subject_to_component[parent]
        previous = component_cohort.setdefault(component, cohort)
        if previous != cohort:
            raise ValueError("a frozen independent component crosses benchmark cohorts")
    return assigned, sites, arrays


def materialize_benchmark_manifest(
    registry: SplitPolicyRegistry,
    declaration: Mapping[str, Any],
    datasets: Sequence[tuple[str, str, ChannelDataset]],
    *,
    registry_attestation: Mapping[str, str],
    declaration_attestation: Mapping[str, str],
) -> dict[str, Any]:
    """Build a deterministic index without duplicating physical parents or channel arrays.

    Dataset hashes must be calculated from the exact stable file bytes used
    to construct each ChannelDataset. The caller verifies both signatures.
    """
    assignments, sites, arrays = _validate_declarations(registry, declaration)
    if not datasets:
        raise ValueError("at least one ChannelDataset is required")
    names: set[str] = set()
    observed_parents: set[str] = set()
    dataset_records: list[dict[str, Any]] = []
    cohort_parents: dict[str, set[str]] = {cohort: set() for cohort in BENCHMARK_COHORTS}
    cohort_components: dict[str, set[str]] = {cohort: set() for cohort in BENCHMARK_COHORTS}
    cohort_rows: dict[str, int] = dict.fromkeys(BENCHMARK_COHORTS, 0)
    for name, digest, dataset in sorted(datasets, key=lambda item: item[0]):
        _identifier(name, "dataset name")
        _sha256(digest, "dataset digest")
        if name in names:
            raise ValueError(f"duplicate dataset name: {name}")
        names.add(name)
        rows: list[dict[str, Any]] = []
        for index, metadata in enumerate(dataset.metadata):
            if not isinstance(metadata, dict):
                raise ValueError(f"{name} row {index} metadata must be an object")
            parent = _identifier(metadata.get("split_subject_id"), "split_subject_id")
            if parent not in registry.subjects:
                raise ValueError(f"{name} row {index} references a parent absent from the lock")
            subject = registry.subjects[parent]
            if metadata.get("split_source_family") != subject.source_family:
                raise ValueError(f"{name} row {index} source family contradicts frozen parent")
            if metadata.get("split_parent_type") != subject.parent_type.value:
                raise ValueError(f"{name} row {index} parent type contradicts frozen parent")
            if (
                "split_partition_id" in metadata
                and metadata["split_partition_id"] != subject.partition_id
            ):
                raise ValueError(f"{name} row {index} partition identity contradicts lock")
            for relation in metadata.get("split_leakage_relations", []):
                if (
                    not isinstance(relation, dict)
                    or set(relation) != {"namespace", "identity"}
                    or (relation["namespace"], relation["identity"])
                    not in {(item.namespace.value, item.identity) for item in subject.relations}
                ):
                    raise ValueError(f"{name} row {index} leakage relation contradicts lock")
            cohort = assignments[parent]
            component = registry.subject_to_component[parent]
            observed_parents.add(parent)
            cohort_parents[cohort].add(parent)
            cohort_components[cohort].add(component)
            cohort_rows[cohort] += 1
            rows.append(
                {
                    "row_index": index,
                    "parent_id": parent,
                    "component_id": component,
                    "cohort": cohort,
                }
            )
        dataset_records.append(
            {
                "name": name,
                "file_sha256": digest,
                "row_count": len(dataset.metadata),
                "label_semantics": dataset.label_semantics,
                "rows": rows,
            }
        )
    if observed_parents != set(registry.subjects):
        raise ValueError("dataset roster must cover every frozen parent exactly")
    for label, attestation in (
        ("registry", registry_attestation),
        ("declaration", declaration_attestation),
    ):
        if not isinstance(attestation, Mapping) or set(attestation) != {
            "artifact_sha256",
            "signature_sha256",
            "allowed_signers_sha256",
            "signer_id",
            "namespace",
        }:
            raise ValueError(f"{label} attestation field roster is invalid")
        for field in ("artifact_sha256", "signature_sha256", "allowed_signers_sha256"):
            _sha256(attestation[field], f"{label} attestation {field}")
        _identifier(attestation["signer_id"], f"{label} signer")
        _identifier(attestation["namespace"], f"{label} namespace")
    if registry_attestation["namespace"] != "umcd-split-policy-v2@github.com":
        raise ValueError("registry attestation namespace is invalid")
    if declaration_attestation["namespace"] != "umcd-benchmark-cohort-v1@github.com":
        raise ValueError("cohort attestation namespace is invalid")
    summary = {
        cohort: {
            "parent_subjects": len(cohort_parents[cohort]),
            "independent_components": len(cohort_components[cohort]),
            "rows": cohort_rows[cohort],
            "site_count": len({sites[parent] for parent in cohort_parents[cohort]}),
            "array_count": len({arrays[parent] for parent in cohort_parents[cohort]}),
        }
        for cohort in BENCHMARK_COHORTS
    }
    core: dict[str, Any] = {
        "schema": "umcd.benchmark_partition_manifest.v1",
        "status": "CANDIDATE_NOT_SCIENTIFICALLY_QUALIFIED",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "registry_attestation": dict(registry_attestation),
        "declaration_attestation": dict(declaration_attestation),
        "cohorts": summary,
        "complete_six_cohort_coverage": all(
            summary[cohort]["rows"] > 0 for cohort in BENCHMARK_COHORTS
        ),
        "datasets": dataset_records,
    }
    canonical = json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {**core, "manifest_sha256": hashlib.sha256(canonical).hexdigest()}
