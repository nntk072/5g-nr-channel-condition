"""Split-policy registry, audits, and signed registry verification."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .policy import (
    _SPLIT_REGISTRY_V2_FIELDS,
    _SPLIT_REGISTRY_V3_FIELDS,
    MAX_SPLIT_POLICY_LOCK_BYTES,
    SPLIT_POLICY_SIGNATURE_NAMESPACE,
    LeakageNamespace,
    LeakageRelation,
    SpatiotemporalFootprint,
    SplitChildBinding,
    SplitParentType,
    SplitPolicy,
    SplitSubject,
    VerifiedOpenSSHArtifact,
    _canonical_json,
    _derive_registry_assignments,
    _fsync_directory,
    _read_immutable_lock_bytes,
    _reject_duplicate_json_keys,
    _require_canonical_identity,
    audit_spatiotemporal_leakage_relations,
    verify_openssh_signed_artifact,
)


def bind_spatiotemporal_audit_to_registry(
    registry: SplitPolicyRegistry,
    footprints: Sequence[SpatiotemporalFootprint],
    *,
    spatial_buffer_m: float,
) -> SplitPolicyRegistry:
    """Return a v3 registry lock that binds the complete footprint audit digest."""
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    audit = audit_spatiotemporal_leakage_relations(
        footprints,
        tuple(registry.subjects.values()),
        spatial_buffer_m=spatial_buffer_m,
    )
    if registry.spatiotemporal_audit is not None and dict(registry.spatiotemporal_audit) != audit:
        raise ValueError("frozen registry spatiotemporal audit cannot be replaced")
    return SplitPolicyRegistry(
        policy=registry.policy,
        subject_to_component=registry.subject_to_component,
        component_to_split=registry.component_to_split,
        subjects=registry.subjects,
        child_bindings=registry.child_bindings,
        spatiotemporal_audit=audit,
    )


def verify_registry_spatiotemporal_audit(
    registry: SplitPolicyRegistry,
    footprints: Sequence[SpatiotemporalFootprint],
    *,
    spatial_buffer_m: float,
) -> dict[str, Any]:
    """Recompute and compare the footprint inventory committed by a v3 registry."""
    if registry.spatiotemporal_audit is None:
        raise ValueError("registry has no frozen spatiotemporal footprint audit")
    observed = audit_spatiotemporal_leakage_relations(
        footprints,
        tuple(registry.subjects.values()),
        spatial_buffer_m=spatial_buffer_m,
    )
    if dict(registry.spatiotemporal_audit) != observed:
        raise ValueError("spatiotemporal footprint inventory differs from frozen registry audit")
    return observed


@dataclass(frozen=True, slots=True)
class SplitPolicyRegistry:
    """Frozen assignments and relation snapshots for parent subjects."""

    policy: SplitPolicy
    subject_to_component: Mapping[str, str]
    component_to_split: Mapping[str, str]
    subjects: Mapping[str, SplitSubject]
    child_bindings: Mapping[str, SplitChildBinding] = field(default_factory=dict)
    spatiotemporal_audit: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        subject_components = dict(self.subject_to_component)
        component_splits = dict(self.component_to_split)
        subjects = dict(self.subjects)
        children = dict(self.child_bindings)
        lineage_audit = (
            None if self.spatiotemporal_audit is None else dict(self.spatiotemporal_audit)
        )
        if not subject_components or set(subject_components) != set(subjects):
            raise ValueError("registry must assign every registered parent subject")
        expected_subject_components, expected_component_splits = _derive_registry_assignments(
            subjects, self.policy
        )
        if subject_components != expected_subject_components:
            raise ValueError("registry components do not match canonical relation reconstruction")
        if component_splits != expected_component_splits:
            raise ValueError(
                "registry assignments do not match deterministic policy reconstruction"
            )
        for key, binding in children.items():
            if not isinstance(binding, SplitChildBinding):
                raise TypeError("child_bindings must contain SplitChildBinding values")
            if key != binding.child_id:
                raise ValueError("child mapping keys must equal child IDs")
            if binding.child_id in subjects:
                raise ValueError(f"child ID collides with a parent subject ID: {binding.child_id}")
            if binding.parent_subject_id not in subjects:
                raise ValueError(
                    f"child references an unknown split parent: {binding.parent_subject_id}"
                )
        if lineage_audit is not None:
            expected_audit_keys = {
                "schema",
                "status",
                "spatial_buffer_m",
                "derivation_algorithm",
                "parent_count",
                "subject_ids_sha256",
                "spatially_linked_parent_count",
                "temporally_linked_parent_count",
                "footprints_sha256",
                "derived_relations_sha256",
                "audit_sha256",
            }
            audit_core = {
                key: value for key, value in lineage_audit.items() if key != "audit_sha256"
            }
            audit_digests = (
                lineage_audit.get("subject_ids_sha256"),
                lineage_audit.get("footprints_sha256"),
                lineage_audit.get("derived_relations_sha256"),
                lineage_audit.get("audit_sha256"),
            )
            digests_are_valid = all(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
                for digest in audit_digests
            )
            buffer_m = lineage_audit.get("spatial_buffer_m")
            counts_are_valid = all(
                isinstance(lineage_audit.get(key), int)
                and not isinstance(lineage_audit.get(key), bool)
                and lineage_audit[key] >= 0
                for key in (
                    "parent_count",
                    "spatially_linked_parent_count",
                    "temporally_linked_parent_count",
                )
            )
            if (
                set(lineage_audit) != expected_audit_keys
                or lineage_audit.get("schema") != "umcd.spatiotemporal_lineage_audit.v2"
                or lineage_audit.get("status") != "PASS_INPUT_SCOPED"
                or lineage_audit.get("derivation_algorithm") != "umcd.spatiotemporal-leakage.v2"
                or isinstance(buffer_m, bool)
                or not isinstance(buffer_m, Real)
                or not math.isfinite(float(buffer_m))
                or buffer_m < 0
                or not counts_are_valid
                or not digests_are_valid
                or lineage_audit.get("parent_count") != len(subjects)
                or lineage_audit.get("subject_ids_sha256")
                != hashlib.sha256(_canonical_json(sorted(subjects))).hexdigest()
                or hashlib.sha256(_canonical_json(audit_core)).hexdigest()
                != lineage_audit.get("audit_sha256")
            ):
                raise ValueError("registry spatiotemporal audit is invalid or roster-mismatched")
        object.__setattr__(self, "subject_to_component", MappingProxyType(subject_components))
        object.__setattr__(self, "component_to_split", MappingProxyType(component_splits))
        object.__setattr__(self, "subjects", MappingProxyType(subjects))
        object.__setattr__(self, "child_bindings", MappingProxyType(children))
        if lineage_audit is not None:
            object.__setattr__(self, "spatiotemporal_audit", MappingProxyType(lineage_audit))

    def split_for_parent(self, subject_id: str) -> str:
        try:
            return self.component_to_split[self.subject_to_component[subject_id]]
        except KeyError as exc:
            raise KeyError(f"parent subject is absent from frozen split: {subject_id}") from exc

    def split_for_child(self, child_id: str) -> str:
        """Resolve a registered child's split through its frozen parent binding."""
        try:
            binding = self.child_bindings[child_id]
        except KeyError as exc:
            raise KeyError(f"child is absent from frozen split registry: {child_id}") from exc
        return self.split_for_parent(binding.parent_subject_id)

    @property
    def independent_component_counts(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                split: sum(value == split for value in self.component_to_split.values())
                for split in self.policy.split_fractions
            }
        )

    def report(self) -> dict[str, Any]:
        """Report parent and independent-component counts overall and per source family."""
        children_by_split = {
            split: sum(self.split_for_child(child_id) == split for child_id in self.child_bindings)
            for split in self.policy.split_fractions
        }
        children_by_kind: dict[str, dict[str, int]] = {}
        for child_id, binding in self.child_bindings.items():
            counts = children_by_kind.setdefault(
                binding.child_kind, dict.fromkeys(self.policy.split_fractions, 0)
            )
            counts[self.split_for_child(child_id)] += 1
        result: dict[str, Any] = {
            "policy_sha256": self.policy.to_dict()["policy_sha256"],
            "splits": {},
            "child_bindings": {
                "total": len(self.child_bindings),
                "by_split": children_by_split,
                "by_kind": {
                    kind: dict(sorted(counts.items()))
                    for kind, counts in sorted(children_by_kind.items())
                },
            },
        }
        for split in self.policy.split_fractions:
            subject_ids = [
                subject_id
                for subject_id in self.subjects
                if self.split_for_parent(subject_id) == split
            ]
            by_family: dict[str, set[str]] = {}
            for subject_id in subject_ids:
                subject = self.subjects[subject_id]
                by_family.setdefault(subject.source_family, set()).add(
                    self.subject_to_component[subject_id]
                )
            result["splits"][split] = {
                "independent_components": self.independent_component_counts[split],
                "parent_subjects": len(subject_ids),
                "by_source_family": {
                    family: {
                        "independent_components": len(components),
                        "parent_subjects": sum(
                            self.subjects[item].source_family == family for item in subject_ids
                        ),
                    }
                    for family, components in sorted(by_family.items())
                },
            }
        return result

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": (
                "umcd.split_registry.v2"
                if self.spatiotemporal_audit is None
                else "umcd.split_registry.v3"
            ),
            "schema_version": 2 if self.spatiotemporal_audit is None else 3,
            "policy": self.policy.to_dict(),
            "subjects": {
                subject_id: {
                    **self.subjects[subject_id].to_dict(),
                    "component_id": self.subject_to_component[subject_id],
                    "split": self.split_for_parent(subject_id),
                }
                for subject_id in sorted(self.subjects)
            },
            "components": {
                component: {"split": split}
                for component, split in sorted(self.component_to_split.items())
            },
            "children": {
                child_id: {
                    **self.child_bindings[child_id].to_dict(),
                    "split": self.split_for_child(child_id),
                }
                for child_id in sorted(self.child_bindings)
            },
            "report": self.report(),
        }
        if self.spatiotemporal_audit is not None:
            core["spatiotemporal_audit"] = dict(self.spatiotemporal_audit)
        return {**core, "registry_sha256": hashlib.sha256(_canonical_json(core)).hexdigest()}


def make_split_policy_registry(
    subjects: Sequence[SplitSubject], *, policy: SplitPolicy
) -> SplitPolicyRegistry:
    """Union parents by sparse qualified relations and assign whole components."""
    if not isinstance(policy, SplitPolicy):
        raise TypeError("policy must be a SplitPolicy")
    subject_by_id: dict[str, SplitSubject] = {}
    for subject in subjects:
        if not isinstance(subject, SplitSubject):
            raise TypeError("subjects must contain SplitSubject values")
        if subject.subject_id in subject_by_id:
            raise ValueError(f"duplicate split subject ID: {subject.subject_id}")
        subject_by_id[subject.subject_id] = subject
    subject_to_component, component_to_split = _derive_registry_assignments(subject_by_id, policy)
    return SplitPolicyRegistry(
        policy=policy,
        subject_to_component=subject_to_component,
        component_to_split=component_to_split,
        subjects=subject_by_id,
    )


def bind_split_children(
    registry: SplitPolicyRegistry, bindings: Sequence[SplitChildBinding]
) -> SplitPolicyRegistry:
    """Return a registry with immutable child-to-parent bindings added.

    Repeating the exact same binding is idempotent. Reusing a child ID for a
    different parent or child kind is rejected.
    """
    children = dict(registry.child_bindings)
    observed: dict[str, SplitChildBinding] = {}
    for binding in bindings:
        if not isinstance(binding, SplitChildBinding):
            raise TypeError("bindings must contain SplitChildBinding values")
        previous_input = observed.setdefault(binding.child_id, binding)
        if previous_input != binding:
            raise ValueError(f"child has conflicting parent or kind: {binding.child_id}")
        previous = children.get(binding.child_id)
        if previous is not None and previous != binding:
            raise ValueError(f"frozen child binding cannot change: {binding.child_id}")
        children[binding.child_id] = binding
    return SplitPolicyRegistry(
        policy=registry.policy,
        subject_to_component=registry.subject_to_component,
        component_to_split=registry.component_to_split,
        subjects=registry.subjects,
        child_bindings=children,
        spatiotemporal_audit=registry.spatiotemporal_audit,
    )


def audit_split_policy_registry(
    registry: SplitPolicyRegistry,
    subjects: Sequence[SplitSubject],
    child_bindings: Sequence[SplitChildBinding] = (),
) -> None:
    """Require exact parent and child coverage and unchanged frozen lineage."""
    observed: dict[str, SplitSubject] = {}
    relation_components: dict[tuple[LeakageNamespace, str], str] = {}
    for subject in subjects:
        if subject.subject_id in observed:
            raise ValueError(f"duplicate split subject in audit: {subject.subject_id}")
        observed[subject.subject_id] = subject
        frozen = registry.subjects.get(subject.subject_id)
        if frozen is None:
            raise ValueError(f"subject is absent from frozen split registry: {subject.subject_id}")
        if subject != frozen:
            raise ValueError(f"frozen split lineage changed for subject {subject.subject_id}")
        component = registry.subject_to_component[subject.subject_id]
        for relation in subject.relations:
            key = (relation.namespace, relation.identity)
            previous = relation_components.setdefault(key, component)
            if previous != component:
                raise ValueError(
                    f"shared leakage relation crosses components: {relation.qualified_identity}"
                )
    if set(observed) != set(registry.subjects):
        raise ValueError("audit subjects do not cover exactly the frozen parent set")
    observed_children: dict[str, SplitChildBinding] = {}
    for binding in child_bindings:
        if binding.child_id in observed_children:
            raise ValueError(f"duplicate child binding in audit: {binding.child_id}")
        frozen_child = registry.child_bindings.get(binding.child_id)
        if frozen_child is None:
            raise ValueError(f"child is absent from frozen split registry: {binding.child_id}")
        if binding != frozen_child:
            raise ValueError(f"frozen parent binding changed for child {binding.child_id}")
        observed_children[binding.child_id] = binding
    if set(observed_children) != set(registry.child_bindings):
        raise ValueError("audit bindings do not cover exactly the frozen child set")


def audit_split_policy_parent_roster(
    registry: SplitPolicyRegistry, subjects: Sequence[SplitSubject]
) -> dict[str, Any]:
    """Verify exact frozen parent coverage without asserting child-data coverage."""
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    observed: dict[str, SplitSubject] = {}
    for subject in subjects:
        if not isinstance(subject, SplitSubject):
            raise TypeError("subjects must contain SplitSubject values")
        if subject.subject_id in observed:
            raise ValueError(f"duplicate split subject in audit: {subject.subject_id}")
        frozen = registry.subjects.get(subject.subject_id)
        if frozen is None:
            raise ValueError(f"subject is absent from frozen split registry: {subject.subject_id}")
        if subject != frozen:
            raise ValueError(f"frozen split lineage changed for subject {subject.subject_id}")
        observed[subject.subject_id] = subject
    if set(observed) != set(registry.subjects):
        raise ValueError("audit subjects do not cover exactly the frozen parent set")
    report: dict[str, Any] = {
        "schema": "umcd.split_parent_roster_audit.v1",
        "status": "PASS_INPUT_SCOPED",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "parent_count": len(observed),
        "parent_ids_sha256": hashlib.sha256(_canonical_json(sorted(observed))).hexdigest(),
    }
    report["audit_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def audit_split_policy_child_roster(
    registry: SplitPolicyRegistry,
    child_bindings: Sequence[SplitChildBinding],
    *,
    child_kind: str,
) -> dict[str, Any]:
    """Verify exact frozen child coverage for one artifact kind.

    This permits auditing a saved derived view against only the corresponding
    child class while the registry also contains other descendant kinds. The
    supplied bindings must exactly match every frozen child of ``child_kind``.
    """
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    child_kind = _require_canonical_identity(child_kind, "child_kind")
    expected = {
        child_id: binding
        for child_id, binding in registry.child_bindings.items()
        if binding.child_kind == child_kind
    }
    if not expected:
        raise ValueError(f"registry has no frozen children of kind {child_kind!r}")

    observed: dict[str, SplitChildBinding] = {}
    for binding in child_bindings:
        if not isinstance(binding, SplitChildBinding):
            raise TypeError("child_bindings must contain SplitChildBinding values")
        if binding.child_kind != child_kind:
            raise ValueError(
                f"child roster contains kind {binding.child_kind!r}; expected {child_kind!r}"
            )
        if binding.child_id in observed:
            raise ValueError(f"duplicate child binding in audit: {binding.child_id}")
        frozen = expected.get(binding.child_id)
        if frozen is None:
            raise ValueError(
                f"child is absent from frozen {child_kind!r} roster: {binding.child_id}"
            )
        if binding != frozen:
            raise ValueError(f"frozen parent binding changed for child {binding.child_id}")
        observed[binding.child_id] = binding
    if set(observed) != set(expected):
        raise ValueError(f"audit bindings do not cover exactly the frozen {child_kind!r} roster")

    report: dict[str, Any] = {
        "schema": "umcd.split_child_roster_audit.v1",
        "status": "PASS_INPUT_SCOPED",
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "child_kind": child_kind,
        "child_count": len(observed),
        "child_ids_sha256": hashlib.sha256(_canonical_json(sorted(observed))).hexdigest(),
        "bindings_sha256": hashlib.sha256(
            _canonical_json([observed[child_id].to_dict() for child_id in sorted(observed)])
        ).hexdigest(),
    }
    report["audit_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def audit_whole_scene_holdout(
    registry: SplitPolicyRegistry,
    scene_subjects: Mapping[str, Sequence[str]],
    *,
    expected_scene_ids: Sequence[str],
    expected_subject_ids: Sequence[str],
) -> dict[str, Any]:
    """Verify complete input-scoped scene membership stays within one split.

    The expected scene and parent rosters must come from a separately reviewed
    corpus inventory. This function verifies exact coverage and the frozen
    registry's assignments; it cannot establish that the supplied inventory
    represents the complete or legally/physically authoritative corpus.
    A parent may belong to multiple scenes, in which case the connected
    relation graph must place those scenes in the same split as well.
    """
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    if not isinstance(scene_subjects, Mapping):
        raise TypeError("scene_subjects must be a mapping of scene IDs to parent IDs")

    def validate_roster(values: Sequence[str], label: str) -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{label} must be a sequence of IDs")
        roster = tuple(values)
        for item in roster:
            _require_canonical_identity(item, f"{label} entry")
        if len(set(roster)) != len(roster):
            raise ValueError(f"{label} must not contain duplicate IDs")
        return roster

    expected_scenes = validate_roster(expected_scene_ids, "expected_scene_ids")
    expected_subjects = validate_roster(expected_subject_ids, "expected_subject_ids")
    if not expected_scenes or not expected_subjects:
        raise ValueError("expected scene and parent rosters must be non-empty")
    normalized_mapping: dict[str, tuple[str, ...]] = {}
    for scene_id, parent_ids in scene_subjects.items():
        normalized_scene_id = _require_canonical_identity(scene_id, "scene_subjects key")
        if normalized_scene_id in normalized_mapping:
            raise ValueError(f"duplicate scene ID: {normalized_scene_id}")
        normalized_mapping[normalized_scene_id] = validate_roster(
            parent_ids, f"scene_subjects[{normalized_scene_id!r}]"
        )
        if not normalized_mapping[normalized_scene_id]:
            raise ValueError(f"scene has no parent subjects: {normalized_scene_id}")
    if set(normalized_mapping) != set(expected_scenes):
        raise ValueError("scene membership does not match expected scene inventory")
    observed_subjects: set[str] = set()
    scene_component_ids: dict[str, str] = {}
    scene_splits: dict[str, str] = {}
    membership_count = 0
    for scene_id in sorted(expected_scenes):
        parent_ids = normalized_mapping[scene_id]
        membership_count += len(parent_ids)
        observed_subjects.update(parent_ids)
        component_ids: set[str] = set()
        splits: set[str] = set()
        required_relation = LeakageRelation(LeakageNamespace.SCENE_REALIZATION, scene_id)
        for parent_id in parent_ids:
            subject = registry.subjects.get(parent_id)
            if subject is None:
                raise ValueError(f"scene parent is absent from frozen registry: {parent_id}")
            if required_relation not in subject.relations:
                raise ValueError(f"scene parent lacks its scene-realization relation: {parent_id}")
            component_ids.add(registry.subject_to_component[parent_id])
            splits.add(registry.split_for_parent(parent_id))
        if len(component_ids) != 1 or len(splits) != 1:
            raise ValueError(f"whole-scene holdout is violated for scene: {scene_id}")
        scene_component_ids[scene_id] = next(iter(component_ids))
        scene_splits[scene_id] = next(iter(splits))

    if observed_subjects != set(expected_subjects):
        raise ValueError("scene memberships do not cover exactly the expected parent roster")
    registered_scene_subjects = {
        subject_id
        for subject_id, subject in registry.subjects.items()
        if any(
            relation.namespace is LeakageNamespace.SCENE_REALIZATION
            for relation in subject.relations
        )
    }
    omitted_scene_subjects = registered_scene_subjects - observed_subjects
    if omitted_scene_subjects:
        raise ValueError(
            "scene memberships omit frozen scene-lineage parents: "
            + ", ".join(sorted(omitted_scene_subjects))
        )
    registered_scene_ids = {
        relation.identity
        for subject in registry.subjects.values()
        for relation in subject.relations
        if relation.namespace is LeakageNamespace.SCENE_REALIZATION
    }
    if registered_scene_ids != set(expected_scenes):
        raise ValueError(
            "expected scene inventory does not cover exactly the frozen "
            "scene-realization identities"
        )
    report = {
        "status": "PASS_INPUT_SCOPED",
        "policy_sha256": registry.policy.to_dict()["policy_sha256"],
        "registry_sha256": registry.to_dict()["registry_sha256"],
        "expected_scene_ids_sha256": hashlib.sha256(
            _canonical_json(sorted(expected_scenes))
        ).hexdigest(),
        "expected_subject_ids_sha256": hashlib.sha256(
            _canonical_json(sorted(expected_subjects))
        ).hexdigest(),
        "scene_membership_sha256": hashlib.sha256(
            _canonical_json(
                {
                    scene_id: sorted(parent_ids)
                    for scene_id, parent_ids in sorted(normalized_mapping.items())
                }
            )
        ).hexdigest(),
        "scene_count": len(expected_scenes),
        "parent_subject_count": len(expected_subjects),
        "scene_membership_count": membership_count,
        "scene_component_ids": dict(sorted(scene_component_ids.items())),
        "scene_splits": dict(sorted(scene_splits.items())),
    }
    report["audit_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def freeze_split_policy_registry(path: str | Path, registry: SplitPolicyRegistry) -> str:
    """Atomically create a canonical immutable registry lock."""
    if not isinstance(registry, SplitPolicyRegistry):
        raise TypeError("registry must be a SplitPolicyRegistry")
    has_spatiotemporal_relations = any(
        relation.namespace
        in {
            LeakageNamespace.SPATIAL_BUFFER,
            LeakageNamespace.OVERLAPPING_TIME_WINDOW,
        }
        for subject in registry.subjects.values()
        for relation in subject.relations
    )
    if has_spatiotemporal_relations and registry.spatiotemporal_audit is None:
        raise ValueError(
            "cannot freeze spatial/time leakage relations without a bound footprint audit"
        )
    destination = Path(path)
    encoded = (json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > MAX_SPLIT_POLICY_LOCK_BYTES:
        raise ValueError(
            f"split-policy lock exceeds its {MAX_SPLIT_POLICY_LOCK_BYTES}-byte reader limit"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _read_immutable_lock_bytes(destination) != encoded:
            raise FileExistsError(f"refusing to modify existing split-policy lock: {destination}")
        return hashlib.sha256(encoded).hexdigest()
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
            temp_path = stream.name
        try:
            os.link(temp_path, destination)
            _fsync_directory(destination.parent)
        except FileExistsError:
            if _read_immutable_lock_bytes(destination) != encoded:
                raise FileExistsError(
                    f"refusing to modify existing split-policy lock: {destination}"
                )
    finally:
        if temp_path is not None:
            try:
                Path(temp_path).unlink(missing_ok=True)
            finally:
                _fsync_directory(destination.parent)
    return hashlib.sha256(encoded).hexdigest()


def read_split_policy_registry(path: str | Path) -> SplitPolicyRegistry:
    """Read a split lock and independently reconstruct its frozen assignment."""
    try:
        payload = json.loads(
            _read_immutable_lock_bytes(Path(path)).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("UMCD split registry is not valid bounded UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("UMCD split registry root must be a JSON object")
    schema = payload.get("schema")
    schema_version = payload.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("UMCD split registry schema_version must be an integer")
    if (schema, schema_version) == ("umcd.split_registry.v2", 2):
        expected_fields = _SPLIT_REGISTRY_V2_FIELDS
    elif (schema, schema_version) == ("umcd.split_registry.v3", 3):
        expected_fields = _SPLIT_REGISTRY_V3_FIELDS
    else:
        raise ValueError("unsupported UMCD split registry schema")
    if set(payload) != expected_fields:
        missing = sorted(expected_fields - set(payload))
        extra = sorted(set(payload) - expected_fields)
        raise ValueError(
            f"UMCD split registry field roster is invalid; missing={missing}, extra={extra}"
        )
    for field_name in ("policy", "subjects", "components", "children", "report"):
        if not isinstance(payload[field_name], dict):
            raise ValueError(f"UMCD split registry {field_name} must be a JSON object")
    if schema_version == 3 and not isinstance(payload["spatiotemporal_audit"], dict):
        raise ValueError("UMCD split registry spatiotemporal_audit must be a JSON object")
    core = {key: value for key, value in payload.items() if key != "registry_sha256"}
    if hashlib.sha256(_canonical_json(core)).hexdigest() != payload.get("registry_sha256"):
        raise ValueError("UMCD split registry hash is invalid")
    policy = SplitPolicy.from_dict(payload["policy"])
    subjects: dict[str, SplitSubject] = {}
    subject_to_component: dict[str, str] = {}
    for subject_id, item in payload["subjects"].items():
        subjects[subject_id] = SplitSubject(
            subject_id=subject_id,
            parent_type=SplitParentType(item["parent_type"]),
            source_family=item["source_family"],
            relations=tuple(
                LeakageRelation(LeakageNamespace(rel["namespace"]), rel["identity"])
                for rel in item["relations"]
            ),
            partition_id=item.get("partition_id", subject_id),
        )
        subject_to_component[subject_id] = item["component_id"]
    component_to_split = {
        component: item["split"] for component, item in payload["components"].items()
    }
    children = {
        child_id: SplitChildBinding(
            child_id=child_id,
            parent_subject_id=item["parent_subject_id"],
            child_kind=item["child_kind"],
        )
        for child_id, item in payload.get("children", {}).items()
    }
    registry = SplitPolicyRegistry(
        policy,
        subject_to_component,
        component_to_split,
        subjects,
        children,
        payload.get("spatiotemporal_audit"),
    )
    if registry.to_dict() != payload:
        raise ValueError("UMCD split registry representation is not canonical")
    return registry


def verify_signed_split_policy_registry_attestation(
    path: str | Path,
    signature_path: str | Path,
    *,
    trusted_allowed_signers_path: str | Path,
    signer_id: str,
) -> tuple[SplitPolicyRegistry, VerifiedOpenSSHArtifact]:
    """Verify a split lock and return its exact artifact/trust attestation.

    The lock, detached signature, and externally governed OpenSSH
    ``allowed_signers`` trust file are snapshotted as single-link regular files
    before verification. The trusted keyring must be supplied independently
    of the release bundle. Signatures use the ``umcd-split-policy-v2`` SSH
    namespace, separating them from signatures for other purposes.
    """
    signer_id = _require_canonical_identity(signer_id, "signer_id")
    if any(character in signer_id for character in ("\n", "\r", "\x00")):
        raise ValueError("signer_id must be a single-line principal")

    verified = verify_openssh_signed_artifact(
        path,
        signature_path,
        trusted_allowed_signers_path=trusted_allowed_signers_path,
        signer_id=signer_id,
        namespace=SPLIT_POLICY_SIGNATURE_NAMESPACE,
    )
    with tempfile.TemporaryDirectory(prefix="umcd-split-policy-verify-") as temporary_directory:
        lock_snapshot = Path(temporary_directory) / "split-registry.json"
        lock_snapshot.write_bytes(verified.payload)
        return read_split_policy_registry(lock_snapshot), verified


def verify_signed_split_policy_registry(
    path: str | Path,
    signature_path: str | Path,
    *,
    trusted_allowed_signers_path: str | Path,
    signer_id: str,
) -> SplitPolicyRegistry:
    """Verify the detached signature and return the frozen split registry."""
    registry, _ = verify_signed_split_policy_registry_attestation(
        path,
        signature_path,
        trusted_allowed_signers_path=trusted_allowed_signers_path,
        signer_id=signer_id,
    )
    return registry
