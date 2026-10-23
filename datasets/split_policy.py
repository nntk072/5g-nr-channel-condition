"""Sparse, versioned leakage relations and immutable UMCD release splits.

The split subjects in this module are physical/statistical parents, not model
views. Network episode descendants and standardized draw observations inherit
their parent's assignment through ``split_for_child``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from evaluation.independent_splits import make_independent_split

from datasets.network import NetworkEpisode
from datasets.standardized import StandardizedChannelRecord

_NAMESPACE_ORDER = (
    "asset_family",
    "site",
    "scene_realization",
    "trajectory_session",
    "capture_session",
    "counterfactual_family",
    "material_realization",
    "equipment_family",
)
_SPLIT_ORDER = ("train", "validation", "calibration", "test")
_DEFAULT_FOUR_WAY = {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class LeakageNamespace(StrEnum):
    """Field-qualified identities that induce conservative component linkage."""

    ASSET_FAMILY = "asset_family"
    SITE = "site"
    SCENE_REALIZATION = "scene_realization"
    TRAJECTORY_SESSION = "trajectory_session"
    CAPTURE_SESSION = "capture_session"
    COUNTERFACTUAL_FAMILY = "counterfactual_family"
    MATERIAL_REALIZATION = "material_realization"
    EQUIPMENT_FAMILY = "equipment_family"


class SplitParentType(StrEnum):
    """Supported statistical parent units in Release Split Policy v2."""

    NETWORK_EPISODE = "network_episode"
    STANDARDIZED_CHANNEL_DRAW = "standardized_channel_draw"


@dataclass(frozen=True, slots=True, order=True)
class LeakageRelation:
    """One sparse, namespace-qualified relation; equal IDs in different fields differ."""

    namespace: LeakageNamespace
    identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, LeakageNamespace):
            try:
                object.__setattr__(self, "namespace", LeakageNamespace(self.namespace))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported leakage namespace: {self.namespace!r}") from exc
        if not isinstance(self.identity, str) or not self.identity.strip():
            raise ValueError("leakage relation identity must be a non-empty string")
        object.__setattr__(self, "identity", self.identity.strip())

    @property
    def qualified_identity(self) -> str:
        """Human-readable field-qualified value for reports."""
        return f"{self.namespace.value}:{self.identity}"

    def to_dict(self) -> dict[str, str]:
        return {"namespace": self.namespace.value, "identity": self.identity}


@dataclass(frozen=True, slots=True)
class SplitSubject:
    """A split parent with only the leakage relations that actually apply."""

    subject_id: str
    parent_type: SplitParentType
    source_family: str
    relations: tuple[LeakageRelation, ...] = ()
    partition_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("subject_id", "source_family"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())
        if not isinstance(self.parent_type, SplitParentType):
            try:
                object.__setattr__(self, "parent_type", SplitParentType(self.parent_type))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported split parent type: {self.parent_type!r}") from exc
        relations = tuple(self.relations)
        if any(not isinstance(item, LeakageRelation) for item in relations):
            raise TypeError("relations must contain LeakageRelation values")
        keys = [(item.namespace, item.identity) for item in relations]
        if len(set(keys)) != len(keys):
            raise ValueError("a split subject cannot repeat a leakage relation")
        object.__setattr__(self, "relations", tuple(sorted(relations)))
        partition_id = self.subject_id if self.partition_id is None else self.partition_id
        if not isinstance(partition_id, str) or not partition_id.strip():
            raise ValueError("partition_id must be a non-empty opaque identity")
        object.__setattr__(self, "partition_id", partition_id.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "parent_type": self.parent_type.value,
            "source_family": self.source_family,
            "partition_id": self.partition_id,
            "relations": [item.to_dict() for item in self.relations],
        }


def network_episode_subject(
    episode: NetworkEpisode,
    *,
    relations: Sequence[LeakageRelation] = (),
) -> SplitSubject:
    """Adapt a network episode; its legacy split group remains a family relation.

    Additional site, scene, trajectory, equipment, or material lineage must be
    supplied explicitly by the caller. ``family_id`` is reporting metadata and
    does not itself join episodes.
    """
    if not isinstance(episode, NetworkEpisode):
        raise TypeError("episode must be a NetworkEpisode")
    return SplitSubject(
        subject_id=episode.episode_id,
        parent_type=SplitParentType.NETWORK_EPISODE,
        source_family=episode.family_id,
        relations=(
            LeakageRelation(LeakageNamespace.ASSET_FAMILY, episode.split_group_id),
            *tuple(relations),
        ),
        partition_id=episode.episode_id,
    )


def standardized_draw_subject(
    record: StandardizedChannelRecord,
    *,
    relations: Sequence[LeakageRelation] = (),
    partition_id: str | None = None,
) -> SplitSubject:
    """Adapt one draw without ranking on profile/category/target metadata.

    The default opaque key is derived from the generation seed and sample
    index, never profile, target, or source-family labels. If independent
    generator streams reuse that seed/index pair, callers must provide a
    stable opaque ``partition_id`` for each draw; registry construction rejects
    collisions instead of silently joining unrelated parents.
    """
    if not isinstance(record, StandardizedChannelRecord):
        raise TypeError("record must be a StandardizedChannelRecord")
    if partition_id is None:
        stored_identity = record.metadata.get("partition_id")
        if isinstance(stored_identity, str) and stored_identity.strip():
            partition_id = stored_identity.strip()
        else:
            seed = record.metadata.get("seed")
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError(
                    "standardized draw needs generation seed metadata or an explicit partition_id"
                )
            opaque_payload = _canonical_json(
                {"draw_seed": seed, "sample_index": record.sample_index}
            )
            partition_id = "standardized-draw-" + hashlib.sha256(opaque_payload).hexdigest()
    return SplitSubject(
        subject_id=record.sample_id,
        parent_type=SplitParentType.STANDARDIZED_CHANNEL_DRAW,
        source_family=f"standardized.{record.family.value}",
        relations=tuple(relations),
        partition_id=partition_id,
    )


@dataclass(frozen=True, slots=True)
class SplitPolicy:
    """Immutable assignment contract; its digest changes with every policy choice."""

    seed: str
    split_fractions: Mapping[str, float]
    relation_namespaces: tuple[LeakageNamespace, ...] = tuple(LeakageNamespace)
    policy_version: str = "umcd.release_split.v2"
    component_algorithm: str = "sha256-connected-component-v2"

    def __post_init__(self) -> None:
        if not isinstance(self.seed, str) or not self.seed.strip():
            raise ValueError("policy seed must be a non-empty string")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be a non-empty string")
        if self.component_algorithm != "sha256-connected-component-v2":
            raise ValueError("unsupported component assignment algorithm")
        fractions = dict(self.split_fractions)
        if set(fractions) not in (
            {"train", "validation", "test"},
            {"train", "validation", "calibration", "test"},
        ):
            raise ValueError("split fractions must define the three-way or four-way split set")
        ordered_fractions = {
            name: float(fractions[name]) for name in _SPLIT_ORDER if name in fractions
        }
        if any(not 0 < value < 1 for value in ordered_fractions.values()):
            raise ValueError("split fractions must be finite and positive")
        if any(not math.isfinite(value) for value in ordered_fractions.values()):
            raise ValueError("split fractions must be finite and positive")
        if abs(sum(ordered_fractions.values()) - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to one")
        namespaces = tuple(
            item if isinstance(item, LeakageNamespace) else LeakageNamespace(item)
            for item in self.relation_namespaces
        )
        if not namespaces or len(set(namespaces)) != len(namespaces):
            raise ValueError("relation_namespaces must be non-empty and unique")
        namespaces = tuple(sorted(namespaces, key=lambda item: _NAMESPACE_ORDER.index(item.value)))
        object.__setattr__(self, "seed", self.seed.strip())
        object.__setattr__(self, "policy_version", self.policy_version.strip())
        object.__setattr__(self, "split_fractions", MappingProxyType(ordered_fractions))
        object.__setattr__(self, "relation_namespaces", namespaces)

    @classmethod
    def release_v2(cls, *, seed: str) -> SplitPolicy:
        """Construct the four-way default release policy."""
        return cls(seed=seed, split_fractions=_DEFAULT_FOUR_WAY)

    def to_dict(self) -> dict[str, Any]:
        core = {
            "schema": "umcd.split_policy.v2",
            "schema_version": 2,
            "policy_version": self.policy_version,
            "seed": self.seed,
            "split_fractions": dict(self.split_fractions),
            "relation_namespaces": [item.value for item in self.relation_namespaces],
            "component_algorithm": self.component_algorithm,
        }
        return {**core, "policy_sha256": hashlib.sha256(_canonical_json(core)).hexdigest()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SplitPolicy:
        if value.get("schema") != "umcd.split_policy.v2" or value.get("schema_version") != 2:
            raise ValueError("unsupported UMCD split policy schema")
        core = {key: item for key, item in value.items() if key != "policy_sha256"}
        if hashlib.sha256(_canonical_json(core)).hexdigest() != value.get("policy_sha256"):
            raise ValueError("UMCD split policy hash is invalid")
        policy = cls(
            seed=value["seed"],
            split_fractions=value["split_fractions"],
            relation_namespaces=tuple(value["relation_namespaces"]),
            policy_version=value["policy_version"],
            component_algorithm=value["component_algorithm"],
        )
        if policy.to_dict() != dict(value):
            raise ValueError("UMCD split policy representation is not canonical")
        return policy


@dataclass(frozen=True, slots=True)
class SplitChildBinding:
    """A persisted release child linked to exactly one frozen parent."""

    child_id: str
    parent_subject_id: str
    child_kind: str

    def __post_init__(self) -> None:
        for name in ("child_id", "parent_subject_id", "child_kind"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())

    def to_dict(self) -> dict[str, str]:
        return {
            "parent_subject_id": self.parent_subject_id,
            "child_kind": self.child_kind,
        }


def _derive_registry_assignments(
    subjects: Mapping[str, SplitSubject], policy: SplitPolicy
) -> tuple[dict[str, str], dict[str, str]]:
    """Rebuild canonical components and assignments from frozen inputs only."""
    if not isinstance(policy, SplitPolicy):
        raise TypeError("policy must be a SplitPolicy")
    if not subjects:
        raise ValueError("at least one split subject is required")
    subject_by_id = dict(subjects)
    if any(not isinstance(subject, SplitSubject) for subject in subject_by_id.values()):
        raise TypeError("subjects must contain SplitSubject values")
    if any(key != subject.subject_id for key, subject in subject_by_id.items()):
        raise ValueError("subject mapping keys must equal subject IDs")
    if len({subject.partition_id for subject in subject_by_id.values()}) != len(subject_by_id):
        raise ValueError("partition_id values must uniquely identify parent subjects")
    for subject in subject_by_id.values():
        unsupported = set(item.namespace for item in subject.relations) - set(
            policy.relation_namespaces
        )
        if unsupported:
            raise ValueError(f"subject has relations excluded by policy: {sorted(unsupported)}")

    parent = {subject_id: subject_id for subject_id in subject_by_id}

    def find(subject_id: str) -> str:
        root = subject_id
        while parent[root] != root:
            parent[root] = parent[parent[root]]
            root = parent[root]
        while parent[subject_id] != subject_id:
            next_id = parent[subject_id]
            parent[subject_id] = root
            subject_id = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_subject_for_relation: dict[tuple[LeakageNamespace, str], str] = {}
    for subject_id in sorted(subject_by_id):
        for relation in subject_by_id[subject_id].relations:
            key = (relation.namespace, relation.identity)
            previous = first_subject_for_relation.setdefault(key, subject_id)
            union(subject_id, previous)

    members: dict[str, list[str]] = {}
    for subject_id in sorted(subject_by_id):
        members.setdefault(find(subject_id), []).append(subject_id)

    subject_to_component: dict[str, str] = {}
    for component_members in members.values():
        partition_ids = sorted(
            str(subject_by_id[subject_id].partition_id) for subject_id in component_members
        )
        component_id = "component-" + hashlib.sha256(_canonical_json(partition_ids)).hexdigest()
        for subject_id in component_members:
            subject_to_component[subject_id] = component_id

    component_ids = sorted(set(subject_to_component.values()))
    independent = make_independent_split(
        [
            {"sample_id": component_id, "component_id": component_id}
            for component_id in component_ids
        ],
        seed=policy.seed,
        group_field="component_id",
        fractions=policy.split_fractions,
    )
    if set(independent.group_to_split.values()) != set(policy.split_fractions):
        raise ValueError("deterministic assignment omitted a required split partition")
    return subject_to_component, dict(independent.group_to_split)


@dataclass(frozen=True, slots=True)
class SplitPolicyRegistry:
    """Frozen assignments and relation snapshots for parent subjects."""

    policy: SplitPolicy
    subject_to_component: Mapping[str, str]
    component_to_split: Mapping[str, str]
    subjects: Mapping[str, SplitSubject]
    child_bindings: Mapping[str, SplitChildBinding] = field(default_factory=dict)

    def __post_init__(self) -> None:
        subject_components = dict(self.subject_to_component)
        component_splits = dict(self.component_to_split)
        subjects = dict(self.subjects)
        children = dict(self.child_bindings)
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
            if binding.parent_subject_id not in subjects:
                raise ValueError(
                    f"child references an unknown split parent: {binding.parent_subject_id}"
                )
        object.__setattr__(self, "subject_to_component", MappingProxyType(subject_components))
        object.__setattr__(self, "component_to_split", MappingProxyType(component_splits))
        object.__setattr__(self, "subjects", MappingProxyType(subjects))
        object.__setattr__(self, "child_bindings", MappingProxyType(children))

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
            "schema": "umcd.split_registry.v2",
            "schema_version": 2,
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


def freeze_split_policy_registry(path: str | Path, registry: SplitPolicyRegistry) -> str:
    """Atomically create a canonical immutable registry lock."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    if destination.exists():
        if destination.read_bytes() != encoded:
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
        except FileExistsError:
            if destination.read_bytes() != encoded:
                raise FileExistsError(
                    f"refusing to modify existing split-policy lock: {destination}"
                )
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def read_split_policy_registry(path: str | Path) -> SplitPolicyRegistry:
    """Read a v2 lock and independently reconstruct its frozen assignment."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "umcd.split_registry.v2" or payload.get("schema_version") != 2:
        raise ValueError("unsupported UMCD split registry schema")
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
    )
    if registry.to_dict() != payload:
        raise ValueError("UMCD split registry representation is not canonical")
    return registry
