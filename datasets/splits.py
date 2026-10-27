"""Immutable parent-episode split registry for leakage-safe UMCD views."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from evaluation.independent_splits import make_independent_split


def _stable_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _require_canonical_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


@dataclass(frozen=True, slots=True)
class EpisodeSplitRegistry:
    """Frozen split assignment for parent episodes and their connected groups.

    Each episode records all configured leakage keys. Episodes sharing any
    field-qualified key are joined into one connected component, so scene,
    route, capture, or counterfactual descendants cannot cross split borders.
    Child links, frames, and receiver observations inherit the parent episode's
    assignment through ``split_for_episode``.
    """

    seed: str
    group_fields: tuple[str, ...]
    episode_to_component: Mapping[str, str]
    component_to_split: Mapping[str, str]
    episode_group_keys: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        seed = _require_canonical_id(self.seed, "seed")
        fields = tuple(self.group_fields)
        components = dict(self.episode_to_component)
        splits = dict(self.component_to_split)
        keys = {episode: tuple(values) for episode, values in self.episode_group_keys.items()}
        if not fields or len(set(fields)) != len(fields):
            raise ValueError("seed and unique group_fields must be non-empty")
        for field in fields:
            _require_canonical_id(field, "group field")
        if not components or set(components) != set(keys):
            raise ValueError("every registered episode must have group keys and a component")
        if set(components.values()) != set(splits):
            raise ValueError("component assignments must cover every registered component")
        if set(splits.values()) - {"train", "validation", "test"}:
            raise ValueError("split names must be train, validation, or test")
        for episode in components:
            _require_canonical_id(episode, "episode ID")
        for component in set(components.values()) | set(splits):
            _require_canonical_id(component, "component ID")
        expected_key_count = len(fields)
        if any(len(value) != expected_key_count for value in keys.values()):
            raise ValueError("each episode must preserve one value for every group field")
        for value in keys.values():
            for item in value:
                _require_canonical_id(item, "episode group-key value")
        key_to_components: dict[tuple[str, str], set[str]] = {}
        for episode, values in keys.items():
            component = components[episode]
            for field, value in zip(fields, values, strict=True):
                key_to_components.setdefault((field, value), set()).add(component)
        if any(len(component_ids) != 1 for component_ids in key_to_components.values()):
            raise ValueError("episodes sharing a leakage key must share one connected component")
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "group_fields", fields)
        object.__setattr__(self, "episode_to_component", MappingProxyType(components))
        object.__setattr__(self, "component_to_split", MappingProxyType(splits))
        object.__setattr__(
            self,
            "episode_group_keys",
            MappingProxyType({episode: keys[episode] for episode in sorted(keys)}),
        )

    @property
    def episode_to_split(self) -> Mapping[str, str]:
        """Read-only split assignment for each parent episode."""
        return MappingProxyType(
            {
                episode: self.component_to_split[component]
                for episode, component in self.episode_to_component.items()
            }
        )

    def split_for_episode(self, episode_id: str) -> str:
        """Return the frozen assignment inherited by all child views."""
        try:
            component = self.episode_to_component[episode_id]
        except KeyError as exc:
            raise KeyError(
                f"episode is absent from the frozen split registry: {episode_id}"
            ) from exc
        return self.component_to_split[component]

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical hashable registry document."""
        core = {
            "schema": "umcd.split_registry.v1",
            "schema_version": 1,
            "seed": self.seed,
            "group_fields": list(self.group_fields),
            "episodes": {
                episode: {
                    "component_id": self.episode_to_component[episode],
                    "split": self.split_for_episode(episode),
                    "group_keys": list(self.episode_group_keys[episode]),
                }
                for episode in sorted(self.episode_to_component)
            },
            "components": {
                component: {"split": split}
                for component, split in sorted(self.component_to_split.items())
            },
        }
        return {**core, "registry_sha256": hashlib.sha256(_stable_json(core)).hexdigest()}


def make_episode_split_registry(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    group_fields: Sequence[str],
    episode_id_field: str = "episode_id",
    fractions: Mapping[str, float] | None = None,
) -> EpisodeSplitRegistry:
    """Freeze parent-episode assignments using all declared leakage relations.

    A connected-component policy is intentionally conservative: if episodes
    share any configured scene, route, capture, intervention, or other group
    identity, the entire connected component stays in one split.
    """
    fields = tuple(group_fields)
    _require_canonical_id(seed, "seed")
    if not records or not fields or len(set(fields)) != len(fields):
        raise ValueError("records, seed, and unique group_fields must be non-empty")
    for field in fields:
        _require_canonical_id(field, "group field")
    episode_keys: dict[str, tuple[str, ...]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each episode split record must be a mapping")
        episode_id = _require_canonical_id(record.get(episode_id_field), episode_id_field)
        if episode_id in episode_keys:
            raise ValueError(f"duplicate episode ID: {episode_id}")
        values = tuple(_require_canonical_id(record.get(field), field) for field in fields)
        episode_keys[episode_id] = values

    parents = {episode: episode for episode in episode_keys}

    def find(episode: str) -> str:
        parent = parents[episode]
        while parent != parents[parent]:
            parents[parent] = parents[parents[parent]]
            parent = parents[parent]
        parents[episode] = parent
        return parent

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    first_for_key: dict[tuple[str, str], str] = {}
    for episode, values in episode_keys.items():
        for field, value in zip(fields, values, strict=True):
            key = (field, value)
            if key in first_for_key:
                union(episode, first_for_key[key])
            else:
                first_for_key[key] = episode

    members: dict[str, list[str]] = {}
    for episode in sorted(episode_keys):
        members.setdefault(find(episode), []).append(episode)
    episode_to_component: dict[str, str] = {}
    for group_members in members.values():
        component_id = "group-" + hashlib.sha256(_stable_json(group_members)).hexdigest()
        episode_to_component.update({episode: component_id for episode in group_members})
    split = make_independent_split(
        [
            {"sample_id": episode, "scene_group_id": episode_to_component[episode]}
            for episode in sorted(episode_keys)
        ],
        seed=seed,
        group_field="scene_group_id",
        fractions=fractions,
    )
    return EpisodeSplitRegistry(
        seed=seed,
        group_fields=fields,
        episode_to_component=episode_to_component,
        component_to_split=split.group_to_split,
        episode_group_keys=episode_keys,
    )


def audit_episode_split_registry(
    registry: EpisodeSplitRegistry,
    records: Sequence[Mapping[str, Any]],
    *,
    episode_id_field: str = "episode_id",
) -> None:
    """Require exact frozen episode coverage and unchanged leakage identities."""
    seen: set[str] = set()
    first_component_for_key: dict[tuple[str, str], str] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("each episode split audit record must be a mapping")
        episode = _require_canonical_id(record.get(episode_id_field), episode_id_field)
        if episode in seen:
            raise ValueError(f"duplicate episode ID in split audit: {episode}")
        seen.add(episode)
        if episode not in registry.episode_to_component:
            raise ValueError(f"episode is absent from frozen split registry: {episode}")
        values = tuple(
            _require_canonical_id(record.get(field), field) for field in registry.group_fields
        )
        if values != registry.episode_group_keys[episode]:
            raise ValueError(f"group identity changed for episode {episode}")
        component = registry.episode_to_component[episode]
        for field, value in zip(registry.group_fields, values, strict=True):
            key = (field, value)
            previous = first_component_for_key.setdefault(key, component)
            if previous != component:
                raise ValueError(f"leakage group crosses split components: {field}={value}")
    if seen != set(registry.episode_to_component):
        raise ValueError("audit records do not cover exactly the frozen episode set")


def freeze_episode_split_registry(path: str | Path, registry: EpisodeSplitRegistry) -> str:
    """Atomically create an immutable registry lock or verify an identical one."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(registry.to_dict(), indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if destination.exists():
        if destination.read_bytes() != encoded:
            raise FileExistsError(f"refusing to modify existing split registry: {destination}")
        return digest
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
                raise FileExistsError(f"refusing to modify existing split registry: {destination}")
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
    return digest


def read_episode_split_registry(path: str | Path) -> EpisodeSplitRegistry:
    """Load and checksum an immutable UMCD episode split registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "umcd.split_registry.v1" or payload.get("schema_version") != 1:
        raise ValueError("unsupported UMCD split registry schema")
    core = {key: value for key, value in payload.items() if key != "registry_sha256"}
    digest = hashlib.sha256(_stable_json(core)).hexdigest()
    if digest != payload.get("registry_sha256"):
        raise ValueError("UMCD split registry hash is invalid")
    episodes = payload["episodes"]
    components = payload["components"]
    registry = EpisodeSplitRegistry(
        seed=payload["seed"],
        group_fields=tuple(payload["group_fields"]),
        episode_to_component={episode: item["component_id"] for episode, item in episodes.items()},
        component_to_split={component: item["split"] for component, item in components.items()},
        episode_group_keys={
            episode: tuple(item["group_keys"]) for episode, item in episodes.items()
        },
    )
    if registry.to_dict() != payload:
        raise ValueError("UMCD split registry representation is not canonical")
    return registry
