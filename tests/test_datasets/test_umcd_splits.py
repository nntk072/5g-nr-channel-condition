"""Leakage-safe episode-level UMCD split registry contracts."""

from __future__ import annotations

import json

import pytest
from datasets.splits import (
    audit_episode_split_registry,
    freeze_episode_split_registry,
    make_episode_split_registry,
    read_episode_split_registry,
)


def _records() -> list[dict[str, str]]:
    return [
        {
            "episode_id": f"episode-{index}",
            "scene_realization_id": scene,
            "trajectory_id": trajectory,
            "counterfactual_family_id": intervention,
        }
        for index, scene, trajectory, intervention in (
            (0, "site-a", "route-0", "cf-0"),
            (1, "site-a", "route-1", "cf-1"),
            (2, "site-b", "route-1", "cf-2"),
            (3, "site-c", "route-3", "cf-3"),
            (4, "site-c", "route-4", "cf-4"),
            (5, "site-d", "route-5", "cf-5"),
            (6, "site-e", "route-6", "cf-6"),
        )
    ]


_GROUP_FIELDS = ("scene_realization_id", "trajectory_id", "counterfactual_family_id")


def test_registry_is_deterministic_and_unions_transitive_leakage_groups():
    records = _records()
    registry = make_episode_split_registry(
        records, seed="umcd-paper-v1", group_fields=_GROUP_FIELDS
    )
    reversed_registry = make_episode_split_registry(
        list(reversed(records)), seed="umcd-paper-v1", group_fields=_GROUP_FIELDS
    )

    assert registry.to_dict() == reversed_registry.to_dict()
    assert registry.episode_to_component["episode-0"] == registry.episode_to_component["episode-2"]
    assert registry.split_for_episode("episode-0") == registry.split_for_episode("episode-2")
    assert registry.split_for_episode("episode-0") in {"train", "validation", "test"}
    audit_episode_split_registry(registry, records)


def test_registry_rejects_duplicate_or_incomplete_episode_identity():
    with pytest.raises(ValueError, match="duplicate episode ID"):
        make_episode_split_registry(
            [*_records(), _records()[0]], seed="v1", group_fields=_GROUP_FIELDS
        )
    incomplete = _records()
    incomplete[0]["trajectory_id"] = ""
    with pytest.raises(ValueError, match="canonical string"):
        make_episode_split_registry(incomplete, seed="v1", group_fields=_GROUP_FIELDS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id", 7),
        ("episode_id", " episode-0"),
        ("trajectory_id", 12),
        ("scene_realization_id", "site-a "),
    ],
)
def test_registry_rejects_coerced_or_padded_split_identities(field, value):
    records = _records()
    records[0][field] = value
    with pytest.raises(ValueError, match="canonical string"):
        make_episode_split_registry(records, seed="v1", group_fields=_GROUP_FIELDS)


def test_registry_audit_rejects_identity_aliases_instead_of_normalizing_them():
    records = _records()
    registry = make_episode_split_registry(records, seed="v1", group_fields=_GROUP_FIELDS)
    changed = [dict(record) for record in records]
    changed[0]["trajectory_id"] = " route-0"
    with pytest.raises(ValueError, match="canonical string"):
        audit_episode_split_registry(registry, changed)


def test_registry_audit_requires_exact_episodes_and_unchanged_groups():
    records = _records()
    registry = make_episode_split_registry(
        records, seed="umcd-paper-v1", group_fields=_GROUP_FIELDS
    )
    with pytest.raises(ValueError, match="exactly the frozen episode set"):
        audit_episode_split_registry(registry, records[:-1])
    changed = [dict(record) for record in records]
    changed[0]["trajectory_id"] = "route-changed"
    with pytest.raises(ValueError, match="group identity changed"):
        audit_episode_split_registry(registry, changed)
    with pytest.raises(ValueError, match="duplicate episode ID"):
        audit_episode_split_registry(registry, [*records, records[0]])


def test_registry_lock_roundtrips_and_refuses_rewrite_or_tampering(tmp_path):
    records = _records()
    registry = make_episode_split_registry(
        records, seed="umcd-paper-v1", group_fields=_GROUP_FIELDS
    )
    path = tmp_path / "splits" / "umcd-paper-v1.json"

    digest = freeze_episode_split_registry(path, registry)
    assert freeze_episode_split_registry(path, registry) == digest
    assert read_episode_split_registry(path).to_dict() == registry.to_dict()
    changed = make_episode_split_registry(records, seed="umcd-paper-v2", group_fields=_GROUP_FIELDS)
    with pytest.raises(FileExistsError, match="refusing to modify"):
        freeze_episode_split_registry(path, changed)

    payload = json.loads(path.read_text())
    payload["episodes"]["episode-0"]["split"] = "test"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash is invalid"):
        read_episode_split_registry(path)
