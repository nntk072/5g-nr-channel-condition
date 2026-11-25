"""Validation and immutable-file guards for UMCD parent split registries."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from datasets import splits


def _registry():
    return splits.EpisodeSplitRegistry(
        seed="seed",
        group_fields=("scene",),
        episode_to_component={"episode-a": "component-a"},
        component_to_split={"component-a": "train"},
        episode_group_keys={"episode-a": ("scene-a",)},
    )


@pytest.mark.parametrize(
    ("fields", "components", "assignments", "keys", "message"),
    [
        (
            (),
            {"episode-a": "component-a"},
            {"component-a": "train"},
            {"episode-a": ()},
            "group_fields",
        ),
        (
            ("scene", "scene"),
            {"episode-a": "component-a"},
            {"component-a": "train"},
            {"episode-a": ("scene-a", "scene-a")},
            "group_fields",
        ),
        (
            ("scene",),
            {"episode-a": "component-a"},
            {"component-a": "train"},
            {},
            "group keys and a component",
        ),
        (
            ("scene",),
            {"episode-a": "component-a"},
            {},
            {"episode-a": ("scene-a",)},
            "cover every registered component",
        ),
        (
            ("scene",),
            {"episode-a": "component-a"},
            {"component-a": "development"},
            {"episode-a": ("scene-a",)},
            "split names",
        ),
        (
            ("scene",),
            {"episode-a": "component-a"},
            {"component-a": "train"},
            {"episode-a": ("scene-a", "route-a")},
            "one value for every group field",
        ),
        (
            ("scene",),
            {"episode-a": "component-a", "episode-b": "component-b"},
            {"component-a": "train", "component-b": "test"},
            {"episode-a": ("scene-shared",), "episode-b": ("scene-shared",)},
            "sharing a leakage key",
        ),
    ],
)
def test_registry_constructor_rejects_inconsistent_assignments(
    fields, components, assignments, keys, message
):
    with pytest.raises(ValueError, match=message):
        splits.EpisodeSplitRegistry("seed", fields, components, assignments, keys)


def test_registry_lookup_and_audit_reject_unknown_or_changed_records():
    registry = _registry()
    assert registry.episode_to_split["episode-a"] == "train"
    with pytest.raises(KeyError, match="absent from the frozen split registry"):
        registry.split_for_episode("unknown")
    with pytest.raises(TypeError, match="each episode split audit record must be a mapping"):
        splits.audit_episode_split_registry(registry, [None])
    with pytest.raises(ValueError, match="group identity changed"):
        splits.audit_episode_split_registry(
            registry, [{"episode_id": "episode-a", "scene": "changed"}]
        )
    with pytest.raises(ValueError, match="absent from frozen split registry"):
        splits.audit_episode_split_registry(
            registry, [{"episode_id": "unknown", "scene": "scene-a"}]
        )
    with pytest.raises(ValueError, match="exactly the frozen episode set"):
        splits.audit_episode_split_registry(registry, [])


def test_split_registry_builder_rejects_empty_bad_and_duplicate_records():
    with pytest.raises(ValueError, match="must be non-empty"):
        splits.make_episode_split_registry([], seed="seed", group_fields=("scene",))
    with pytest.raises(TypeError, match="record must be a mapping"):
        splits.make_episode_split_registry([None], seed="seed", group_fields=("scene",))
    with pytest.raises(ValueError, match="duplicate episode ID"):
        splits.make_episode_split_registry(
            [
                {"episode_id": "same", "scene": "a"},
                {"episode_id": "same", "scene": "b"},
            ],
            seed="seed",
            group_fields=("scene",),
        )


def test_registry_builder_coalesces_transitive_groups_and_audits_them():
    records = [
        {"episode_id": "a", "scene": "s1", "route": "r1", "capture": "c1"},
        {"episode_id": "b", "scene": "s2", "route": "r1", "capture": "c1"},
        {"episode_id": "c", "scene": "s2", "route": "r3", "capture": "c3"},
        {"episode_id": "d", "scene": "s4", "route": "r4", "capture": "c4"},
        {"episode_id": "e", "scene": "s5", "route": "r5", "capture": "c5"},
    ]
    registry = splits.make_episode_split_registry(
        records, seed="seed", group_fields=("scene", "route", "capture")
    )
    assert registry.episode_to_component["a"] == registry.episode_to_component["b"]
    assert registry.episode_to_component["b"] == registry.episode_to_component["c"]
    assert registry.episode_to_component["d"] != registry.episode_to_component["e"]
    splits.audit_episode_split_registry(registry, records)


def test_frozen_registry_write_is_idempotent_and_refuses_drift(tmp_path):
    destination = tmp_path / "registry.json"
    registry = _registry()
    digest = splits.freeze_episode_split_registry(destination, registry)
    assert splits.freeze_episode_split_registry(destination, registry) == digest
    assert splits.read_episode_split_registry(destination) == registry
    with pytest.raises(FileExistsError, match="refusing to modify"):
        splits.freeze_episode_split_registry(
            destination,
            splits.EpisodeSplitRegistry(
                seed="other",
                group_fields=("scene",),
                episode_to_component={"episode-a": "component-a"},
                component_to_split={"component-a": "train"},
                episode_group_keys={"episode-a": ("scene-a",)},
            ),
        )


@pytest.mark.parametrize("identical", [True, False])
def test_frozen_registry_link_race_checks_existing_content(monkeypatch, tmp_path, identical):
    destination = tmp_path / "registry.json"
    registry = _registry()
    real_link = os.link

    def racing_link(source, target):
        destination.write_bytes(Path(source).read_bytes() if identical else b"different")
        raise FileExistsError(target)

    monkeypatch.setattr(splits.os, "link", racing_link)
    if identical:
        assert splits.freeze_episode_split_registry(destination, registry)
    else:
        with pytest.raises(FileExistsError, match="refusing to modify"):
            splits.freeze_episode_split_registry(destination, registry)
    monkeypatch.setattr(splits.os, "link", real_link)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema": "unsupported", "schema_version": 1},
        {"schema": "umcd.split_registry.v1", "schema_version": 1},
    ],
)
def test_registry_reader_rejects_unsupported_schema_and_bad_hash(tmp_path, payload):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    expected_message = (
        "unsupported UMCD split registry schema"
        if payload["schema"] == "unsupported"
        else "hash is invalid"
    )
    with pytest.raises(ValueError, match=expected_message):
        splits.read_episode_split_registry(path)


def test_registry_reader_rejects_noncanonical_representation_with_valid_digest(tmp_path):
    payload = _registry().to_dict()
    payload["extra"] = True
    core = {key: value for key, value in payload.items() if key != "registry_sha256"}
    payload["registry_sha256"] = hashlib.sha256(splits._stable_json(core)).hexdigest()
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="representation is not canonical"):
        splits.read_episode_split_registry(path)


def test_registry_audit_detects_corrupt_leakage_component_links():
    registry = object.__new__(splits.EpisodeSplitRegistry)
    object.__setattr__(registry, "group_fields", ("scene",))
    object.__setattr__(registry, "episode_group_keys", {"a": ("shared",), "b": ("shared",)})
    object.__setattr__(registry, "episode_to_component", {"a": "component-a", "b": "component-b"})
    with pytest.raises(ValueError, match="leakage group crosses split components"):
        splits.audit_episode_split_registry(
            registry,
            [
                {"episode_id": "a", "scene": "shared"},
                {"episode_id": "b", "scene": "shared"},
            ],
        )


def test_registry_freeze_cleans_up_when_temporary_file_creation_fails(monkeypatch, tmp_path):
    def fail_create(**_kwargs):
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(splits.tempfile, "NamedTemporaryFile", fail_create)
    with pytest.raises(OSError, match="temporary storage unavailable"):
        splits.freeze_episode_split_registry(tmp_path / "registry.json", _registry())


def test_registry_freeze_handles_temporary_stream_without_a_path(monkeypatch, tmp_path):
    class StreamWithoutPath:
        name = None

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def write(self, _value):
            return None

        def flush(self):
            return None

        def fileno(self):
            return 1

    monkeypatch.setattr(
        splits.tempfile, "NamedTemporaryFile", lambda **_kwargs: StreamWithoutPath()
    )
    monkeypatch.setattr(splits.os, "fsync", lambda _fd: None)
    monkeypatch.setattr(splits.os, "link", lambda *_args: None)
    digest = splits.freeze_episode_split_registry(tmp_path / "registry.json", _registry())
    assert len(digest) == 64
