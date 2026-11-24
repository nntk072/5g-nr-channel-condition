"""Episode-bundle manifests: pointer, schema, layer-roster and identity checks on disk."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from datasets.episode_store import read_episode_bundle, write_episode_bundle
from tests.test_datasets.test_umcd_store_integrity import _episode_bundle


def _write(root: Path):
    stored = write_episode_bundle(root, _episode_bundle())
    return stored, root / stored.relative_path


def _rewrite(manifest: Path, mutate: Any) -> dict[str, Any]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    mutate(document)
    manifest.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    return document


def _stored_with(stored, **changes):
    return dataclasses.replace(stored, **changes)


def test_bundle_round_trips_from_its_manifest(tmp_path: Path) -> None:
    stored, _ = _write(tmp_path / "store")
    restored = read_episode_bundle(tmp_path / "store", stored)
    assert restored.episode.episode_id == stored.episode_id


def test_pointer_must_stay_inside_the_store(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stored, _ = _write(store)
    with pytest.raises(ValueError, match="outside the store|inside the store"):
        read_episode_bundle(store, _stored_with(stored, relative_path="../escape.json"))


def test_manifest_must_be_a_json_object(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stored, manifest = _write(store)
    manifest.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest must be a JSON object"):
        read_episode_bundle(store, stored)


def test_manifest_schema_and_identity_are_checked(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stored, manifest = _write(store)
    _rewrite(manifest, lambda d: d.update(schema="umcd.bundle.v9"))
    with pytest.raises(ValueError, match="unsupported episode bundle schema"):
        read_episode_bundle(store, stored)


def test_manifest_hash_must_match_its_content(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stored, manifest = _write(store)
    _rewrite(manifest, lambda d: d.update(episode_id="someone-else"))
    with pytest.raises(ValueError):
        read_episode_bundle(store, stored)


def test_layer_roster_must_be_a_nonempty_list(tmp_path: Path) -> None:
    store = tmp_path / "store"
    stored, manifest = _write(store)
    _rewrite(manifest, lambda d: d.update(layers=[]))
    with pytest.raises(ValueError):
        read_episode_bundle(store, stored)
