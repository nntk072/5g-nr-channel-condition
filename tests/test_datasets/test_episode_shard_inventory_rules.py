"""Episode shard inventory: manifest name, pointer, layer-roster and byte-budget rules."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from datasets.episode_store import iter_episode_bundle_shards, write_episode_bundle
from tests.test_datasets.test_umcd_store_integrity import _episode_bundle


def _store(tmp_path: Path) -> tuple[Path, Path]:
    store = tmp_path / "store"
    write_episode_bundle(store, _episode_bundle())
    manifests = sorted((store / "manifests").rglob("*.json"))
    assert len(manifests) == 1
    return store, manifests[0]


def _edit(manifest: Path, mutate) -> None:
    document = json.loads(manifest.read_text())
    mutate(document)
    core = {k: document[k] for k in ("schema", "schema_version", "episode_id", "layers")}
    document["bundle_sha256"] = hashlib.sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    manifest.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda d: d.update(schema="umcd.bundle.v9"),
            "unsupported episode bundle schema",
            id="schema",
        ),
        pytest.param(
            lambda d: d.update(episode_id=""), "episode_id must be nonempty", id="episode-id"
        ),
        pytest.param(
            lambda d: d.update(layers=[]), "layers must be a nonempty list", id="layers-empty"
        ),
        pytest.param(
            lambda d: d.update(layers=[l for l in d["layers"] if l["layer"] != "episode"]),
            "exactly one episode layer",
            id="no-episode",
        ),
        pytest.param(
            lambda d: d["layers"].append({**d["layers"][0], "layer": "episode"}),
            "exactly one episode layer",
            id="two-episodes",
        ),
        pytest.param(
            lambda d: d["layers"][1].update(extra=1),
            "layer reference fields are invalid",
            id="reference-fields",
        ),
        pytest.param(
            lambda d: d["layers"][1].update(sha256="xyz" * 21 + "x"),
            "must be a SHA-256 hex digest|must be hexadecimal",
            id="digest",
        ),
    ],
)
def test_manifest_layer_roster_is_checked_before_reading(tmp_path, mutate, message) -> None:
    store, manifest = _store(tmp_path)
    _edit(manifest, mutate)
    with pytest.raises(ValueError, match=message):
        list(iter_episode_bundle_shards(store, max_shard_bytes=10_000_000))


def test_manifest_file_name_must_match_its_episode(tmp_path) -> None:
    store, manifest = _store(tmp_path)
    renamed = manifest.with_name("0" * 64 + ".json")
    manifest.rename(renamed)
    with pytest.raises(ValueError, match="filename does not match its episode ID"):
        list(iter_episode_bundle_shards(store, max_shard_bytes=10_000_000))


def test_shard_budget_must_be_positive(tmp_path) -> None:
    store, _ = _store(tmp_path)
    with pytest.raises(ValueError, match="max_shard_bytes must be positive"):
        list(iter_episode_bundle_shards(store, max_shard_bytes=0))
