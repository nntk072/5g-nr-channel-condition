"""Scene asset manifests: closed field rosters, canonical paths, byte counts, digests and geometry relations."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest
from datasets.split_lineage import _scene_geometry_asset_relations_from_manifest


def _asset(path="meshes/wall.obj", data=b"v 0 0 0\n"):
    return {"path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _manifest(assets=None, *, complete=True):
    assets = assets if assets is not None else [_asset()]
    normalized = sorted(
        ({"path": a["path"], "bytes": a["bytes"], "sha256": a["sha256"]} for a in assets),
        key=lambda row: row["path"],
    )
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return {
        "complete": complete,
        "asset_set_sha256": hashlib.sha256(canonical).hexdigest(),
        "assets": assets,
    }


def _check(manifest):
    return _scene_geometry_asset_relations_from_manifest(manifest, context="scene")


def test_complete_manifest_yields_geometry_relations() -> None:
    complete, digest, relations = _check(_manifest())
    assert complete is True
    assert len(digest) == 64
    assert len(relations) == 1


def test_incomplete_manifest_is_accepted_without_geometry_relations() -> None:
    complete, digest, relations = _check(_manifest(complete=False))
    assert complete is False
    assert relations == frozenset()
    assert len(digest) == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(lambda m: m.update(complete="yes"), "invalid completeness flag", id="flag"),
        pytest.param(
            lambda m: m.update(asset_set_sha256="ABC"), "invalid set digest", id="digest-shape"
        ),
        pytest.param(lambda m: m.update(assets=[]), "must contain assets", id="no-assets"),
        pytest.param(lambda m: m["assets"][0].pop("bytes"), "invalid field roster", id="roster"),
        pytest.param(
            lambda m: m["assets"][0].update(path="/abs/wall.obj"),
            "noncanonical relative path",
            id="absolute-path",
        ),
        pytest.param(
            lambda m: m["assets"][0].update(path="../escape.obj"),
            "noncanonical relative path",
            id="parent-path",
        ),
        pytest.param(
            lambda m: m["assets"][0].update(bytes=-1), "invalid byte count", id="negative-bytes"
        ),
        pytest.param(
            lambda m: m["assets"][0].update(bytes=True), "invalid byte count", id="bool-bytes"
        ),
        pytest.param(
            lambda m: m["assets"][0].update(sha256="Z" * 64), "invalid SHA-256", id="bad-sha"
        ),
        pytest.param(
            lambda m: m.update(asset_set_sha256="0" * 64), "does not match its entries", id="digest"
        ),
    ],
)
def test_manifest_refusals(mutate, message) -> None:
    manifest = copy.deepcopy(_manifest())
    mutate(manifest)
    with pytest.raises(ValueError, match=message):
        _check(manifest)


def test_manifest_must_be_an_object() -> None:
    with pytest.raises(ValueError, match="scene_asset_manifest must be an object"):
        _check(["not", "a", "manifest"])


def test_duplicate_asset_paths_are_refused() -> None:
    first = _asset()
    manifest = _manifest([first, dict(first)])
    with pytest.raises(ValueError, match="paths must be unique"):
        _check(manifest)
