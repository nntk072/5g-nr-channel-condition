"""Regression checks for the source-only ODbL scene snapshot audit."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "audit_odbl_scene_source_snapshot.py"
_SPEC = importlib.util.spec_from_file_location("audit_odbl_scene_source_snapshot", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


ROOT_COMMIT = "1" * 40
RT_COMMIT = "2" * 40
ASSET_HASH = "a" * 64


def _policy() -> dict[str, object]:
    return {
        "software_source": {
            "source_commit": ROOT_COMMIT,
            "rt_submodule_commit": RT_COMMIT,
        },
        "scene_bindings": [
            {
                "scene_id": scene,
                "source_scene_xml": f"src/sionna/rt/scenes/{scene}/{scene}.xml",
                "asset_count": 1,
                "asset_tree_sha256": ASSET_HASH,
            }
            for scene in ("etoile", "florence", "munich")
        ],
    }


@pytest.fixture
def snapshot_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    checkout = tmp_path / "sionna"
    rt = checkout / "ext/sionna-rt"
    rt.mkdir(parents=True)
    monkeypatch.setattr(
        audit,
        "_git",
        lambda repo, *args: (
            ROOT_COMMIT
            if args == ("rev-parse", "HEAD") and repo == checkout
            else RT_COMMIT
            if args == ("rev-parse", "HEAD")
            else ""
            if args == ("status", "--porcelain", "--untracked-files=all")
            else f"160000 commit {RT_COMMIT}\text/sionna-rt"
        ),
    )
    monkeypatch.setattr(
        audit,
        "_tree_inventory",
        lambda *_args: {
            "tracked_file_count": 1,
            "git_tree_inventory_sha256": "b" * 64,
            "git_blob_inventory_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(
        audit,
        "_last_xml_change",
        lambda *_args: {"commit": RT_COMMIT, "authored_at": "2025-01-01T00:00:00Z"},
    )
    from integrations.sionna import scene_assets

    monkeypatch.setattr(
        scene_assets,
        "scene_asset_manifest",
        lambda *_args: {
            "asset_count": 1,
            "asset_set_sha256": ASSET_HASH,
            "complete": True,
            "unresolved_references": [],
        },
    )
    return checkout


def test_odbl_source_snapshot_report_is_verified_but_never_release_eligible(
    snapshot_mocks: Path,
) -> None:
    result = audit.audit_source_snapshot(snapshot_mocks, _policy())

    assert result["status"] == "SOURCE_SNAPSHOT_VERIFIED"
    assert result["public_release_eligible"] is False
    assert all(scene["policy_asset_tree_matches"] for scene in result["scenes"])
    assert any("OSM extract/update" in item for item in result["unresolved_provenance"])


def test_odbl_source_snapshot_mismatch_keeps_release_gate_closed(
    snapshot_mocks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from integrations.sionna import scene_assets

    monkeypatch.setattr(
        scene_assets,
        "scene_asset_manifest",
        lambda *_args: {
            "asset_count": 2,
            "asset_set_sha256": "d" * 64,
            "complete": True,
            "unresolved_references": [],
        },
    )
    result = audit.audit_source_snapshot(snapshot_mocks, _policy())

    assert result["status"] == "SOURCE_SNAPSHOT_MISMATCH"
    assert result["public_release_eligible"] is False
    assert all(not scene["policy_asset_tree_matches"] for scene in result["scenes"])
