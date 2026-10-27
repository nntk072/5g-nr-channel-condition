"""Audit the pinned Sionna source snapshot for the ODbL-derived built-in scenes.

This records what the upstream Git tree proves. It deliberately cannot recover
OSM extract metadata, conversion history, or third-party rights that the source
tree does not contain, and it never authorizes public redistribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLICY = ROOT / "docs/research/licenses/odbl-osm-scene-release-policy-20261023.json"
SCHEMA = "umcd.odbl_scene_source_snapshot_audit.v1"


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True, stderr=subprocess.PIPE
    ).strip()


def _tree_inventory(repo: Path, revision: str, scene_path: str) -> dict[str, Any]:
    raw = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "-z", revision, "--", scene_path]
    )
    rows: list[dict[str, str]] = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ", 2)
        rows.append(
            {
                "mode": mode,
                "object_type": object_type,
                "object_id": object_id,
                "path": path.decode("utf-8", errors="strict"),
            }
        )
    rows.sort(key=lambda row: row["path"])
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "tracked_file_count": len(rows),
        "git_tree_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "git_blob_inventory_sha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['object_id']}" for row in rows).encode()
        ).hexdigest(),
    }


def _last_xml_change(repo: Path, revision: str, xml_path: str) -> dict[str, str]:
    value = _git(repo, "log", "-1", "--follow", "--format=%H%x09%aI", revision, "--", xml_path)
    commit, authored_at = value.split("\t", 1)
    return {"commit": commit, "authored_at": authored_at}


def audit_source_snapshot(source_checkout: Path, policy: dict[str, Any]) -> dict[str, Any]:
    """Compare the pinned source tree and recursive scene hashes to policy."""
    source_checkout = source_checkout.expanduser().resolve(strict=True)
    rt_checkout = source_checkout / "ext/sionna-rt"
    expected_software = policy["software_source"]
    expected_root = expected_software["source_commit"]
    expected_rt = expected_software["rt_submodule_commit"]
    root_revision = _git(source_checkout, "rev-parse", "HEAD")
    rt_revision = _git(rt_checkout, "rev-parse", "HEAD")
    root_dirty = bool(_git(source_checkout, "status", "--porcelain", "--untracked-files=all"))
    rt_dirty = bool(_git(rt_checkout, "status", "--porcelain", "--untracked-files=all"))
    gitlink = _git(source_checkout, "ls-tree", "HEAD", "ext/sionna-rt").split()
    gitlink_revision = gitlink[2] if len(gitlink) >= 3 and gitlink[1] == "commit" else None

    scenes: list[dict[str, Any]] = []
    for binding in policy["scene_bindings"]:
        scene_id = binding["scene_id"]
        source_xml = binding["source_scene_xml"]
        prefix = "src/sionna/rt/"
        if not source_xml.startswith(prefix):
            raise ValueError(f"unsupported policy scene path: {source_xml}")
        xml_relative_to_rt = source_xml.removeprefix(prefix)
        # Policy paths are relative to the Sionna source root; RT scene assets
        # are tracked by its pinned submodule.
        rt_scene_path = f"{prefix}{xml_relative_to_rt}"
        rt_scene_root = str(Path(rt_scene_path).parent)
        actual_xml = source_checkout / source_xml
        if not actual_xml.exists():
            actual_xml = rt_checkout / rt_scene_path
        from integrations.sionna.scene_assets import scene_asset_manifest

        assets = scene_asset_manifest(actual_xml)
        inventory = _tree_inventory(rt_checkout, expected_rt, rt_scene_root)
        xml_commit = _last_xml_change(rt_checkout, expected_rt, rt_scene_path)
        scenes.append(
            {
                "scene_id": scene_id,
                "source_scene_xml": source_xml,
                "rt_git_path": rt_scene_path,
                **inventory,
                "xml_last_change": xml_commit,
                "asset_count": assets["asset_count"],
                "asset_tree_sha256": assets["asset_set_sha256"],
                "assets_complete": assets["complete"],
                "unresolved_references": assets["unresolved_references"],
                "policy_asset_count_matches": assets["asset_count"] == binding["asset_count"],
                "policy_asset_tree_matches": assets["asset_set_sha256"]
                == binding["asset_tree_sha256"],
            }
        )

    commits_match = (
        root_revision == expected_root
        and rt_revision == expected_rt
        and gitlink_revision == expected_rt
    )
    worktrees_clean = not root_dirty and not rt_dirty
    scene_hashes_match = all(
        scene["policy_asset_count_matches"]
        and scene["policy_asset_tree_matches"]
        and scene["assets_complete"]
        for scene in scenes
    )
    verified = commits_match and worktrees_clean and scene_hashes_match
    return {
        "schema": SCHEMA,
        "status": "SOURCE_SNAPSHOT_VERIFIED" if verified else "SOURCE_SNAPSHOT_MISMATCH",
        "public_release_eligible": False,
        "source_checkout": str(source_checkout),
        "policy_sha256": hashlib.sha256(POLICY.read_bytes()).hexdigest(),
        "software_source": {
            "expected_commit": expected_root,
            "observed_commit": root_revision,
            "rt_submodule_expected_commit": expected_rt,
            "rt_submodule_gitlink_commit": gitlink_revision,
            "rt_submodule_observed_commit": rt_revision,
            "root_worktree_clean": not root_dirty,
            "rt_worktree_clean": not rt_dirty,
            "commits_match_policy": commits_match,
        },
        "scenes": scenes,
        "unresolved_provenance": [
            "OSM extract/update timestamp and upstream object/change-set identifiers are not established by Git scene history.",
            "Original scene conversion recipe is not established by the inspected source tree.",
            "Per-file third-party content and license closure remains unreviewed.",
            "Release-specific legal classification, attribution placement, and source-data offer remain unreviewed.",
        ],
        "limitations": [
            "Git object identity and scene byte hashes prove a source snapshot only; they do not establish OSM extraction provenance or legal rights.",
            "Git author dates identify source-history events, not OSM extract or conversion dates.",
            "No scene generation, public packaging, legal approval, or scientific qualification was performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_checkout", type=Path, help="pinned Sionna source checkout")
    parser.add_argument("output", type=Path, help="new JSON report path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        policy = json.loads(POLICY.read_text(encoding="utf-8"))
        result = audit_source_snapshot(args.source_checkout, policy)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"{result['status']}: {args.output}")
    if not result["software_source"]["commits_match_policy"]:
        raise SystemExit(2)
    if result["status"] != "SOURCE_SNAPSHOT_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
