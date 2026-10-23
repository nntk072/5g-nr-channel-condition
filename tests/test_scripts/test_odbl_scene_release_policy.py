"""Tests binding the ODbL release policy to the audited Sionna scene assets."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_odbl_policy_binds_all_audited_osm_scenes_and_keeps_release_gated() -> None:
    policy = json.loads(
        (ROOT / "docs/research/licenses/odbl-osm-scene-release-policy-20261023.json").read_text()
    )
    audit = json.loads(
        (ROOT / "docs/research/results/g3-scene-asset-material-audit-20261023.json").read_text()
    )

    assert policy["schema"] == "umcd.odbl_osm_scene_release_policy.v1"
    assert policy["decision"] == "approved_for_research_use_subject_to_odbl_compliance"
    assert policy["distribution_status"] == "not_cleared_for_public_dataset_or_asset_redistribution"
    assert policy["source_database"]["required_credit"] == "© OpenStreetMap contributors"
    assert policy["source_database"]["license_url"].endswith("/odbl/1-0/")

    bindings = {item["scene_id"]: item for item in policy["scene_bindings"]}
    assert set(bindings) == {"etoile", "florence", "munich"}
    for scene_id, binding in bindings.items():
        audited_scene = audit["scenes"][scene_id]
        assert binding["asset_count"] == audited_scene["asset_count"]
        assert binding["asset_tree_sha256"] == audited_scene["asset_set_sha256"]
        assert binding["data_license"] == "ODbL-1.0"
        assert binding["osm_extract_or_update_timestamp"] is None
        assert binding["osm_object_or_change_set_ids"] is None
        assert binding["conversion_recipe"] is None
