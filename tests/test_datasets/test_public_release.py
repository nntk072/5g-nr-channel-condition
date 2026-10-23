from __future__ import annotations

import hashlib
import json
import zipfile

import pytest
from datasets.public_release import export_public_release, validate_public_release_manifest


def _manifest(payload: bytes) -> dict[str, object]:
    return {
        "schema": "umcd.public_release.v1",
        "release_id": "test-release",
        "source_software_revision": "a" * 40,
        "scenes": [
            {
                "scene_id": "etoile",
                "asset_tree_sha256": "d0b6ee5c9406e8b75c5fc17388e16037e988b5057d37bc40fe55699fefba115d",
                "classification": "odbl_produced_work",
                "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
                "copyright_notice_url": "https://www.openstreetmap.org/copyright",
                "required_credit": "© OpenStreetMap contributors",
                "source_software_revision": "a" * 40,
                "attribution_location": "LICENSES/README.txt",
                "source_data_offer_url": "https://example.org/osm-source-data",
                "extract_or_update_date_or_documented_gap": "upstream source records recovered",
                "source_ids_or_documented_gap": "OSM replication sequence 1",
                "conversion_recipe_or_documented_gap": "conversion recipe archived at release URL",
                "third_party_license_closure": "complete",
            }
        ],
        "artifacts": [
            {
                "path": "data/sample.bin",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "license": "ODbL-1.0",
                "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
                "scene_ids": ["etoile"],
                "source_scene_tree_sha256": [
                    "d0b6ee5c9406e8b75c5fc17388e16037e988b5057d37bc40fe55699fefba115d"
                ],
                "redistribution": "permitted",
                "lineage_complete": True,
            }
        ],
    }


def test_odbl_release_export_requires_complete_scene_assessment(tmp_path) -> None:
    payload = b"sample derived data"
    manifest = _manifest(payload)
    manifest["scenes"][0]["third_party_license_closure"] = "not independently recorded"
    with pytest.raises(ValueError, match="third-party license closure is incomplete"):
        validate_public_release_manifest(manifest)


def test_odbl_release_export_binds_asset_hash_and_adds_notices(tmp_path) -> None:
    payload = b"sample derived data"
    (tmp_path / "data").mkdir()
    (tmp_path / "data/sample.bin").write_bytes(payload)
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(payload)), encoding="utf-8")
    archive_path = tmp_path / "release.zip"

    result = export_public_release(tmp_path, manifest_path, archive_path)

    assert result == archive_path
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("data/sample.bin") == payload
        notices = archive.read("LICENSES/README.txt").decode()
        assert "© OpenStreetMap contributors" in notices
        assert "https://opendatacommons.org/licenses/odbl/1-0/" in notices
        assert "Source database offer: https://example.org/osm-source-data" in notices


def test_public_release_export_rejects_artifact_hash_mismatch(tmp_path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/sample.bin").write_bytes(b"tampered")
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(b"original")), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        export_public_release(tmp_path, manifest_path, tmp_path / "release.zip")
    assert not (tmp_path / "release.zip").exists()


def test_public_release_export_fails_closed_for_restricted_license(tmp_path) -> None:
    manifest = _manifest(b"payload")
    manifest["artifacts"][0]["redistribution"] = "prohibited"

    with pytest.raises(ValueError, match="not cleared for public redistribution"):
        validate_public_release_manifest(manifest)


def test_public_release_export_requires_scene_lineage_to_match_assessment() -> None:
    manifest = _manifest(b"payload")
    manifest["artifacts"][0]["source_scene_tree_sha256"] = []

    with pytest.raises(ValueError, match="scene IDs disagree with its lineage hashes"):
        validate_public_release_manifest(manifest)
