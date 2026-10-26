from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest
from datasets.public_release import (
    PUBLIC_RELEASE_SIGNATURE_NAMESPACE,
    export_public_release,
    validate_public_release_manifest,
)

_RELEASE_SIGNER_ID = "release-reviewer@example.org"


def _export_with_external_approval(tmp_path, manifest_path, archive_path):
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH ssh-keygen is required for public-release signature tests")
    key_path = tmp_path / "release-reviewer-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(key_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = key_path.with_suffix(".pub").read_text(encoding="utf-8").split()
    allowed_signers = tmp_path / "release-reviewers.allowed_signers"
    allowed_signers.write_text(
        f'{_RELEASE_SIGNER_ID} namespaces="{PUBLIC_RELEASE_SIGNATURE_NAMESPACE}" '
        f"{public_key[0]} {public_key[1]}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-f",
            str(key_path),
            "-n",
            PUBLIC_RELEASE_SIGNATURE_NAMESPACE,
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return export_public_release(
        tmp_path,
        manifest_path,
        archive_path,
        manifest_signature_path=Path(f"{manifest_path}.sig"),
        trusted_allowed_signers_path=allowed_signers,
        signer_id=_RELEASE_SIGNER_ID,
    )


def _manifest(payload: bytes) -> dict[str, Any]:
    return {
        "schema": "umcd.public_release.v2",
        "release_id": "test-release",
        "source_software_revision": "a" * 40,
        "scenes": [
            {
                "scene_id": "etoile",
                "asset_tree_sha256": "d0b6ee5c9406e8b75c5fc17388e16037e988b5057d37bc40fe55699fefba115d",
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
                "scene_classifications": {"etoile": "odbl_database"},
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


def test_public_release_manifest_rejects_unversioned_top_level_fields() -> None:
    manifest = _manifest(b"payload")
    manifest["review_status"] = "approved"

    with pytest.raises(ValueError, match="unsupported public release manifest"):
        validate_public_release_manifest(manifest)


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        ("top", "release_id", " test-release"),
        ("scene", "scene_id", "etoile "),
        ("scene", "source_data_offer_url", " https://example.org/osm-source-data"),
        ("artifact", "path", " data/sample.bin"),
        ("artifact", "license", "ODbL-1.0 "),
    ],
)
def test_public_release_rejects_padded_signed_text_fields(
    location: str, field: str, value: str
) -> None:
    manifest = _manifest(b"payload")
    if location == "top":
        manifest[field] = value
    elif location == "scene":
        manifest["scenes"][0][field] = value
    else:
        manifest["artifacts"][0][field] = value

    with pytest.raises(ValueError, match="canonical"):
        validate_public_release_manifest(manifest)


def test_public_release_export_requires_external_manifest_approval(tmp_path) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(b"payload")), encoding="utf-8")

    with pytest.raises(ValueError, match="requires a detached manifest signature"):
        export_public_release(tmp_path, manifest_path, tmp_path / "release.zip")


def test_public_release_export_rejects_duplicate_manifest_json_fields(tmp_path) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(
        '{"schema":"umcd.public_release.v2","schema":"umcd.public_release.v2"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        _export_with_external_approval(tmp_path, manifest_path, tmp_path / "release.zip")
    assert not (tmp_path / "release.zip").exists()


def test_public_release_scene_rejects_unbound_fields() -> None:
    manifest = _manifest(b"payload")
    manifest["scenes"][0]["additional_license"] = "Apache-2.0"

    with pytest.raises(ValueError, match="exactly v2 fields"):
        validate_public_release_manifest(manifest)


def test_public_release_artifact_rejects_unbound_fields() -> None:
    manifest = _manifest(b"payload")
    manifest["artifacts"][0]["reviewer"] = "reviewed"

    with pytest.raises(ValueError, match="exactly v2 fields"):
        validate_public_release_manifest(manifest)


def test_public_release_artifact_requires_lowercase_sha256() -> None:
    manifest = _manifest(b"payload")
    manifest["artifacts"][0]["sha256"] = "A" * 64

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_public_release_manifest(manifest)


def test_public_release_v2_schema_is_packaged_and_accepts_valid_manifest() -> None:
    import jsonschema

    schema = json.loads(
        files("datasets")
        .joinpath("schema", "public-release-manifest.v2.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(_manifest(b"payload"))


def test_odbl_release_export_binds_asset_hash_and_adds_notices(tmp_path) -> None:
    payload = b"sample derived data"
    (tmp_path / "data").mkdir()
    (tmp_path / "data/sample.bin").write_bytes(payload)
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(payload)), encoding="utf-8")
    archive_path = tmp_path / "release.zip"

    result = _export_with_external_approval(tmp_path, manifest_path, archive_path)

    assert result == archive_path
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.read("data/sample.bin") == payload
        notices = archive.read("LICENSES/README.txt").decode()
        assert "© OpenStreetMap contributors" in notices
        assert "https://opendatacommons.org/licenses/odbl/1-0/" in notices
        assert "Source database offer: https://example.org/osm-source-data" in notices
        approval = json.loads(archive.read("release-approval.json"))
        assert approval["signer_id"] == _RELEASE_SIGNER_ID
        assert approval["signature_namespace"] == PUBLIC_RELEASE_SIGNATURE_NAMESPACE
        import jsonschema

        approval_schema = json.loads(
            files("datasets")
            .joinpath("schema", "public-release-approval.v1.schema.json")
            .read_text(encoding="utf-8")
        )
        jsonschema.Draft202012Validator.check_schema(approval_schema)
        jsonschema.Draft202012Validator(approval_schema).validate(approval)
        assert archive.read("release-manifest.json") == manifest_path.read_bytes()
        assert (
            archive.read("release-manifest.json.sig") == Path(f"{manifest_path}.sig").read_bytes()
        )


def test_public_release_export_rejects_artifact_hash_mismatch(tmp_path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data/sample.bin").write_bytes(b"tampered")
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(b"original")), encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        _export_with_external_approval(tmp_path, manifest_path, tmp_path / "release.zip")
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


@pytest.mark.parametrize("url", ("http://example.org/source", "not a URL", "https:///missing-host"))
def test_public_release_requires_valid_https_source_data_offer(url: str) -> None:
    manifest = _manifest(b"payload")
    manifest["scenes"][0]["source_data_offer_url"] = url

    with pytest.raises(ValueError, match="valid HTTPS URL"):
        validate_public_release_manifest(manifest)


def test_public_release_rejects_scene_wide_classification() -> None:
    manifest = _manifest(b"payload")
    manifest["scenes"][0]["classification"] = "odbl_produced_work"

    with pytest.raises(ValueError, match="belongs to each artifact-scene pair"):
        validate_public_release_manifest(manifest)


@pytest.mark.parametrize(
    "classifications",
    (None, {}, {"etoile": "unreviewed"}),
)
def test_public_release_requires_one_supported_classification_per_artifact_scene(
    classifications: object,
) -> None:
    manifest = _manifest(b"payload")
    manifest["artifacts"][0]["scene_classifications"] = classifications

    with pytest.raises(ValueError, match="classification"):
        validate_public_release_manifest(manifest)


def test_public_release_requires_odbl_for_each_database_artifact() -> None:
    manifest = _manifest(b"payload")
    artifact = manifest["artifacts"][0]
    artifact["scene_classifications"] = {"etoile": "odbl_database"}
    artifact["license"] = "CC-BY-4.0"
    artifact["license_url"] = "https://creativecommons.org/licenses/by/4.0/"

    with pytest.raises(ValueError, match="must be licensed under ODbL"):
        validate_public_release_manifest(manifest)


def test_public_release_rejects_duplicate_scene_lineage_hashes() -> None:
    manifest = _manifest(b"payload")
    digest = manifest["artifacts"][0]["source_scene_tree_sha256"][0]
    manifest["artifacts"][0]["source_scene_tree_sha256"] = [digest, digest]

    with pytest.raises(ValueError, match="unknown OSM scene lineage hash"):
        validate_public_release_manifest(manifest)


def test_public_release_classifies_each_artifact_scene_pair_and_emits_both_notices(
    tmp_path,
) -> None:
    payload = b"shared source-derived bytes"
    (tmp_path / "data").mkdir()
    (tmp_path / "data/database.bin").write_bytes(payload)
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures/plot.bin").write_bytes(payload)
    manifest = _manifest(payload)
    database = manifest["artifacts"][0]
    database["path"] = "data/database.bin"
    figure = json.loads(json.dumps(database))
    figure["path"] = "figures/plot.bin"
    figure["license"] = "CC-BY-4.0"
    figure["license_url"] = "https://creativecommons.org/licenses/by/4.0/"
    figure["scene_classifications"] = {"etoile": "odbl_produced_work"}
    manifest["artifacts"].append(figure)
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validate_public_release_manifest(manifest)
    archive_path = _export_with_external_approval(tmp_path, manifest_path, tmp_path / "release.zip")

    with zipfile.ZipFile(archive_path) as archive:
        notices = archive.read("LICENSES/README.txt").decode()
    assert "Artifact data/database.bin derived from scene etoile" in notices
    assert "Classification: odbl_database" in notices
    assert "Artifact figures/plot.bin derived from scene etoile" in notices
    assert "Classification: odbl_produced_work" in notices


@pytest.mark.parametrize(
    "artifact_path",
    (
        "./data/sample.bin",
        "data//sample.bin",
        "data/../sample.bin",
        "C:/data/sample.bin",
        "data:stream.bin",
        "data/sample\x00.bin",
    ),
)
def test_public_release_manifest_rejects_noncanonical_artifact_paths(
    artifact_path: str,
) -> None:
    manifest = _manifest(b"payload")
    manifest["artifacts"][0]["path"] = artifact_path

    with pytest.raises(ValueError, match="safe relative POSIX paths"):
        validate_public_release_manifest(manifest)


@pytest.mark.parametrize(
    "artifact_path",
    (
        "release-manifest.json",
        "release-manifest.json/payload.bin",
        "release-manifest.json.sig",
        "release-approval.json",
        "LICENSES",
        "LICENSES/README.txt",
        "LICENSES/README.txt/extra.txt",
    ),
)
def test_public_release_manifest_rejects_generated_metadata_path_conflicts(
    artifact_path: str,
) -> None:
    manifest = _manifest(b"payload")
    manifest["artifacts"][0]["path"] = artifact_path

    with pytest.raises(ValueError, match="conflicts with generated release metadata"):
        validate_public_release_manifest(manifest)


@pytest.mark.parametrize("symlink_parent", (False, True))
def test_public_release_export_rejects_symlink_artifact_paths(
    tmp_path, symlink_parent: bool
) -> None:
    payload = b"sample derived data"
    target_directory = tmp_path / "actual-data"
    target_directory.mkdir()
    (target_directory / "sample.bin").write_bytes(payload)
    if symlink_parent:
        (tmp_path / "data").symlink_to(target_directory, target_is_directory=True)
    else:
        (tmp_path / "data").mkdir()
        (tmp_path / "data/sample.bin").symlink_to(target_directory / "sample.bin")
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(payload)), encoding="utf-8")

    with pytest.raises(OSError):
        _export_with_external_approval(tmp_path, manifest_path, tmp_path / "release.zip")
    assert not (tmp_path / "release.zip").exists()


def test_public_release_export_rejects_hardlinked_artifacts(tmp_path) -> None:
    payload = b"sample derived data"
    artifact_path = tmp_path / "data/sample.bin"
    artifact_path.parent.mkdir()
    artifact_path.write_bytes(payload)
    os.link(artifact_path, tmp_path / "external-alias.bin")
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(payload)), encoding="utf-8")

    with pytest.raises(ValueError, match="single-link regular file"):
        _export_with_external_approval(tmp_path, manifest_path, tmp_path / "release.zip")
    assert not (tmp_path / "release.zip").exists()


def test_public_release_hashes_the_exact_bytes_written_to_zip(tmp_path, monkeypatch) -> None:
    payload = b"original release bytes"
    artifact_path = tmp_path / "data/sample.bin"
    artifact_path.parent.mkdir()
    artifact_path.write_bytes(payload)
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(payload)), encoding="utf-8")
    archive_path = tmp_path / "release.zip"
    archive_path.write_bytes(b"previous archive remains intact")
    original_open = zipfile.ZipFile.open
    mutated = False

    def mutate_after_source_open(self, name, mode="r", pwd=None, *, force_zip64=False):
        nonlocal mutated
        if name == "data/sample.bin" and mode == "w" and not mutated:
            mutated = True
            artifact_path.write_bytes(b"tampered release bytes")
        return original_open(self, name, mode, pwd, force_zip64=force_zip64)

    monkeypatch.setattr(zipfile.ZipFile, "open", mutate_after_source_open)
    with pytest.raises(ValueError, match="release artifact hash mismatch"):
        _export_with_external_approval(tmp_path, manifest_path, archive_path)

    assert mutated
    assert archive_path.read_bytes() == b"previous archive remains intact"
