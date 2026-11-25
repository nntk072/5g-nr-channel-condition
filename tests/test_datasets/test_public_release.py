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

import datasets.public_release as release
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
    recipe = b"reproducible conversion recipe v1\n"
    return {
        "schema": "umcd.public_release.v3",
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
                "osm_extract_timestamp": "2026-01-01T00:00:00Z",
                "osm_source_object_ids": ["node/1", "way/2", "relation/3"],
                "conversion_recipe_artifact_path": "metadata/conversion-recipe.txt",
                "conversion_recipe_sha256": hashlib.sha256(recipe).hexdigest(),
                "third_party_license_closure": {
                    "review_evidence_sha256": "b" * 64,
                    "assets": [
                        {
                            "asset_path": "textures/provider-logo.png",
                            "asset_sha256": "c" * 64,
                            "rights_holder": "Example rights holder",
                            "license_id": "CC-BY-4.0",
                            "license_url": "https://creativecommons.org/licenses/by/4.0/",
                            "source_url": "https://example.org/assets/logo",
                            "redistribution_permission": "permitted",
                            "attribution_text": "Example creator",
                        }
                    ],
                },
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
            },
            {
                "path": "metadata/conversion-recipe.txt",
                "sha256": hashlib.sha256(recipe).hexdigest(),
                "license": "Apache-2.0",
                "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
                "scene_ids": [],
                "scene_classifications": {},
                "source_scene_tree_sha256": [],
                "redistribution": "permitted",
                "lineage_complete": True,
            },
        ],
    }


def test_odbl_release_export_requires_complete_scene_assessment(tmp_path) -> None:
    payload = b"sample derived data"
    manifest = _manifest(payload)
    manifest["scenes"][0]["third_party_license_closure"] = {
        "review_evidence_sha256": "not-a-digest",
        "assets": [],
    }
    with pytest.raises(ValueError, match="third-party license closure is incomplete"):
        validate_public_release_manifest(manifest)


def test_odbl_release_rejects_documented_provenance_gaps() -> None:
    manifest = _manifest(b"payload")
    manifest["scenes"][0]["osm_source_object_ids"] = ["source IDs unavailable"]

    with pytest.raises(ValueError, match="canonical OSM object IDs"):
        validate_public_release_manifest(manifest)


def test_odbl_release_conversion_recipe_must_be_the_declared_artifact() -> None:
    manifest = _manifest(b"payload")
    manifest["scenes"][0]["conversion_recipe_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="hash-bound"):
        validate_public_release_manifest(manifest)


def test_odbl_release_rejects_unsafe_third_party_asset_paths() -> None:
    manifest = _manifest(b"payload")
    manifest["scenes"][0]["third_party_license_closure"]["assets"][0]["asset_path"] = (
        "../license.txt"
    )

    with pytest.raises(ValueError, match="unsafe third-party asset path"):
        validate_public_release_manifest(manifest)


def test_odbl_release_requires_third_party_redistribution_permission() -> None:
    manifest = _manifest(b"payload")
    manifest["scenes"][0]["third_party_license_closure"]["assets"][0][
        "redistribution_permission"
    ] = "not_permitted"

    with pytest.raises(ValueError, match="not cleared for redistribution"):
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
        '{"schema":"umcd.public_release.v3","schema":"umcd.public_release.v3"}',
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


def test_public_release_v3_schema_is_packaged_and_accepts_valid_manifest() -> None:
    import jsonschema

    schema = json.loads(
        files("datasets")
        .joinpath("schema", "public-release-manifest.v3.schema.json")
        .read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(_manifest(b"payload"))


def test_odbl_release_export_binds_asset_hash_and_adds_notices(tmp_path) -> None:
    payload = b"sample derived data"
    (tmp_path / "data").mkdir()
    (tmp_path / "data/sample.bin").write_bytes(payload)
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/conversion-recipe.txt").write_bytes(
        b"reproducible conversion recipe v1\n"
    )
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
        assert "OSM source object count: 3" in notices
        assert "conversion-recipe.txt" in notices
        assert "textures/provider-logo.png" in notices
        assert "attribution: Example creator" in notices
        approval = json.loads(archive.read("release-approval.json"))
        assert approval["signer_id"] == _RELEASE_SIGNER_ID
        assert approval["signature_namespace"] == PUBLIC_RELEASE_SIGNATURE_NAMESPACE
        import jsonschema

        approval_schema = json.loads(
            files("datasets")
            .joinpath("schema", "public-release-approval.v2.schema.json")
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
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/conversion-recipe.txt").write_bytes(
        b"reproducible conversion recipe v1\n"
    )
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
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata/conversion-recipe.txt").write_bytes(
        b"reproducible conversion recipe v1\n"
    )
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


def test_stable_release_reader_rejects_nonregular_and_oversized_files(tmp_path) -> None:
    with pytest.raises(ValueError, match="single-link regular file"):
        release._read_stable_release_file(tmp_path, limit=10, label="fixture")
    path = tmp_path / "large.bin"
    path.write_bytes(b"payload")
    with pytest.raises(ValueError, match="exceeds 2 bytes"):
        release._read_stable_release_file(path, limit=2, label="fixture")


def test_stable_release_reader_rejects_identity_change_before_open(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"payload")
    real_fstat = os.fstat

    def changed_identity(descriptor: int):
        actual = real_fstat(descriptor)
        return os.stat_result(
            (
                actual.st_mode,
                actual.st_ino + 1,
                actual.st_dev,
                actual.st_nlink,
                actual.st_uid,
                actual.st_gid,
                actual.st_size,
                actual.st_atime,
                actual.st_mtime,
                actual.st_ctime,
            )
        )

    monkeypatch.setattr(release.os, "fstat", changed_identity)
    with pytest.raises(ValueError, match="changed while opening"):
        release._read_stable_release_file(path, limit=20, label="fixture")


def test_stable_release_reader_rejects_growth_and_mutation_during_read(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"x")
    monkeypatch.setattr(release.os, "read", lambda _descriptor, _size: b"xxx")
    with pytest.raises(ValueError, match="exceeds 2 bytes"):
        release._read_stable_release_file(path, limit=2, label="fixture")

    monkeypatch.undo()
    real_fstat = os.fstat
    calls = 0

    def changed_snapshot(descriptor: int):
        nonlocal calls
        calls += 1
        actual = real_fstat(descriptor)
        if calls == 2:
            return os.stat_result(
                (
                    actual.st_mode,
                    actual.st_ino,
                    actual.st_dev,
                    actual.st_nlink,
                    actual.st_uid,
                    actual.st_gid,
                    actual.st_size,
                    actual.st_atime,
                    actual.st_mtime + 1,
                    actual.st_ctime,
                )
            )
        return actual

    monkeypatch.setattr(release.os, "fstat", changed_snapshot)
    with pytest.raises(ValueError, match="changed while being read"):
        release._read_stable_release_file(path, limit=20, label="fixture")


def test_open_release_artifact_rejects_missing_dir_fd_support(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(release.os, "supports_dir_fd", set())
    try:
        with (
            pytest.raises(ValueError, match="lacks safe directory-relative"),
            release._open_release_artifact(root_fd, release.PurePosixPath("data/item.bin")),
        ):
            pass
    finally:
        os.close(root_fd)


def test_public_release_rejects_nonfinite_manifest_json(tmp_path) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text('{"value":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON value"):
        export_public_release(
            tmp_path,
            manifest_path,
            tmp_path / "release.zip",
            manifest_signature_path=tmp_path / "release.json.sig",
            trusted_allowed_signers_path=tmp_path / "allowed_signers",
            signer_id=_RELEASE_SIGNER_ID,
        )


def _mock_export_signature(monkeypatch: pytest.MonkeyPatch, signature_bytes: bytes) -> None:
    from evaluation.acceptance import signed_attestation

    monkeypatch.setattr(
        signed_attestation,
        "verify_openssh_signature_for_bytes",
        lambda *_args, **kwargs: {
            "signature_sha256": hashlib.sha256(signature_bytes).hexdigest(),
            "allowed_signers_sha256": "e" * 64,
            "signer_id": kwargs["signer_id"],
            "namespace": kwargs["namespace"],
        },
    )


def test_public_release_detects_signature_changed_after_verification(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(b"payload")), encoding="utf-8")
    signature_path = tmp_path / "release.json.sig"
    signature_path.write_bytes(b"changed-signature")
    trust_path = tmp_path / "allowed_signers"
    trust_path.write_text("fixture trust", encoding="utf-8")
    _mock_export_signature(monkeypatch, b"verified-signature")

    with pytest.raises(ValueError, match="signature changed after verification"):
        export_public_release(
            tmp_path,
            manifest_path,
            tmp_path / "release.zip",
            manifest_signature_path=signature_path,
            trusted_allowed_signers_path=trust_path,
            signer_id=_RELEASE_SIGNER_ID,
        )


def test_public_release_refuses_archive_overwrite_of_manifest(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(b"payload")), encoding="utf-8")
    signature_path = tmp_path / "release.json.sig"
    signature_path.write_bytes(b"signature")
    trust_path = tmp_path / "allowed_signers"
    trust_path.write_text("fixture trust", encoding="utf-8")
    _mock_export_signature(monkeypatch, b"signature")

    with pytest.raises(ValueError, match="cannot overwrite an input artifact or manifest"):
        export_public_release(
            tmp_path,
            manifest_path,
            manifest_path,
            manifest_signature_path=signature_path,
            trusted_allowed_signers_path=trust_path,
            signer_id=_RELEASE_SIGNER_ID,
        )


def test_public_release_rejects_artifact_symlink_resolving_outside_source(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(b"payload")), encoding="utf-8")
    signature_path = tmp_path / "release.json.sig"
    signature_path.write_bytes(b"signature")
    trust_path = tmp_path / "allowed_signers"
    trust_path.write_text("fixture trust", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"payload")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/sample.bin").symlink_to(outside)
    _mock_export_signature(monkeypatch, b"signature")

    with pytest.raises(ValueError, match="artifact path resolves outside source_root"):
        export_public_release(
            tmp_path,
            manifest_path,
            tmp_path / "release.zip",
            manifest_signature_path=signature_path,
            trusted_allowed_signers_path=trust_path,
            signer_id=_RELEASE_SIGNER_ID,
        )


def test_public_release_requires_manifest_inside_source_root(tmp_path) -> None:
    outside_manifest = tmp_path.parent / f"{tmp_path.name}-release.json"
    outside_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest must be inside source_root"):
        export_public_release(tmp_path, outside_manifest, tmp_path / "release.zip")


def test_public_release_rejects_manifest_symlink_traversal(tmp_path) -> None:
    outside_manifest = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside_manifest.write_text("{}", encoding="utf-8")
    linked_manifest = tmp_path / "release.json"
    linked_manifest.symlink_to(outside_manifest)
    with pytest.raises(ValueError, match="manifest path cannot traverse outside"):
        export_public_release(tmp_path, linked_manifest, tmp_path / "release.zip")


def test_public_release_requires_safe_directory_relative_opens(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "release.json"
    manifest_path.write_text(json.dumps(_manifest(b"payload")), encoding="utf-8")
    signature_path = tmp_path / "release.json.sig"
    signature_path.write_bytes(b"signature")
    trust_path = tmp_path / "allowed_signers"
    trust_path.write_text("fixture trust", encoding="utf-8")
    _mock_export_signature(monkeypatch, b"signature")
    monkeypatch.setattr(release.os, "supports_dir_fd", set())

    with pytest.raises(ValueError, match="lacks safe directory-relative"):
        export_public_release(
            tmp_path,
            manifest_path,
            tmp_path / "release.zip",
            manifest_signature_path=signature_path,
            trusted_allowed_signers_path=trust_path,
            signer_id=_RELEASE_SIGNER_ID,
        )
