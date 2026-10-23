"""Fail-closed packaging for reviewed public research releases."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/research/licenses/odbl-osm-scene-release-policy-20261023.json"
)
_ODBL_URL = "https://opendatacommons.org/licenses/odbl/1-0/"
_OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"
_OSM_CREDIT = "© OpenStreetMap contributors"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"public release requires non-empty {name}")
    return value.strip()


def validate_public_release_manifest(manifest: Any) -> dict[str, Any]:
    """Validate artifact licenses, hashes, ODbL provenance, and release notices.

    This is a release-time check, not a license-classification service. Every
    artifact needs an explicit license and every OSM-derived scene needs a
    reviewed, complete assessment before the archive can be created.
    """
    if not isinstance(manifest, dict) or manifest.get("schema") != "umcd.public_release.v1":
        raise ValueError("unsupported public release manifest")
    _required_text(manifest.get("release_id"), name="release_id")
    _required_text(manifest.get("source_software_revision"), name="source_software_revision")
    artifacts = manifest.get("artifacts")
    scenes = manifest.get("scenes")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("public release must enumerate at least one artifact")
    if not isinstance(scenes, list):
        raise ValueError("public release scenes must be a list")

    policy = json.loads(_POLICY_PATH.read_text(encoding="utf-8"))
    known_scenes = {
        scene["scene_id"]: scene["asset_tree_sha256"] for scene in policy["scene_bindings"]
    }
    known_hashes = {value: key for key, value in known_scenes.items()}
    declared_hashes: set[str] = set()
    declared_ids: set[str] = set()
    for scene in scenes:
        if not isinstance(scene, dict):
            raise ValueError("each public release scene assessment must be an object")
        scene_id = _required_text(scene.get("scene_id"), name="scene_id")
        tree_hash = _required_text(scene.get("asset_tree_sha256"), name="asset_tree_sha256")
        if scene_id in declared_ids or tree_hash in declared_hashes:
            raise ValueError("public release scene assessments must be unique")
        if known_scenes.get(scene_id) != tree_hash:
            raise ValueError(f"scene {scene_id!r} does not match the audited OSM asset hash")
        declared_ids.add(scene_id)
        declared_hashes.add(tree_hash)
        if scene.get("classification") not in {"odbl_database", "odbl_produced_work"}:
            raise ValueError(f"scene {scene_id!r} needs an artifact-specific ODbL classification")
        if scene.get("license_url") != _ODBL_URL:
            raise ValueError(f"scene {scene_id!r} must link directly to ODbL v1.0")
        if scene.get("copyright_notice_url") != _OSM_COPYRIGHT_URL:
            raise ValueError(f"scene {scene_id!r} must link to the OSM copyright notice")
        if scene.get("required_credit") != _OSM_CREDIT:
            raise ValueError(f"scene {scene_id!r} must include the required OSM credit")
        for field in (
            "source_software_revision",
            "attribution_location",
            "source_data_offer_url",
            "extract_or_update_date_or_documented_gap",
            "source_ids_or_documented_gap",
            "conversion_recipe_or_documented_gap",
        ):
            _required_text(scene.get(field), name=f"{scene_id}.{field}")
        if scene["attribution_location"] not in {"LICENSES/README.txt"}:
            raise ValueError(
                f"scene {scene_id!r} attribution_location must name the generated OSM notice"
            )
        if scene["source_software_revision"] != manifest["source_software_revision"]:
            raise ValueError(f"scene {scene_id!r} software revision differs from the release")
        if scene.get("third_party_license_closure") != "complete":
            raise ValueError(f"scene {scene_id!r} third-party license closure is incomplete")

    scene_by_id = {scene["scene_id"]: scene for scene in scenes}
    artifact_paths: set[str] = set()
    referenced_scene_ids: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise ValueError("each public release artifact must be an object")
        raw_path = _required_text(artifact.get("path"), name="artifact.path")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts or "\\" in raw_path:
            raise ValueError("artifact paths must be safe relative POSIX paths")
        if raw_path in artifact_paths:
            raise ValueError("public release artifact paths must be unique")
        artifact_paths.add(raw_path)
        _required_text(artifact.get("license"), name=f"artifact {raw_path} license")
        _required_text(artifact.get("license_url"), name=f"artifact {raw_path} license_url")
        _required_text(artifact.get("sha256"), name=f"artifact {raw_path} sha256")
        if artifact.get("redistribution") != "permitted":
            raise ValueError(f"artifact {raw_path} is not cleared for public redistribution")
        if artifact.get("lineage_complete") is not True:
            raise ValueError(f"artifact {raw_path} needs complete source-lineage review")
        if not isinstance(artifact.get("sha256"), str) or len(artifact["sha256"]) != 64:
            raise ValueError(f"artifact {raw_path} sha256 must be a SHA-256 hex digest")
        try:
            int(artifact["sha256"], 16)
        except ValueError as error:
            raise ValueError(f"artifact {raw_path} sha256 must be hexadecimal") from error
        artifact_scenes = artifact.get("scene_ids", [])
        source_scene_hashes = artifact.get("source_scene_tree_sha256", [])
        if not isinstance(source_scene_hashes, list) or any(
            not isinstance(digest, str) or digest not in known_hashes
            for digest in source_scene_hashes
        ):
            raise ValueError(f"artifact {raw_path} has an unknown OSM scene lineage hash")
        expected_scene_ids = {known_hashes[digest] for digest in source_scene_hashes}
        if not isinstance(artifact_scenes, list) or any(
            not isinstance(scene_id, str) or scene_id not in declared_ids
            for scene_id in artifact_scenes
        ):
            raise ValueError(f"artifact {raw_path} references an undeclared scene")
        if set(artifact_scenes) != expected_scene_ids:
            raise ValueError(f"artifact {raw_path} scene IDs disagree with its lineage hashes")
        referenced_scene_ids.update(artifact_scenes)
        if artifact_scenes and any(
            scene_by_id[scene_id]["classification"] == "odbl_database"
            and (artifact["license"] != "ODbL-1.0" or artifact["license_url"] != _ODBL_URL)
            for scene_id in artifact_scenes
        ):
            raise ValueError(
                f"OSM-derived database artifact {raw_path} must be licensed under ODbL"
            )
    if referenced_scene_ids != declared_ids:
        raise ValueError("every ODbL scene assessment must be referenced by a release artifact")
    return manifest


def export_public_release(
    source_root: str | Path,
    manifest_path: str | Path,
    archive_path: str | Path,
) -> Path:
    """Create a ZIP only after release-specific license and integrity checks pass."""
    source = Path(source_root).resolve()
    manifest_file = Path(manifest_path).resolve()
    try:
        manifest_file.relative_to(source)
    except ValueError as error:
        raise ValueError("release manifest must be inside source_root") from error
    manifest = validate_public_release_manifest(
        json.loads(manifest_file.read_text(encoding="utf-8"))
    )

    files: list[tuple[str, Path]] = []
    for artifact in manifest["artifacts"]:
        relative = PurePosixPath(artifact["path"])
        file_path = (source / Path(*relative.parts)).resolve()
        try:
            file_path.relative_to(source)
        except ValueError as error:
            raise ValueError("artifact path resolves outside source_root") from error
        if not file_path.is_file():
            raise ValueError(f"release artifact is missing or not a file: {relative}")
        if _sha256_file(file_path) != artifact["sha256"]:
            raise ValueError(f"release artifact hash mismatch: {relative}")
        files.append((relative.as_posix(), file_path))

    notices = [
        f"Release: {manifest['release_id']}",
        f"Source software revision: {manifest['source_software_revision']}",
        "",
    ]
    for scene in manifest["scenes"]:
        notices.extend(
            [
                f"Scene {scene['scene_id']} ({scene['asset_tree_sha256']}):",
                f"  Classification: {scene['classification']}",
                f"  Attribution: {scene['required_credit']}",
                f"  ODbL notice: {scene['license_url']}",
                f"  Copyright notice: {scene['copyright_notice_url']}",
                f"  Attribution location: {scene['attribution_location']}",
                f"  Source database offer: {scene['source_data_offer_url']}",
                "",
            ]
        )
    notices.extend(
        [
            "Artifact licenses:",
            *(f"  {a['path']}: {a['license']} ({a['license_url']})" for a in manifest["artifacts"]),
        ]
    )

    destination = Path(archive_path)
    resolved_destination = destination.resolve()
    if resolved_destination == manifest_file or any(
        resolved_destination == path for _, path in files
    ):
        raise ValueError("release archive cannot overwrite an input artifact or manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for archive_name, path in files:
                archive.write(path, archive_name)
            archive.writestr(
                "release-manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            archive.writestr("LICENSES/README.txt", "\n".join(notices) + "\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination
