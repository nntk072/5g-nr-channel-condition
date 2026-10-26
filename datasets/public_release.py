"""Fail-closed packaging for reviewed public research releases."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from urllib.parse import urlsplit

_POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/research/licenses/odbl-osm-scene-release-policy-20261023.json"
)
_ODBL_URL = "https://opendatacommons.org/licenses/odbl/1-0/"
_OSM_COPYRIGHT_URL = "https://www.openstreetmap.org/copyright"
_OSM_CREDIT = "© OpenStreetMap contributors"
PUBLIC_RELEASE_SIGNATURE_NAMESPACE = "umcd-public-release-manifest-v2@github.com"
_MAX_RELEASE_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_RELEASE_SIGNATURE_BYTES = 64 * 1024
_RESERVED_ARCHIVE_PATHS = (
    PurePosixPath("release-manifest.json"),
    PurePosixPath("release-manifest.json.sig"),
    PurePosixPath("release-approval.json"),
    PurePosixPath("LICENSES/README.txt"),
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MANIFEST_FIELDS = {"schema", "release_id", "source_software_revision", "artifacts", "scenes"}
_SCENE_FIELDS = {
    "scene_id",
    "asset_tree_sha256",
    "license_url",
    "copyright_notice_url",
    "required_credit",
    "source_software_revision",
    "attribution_location",
    "source_data_offer_url",
    "extract_or_update_date_or_documented_gap",
    "source_ids_or_documented_gap",
    "conversion_recipe_or_documented_gap",
    "third_party_license_closure",
}
_ARTIFACT_FIELDS = {
    "path",
    "sha256",
    "license",
    "license_url",
    "scene_ids",
    "scene_classifications",
    "source_scene_tree_sha256",
    "redistribution",
    "lineage_complete",
}


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_stable_release_file(path: Path, *, limit: int, label: str) -> bytes:
    """Read a bounded single-link regular file without following symlinks."""
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError(f"{label} must be a single-link regular file")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe no-symlink file opening")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
    )
    try:
        before = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(before_path):
            raise ValueError(f"{label} changed while opening")
        if before.st_size > limit:
            raise ValueError(f"{label} exceeds {limit} bytes")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise ValueError(f"{label} exceeds {limit} bytes")
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
        if (
            size != before.st_size
            or _file_identity(after) != _file_identity(before)
            or _file_identity(path_after) != _file_identity(before)
        ):
            raise ValueError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


@contextmanager
def _open_release_artifact(root_fd: int, relative_path: PurePosixPath) -> Iterator[int]:
    """Open a package artifact by directory descriptors without following links."""
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise ValueError("platform lacks safe directory-relative no-symlink file opening")

    regular_flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    directory_flags = regular_flags | os.O_DIRECTORY
    opened: list[int] = []
    try:
        parent_fd = root_fd
        for component in relative_path.parts[:-1]:
            directory_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            opened.append(directory_fd)
            parent_fd = directory_fd
        artifact_fd = os.open(relative_path.parts[-1], regular_flags, dir_fd=parent_fd)
        opened.append(artifact_fd)
        metadata = os.fstat(artifact_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("release artifact must be a single-link regular file")
        yield artifact_fd
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def _write_verified_artifact(
    archive: zipfile.ZipFile,
    archive_name: str,
    root_fd: int,
    relative_path: PurePosixPath,
    expected_sha256: str,
) -> None:
    """Stream the exact hash-checked descriptor bytes into the temporary ZIP."""
    digest = hashlib.sha256()
    with _open_release_artifact(root_fd, relative_path) as source_fd:
        with archive.open(archive_name, "w", force_zip64=True) as destination:
            while block := os.read(source_fd, 1024 * 1024):
                digest.update(block)
                destination.write(block)
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"release artifact hash mismatch: {relative_path.as_posix()}")


def _required_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"public release requires non-empty canonical {name}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"public release manifest has duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"public release manifest contains a non-finite JSON value: {value}")


def _required_https_url(value: Any, *, name: str) -> str:
    """Require an absolute HTTPS URL suitable for a release-time data offer."""
    text = _required_text(value, name=name)
    try:
        parsed = urlsplit(text)
        hostname = parsed.hostname
        parsed.port
    except ValueError as error:
        raise ValueError(f"public release requires a valid HTTPS URL for {name}") from error
    if parsed.scheme != "https" or not hostname or parsed.username or parsed.password:
        raise ValueError(f"public release requires a valid HTTPS URL for {name}")
    return text


def validate_public_release_manifest(manifest: Any) -> dict[str, Any]:
    """Validate artifact licenses, hashes, ODbL provenance, and release notices.

    This is a release-time check, not a license-classification service. Every
    artifact needs an explicit license and every OSM-derived scene needs a
    reviewed, complete assessment before the archive can be created.
    """
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_FIELDS
        or manifest.get("schema") != "umcd.public_release.v2"
    ):
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
        if "classification" in scene:
            raise ValueError(
                f"scene {scene_id!r} classification belongs to each artifact-scene pair"
            )
        if set(scene) != _SCENE_FIELDS:
            raise ValueError("each public release scene assessment must contain exactly v2 fields")
        tree_hash = _required_text(scene.get("asset_tree_sha256"), name="asset_tree_sha256")
        if scene_id in declared_ids or tree_hash in declared_hashes:
            raise ValueError("public release scene assessments must be unique")
        if known_scenes.get(scene_id) != tree_hash:
            raise ValueError(f"scene {scene_id!r} does not match the audited OSM asset hash")
        declared_ids.add(scene_id)
        declared_hashes.add(tree_hash)
        if scene.get("license_url") != _ODBL_URL:
            raise ValueError(f"scene {scene_id!r} must link directly to ODbL v1.0")
        if scene.get("copyright_notice_url") != _OSM_COPYRIGHT_URL:
            raise ValueError(f"scene {scene_id!r} must link to the OSM copyright notice")
        if scene.get("required_credit") != _OSM_CREDIT:
            raise ValueError(f"scene {scene_id!r} must include the required OSM credit")
        for field in (
            "source_software_revision",
            "attribution_location",
            "extract_or_update_date_or_documented_gap",
            "source_ids_or_documented_gap",
            "conversion_recipe_or_documented_gap",
        ):
            _required_text(scene.get(field), name=f"{scene_id}.{field}")
        _required_https_url(
            scene.get("source_data_offer_url"), name=f"{scene_id}.source_data_offer_url"
        )
        if scene["attribution_location"] not in {"LICENSES/README.txt"}:
            raise ValueError(
                f"scene {scene_id!r} attribution_location must name the generated OSM notice"
            )
        if scene["source_software_revision"] != manifest["source_software_revision"]:
            raise ValueError(f"scene {scene_id!r} software revision differs from the release")
        if scene.get("third_party_license_closure") != "complete":
            raise ValueError(f"scene {scene_id!r} third-party license closure is incomplete")

    artifact_paths: set[str] = set()
    referenced_scene_ids: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            raise ValueError("each public release artifact must contain exactly v2 fields")
        raw_path = _required_text(artifact.get("path"), name="artifact.path")
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or "\\" in raw_path
            or "\x00" in raw_path
            or path.as_posix() != raw_path
            or any(part in {"", ".", ".."} or ":" in part for part in raw_path.split("/"))
        ):
            raise ValueError("artifact paths must be safe relative POSIX paths")
        if any(
            path.parts[: len(reserved.parts)] == reserved.parts
            or reserved.parts[: len(path.parts)] == path.parts
            for reserved in _RESERVED_ARCHIVE_PATHS
        ):
            raise ValueError("artifact path conflicts with generated release metadata")
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
        if (
            not isinstance(artifact.get("sha256"), str)
            or _SHA256_RE.fullmatch(artifact["sha256"]) is None
        ):
            raise ValueError(f"artifact {raw_path} sha256 must be lowercase SHA-256 hex")
        artifact_scenes = artifact.get("scene_ids", [])
        source_scene_hashes = artifact.get("source_scene_tree_sha256", [])
        if (
            not isinstance(source_scene_hashes, list)
            or any(
                not isinstance(digest, str) or digest not in known_hashes
                for digest in source_scene_hashes
            )
            or len(set(source_scene_hashes)) != len(source_scene_hashes)
        ):
            raise ValueError(f"artifact {raw_path} has an unknown OSM scene lineage hash")
        expected_scene_ids = {known_hashes[digest] for digest in source_scene_hashes}
        if (
            not isinstance(artifact_scenes, list)
            or any(
                not isinstance(scene_id, str) or scene_id not in declared_ids
                for scene_id in artifact_scenes
            )
            or len(set(artifact_scenes)) != len(artifact_scenes)
        ):
            raise ValueError(f"artifact {raw_path} references an undeclared scene")
        if set(artifact_scenes) != expected_scene_ids:
            raise ValueError(f"artifact {raw_path} scene IDs disagree with its lineage hashes")
        classifications = artifact.get("scene_classifications")
        if not isinstance(classifications, dict) or set(classifications) != expected_scene_ids:
            raise ValueError(
                f"artifact {raw_path} needs exactly one classification per source scene"
            )
        if any(
            classification not in {"odbl_database", "odbl_produced_work"}
            for classification in classifications.values()
        ):
            raise ValueError(f"artifact {raw_path} has an unsupported ODbL classification")
        referenced_scene_ids.update(artifact_scenes)
        if artifact_scenes and any(
            classifications[scene_id] == "odbl_database"
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
    *,
    manifest_signature_path: str | Path | None = None,
    trusted_allowed_signers_path: str | Path | None = None,
    signer_id: str | None = None,
) -> Path:
    """Create an approved ZIP with exact manifest/signature bytes and an approval receipt.

    ``manifest_signature_path`` must be an OpenSSH detached signature over the
    exact manifest file bytes. Its principal must be present in the separately
    governed ``trusted_allowed_signers_path`` under the release namespace.
    """
    source = Path(source_root).resolve()
    manifest_file = Path(os.path.abspath(manifest_path))
    try:
        manifest_file.relative_to(source)
    except ValueError as error:
        raise ValueError("release manifest must be inside source_root") from error
    try:
        manifest_file.resolve().relative_to(source)
    except ValueError as error:
        raise ValueError("release manifest path cannot traverse outside source_root") from error
    signature_inputs = (
        manifest_signature_path,
        trusted_allowed_signers_path,
        signer_id,
    )
    if any(value is None for value in signature_inputs):
        raise ValueError(
            "public release export requires a detached manifest signature, external "
            "allowed_signers file, and expected signer ID"
        )
    assert manifest_signature_path is not None
    assert trusted_allowed_signers_path is not None
    assert signer_id is not None
    manifest_bytes = _read_stable_release_file(
        manifest_file,
        limit=_MAX_RELEASE_MANIFEST_BYTES,
        label="public release manifest",
    )
    try:
        manifest_value = json.loads(
            manifest_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as error:
        raise ValueError(f"public release manifest must be strict UTF-8 JSON: {error}") from error
    manifest = validate_public_release_manifest(manifest_value)

    from evaluation.g3_signed_attestation import verify_openssh_signature_for_bytes

    signature_path = Path(os.path.abspath(manifest_signature_path))
    trust_path = Path(os.path.abspath(trusted_allowed_signers_path))
    signature_identity = verify_openssh_signature_for_bytes(
        manifest_bytes,
        signature_path,
        trust_path,
        signer_id=signer_id,
        namespace=PUBLIC_RELEASE_SIGNATURE_NAMESPACE,
    )
    signature_bytes = _read_stable_release_file(
        signature_path,
        limit=_MAX_RELEASE_SIGNATURE_BYTES,
        label="public release manifest signature",
    )
    if hashlib.sha256(signature_bytes).hexdigest() != signature_identity["signature_sha256"]:
        raise ValueError("public release manifest signature changed after verification")
    approval = {
        "schema": "umcd.public_release_approval.v1",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "signature_sha256": signature_identity["signature_sha256"],
        "allowed_signers_sha256": signature_identity["allowed_signers_sha256"],
        "signer_id": signature_identity["signer_id"],
        "signature_namespace": signature_identity["namespace"],
    }
    if (
        not hasattr(os, "O_NOFOLLOW")
        or not hasattr(os, "O_DIRECTORY")
        or os.open not in os.supports_dir_fd
    ):
        raise ValueError("platform lacks safe directory-relative no-symlink file opening")

    files: list[tuple[str, PurePosixPath, Path, str]] = []
    for artifact in manifest["artifacts"]:
        relative = PurePosixPath(artifact["path"])
        file_path = (source / Path(*relative.parts)).resolve()
        try:
            file_path.relative_to(source)
        except ValueError as error:
            raise ValueError("artifact path resolves outside source_root") from error
        files.append((relative.as_posix(), relative, file_path, artifact["sha256"]))

    notices = [
        f"Release: {manifest['release_id']}",
        f"Source software revision: {manifest['source_software_revision']}",
        "",
    ]
    scene_by_id = {scene["scene_id"]: scene for scene in manifest["scenes"]}
    for artifact in manifest["artifacts"]:
        for scene_id in artifact.get("scene_ids", []):
            scene = scene_by_id[scene_id]
            notices.extend(
                [
                    f"Artifact {artifact['path']} derived from scene {scene_id} ({scene['asset_tree_sha256']}):",
                    f"  Classification: {artifact['scene_classifications'][scene_id]}",
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
    if resolved_destination in {
        manifest_file.resolve(),
        signature_path.resolve(),
        trust_path.resolve(),
    } or any(resolved_destination == file_path for _, _, file_path, _ in files):
        raise ValueError("release archive cannot overwrite an input artifact or manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        source_fd = os.open(
            source,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for archive_name, relative, _, expected_sha256 in files:
                    _write_verified_artifact(
                        archive,
                        archive_name,
                        source_fd,
                        relative,
                        expected_sha256,
                    )
                archive.writestr("release-manifest.json", manifest_bytes)
                archive.writestr("release-manifest.json.sig", signature_bytes)
                archive.writestr(
                    "release-approval.json",
                    json.dumps(approval, indent=2, sort_keys=True) + "\n",
                )
                archive.writestr("LICENSES/README.txt", "\n".join(notices) + "\n")
        finally:
            os.close(source_fd)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination
