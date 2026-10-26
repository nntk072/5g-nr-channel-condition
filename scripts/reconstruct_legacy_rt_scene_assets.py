"""Reconstruct RT scene-asset identities from a frozen source revision and lockfile."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

from datasets.channel import ChannelDataset
from integrations.sionna.scene_assets import scene_asset_manifest

_SCHEMA = "umcd.legacy_rt_scene_asset_reconstruction.v1"
_PACKAGES = (
    "artifacts",
    "configuration",
    "datasets",
    "propagation",
    "receivers",
    "models",
    "learning",
    "evaluation",
    "experiments",
    "integrations",
    "compute",
    "loslab",
)


def _git_bytes(revision: str, path: str, *, repo_root: Path) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=repo_root)


def _historical_package_source_sha256(revision: str, *, repo_root: Path) -> str:
    digest = hashlib.sha256()
    for package in _PACKAGES:
        paths = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", revision, "--", package],
            text=True,
            cwd=repo_root,
        ).splitlines()
        for path in sorted(
            item for item in paths if item.endswith(".py") and "__pycache__" not in item.split("/")
        ):
            digest.update(path.encode("utf-8"))
            digest.update(_git_bytes(revision, path, repo_root=repo_root))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locked_wheel(lock: dict[str, Any], wheel_path: Path) -> tuple[str, str]:
    wheel_hash = _sha256_file(wheel_path)
    matches = [
        package
        for package in lock.get("package", [])
        if package.get("name") == "sionna-rt"
        and any(
            artifact.get("hash") == f"sha256:{wheel_hash}" for artifact in package.get("wheels", [])
        )
    ]
    if len(matches) != 1:
        raise ValueError("Sionna-RT wheel is not pinned by the historical lockfile")
    return str(matches[0]["version"]), wheel_hash


def _scene_from_wheel(wheel_root: Path, scene_name: str) -> Path:
    if not scene_name or "/" in scene_name or "\\" in scene_name or ".." in scene_name:
        raise ValueError("legacy scene name is not a supported built-in scene identifier")
    scene_path = wheel_root / "sionna" / "rt" / "scenes" / scene_name / f"{scene_name}.xml"
    if not scene_path.is_file():
        raise ValueError(f"scene {scene_name!r} is neither a tracked scene nor in the wheel")
    return scene_path


def _historical_scene(
    scene_name: str, *, revision: str, repo_root: Path, wheel_root: Path, temp_root: Path
) -> Path:
    if not scene_name.startswith("/"):
        return _scene_from_wheel(wheel_root, scene_name)
    scene_path = Path(scene_name).resolve()
    try:
        relative = scene_path.relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError("legacy scene path is outside the repository") from error
    relative_parent = relative.parent.as_posix()
    subprocess.run(
        ["git", "cat-file", "-e", f"{revision}:{relative.as_posix()}"],
        check=True,
        capture_output=True,
    )
    archive = subprocess.check_output(["git", "archive", revision, relative_parent], cwd=repo_root)
    archive_root = temp_root / "historical-scenes"
    archive_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as stream:
        stream.extractall(archive_root, filter="data")
    scene_file = archive_root / relative
    if not scene_file.is_file():
        raise ValueError("historical Git archive did not contain the declared scene")
    return scene_file


def reconstruct_legacy_scene_assets(
    inputs: list[tuple[str, Path]], *, sionna_rt_wheel: Path, repo_root: Path
) -> dict[str, Any]:
    """Reconstruct complete scene closures bound to legacy dataset and source hashes."""
    if not inputs:
        raise ValueError("at least one legacy ChannelDataset is required")
    wheel_path = Path(sionna_rt_wheel).resolve(strict=True)
    wheel_sha256 = _sha256_file(wheel_path)
    datasets: list[dict[str, Any]] = []
    observed_revision: str | None = None
    observed_source_hash: str | None = None
    lock_sha256: str | None = None
    sionna_rt_version: str | None = None
    seen_names: set[str] = set()

    with tempfile.TemporaryDirectory(prefix="legacy-rt-assets-") as temp_name:
        temp_root = Path(temp_name)
        wheel_root = temp_root / "wheel"
        wheel_root.mkdir()
        with zipfile.ZipFile(wheel_path) as wheel:
            wheel.extractall(wheel_root)

        for name, raw_path in inputs:
            if not name.strip() or name.strip() != name or name in seen_names:
                raise ValueError("dataset names must be unique, nonempty, and trimmed")
            seen_names.add(name)
            path = Path(raw_path).resolve(strict=True)
            payload = path.read_bytes()
            dataset = ChannelDataset.load_bytes(payload)
            report = dataset.generation_report
            revision = report.get("environment", {}).get("git_sha")
            source_hash = report.get("package_source_sha256")
            if not isinstance(revision, str) or len(revision) != 40:
                raise ValueError(f"{name} does not record a full historical Git revision")
            if observed_revision is not None and revision != observed_revision:
                raise ValueError("all legacy datasets must come from the same Git revision")
            observed_revision = revision
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ValueError(f"{name} does not record its package-source fingerprint")
            if observed_source_hash is not None and source_hash != observed_source_hash:
                raise ValueError("legacy datasets have different package-source fingerprints")
            observed_source_hash = source_hash

            lock_bytes = _git_bytes(revision, "uv.lock", repo_root=repo_root)
            current_lock_sha = hashlib.sha256(lock_bytes).hexdigest()
            lock = tomllib.loads(lock_bytes.decode("utf-8"))
            current_version, current_wheel_hash = _locked_wheel(lock, wheel_path)
            if current_wheel_hash != wheel_sha256:
                raise ValueError("historical Sionna-RT wheel hash changed during reconstruction")
            if report.get("environment", {}).get("sionna-rt") != current_version:
                raise ValueError(f"{name} runtime version differs from the historical lockfile")
            if lock_sha256 is not None and current_lock_sha != lock_sha256:
                raise ValueError("legacy datasets use different historical lockfiles")
            lock_sha256 = current_lock_sha
            sionna_rt_version = current_version

            if (
                _historical_package_source_sha256(revision, repo_root=Path(repo_root))
                != source_hash
            ):
                raise ValueError(
                    f"{name} package-source fingerprint does not match its Git revision"
                )

            scene_name = str(dataset.metadata[0].get("scene", "")) if dataset.metadata else ""
            scene_config = dataset.metadata[0].get("scene_config") if dataset.metadata else None
            if (
                not isinstance(scene_config, dict)
                or scene_config.get("name") != scene_name
                or report.get("scene_config") != scene_config
                or any(
                    metadata.get("scene") != scene_name
                    or metadata.get("scene_config") != scene_config
                    for metadata in dataset.metadata
                )
            ):
                raise ValueError(f"{name} rows disagree on the historical scene configuration")
            if any(str(source) != "sionna_rt" for source in dataset.source):
                raise ValueError(f"{name} contains non-RT observations")
            if any("split_identity_complete" in metadata for metadata in dataset.metadata):
                raise ValueError(
                    f"{name} already has split identities; reconstruction is legacy-only"
                )

            scene_path = _historical_scene(
                scene_name,
                revision=revision,
                repo_root=Path(repo_root),
                wheel_root=wheel_root,
                temp_root=temp_root,
            )
            asset_manifest = scene_asset_manifest(scene_path)
            if asset_manifest.get("complete") is not True:
                raise ValueError(f"{name} historical scene has unresolved asset references")
            asset_manifest["scene_xml_path"] = scene_name
            datasets.append(
                {
                    "name": name,
                    "input_sha256": hashlib.sha256(payload).hexdigest(),
                    "input_bytes": len(payload),
                    "rows": len(dataset.y),
                    "source_revision": revision,
                    "package_source_sha256": source_hash,
                    "scene": scene_name,
                    "scene_config": scene_config,
                    "scene_asset_manifest": asset_manifest,
                }
            )

    assert observed_revision is not None
    assert observed_source_hash is not None
    assert lock_sha256 is not None
    assert sionna_rt_version is not None
    if len({item["scene"] for item in datasets}) != len(datasets):
        raise ValueError("scene reconstruction manifest requires one dataset per scene identity")
    return {
        "schema": _SCHEMA,
        "status": "NOT_QUALIFIED",
        "reconstruction_method": "historical_git_tree_and_lockfile_pinned_sionna_rt_wheel",
        "reconstruction_tool_source_sha256": _sha256_file(Path(__file__).resolve()),
        "asset_manifest_builder_source_sha256": _sha256_file(
            Path(scene_asset_manifest.__code__.co_filename).resolve()
        ),
        "source_revision": observed_revision,
        "package_source_sha256": observed_source_hash,
        "uv_lock_sha256": lock_sha256,
        "sionna_rt_version": sionna_rt_version,
        "sionna_rt_wheel_sha256": wheel_sha256,
        "datasets": sorted(datasets, key=lambda item: item["name"]),
        "limitations": [
            "The source revision and package wheel are content-verified against the recorded source fingerprint and historical lockfile.",
            "This reconstructed manifest is not externally signed and does not prove corpus completeness or acquisition authority.",
            "The original generation report did not capture the lockfile or asset digest; this is a reproducible retrospective reconstruction, not original run-time attestation.",
        ],
    }


def _parse_dataset(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("dataset must be NAME=PATH")
    return name.strip(), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", type=_parse_dataset, required=True)
    parser.add_argument("--sionna-rt-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    try:
        manifest = reconstruct_legacy_scene_assets(
            args.dataset,
            sionna_rt_wheel=args.sionna_rt_wheel,
            repo_root=Path(__file__).resolve().parents[1],
        )
    except (OSError, RuntimeError, TypeError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(2, f"error: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "datasets": len(manifest["datasets"]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
