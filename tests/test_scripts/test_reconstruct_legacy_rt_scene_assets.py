from __future__ import annotations

import hashlib
import subprocess
import zipfile
from pathlib import Path

import numpy as np
from datasets.channel import ChannelDataset
from scripts.reconstruct_legacy_rt_scene_assets import reconstruct_legacy_scene_assets


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def test_reconstructs_asset_manifest_from_historical_commit_and_locked_wheel(tmp_path) -> None:
    repo = tmp_path / "historical-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "Test")
    (repo / "artifacts").mkdir()
    source_path = repo / "artifacts" / "__init__.py"
    source_path.write_text("SOURCE_MARKER = 1\n", encoding="utf-8")

    wheel_path = tmp_path / "sionna_rt_fixture.whl"
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        wheel.writestr(
            "sionna/rt/scenes/fixture/fixture.xml",
            '<scene version="3.0"><integrator type="path"/></scene>\n',
        )
    wheel_sha256 = hashlib.sha256(wheel_path.read_bytes()).hexdigest()
    (repo / "uv.lock").write_text(
        "[[package]]\n"
        'name = "sionna-rt"\n'
        'version = "2.2.0"\n'
        f'wheels = [{{ url = "fixture.whl", hash = "sha256:{wheel_sha256}" }}]\n',
        encoding="utf-8",
    )
    _git(repo, "add", "artifacts/__init__.py", "uv.lock")
    _git(repo, "commit", "-q", "-m", "fixture")
    revision = _git(repo, "rev-parse", "HEAD")

    from scripts.reconstruct_legacy_rt_scene_assets import _historical_package_source_sha256

    package_source_sha256 = _historical_package_source_sha256(revision, repo_root=repo)
    scene_config = {"name": "fixture", "block_size_m": 20.0}
    dataset_path = tmp_path / "legacy.npz"
    ChannelDataset(
        h=np.ones((1, 1, 8), dtype=np.complex64),
        y=np.zeros(1, dtype=np.uint8),
        group=np.asarray(["legacy-parent"]),
        domain=np.asarray(["rt_fixture"]),
        source=np.asarray(["sionna_rt"]),
        metadata=({"scene": "fixture", "scene_config": scene_config, "rx": [1.0, 2.0, 1.5]},),
        generation_report={
            "environment": {"git_sha": revision, "sionna-rt": "2.2.0"},
            "package_source_sha256": package_source_sha256,
            "scene_config": scene_config,
        },
        label_semantics="fixture",
    ).save(dataset_path)

    manifest = reconstruct_legacy_scene_assets(
        [("fixture", dataset_path)], sionna_rt_wheel=wheel_path, repo_root=repo
    )

    assert manifest["status"] == "NOT_QUALIFIED"
    assert manifest["source_revision"] == revision
    assert manifest["sionna_rt_wheel_sha256"] == wheel_sha256
    assert manifest["datasets"][0]["scene_asset_manifest"]["complete"] is True
    assert manifest["datasets"][0]["scene_asset_manifest"]["scene_xml_path"] == "fixture"
    assert (
        manifest["datasets"][0]["input_sha256"]
        == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    )
