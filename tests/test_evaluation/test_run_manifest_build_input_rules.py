"""Joint-evidence run manifest builder: run identity, directory roots, distinct paths and typed contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from evaluation.run_manifest.build import build_joint_evidence_run_manifest


def _files(tmp_path: Path) -> tuple[Path, Path]:
    run_input = tmp_path / "input.json"
    run_input.write_text("{}", encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    return run_input, artifact


@pytest.mark.parametrize("run_id", ["ab", "Upper-Case", "has space", 7])
def test_run_id_pattern_is_enforced(tmp_path: Path, run_id) -> None:
    run_input, artifact = _files(tmp_path)
    with pytest.raises(ValueError, match="run_id must be 3"):
        build_joint_evidence_run_manifest(run_id, run_input, artifact)


def test_package_and_source_roots_must_be_directories(tmp_path: Path) -> None:
    run_input, artifact = _files(tmp_path)
    plain = tmp_path / "plain.txt"
    plain.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="must be directories"):
        build_joint_evidence_run_manifest(
            "run-0001", run_input, artifact, package_root=tmp_path, source_root=plain
        )
