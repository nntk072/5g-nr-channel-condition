"""Joint evidence artifact CLI and file-handling rules: arguments, paths, sizes and exact bytes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation import joint_evidence_artifact as artifact
from tests.test_evaluation.joint_evidence_artifact_support import (
    _install_scorers,
    _payload,
    _write_canonical,
)


def test_cli_evaluate_writes_a_new_artifact_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "inputs.json"
    _write_canonical(source, _payload())
    saved = tmp_path / "bundle.json"

    assert artifact.main(["evaluate", str(source), str(saved)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert set(printed) == {"schema", "status", "bundle_sha256"}
    assert artifact.main(["verify", str(source), str(saved)]) == 0

    assert artifact.main(["evaluate", str(source), str(saved)]) == 2
    assert "File exists" in capsys.readouterr().err


def test_cli_contract_bound_options_are_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "inputs.json"
    _write_canonical(source, _payload())
    with pytest.raises(SystemExit) as missing:
        artifact.main(["evaluate-preregistered", str(source), str(tmp_path / "out.json")])
    assert missing.value.code == 2
    assert "requires --qualification-contract and --run-manifest" in capsys.readouterr().err
    with pytest.raises(SystemExit) as misplaced:
        artifact.main(
            ["evaluate", str(source), str(tmp_path / "out.json"), "--run-manifest", str(source)]
        )
    assert misplaced.value.code == 2
    assert "require evaluate-preregistered" in capsys.readouterr().err


def test_cli_reports_unreadable_inputs_as_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert (
        artifact.main(["verify", str(tmp_path / "missing.json"), str(tmp_path / "bundle.json")])
        == 2
    )
    assert "error:" in capsys.readouterr().err


def test_input_root_must_be_a_top_level_object(tmp_path: Path) -> None:
    source = tmp_path / "inputs.json"
    source.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="joint input root must be an object"):
        artifact.evaluate_joint_evidence_file(source)


def test_input_and_artifact_size_limits_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "inputs.json"
    _write_canonical(source, _payload())
    monkeypatch.setattr(artifact, "MAX_INPUT_BYTES", 10)
    with pytest.raises(ValueError, match="exceeds 10 bytes"):
        artifact.evaluate_joint_evidence_file(source)


def test_reading_refuses_symbolic_links(tmp_path: Path) -> None:
    target = tmp_path / "real.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="single-link regular file"):
        artifact._read_regular_file(link, limit=100, label="G3 joint input")
