"""Path-roster command line: bounded input, schema binding and the diagnostic report."""

from __future__ import annotations

import json
import os

import pytest

from evaluation.acceptance import path_equivalence_roster_cli as roster_cli
from evaluation.acceptance.joint_acceptance import build_joint_acceptance_family_v4
from tests.test_evaluation.test_path_equivalence_assurance import _comparison, _record

SCENARIO = ("site-family-a", "LOS", "episode-1")


def _family(**overrides):
    kwargs = {
        "family_id": "protocol-v4",
        "material_expected_strata": [("concrete", "V", "20deg")],
        "complex_csi_expected_strata": [("site-a", "LOS")],
        "propagation_expected_strata": [("site-a", "LOS", "total")],
        "receiver_impact_expected_strata": [("site-a", "LOS")],
        "path_equivalence_expected_scenarios": [SCENARIO],
    }
    kwargs.update(overrides)
    return build_joint_acceptance_family_v4(**kwargs)


def _payload(**overrides) -> dict:
    payload = {
        "schema": "loslab.g3_path_equivalence_roster_input.v1",
        "joint_acceptance_family": _family(),
        "records": [_record("episode-1", _comparison("PASS", 1.0, 1.0))],
    }
    payload.update(overrides)
    return payload


def _write(path, payload) -> object:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_valid_input_produces_the_diagnostic_report(tmp_path, capsys) -> None:
    source = _write(tmp_path / "roster.json", _payload())
    assert roster_cli.main([str(source)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "loslab.g3_path_equivalence_roster_report.v1" or report["status"]
    assert report["status"] == "ROSTER_PATH_COVERAGE_DIAGNOSTIC_PASS"
    assert report["all_expected_episodes_pass"] is True


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema": "other", "joint_acceptance_family": _family(), "records": [{}]}, "input schema must be"),
        ({"schema": "loslab.g3_path_equivalence_roster_input.v1", "joint_acceptance_family": _family()}, "exactly schema"),
        (_payload(records=[]), "records must be a nonempty list"),
        (_payload(records="not-a-list"), "records must be a nonempty list"),
        (_payload(joint_acceptance_family={**_family(), "schema": "other"}), "schema is unsupported"),
    ],
)
def test_input_envelope_is_validated(tmp_path, capsys, payload, message) -> None:
    source = _write(tmp_path / "bad.json", payload)
    with pytest.raises(SystemExit) as error:
        roster_cli.main([str(source)])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_v3_family_is_refused_by_the_v4_roster_cli(tmp_path, capsys, monkeypatch) -> None:
    family_v4 = _family()
    family_v4["schema"] = "loslab.g3_joint_acceptance_family.v3"
    monkeypatch.setattr(roster_cli, "FAMILY_SCHEMA_V4", "loslab.g3_joint_acceptance_family.v4")
    monkeypatch.setattr(roster_cli, "validate_joint_acceptance_family", lambda value: value)
    source = _write(tmp_path / "v3.json", _payload(joint_acceptance_family=family_v4))
    with pytest.raises(SystemExit):
        roster_cli.main([str(source)])
    assert "requires a joint-acceptance family v4" in capsys.readouterr().err


def test_root_must_be_an_object(tmp_path, capsys) -> None:
    source = _write(tmp_path / "list.json", [1, 2])
    with pytest.raises(SystemExit):
        roster_cli.main([str(source)])
    assert "root must be an object" in capsys.readouterr().err


def test_reader_refuses_links_directories_and_duplicate_or_nonfinite_json(tmp_path) -> None:
    target = _write(tmp_path / "target.json", _payload())
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(OSError):
        roster_cli._read_input(alias)
    hard = tmp_path / "hard.json"
    os.link(target, hard)
    with pytest.raises(ValueError, match="single-link regular file"):
        roster_cli._read_input(target)
    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(ValueError, match="single-link regular file"):
        roster_cli._read_input(directory)
    duplicate = tmp_path / "dup.json"
    duplicate.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        roster_cli._read_input(duplicate)
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"schema": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON value"):
        roster_cli._read_input(nonfinite)


def test_reader_enforces_the_size_limit(tmp_path, monkeypatch) -> None:
    source = _write(tmp_path / "big.json", _payload())
    monkeypatch.setattr(roster_cli, "_MAX_INPUT_BYTES", 32)
    with pytest.raises(ValueError, match="size limit"):
        roster_cli._read_input(source)


def test_platform_without_no_follow_open_is_refused(tmp_path, monkeypatch) -> None:
    source = _write(tmp_path / "platform.json", _payload())
    monkeypatch.delattr(roster_cli.os, "O_NOFOLLOW", raising=False)
    with pytest.raises(RuntimeError, match="no-follow input"):
        roster_cli._read_input(source)
