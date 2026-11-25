"""Pilot-sampling command line: bounded JSON input, refusal to overwrite, exit status."""

from __future__ import annotations

import json
import os

import pytest
from experiments import pilot_cli
from tests.test_experiments.test_pilot_sampling_design import _candidates, _design, _outcomes


def _write(path, payload) -> object:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _run(monkeypatch, *argv) -> None:
    monkeypatch.setattr("sys.argv", ["pilot", *map(str, argv)])
    pilot_cli.main()


def test_sample_command_writes_a_complete_roster_and_exits_cleanly(
    tmp_path, monkeypatch, capsys
) -> None:
    design = _write(tmp_path / "design.json", _design(groups=2))
    candidates = _write(tmp_path / "candidates.json", _candidates(groups_per_cell=3))
    output = tmp_path / "sample.json"
    _run(monkeypatch, "sample", "--design", design, "--candidates", candidates, "--output", output)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "SAMPLE_ROSTER_COMPLETE"
    assert capsys.readouterr().out.startswith("SAMPLE_ROSTER_COMPLETE:")


def test_incomplete_sample_is_written_but_exits_nonzero(tmp_path, monkeypatch, capsys) -> None:
    design = _write(tmp_path / "design.json", _design(groups=3))
    candidates = _write(tmp_path / "candidates.json", _candidates(groups_per_cell=2))
    output = tmp_path / "short.json"
    with pytest.raises(SystemExit) as error:
        _run(
            monkeypatch,
            "sample",
            "--design",
            design,
            "--candidates",
            candidates,
            "--output",
            output,
        )
    assert error.value.code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "INCOMPLETE_DESIGN"
    assert "INCOMPLETE_DESIGN" in capsys.readouterr().out


def test_audit_command_binds_outcomes_to_the_frozen_sample(tmp_path, monkeypatch) -> None:
    from experiments.pilot_sampling import build_pilot_sample

    sample = build_pilot_sample(_design(), _candidates())
    sample_path = _write(tmp_path / "sample.json", sample)
    outcomes_path = _write(tmp_path / "outcomes.json", _outcomes(sample))
    output = tmp_path / "audit.json"
    _run(
        monkeypatch,
        "audit",
        "--sample",
        sample_path,
        "--outcomes",
        outcomes_path,
        "--output",
        output,
    )
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "COMPLETE_DIAGNOSTIC"


def test_audit_with_partial_outcomes_exits_nonzero(tmp_path, monkeypatch) -> None:
    from experiments.pilot_sampling import build_pilot_sample

    sample = build_pilot_sample(_design(), _candidates())
    outcomes = _outcomes(sample)
    outcomes["outcomes"] = outcomes["outcomes"][:-1]
    with pytest.raises(SystemExit) as error:
        _run(
            monkeypatch,
            "audit",
            "--sample",
            _write(tmp_path / "s.json", sample),
            "--outcomes",
            _write(tmp_path / "o.json", outcomes),
            "--output",
            tmp_path / "audit.json",
        )
    assert error.value.code == 1


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch) -> None:
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    design = _write(tmp_path / "design.json", _design())
    candidates = _write(tmp_path / "candidates.json", _candidates())
    with pytest.raises(SystemExit) as error:
        _run(
            monkeypatch,
            "sample",
            "--design",
            design,
            "--candidates",
            candidates,
            "--output",
            output,
        )
    assert error.value.code == 2
    assert output.read_text(encoding="utf-8") == "keep"


def test_invalid_inputs_are_reported_through_the_parser(tmp_path, monkeypatch) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    candidates = _write(tmp_path / "candidates.json", _candidates())
    with pytest.raises(SystemExit) as error:
        _run(
            monkeypatch,
            "sample",
            "--design",
            broken,
            "--candidates",
            candidates,
            "--output",
            tmp_path / "out.json",
        )
    assert error.value.code == 2


def test_json_reader_refuses_links_directories_and_oversized_files(tmp_path, monkeypatch) -> None:
    target = _write(tmp_path / "target.json", {"a": 1})
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="single-link regular file"):
        pilot_cli._read_json(alias, label="design")
    hard = tmp_path / "hard.json"
    os.link(target, hard)
    with pytest.raises(ValueError, match="single-link regular file"):
        pilot_cli._read_json(target, label="design")
    directory = tmp_path / "dir"
    directory.mkdir()
    with pytest.raises(ValueError, match="single-link regular file"):
        pilot_cli._read_json(directory, label="design")
    monkeypatch.setattr(pilot_cli, "_MAX_INPUT_BYTES", 4)
    with pytest.raises(ValueError, match="exceeds"):
        pilot_cli._read_json(_write(tmp_path / "big.json", {"abcdef": 1}), label="design")


def test_json_reader_rejects_duplicate_and_nonfinite_values(tmp_path) -> None:
    duplicate = tmp_path / "dup.json"
    duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        pilot_cli._read_json(duplicate, label="design")
    nonfinite = tmp_path / "nan.json"
    nonfinite.write_text('{"a": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        pilot_cli._read_json(nonfinite, label="design")
    invalid = tmp_path / "bytes.json"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="bounded valid UTF-8 JSON"):
        pilot_cli._read_json(invalid, label="design")
    deep = tmp_path / "deep.json"
    deep.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
    with pytest.raises(ValueError, match="bounded valid UTF-8 JSON"):
        pilot_cli._read_json(deep, label="design")


def test_json_reader_detects_file_change_while_opening(tmp_path, monkeypatch) -> None:
    source = _write(tmp_path / "opening.json", {"a": 1})
    identity = pilot_cli._identity
    calls = {"count": 0}

    def changed_identity(metadata):
        calls["count"] += 1
        value = identity(metadata)
        return (0, *value[1:]) if calls["count"] == 2 else value

    monkeypatch.setattr(pilot_cli, "_identity", changed_identity)
    with pytest.raises(ValueError, match="changed while opening"):
        pilot_cli._read_json(source, label="design")


def test_json_reader_detects_file_growth_during_read(tmp_path, monkeypatch) -> None:
    source = tmp_path / "growing.json"
    source.write_bytes(b"x")
    monkeypatch.setattr(pilot_cli, "_MAX_INPUT_BYTES", 4)
    monkeypatch.setattr(pilot_cli.os, "read", lambda *_args: b"12345")

    with pytest.raises(ValueError, match="exceeds the 4-byte limit"):
        pilot_cli._read_json(source, label="design")


def test_json_reader_detects_file_change_while_reading(tmp_path, monkeypatch) -> None:
    source = _write(tmp_path / "reading.json", {"a": 1})
    identity = pilot_cli._identity
    calls = {"count": 0}

    def change_after_read(metadata):
        calls["count"] += 1
        value = identity(metadata)
        return (0, *value[1:]) if calls["count"] == 4 else value

    monkeypatch.setattr(pilot_cli, "_identity", change_after_read)
    with pytest.raises(ValueError, match="changed while being read"):
        pilot_cli._read_json(source, label="design")
