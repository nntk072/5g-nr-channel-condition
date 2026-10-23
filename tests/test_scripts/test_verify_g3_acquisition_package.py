"""CLI tests for the bounded G3.03 semantic consistency verifier."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

from scripts.verify_g3_acquisition_package import main

FIXTURE_HELPERS = (
    Path(__file__).parents[1] / "test_evaluation/test_g3_acquisition_package_verifier.py"
)
_make_package = runpy.run_path(str(FIXTURE_HELPERS))["_make_package"]


def test_cli_writes_machine_readable_consistency_report(tmp_path: Path, capsys):
    package = _make_package(tmp_path)
    report_path = tmp_path / "verification.json"

    result = main([str(package), "--output", str(report_path)])

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert result == 0
    assert report["status"] == "consistent_proposal_only"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert capsys.readouterr().out == ""


def test_cli_emits_failure_report_and_nonzero_exit_for_inconsistent_package(tmp_path: Path, capsys):
    package = _make_package(tmp_path)
    payload = json.loads(package.read_text(encoding="utf-8"))
    payload["per_file_rights"].pop()
    package.write_text(json.dumps(payload), encoding="utf-8")

    result = main([str(package)])

    report = json.loads(capsys.readouterr().out)
    assert result == 1
    assert report["status"] == "inconsistent"
    assert report["proposal_only"] is True
    assert report["qualification_status"] == "NOT_QUALIFIED"
