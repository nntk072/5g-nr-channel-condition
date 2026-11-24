"""Standardized capability report command: writes the two hash-bound reports and nothing else."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.report_standardized_capabilities import main


def test_command_writes_both_reports_into_the_requested_directory(tmp_path: Path, capsys) -> None:
    output = tmp_path / "reports"
    status = main(["--output-dir", str(output)])
    assert status == 0
    names = sorted(path.name for path in output.iterdir())
    assert names == [
        "standardized-generator-capabilities.v2.json",
        "standardized-profile-applicability.v1.json",
    ]
    for name in names:
        payload = json.loads((output / name).read_text(encoding="utf-8"))
        assert isinstance(payload, dict) and payload
    assert capsys.readouterr() is not None
