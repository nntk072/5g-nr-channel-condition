from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from scripts import report_standardized_capabilities as capabilities
from scripts.report_standardized_capabilities import publish_standardized_capability_reports


def test_standardized_capability_report_publisher_writes_schema_valid_deterministic_reports(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "reports"
    first_paths = publish_standardized_capability_reports(output_dir)
    first_bytes = tuple(path.read_bytes() for path in first_paths)

    assert tuple(path.name for path in first_paths) == (
        "standardized-profile-applicability.v1.json",
        "standardized-generator-capabilities.v2.json",
    )
    for report_path, schema_name in zip(
        first_paths,
        (
            "standardized-profile-applicability.v1.schema.json",
            "standardized-generator-capabilities.v2.schema.json",
        ),
        strict=True,
    ):
        schema_path = Path(__file__).parents[2] / "datasets" / "schema" / schema_name
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(
            json.loads(report_path.read_text(encoding="utf-8"))
        )

    second_paths = publish_standardized_capability_reports(output_dir)
    assert second_paths == first_paths
    assert tuple(path.read_bytes() for path in second_paths) == first_bytes


def test_atomic_json_writer_removes_temporary_file_after_replace_failure(
    tmp_path: Path, monkeypatch
) -> None:
    destination = tmp_path / "report.json"

    def fail_replace(_source, _destination):
        raise OSError("destination is read only")

    monkeypatch.setattr(capabilities.os, "replace", fail_replace)
    with pytest.raises(OSError, match="destination is read only"):
        capabilities._write_json_atomic(destination, {"status": "fixture"})

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []
