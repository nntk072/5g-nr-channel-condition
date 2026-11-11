"""Publish schema-validated, deterministic standardized generator scope reports."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jsonschema
from datasets.standardized_specs import (
    standardized_generator_capability_report,
    standardized_profile_applicability_report,
)

_REPORTS = (
    (
        "standardized-profile-applicability.v1.json",
        "standardized-profile-applicability.v1.schema.json",
        standardized_profile_applicability_report,
    ),
    (
        "standardized-generator-capabilities.v2.json",
        "standardized-generator-capabilities.v2.schema.json",
        standardized_generator_capability_report,
    ),
)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def publish_standardized_capability_reports(
    output_dir: Path,
    *,
    schema_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Validate and atomically publish both code-scope reports.

    These reports describe implemented adapter scope only. Their schema checks
    cannot establish statistical conformance or scientific qualification.
    """
    schemas = schema_dir or Path(__file__).resolve().parents[1] / "datasets" / "schema"
    published: list[Path] = []
    for report_name, schema_name, build_report in _REPORTS:
        with (schemas / schema_name).open("r", encoding="utf-8") as stream:
            schema = json.load(stream)
        jsonschema.Draft202012Validator.check_schema(schema)
        report = build_report()
        jsonschema.Draft202012Validator(schema).validate(report)
        destination = output_dir / report_name
        _write_json_atomic(destination, report)
        published.append(destination)
    return tuple(published)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="directory for the two JSON reports"
    )
    args = parser.parse_args(argv)
    for path in publish_standardized_capability_reports(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
