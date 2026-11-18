"""Create an immutable, deterministic S00–S11 scene-candidate sample report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import jsonschema
from datasets.scene_families import (
    DEFAULT_REGISTRY,
    ROOT,
    load_scene_family_registry,
    sample_scene_candidates,
)

SCHEMA = ROOT / "datasets/schema/scene-family-sample.v1.schema.json"


def _publish_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidates-per-family", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    try:
        registry = load_scene_family_registry(args.registry)
        report = sample_scene_candidates(
            registry,
            seed=args.seed,
            candidates_per_family=args.candidates_per_family,
        )
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(report)
    except (OSError, ValueError, json.JSONDecodeError, jsonschema.ValidationError) as error:
        parser.error(str(error))
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        _publish_no_replace(output, payload)
    except FileExistsError:
        parser.error(f"refusing to overwrite existing report: {output}")
    print(
        f"{sum(row['shortfall'] == 0 for row in report['family_coverage'])}/12 families "
        f"meet the requested candidate count; NOT_QUALIFIED; sha256={hashlib.sha256(payload).hexdigest()}"
    )
    return 0 if all(row["shortfall"] == 0 for row in report["family_coverage"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
