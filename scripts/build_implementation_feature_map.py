"""Build a traceable, non-qualifying index from the implementation backlog."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_BACKLOG = Path("docs/research/IMPLEMENTATION_BACKLOG.md")
_DEFAULT_OUTPUT = Path("docs/research/IMPLEMENTATION_FEATURE_MAP.json")
_SCHEMA = "loslab.implementation_feature_map.v1"
_HEADING = re.compile(r"^###\s+(\d+)\.\s+(.+?)\s*$")
_ITEM = re.compile(r"^\s*-\s+\[([ xX])\]\s+(.+?)\s*$")
_PLAN_GATES = re.compile(r"\bG\d+(?:\.\d+)?\b")
_BACKTICK_PATH = re.compile(r"`([^`]+)`")
_PATH_SUFFIXES = {".json", ".md", ".py", ".yaml", ".yml", ".toml", ".lock"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative_path(value: str, root: Path) -> str | None:
    candidate = value.strip().rstrip(".,;:")
    if not candidate or "\\" in candidate:
        return None
    path = Path(candidate)
    if path.as_posix() == _DEFAULT_OUTPUT.as_posix():
        return None
    if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in _PATH_SUFFIXES:
        return None
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return path.as_posix() if resolved.is_file() else None


def _classify_paths(paths: list[str]) -> dict[str, list[str]]:
    result = {
        "source_modules": [],
        "schemas": [],
        "cli_entries": [],
        "regression_files": [],
        "documentation": [],
        "other_files": [],
    }
    for value in paths:
        if value.startswith("datasets/schema/") or value.startswith("evaluation/schema/"):
            bucket = "schemas"
        elif value.startswith("tests/"):
            bucket = "regression_files"
        elif value.startswith("scripts/") and value.endswith(".py"):
            bucket = "cli_entries"
        elif value.startswith("docs/"):
            bucket = "documentation"
        elif value.endswith(".py"):
            bucket = "source_modules"
        else:
            bucket = "other_files"
        result[bucket].append(value)
    return result


def build_feature_map(markdown: str, *, root: Path = _REPOSITORY_ROOT) -> dict[str, Any]:
    """Parse every checklist item and preserve only explicitly linked evidence paths."""
    features: list[dict[str, Any]] = []
    package_number: str | None = None
    package_title: str | None = None
    package_ordinal = 0
    for line_number, line in enumerate(markdown.splitlines(), start=1):
        heading = _HEADING.match(line)
        if heading:
            package_number, package_title = heading.groups()
            package_ordinal = 0
            continue
        item = _ITEM.match(line)
        if not item:
            continue
        if package_number is None or package_title is None:
            raise ValueError(f"checklist item at line {line_number} appears outside a package")
        package_ordinal += 1
        checked, description = item.groups()
        paths = sorted(
            {
                path
                for token in _BACKTICK_PATH.findall(description)
                if (path := _relative_path(token, root)) is not None
            }
        )
        explicit_gates = sorted(set(_PLAN_GATES.findall(description)))
        features.append(
            {
                "feature_id": f"PKG{int(package_number):02d}-{package_ordinal:03d}",
                "backlog_line": line_number,
                "package": {"number": int(package_number), "title": package_title},
                "plan_gates": explicit_gates or [f"backlog-package-{package_number}"],
                "backlog_status": "marked_done" if checked.lower() == "x" else "open",
                "description": description,
                "evidence_links": _classify_paths(paths),
                "link_status": "linked" if paths else "unmapped",
                "qualification_status": "not_assessed_by_inventory",
            }
        )
    if not features:
        raise ValueError("backlog contains no package checklist items")
    return {
        "schema": _SCHEMA,
        "source": {
            "path": _DEFAULT_BACKLOG.as_posix(),
            "sha256": _sha256(markdown.encode("utf-8")),
        },
        "policy": {
            "status_semantics": (
                "backlog_status records only the checkbox state; it is not an independent "
                "implementation audit or scientific qualification"
            ),
            "qualification_claim": "none",
        },
        "summary": {
            "feature_count": len(features),
            "marked_done_count": sum(row["backlog_status"] == "marked_done" for row in features),
            "open_count": sum(row["backlog_status"] == "open" for row in features),
            "linked_count": sum(row["link_status"] == "linked" for row in features),
            "unmapped_count": sum(row["link_status"] == "unmapped" for row in features),
        },
        "features": features,
    }


def _stable_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", type=Path, default=_REPOSITORY_ROOT / _DEFAULT_BACKLOG)
    parser.add_argument("--output", type=Path, default=_REPOSITORY_ROOT / _DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail if the saved map is stale")
    args = parser.parse_args(argv)
    try:
        backlog_path = args.backlog.resolve(strict=True)
        backlog_path.relative_to(_REPOSITORY_ROOT)
        raw = backlog_path.read_bytes()
        feature_map = build_feature_map(raw.decode("utf-8"), root=_REPOSITORY_ROOT)
        rendered = _stable_json(feature_map)
        if args.check:
            if not args.output.is_file() or args.output.read_bytes() != rendered:
                raise ValueError(f"feature map is missing or stale: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(rendered)
        print(json.dumps(feature_map["summary"], sort_keys=True))
        return 0
    except (OSError, UnicodeDecodeError, ValueError) as error:
        print(f"implementation feature map: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
