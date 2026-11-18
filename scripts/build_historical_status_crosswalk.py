"""Build a source-bound inventory of dated research and MVP status notes.

The index records only exact requirement IDs already present in the ledger.
It does not infer current status, scientific qualification, or gate equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = Path("docs/research/REQUIREMENT_LEDGER.json")
SCHEMA_PATH = Path("docs/research/schema/historical-status-crosswalk.v1.schema.json")
OUTPUT_PATH = Path("docs/research/HISTORICAL_STATUS_CROSSWALK.json")
MANUAL_MAPPINGS_PATH = Path("docs/research/HISTORICAL_STATUS_MANUAL_MAPPINGS.json")
MANUAL_SCHEMA_PATH = Path("docs/research/schema/historical-status-manual-mappings.v1.schema.json")
SOURCES = (
    Path("docs/research/MASTER_RESEARCH_PLAN.md"),
    Path("docs/research/MVP_STATUS.md"),
)
_DATE = re.compile(r"\((\d{4}-\d{2}-\d{2})(?:\b|,)")
_BOLD = re.compile(r"^\*\*(.+?)\*\*")
_EXACT_ID = re.compile(r"\b(?:PKG\d{2}-\d{3}|DS\.G\d+\.\d{2}|RS\.Phase\d+|RS\.AppendixE\.G\d+)\b")
_DATASET_CRITERION = re.compile(r"\bG(\d+)\.(\d{2})\b")
_MARKDOWN_LINK = re.compile(r"\[[^\]]+\]\(([^)#]+)(?:#[^)]*)?\)")
_BACKTICK = re.compile(r"`([^`]+)`")
_BARE_REPO_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])((?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
    r"\.(?:json|jsonl|md|py|yaml|yml|toml|csv|npz))(?![A-Za-z0-9_-])"
)
_PATH_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".yaml", ".yml", ".toml", ".csv", ".npz"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _requirement_index(ledger: dict[str, Any]) -> dict[str, str]:
    index: dict[str, str] = {}
    for row in ledger["requirements"]:
        index[row["requirement_id"]] = row["requirement_id"]
        for alias in row["backlog_aliases"]:
            index[alias] = row["requirement_id"]
    return index


def _load_manual_mappings(
    root: Path, id_index: dict[str, str]
) -> tuple[dict[str, dict[str, Any]], str]:
    path = root / MANUAL_MAPPINGS_PATH
    raw = path.read_bytes()
    payload = json.loads(raw)
    schema = json.loads((root / MANUAL_SCHEMA_PATH).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    if not isinstance(payload, dict) or set(payload) != {"schema", "entries"}:
        raise ValueError("manual crosswalk mappings must contain schema and entries")
    if payload["schema"] != "loslab.historical_status_manual_mappings.v1":
        raise ValueError("unsupported manual crosswalk mapping schema")
    if not isinstance(payload["entries"], list):
        raise ValueError("manual crosswalk mappings entries must be an array")
    result: dict[str, dict[str, Any]] = {}
    for row in payload["entries"]:
        if not isinstance(row, dict) or set(row) != {
            "entry_key",
            "heading_sha256",
            "block_sha256",
            "requirement_ids",
            "rationale",
        }:
            raise ValueError("manual crosswalk mapping fields are invalid")
        entry_key = row["entry_key"]
        heading_sha256 = row["heading_sha256"]
        block_sha256 = row["block_sha256"]
        rationale = row["rationale"]
        ids = row["requirement_ids"]
        if (
            not isinstance(entry_key, str)
            or not re.fullmatch(r"[0-9a-f]{64}", entry_key)
            or entry_key in result
        ):
            raise ValueError("manual crosswalk entry keys must be unique lowercase SHA-256 values")
        if not isinstance(heading_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", heading_sha256):
            raise ValueError("manual crosswalk mapping must pin the exact heading digest")
        if not isinstance(block_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", block_sha256):
            raise ValueError("manual crosswalk mapping must pin the exact note-block digest")
        if not isinstance(rationale, str) or len(rationale.strip()) < 20:
            raise ValueError("manual crosswalk rationale must contain at least 20 characters")
        if (
            not isinstance(ids, list)
            or not ids
            or any(not isinstance(item, str) or id_index.get(item) != item for item in ids)
            or len(ids) != len(set(ids))
        ):
            raise ValueError("manual crosswalk mappings must name unique existing requirement IDs")
        result[entry_key] = {
            "heading_sha256": heading_sha256,
            "block_sha256": block_sha256,
            "requirement_ids": sorted(ids),
            "rationale": rationale.strip(),
        }
    return result, _sha256(raw)


def _entry_starts(lines: list[str]) -> list[int]:
    starts = []
    for index, line in enumerate(lines):
        if line.startswith("**") and _DATE.search(line) and _BOLD.match(line):
            starts.append(index)
    return starts


def _evidence_paths(block: str, *, root: Path, source: Path) -> list[str]:
    found: set[str] = set()
    candidates = [
        (_MARKDOWN_LINK.findall(block), source.parent),
        (_BACKTICK.findall(block), Path(".")),
        (_BARE_REPO_PATH.findall(block), Path(".")),
    ]
    for values, base in candidates:
        for candidate in values:
            value = candidate.strip().rstrip(".,;:")
            path = Path(value.split("#", 1)[0])
            if not value or path.is_absolute() or path.suffix.lower() not in _PATH_SUFFIXES:
                continue
            joined = path if path.is_absolute() else base / path
            resolved = (root / joined).resolve()
            try:
                relative = resolved.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if resolved.is_file():
                found.add(relative)
    return sorted(found)


def build_crosswalk(*, root: Path = ROOT) -> dict[str, Any]:
    """Inventory dated notes with exact aliases and existing cited-file paths."""
    ledger_path = root / LEDGER_PATH
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    id_index = _requirement_index(ledger)
    manual_mappings, manual_mappings_sha256 = _load_manual_mappings(root, id_index)
    entries: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for source in SOURCES:
        source_path = root / source
        raw = source_path.read_bytes()
        lines = raw.decode("utf-8").splitlines()
        starts = _entry_starts(lines)
        spans = [
            (start, starts[position + 1] if position + 1 < len(starts) else len(lines))
            for position, start in enumerate(starts)
        ]
        sources.append({"path": source.as_posix(), "sha256": _sha256(raw)})
        for start, stop in spans:
            first = lines[start]
            title_match = _BOLD.match(first)
            date_match = _DATE.search(first)
            if title_match is None or date_match is None:
                continue
            block = "\n".join(lines[start:stop])
            aliases = sorted(set(_EXACT_ID.findall(block)))
            dataset_criteria = sorted(
                {
                    f"G{major}.{minor}"
                    for major, minor in _DATASET_CRITERION.findall(block)
                    if f"DS.G{major}.{minor}" in id_index
                }
            )
            matched_tokens = {*aliases, *dataset_criteria}
            requirement_ids = sorted(
                {
                    id_index[token] if token in id_index else f"DS.{token}"
                    for token in matched_tokens
                }
            )
            entry_key = _sha256(
                f"{source.as_posix()}\0{start + 1}\0{title_match.group(1)}".encode("utf-8")
            )
            manual = manual_mappings.pop(entry_key, None)
            if manual is not None:
                if manual["heading_sha256"] != _sha256(title_match.group(1).encode("utf-8")):
                    raise ValueError(f"manual mapping heading digest is stale: {entry_key}")
                if manual["block_sha256"] != _sha256(block.encode("utf-8")):
                    raise ValueError(f"manual mapping note-block digest is stale: {entry_key}")
                if set(manual["requirement_ids"]).intersection(requirement_ids):
                    raise ValueError(
                        f"manual mapping duplicates an exact source match: {entry_key}"
                    )
                requirement_ids = sorted({*requirement_ids, *manual["requirement_ids"]})
            entries.append(
                {
                    "entry_key": entry_key,
                    "source_path": source.as_posix(),
                    "line": start + 1,
                    "date": date_match.group(1),
                    "heading": title_match.group(1),
                    "heading_sha256": _sha256(title_match.group(1).encode("utf-8")),
                    "direct_requirement_aliases": aliases,
                    "numbered_dataset_criteria": dataset_criteria,
                    "requirement_ids": requirement_ids,
                    **(
                        {
                            "manual_mapping": {
                                "rationale": manual["rationale"],
                                "heading_sha256": manual["heading_sha256"],
                                "block_sha256": manual["block_sha256"],
                                "requirement_ids": manual["requirement_ids"],
                            }
                        }
                        if manual is not None
                        else {}
                    ),
                    "cited_existing_paths": _evidence_paths(block, root=root, source=source),
                    "mapping_state": (
                        "resolved_manual_review"
                        if manual is not None
                        else "resolved_exact_match"
                        if requirement_ids
                        else "needs_manual_mapping"
                    ),
                    "current_status_authority": "docs/research/REQUIREMENT_LEDGER.json",
                }
            )
    if manual_mappings:
        raise ValueError(
            "manual mappings reference entries absent from current sources: "
            + ", ".join(sorted(manual_mappings))
        )
    entries.sort(key=lambda row: (row["source_path"], row["line"]))
    payload = {
        "schema": "loslab.historical_status_crosswalk.v1",
        "authority_note": "Inventory only: historical notes remain unchanged; exact ID matches are candidates for review, and no current status or qualification is inferred.",
        "sources": sorted(sources, key=lambda row: row["path"]),
        "source_ledger_sha256": _sha256(ledger_path.read_bytes()),
        "manual_mappings_sha256": manual_mappings_sha256,
        "entry_count": len(entries),
        "resolved_match_count": sum(
            row["mapping_state"] == "resolved_exact_match" for row in entries
        ),
        "resolved_manual_review_count": sum(
            row["mapping_state"] == "resolved_manual_review" for row in entries
        ),
        "needs_manual_mapping_count": sum(
            row["mapping_state"] == "needs_manual_mapping" for row in entries
        ),
        "entries": entries,
    }
    schema = json.loads((root / SCHEMA_PATH).read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(payload, schema)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--check", action="store_true", help="fail if the saved crosswalk is stale")
    args = parser.parse_args(argv)
    payload = build_crosswalk()
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    output = ROOT / args.output
    if args.check:
        try:
            current = output.read_bytes()
        except OSError:
            print(f"historical crosswalk is missing: {output}", file=sys.stderr)
            return 1
        if current != encoded:
            print(f"historical crosswalk is stale: {output}", file=sys.stderr)
            return 1
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "entries": payload["entry_count"],
                "resolved_exact_matches": payload["resolved_match_count"],
                "resolved_manual_review": payload["resolved_manual_review_count"],
                "needs_manual_mapping": payload["needs_manual_mapping_count"],
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
