"""Validate requirement-ledger structure and plan/backlog coverage."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "docs/research/REQUIREMENT_LEDGER.json"
SCHEMA_PATH = ROOT / "docs/research/schema/requirement-ledger.v1.schema.json"
FEATURE_MAP_PATH = ROOT / "docs/research/IMPLEMENTATION_FEATURE_MAP.json"


def validate_requirement_ledger() -> dict[str, int]:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    feature_map = json.loads(FEATURE_MAP_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(ledger, schema)

    rows = ledger["requirements"]
    identifiers = [row["requirement_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("requirement_id values must be unique")
    aliases = [alias for row in rows for alias in row["backlog_aliases"]]
    if len(aliases) != len(set(aliases)):
        raise ValueError("backlog aliases must be unique")

    expected_backlog_ids = {row["feature_id"] for row in feature_map["features"]}
    indexed_backlog_ids = set(aliases)
    if expected_backlog_ids != indexed_backlog_ids:
        missing = sorted(expected_backlog_ids - indexed_backlog_ids)
        extra = sorted(indexed_backlog_ids - expected_backlog_ids)
        raise ValueError(f"backlog coverage mismatch; missing={missing}, extra={extra}")

    expected_gate_ids = {
        *(f"DS.G{number}" for number in range(9)),
        *(f"RS.Phase{number}" for number in range(11)),
        *(f"RS.AppendixE.G{number}" for number in range(10)),
    }
    indexed_gate_ids = {identifier for identifier in identifiers if identifier in expected_gate_ids}
    if indexed_gate_ids != expected_gate_ids:
        raise ValueError(
            "governing gate coverage mismatch; "
            f"missing={sorted(expected_gate_ids - indexed_gate_ids)}"
        )

    for row in rows:
        for authority in row["authority"]:
            relative_path = authority.split("#", 1)[0]
            if not (ROOT / relative_path).is_file():
                raise ValueError(
                    f"{row['requirement_id']} references missing authority {relative_path}"
                )
    return {
        "requirements": len(rows),
        "backlog_requirements": len(expected_backlog_ids),
        "governing_gates": len(expected_gate_ids),
    }


def main() -> None:
    print(json.dumps(validate_requirement_ledger(), sort_keys=True))


if __name__ == "__main__":
    main()
