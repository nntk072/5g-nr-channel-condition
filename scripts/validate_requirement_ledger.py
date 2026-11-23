"""Validate requirement-ledger structure and plan/backlog coverage."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "docs/research/REQUIREMENT_LEDGER.json"
SCHEMA_PATH = ROOT / "docs/research/schema/requirement-ledger.v1.schema.json"
FEATURE_MAP_PATH = ROOT / "docs/research/IMPLEMENTATION_FEATURE_MAP.json"
UNIFIED_PLAN_PATH = ROOT / "docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md"
# The ledger is frozen against the evidence-base sources. Its criterion text was
# copied from the unified plan at the evidence base, and its Appendix E gate text from the
# master research plan archived at a tag. Both are read with git and checked by digest,
# because the working-tree copies contain later authorised edits and the plan itself is
# no longer in the tree.
EVIDENCE_BASE_REF = "016168e"
UNIFIED_PLAN_PATH_AT_BASE = "docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md"
UNIFIED_PLAN_SHA256 = "3285eb1398ee83ea4d79b4b0116478ceed9d1374537fe2d3510a3c65a7585da2"
RESEARCH_PLAN_ARCHIVE_REF = "docs/master-research-plan-archive"
RESEARCH_PLAN_ARCHIVE_PATH = "docs/research/MASTER_RESEARCH_PLAN.md"
RESEARCH_PLAN_SHA256 = "f30d8f567f97443aec722eabc459843aa4e5821f7574ce67080ab99747c5de93"


def _git_text(ref: str, path: str, sha256: str) -> str:
    """Return the text of ``path`` at ``ref``, checked against its recorded SHA-256."""
    data = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{path} at {ref} does not match its recorded digest")
    return data.decode("utf-8")


def _recorded_moves() -> dict[str, str]:
    payload = json.loads(
        (ROOT / "docs/reference/digest-rebindings.json").read_text(encoding="utf-8")
    )
    return {
        item["old_path"]: item["new_path"]
        for item in payload.get("file_moves", [])
        if item.get("old_path") != item.get("new_path")
    }


def _evidence_exceptions() -> dict[str, list[str]]:
    payload = json.loads(
        (ROOT / "docs/reference/ledger-evidence-exceptions.json").read_text(encoding="utf-8")
    )
    return {item["path"]: item["replaced_by"] for item in payload["entries"]}


def _cited_path_exists(relative_path: str) -> bool:
    """Return True when a frozen citation resolves to a file, through the recorded moves.

    A citation resolves when the path exists, when its recorded move chain ends at an
    existing file, when it names a module that became a package, or when it is an
    explicit exception whose replacements all exist. Developer records moved to the local
    ``dev-docs`` folder resolve when that local file exists.
    """
    if (ROOT / relative_path).is_file():
        return True
    moves = _recorded_moves()
    end, seen = relative_path, {relative_path}
    while end in moves and moves[end] not in seen:
        end = moves[end]
        seen.add(end)
    if (ROOT / end).is_file():
        return True
    if end.endswith(".py") and (ROOT / end[:-3] / "__init__.py").is_file():
        return True
    replacements = _evidence_exceptions().get(relative_path)
    if replacements is not None:
        return all((ROOT / item).is_file() for item in replacements)
    return False


def validate_requirement_ledger() -> dict[str, int]:
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    feature_map = json.loads(FEATURE_MAP_PATH.read_text(encoding="utf-8"))
    unified_plan = _git_text(EVIDENCE_BASE_REF, UNIFIED_PLAN_PATH_AT_BASE, UNIFIED_PLAN_SHA256)
    research_plan = _git_text(
        RESEARCH_PLAN_ARCHIVE_REF, RESEARCH_PLAN_ARCHIVE_PATH, RESEARCH_PLAN_SHA256
    )
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

    planned_criteria = {
        f"DS.{criterion}": ("checked" if mark == "x" else "open", statement.strip())
        for mark, criterion, statement in re.findall(
            r"^\s*- \[([ x])\] \*\*(G\d{1,2}\.\d{2})\*\*\s*(.*)$",
            unified_plan,
            re.MULTILINE,
        )
    }
    indexed_criteria = {
        row["requirement_id"]: row
        for row in rows
        if row["requirement_id"].startswith("DS.G") and "." in row["requirement_id"][5:]
    }
    if set(indexed_criteria) != set(planned_criteria):
        raise ValueError(
            "numbered dataset-gate criterion coverage mismatch; "
            f"missing={sorted(set(planned_criteria) - set(indexed_criteria))}, "
            f"extra={sorted(set(indexed_criteria) - set(planned_criteria))}"
        )
    for identifier, (mark, statement) in planned_criteria.items():
        row = indexed_criteria[identifier]
        if row["plan_mark"] != mark or row["requirement"] != statement:
            raise ValueError(f"{identifier} does not match its source criterion text/status")

    appendix_e_gates = {
        f"RS.AppendixE.G{number}": {
            "title": title.strip(),
            "evidence": evidence.strip(),
            "failure_disposition": failure.strip(),
        }
        for number, title, evidence, failure in re.findall(
            r"^\| G(\d+) ([^|]+) \| ([^|]+) \| ([^|]+) \|$",
            research_plan,
            re.MULTILINE,
        )
    }
    indexed_appendix_gates = {
        row["requirement_id"]: row["requirement"]
        for row in rows
        if row["requirement_id"].startswith("RS.AppendixE.G")
    }
    if set(indexed_appendix_gates) != set(appendix_e_gates):
        raise ValueError("research Appendix E acceptance-gate coverage mismatch")
    if indexed_appendix_gates != appendix_e_gates:
        raise ValueError("research Appendix E acceptance-gate source text mismatch")

    for row in rows:
        for authority in row["authority"]:
            relative_path = authority.split("#", 1)[0]
            if not _cited_path_exists(relative_path):
                raise ValueError(
                    f"{row['requirement_id']} references missing authority {relative_path}"
                )
        for evidence_path in row["implementation"]["evidence"]:
            if not _cited_path_exists(evidence_path):
                raise ValueError(
                    f"{row['requirement_id']} references missing evidence {evidence_path}"
                )
    return {
        "requirements": len(rows),
        "backlog_requirements": len(expected_backlog_ids),
        "governing_gates": len(expected_gate_ids),
        "dataset_gate_criteria": len(planned_criteria),
        "research_appendix_e_gates": len(appendix_e_gates),
    }


def main() -> None:
    print(json.dumps(validate_requirement_ledger(), sort_keys=True))


if __name__ == "__main__":
    main()
