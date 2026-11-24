"""The requirement ledger validates against its schema, its sources, and its cited paths."""

from __future__ import annotations

import json

from scripts.validate_requirement_ledger import (
    ROOT,
    developer_local_records,
    developer_local_unverified,
    validate_requirement_ledger,
)


def test_requirement_ledger_validates_against_its_sources_and_cited_paths() -> None:
    summary = validate_requirement_ledger()
    assert summary == {
        "backlog_requirements": 182,
        "dataset_gate_criteria": 104,
        "governing_gates": 30,
        "requirements": 316,
        "research_appendix_e_gates": 10,
    }


def test_developer_local_citations_are_declared_cited_and_checked_where_present() -> None:
    declared = developer_local_records()
    ledger = json.loads((ROOT / "docs/research/REQUIREMENT_LEDGER.json").read_text(encoding="utf-8"))
    cited = {
        authority.split("#", 1)[0]
        for row in ledger["requirements"]
        for authority in row["authority"]
    } | {
        evidence
        for row in ledger["requirements"]
        for evidence in row["implementation"]["evidence"]
    }
    # A declaration without a ledger citation is stale and must be removed.
    assert set(declared) <= cited
    if (ROOT / "dev-docs").is_dir():
        # Developer records are checked against the local folder, so none may be unverified.
        assert developer_local_unverified() == []
    else:
        # A clean checkout cannot see dev-docs, so exactly the declared records are unverified.
        assert developer_local_unverified() == sorted(declared)
