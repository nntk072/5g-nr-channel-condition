"""The requirement ledger validates against its schema, its sources, and its cited paths."""

from __future__ import annotations

from scripts.validate_requirement_ledger import validate_requirement_ledger


def test_requirement_ledger_validates_against_its_sources_and_cited_paths() -> None:
    summary = validate_requirement_ledger()
    assert summary == {
        "backlog_requirements": 182,
        "dataset_gate_criteria": 104,
        "governing_gates": 30,
        "requirements": 316,
        "research_appendix_e_gates": 10,
    }
