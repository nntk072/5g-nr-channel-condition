from pathlib import Path

import pytest
from agentops.context import SharedFact
from agentops.contracts import ContractError, current_git_revision

ROOT = Path(__file__).resolve().parents[2]


def test_shared_fact_binds_current_revision() -> None:
    fact = SharedFact.from_mapping(
        {
            "fact": "queue is the active engineering action list",
            "source_sha": current_git_revision(ROOT),
            "path": "docs/research/EXECUTION_QUEUE.md",
            "symbol_or_line_range": "active execution queue",
            "verified_by": "architect",
            "verified_at_utc": "2026-11-19T00:00:00Z",
            "stale_if": ["docs/research/EXECUTION_QUEUE.md"],
            "evidence_digest": "a" * 64,
        },
        ROOT,
    )
    assert fact.verified_by == "architect"


def test_shared_fact_rejects_stale_revision() -> None:
    with pytest.raises(ContractError, match="stale"):
        SharedFact.from_mapping(
            {
                "fact": "x",
                "source_sha": "0" * 40,
                "path": "AGENTS.md",
                "symbol_or_line_range": "Authority",
                "verified_by": "architect",
                "verified_at_utc": "2026-11-19T00:00:00Z",
                "stale_if": ["AGENTS.md"],
                "evidence_digest": "b" * 64,
            },
            ROOT,
        )


def test_shared_fact_rejects_bad_evidence_digest() -> None:
    with pytest.raises(ContractError, match="evidence_digest"):
        SharedFact.from_mapping(
            {
                "fact": "x",
                "source_sha": current_git_revision(ROOT),
                "path": "AGENTS.md",
                "symbol_or_line_range": "Authority",
                "verified_by": "architect",
                "verified_at_utc": "2026-11-19T00:00:00Z",
                "stale_if": ["AGENTS.md"],
                "evidence_digest": "invalid",
            },
            ROOT,
        )
