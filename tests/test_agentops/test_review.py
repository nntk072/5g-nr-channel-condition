import pytest
from agentops.contracts import ContractError
from agentops.review import allow_fix_cycle, bind_review, require_review_approval, validate_review


def _review(*, reviewer_id: str = "independent_reviewer", verdict: str = "APPROVE") -> dict:
    return bind_review(
        {"reviewer_id": reviewer_id, "verdict": verdict, "findings": []},
        task_id="TASK",
        source_revision="a" * 40,
        task_card_sha256="b" * 64,
        worktree_digest="c" * 64,
    )


def test_reviewer_cannot_self_approve() -> None:
    review = _review(reviewer_id="architect")
    with pytest.raises(ContractError, match="self-approve"):
        validate_review(
            review,
            author_id="architect",
            expected_reviewer_id="architect",
        )


def test_reviewer_must_match_configured_identity() -> None:
    review = _review(reviewer_id="anything")
    with pytest.raises(ContractError, match="configured reviewer"):
        validate_review(
            review,
            author_id="architect",
            expected_reviewer_id="independent_reviewer",
        )


def test_review_veto_and_fix_bound() -> None:
    review = _review(verdict="REQUEST_CHANGES")
    review["findings"] = [{"severity": "high", "summary": "scope leak", "evidence": "diff"}]
    with pytest.raises(ContractError, match="did not approve"):
        require_review_approval(
            review,
            author_id="architect",
            expected_reviewer_id="independent_reviewer",
        )
    allow_fix_cycle(0, 1)
    with pytest.raises(ContractError, match="exhausted"):
        allow_fix_cycle(1, 1)
    with pytest.raises(ContractError, match="outside the allowed range"):
        allow_fix_cycle(0, 3)


def test_review_binding_must_match_current_worktree() -> None:
    review = _review()
    with pytest.raises(ContractError, match="worktree_digest"):
        require_review_approval(
            review,
            author_id="architect",
            expected_reviewer_id="independent_reviewer",
            task_id="TASK",
            source_revision="a" * 40,
            task_card_sha256="b" * 64,
            worktree_digest="d" * 64,
        )
