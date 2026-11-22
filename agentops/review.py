"""Independent structured review contracts and bounded fix-loop policy."""

from __future__ import annotations

from typing import Any

from .contracts import ContractError

VERDICTS = {"APPROVE", "REQUEST_CHANGES", "UNABLE_TO_VERIFY"}
_BODY_KEYS = {"reviewer_id", "verdict", "findings"}
_BINDING_KEYS = {"task_id", "source_revision", "task_card_sha256", "worktree_digest"}


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def bind_review(
    value: dict[str, Any],
    *,
    task_id: str,
    source_revision: str,
    task_card_sha256: str,
    worktree_digest: str,
) -> dict[str, Any]:
    """Bind a provider-produced review body to the exact reviewed source state."""
    if set(value) != _BODY_KEYS:
        raise ContractError("review body keys do not match the canonical contract")
    return {
        **value,
        "task_id": task_id,
        "source_revision": source_revision,
        "task_card_sha256": task_card_sha256,
        "worktree_digest": worktree_digest,
    }


def validate_review(
    value: dict[str, Any],
    *,
    author_id: str,
    expected_reviewer_id: str,
    task_id: str | None = None,
    source_revision: str | None = None,
    task_card_sha256: str | None = None,
    worktree_digest: str | None = None,
) -> dict[str, Any]:
    required = _BODY_KEYS | _BINDING_KEYS
    if set(value) != required:
        raise ContractError("review keys do not match the canonical contract")
    reviewer_id = value["reviewer_id"]
    if not isinstance(reviewer_id, str) or not reviewer_id:
        raise ContractError("reviewer_id must be a nonempty string")
    if reviewer_id == author_id:
        raise ContractError("implementation author cannot self-approve")
    if not isinstance(expected_reviewer_id, str) or not expected_reviewer_id:
        raise ContractError("configured reviewer_id must be a nonempty string")
    if reviewer_id != expected_reviewer_id:
        raise ContractError("reviewer_id does not match configured reviewer")
    if not isinstance(value["verdict"], str) or value["verdict"] not in VERDICTS:
        raise ContractError("unknown review verdict")
    if not isinstance(value["task_id"], str) or not value["task_id"]:
        raise ContractError("review task_id must be a nonempty string")
    if not _is_lower_hex(value["source_revision"], 40):
        raise ContractError("review source_revision must be a lowercase Git SHA")
    if not _is_lower_hex(value["task_card_sha256"], 64):
        raise ContractError("review task_card_sha256 must be a lowercase SHA-256")
    if not _is_lower_hex(value["worktree_digest"], 64):
        raise ContractError("review worktree_digest must be a lowercase SHA-256")
    expected = {
        "task_id": task_id,
        "source_revision": source_revision,
        "task_card_sha256": task_card_sha256,
        "worktree_digest": worktree_digest,
    }
    for key, expected_value in expected.items():
        if expected_value is not None and value[key] != expected_value:
            raise ContractError(f"review {key} does not match the current task state")
    findings = value["findings"]
    if not isinstance(findings, list) or not all(
        isinstance(item, dict)
        and set(item) == {"severity", "summary", "evidence"}
        and isinstance(item["severity"], str)
        and item["severity"] in {"info", "low", "medium", "high", "critical"}
        and isinstance(item["summary"], str)
        and item["summary"]
        and isinstance(item["evidence"], str)
        and item["evidence"]
        for item in findings
    ):
        raise ContractError("review findings violate the canonical contract")
    return value


def require_review_approval(
    value: dict[str, Any],
    *,
    author_id: str,
    expected_reviewer_id: str,
    task_id: str | None = None,
    source_revision: str | None = None,
    task_card_sha256: str | None = None,
    worktree_digest: str | None = None,
) -> None:
    review = validate_review(
        value,
        author_id=author_id,
        expected_reviewer_id=expected_reviewer_id,
        task_id=task_id,
        source_revision=source_revision,
        task_card_sha256=task_card_sha256,
        worktree_digest=worktree_digest,
    )
    if review["verdict"] != "APPROVE":
        raise ContractError(f"review did not approve: {review['verdict']}")
    blocking = [item for item in review["findings"] if item["severity"] in {"high", "critical"}]
    if blocking:
        # An approval cannot coexist with blocking findings; the reviewer must request changes.
        raise ContractError("review approval carries high or critical findings")


def allow_fix_cycle(attempt: int, max_fix_attempts: int) -> None:
    if max_fix_attempts < 0 or max_fix_attempts > 2:
        raise ContractError("max_fix_attempts is outside the allowed range")
    if attempt >= max_fix_attempts:
        raise ContractError("bounded fix loop exhausted")
