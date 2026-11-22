"""Deterministic engineering handoff construction and validation."""

from __future__ import annotations

from typing import Any

import jsonschema

from .contracts import ContractError, load_schema


def validate_handoff(report: dict[str, Any]) -> dict[str, Any]:
    schema = load_schema("handoff.v1.schema.json")
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(report))
    if errors:
        raise ContractError(
            "handoff violates schema: " + "; ".join(error.message for error in errors[:8])
        )
    return report


def build_handoff(
    *,
    task_id: str,
    engineering_status: str,
    scientific_disposition: str,
    source_revision: str,
    task_card_sha256: str,
    changed_paths: list[str],
    checks: list[dict[str, Any]],
    review: dict[str, Any],
    blockers: list[str],
    next_task: str | None,
) -> dict[str, Any]:
    if not checks:
        raise ContractError("handoff requires at least one verification receipt")
    final_receipt = checks[-1]
    source_digest = final_receipt.get("worktree_digest")
    if not isinstance(source_digest, str):
        raise ContractError("handoff verification receipt is missing worktree_digest")
    review_verdict = review.get("verdict")
    if not isinstance(review_verdict, str):
        raise ContractError("handoff review is missing verdict")
    if engineering_status == "DONE":
        # A DONE handoff must rest on an approval and on gates that actually passed.
        if review_verdict != "APPROVE":
            raise ContractError("DONE handoff requires an APPROVE review verdict")
        for check in checks:
            if check.get("exit_code") != 0 or check.get("skipped") is not False:
                raise ContractError("DONE handoff requires every check to pass and none skipped")
    return validate_handoff(
        {
            "schema": "nr-agentops.handoff.v1",
            "task_id": task_id,
            "base_sha": source_revision,
            "result": engineering_status,
            "changed_files": sorted(set(changed_paths)),
            "source_digest": source_digest,
            "verification_receipt": final_receipt,
            "review_verdict": review_verdict,
            "unresolved_requirements": list(blockers),
            "next_task_id": next_task,
            "limitations": list(blockers),
            "engineering_status": engineering_status,
            "scientific_disposition": scientific_disposition,
            "source_revision": source_revision,
            "task_card_sha256": task_card_sha256,
            "changed_paths": sorted(set(changed_paths)),
            "checks": checks,
            "review": review,
            "blockers": blockers,
            "next_task": next_task,
        }
    )
