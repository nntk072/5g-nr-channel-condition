"""Curated task queue tied to the existing requirement ledger."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .contracts import ROOT, ContractError, load_strict_json_bytes, load_task_card
from .ownership import validate_ready_ownership
from .task_state import TaskStateStore


@dataclass(frozen=True)
class QueueSnapshot:
    cards: tuple[dict[str, Any], ...]

    def first_ready(self) -> dict[str, Any] | None:
        ready = [
            card for card in self.cards if card.get("effective_status", card["status"]) == "ready"
        ]
        if not ready:
            return None
        return sorted(ready, key=lambda card: (card["priority"], card["task_id"]))[0]

    def by_id(self, task_id: str) -> dict[str, Any]:
        for card in self.cards:
            if card["task_id"] == task_id:
                return card
        raise ContractError(f"unknown task ID {task_id}")


def _ledger_ids(repo_root: Path) -> set[str]:
    try:
        ledger = load_strict_json_bytes(
            (repo_root / "docs/research/REQUIREMENT_LEDGER.json").read_bytes()
        )
    except (OSError, ContractError) as error:
        raise ContractError("cannot load existing requirement ledger") from error
    if not isinstance(ledger, dict):
        raise ContractError("existing requirement ledger has unsupported structure")
    requirements = ledger.get("requirements")
    if not isinstance(requirements, list):
        raise ContractError("existing requirement ledger has unsupported structure")
    result = {row.get("requirement_id") for row in requirements if isinstance(row, dict)}
    return {item for item in result if isinstance(item, str)}


def requirement_anchored(requirement: str, queue_text: str) -> bool:
    """Return whether the requirement ID occurs as a whole identifier.

    A match is rejected when the ID is a prefix of a longer identifier or of a
    dotted sub-identifier, such as ``PKG06-003.1`` or ``PKG06-003.v2``.

    Args:
        requirement: The requirement identifier to search for.
        queue_text: The text of the execution queue.

    Returns:
        True when the identifier appears as a whole identifier in ``queue_text``.
    """
    pattern = rf"(?<![A-Za-z0-9_.-]){re.escape(requirement)}(?![A-Za-z0-9_-]|\.[A-Za-z0-9_])"
    return re.search(pattern, queue_text) is not None


def _execution_queue_text(repo_root: Path) -> str:
    try:
        return (repo_root / "docs/research/EXECUTION_QUEUE.md").read_text(encoding="utf-8")
    except OSError as error:
        raise ContractError("cannot load existing execution queue") from error


def _effective_status(declared: str, phase: str | None) -> str:
    if phase is None:
        return declared
    if phase == "STALE_TASK_CARD":
        return "done" if declared == "done" else "blocked"
    if phase == "DONE":
        return "done"
    return "blocked"


def _check_acyclic(cards: list[dict[str, Any]]) -> None:
    by_id = {card["task_id"]: card for card in cards}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise ContractError(f"task dependency cycle includes {task_id}")
        visiting.add(task_id)
        for dependency in by_id[task_id]["dependencies"]:
            if dependency not in by_id:
                raise ContractError(f"task {task_id} depends on unknown task {dependency}")
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in sorted(by_id):
        visit(task_id)


def load_queue(
    config_path: str | Path = "config/agentops/default.yaml",
    repo_root: Path = ROOT,
    *,
    reconcile_state: bool = True,
) -> QueueSnapshot:
    config = load_config(config_path, repo_root)
    loaded_cards = [load_task_card(card_path, repo_root) for card_path in config.task_cards]
    loaded: list[dict[str, Any]] = []
    for loaded_card in loaded_cards:
        card = dict(loaded_card.data)
        phase = None
        evidence_status = None
        evidence_reason = None
        if reconcile_state:
            store = TaskStateStore(
                card["task_id"],
                repo_root / config.state_root,
                repo_root=repo_root,
            )
            if store.state_file.exists():
                state = store.load()
                if state["task_card_sha256"] != loaded_card.raw_sha256:
                    phase = "STALE_TASK_CARD"
                    evidence_status = "stale"
                    evidence_reason = "task card changed since operational state was recorded"
                else:
                    phase = state["phase"]
                    if phase == "DONE":
                        try:
                            store.assert_current_done_evidence(loaded_card.raw_sha256)
                        except ContractError as error:
                            evidence_status = "stale"
                            evidence_reason = str(error)
                        else:
                            evidence_status = "current"
        card["effective_status"] = _effective_status(card["status"], phase)
        card["operational_phase"] = phase
        card["verification_evidence_status"] = evidence_status
        card["verification_evidence_reason"] = evidence_reason
        loaded.append(card)
    ids = [card["task_id"] for card in loaded]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate task IDs in curated queue")
    ledger_ids = _ledger_ids(repo_root)
    execution_queue = _execution_queue_text(repo_root)
    by_id = {card["task_id"]: card for card in loaded}
    for card in loaded:
        unknown = sorted(set(card["requirement_ids"]) - ledger_ids)
        if unknown:
            raise ContractError(
                f"task {card['task_id']} references unknown requirements: {unknown}"
            )
        missing_queue_anchors = [
            requirement
            for requirement in card["requirement_ids"]
            if not requirement_anchored(requirement, execution_queue)
        ]
        if missing_queue_anchors:
            raise ContractError(
                f"task {card['task_id']} is not anchored in the live execution queue: "
                f"{missing_queue_anchors}"
            )
        if card["effective_status"] == "ready":
            for dependency_id in card["dependencies"]:
                dependency = by_id.get(dependency_id)
                if dependency is None or dependency["effective_status"] != "done":
                    raise ContractError(
                        f"ready task {card['task_id']} has unmet dependency {dependency_id}"
                    )
    _check_acyclic(loaded)
    validate_ready_ownership(loaded)
    return QueueSnapshot(tuple(loaded))
