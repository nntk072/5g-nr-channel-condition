from pathlib import Path
from types import SimpleNamespace

import pytest
from agentops.contracts import ContractError
from agentops.ownership import validate_ready_ownership
from agentops.queue import QueueSnapshot, _check_acyclic, _effective_status, load_queue

ROOT = Path(__file__).resolve().parents[2]


def test_curated_queue_is_deterministic() -> None:
    queue = load_queue(repo_root=ROOT, reconcile_state=False)
    assert queue.by_id("AGENT.INFRA.SELFTEST")["status"] == "done"
    assert queue.by_id("E04.PKG06-003.HOP_EST_DEMAP")["status"] == "ready"
    assert queue.first_ready()["task_id"] == "E04.PKG06-003.HOP_EST_DEMAP"
    assert {card["task_id"] for card in queue.cards} == {
        "AGENT.INFRA.SELFTEST",
        "E04.PKG06-003.HOP_EST_DEMAP",
    }


def test_operational_done_retires_ready_card() -> None:
    assert _effective_status("ready", "DONE") == "done"
    assert _effective_status("ready", "VERIFYING") == "blocked"
    assert _effective_status("ready", "STALE_TASK_CARD") == "blocked"
    assert _effective_status("done", "STALE_TASK_CARD") == "done"
    assert _effective_status("ready", None) == "ready"


def test_queue_cycle_overlap_blocked_and_unmet_dependency(monkeypatch) -> None:
    cycle = [
        {"task_id": "TASK.ONE", "dependencies": ["TASK.TWO"]},
        {"task_id": "TASK.TWO", "dependencies": ["TASK.ONE"]},
    ]
    with pytest.raises(ContractError, match="cycle"):
        _check_acyclic(cycle)

    with pytest.raises(ContractError, match="overlap"):
        validate_ready_ownership(
            [
                {
                    "task_id": "TASK.ONE",
                    "status": "ready",
                    "effective_status": "ready",
                    "owned_paths": ["agentops"],
                },
                {
                    "task_id": "TASK.TWO",
                    "status": "ready",
                    "effective_status": "ready",
                    "owned_paths": ["agentops/runner.py"],
                },
            ]
        )

    snapshot = QueueSnapshot(
        (
            {"task_id": "TASK.BLOCKED", "status": "blocked", "priority": 0},
            {"task_id": "TASK.READY", "status": "ready", "priority": 1},
        )
    )
    assert snapshot.first_ready()["task_id"] == "TASK.READY"

    cards = {
        "one": SimpleNamespace(
            data={
                "task_id": "TASK.ONE",
                "status": "blocked",
                "priority": 1,
                "requirement_ids": ["REQ-1"],
                "dependencies": [],
                "owned_paths": ["agentops/one.py"],
            },
            raw_sha256="a" * 64,
        ),
        "two": SimpleNamespace(
            data={
                "task_id": "TASK.TWO",
                "status": "ready",
                "priority": 2,
                "requirement_ids": ["REQ-1"],
                "dependencies": ["TASK.ONE"],
                "owned_paths": ["agentops/two.py"],
            },
            raw_sha256="b" * 64,
        ),
    }
    monkeypatch.setattr(
        "agentops.queue.load_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            task_cards=("one", "two"), state_root="outputs/agentops/tasks"
        ),
    )
    monkeypatch.setattr("agentops.queue.load_task_card", lambda path, _root: cards[str(path)])
    monkeypatch.setattr("agentops.queue._ledger_ids", lambda _root: {"REQ-1"})
    monkeypatch.setattr("agentops.queue._execution_queue_text", lambda _root: "REQ-1")
    with pytest.raises(ContractError, match="unmet dependency"):
        load_queue(repo_root=ROOT, reconcile_state=False)


def test_queue_rejects_duplicate_ids_and_unknown_requirements(monkeypatch) -> None:
    def loaded(task_id, requirement):
        return SimpleNamespace(
            data={
                "task_id": task_id,
                "status": "blocked",
                "priority": 1,
                "requirement_ids": [requirement],
                "dependencies": [],
                "owned_paths": [f"agentops/{task_id}.py"],
            },
            raw_sha256="a" * 64,
        )

    monkeypatch.setattr(
        "agentops.queue.load_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            task_cards=("one", "two"), state_root="outputs/agentops/tasks"
        ),
    )
    monkeypatch.setattr("agentops.queue._execution_queue_text", lambda _root: "REQ-1 REQ-2")
    monkeypatch.setattr("agentops.queue._ledger_ids", lambda _root: {"REQ-1", "REQ-2"})
    duplicate = {"one": loaded("TASK.ONE", "REQ-1"), "two": loaded("TASK.ONE", "REQ-2")}
    monkeypatch.setattr("agentops.queue.load_task_card", lambda path, _root: duplicate[str(path)])
    with pytest.raises(ContractError, match="duplicate task IDs"):
        load_queue(repo_root=ROOT, reconcile_state=False)

    unknown = {"one": loaded("TASK.ONE", "REQ-404"), "two": loaded("TASK.TWO", "REQ-2")}
    monkeypatch.setattr("agentops.queue.load_task_card", lambda path, _root: unknown[str(path)])
    with pytest.raises(ContractError, match="unknown requirements"):
        load_queue(repo_root=ROOT, reconcile_state=False)
