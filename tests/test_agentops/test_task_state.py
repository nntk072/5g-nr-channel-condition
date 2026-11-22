from pathlib import Path

import pytest
from agentops.contracts import ContractError, current_git_revision, current_worktree_digest
from agentops.task_state import TaskStateStore

ROOT = Path(__file__).resolve().parents[2]


def test_state_requires_non_skipped_source_bound_receipt_for_done(tmp_path: Path) -> None:
    revision = current_git_revision(ROOT)
    store = TaskStateStore("TASK.STATE", tmp_path)
    store.create("a" * 64, revision, owners=["architect"])
    for phase in ("PREFLIGHT", "IMPLEMENTING", "REVIEWING", "VERIFYING", "REPORTING"):
        store.transition(phase)
    with pytest.raises(ContractError, match="verification receipt"):
        store.transition("DONE")
    receipt = {
        "exit_code": 0,
        "skipped": False,
        "source_revision": revision,
        "worktree_digest": current_worktree_digest(ROOT),
        "task_card_sha256": "a" * 64,
    }
    assert store.transition("DONE", verification_receipt=receipt)["status"] == "done"


def test_attempt_history_is_append_only_and_resume_checks_card(tmp_path: Path) -> None:
    revision = current_git_revision(ROOT)
    store = TaskStateStore("TASK.RESUME", tmp_path)
    store.create("b" * 64, revision)
    store.append_attempt({"kind": "first"})
    store.append_attempt({"kind": "second"})
    state = store.assert_resumable("b" * 64, revision)
    assert [row["kind"] for row in state["attempt_history"]] == ["first", "second"]
    with pytest.raises(ContractError, match="task card changed"):
        store.assert_resumable("c" * 64, revision)


def test_task_state_rejects_unsafe_id_and_detects_stale_done_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    with pytest.raises(ContractError, match="invalid task ID"):
        TaskStateStore("../TASK", tmp_path, repo_root=tmp_path)

    revision = "a" * 40
    digest = {"value": "b" * 64}
    monkeypatch.setattr("agentops.task_state.current_git_revision", lambda _root: revision)
    monkeypatch.setattr(
        "agentops.task_state.current_worktree_digest", lambda _root: digest["value"]
    )
    store = TaskStateStore("TASK.EVIDENCE", tmp_path / "state", repo_root=tmp_path)
    store.create("c" * 64, revision)
    for phase in ("PREFLIGHT", "IMPLEMENTING", "REVIEWING", "VERIFYING", "REPORTING"):
        store.transition(phase)
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    receipt_path = receipt_dir / "final.json"
    receipt = {
        "schema": "nr-agentops.validation-receipt.v1",
        "gate": "fast",
        "command_argv": ["fake-check", "fast"],
        "source_revision": revision,
        "worktree_digest": digest["value"],
        "task_card_sha256": "c" * 64,
        "environment_id": "d" * 64,
        "started_at": "2026-11-19T00:00:00+00:00",
        "ended_at": "2026-11-19T00:00:01+00:00",
        "exit_code": 0,
        "skipped": False,
        "stdout_sha256": "e" * 64,
        "stderr_sha256": "f" * 64,
        "log_path": "receipts/final.stdout.log",
        "scope": "tools/check:fast",
        "provenance": "engineering_verification_only",
    }
    receipt_path.write_text(__import__("json").dumps(receipt), encoding="utf-8")
    store.add_validation_receipt("receipts/final.json")
    store.transition("DONE", verification_receipt=receipt)
    assert store.assert_current_done_evidence("c" * 64)["gate"] == "fast"

    digest["value"] = "0" * 64
    with pytest.raises(ContractError, match="stale for the current worktree"):
        store.assert_current_done_evidence("c" * 64)


def test_state_rejects_unknown_phase_and_atomic_write_failure_preserves_state(
    tmp_path: Path, monkeypatch
) -> None:
    revision = current_git_revision(ROOT)
    store = TaskStateStore("TASK.ATOMIC", tmp_path)
    store.create("d" * 64, revision)
    original = store.state_file.read_bytes()

    with pytest.raises(ContractError, match="unknown task phase"):
        store.transition("NOT_A_PHASE")
    assert store.state_file.read_bytes() == original

    def fail_replace(_source, _destination):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr("artifacts.provenance.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        store.append_attempt({"kind": "must_not_commit"})

    assert store.state_file.read_bytes() == original
    assert not store.lock_file.exists()
    assert not list(store.task_root.glob(".state.json.*"))
