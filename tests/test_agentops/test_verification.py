import sys
from pathlib import Path

import pytest
from agentops.contracts import ContractError, current_git_revision, current_worktree_digest
from agentops.task_state import TaskStateStore
from agentops.verification import run_gate

ROOT = Path(__file__).resolve().parents[2]


def test_scoped_failure_propagates_nonzero_receipt(tmp_path: Path) -> None:
    receipt = run_gate(
        "scoped",
        "a" * 64,
        scoped_argv=[sys.executable, "-c", "raise SystemExit(7)"],
        receipt_dir=tmp_path,
    )
    assert receipt["exit_code"] == 7
    assert receipt["skipped"] is False
    assert receipt["worktree_digest"] == current_worktree_digest(ROOT)


def test_done_rejects_receipt_after_worktree_changes(tmp_path: Path, monkeypatch) -> None:
    revision = current_git_revision(ROOT)
    store = TaskStateStore("TASK.DIGEST", tmp_path, repo_root=ROOT)
    store.create("b" * 64, revision)
    for phase in ("PREFLIGHT", "IMPLEMENTING", "REVIEWING", "VERIFYING", "REPORTING"):
        store.transition(phase)
    receipt = {
        "exit_code": 0,
        "skipped": False,
        "source_revision": revision,
        "worktree_digest": current_worktree_digest(ROOT),
        "task_card_sha256": "b" * 64,
    }
    monkeypatch.setattr("agentops.task_state.current_worktree_digest", lambda _root: "f" * 64)
    with pytest.raises(ContractError, match="stale for the current worktree"):
        store.transition("DONE", verification_receipt=receipt)


def test_scoped_uv_falls_back_to_user_local_binary(tmp_path: Path, monkeypatch) -> None:
    uv = tmp_path / ".local/bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    uv.chmod(0o755)
    monkeypatch.setattr("agentops.verification.shutil.which", lambda _name: None)
    monkeypatch.setattr("agentops.verification.Path.home", lambda *args: tmp_path)

    receipt = run_gate(
        "scoped",
        "c" * 64,
        scoped_argv=["uv", "--version"],
        receipt_dir=tmp_path / "receipts",
    )

    assert receipt["exit_code"] == 0
    assert receipt["command_argv"][0] == str(uv)
