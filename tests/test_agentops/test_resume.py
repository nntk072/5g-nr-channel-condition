import json
from pathlib import Path

import pytest
from agentops.contracts import ContractError, current_git_revision
from agentops.task_state import TaskStateStore
from artifacts.provenance import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def test_resume_survives_reload_but_binds_workspace(tmp_path: Path) -> None:
    revision = current_git_revision(ROOT)
    store = TaskStateStore("TASK.RELOAD", tmp_path, repo_root=ROOT)
    store.create("a" * 64, revision)
    reloaded = TaskStateStore("TASK.RELOAD", tmp_path, repo_root=ROOT)
    assert reloaded.assert_resumable("a" * 64, revision)["phase"] == "PENDING"

    state = json.loads(reloaded.state_file.read_text(encoding="utf-8"))
    state["workspace_root"] = "/different/workspace"
    atomic_json(reloaded.state_file, state)
    with pytest.raises(ContractError, match="workspace changed"):
        reloaded.assert_resumable("a" * 64, revision)


def test_resume_requires_resource_revalidation_for_native_authority(tmp_path: Path) -> None:
    revision = current_git_revision(ROOT)
    store = TaskStateStore("TASK.NATIVE", tmp_path, repo_root=ROOT)
    store.create("b" * 64, revision)
    store.set_resource_authorization("c" * 64)
    with pytest.raises(ContractError, match="resource revalidation"):
        store.assert_resumable("b" * 64, revision)
    assert (
        store.assert_resumable(
            "b" * 64,
            revision,
            resource_authorization_digest="c" * 64,
        )["phase"]
        == "PENDING"
    )
    with pytest.raises(ContractError, match="resource authorization changed"):
        store.assert_resumable(
            "b" * 64,
            revision,
            resource_authorization_digest="d" * 64,
        )


def test_stale_writer_lock_fails_closed_without_deleting_it(tmp_path: Path) -> None:
    revision = current_git_revision(ROOT)
    store = TaskStateStore("TASK.LOCK", tmp_path, repo_root=ROOT)
    store.create("d" * 64, revision)
    store.lock_file.write_text("999999", encoding="ascii")
    with pytest.raises(ContractError, match="writer lock"):
        store.append_attempt({"kind": "should_not_write"})
    assert store.lock_file.exists()
