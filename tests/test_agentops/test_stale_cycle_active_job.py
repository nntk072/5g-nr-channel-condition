"""Regression tests for stale_if evaluation, per-cycle review retention, and active-job refusal.

Git-dependent tests run in a temporary repository so the live worktree is never modified.
Process tests start real child processes inside temporary directories and always stop them.
Runner tests isolate the task-state store and log root in tmp_path, as the other runner tests do.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from agentops import runner as runner_module
from agentops.active_jobs import find_active_jobs
from agentops.context import SharedFact
from agentops.contracts import ROOT, ContractError
from agentops.runner import TaskRunner
from agentops.task_state import TaskStateStore

VALIDATION_TASK = "AGENT.INFRA.SELFTEST"

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "agentops-test",
    "GIT_AUTHOR_EMAIL": "agentops-test@example.invalid",
    "GIT_COMMITTER_NAME": "agentops-test",
    "GIT_COMMITTER_EMAIL": "agentops-test@example.invalid",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, env=GIT_ENV, text=True).strip()


def _temp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "data").mkdir()
    (repo / "data" / "source.txt").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    return repo


def _fact(repo: Path, stale_if: list[str]) -> SharedFact:
    return SharedFact.from_mapping(
        {
            "fact": "source file content",
            "source_sha": _git(repo, "rev-parse", "HEAD"),
            "path": "data/source.txt",
            "symbol_or_line_range": "whole file",
            "verified_by": "test",
            "verified_at_utc": "2026-11-22T00:00:00Z",
            "stale_if": stale_if,
            "evidence_digest": "c" * 64,
        },
        repo,
    )


def _isolated_runner(tmp_path: Path) -> TaskRunner:
    runner = TaskRunner(VALIDATION_TASK, repo_root=ROOT)
    runner.config = type(runner.config)(
        task_cards=runner.config.task_cards,
        state_root=runner.config.state_root,
        log_root=str(tmp_path / "logs"),
        provider=None,
        reviewer_provider=None,
    )
    runner.store = TaskStateStore(VALIDATION_TASK, tmp_path / "state", repo_root=ROOT)
    runner.store.create(runner.card.raw_sha256, runner.card.source_revision)
    return runner


def test_stale_if_clean_worktree_is_fresh(tmp_path: Path) -> None:
    repo = _temp_repo(tmp_path)
    fact = _fact(repo, ["data/source.txt"])
    assert fact.changed_stale_if_paths(repo) == ()
    fact.assert_fresh(repo)


def test_stale_if_tracked_change_marks_fact_stale(tmp_path: Path) -> None:
    repo = _temp_repo(tmp_path)
    fact = _fact(repo, ["data/source.txt"])
    (repo / "data" / "source.txt").write_text("v2\n", encoding="utf-8")
    assert fact.changed_stale_if_paths(repo) == ("data/source.txt",)
    with pytest.raises(ContractError, match="stale_if paths changed"):
        fact.assert_fresh(repo)


def test_stale_if_untracked_file_under_directory_marks_fact_stale(tmp_path: Path) -> None:
    repo = _temp_repo(tmp_path)
    fact = _fact(repo, ["data"])
    (repo / "data" / "new.txt").write_text("x\n", encoding="utf-8")
    assert "data/new.txt" in fact.changed_stale_if_paths(repo)


def test_stale_if_ignores_unlisted_changes(tmp_path: Path) -> None:
    repo = _temp_repo(tmp_path)
    (repo / "other.txt").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "other")
    fact = _fact(repo, ["data/source.txt"])
    (repo / "other.txt").write_text("v2\n", encoding="utf-8")
    fact.assert_fresh(repo)


def test_empty_stale_if_never_marks_stale(tmp_path: Path) -> None:
    repo = _temp_repo(tmp_path)
    fact = _fact(repo, [])
    (repo / "data" / "source.txt").write_text("v2\n", encoding="utf-8")
    fact.assert_fresh(repo)


def test_active_job_detects_process_with_cwd_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "owned"
    root.mkdir()
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=root)
    try:
        pids = {job["pid"] for job in find_active_jobs([root])["active"]}
        assert child.pid in pids
        assert os.getpid() not in pids
    finally:
        child.kill()
        child.wait()


def test_active_job_detects_open_file_inside_root(tmp_path: Path) -> None:
    root = tmp_path / "owned"
    root.mkdir()
    target = root / "held.txt"
    target.write_text("held\n", encoding="utf-8")
    script = (
        "import sys, time\nhandle = open(sys.argv[1])\nprint('ready', flush=True)\ntime.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script, str(target)],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        matches = [job for job in find_active_jobs([root])["active"] if job["pid"] == child.pid]
        assert matches
        assert any(hit["kind"] == "fd" for hit in matches[0]["hits"])
    finally:
        child.kill()
        child.wait()


def test_active_job_absent_when_no_process_touches_root(tmp_path: Path) -> None:
    root = tmp_path / "unused"
    root.mkdir()
    assert find_active_jobs([root])["active"] == []


def test_active_job_check_fails_closed_without_process_table(tmp_path: Path) -> None:
    with pytest.raises(ContractError, match="process table"):
        find_active_jobs([tmp_path], proc_root=tmp_path / "missing-proc")


def test_runner_refuses_when_active_job_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _isolated_runner(tmp_path)

    def fake_find(roots: list[Path], **kwargs: object) -> dict[str, object]:
        return {
            "active": [{"pid": 4242, "hits": [{"kind": "cwd", "path": str(roots[0])}]}],
            "uninspected": [],
        }

    monkeypatch.setattr(runner_module, "find_active_jobs", fake_find)
    with pytest.raises(ContractError, match="active process"):
        runner._refuse_active_jobs()


def test_runner_admits_when_no_active_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _isolated_runner(tmp_path)
    monkeypatch.setattr(
        runner_module,
        "find_active_jobs",
        lambda roots, **kwargs: {"active": [], "uninspected": []},
    )
    runner._refuse_active_jobs()


def test_review_cycles_are_numbered_and_retained(tmp_path: Path) -> None:
    runner = _isolated_runner(tmp_path)
    first = runner._new_review_cycle_dir()
    second = runner._new_review_cycle_dir()
    assert first.name == "0001"
    assert second.name == "0002"
    assert first.parent == runner._log_dir() / "review-cycles"


def test_review_cycle_reports_do_not_overwrite_each_other(tmp_path: Path) -> None:
    runner = _isolated_runner(tmp_path)
    first = runner._new_review_cycle_dir()
    second = runner._new_review_cycle_dir()
    runner._write_reviewer_report({"verdict": "REQUEST_CHANGES"}, first)
    runner._write_reviewer_report({"verdict": "APPROVE"}, second)
    assert "REQUEST_CHANGES" in (first / "reviewer-report.md").read_text(encoding="utf-8")
    assert "APPROVE" in (second / "reviewer-report.md").read_text(encoding="utf-8")
