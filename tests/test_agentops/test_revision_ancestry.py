"""Task cards bind to HEAD or an ancestor of HEAD, so committing the work does not stale them."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from agentops.contracts import is_ancestor_revision

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "agentops-test",
    "GIT_AUTHOR_EMAIL": "agentops-test@example.invalid",
    "GIT_COMMITTER_NAME": "agentops-test",
    "GIT_COMMITTER_EMAIL": "agentops-test@example.invalid",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, env=GIT_ENV, text=True).strip()


def _two_commit_repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "a.txt").write_text("two\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "second")
    second = _git(repo, "rev-parse", "HEAD")
    return repo, first, second


def test_parent_revision_is_an_ancestor_of_head(tmp_path: Path) -> None:
    repo, first, second = _two_commit_repo(tmp_path)
    assert is_ancestor_revision(first, second, repo)


def test_head_is_an_ancestor_of_itself(tmp_path: Path) -> None:
    repo, _first, second = _two_commit_repo(tmp_path)
    assert is_ancestor_revision(second, second, repo)


def test_descendant_revision_is_not_an_ancestor(tmp_path: Path) -> None:
    repo, first, second = _two_commit_repo(tmp_path)
    assert not is_ancestor_revision(second, first, repo)


def test_unknown_or_malformed_revision_is_rejected(tmp_path: Path) -> None:
    repo, _first, second = _two_commit_repo(tmp_path)
    assert not is_ancestor_revision("0" * 40, second, repo)
    assert not is_ancestor_revision("HEAD", second, repo)
    assert not is_ancestor_revision("", second, repo)
