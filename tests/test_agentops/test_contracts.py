import shutil
import subprocess
from pathlib import Path

import pytest
from agentops.contracts import (
    ContractError,
    current_worktree_digest,
    load_strict_json_bytes,
    load_task_card,
    path_state_fingerprint,
)

ROOT = Path(__file__).resolve().parents[2]


def test_curated_selftest_card_loads() -> None:
    card = load_task_card("config/agentops/tasks/AGENT.INFRA.SELFTEST.yaml", ROOT)
    assert card.data["task_id"] == "AGENT.INFRA.SELFTEST"
    assert len(card.raw_sha256) == 64


def test_strict_json_rejects_duplicate_keys_and_nonfinite() -> None:
    with pytest.raises(ContractError, match="duplicate JSON key"):
        load_strict_json_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ContractError, match="non-finite"):
        load_strict_json_bytes(b'{"a":NaN}')


def _tmp_repo(tmp_path: Path) -> Path:
    # Negative task cards are written only inside a temporary repository copy. Tracked
    # config/agentops files are never created, changed, or deleted by these tests.
    repo = tmp_path / "repo"
    (repo / "config/agentops").mkdir(parents=True)
    shutil.copytree(ROOT / "config/agentops/tasks", repo / "config/agentops/tasks")
    (repo / "agentops").mkdir()
    return repo


def test_task_card_rejects_missing_required_field_and_invalid_gate(tmp_path: Path) -> None:
    repo = _tmp_repo(tmp_path)
    source = repo / "config/agentops/tasks/AGENT.INFRA.SELFTEST.yaml"
    original = source.read_text(encoding="utf-8")
    candidate = repo / "config/agentops/tasks/negative.yaml"
    cases = [
        (original.replace("task_id: AGENT.INFRA.SELFTEST\n", ""), "violates schema"),
        (original.replace("  - agent\n", "  - invented-gate\n"), "violates schema"),
        (original.replace("max_fix_attempts: 0", "max_fix_attempts: 3"), "violates schema"),
    ]
    for data, message in cases:
        candidate.write_text(data, encoding="utf-8")
        with pytest.raises(ContractError, match=message):
            load_task_card(candidate, repo)


def test_task_card_rejects_parent_traversal(tmp_path: Path) -> None:
    repo = _tmp_repo(tmp_path)
    source = repo / "config/agentops/tasks/AGENT.INFRA.SELFTEST.yaml"
    data = source.read_text(encoding="utf-8").replace("  - agentops\n", "  - ../escape\n")
    candidate = repo / "config/agentops/tasks/unsafe.yaml"
    candidate.write_text(data, encoding="utf-8")
    with pytest.raises(ContractError, match="unsafe repository-relative path"):
        load_task_card(candidate, repo)


def test_task_card_rejects_absolute_symlink_and_unknown_schema(tmp_path: Path) -> None:
    repo = _tmp_repo(tmp_path)
    source = repo / "config/agentops/tasks/AGENT.INFRA.SELFTEST.yaml"
    original = source.read_text(encoding="utf-8")
    candidate = repo / "config/agentops/tasks/contract.yaml"
    link = repo / "config/agentops/tasks/contract-link"
    cases = [
        (
            original.replace("  - agentops\n", "  - /tmp/escape\n"),
            "unsafe repository-relative path",
        ),
        (
            original.replace(
                "schema: nr-agentops.task-card.v1",
                "schema: nr-agentops.task-card.v99",
            ),
            "violates schema",
        ),
        (
            original.replace("scientific_qualification: false", "scientific_qualification: true"),
            "violates schema",
        ),
    ]
    for data, message in cases:
        candidate.write_text(data, encoding="utf-8")
        with pytest.raises(ContractError, match=message):
            load_task_card(candidate, repo)
    link.symlink_to(repo / "agentops", target_is_directory=True)
    candidate.write_text(
        original.replace(
            "  - agentops\n",
            "  - config/agentops/tasks/contract-link\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractError, match="symlink"):
        load_task_card(candidate, repo)


def test_worktree_digest_tracks_nested_untracked_repository_mutations(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "agentops@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "AgentOps Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("baseline\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)

    nested = repo / ".claude" / "worktrees" / "session-a"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=nested, check=True)
    payload = nested / "payload.txt"
    payload.write_text("before\n", encoding="utf-8")

    first = current_worktree_digest(repo)
    first_nested = path_state_fingerprint(nested)
    payload.write_text("after\n", encoding="utf-8")
    second = current_worktree_digest(repo)
    second_nested = path_state_fingerprint(nested)

    assert first != second
    assert first_nested != second_nested
