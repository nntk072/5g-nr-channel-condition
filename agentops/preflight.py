"""Read-only preflight checks for source, task scope, and dirty-path visibility."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .contracts import ROOT, ContractError, LoadedTaskCard, current_git_revision


def dirty_paths(repo_root: Path = ROOT) -> tuple[str, ...]:
    try:
        raw = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ContractError("cannot inspect repository worktree state") from error
    entries = iter(raw.decode("utf-8", errors="strict").split("\0"))
    paths: list[str] = []
    for entry in entries:
        if not entry:
            continue
        if len(entry) < 4 or entry[2] != " ":
            raise ContractError("git status returned malformed porcelain data")
        status = entry[:2]
        paths.append(entry[3:])
        if "R" in status or "C" in status:
            source = next(entries, "")
            if not source:
                raise ContractError("git status returned incomplete rename/copy data")
            paths.append(source)
    return tuple(sorted(paths))


def task_preflight(card: LoadedTaskCard, repo_root: Path = ROOT) -> dict[str, Any]:
    current = current_git_revision(repo_root)
    if current != card.source_revision:
        raise ContractError("task source revision changed before preflight")
    visible_dirty = dirty_paths(repo_root)
    return {
        "task_id": card.data["task_id"],
        "source_revision": current,
        "task_card_sha256": card.raw_sha256,
        "dirty_paths": list(visible_dirty),
        "owned_paths": list(card.data["owned_paths"]),
        "readonly_paths": list(card.data["readonly_paths"]),
        "scientific_disposition": card.data["scientific_disposition"],
    }
