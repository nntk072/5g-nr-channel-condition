"""Source-bound shared facts for agent handoffs."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ContractError, current_git_revision, validate_relative_path


@dataclass(frozen=True)
class SharedFact:
    fact: str
    source_sha: str
    path: str
    symbol_or_line_range: str
    verified_by: str
    verified_at_utc: str
    stale_if: tuple[str, ...]
    evidence_digest: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any], repo_root: Path) -> "SharedFact":
        required = {
            "fact",
            "source_sha",
            "path",
            "symbol_or_line_range",
            "verified_by",
            "verified_at_utc",
            "stale_if",
            "evidence_digest",
        }
        if set(value) != required:
            raise ContractError("shared fact keys do not match the canonical contract")
        strings = [
            value[name]
            for name in (
                "fact",
                "source_sha",
                "path",
                "symbol_or_line_range",
                "verified_by",
                "verified_at_utc",
                "evidence_digest",
            )
        ]
        if not all(isinstance(item, str) and item.strip() == item and item for item in strings):
            raise ContractError("shared fact string fields must be nonempty and trimmed")
        if value["source_sha"] != current_git_revision(repo_root):
            raise ContractError("shared fact is stale for the current source revision")
        if len(value["evidence_digest"]) != 64 or any(
            character not in "0123456789abcdef" for character in value["evidence_digest"]
        ):
            raise ContractError("shared fact evidence_digest must be lowercase SHA-256")
        checked_path = validate_relative_path(value["path"], repo_root)
        stale = value["stale_if"]
        if not isinstance(stale, list) or not all(isinstance(item, str) for item in stale):
            raise ContractError("stale_if must be a list of repository-relative paths")
        checked = tuple(validate_relative_path(item, repo_root) for item in stale)
        return cls(
            fact=value["fact"],
            source_sha=value["source_sha"],
            path=checked_path,
            symbol_or_line_range=value["symbol_or_line_range"],
            verified_by=value["verified_by"],
            verified_at_utc=value["verified_at_utc"],
            stale_if=checked,
            evidence_digest=value["evidence_digest"],
        )

    def changed_stale_if_paths(self, repo_root: Path) -> tuple[str, ...]:
        """Return the stale_if paths that differ from the bound revision in the worktree.

        A path counts as changed when git reports a tracked difference from HEAD (the bound
        revision, enforced by ``from_mapping``) or an untracked file under it. An empty
        ``stale_if`` list never marks the fact stale.
        """
        if not self.stale_if:
            return ()
        pathspec = ["--", *self.stale_if]
        try:
            tracked = subprocess.check_output(
                ["git", "diff", "--name-only", "-z", "--no-renames", "HEAD", *pathspec],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            untracked = subprocess.check_output(
                ["git", "ls-files", "-o", "--exclude-standard", "-z", *pathspec],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            raise ContractError("cannot evaluate stale_if for shared fact") from error
        changed = {
            item
            for raw in (tracked, untracked)
            for item in raw.decode("utf-8", errors="strict").split("\0")
            if item
        }
        return tuple(sorted(changed))

    def assert_fresh(self, repo_root: Path) -> None:
        """Fail closed when any stale_if path changed since the bound revision."""
        changed = self.changed_stale_if_paths(repo_root)
        if changed:
            raise ContractError(
                f"shared fact {self.fact!r} is stale: stale_if paths changed: {list(changed)}"
            )
