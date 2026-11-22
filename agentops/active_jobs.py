"""Detect live processes working inside a task's owned paths before mutation.

Definition used by AgentOps: an active job is another live process whose current working
directory, or any open file descriptor, resolves to a path inside one of the protected roots.
Mutating entry points refuse to run while such a process exists, so unrelated work in progress
is not overwritten. Processes whose ``/proc`` entries are unreadable (other users) cannot be
inspected; they are reported in ``uninspected`` rather than silently treated as idle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .contracts import ContractError


def _inside(candidate: str, roots: list[Path]) -> Path | None:
    if not candidate.startswith("/"):
        return None
    path = Path(candidate.removesuffix(" (deleted)"))
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        return root
    return None


def find_active_jobs(
    roots: list[Path],
    *,
    proc_root: Path = Path("/proc"),
    own_pid: int | None = None,
) -> dict[str, Any]:
    """Return live processes (other than this one) working inside any root.

    Raises ContractError when the process table cannot be enumerated, because an unchecked
    process table must not be treated as empty.
    """
    if not proc_root.is_dir():
        raise ContractError("active-job check needs a readable process table; refusing to proceed")
    resolved = [root.resolve() for root in roots]
    excluded = {os.getpid()} if own_pid is None else {os.getpid(), own_pid}
    jobs: list[dict[str, Any]] = []
    uninspected: list[int] = []
    try:
        entries = sorted(proc_root.iterdir())
    except OSError as error:
        raise ContractError("cannot enumerate the process table") from error
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in excluded:
            continue
        hits: list[dict[str, str]] = []
        try:
            cwd = os.readlink(entry / "cwd")
        except OSError:
            cwd = ""
            uninspected.append(pid)
        root = _inside(cwd, resolved) if cwd else None
        if root is not None:
            hits.append({"kind": "cwd", "path": cwd})
        try:
            descriptors = sorted((entry / "fd").iterdir())
        except OSError:
            descriptors = []
            if pid not in uninspected:
                uninspected.append(pid)
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except OSError:
                continue
            if _inside(target, resolved) is not None:
                hits.append({"kind": "fd", "path": target})
        if hits:
            jobs.append({"pid": pid, "hits": hits})
    return {"active": jobs, "uninspected": uninspected}
