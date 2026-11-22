"""Canonical verification dispatcher adapter with source-bound receipts."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import jsonschema
from artifacts.provenance import atomic_json

from .contracts import (
    ROOT,
    ContractError,
    current_git_revision,
    current_worktree_digest,
    load_schema,
)

TIMEOUTS = {
    "agent": 300,
    "fast": 900,
    "core": 7200,
    "build": 1200,
    "full": 10800,
    "native-smoke": 1800,
    "scoped": 1800,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _environment_id() -> str:
    payload = f"{os.uname().sysname}|{os.uname().machine}|{os.uname().nodename}".encode()
    return hashlib.sha256(payload).hexdigest()


def validate_receipt(receipt: dict[str, Any]) -> None:
    schema = load_schema("validation-receipt.v1.schema.json")
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(receipt))
    if errors:
        raise ContractError(
            "verification receipt violates schema: "
            + "; ".join(error.message for error in errors[:8])
        )


def _resolve_argv(argv: Sequence[str]) -> list[str]:
    if not argv:
        raise ContractError("verification command is empty")
    executable = argv[0]
    candidate = Path(executable)
    if candidate.is_absolute():
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            raise ContractError(f"verification executable is unavailable: {executable}")
        return list(argv)
    resolved = shutil.which(executable)
    if resolved is None and executable == "uv":
        for fallback in (Path.home() / ".local/bin/uv", Path.home() / ".cargo/bin/uv"):
            if fallback.is_file() and os.access(fallback, os.X_OK):
                resolved = str(fallback)
                break
    if resolved is None:
        raise ContractError(f"verification executable is unavailable: {executable}")
    return [resolved, *argv[1:]]


def _run(
    gate: str,
    argv: Sequence[str],
    task_card_sha256: str,
    *,
    timeout: int,
    scope: str,
    receipt_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_argv = _resolve_argv(argv)
    source_before = current_git_revision(ROOT)
    worktree_before = current_worktree_digest(ROOT)
    started = _now()
    try:
        result = subprocess.run(
            resolved_argv,
            cwd=ROOT,
            shell=False,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise ContractError(f"verification gate {gate} timed out after {timeout}s") from error
    ended = _now()
    source_after = current_git_revision(ROOT)
    worktree_after = current_worktree_digest(ROOT)
    if source_after != source_before or worktree_after != worktree_before:
        raise ContractError("source or worktree state changed while verification was running")
    root = receipt_dir or (ROOT / "outputs/agentops/verification")
    root.mkdir(parents=True, exist_ok=True)
    stem = f"{gate}-{task_card_sha256[:12]}-{hashlib.sha256(started.encode()).hexdigest()[:8]}"
    stdout_path = root / f"{stem}.stdout.log"
    stderr_path = root / f"{stem}.stderr.log"
    stdout_path.write_text(result.stdout[-2_000_000:], encoding="utf-8")
    stderr_path.write_text(result.stderr[-2_000_000:], encoding="utf-8")
    try:
        log_path = stdout_path.relative_to(ROOT).as_posix()
    except ValueError:
        log_path = str(stdout_path)
    receipt = {
        "schema": "nr-agentops.validation-receipt.v1",
        "gate": gate,
        # Persist the executable and the argument count, never argument values (same rule as provider receipts).
        "command_argv": [resolved_argv[0], f"<{len(resolved_argv) - 1} arguments>"]
        if resolved_argv
        else [],
        "source_revision": source_after,
        "worktree_digest": worktree_after,
        "task_card_sha256": task_card_sha256,
        "environment_id": _environment_id(),
        "started_at": started,
        "ended_at": ended,
        "exit_code": result.returncode,
        "skipped": False,
        "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
        "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        "log_path": log_path,
        "scope": scope,
        "provenance": "engineering_verification_only",
    }
    validate_receipt(receipt)
    atomic_json(root / f"{stem}.receipt.json", receipt)
    return receipt


def run_gate(
    gate: str,
    task_card_sha256: str,
    *,
    timeout: int | None = None,
    scoped_argv: Sequence[str] | None = None,
    receipt_dir: Path | None = None,
) -> dict[str, Any]:
    if gate not in TIMEOUTS:
        raise ContractError(f"unknown verification gate {gate}")
    if gate == "scoped":
        if not scoped_argv:
            raise ContractError("scoped verification requires an exact argv")
        argv = list(scoped_argv)
        scope = "task_card_scoped"
    else:
        argv = [str(ROOT / "tools/check"), gate]
        scope = f"tools/check:{gate}"
    return _run(
        gate,
        argv,
        task_card_sha256,
        timeout=timeout or TIMEOUTS[gate],
        scope=scope,
        receipt_dir=receipt_dir,
    )
