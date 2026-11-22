"""Durable operational state for engineering task execution."""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import jsonschema
from artifacts.provenance import atomic_json

from .contracts import (
    ROOT,
    ContractError,
    current_git_revision,
    current_worktree_digest,
    load_schema,
    load_strict_json_bytes,
)

PHASES = (
    "PENDING",
    "PREFLIGHT",
    "IMPLEMENTING",
    "REVIEWING",
    "FIXING",
    "VERIFYING",
    "REPORTING",
    "DONE",
    "FAILED",
    "BLOCKED",
    "STALE_SOURCE",
    "NEEDS_HUMAN_APPROVAL",
    "DRY_RUN_ONLY",
)
ALLOWED = {
    "PENDING": {"PREFLIGHT", "BLOCKED", "STALE_SOURCE", "DRY_RUN_ONLY"},
    "PREFLIGHT": {"IMPLEMENTING", "BLOCKED", "STALE_SOURCE", "NEEDS_HUMAN_APPROVAL", "FAILED"},
    "IMPLEMENTING": {"REVIEWING", "FAILED", "BLOCKED", "STALE_SOURCE"},
    "REVIEWING": {"FIXING", "VERIFYING", "BLOCKED", "FAILED", "STALE_SOURCE"},
    "FIXING": {"REVIEWING", "FAILED", "BLOCKED", "STALE_SOURCE"},
    "VERIFYING": {"REPORTING", "FIXING", "FAILED", "BLOCKED", "STALE_SOURCE"},
    "REPORTING": {"DONE", "BLOCKED", "FAILED", "STALE_SOURCE"},
    "DONE": set(),
    "FAILED": set(),
    "BLOCKED": set(),
    "STALE_SOURCE": set(),
    "NEEDS_HUMAN_APPROVAL": set(),
    "DRY_RUN_ONLY": set(),
}
TERMINAL = {"DONE", "FAILED", "BLOCKED", "STALE_SOURCE", "NEEDS_HUMAN_APPROVAL", "DRY_RUN_ONLY"}
TASK_ID_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9._-]{2,127}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_for_phase(phase: str) -> str:
    if phase == "DONE":
        return "done"
    if phase == "FAILED":
        return "failed"
    if phase == "BLOCKED":
        return "blocked"
    if phase == "STALE_SOURCE":
        return "stale_source"
    if phase == "NEEDS_HUMAN_APPROVAL":
        return "needs_human_approval"
    if phase == "DRY_RUN_ONLY":
        return "dry_run_only"
    return "active"


class TaskStateStore:
    def __init__(
        self,
        task_id: str,
        state_root: Path | None = None,
        *,
        repo_root: Path = ROOT,
    ) -> None:
        if not TASK_ID_PATTERN.fullmatch(task_id):
            raise ContractError(f"invalid task ID {task_id!r}")
        self.task_id = task_id
        self.repo_root = repo_root
        self.state_root = state_root or (repo_root / "outputs/agentops/tasks")
        self.task_root = self.state_root / task_id
        self.state_file = self.task_root / "state.json"
        self.lock_file = self.task_root / ".writer.lock"
        self._assert_safe_task_root()

    def _assert_safe_task_root(self) -> None:
        try:
            root = self.state_root.resolve(strict=False)
            resolved = self.task_root.resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as error:
            raise ContractError("task state path escapes the configured state root") from error
        if self.task_root.exists() and self.task_root.is_symlink():
            raise ContractError("task state directory may not be a symlink")

    @contextmanager
    def lock(self) -> Iterator[None]:
        self._assert_safe_task_root()
        self.task_root.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(self.lock_file, flags, 0o600)
        except FileExistsError as error:
            raise ContractError(
                f"active task-state writer lock exists for {self.task_id}"
            ) from error
        try:
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.fsync(fd)
            os.close(fd)
            yield
        finally:
            try:
                self.lock_file.unlink()
            except FileNotFoundError:
                pass

    def create(
        self,
        task_card_sha256: str,
        base_revision: str,
        *,
        owners: list[str] | None = None,
        scientific_disposition: str = "NOT_APPLICABLE",
    ) -> dict[str, Any]:
        if self.state_file.exists():
            raise ContractError(f"task state already exists for {self.task_id}")
        now = utc_now()
        state = {
            "schema": "nr-agentops.task-state.v1",
            "task_id": self.task_id,
            "task_card_sha256": task_card_sha256,
            "source_base_sha": base_revision,
            "current_source_sha": current_git_revision(self.repo_root),
            "workspace_root": str(self.repo_root.resolve()),
            "phase": "PENDING",
            "status": "active",
            "attempt": 0,
            "attempt_history": [],
            "owners": owners or [],
            "start_utc": now,
            "last_update_utc": now,
            "validation_receipt_paths": [],
            "resource_authorization_digest": None,
            "provider_calls": [],
            "blocked_reason": None,
            "scientific_disposition": scientific_disposition,
        }
        self._validate(state)
        atomic_json(self.state_file, state)
        return state

    def load(self) -> dict[str, Any]:
        self._assert_safe_task_root()
        try:
            raw = self.state_file.read_bytes()
        except OSError as error:
            raise ContractError(f"cannot read task state for {self.task_id}") from error
        value = load_strict_json_bytes(raw)
        if not isinstance(value, dict):
            raise ContractError("task state must be an object")
        self._validate(value)
        return value

    def transition(
        self,
        target: str,
        *,
        verification_receipt: dict[str, Any] | None = None,
        blocked_reason: str | None = None,
    ) -> dict[str, Any]:
        if target not in PHASES:
            raise ContractError(f"unknown task phase {target}")
        with self.lock():
            state = self.load()
            current = state["phase"]
            if target not in ALLOWED[current]:
                raise ContractError(f"invalid transition {current} -> {target}")
            if target == "DONE":
                if not verification_receipt:
                    raise ContractError("DONE requires a verification receipt")
                if (
                    verification_receipt.get("skipped")
                    or verification_receipt.get("exit_code") != 0
                ):
                    raise ContractError("DONE requires successful non-skipped verification")
                if verification_receipt.get("source_revision") != current_git_revision(
                    self.repo_root
                ):
                    raise ContractError("DONE verification receipt is stale for the current source")
                if verification_receipt.get("worktree_digest") != current_worktree_digest(
                    self.repo_root
                ):
                    raise ContractError(
                        "DONE verification receipt is stale for the current worktree"
                    )
                if verification_receipt.get("task_card_sha256") != state["task_card_sha256"]:
                    raise ContractError(
                        "DONE verification receipt belongs to a different task card"
                    )
            if target in {"BLOCKED", "STALE_SOURCE", "NEEDS_HUMAN_APPROVAL", "FAILED"}:
                if blocked_reason is not None and (
                    not blocked_reason or blocked_reason.strip() != blocked_reason
                ):
                    raise ContractError("blocked_reason must be a nonempty trimmed string")
                state["blocked_reason"] = blocked_reason
            state["phase"] = target
            state["status"] = _status_for_phase(target)
            state["current_source_sha"] = current_git_revision(self.repo_root)
            state["last_update_utc"] = utc_now()
            self._validate(state)
            atomic_json(self.state_file, state)
            return state

    def append_attempt(self, attempt: dict[str, Any]) -> dict[str, Any]:
        with self.lock():
            state = self.load()
            state["attempt"] += 1
            row = dict(attempt)
            row.setdefault("attempt", state["attempt"])
            row.setdefault("recorded_at_utc", utc_now())
            state["attempt_history"].append(row)
            state["last_update_utc"] = utc_now()
            self._validate(state)
            atomic_json(self.state_file, state)
            return state

    def add_validation_receipt(self, relative_path: str) -> dict[str, Any]:
        with self.lock():
            state = self.load()
            state["validation_receipt_paths"].append(relative_path)
            state["last_update_utc"] = utc_now()
            self._validate(state)
            atomic_json(self.state_file, state)
            return state

    def assert_current_done_evidence(self, task_card_sha256: str) -> dict[str, Any]:
        state = self.load()
        if state["phase"] != "DONE":
            raise ContractError("task does not have historical DONE evidence")
        if state["task_card_sha256"] != task_card_sha256:
            raise ContractError("task card changed since DONE evidence was recorded")
        current_revision = current_git_revision(self.repo_root)
        if state["current_source_sha"] != current_revision:
            raise ContractError("DONE evidence is stale for the current source revision")
        paths = state["validation_receipt_paths"]
        if not paths:
            raise ContractError("DONE state has no retained verification receipt")
        raw_path = paths[-1]
        path = Path(raw_path)
        if not path.is_absolute():
            path = self.repo_root / path
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self.repo_root.resolve())
        except (OSError, RuntimeError, ValueError) as error:
            raise ContractError(
                "retained verification receipt path is unsafe or unavailable"
            ) from error
        value = load_strict_json_bytes(resolved.read_bytes())
        if not isinstance(value, dict):
            raise ContractError("retained verification receipt must be a JSON object")
        schema = load_schema("validation-receipt.v1.schema.json")
        errors = list(jsonschema.Draft202012Validator(schema).iter_errors(value))
        if errors:
            raise ContractError(
                "retained verification receipt violates schema: "
                + "; ".join(error.message for error in errors[:8])
            )
        if value["skipped"] or value["exit_code"] != 0:
            raise ContractError("DONE evidence is not a successful non-skipped verification")
        if value["task_card_sha256"] != task_card_sha256:
            raise ContractError("DONE evidence belongs to a different task card")
        if value["source_revision"] != current_revision:
            raise ContractError("DONE evidence is stale for the current source revision")
        if value["worktree_digest"] != current_worktree_digest(self.repo_root):
            raise ContractError("DONE evidence is stale for the current worktree")
        return value

    def add_provider_call(self, receipt: dict[str, Any]) -> dict[str, Any]:
        with self.lock():
            state = self.load()
            state["provider_calls"].append(dict(receipt))
            state["last_update_utc"] = utc_now()
            self._validate(state)
            atomic_json(self.state_file, state)
            return state

    def set_resource_authorization(self, digest: str | None) -> dict[str, Any]:
        with self.lock():
            state = self.load()
            state["resource_authorization_digest"] = digest
            state["last_update_utc"] = utc_now()
            self._validate(state)
            atomic_json(self.state_file, state)
            return state

    def assert_resumable(
        self,
        task_card_sha256: str,
        base_revision: str,
        *,
        resource_authorization_digest: str | None = None,
    ) -> dict[str, Any]:
        state = self.load()
        current = current_git_revision(self.repo_root)
        if state["task_card_sha256"] != task_card_sha256:
            raise ContractError("task card changed since task state was created")
        if state["source_base_sha"] != base_revision or state["current_source_sha"] != current:
            raise ContractError("source revision changed since the last safe checkpoint")
        if state.get("workspace_root") != str(self.repo_root.resolve()):
            raise ContractError("workspace changed since task state was created")
        retained_authorization = state["resource_authorization_digest"]
        if retained_authorization is not None:
            if resource_authorization_digest is None:
                raise ContractError(
                    "resource-authorized task requires explicit resource revalidation before resume"
                )
            if resource_authorization_digest != retained_authorization:
                raise ContractError("resource authorization changed since the last safe checkpoint")
        elif resource_authorization_digest is not None:
            raise ContractError("resume supplied resource authorization for a task that had none")
        if state["phase"] in TERMINAL:
            raise ContractError(f"terminal task state {state['phase']} cannot be resumed")
        return state

    @staticmethod
    def _validate(state: dict[str, Any]) -> None:
        schema = load_schema("task-state.v1.schema.json")
        errors = list(jsonschema.Draft202012Validator(schema).iter_errors(state))
        if errors:
            raise ContractError(
                "task state violates schema: " + "; ".join(error.message for error in errors[:8])
            )
