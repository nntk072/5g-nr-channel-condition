"""Finite task execution flow for reviewed engineering task cards."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from artifacts.provenance import atomic_json

from .active_jobs import find_active_jobs
from .config import AgentConfig, load_config
from .contracts import (
    ROOT,
    ContractError,
    LoadedTaskCard,
    canonical_json_bytes,
    current_git_revision,
    current_worktree_digest,
    load_strict_json_bytes,
    load_task_card,
    path_state_fingerprint,
)
from .ownership import paths_overlap
from .preflight import dirty_paths, task_preflight
from .provider import ProviderError, run_provider
from .reporting import build_handoff, validate_handoff
from .resources import (
    LOCAL_CLASSES,
    ResourceUsageLedger,
    authorize_resource_class,
    host_inventory,
    metered_gpu_execution,
)
from .review import allow_fix_cycle, bind_review, require_review_approval, validate_review
from .task_state import TaskStateStore
from .telemetry import append_event, normalize_provider_usage
from .verification import TIMEOUTS, run_gate, validate_receipt


def _card_path(task_id: str, config: AgentConfig, repo_root: Path = ROOT) -> str:
    for path in config.task_cards:
        card = load_task_card(path, repo_root)
        if card.data["task_id"] == task_id:
            return path
    raise ContractError(f"unknown task ID {task_id}")


class TaskRunner:
    def __init__(
        self,
        task_id: str,
        *,
        repo_root: Path = ROOT,
        config_path: str = "config/agentops/default.yaml",
    ) -> None:
        self.repo_root = repo_root
        self.config = load_config(config_path, repo_root)
        self.card: LoadedTaskCard = load_task_card(
            _card_path(task_id, self.config, repo_root), repo_root
        )
        self.task_id = task_id
        self.state_root = repo_root / self.config.state_root
        self.store = TaskStateStore(task_id, self.state_root, repo_root=self.repo_root)

    def plan(self) -> dict[str, Any]:
        preflight = task_preflight(self.card, self.repo_root)
        return {
            "task": self.card.data,
            "task_card_sha256": self.card.raw_sha256,
            "preflight": preflight,
            "provider_configured": self.config.provider is not None,
        }

    def dry_run(self) -> dict[str, Any]:
        plan = self.plan()
        dry_store = TaskStateStore(
            self.task_id,
            self.repo_root / "outputs/agentops/dry-runs",
            repo_root=self.repo_root,
        )
        if not dry_store.state_file.exists():
            dry_store.create(
                self.card.raw_sha256,
                self.card.source_revision,
                owners=[self.card.data["owner_id"]],
                scientific_disposition=self.card.data["scientific_disposition"],
            )
            dry_store.transition("DRY_RUN_ONLY")
        plan["engineering_status"] = "DRY_RUN_ONLY"
        return plan

    def _ensure_state(self) -> None:
        if not self.store.state_file.exists():
            self.store.create(
                self.card.raw_sha256,
                self.card.source_revision,
                owners=[self.card.data["owner_id"]],
                scientific_disposition=self.card.data["scientific_disposition"],
            )
        self._ensure_packet()

    @staticmethod
    def _atomic_bytes(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def _log_dir(self) -> Path:
        state = self.store.load()
        date = state["start_utc"][:10].replace("-", "")
        return (
            self.repo_root
            / self.config.log_root
            / self.card.data["owner_id"]
            / f"{date}_{self.task_id}"
        )

    def _ensure_packet(self) -> None:
        task_root = self.store.task_root
        snapshot = task_root / "task-card.snapshot.yaml"
        if not snapshot.exists():
            self._atomic_bytes(snapshot, self.card.path.read_bytes())
        ownership = task_root / "ownership.json"
        if not ownership.exists():
            atomic_json(
                ownership,
                {
                    "task_id": self.task_id,
                    "source_revision": self.card.source_revision,
                    "task_card_sha256": self.card.raw_sha256,
                    "owner_id": self.card.data["owner_id"],
                    "reviewer_id": self.card.data["reviewer_id"],
                    "owned_paths": self.card.data["owned_paths"],
                    "readonly_paths": self.card.data["readonly_paths"],
                },
            )
        facts = task_root / "facts.jsonl"
        if not facts.exists():
            self._atomic_bytes(facts, b"")
        (task_root / "phase-receipts").mkdir(parents=True, exist_ok=True)
        log_dir = self._log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        execution_log = log_dir / "execution.log"
        if not execution_log.exists():
            self._atomic_bytes(execution_log, b"")

    def _write_source_diff(self) -> None:
        context = self._review_source_context()
        self._write_review_source_context(context)

    def _write_review_source_context(
        self, context: dict[str, Any], directory: Path | None = None
    ) -> None:
        target = directory or self._log_dir()
        self._atomic_bytes(target / "source-diff.patch", context["diff"].encode("utf-8"))
        atomic_json(target / "review-source.json", context)

    def _new_review_cycle_dir(self) -> Path:
        """Allocate a numbered directory that keeps one review cycle's artifacts.

        Each review call gets its own directory, so earlier cycles' source diffs and reports
        are not overwritten by later ones. The top-level files remain the latest copy.
        """
        base = self._log_dir() / "review-cycles"
        base.mkdir(parents=True, exist_ok=True)
        existing = [int(item.name) for item in base.iterdir() if item.name.isdigit()]
        cycle = base / f"{max(existing, default=0) + 1:04d}"
        cycle.mkdir()
        return cycle

    def _write_reviewer_report(self, review: dict[str, Any], directory: Path | None = None) -> None:
        body = (
            "# Independent Review\n\n```json\n"
            + json.dumps(review, indent=2, sort_keys=True)
            + "\n```\n"
        )
        target = directory or self._log_dir()
        self._atomic_bytes(target / "reviewer-report.md", body.encode("utf-8"))

    def _telemetry(self, phase: str, event: str, **fields: Any) -> None:
        row = {
            "task_id": self.task_id,
            "source_sha": current_git_revision(self.repo_root),
            "task_card_sha256": self.card.raw_sha256,
            "role": self.card.data["owner_id"],
            "phase": phase,
            "event": event,
            "scientific_disposition": self.card.data["scientific_disposition"],
            **fields,
        }
        append_event(self.store.task_root / "telemetry.jsonl", row)
        append_event(self._log_dir() / "execution.log", row)

    def _is_owned(self, path: str) -> bool:
        return any(paths_overlap(owner, path) for owner in self.card.data["owned_paths"])

    def _path_fingerprint(self, path: str) -> str:
        return path_state_fingerprint(self.repo_root / path)

    def _snapshot_unowned_dirty(self, paths: tuple[str, ...]) -> dict[str, str]:
        return {path: self._path_fingerprint(path) for path in paths if not self._is_owned(path)}

    def _owned_dirty_paths(self) -> list[str]:
        return sorted(path for path in dirty_paths(self.repo_root) if self._is_owned(path))

    def _review_path_record(self, path: str, *, include_content: bool) -> dict[str, Any]:
        candidate = self.repo_root / path
        if candidate.is_symlink():
            target = os.readlink(candidate)
            target_bytes = os.fsencode(target)
            return {
                "path": path,
                "kind": "symlink",
                "mode": "120000",
                "sha256": hashlib.sha256(target_bytes).hexdigest(),
                "target": target,
            }
        if not candidate.exists():
            return {
                "path": path,
                "kind": "missing",
                "mode": None,
                "sha256": hashlib.sha256(b"MISSING").hexdigest(),
            }
        if not candidate.is_file():
            raise ContractError(f"cannot represent changed review path: {path}")
        raw = candidate.read_bytes()
        record: dict[str, Any] = {
            "path": path,
            "kind": "file",
            "mode": f"{candidate.stat().st_mode & 0o777:o}",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if include_content:
            record["content_base64"] = base64.b64encode(raw).decode("ascii")
        return record

    def _review_source_context(self) -> dict[str, Any]:
        diff_argv = [
            "git",
            "diff",
            "--no-ext-diff",
            "--binary",
            "HEAD",
            "--",
            *self.card.data["owned_paths"],
        ]
        try:
            result = subprocess.run(
                diff_argv,
                cwd=self.repo_root,
                shell=False,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ContractError("cannot capture provider review source diff") from error
        if result.returncode != 0:
            raise ContractError("cannot capture provider review source diff")
        untracked_argv = [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            *self.card.data["owned_paths"],
        ]
        try:
            untracked_result = subprocess.run(
                untracked_argv,
                cwd=self.repo_root,
                shell=False,
                capture_output=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ContractError("cannot capture provider review untracked paths") from error
        if untracked_result.returncode != 0:
            raise ContractError("cannot capture provider review untracked paths")
        try:
            untracked_paths = {
                item
                for item in untracked_result.stdout.decode("utf-8", errors="strict").split("\0")
                if item
            }
        except UnicodeDecodeError as error:
            raise ContractError("provider review path names are not valid UTF-8") from error
        changed_paths = self._owned_dirty_paths()
        path_records = [
            self._review_path_record(path, include_content=path in untracked_paths)
            for path in changed_paths
        ]
        represented = {record["path"] for record in path_records}
        if represented != set(changed_paths) or not untracked_paths.issubset(represented):
            raise ContractError(
                "provider review source context does not represent every changed path"
            )
        context = {
            "task_id": self.task_id,
            "source_revision": current_git_revision(self.repo_root),
            "task_card_sha256": self.card.raw_sha256,
            "worktree_digest": current_worktree_digest(self.repo_root),
            "changed_paths": changed_paths,
            "diff": result.stdout.decode("utf-8", errors="replace"),
            "path_records": path_records,
            "contract_tests": self.card.data.get("scoped_verification", []),
        }
        if len(canonical_json_bytes(context)) > 1_000_000:
            raise ContractError("provider review source context exceeds the configured limit")
        return context

    def _validate_current_review(self, review: dict[str, Any], *, author_id: str) -> dict[str, Any]:
        if current_git_revision(self.repo_root) != self.card.source_revision:
            raise ContractError("review source revision does not match the current Git revision")
        return validate_review(
            review,
            author_id=author_id,
            expected_reviewer_id=self.card.data["reviewer_id"],
            task_id=self.task_id,
            source_revision=self.card.source_revision,
            task_card_sha256=self.card.raw_sha256,
            worktree_digest=current_worktree_digest(self.repo_root),
        )

    def _require_current_review_approval(self, review: dict[str, Any], *, author_id: str) -> None:
        require_review_approval(
            review,
            author_id=author_id,
            expected_reviewer_id=self.card.data["reviewer_id"],
            task_id=self.task_id,
            source_revision=self.card.source_revision,
            task_card_sha256=self.card.raw_sha256,
            worktree_digest=current_worktree_digest(self.repo_root),
        )

    def _authorize_resources(
        self,
        authorization: dict[str, Any] | None,
        *,
        budget_report_path: Path | None = None,
    ) -> str | None:
        resource_class = self.card.data["resource_class"]
        inventory = (
            None if resource_class in {"docs_only", "cpu_test"} else host_inventory(self.repo_root)
        )
        return authorize_resource_class(
            resource_class,
            authorization=authorization,
            task_id=self.task_id,
            source_revision=self.card.source_revision,
            task_card_sha256=self.card.raw_sha256,
            host=socket.gethostname(),
            budget_report_path=budget_report_path,
            inventory=inventory,
        )

    def _run_gate_with_resources(
        self,
        gate: str,
        *,
        scoped_argv: list[str] | None = None,
        receipt_dir: Path | None = None,
        resource_authorization: dict[str, Any] | None = None,
        resource_authorization_digest: str | None = None,
    ) -> dict[str, Any]:
        if gate != "native-smoke":
            # Only native-smoke is metered. Other gates must never run for a native class.
            if self.card.data["resource_class"] not in LOCAL_CLASSES:
                raise ContractError(
                    "non-native-smoke gates run only for local resource classes; "
                    "native work requires the metered native-smoke gate"
                )
            return run_gate(
                gate,
                self.card.raw_sha256,
                scoped_argv=scoped_argv,
                receipt_dir=receipt_dir,
            )
        if resource_authorization is None or resource_authorization_digest is None:
            raise ContractError("native-smoke requires validated resource authorization")
        ledger = ResourceUsageLedger(self.repo_root / "outputs/agentops/resource-usage")
        with metered_gpu_execution(
            ledger,
            resource_authorization_digest,
            resource_authorization,
            maximum_timeout_seconds=TIMEOUTS[gate],
        ) as timeout:
            return run_gate(
                gate,
                self.card.raw_sha256,
                timeout=timeout,
                receipt_dir=receipt_dir,
            )

    def _persist_provider_receipt(self, receipt: dict[str, Any], operation: str) -> str:
        call_dir = self.store.task_root / "provider-calls"
        call_index = len(list(call_dir.glob("*.json"))) + 1 if call_dir.exists() else 1
        call_path = call_dir / f"{call_index:03d}-{operation}.json"
        atomic_json(call_path, receipt)
        try:
            receipt_path = call_path.relative_to(self.repo_root).as_posix()
        except ValueError:
            receipt_path = str(call_path)
        summary = {
            key: receipt.get(key)
            for key in (
                "provider",
                "command_sha256",
                "started_at",
                "ended_at",
                "exit_code",
                "stdout_truncated",
                "stderr_truncated",
                "error",
            )
        }
        summary["receipt_path"] = receipt_path
        self.store.add_provider_call(summary)
        return receipt_path

    def _provider_call(self, payload: dict[str, Any]) -> dict[str, Any]:
        operation = payload.get("operation")
        if operation not in {"implement", "fix", "review"}:
            raise ProviderError("provider operation is not allowed")
        # Review must come from the separately configured reviewer provider, never from the
        # implementer, so that the implementing executable cannot approve its own change.
        provider = self.config.reviewer_provider if operation == "review" else self.config.provider
        if provider is None:
            if operation == "review":
                raise ProviderError("no independent reviewer provider is configured")
            raise ProviderError("no approved provider is configured")
        source_before = current_git_revision(self.repo_root)
        worktree_before = current_worktree_digest(self.repo_root)
        dirty_before = dirty_paths(self.repo_root)
        unowned_before = self._snapshot_unowned_dirty(dirty_before)
        try:
            receipt = run_provider(
                provider.executable,
                provider.args,
                payload,
                env_allowlist=provider.env_allowlist,
                timeout_seconds=provider.timeout_seconds,
                output_limit_bytes=provider.output_limit_bytes,
                cwd=self.repo_root,
            )
        except ProviderError as error:
            receipt_path = None
            command_sha256 = None
            if error.receipt is not None:
                receipt_path = self._persist_provider_receipt(error.receipt, operation)
                command_sha256 = error.receipt.get("command_sha256")
            self._telemetry(
                "REVIEWING" if operation == "review" else "IMPLEMENTING",
                "provider_failed",
                operation=operation,
                error=str(error),
                command_sha256=command_sha256,
                receipt_path=receipt_path,
            )
            raise
        receipt_path = self._persist_provider_receipt(receipt, operation)
        response = receipt["stdout"]
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else None
        self._telemetry(
            "REVIEWING" if operation == "review" else "IMPLEMENTING",
            "provider_completed",
            operation=operation,
            provider=receipt["provider"],
            model=response.get("model", "unknown"),
            command_sha256=receipt["command_sha256"],
            exit_code=receipt["exit_code"],
            usage=normalize_provider_usage(usage),
            receipt_path=receipt_path,
        )
        if current_git_revision(self.repo_root) != source_before:
            raise ProviderError("provider changed the pinned Git source revision")
        worktree_after = current_worktree_digest(self.repo_root)
        dirty_after = dirty_paths(self.repo_root)
        unowned_after = self._snapshot_unowned_dirty(dirty_after)
        if operation == "review" and worktree_after != worktree_before:
            raise ProviderError("read-only reviewer mutated the worktree")
        if operation in {"implement", "fix"}:
            if unowned_after != unowned_before:
                raise ProviderError("provider mutated a path outside task ownership")
            if worktree_after == worktree_before:
                raise ProviderError(f"provider {operation} produced no worktree change")
        return receipt

    def _run_review(self, author_id: str) -> dict[str, Any]:
        # Every kind, including validation, requires a provider-produced independent review.
        # Runner-synthesized approvals are never accepted as review evidence.
        cycle_dir = self._new_review_cycle_dir()
        source_context = self._review_source_context()
        self._write_review_source_context(source_context, cycle_dir)
        self._write_review_source_context(source_context)
        receipt = self._provider_call(
            {
                "operation": "review",
                "read_only": True,
                "task_card": self.card.data,
                "task_card_sha256": self.card.raw_sha256,
                "source_context": source_context,
            }
        )
        response = receipt["stdout"]
        raw_review = response.get("review")
        if not isinstance(raw_review, dict):
            raise ContractError("provider review response is missing a structured review")
        review = bind_review(
            raw_review,
            task_id=self.task_id,
            source_revision=source_context["source_revision"],
            task_card_sha256=self.card.raw_sha256,
            worktree_digest=source_context["worktree_digest"],
        )
        checked = self._validate_current_review(review, author_id=author_id)
        self._write_reviewer_report(checked, cycle_dir)
        self._write_reviewer_report(checked)
        return checked

    def _run_verification(
        self,
        *,
        resource_authorization: dict[str, Any] | None = None,
        resource_authorization_digest: str | None = None,
    ) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        scoped = self.card.data.get("scoped_verification", [])

        def record(receipt: dict[str, Any]) -> None:
            receipts.append(receipt)
            log_path = Path(receipt["log_path"])
            if log_path.name.endswith(".stdout.log"):
                stem = log_path.name[: -len(".stdout.log")]
                receipt_path = (log_path.parent / f"{stem}.receipt.json").as_posix()
            else:
                receipt_path = receipt["log_path"]
            self.store.add_validation_receipt(receipt_path)
            atomic_json(
                self.store.task_root / "verification.json",
                {
                    "task_id": self.task_id,
                    "source_revision": self.card.source_revision,
                    "task_card_sha256": self.card.raw_sha256,
                    "receipts": receipts,
                },
            )
            self._telemetry(
                "VERIFYING",
                "verification_gate",
                gate=receipt["gate"],
                command_argv=receipt["command_argv"],
                started_at=receipt["started_at"],
                ended_at=receipt["ended_at"],
                exit_code=receipt["exit_code"],
                stdout_sha256=receipt["stdout_sha256"],
                stderr_sha256=receipt["stderr_sha256"],
                receipt_path=receipt_path,
            )
            if receipt["exit_code"] != 0:
                raise ContractError(f"verification gate {receipt['gate']} failed")

        for gate in self.card.data["verification_gates"]:
            if gate == "scoped":
                if not scoped:
                    raise ContractError(
                        "task requires scoped verification but defines no exact command"
                    )
                for argv in scoped:
                    receipt = self._run_gate_with_resources(
                        gate,
                        scoped_argv=argv,
                        receipt_dir=self.store.task_root / "phase-receipts",
                        resource_authorization=resource_authorization,
                        resource_authorization_digest=resource_authorization_digest,
                    )
                    record(receipt)
            else:
                receipt = self._run_gate_with_resources(
                    gate,
                    receipt_dir=self.store.task_root / "phase-receipts",
                    resource_authorization=resource_authorization,
                    resource_authorization_digest=resource_authorization_digest,
                )
                record(receipt)
        return receipts

    def _manual_resume_context(self, state: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
        implementation = None
        review = None
        for attempt in state["attempt_history"]:
            if attempt.get("kind") == "manual_implementation":
                implementation = attempt.get("result")
            elif attempt.get("kind") == "independent_review":
                review = attempt.get("result")
        if not isinstance(implementation, dict) or not isinstance(review, dict):
            raise ContractError(
                "resume requires retained manual implementation and review evidence"
            )
        changed_paths = implementation.get("changed_paths")
        if (
            not isinstance(changed_paths, list)
            or not changed_paths
            or not all(isinstance(path, str) for path in changed_paths)
        ):
            raise ContractError("retained manual implementation has invalid changed paths")
        checked_review = self._validate_current_review(review, author_id=self.card.data["owner_id"])
        self._require_current_review_approval(checked_review, author_id=self.card.data["owner_id"])
        if implementation.get("source_revision") != self.card.source_revision:
            raise ContractError("retained manual implementation source revision is stale")
        if implementation.get("task_card_sha256") != self.card.raw_sha256:
            raise ContractError("retained manual implementation task card is stale")
        if implementation.get("worktree_digest") != checked_review["worktree_digest"]:
            raise ContractError("retained manual implementation review binding is stale")
        return changed_paths, checked_review

    def _load_validation_receipts(self, state: dict[str, Any]) -> list[dict[str, Any]]:
        raw_paths = state["validation_receipt_paths"]
        expected_gates: list[str] = []
        scoped = self.card.data.get("scoped_verification", [])
        for gate in self.card.data["verification_gates"]:
            if gate == "scoped":
                expected_gates.extend(["scoped"] * len(scoped))
            else:
                expected_gates.append(gate)
        if len(raw_paths) != len(expected_gates):
            raise ContractError("retained verification receipt roster does not match the task card")

        receipt_root = self.store.task_root / "phase-receipts"
        try:
            if receipt_root.is_symlink():
                raise ContractError("retained verification receipt directory may not be a symlink")
            resolved_root = receipt_root.resolve(strict=True)
        except OSError as error:
            raise ContractError("retained verification receipt directory is unavailable") from error

        current_worktree = current_worktree_digest(self.repo_root)
        receipts: list[dict[str, Any]] = []
        for raw_path, expected_gate in zip(raw_paths, expected_gates, strict=True):
            path = Path(raw_path)
            if not path.is_absolute():
                path = self.repo_root / path
            try:
                if path.is_symlink():
                    raise ContractError("retained verification receipt may not be a symlink")
                resolved = path.resolve(strict=True)
                resolved.relative_to(resolved_root)
            except (OSError, RuntimeError, ValueError) as error:
                raise ContractError(
                    f"retained verification receipt path is unsafe or unavailable: {raw_path}"
                ) from error
            if resolved.parent != resolved_root:
                raise ContractError(
                    "retained verification receipt must be directly in phase-receipts"
                )
            try:
                value = load_strict_json_bytes(resolved.read_bytes())
            except OSError as error:
                raise ContractError(
                    f"cannot read retained verification receipt {raw_path}"
                ) from error
            if not isinstance(value, dict):
                raise ContractError("retained verification receipt must be a JSON object")
            validate_receipt(value)
            if value["gate"] != expected_gate:
                raise ContractError("retained verification receipt gate roster is stale")
            expected_scope = (
                "task_card_scoped" if expected_gate == "scoped" else f"tools/check:{expected_gate}"
            )
            if value["scope"] != expected_scope:
                raise ContractError("retained verification receipt scope is stale")
            if value["skipped"] or value["exit_code"] != 0:
                raise ContractError("retained verification receipt is not successful")
            if value["task_card_sha256"] != self.card.raw_sha256:
                raise ContractError(
                    "retained verification receipt belongs to a different task card"
                )
            if value["source_revision"] != self.card.source_revision:
                raise ContractError("retained verification receipt source revision is stale")
            if value["worktree_digest"] != current_worktree:
                raise ContractError("retained verification receipt worktree binding is stale")
            receipts.append(value)
        return receipts

    def _finish_reporting(
        self,
        *,
        receipts: list[dict[str, Any]],
        review: dict[str, Any],
        changed_paths: list[str],
        next_task: str | None = None,
    ) -> dict[str, Any]:
        if not receipts:
            raise ContractError("DONE requires at least one verification receipt")
        author_id = self.card.data["owner_id"]
        self._require_current_review_approval(review, author_id=author_id)
        if any(receipt.get("worktree_digest") != review["worktree_digest"] for receipt in receipts):
            # Every gate run must have executed on the approved worktree, not only the last one.
            raise ContractError("verification worktree does not match the approved review")
        handoff = build_handoff(
            task_id=self.task_id,
            engineering_status="DONE",
            scientific_disposition=self.card.data["scientific_disposition"],
            source_revision=self.card.source_revision,
            task_card_sha256=self.card.raw_sha256,
            changed_paths=changed_paths,
            checks=receipts,
            review=review,
            blockers=[],
            next_task=next_task,
        )
        # Stage the handoff, make the guarded DONE transition, and only then publish the handoff.
        # A refused transition therefore leaves no DONE handoff on disk.
        pending = self.store.task_root / "handoff.json.pending"
        atomic_json(pending, handoff)
        try:
            self.store.transition("DONE", verification_receipt=receipts[-1])
        except BaseException:
            # Includes KeyboardInterrupt: no staged DONE handoff may survive a refused transition.
            pending.unlink(missing_ok=True)
            raise
        os.replace(pending, self.store.task_root / "handoff.json")
        self._telemetry("DONE", "handoff_completed", changed_paths=changed_paths)
        return handoff

    def _require_dependencies_done(self) -> None:
        """Refuse to start, finalize, or resume a task whose dependencies are not done.

        A dependency counts as done only when its reconciled operational phase is DONE and its
        verification evidence is current for the present worktree. Declared status alone is not
        sufficient, because a retained DONE can be stale or blocked by a later state change.
        """
        from .queue import load_queue

        snapshot = load_queue(repo_root=self.repo_root, reconcile_state=True)
        for dependency_id in self.card.data["dependencies"]:
            dependency = snapshot.by_id(dependency_id)
            satisfied = (
                dependency.get("operational_phase") == "DONE"
                and dependency.get("verification_evidence_status") == "current"
            )
            if not satisfied:
                raise ContractError(f"task {self.task_id} has unmet dependency {dependency_id}")

    def _refuse_active_jobs(self) -> None:
        """Refuse to mutate while another live process works inside owned paths or task state.

        See ``agentops.active_jobs`` for the definition of an active job.
        """
        roots = [self.repo_root / path for path in self.card.data["owned_paths"]]
        roots.append(self.store.state_file.parent)
        found = find_active_jobs(roots)
        if found["active"]:
            summary = [
                {"pid": job["pid"], "paths": [hit["path"] for hit in job["hits"]]}
                for job in found["active"]
            ]
            raise ContractError(
                f"task {self.task_id} refused: active process(es) work inside owned paths "
                f"or task state: {summary}"
            )

    def run(
        self,
        *,
        resource_authorization: dict[str, Any] | None = None,
        budget_report_path: Path | None = None,
    ) -> dict[str, Any]:
        if self.card.data["status"] != "ready":
            raise ContractError(f"task {self.task_id} is not ready")
        self._require_dependencies_done()
        self._refuse_active_jobs()
        self._ensure_state()
        state = self.store.load()
        if state["phase"] != "PENDING":
            raise ContractError("task already has execution state; use resume or status")
        if state["task_card_sha256"] != self.card.raw_sha256:
            raise ContractError("recorded task card digest does not match the current card")
        self.store.transition("PREFLIGHT")
        plan = self.plan()
        authorization_digest = self._authorize_resources(
            resource_authorization,
            budget_report_path=budget_report_path,
        )
        self.store.set_resource_authorization(authorization_digest)
        self.store.transition("IMPLEMENTING")
        author_id = self.card.data["owner_id"]
        # The already-dirty guard applies to every kind: owned dirty work that predates this
        # run must never be attributed to this task's DONE handoff.
        already_dirty = self._owned_dirty_paths()
        if already_dirty:
            raise ContractError(
                f"task owns paths that were already dirty before this run: {already_dirty}"
            )
        if self.card.data["kind"] == "validation":
            self.store.append_attempt(
                {"kind": "internal_validation", "result": "NO_SOURCE_MUTATION"}
            )
        else:
            permissions = self.card.data["permissions"]
            if not permissions["source_write"]:
                raise ContractError("task card does not grant source_write for implementation")
            implementation = self._provider_call(
                {
                    "operation": "implement",
                    "task_card": self.card.data,
                    "task_card_sha256": self.card.raw_sha256,
                    "preflight": plan["preflight"],
                }
            )
            self.store.append_attempt(
                {"kind": "provider_implementation", "result": implementation["stdout"]}
            )
        self.store.transition("REVIEWING")
        review = self._run_review(author_id)
        self.store.append_attempt({"kind": "independent_review", "result": review})
        fix_attempt = 0
        while review["verdict"] == "REQUEST_CHANGES":
            allow_fix_cycle(fix_attempt, self.card.data["max_fix_attempts"])
            self.store.transition("FIXING")
            fix = self._provider_call(
                {
                    "operation": "fix",
                    "task_card": self.card.data,
                    "review": review,
                    "attempt": fix_attempt + 1,
                }
            )
            self.store.append_attempt({"kind": "provider_fix", "result": fix["stdout"]})
            fix_attempt += 1
            self.store.transition("REVIEWING")
            review = self._run_review(author_id)
            self.store.append_attempt({"kind": "independent_review", "result": review})
        self._require_current_review_approval(review, author_id=author_id)
        changed_paths = self._owned_dirty_paths()
        self.store.transition("VERIFYING")
        receipts = self._run_verification(
            resource_authorization=resource_authorization,
            resource_authorization_digest=authorization_digest,
        )
        self.store.transition("REPORTING")
        return self._finish_reporting(
            receipts=receipts,
            review=review,
            changed_paths=changed_paths,
        )

    def finalize_manual(
        self,
        *,
        review: dict[str, Any],
        changed_paths: list[str],
        next_task: str | None = None,
        resource_authorization: dict[str, Any] | None = None,
        budget_report_path: Path | None = None,
    ) -> dict[str, Any]:
        """Finalize authorized manual implementation through the canonical task state."""
        if self.card.data["status"] != "ready":
            raise ContractError(f"task {self.task_id} is not ready")
        self._require_dependencies_done()
        self._refuse_active_jobs()
        if self.store.state_file.exists():
            raise ContractError("task already has execution state; use status before retrying")
        if not changed_paths:
            raise ContractError("manual finalization requires at least one changed path")
        normalized = sorted(set(changed_paths))
        if len(normalized) != len(changed_paths):
            raise ContractError("manual changed paths must be unique")
        owned = self.card.data["owned_paths"]
        for path in normalized:
            if not any(paths_overlap(owner, path) for owner in owned):
                raise ContractError(f"manual changed path is outside task ownership: {path}")
        visible_dirty = set(dirty_paths(self.repo_root))
        missing = [path for path in normalized if path not in visible_dirty]
        if missing:
            raise ContractError(
                f"manual changed paths are not dirty in the current worktree: {missing}"
            )
        owned_dirty = set(self._owned_dirty_paths())
        if set(normalized) != owned_dirty:
            omitted = sorted(owned_dirty - set(normalized))
            extra = sorted(set(normalized) - owned_dirty)
            raise ContractError(
                f"manual changed paths must exactly match dirty owned paths; omitted={omitted}, extra={extra}"
            )

        author_id = self.card.data["owner_id"]
        checked_review = self._validate_current_review(review, author_id=author_id)
        self._require_current_review_approval(checked_review, author_id=author_id)

        self._ensure_state()
        self.store.transition("PREFLIGHT")
        authorization_digest = self._authorize_resources(
            resource_authorization,
            budget_report_path=budget_report_path,
        )
        self.store.set_resource_authorization(authorization_digest)
        self.store.transition("IMPLEMENTING")
        self.store.append_attempt(
            {
                "kind": "manual_implementation",
                "result": {
                    "changed_paths": normalized,
                    "source_revision": self.card.source_revision,
                    "task_card_sha256": self.card.raw_sha256,
                    "worktree_digest": checked_review["worktree_digest"],
                },
            }
        )
        self._write_source_diff()
        self.store.transition("REVIEWING")
        self.store.append_attempt(
            {
                "kind": "independent_review",
                "result": checked_review,
            }
        )
        self._write_reviewer_report(checked_review)
        self.store.transition("VERIFYING")
        receipts = self._run_verification(
            resource_authorization=resource_authorization,
            resource_authorization_digest=authorization_digest,
        )
        self.store.transition("REPORTING")
        return self._finish_reporting(
            receipts=receipts,
            review=checked_review,
            changed_paths=normalized,
            next_task=next_task,
        )

    def resume(
        self,
        *,
        resource_authorization: dict[str, Any] | None = None,
        budget_report_path: Path | None = None,
    ) -> dict[str, Any]:
        self._require_dependencies_done()
        self._refuse_active_jobs()
        authorization_digest = self._authorize_resources(
            resource_authorization,
            budget_report_path=budget_report_path,
        )
        state = self.store.assert_resumable(
            self.card.raw_sha256,
            self.card.source_revision,
            resource_authorization_digest=authorization_digest,
        )
        if state["phase"] == "VERIFYING":
            changed_paths, review = self._manual_resume_context(state)
            receipts = self._run_verification(
                resource_authorization=resource_authorization,
                resource_authorization_digest=authorization_digest,
            )
            self.store.transition("REPORTING")
            return self._finish_reporting(
                receipts=receipts,
                review=review,
                changed_paths=changed_paths,
            )
        if state["phase"] == "REPORTING":
            changed_paths, review = self._manual_resume_context(state)
            receipts = self._load_validation_receipts(state)
            return self._finish_reporting(
                receipts=receipts,
                review=review,
                changed_paths=changed_paths,
            )
        raise ContractError(
            f"phase {state['phase']} has no safe automated resume checkpoint; preserve state for review"
        )

    def handoff(self) -> dict[str, Any]:
        path = self.store.task_root / "handoff.json"
        if not path.exists():
            raise ContractError("task has no handoff yet")
        value = load_strict_json_bytes(path.read_bytes())
        if not isinstance(value, dict):
            raise ContractError("handoff must be a JSON object")
        try:
            validated = validate_handoff(value)
        except ContractError:
            legacy = {
                "task_id",
                "engineering_status",
                "scientific_disposition",
                "source_revision",
                "task_card_sha256",
                "changed_paths",
                "checks",
                "review",
                "blockers",
                "next_task",
            }
            if not legacy.issubset(value):
                raise
            if value.get("engineering_status") == "DONE":
                # A legacy DONE cannot be reconstructed safely: its checks, review, and digests
                # come from unsigned file content. Re-run verification to publish a handoff.
                raise ContractError("legacy DONE handoff cannot be rebuilt; re-run verification")
            checks = value.get("checks")
            if not isinstance(checks, list) or not all(isinstance(item, dict) for item in checks):
                raise ContractError("legacy handoff checks must be a list of objects")
            return build_handoff(
                task_id=value["task_id"],
                engineering_status=value["engineering_status"],
                scientific_disposition=value["scientific_disposition"],
                source_revision=value["source_revision"],
                task_card_sha256=value["task_card_sha256"],
                changed_paths=value["changed_paths"],
                checks=value["checks"],
                review=value["review"],
                blockers=value["blockers"],
                next_task=value["next_task"],
            )
        if validated.get("engineering_status") == "DONE" or validated.get("result") == "DONE":
            # A DONE handoff shown to an operator must rest on current evidence.
            self.store.assert_current_done_evidence(self.card.raw_sha256)
        return validated
