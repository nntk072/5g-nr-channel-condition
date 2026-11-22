"""Engineering-only command line interface for reviewed agent task cards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .config import load_config
from .contracts import ROOT, ContractError, load_strict_json_bytes, load_task_card
from .doctor import doctor_report
from .queue import load_queue
from .resources import LOCAL_CLASSES, host_inventory
from .runner import TaskRunner
from .task_state import TaskStateStore
from .verification import run_gate

EXIT_INVALID = 2
EXIT_BLOCKED = 3
EXIT_CAPABILITY = 4
EXIT_VERIFICATION = 5
EXIT_INTERRUPTED = 6
EXIT_INTERNAL = 70


def _emit(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _task_card(task_id: str):
    config = load_config()
    for path in config.task_cards:
        card = load_task_card(path)
        if card.data["task_id"] == task_id:
            return card
    raise ContractError(f"unknown task ID {task_id}")


def _load_optional_json(path: str | None, label: str) -> dict[str, Any] | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        value = load_strict_json_bytes(candidate.read_bytes())
    except OSError as error:
        raise ContractError(f"cannot read {label} file") from error
    if not isinstance(value, dict):
        raise ContractError(f"{label} file must contain one JSON object")
    return value


def _doctor() -> dict[str, Any]:
    report = doctor_report(ROOT)
    report["host"] = host_inventory(ROOT)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nr-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    sub.add_parser("validate-tasks")
    sub.add_parser("list")
    for name in ("status", "plan", "handoff"):
        child = sub.add_parser(name)
        child.add_argument("task_id")
    resume = sub.add_parser("resume")
    resume.add_argument("task_id")
    resume.add_argument("--resource-authorization-file")
    resume.add_argument("--budget-report-file")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("task_id")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--resource-authorization-file")
    run_parser.add_argument("--budget-report-file")
    manual = sub.add_parser("finalize-manual")
    manual.add_argument("task_id")
    manual.add_argument("--review-file", required=True)
    manual.add_argument("--changed-path", action="append", required=True)
    manual.add_argument("--next-task")
    manual.add_argument("--resource-authorization-file")
    manual.add_argument("--budget-report-file")
    verify = sub.add_parser("verify")
    verify.add_argument("task_id")
    verify.add_argument(
        "--gate",
        required=True,
        choices=("agent", "fast", "core", "build", "full", "native-smoke", "scoped"),
    )
    verify.add_argument("--resource-authorization-file")
    verify.add_argument("--budget-report-file")
    return parser


def _main(argv: Sequence[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        report = _doctor()
        _emit(report)
        return 0 if report["ok"] else EXIT_CAPABILITY
    if args.command == "validate-tasks":
        queue = load_queue(reconcile_state=False)
        _emit({"valid": True, "task_count": len(queue.cards)})
        return 0
    if args.command == "list":
        queue = load_queue()
        _emit(
            [
                {
                    "task_id": card["task_id"],
                    "status": card.get("effective_status", card["status"]),
                    "declared_status": card["status"],
                    "operational_phase": card.get("operational_phase"),
                    "verification_evidence_status": card.get("verification_evidence_status"),
                    "verification_evidence_reason": card.get("verification_evidence_reason"),
                    "priority": card["priority"],
                    "scientific_disposition": card["scientific_disposition"],
                }
                for card in sorted(queue.cards, key=lambda row: (row["priority"], row["task_id"]))
            ]
        )
        return 0
    if args.command == "status":
        card = _task_card(args.task_id)
        store = TaskStateStore(args.task_id)
        if not store.state_file.exists():
            _emit({"task_id": card.data["task_id"], "status": "not_started"})
            return 0
        state = dict(store.load())
        if state["phase"] == "DONE":
            try:
                store.assert_current_done_evidence(card.raw_sha256)
            except ContractError as error:
                state["verification_evidence_status"] = "stale"
                state["verification_evidence_reason"] = str(error)
            else:
                state["verification_evidence_status"] = "current"
                state["verification_evidence_reason"] = None
        _emit(state)
        return 0
    runner = TaskRunner(args.task_id)
    if args.command == "plan":
        _emit(runner.plan())
        return 0
    if args.command == "run":
        if args.dry_run:
            _emit(runner.dry_run())
        else:
            authorization = _load_optional_json(
                args.resource_authorization_file, "resource authorization"
            )
            budget_report = Path(args.budget_report_file) if args.budget_report_file else None
            _emit(
                runner.run(
                    resource_authorization=authorization,
                    budget_report_path=budget_report,
                )
            )
        return 0
    if args.command == "finalize-manual":
        review_path = Path(args.review_file)
        try:
            review_value = load_strict_json_bytes(review_path.read_bytes())
        except OSError as error:
            raise ContractError("cannot read manual review file") from error
        if not isinstance(review_value, dict):
            raise ContractError("manual review file must contain one JSON object")
        authorization = _load_optional_json(
            args.resource_authorization_file, "resource authorization"
        )
        budget_report = Path(args.budget_report_file) if args.budget_report_file else None
        _emit(
            runner.finalize_manual(
                review=review_value,
                changed_paths=args.changed_path,
                next_task=args.next_task,
                resource_authorization=authorization,
                budget_report_path=budget_report,
            )
        )
        return 0
    if args.command == "handoff":
        _emit(runner.handoff())
        return 0
    if args.command == "resume":
        authorization = _load_optional_json(
            args.resource_authorization_file, "resource authorization"
        )
        budget_report = Path(args.budget_report_file) if args.budget_report_file else None
        _emit(
            runner.resume(
                resource_authorization=authorization,
                budget_report_path=budget_report,
            )
        )
        return 0
    if args.command == "verify":
        card = runner.card
        if args.gate == "native-smoke":
            if (
                card.data["resource_class"] != "native_smoke"
                or not card.data["permissions"]["native_gpu"]
            ):
                raise ContractError(
                    "native GPU verification requires task permission and resource authorization"
                )
            authorization = _load_optional_json(
                args.resource_authorization_file, "resource authorization"
            )
            budget_report = Path(args.budget_report_file) if args.budget_report_file else None
            authorization_digest = runner._authorize_resources(
                authorization,
                budget_report_path=budget_report,
            )
            receipt = runner._run_gate_with_resources(
                args.gate,
                resource_authorization=authorization,
                resource_authorization_digest=authorization_digest,
            )
            _emit(receipt)
            return 0 if receipt["exit_code"] == 0 else EXIT_VERIFICATION
        _refuse_native_for_verify(card)
        scoped = card.data.get("scoped_verification", [])
        if args.gate == "scoped":
            if not scoped:
                raise ContractError("task has no exact scoped verification command")
            receipts = [run_gate("scoped", card.raw_sha256, scoped_argv=item) for item in scoped]
            _emit(receipts)
            return 0 if all(item["exit_code"] == 0 for item in receipts) else EXIT_VERIFICATION
        receipt = run_gate(args.gate, card.raw_sha256)
        _emit(receipt)
        return 0 if receipt["exit_code"] == 0 else EXIT_VERIFICATION
    raise ContractError("unsupported command")


def _refuse_native_for_verify(card: Any) -> None:
    """Refuse verification gates for non-local resource classes.

    The scoped and standard gates run only for local classes. native-smoke has its own branch that
    requires the native_smoke class and a resource authorization.
    """
    if card.data["resource_class"] not in LOCAL_CLASSES:
        raise ContractError(
            "verify runs only local resource classes; native work requires resource authorization"
        )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main(sys.argv[1:] if argv is None else argv)
    except KeyboardInterrupt:
        return EXIT_INTERRUPTED
    except ContractError as error:
        print(str(error), file=sys.stderr)
        message = str(error).lower()
        if "provider" in message or "authorization" in message or "permission" in message:
            return EXIT_CAPABILITY
        if "not ready" in message or "stale" in message or "blocked" in message:
            return EXIT_BLOCKED
        if "verification" in message or "review" in message:
            return EXIT_VERIFICATION
        return EXIT_INVALID
    except Exception as error:  # pragma: no cover - final containment boundary
        print(f"internal error: {error}", file=sys.stderr)
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
