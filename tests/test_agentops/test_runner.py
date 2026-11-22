import base64
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from agentops.config import ProviderConfig
from agentops.contracts import ContractError, current_git_revision, current_worktree_digest
from agentops.preflight import dirty_paths
from agentops.provider import ProviderError
from agentops.runner import TaskRunner
from agentops.task_state import TaskStateStore

ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "E04.PKG06-003.HOP_EST_DEMAP"


@pytest.fixture(autouse=True)
def _skip_dependency_admission(monkeypatch):
    # These tests exercise run, resume and finalize behavior unrelated to scheduling. E04 depends
    # on AGENT.INFRA.SELFTEST, whose verification is stale, so the admission gate would refuse
    # them for an unrelated reason. Dependency admission is tested directly in
    # tests/test_agentops/test_review_hardening.py.
    monkeypatch.setattr(TaskRunner, "_require_dependencies_done", lambda self: None)


CHANGED = [
    "receivers/nr/__init__.py",
    "receivers/nr/pusch.py",
    "tests/test_receivers/test_nr_pusch.py",
]


def test_dirty_paths_preserves_porcelain_z_rename_and_copy_paths(monkeypatch) -> None:
    raw = (
        b"R  renamed/new.py\0renamed/old.py\0 C copied/new.py\0copied/source.py\0?? untracked.txt\0"
    )
    monkeypatch.setattr("agentops.preflight.subprocess.check_output", lambda *args, **kwargs: raw)
    assert dirty_paths(ROOT) == (
        "copied/new.py",
        "copied/source.py",
        "renamed/new.py",
        "renamed/old.py",
        "untracked.txt",
    )


def test_dirty_paths_rejects_incomplete_porcelain_z_rename(monkeypatch) -> None:
    monkeypatch.setattr(
        "agentops.preflight.subprocess.check_output",
        lambda *args, **kwargs: b"R  renamed/new.py\0",
    )
    with pytest.raises(ContractError, match="incomplete rename/copy"):
        dirty_paths(ROOT)


def _review(
    runner: TaskRunner,
    *,
    verdict: str = "APPROVE",
    findings: list[dict] | None = None,
    worktree_digest: str | None = None,
) -> dict:
    return {
        "reviewer_id": "independent_reviewer",
        "verdict": verdict,
        "findings": findings or [],
        "task_id": runner.task_id,
        "source_revision": runner.card.source_revision,
        "task_card_sha256": runner.card.raw_sha256,
        "worktree_digest": worktree_digest or current_worktree_digest(ROOT),
    }


def _runner(tmp_path: Path) -> TaskRunner:
    runner = TaskRunner(TASK_ID, repo_root=ROOT)
    runner.config = type(runner.config)(
        task_cards=runner.config.task_cards,
        state_root=runner.config.state_root,
        log_root=str(tmp_path / "logs"),
        provider=runner.config.provider,
    )
    runner.store = TaskStateStore(TASK_ID, tmp_path, repo_root=ROOT)
    return runner


def _receipt(
    runner: TaskRunner,
    gate: str,
    exit_code: int = 0,
    *,
    worktree_digest: str | None = None,
) -> dict:
    return {
        "schema": "nr-agentops.validation-receipt.v1",
        "gate": gate,
        "command_argv": ["fake-check", gate],
        "started_at": "2026-11-19T00:00:00+00:00",
        "ended_at": "2026-11-19T00:00:01+00:00",
        "exit_code": exit_code,
        "skipped": False,
        "stdout_sha256": "1" * 64,
        "stderr_sha256": "2" * 64,
        "source_revision": current_git_revision(ROOT),
        "worktree_digest": worktree_digest or current_worktree_digest(ROOT),
        "task_card_sha256": runner.card.raw_sha256,
        "environment_id": "4" * 64,
        "log_path": f"outputs/agentops/{gate}.log",
        "scope": "task_card_scoped" if gate == "scoped" else f"tools/check:{gate}",
        "provenance": "engineering_verification_only",
    }


def _provider_receipt(stdout: dict | None = None) -> dict:
    return {
        "provider": "fake-provider",
        "argv": ["fake-provider"],
        "command_sha256": "3" * 64,
        "started_at": "2026-11-19T00:00:00+00:00",
        "ended_at": "2026-11-19T00:00:01+00:00",
        "exit_code": 0,
        "stdout": stdout or {"ok": True},
        "stderr": "",
        "stdout_truncated": False,
        "stderr_truncated": False,
    }


def _provider_runner(tmp_path: Path) -> TaskRunner:
    runner = _runner(tmp_path)
    runner.config = type(runner.config)(
        task_cards=runner.config.task_cards,
        state_root=runner.config.state_root,
        log_root=runner.config.log_root,
        provider=ProviderConfig(
            executable="fake-provider",
            args=(),
            env_allowlist=(),
            timeout_seconds=30,
            output_limit_bytes=4096,
        ),
        reviewer_provider=ProviderConfig(
            executable="fake-reviewer",
            args=(),
            env_allowlist=(),
            timeout_seconds=30,
            output_limit_bytes=4096,
        ),
    )
    runner.store.create(
        runner.card.raw_sha256,
        runner.card.source_revision,
        owners=[runner.card.data["owner_id"]],
        scientific_disposition="NOT_QUALIFIED",
    )
    return runner


def test_finalize_manual_records_review_verification_and_changed_paths(
    tmp_path, monkeypatch
) -> None:
    runner = _runner(tmp_path)
    snapshot_digest = current_worktree_digest(ROOT)
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: tuple(CHANGED))
    monkeypatch.setattr("agentops.runner.current_worktree_digest", lambda _root: snapshot_digest)
    monkeypatch.setattr(
        "agentops.runner.run_gate",
        lambda gate, task_card_sha256, **kwargs: _receipt(
            runner, gate, worktree_digest=snapshot_digest
        ),
    )
    approval = _review(runner, worktree_digest=snapshot_digest)

    handoff = runner.finalize_manual(
        review=approval,
        changed_paths=list(reversed(CHANGED)),
        next_task="E04.PKG06-003.NEXT",
    )

    assert handoff["changed_paths"] == sorted(CHANGED)
    assert handoff["review"]["verdict"] == "APPROVE"
    assert handoff["review_verdict"] == "APPROVE"
    assert handoff["base_sha"] == runner.card.source_revision
    assert handoff["changed_files"] == sorted(CHANGED)
    assert handoff["source_digest"] == snapshot_digest
    assert handoff["verification_receipt"]["exit_code"] == 0
    assert handoff["unresolved_requirements"] == []
    assert handoff["limitations"] == []
    assert handoff["scientific_disposition"] == "NOT_QUALIFIED"
    assert runner.store.load()["phase"] == "DONE"
    assert [item["kind"] for item in runner.store.load()["attempt_history"]] == [
        "manual_implementation",
        "independent_review",
    ]
    assert (runner.store.task_root / "telemetry.jsonl").exists()
    assert (
        runner.store.task_root / "task-card.snapshot.yaml"
    ).read_bytes() == runner.card.path.read_bytes()
    assert (runner.store.task_root / "ownership.json").exists()
    assert (runner.store.task_root / "facts.jsonl").read_bytes() == b""
    assert (runner.store.task_root / "verification.json").exists()
    log_root = runner.repo_root / runner.config.log_root / runner.card.data["owner_id"]
    log_dirs = list(log_root.glob(f"*_{TASK_ID}"))
    assert log_dirs
    assert (log_dirs[-1] / "execution.log").exists()
    assert (log_dirs[-1] / "source-diff.patch").exists()
    assert (log_dirs[-1] / "reviewer-report.md").exists()


@pytest.mark.parametrize(
    ("changed_paths", "dirty", "message"),
    (
        ([CHANGED[0], CHANGED[0]], CHANGED, "unique"),
        (["agentops/runner.py"], ["agentops/runner.py"], "outside task ownership"),
        ([CHANGED[0]], [], "not dirty"),
        ([CHANGED[0]], CHANGED, "exactly match dirty owned paths"),
    ),
)
def test_finalize_manual_rejects_invalid_changed_path_roster(
    tmp_path, monkeypatch, changed_paths, dirty, message
) -> None:
    runner = _runner(tmp_path)
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: tuple(dirty))
    with pytest.raises(ContractError, match=message):
        runner.finalize_manual(review=_review(runner), changed_paths=changed_paths)


@pytest.mark.parametrize(
    "review",
    (
        {"reviewer_id": "nr_phy_receiver_engineer", "verdict": "APPROVE", "findings": []},
        {"reviewer_id": "anything", "verdict": "APPROVE", "findings": []},
        {
            "reviewer_id": "independent_reviewer",
            "verdict": "REQUEST_CHANGES",
            "findings": [
                {"severity": "high", "summary": "needs fix", "evidence": "review evidence"}
            ],
        },
    ),
)
def test_finalize_manual_requires_independent_approval(tmp_path, monkeypatch, review) -> None:
    runner = _runner(tmp_path)
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: tuple(CHANGED))
    review = {
        **review,
        "task_id": runner.task_id,
        "source_revision": runner.card.source_revision,
        "task_card_sha256": runner.card.raw_sha256,
        "worktree_digest": current_worktree_digest(ROOT),
    }
    with pytest.raises(ContractError):
        runner.finalize_manual(review=review, changed_paths=CHANGED)
    assert not runner.store.state_file.exists()


def test_finalize_manual_rejects_existing_execution_state(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path)
    runner.store.create(
        runner.card.raw_sha256,
        runner.card.source_revision,
        scientific_disposition="NOT_QUALIFIED",
    )
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: tuple(CHANGED))
    with pytest.raises(ContractError, match="already has execution state"):
        runner.finalize_manual(review=_review(runner), changed_paths=CHANGED)


def test_finalize_manual_verification_failure_never_produces_done(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path)
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: tuple(CHANGED))
    monkeypatch.setattr(
        "agentops.runner.run_gate",
        lambda gate, task_card_sha256, **kwargs: _receipt(runner, gate, exit_code=1),
    )
    with pytest.raises(ContractError, match="verification gate"):
        runner.finalize_manual(review=_review(runner), changed_paths=CHANGED)
    assert runner.store.load()["phase"] == "VERIFYING"
    assert not (runner.store.task_root / "handoff.json").exists()


def test_resume_finishes_manual_verifying_checkpoint(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path)
    approval = _review(runner)
    runner.store.create(
        runner.card.raw_sha256,
        runner.card.source_revision,
        owners=[runner.card.data["owner_id"]],
        scientific_disposition="NOT_QUALIFIED",
    )
    runner.store.transition("PREFLIGHT")
    runner.store.transition("IMPLEMENTING")
    runner.store.append_attempt(
        {
            "kind": "manual_implementation",
            "result": {
                "changed_paths": CHANGED,
                "source_revision": runner.card.source_revision,
                "task_card_sha256": runner.card.raw_sha256,
                "worktree_digest": approval["worktree_digest"],
            },
        }
    )
    runner.store.transition("REVIEWING")
    runner.store.append_attempt({"kind": "independent_review", "result": approval})
    runner.store.transition("VERIFYING")
    monkeypatch.setattr(
        "agentops.runner.run_gate",
        lambda gate, task_card_sha256, **kwargs: _receipt(runner, gate),
    )

    handoff = runner.resume()

    assert handoff["engineering_status"] == "DONE"
    assert handoff["changed_paths"] == sorted(CHANGED)
    assert handoff["review"] == approval
    assert runner.store.load()["phase"] == "DONE"


def test_reporting_resume_rejects_tampered_retained_receipt(tmp_path) -> None:
    runner = _runner(tmp_path)
    receipt_root = runner.store.task_root / "phase-receipts"
    receipt_root.mkdir(parents=True)
    scoped = _receipt(runner, "scoped")
    fast = _receipt(runner, "fast")
    scoped_path = receipt_root / "scoped.receipt.json"
    fast_path = receipt_root / "fast.receipt.json"
    scoped_path.write_text(json.dumps(scoped), encoding="utf-8")
    fast_path.write_text(json.dumps(fast), encoding="utf-8")
    state = {"validation_receipt_paths": [str(scoped_path), str(fast_path)]}

    loaded = runner._load_validation_receipts(state)
    assert [receipt["gate"] for receipt in loaded] == ["scoped", "fast"]

    scoped["task_card_sha256"] = "f" * 64
    scoped_path.write_text(json.dumps(scoped), encoding="utf-8")
    with pytest.raises(ContractError, match="different task card"):
        runner._load_validation_receipts(state)

    with pytest.raises(ContractError, match="roster"):
        runner._load_validation_receipts({"validation_receipt_paths": [str(fast_path)]})


def test_resume_rejects_review_after_worktree_changes(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path)
    approval = _review(runner, worktree_digest="a" * 64)
    runner.store.create(
        runner.card.raw_sha256,
        runner.card.source_revision,
        owners=[runner.card.data["owner_id"]],
        scientific_disposition="NOT_QUALIFIED",
    )
    runner.store.transition("PREFLIGHT")
    runner.store.transition("IMPLEMENTING")
    runner.store.append_attempt(
        {
            "kind": "manual_implementation",
            "result": {
                "changed_paths": CHANGED,
                "source_revision": runner.card.source_revision,
                "task_card_sha256": runner.card.raw_sha256,
                "worktree_digest": approval["worktree_digest"],
            },
        }
    )
    runner.store.transition("REVIEWING")
    runner.store.append_attempt({"kind": "independent_review", "result": approval})
    runner.store.transition("VERIFYING")
    monkeypatch.setattr("agentops.runner.current_worktree_digest", lambda _root: "b" * 64)

    with pytest.raises(ContractError, match="worktree_digest"):
        runner.resume()


def test_review_source_context_includes_untracked_owned_content(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=AgentOps",
            "-c",
            "user.email=agent@example.invalid",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        ],
        cwd=repo,
        check=True,
    )
    owned = repo / "owned"
    owned.mkdir()
    payload = b"\x00binary-review\xff"
    (owned / "new.bin").write_bytes(payload)

    runner = TaskRunner.__new__(TaskRunner)
    runner.repo_root = repo
    runner.task_id = "TASK"
    runner.card = SimpleNamespace(
        data={"owned_paths": ["owned"], "scoped_verification": []},
        raw_sha256="b" * 64,
        source_revision=current_git_revision(repo),
    )

    context = runner._review_source_context()

    record = next(item for item in context["path_records"] if item["path"] == "owned/new.bin")
    assert record["kind"] == "file"
    assert base64.b64decode(record["content_base64"]) == payload
    assert record["sha256"]
    assert context["changed_paths"] == ["owned/new.bin"]


def test_resume_refuses_unsafe_implementing_checkpoint(tmp_path) -> None:
    runner = _runner(tmp_path)
    runner.store.create(runner.card.raw_sha256, runner.card.source_revision)
    runner.store.transition("PREFLIGHT")
    runner.store.transition("IMPLEMENTING")
    with pytest.raises(ContractError, match="no safe automated resume checkpoint"):
        runner.resume()


def test_provider_read_only_review_rejects_worktree_mutation(tmp_path, monkeypatch) -> None:
    runner = _provider_runner(tmp_path)
    monkeypatch.setattr("agentops.runner.run_provider", lambda *args, **kwargs: _provider_receipt())
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: ())
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr("agentops.runner.current_worktree_digest", lambda _root: next(digests))

    with pytest.raises(ProviderError, match="read-only reviewer mutated"):
        runner._provider_call({"operation": "review", "read_only": True})


def test_provider_implementation_rejects_unowned_mutation(tmp_path, monkeypatch) -> None:
    runner = _provider_runner(tmp_path)
    monkeypatch.setattr("agentops.runner.run_provider", lambda *args, **kwargs: _provider_receipt())
    dirty = iter(((), ("agentops/runner.py",)))
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: next(dirty))
    digests = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr("agentops.runner.current_worktree_digest", lambda _root: next(digests))

    with pytest.raises(ProviderError, match="outside task ownership"):
        runner._provider_call({"operation": "implement"})


def test_provider_implementation_rejects_noop(tmp_path, monkeypatch) -> None:
    runner = _provider_runner(tmp_path)
    monkeypatch.setattr("agentops.runner.run_provider", lambda *args, **kwargs: _provider_receipt())
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: ())
    monkeypatch.setattr("agentops.runner.current_worktree_digest", lambda _root: "a" * 64)

    with pytest.raises(ProviderError, match="produced no worktree change"):
        runner._provider_call({"operation": "implement"})


def test_provider_run_executes_bounded_fix_loop(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path)
    initial_changed = ["receivers/nr/pusch.py"]
    final_changed = ["receivers/nr/pusch.py", "tests/test_receivers/test_nr_pusch.py"]
    current_changed = []
    review_paths = []
    monkeypatch.setattr(runner, "_owned_dirty_paths", lambda: list(current_changed))
    operations = []

    def provider(payload):
        operations.append(payload["operation"])
        if payload["operation"] == "implement":
            current_changed[:] = initial_changed
        elif payload["operation"] == "fix":
            current_changed[:] = final_changed
        return _provider_receipt()

    request_changes = _review(
        runner,
        verdict="REQUEST_CHANGES",
        findings=[{"severity": "high", "summary": "fix required", "evidence": "contract evidence"}],
    )
    reviews = iter((request_changes, _review(runner)))
    monkeypatch.setattr(runner, "_provider_call", provider)

    def run_review(_author):
        review_paths.append(list(current_changed))
        return next(reviews)

    monkeypatch.setattr(runner, "_run_review", run_review)
    monkeypatch.setattr(runner, "_run_verification", lambda **_kwargs: [_receipt(runner, "fast")])

    handoff = runner.run()

    assert operations == ["implement", "fix"]
    assert review_paths == [initial_changed, final_changed]
    assert handoff["engineering_status"] == "DONE"
    assert handoff["changed_paths"] == final_changed
    assert handoff["changed_files"] == final_changed
    assert runner.store.load()["phase"] == "DONE"


def test_provider_run_stops_after_fix_budget(tmp_path, monkeypatch) -> None:
    runner = _runner(tmp_path)
    dirty = iter(([], CHANGED))
    monkeypatch.setattr(runner, "_owned_dirty_paths", lambda: next(dirty))
    operations = []

    def provider(payload):
        operations.append(payload["operation"])
        return _provider_receipt()

    request_changes = _review(
        runner,
        verdict="REQUEST_CHANGES",
        findings=[{"severity": "high", "summary": "still wrong", "evidence": "contract evidence"}],
    )
    monkeypatch.setattr(runner, "_provider_call", provider)
    monkeypatch.setattr(runner, "_run_review", lambda _author: request_changes)

    with pytest.raises(ContractError, match="bounded fix loop exhausted"):
        runner.run()

    assert operations == ["implement", "fix", "fix"]
    assert runner.store.load()["phase"] == "REVIEWING"


def test_provider_failure_receipt_is_persisted(tmp_path, monkeypatch) -> None:
    runner = _provider_runner(tmp_path)
    failure = {
        "provider": "fake-provider",
        "argv": ["fake-provider"],
        "command_sha256": "9" * 64,
        "started_at": "2026-11-19T00:00:00+00:00",
        "ended_at": "2026-11-19T00:00:01+00:00",
        "exit_code": 9,
        "stdout_raw": "",
        "stderr": "boom",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "error": "nonzero_exit",
    }

    def fail(*args, **kwargs):
        raise ProviderError("provider exited with code 9", receipt=failure)

    monkeypatch.setattr("agentops.runner.run_provider", fail)
    monkeypatch.setattr("agentops.runner.dirty_paths", lambda _root: ())
    with pytest.raises(ProviderError, match="code 9"):
        runner._provider_call({"operation": "implement"})

    state = runner.store.load()
    assert state["provider_calls"][-1]["exit_code"] == 9
    receipt_path = Path(state["provider_calls"][-1]["receipt_path"])
    if not receipt_path.is_absolute():
        receipt_path = runner.repo_root / receipt_path
    assert receipt_path.exists()
