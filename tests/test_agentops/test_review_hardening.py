"""Regression tests for independent-review, redaction, ownership, and admission hardening.

Every test that builds a TaskRunner isolates both the task-state store and the log root in
tmp_path, so the suite never appends to retained evidence under outputs/ or logs/.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from agentops.config import ProviderConfig, load_config
from agentops.contracts import ROOT, ContractError, validate_relative_path
from agentops.ownership import paths_overlap
from agentops.provider import ProviderError, redact
from agentops.runner import TaskRunner
from agentops.task_state import TaskStateStore

VALIDATION_TASK = "AGENT.INFRA.SELFTEST"


def _isolated_runner(
    tmp_path: Path,
    *,
    provider: ProviderConfig | None = None,
    reviewer_provider: ProviderConfig | None = None,
) -> TaskRunner:
    runner = TaskRunner(VALIDATION_TASK, repo_root=ROOT)
    runner.config = type(runner.config)(
        task_cards=runner.config.task_cards,
        state_root=runner.config.state_root,
        log_root=str(tmp_path / "logs"),
        provider=provider,
        reviewer_provider=reviewer_provider,
    )
    runner.store = TaskStateStore(VALIDATION_TASK, tmp_path / "state", repo_root=ROOT)
    runner.store.create(runner.card.raw_sha256, runner.card.source_revision)
    return runner


def _provider(executable: str, args: tuple[str, ...] = ()) -> ProviderConfig:
    return ProviderConfig(
        executable=executable,
        args=args,
        env_allowlist=(),
        timeout_seconds=30,
        output_limit_bytes=4096,
    )


@pytest.mark.parametrize(
    ("raw", "leaked"),
    [
        ("Authorization: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA=="),
        ("Authorization: Token abc123secret", "abc123secret"),
        ('Authorization: Digest username="u", response="r1x"', "r1x"),
        ('Authorization: Digest username="u", response="r1x"', "username"),
        ("auth: Basic dXNlcjpwYXNzd29yZA==", "dXNlcjpwYXNzd29yZA=="),
        ('{"auth": "Basic dXNlcjpwYXNzd29yZA=="}', "dXNlcjpwYXNzd29yZA=="),
        ("token sk-ant-api03-UNLABELLED-LEAK", "UNLABELLED-LEAK"),
        ("leak ghp_abcdefghijklmnopqrstuvwxyz0123", "abcdefghijklmnopqrstuvwxyz0123"),
        ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.sigPART", "eyJzdWIiOiIxIn0"),
        ("key AIzaSyA1234567890abcdefghijklmnopqrs", "AIzaSyA1234567890"),
        ("Bearer eyJhbGciOiJIUzI1NiJ9.payload", "eyJhbGciOiJIUzI1NiJ9.payload"),
    ],
)
def test_redact_removes_credentials_and_their_values(raw: str, leaked: str) -> None:
    result = redact(raw)
    assert leaked not in result
    assert "[REDACTED]" in result


def test_redact_preserves_ordinary_text() -> None:
    assert redact("plain status line, exit 0") == "plain status line, exit 0"


def test_redact_keeps_redacted_json_parseable() -> None:
    redacted = redact('{"usage": {"token": 12}, "password": null, "ok": true}')
    parsed = json.loads(redacted)
    assert parsed["usage"]["token"] == "[REDACTED]"
    assert parsed["password"] == "[REDACTED]"
    assert parsed["ok"] is True


@pytest.mark.parametrize("value", [".", "./"])
def test_repository_root_is_not_a_valid_owned_path(value: str) -> None:
    with pytest.raises(ContractError, match="unsafe repository-relative path"):
        validate_relative_path(value, ROOT)


def test_repository_root_overlaps_every_path_fail_closed() -> None:
    assert paths_overlap(".", "agentops/runner.py")
    assert paths_overlap("receivers/nr", ".")


def _write_config(tmp_path: Path, provider_block: str, reviewer_block: str | None) -> Path:
    lines = [
        "schema: nr-agentops.config.v1",
        "task_cards: [config/agentops/tasks/AGENT.INFRA.SELFTEST.yaml]",
        provider_block,
    ]
    if reviewer_block is not None:
        lines.append(reviewer_block)
    config = tmp_path / "agent.yaml"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config


def _block(key: str, executable: str, args: list[str]) -> str:
    args_text = ", ".join(args)
    return "\n".join(
        [
            f"{key}:",
            f"  executable: {executable}",
            f"  args: [{args_text}]",
            "  env_allowlist: []",
            "  timeout_seconds: 30",
            "  output_limit_bytes: 4096",
        ]
    )


def test_reviewer_resolving_to_implementer_binary_is_rejected(tmp_path: Path) -> None:
    binary = shutil.which("python3")
    assert binary is not None
    # Same binary under a different path spelling (bare name vs absolute path).
    config = _write_config(
        tmp_path,
        _block("provider", "python3", ["fake.py"]),
        _block("reviewer_provider", binary, ["fake.py"]),
    )
    with pytest.raises(ContractError, match="same executable"):
        load_config(config, repo_root=ROOT)


def test_reviewer_with_extra_flags_on_implementer_binary_is_rejected(tmp_path: Path) -> None:
    binary = shutil.which("python3")
    assert binary is not None
    config = _write_config(
        tmp_path,
        _block("provider", "python3", ["fake.py"]),
        _block("reviewer_provider", "python3", ["fake.py", "--model", "other"]),
    )
    with pytest.raises(ContractError, match="same executable"):
        load_config(config, repo_root=ROOT)


def test_reviewer_on_distinct_binary_is_accepted(tmp_path: Path) -> None:
    binary = shutil.which("python3")
    assert binary is not None
    config = _write_config(
        tmp_path,
        _block("provider", "python3", ["fake.py"]),
        _block("reviewer_provider", "/bin/sh", ["fake.py"]),
    )
    loaded = load_config(config, repo_root=ROOT)
    assert loaded.reviewer_provider is not None
    assert loaded.reviewer_provider.executable == "/bin/sh"


def test_config_without_reviewer_provider_loads_with_no_reviewer(tmp_path: Path) -> None:
    config = tmp_path / "agent.yaml"
    config.write_text(
        "schema: nr-agentops.config.v1\n"
        "task_cards: [config/agentops/tasks/AGENT.INFRA.SELFTEST.yaml]\n"
        "provider: null\n",
        encoding="utf-8",
    )
    loaded = load_config(config, repo_root=ROOT)
    assert loaded.provider is None
    assert loaded.reviewer_provider is None


def test_validation_card_cannot_self_approve_without_independent_reviewer(tmp_path) -> None:
    runner = _isolated_runner(tmp_path, provider=_provider("fake-implementer"))
    with pytest.raises(ProviderError, match="no independent reviewer provider is configured"):
        runner._run_review(runner.card.data["owner_id"])


def test_validation_card_with_preexisting_owned_dirty_work_is_refused(
    tmp_path, monkeypatch
) -> None:
    runner = _isolated_runner(
        tmp_path,
        provider=_provider("fake-implementer"),
        reviewer_provider=_provider("fake-reviewer"),
    )
    # In-memory only: the readiness gate is bypassed so the dirty-work guard is what is tested.
    monkeypatch.setitem(runner.card.data, "status", "ready")
    monkeypatch.setattr(runner, "_owned_dirty_paths", lambda: ["agentops/runner.py"])
    calls: list[str] = []
    monkeypatch.setattr(
        "agentops.runner.run_provider",
        lambda executable, *args, **kwargs: calls.append(executable),
    )
    with pytest.raises(ContractError, match="already dirty before this run"):
        runner.run()
    assert calls == []


def test_reviewer_provider_is_used_for_review_and_implementer_for_implement(
    tmp_path, monkeypatch
) -> None:
    runner = _isolated_runner(
        tmp_path,
        provider=_provider("fake-implementer"),
        reviewer_provider=_provider("fake-reviewer"),
    )
    seen: list[str] = []

    def fake_run_provider(executable, args, payload, **kwargs):
        seen.append(executable)
        raise ProviderError("stop after dispatch")

    monkeypatch.setattr("agentops.runner.run_provider", fake_run_provider)
    with pytest.raises(ProviderError):
        runner._provider_call({"operation": "review", "read_only": True})
    with pytest.raises(ProviderError):
        runner._provider_call({"operation": "implement"})
    assert seen == ["fake-reviewer", "fake-implementer"]


@pytest.mark.parametrize(
    ("text", "anchored"),
    [
        ("PKG06-003 is the pilot.", True),
        ("see PKG06-003.", True),
        ("PKG06-003.1 sub-item", False),
        ("PKG06-003.v2 variant", False),
        ("PKG06-003.HOP suffix", False),
        ("PKG06-0031 different id", False),
        ("xPKG06-003 embedded", False),
    ],
)
def test_requirement_anchor_matches_whole_identifiers_only(text: str, anchored: bool) -> None:
    from agentops.queue import requirement_anchored

    assert requirement_anchored("PKG06-003", text) is anchored


@pytest.mark.parametrize(
    ("raw", "leaked"),
    [
        ('{"password": "hun\\"ter2pw"}', "ter2pw"),
        ('{"auth": {"token": "ABC123"}}', "ABC123"),
        ('{"credentials": {"user": "u", "key": "SECRETVAL123"}}', "SECRETVAL123"),
        ('{"token": ["a", "b"]}', '"b"'),
        ("Set-Cookie: sid=xyz789; Path=/", "xyz789"),
        ('{"cookie": "session=abc123"}', "abc123"),
        ("https://user:hunter2pw@example.com/repo.git", "hunter2pw"),
        ("aws_secret_access_key = AKIAVALUE123456789", "AKIAVALUE123456789"),
        ('{"secret_key": "zzz111"}', "zzz111"),
        ('{"private_key": "-----KEY-----"}', "KEY"),
        ("x-api-key: AKV-live-0123456789abcdef", "AKV-live-0123456789abcdef"),
    ],
)
def test_structured_and_text_redaction_cover_claimed_categories(raw: str, leaked: str) -> None:
    import json

    from agentops.provider import _redact_structured

    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        assert leaked not in redact(raw)
        return
    assert leaked not in json.dumps(_redact_structured(value))


def test_redacted_json_with_escaped_quotes_stays_parseable_and_clean() -> None:
    import json

    from agentops.provider import _redact_structured

    raw = '{"headers": {"Authorization": "Digest username=\\"u\\", response=\\"r1x\\""}}'
    result = json.dumps(_redact_structured(json.loads(raw)))
    assert "r1x" not in result
    assert json.loads(result)["headers"]["Authorization"] == "[REDACTED]"


@pytest.mark.parametrize(
    "raw",
    ['{"usage": {"token": 1e5}}', '{"usage": {"token": 1.5E+3}}', '{"password": null}'],
)
def test_bare_scalar_redaction_keeps_json_valid(raw: str) -> None:
    import json

    parsed = json.loads(redact(raw))
    assert parsed is not None


def test_scientific_disposition_cannot_claim_qualification() -> None:
    # A card or handoff may only record a non-qualifying disposition. Qualification belongs to
    # a separate authoritative scientific process, never to AgentOps.
    import json

    from agentops.contracts import load_schema
    from jsonschema import Draft202012Validator

    card_schema = load_schema("task-card.v1.schema.json")
    enum = card_schema["properties"]["scientific_disposition"]["enum"]
    assert set(enum) == {"NOT_APPLICABLE", "NOT_QUALIFIED"}
    handoff = json.loads((ROOT / "agentops/schema/handoff.v1.schema.json").read_text())
    assert handoff["properties"]["scientific_disposition"]["enum"] == enum
    Draft202012Validator.check_schema(card_schema)
    Draft202012Validator.check_schema(handoff)


def test_failure_receipts_omit_raw_provider_output_and_keep_digests() -> None:
    import hashlib

    from agentops.provider import _failure_receipt

    secret_text = 'warning\n{"auth": {"token": "SECRET-NEST-9"}}'
    receipt = _failure_receipt(
        resolved="/bin/fake",
        argv=["fake", "--password", "hunter2"],
        command_digest="0" * 64,
        started="2026-11-22T00:00:00+00:00",
        ended="2026-11-22T00:00:01+00:00",
        exit_code=0,
        stdout_text=secret_text,
        stderr_text="Set-Cookie: sid=xyz789; other=SECRET",
        output_limit_bytes=4096,
        error="malformed_json",
    )
    rendered = repr(receipt)
    assert "SECRET-NEST-9" not in rendered
    assert "xyz789" not in rendered
    assert "other=SECRET" not in rendered
    assert "hunter2" not in rendered
    assert hashlib.sha256(secret_text.encode()).hexdigest() in receipt["stdout_raw"]


def test_command_digest_does_not_expose_argv_secret(monkeypatch) -> None:
    import hashlib
    import subprocess

    from agentops import provider as provider_module
    from agentops.contracts import canonical_json_bytes

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=b'{"ok": true}', stderr=b"")

    monkeypatch.setattr(provider_module.subprocess, "run", fake_run)
    receipt = provider_module.run_provider(
        "/bin/true",
        ("--password", "hunter2"),
        {"operation": "review"},
        env_allowlist=(),
        timeout_seconds=30,
        output_limit_bytes=4096,
        cwd=ROOT,
    )
    expected = hashlib.sha256(
        canonical_json_bytes(provider_module._argv_summary(["/bin/true", "--password", "hunter2"]))
    ).hexdigest()
    assert receipt["command_sha256"] == expected
    assert "hunter2" not in repr(receipt)
    assert receipt["argv"] == ["true", "<2 arguments>"]


def test_camelcase_credential_keys_are_replaced_on_the_structured_path() -> None:
    import json

    from agentops.provider import _redact_structured

    value = {"accessToken": "a1", "authToken": "a2", "clientSecret": "a3", "normal": "ok"}
    result = json.loads(json.dumps(_redact_structured(value)))
    assert result == {
        "accessToken": "[REDACTED]",
        "authToken": "[REDACTED]",
        "clientSecret": "[REDACTED]",
        "normal": "ok",
    }


def test_multi_cookie_header_is_fully_redacted() -> None:
    text = redact("Cookie: sid=abc123; other=SECRET; third=X")
    assert "SECRET" not in text and "abc123" not in text and "third=X" not in text


def test_provider_error_text_carries_no_stderr(monkeypatch) -> None:
    import subprocess

    from agentops import provider as provider_module

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 3, stdout=b"", stderr=b"Set-Cookie: sid=LEAKME")

    monkeypatch.setattr(provider_module.subprocess, "run", fake_run)
    with pytest.raises(ProviderError) as info:
        provider_module.run_provider(
            "/bin/true",
            (),
            {"operation": "review"},
            env_allowlist=(),
            timeout_seconds=30,
            output_limit_bytes=4096,
            cwd=ROOT,
        )
    assert "LEAKME" not in str(info.value)
    assert "LEAKME" not in repr(info.value.receipt)


def test_quoted_and_bracketed_cookie_values_are_fully_redacted() -> None:
    text = redact('Cookie: sid="s3cr3t"; theme=dark')
    assert "s3cr3t" not in text and "theme=dark" not in text
    tail = redact("Cookie: a=x]tail; b=y}z")
    assert "tail" not in tail and "y}z" not in tail


def test_lowercase_concatenated_credential_keys_are_replaced() -> None:
    from agentops.provider import _is_secret_key

    for key in ("accesstoken", "clientsecret", "apikey", "Tokenizer", "keyboard", "max_tokens"):
        assert _is_secret_key(key) is (key in {"accesstoken", "clientsecret", "apikey"})


def test_task_state_schema_rejects_qualification_disposition() -> None:
    import json

    from agentops.contracts import load_schema

    schema = load_schema("task-state.v1.schema.json")
    enum = schema["properties"]["scientific_disposition"]["enum"]
    assert set(enum) == {"NOT_APPLICABLE", "NOT_QUALIFIED"}
    assert json.dumps(enum).count("QUALIFIED_FROM") == 0


def test_flat_cloud_credential_names_are_redacted() -> None:
    from agentops.provider import redact

    assert "KEYVAL123" not in redact("SecretAccessKey=KEYVAL123")
    assert "TOK99" not in redact('{"SessionToken": "TOK99"}')
    assert "KEYVAL123" not in redact("SECRET_ACCESS_KEY: KEYVAL123")


def test_hyphenated_cookie_labels_redact_every_pair() -> None:
    text = redact("X-Set-Cookie: a=1; sid=SECRET")
    assert "SECRET" not in text and "a=1" not in text


def test_review_approval_with_blocking_findings_is_refused() -> None:
    from agentops.review import require_review_approval

    base = {
        "reviewer_id": "independent_reviewer",
        "verdict": "APPROVE",
        "findings": [{"severity": "critical", "summary": "x", "evidence": "y"}],
        "task_id": "T",
        "source_revision": "a" * 40,
        "task_card_sha256": "b" * 64,
        "worktree_digest": "c" * 64,
    }
    with pytest.raises(ContractError, match="high or critical findings"):
        require_review_approval(
            base,
            author_id="owner",
            expected_reviewer_id="independent_reviewer",
            task_id="T",
            source_revision="a" * 40,
            task_card_sha256="b" * 64,
            worktree_digest="c" * 64,
        )


def test_dependency_admission_refuses_unmet_dependency(tmp_path, monkeypatch) -> None:
    runner = _isolated_runner(tmp_path, provider=_provider("fake-implementer"))
    runner.card.data["dependencies"] = ["UPSTREAM.TASK"]

    class Snapshot:
        def by_id(self, task_id):
            return {"task_id": task_id, "status": "ready", "effective_status": "ready"}

    monkeypatch.setattr("agentops.queue.load_queue", lambda **kwargs: Snapshot())
    with pytest.raises(ContractError, match="unmet dependency UPSTREAM.TASK"):
        runner._require_dependencies_done()


def test_deeply_nested_provider_json_becomes_a_receipt_not_a_crash(monkeypatch) -> None:
    import json
    import subprocess

    from agentops import provider as provider_module

    deep = "[" * 1500 + "]" * 1500
    payload = '{"review": ' + deep + "}"

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=payload.encode(), stderr=b"")

    monkeypatch.setattr(provider_module.subprocess, "run", fake_run)
    with pytest.raises(ProviderError) as info:
        provider_module.run_provider(
            "/bin/true",
            (),
            {"operation": "review"},
            env_allowlist=(),
            timeout_seconds=30,
            output_limit_bytes=16_000_000,
            cwd=ROOT,
        )
    assert info.value.receipt is not None
    json.dumps(info.value.receipt)


def test_cookie_pattern_is_linear_on_long_word_runs() -> None:
    import time

    started = time.perf_counter()
    redact("a" * 200_000 + " cookie")
    assert time.perf_counter() - started < 2.0


def test_dependency_with_stale_evidence_is_not_satisfied(tmp_path, monkeypatch) -> None:
    runner = _isolated_runner(tmp_path, provider=_provider("fake-implementer"))
    runner.card.data["dependencies"] = ["UPSTREAM.TASK"]

    class Snapshot:
        def by_id(self, task_id):
            return {
                "task_id": task_id,
                "status": "done",
                "operational_phase": "DONE",
                "verification_evidence_status": "stale",
            }

    monkeypatch.setattr("agentops.queue.load_queue", lambda **kwargs: Snapshot())
    with pytest.raises(ContractError, match="unmet dependency UPSTREAM.TASK"):
        runner._require_dependencies_done()


def test_json_nesting_beyond_limit_is_refused_with_receipt(monkeypatch) -> None:
    import subprocess

    from agentops import provider as provider_module

    deep = "[" * 300 + "]" * 300
    payload = '{"review": ' + deep + "}"

    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, stdout=payload.encode(), stderr=b"")

    monkeypatch.setattr(provider_module.subprocess, "run", fake_run)
    with pytest.raises(ProviderError) as info:
        provider_module.run_provider(
            "/bin/true",
            (),
            {"operation": "review"},
            env_allowlist=(),
            timeout_seconds=30,
            output_limit_bytes=16_000_000,
            cwd=ROOT,
        )
    assert info.value.receipt is not None
    assert info.value.receipt["error"] == "malformed_json"


def test_refused_done_transition_leaves_no_handoff(tmp_path, monkeypatch) -> None:
    runner = _isolated_runner(tmp_path, provider=_provider("fake-implementer"))
    handoff_path = runner.store.task_root / "handoff.json"

    def refuse(*args, **kwargs):
        raise ContractError("DONE refused: evidence stale")

    monkeypatch.setattr(runner.store, "transition", refuse)
    monkeypatch.setattr(runner, "_require_current_review_approval", lambda *a, **k: None)
    review = {"worktree_digest": "c" * 64, "verdict": "APPROVE", "reviewer_id": "x"}
    receipts = [{"worktree_digest": "c" * 64, "gate": "scoped", "exit_code": 0, "skipped": False}]
    with pytest.raises(ContractError, match="DONE refused"):
        runner._finish_reporting(receipts=receipts, review=review, changed_paths=[])
    assert not handoff_path.exists()
    assert not (runner.store.task_root / "handoff.json.pending").exists()


def test_done_handoff_requires_approval_and_passing_unskipped_checks() -> None:
    from agentops.reporting import build_handoff

    good_check = {"worktree_digest": "d" * 64, "exit_code": 0, "skipped": False, "gate": "scoped"}
    base = {
        "task_id": "T",
        "engineering_status": "DONE",
        "scientific_disposition": "NOT_QUALIFIED",
        "source_revision": "a" * 40,
        "task_card_sha256": "b" * 64,
        "changed_paths": [],
        "blockers": [],
        "next_task": None,
    }
    with pytest.raises(ContractError, match="APPROVE"):
        build_handoff(
            **base,
            checks=[good_check],
            review={"verdict": "REQUEST_CHANGES"},
        )
    with pytest.raises(ContractError, match="every check to pass"):
        build_handoff(
            **base,
            checks=[{**good_check, "exit_code": 1}],
            review={"verdict": "APPROVE"},
        )
    with pytest.raises(ContractError, match="every check to pass"):
        build_handoff(
            **base,
            checks=[{**good_check, "skipped": True}],
            review={"verdict": "APPROVE"},
        )


def test_credential_patterns_are_linear_on_separator_rich_input() -> None:
    import time

    started = time.perf_counter()
    redact("eyJa-" * 40_000 + "x")
    redact("AIza" * 50_000)
    assert time.perf_counter() - started < 3.0


def test_legacy_done_handoff_refuses_stale_evidence(tmp_path) -> None:
    import json

    runner = _isolated_runner(tmp_path, provider=_provider("fake-implementer"))
    legacy = {
        "task_id": runner.task_id,
        "engineering_status": "DONE",
        "scientific_disposition": "NOT_QUALIFIED",
        "source_revision": "a" * 40,
        "task_card_sha256": runner.card.raw_sha256,
        "changed_paths": [],
        "checks": [{"worktree_digest": "d" * 64, "exit_code": 0, "skipped": False}],
        "review": {"verdict": "APPROVE"},
        "blockers": [],
        "next_task": None,
    }
    path = runner.store.task_root / "handoff.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(legacy), encoding="utf-8")

    def stale(*args, **kwargs):
        raise ContractError("DONE evidence is stale for the current worktree")

    runner.store.assert_current_done_evidence = stale
    with pytest.raises(ContractError, match="cannot be rebuilt"):
        runner.handoff()


def test_deep_json_never_escapes_as_an_internal_error() -> None:
    from agentops.contracts import load_strict_json_bytes

    deep = b"[" * 5000 + b"]" * 5000
    try:
        load_strict_json_bytes(deep)
    except ContractError:
        pass  # refusing deep input with a contract error is acceptable


def test_worktree_digest_sees_a_staged_deletion(tmp_path) -> None:
    import subprocess

    from agentops.contracts import current_worktree_digest

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty", "-qm", "i"],
        cwd=repo,
        check=True,
    )
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "a"],
        cwd=repo,
        check=True,
    )
    before = current_worktree_digest(repo)
    subprocess.run(["git", "rm", "-q", "a.txt"], cwd=repo, check=True)
    (repo / "b.txt").write_text("one\n", encoding="utf-8")
    subprocess.run(["git", "add", "b.txt"], cwd=repo, check=True)
    after = current_worktree_digest(repo)
    assert before != after


def test_oversized_integer_literal_is_refused_as_contract_error() -> None:
    from agentops.contracts import load_strict_json_bytes

    with pytest.raises(ContractError):
        load_strict_json_bytes(b'{"a": ' + b"1" * 5000 + b"}")


def test_done_refuses_when_any_gate_ran_on_a_different_worktree(tmp_path, monkeypatch) -> None:
    runner = _isolated_runner(tmp_path, provider=_provider("fake-implementer"))
    monkeypatch.setattr(runner, "_require_current_review_approval", lambda *a, **k: None)
    review = {"worktree_digest": "c" * 64, "verdict": "APPROVE", "reviewer_id": "x"}
    receipts = [
        {"worktree_digest": "e" * 64, "gate": "scoped", "exit_code": 0, "skipped": False},
        {"worktree_digest": "c" * 64, "gate": "fast", "exit_code": 0, "skipped": False},
    ]
    with pytest.raises(ContractError, match="does not match the approved review"):
        runner._finish_reporting(receipts=receipts, review=review, changed_paths=[])


def test_yaml_style_alias_expansion_is_not_exponential() -> None:
    import time

    from agentops.contracts import reject_nonfinite

    # Build a shared-node chain: each level references the previous one twice (alias-like).
    node: list = ["leaf"]
    for _ in range(60):
        node = [node, node]
    started = time.perf_counter()
    reject_nonfinite(node)
    assert time.perf_counter() - started < 2.0


def test_non_string_verdict_is_a_contract_error() -> None:
    from agentops.review import validate_review

    value = {
        "reviewer_id": "independent_reviewer",
        "verdict": ["APPROVE"],
        "findings": [],
        "task_id": "T",
        "source_revision": "a" * 40,
        "task_card_sha256": "b" * 64,
        "worktree_digest": "c" * 64,
    }
    with pytest.raises(ContractError):
        validate_review(
            value,
            author_id="owner",
            expected_reviewer_id="independent_reviewer",
            task_id="T",
            source_revision="a" * 40,
            task_card_sha256="b" * 64,
            worktree_digest="c" * 64,
        )


def test_verification_receipt_persists_no_argument_values() -> None:
    import inspect

    from agentops import verification

    source = inspect.getsource(verification)
    assert '"command_argv": resolved_argv,' not in source
    assert "arguments>" in source


def test_verify_refuses_native_resource_classes_for_scoped_gates() -> None:
    from agentops import cli

    class Card:
        data = {"resource_class": "production_d1", "scoped_verification": [["true"]]}
        raw_sha256 = "a" * 64

    class Runner:
        card = Card()

    with pytest.raises(ContractError, match="verify runs only local resource classes"):
        cli._refuse_native_for_verify(Runner.card)


def test_runner_scoped_gate_refuses_native_class_without_metering(tmp_path, monkeypatch) -> None:
    runner = _isolated_runner(tmp_path, provider=_provider("fake-implementer"))
    runner.card.data["resource_class"] = "native_smoke"
    called: list[str] = []
    monkeypatch.setattr("agentops.runner.run_gate", lambda *a, **k: called.append("ran") or {})
    with pytest.raises(ContractError, match="only for local resource classes"):
        runner._run_gate_with_resources("scoped", scoped_argv=["true"])
    assert called == []
