import json

from agentops.cli import build_parser, main


def test_cli_validate_list_and_plan(capsys) -> None:
    assert main(["validate-tasks"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["valid"] is True
    assert main(["list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert any(row["task_id"] == "AGENT.INFRA.SELFTEST" for row in listed)
    assert main(["plan", "AGENT.INFRA.SELFTEST"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["task"]["scientific_disposition"] == "NOT_APPLICABLE"


def test_run_and_resume_accept_explicit_resource_authorization_file() -> None:
    parser = build_parser()
    run = parser.parse_args(
        ["run", "TASK.ONE", "--resource-authorization-file", "authorization.json"]
    )
    resume = parser.parse_args(
        ["resume", "TASK.ONE", "--resource-authorization-file", "authorization.json"]
    )
    assert run.resource_authorization_file == "authorization.json"
    assert resume.resource_authorization_file == "authorization.json"


def test_cli_native_verify_is_authorization_gated(capsys) -> None:
    code = main(["verify", "E04.PKG06-003.HOP_EST_DEMAP", "--gate", "native-smoke"])
    captured = capsys.readouterr()
    assert code == 4
    assert "resource authorization" in captured.err


def test_cli_exit_code_contract(monkeypatch, capsys) -> None:
    import agentops.cli as cli
    from agentops.contracts import ContractError

    assert cli.main(["plan", "UNKNOWN.TASK"]) == 2
    capsys.readouterr()
    assert cli.main(["run", "AGENT.INFRA.SELFTEST"]) == 3
    capsys.readouterr()

    monkeypatch.setattr(
        cli, "_main", lambda _argv: (_ for _ in ()).throw(ContractError("review failed"))
    )
    assert cli.main([]) == 5
    capsys.readouterr()
    monkeypatch.setattr(cli, "_main", lambda _argv: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert cli.main([]) == 6
    monkeypatch.setattr(cli, "_main", lambda _argv: (_ for _ in ()).throw(RuntimeError("boom")))
    assert cli.main([]) == 70


def test_budget_report_file_is_forwarded_for_resource_aware_commands(
    tmp_path, monkeypatch, capsys
) -> None:
    import agentops.cli as cli

    authorization_path = tmp_path / "authorization.json"
    authorization_path.write_text('{"approved": true}', encoding="utf-8")
    review_path = tmp_path / "review.json"
    review_path.write_text(
        '{"reviewer_id":"independent_reviewer","verdict":"APPROVE","findings":[],'
        '"task_id":"TASK","source_revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"task_card_sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
        '"worktree_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}',
        encoding="utf-8",
    )
    budget_path = tmp_path / "budget.json"
    calls = []

    class FakeRunner:
        def __init__(self, task_id):
            self.task_id = task_id
            self.card = type(
                "Card",
                (),
                {
                    "data": {
                        "resource_class": "native_smoke",
                        "permissions": {"native_gpu": True},
                    }
                },
            )()

        def run(self, **kwargs):
            calls.append(("run", kwargs))
            return {"ok": True}

        def resume(self, **kwargs):
            calls.append(("resume", kwargs))
            return {"ok": True}

        def finalize_manual(self, **kwargs):
            calls.append(("finalize-manual", kwargs))
            return {"ok": True}

        def _authorize_resources(self, authorization, *, budget_report_path):
            calls.append(
                (
                    "verify-authorize",
                    {
                        "resource_authorization": authorization,
                        "budget_report_path": budget_report_path,
                    },
                )
            )
            return "d" * 64

        def _run_gate_with_resources(self, gate, **kwargs):
            calls.append(("verify-gate", {"gate": gate, **kwargs}))
            return {"exit_code": 0}

    monkeypatch.setattr(cli, "TaskRunner", FakeRunner)

    common = [
        "--resource-authorization-file",
        str(authorization_path),
        "--budget-report-file",
        str(budget_path),
    ]
    assert cli.main(["run", "TASK.ONE", *common]) == 0
    capsys.readouterr()
    assert cli.main(["resume", "TASK.ONE", *common]) == 0
    capsys.readouterr()
    assert (
        cli.main(
            [
                "finalize-manual",
                "TASK.ONE",
                "--review-file",
                str(review_path),
                "--changed-path",
                "receivers/nr/pusch.py",
                *common,
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert cli.main(["verify", "TASK.ONE", "--gate", "native-smoke", *common]) == 0
    capsys.readouterr()

    expected_budget = budget_path
    for command in ("run", "resume", "finalize-manual"):
        kwargs = next(value for name, value in calls if name == command)
        assert kwargs["budget_report_path"] == expected_budget
        assert kwargs["resource_authorization"] == {"approved": True}
    authorize = next(value for name, value in calls if name == "verify-authorize")
    assert authorize["budget_report_path"] == expected_budget
    assert authorize["resource_authorization"] == {"approved": True}
    verify_gate = next(value for name, value in calls if name == "verify-gate")
    assert verify_gate["resource_authorization"] == {"approved": True}
    assert verify_gate["resource_authorization_digest"] == "d" * 64
