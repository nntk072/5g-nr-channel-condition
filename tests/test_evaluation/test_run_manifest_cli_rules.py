from __future__ import annotations

import json

import evaluation.run_manifest.cli as cli
import pytest


def test_print_source_bundle_emits_requested_inventory(monkeypatch, capsys, tmp_path):
    observed = {}
    identity = {"schema": "fixture", "sources": []}

    def source_identity(**kwargs):
        observed.update(kwargs)
        return identity

    monkeypatch.setattr(cli, "evaluator_source_bundle_identity", source_identity)

    assert (
        cli.main(
            [
                "--print-source-bundle",
                "--include-runtime-inventory-sources",
                "--source-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert observed == {"source_root": tmp_path, "include_runtime_inventory": True}
    assert json.loads(capsys.readouterr().out) == identity


def test_print_source_bundle_reports_inventory_errors(monkeypatch, capsys):
    def fail(**_kwargs):
        raise RuntimeError("source tree is incomplete")

    monkeypatch.setattr(cli, "evaluator_source_bundle_identity", fail)

    assert cli.main(["--print-source-bundle"]) == 2
    assert "source inventory error: source tree is incomplete" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--print-source-bundle", "--create-run-id", "r1"], "cannot be combined"),
        (["--include-runtime-inventory-sources"], "requires --print-source-bundle"),
        (["--runtime-inventory", "inventory.json"], "requires --create-run-id"),
        (["--qualification-contract", "contract.json"], "requires --create-run-id"),
        (
            [
                "--create-run-id",
                "r1",
                "--input",
                "i",
                "--artifact",
                "a",
                "--output",
                "o",
                "--signature",
                "s",
            ],
            "signature options are only valid",
        ),
        (
            ["--create-run-id", "r1", "--require-signed-reviewer-admission"],
            "only valid when verifying",
        ),
        (["--create-run-id", "r1", "--report-output", "report.json"], "only valid when verifying"),
        (["--create-run-id", "r1"], "requires --input, --artifact, and --output"),
        ([], "provide a manifest to verify"),
    ],
)
def test_cli_rejects_invalid_option_combinations(arguments, message):
    with pytest.raises(SystemExit, match="2"):
        cli.main(arguments)
    # argparse writes usage and the specific diagnostic to stderr.


def test_create_run_manifest_writes_new_canonical_file(monkeypatch, capsys, tmp_path):
    manifest = {"schema": "fixture", "run_id": "run-1"}
    observed = {}

    def build(run_id, source, artifact, **kwargs):
        observed.update(run_id=run_id, source=source, artifact=artifact, **kwargs)
        return manifest

    monkeypatch.setattr(cli, "build_joint_evidence_run_manifest", build)
    output = tmp_path / "manifest.json"
    arguments = [
        "--create-run-id",
        "run-1",
        "--input",
        "input.json",
        "--artifact",
        "artifact.json",
        "--output",
        str(output),
    ]

    assert cli.main(arguments) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    assert observed["run_id"] == "run-1"
    assert json.loads(capsys.readouterr().out) == {"schema": "fixture", "run_id": "run-1"}


def test_create_run_manifest_reports_builder_and_output_errors(monkeypatch, capsys, tmp_path):
    def fail(*_args, **_kwargs):
        raise ValueError("invalid source bundle")

    monkeypatch.setattr(cli, "build_joint_evidence_run_manifest", fail)
    arguments = [
        "--create-run-id",
        "run-1",
        "--input",
        "input.json",
        "--artifact",
        "artifact.json",
        "--output",
        str(tmp_path / "manifest.json"),
    ]

    assert cli.main(arguments) == 2
    assert "run manifest creation error: invalid source bundle" in capsys.readouterr().err


def test_create_run_manifest_refuses_existing_destination(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        cli,
        "build_joint_evidence_run_manifest",
        lambda *_args, **_kwargs: {"schema": "fixture", "run_id": "run-1"},
    )
    output = tmp_path / "manifest.json"
    output.write_text("existing", encoding="utf-8")
    arguments = [
        "--create-run-id",
        "run-1",
        "--input",
        "input.json",
        "--artifact",
        "artifact.json",
        "--output",
        str(output),
    ]

    assert cli.main(arguments) == 2
    assert "File exists" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "existing"


def test_verification_report_write_failure_is_reported(monkeypatch, capsys, tmp_path):
    report = {"status": "PASS_INPUT_SCOPED_RECOMPUTATION"}
    monkeypatch.setattr(cli, "verify_joint_evidence_run_manifest", lambda *_args, **_kwargs: report)

    def fail_write(*_args, **_kwargs):
        raise OSError("report destination unavailable")

    monkeypatch.setattr(cli, "_write_new_artifact", fail_write)

    assert cli.main(["manifest.json", "--report-output", str(tmp_path / "report.json")]) == 2
    assert (
        "verification report write error: report destination unavailable" in capsys.readouterr().err
    )


def test_verification_returns_failure_for_nonpassing_report(monkeypatch, capsys):
    report = {"status": "FAIL_INPUT_SCOPED_RECOMPUTATION"}
    monkeypatch.setattr(cli, "verify_joint_evidence_run_manifest", lambda *_args, **_kwargs: report)

    assert cli.main(["manifest.json"]) == 1
    assert json.loads(capsys.readouterr().out) == report
