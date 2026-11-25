from __future__ import annotations

import json

import pytest
from scripts.run_channel_experiment import main


def test_experiment_config_rejects_unknown_keys(tmp_path, monkeypatch) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "schema: loslab.experiment.v1\ninputs: [data.npz]\noutput: run\ntyop: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("sys.argv", ["run_channel_experiment", "--config", str(config)])

    with pytest.raises(ValueError, match="Unknown experiment config keys"):
        main()


def test_experiment_config_dispatches_resolved_options(tmp_path, monkeypatch, capsys) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text(
        "schema: loslab.experiment.v1\ninputs: [data.npz]\noutput: run\n"
        "methods: [logreg]\nseeds: [13]\nbootstrap: 10\ncapacity: large\n"
        "split_lock_path: lock.json\nsplit_signature_path: lock.json.sig\n"
        "trusted_split_signers_path: allowed_signers\nsplit_signer_id: custodian@example.org\n"
        "prior_shift:\n"
        "  schema: loslab.prior_shift.v1\n"
        "  training_prior_kind: enriched_stress\n"
        "  training_prior_id: enriched-v1\n"
        "  target_prior_kind: natural\n"
        "  target_prior_id: field-v1\n"
        "  target_positive_prevalence: 0.2\n"
        f"  target_prior_evidence_sha256: {'a' * 64}\n"
        "low_confidence_margin: 0.08\n",
        encoding="utf-8",
    )
    calls = []

    def fake_run(inputs, output, **options):
        calls.append((inputs, output, options))
        return {"status": "ok"}

    monkeypatch.setattr("experiments.runner.run_experiment", fake_run)
    monkeypatch.setattr("sys.argv", ["run_channel_experiment", "--config", str(config)])

    main()

    inputs, output, options = calls[0]
    assert inputs == ["data.npz"]
    assert output == "run"
    assert options["methods"] == ["logreg"]
    assert options["seeds"] == [13]
    assert options["bootstrap"] == 10
    assert options["capacity"] == "large"
    assert options["split_lock_path"] == "lock.json"
    assert options["split_signature_path"] == "lock.json.sig"
    assert options["trusted_split_signers_path"] == "allowed_signers"
    assert options["split_signer_id"] == "custodian@example.org"
    assert options["prior_shift"] == {
        "schema": "loslab.prior_shift.v1",
        "training_prior_kind": "enriched_stress",
        "training_prior_id": "enriched-v1",
        "target_prior_kind": "natural",
        "target_prior_id": "field-v1",
        "target_positive_prevalence": 0.2,
        "target_prior_evidence_sha256": "a" * 64,
    }
    assert options["low_confidence_margin"] == 0.08
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_experiment_config_rejects_wrong_schema(tmp_path, monkeypatch) -> None:
    config = tmp_path / "experiment.yaml"
    config.write_text("schema: other.v1\ninputs: [data.npz]\noutput: run\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["run_channel_experiment", "--config", str(config)])

    with pytest.raises(ValueError, match="schema must be loslab.experiment.v1"):
        main()


def test_experiment_cli_dispatches_input_list_and_options(monkeypatch, capsys) -> None:
    calls = []

    def fake_run(inputs, output, **options):
        calls.append((inputs, output, options))
        return {"status": "ok"}

    monkeypatch.setattr("experiments.runner.run_experiment", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_channel_experiment",
            "--input",
            "one.npz",
            "two.npz",
            "--output",
            "run",
            "--phase",
            "--attempt-index",
            "3",
        ],
    )

    main()

    inputs, output, options = calls[0]
    assert inputs == ["one.npz", "two.npz"]
    assert output == "run"
    assert options["include_phase"] is True
    assert options["attempt_index"] == 3
    assert options["prior_shift"] is None
    assert json.loads(capsys.readouterr().out) == {"status": "ok"}


def test_experiment_cli_requires_input_and_output(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_channel_experiment"])

    with pytest.raises(SystemExit, match="2"):
        main()


def test_config_accepts_single_input_string(monkeypatch, tmp_path):
    config = tmp_path / "single-input.yaml"
    config.write_text(
        "schema: loslab.experiment.v1\ninput: data.npz\noutput: run\n", encoding="utf-8"
    )
    calls = []
    monkeypatch.setattr(
        "experiments.runner.run_experiment",
        lambda inputs, output, **options: (
            calls.append((inputs, output, options)) or {"status": "ok"}
        ),
    )
    monkeypatch.setattr("sys.argv", ["run_channel_experiment", "--config", str(config)])

    main()

    assert calls[0][0] == ["data.npz"]
