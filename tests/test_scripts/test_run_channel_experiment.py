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
