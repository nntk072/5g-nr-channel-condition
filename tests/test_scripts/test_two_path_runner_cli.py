"""End-to-end artifact contract for the grouped two-path pilot runner."""

from __future__ import annotations

import json

import pytest
import scripts.run_observable_multipath_experiment as observable_runner
from artifacts.provenance import canonical_hash
from artifacts.run_replay import validate_run_replay_provenance
from scripts.audit_interrupted_runs import audit_root
from scripts.run_observable_multipath_experiment import run_experiment
from scripts.summarize_observable_multipath_experiment import summarize_run


def test_runner_persists_running_receipt_before_decoder_training(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    output = tmp_path / "interrupted-pilot"
    observed = {}

    def interrupt_after_receipt(*args, **kwargs):
        staging_dirs = list(tmp_path.glob(".interrupted-pilot.staging-*"))
        assert len(staging_dirs) == 1
        receipt = json.loads((staging_dirs[0] / "run_manifest.json").read_text())
        observed.update(receipt)
        assert receipt["schema"] == "loslab.observable_multipath_staging.v2"
        assert receipt["phase"] == "model_fitting"
        assert receipt["state"] == receipt["run_replay"]["state"] == "running"
        audited = audit_root(tmp_path)["manifests"][0]
        assert audited["recovery_supported"] is True
        raise RuntimeError("simulated interruption after durable receipt")

    monkeypatch.setattr(observable_runner, "fit_two_path_decoder", interrupt_after_receipt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        observable_runner.run_experiment(
            output,
            num_groups=40,
            samples_per_group=1,
            n_tones=8,
            seed=10,
            epochs=1,
            batch_size=32,
            patience=1,
            device="cpu",
            bootstrap_draws=100,
        )

    assert observed["qualification_status"] == "NOT_QUALIFIED"
    assert not output.exists()
    assert list(tmp_path.glob(".interrupted-pilot.staging-*")) == []


def test_runner_persists_preparation_receipt_before_dataset_generation(tmp_path, monkeypatch):
    output = tmp_path / "interrupted-before-dataset"
    observed = {}

    def interrupt_during_generation(**kwargs):
        staging_dirs = list(tmp_path.glob(".interrupted-before-dataset.staging-*"))
        assert len(staging_dirs) == 1
        receipt = json.loads((staging_dirs[0] / "run_manifest.json").read_text())
        observed.update(receipt)
        assert receipt["schema"] == "loslab.observable_multipath_staging.v2"
        assert receipt["phase"] == "dataset_generation"
        assert receipt["state"] == "running"
        assert "run_replay" not in receipt
        assert receipt["qualification_status"] == "NOT_QUALIFIED"
        raise RuntimeError("simulated interruption during dataset generation")

    monkeypatch.setattr(
        observable_runner, "generate_grouped_two_path_dataset", interrupt_during_generation
    )
    with pytest.raises(RuntimeError, match="during dataset generation"):
        observable_runner.run_experiment(
            output,
            num_groups=40,
            samples_per_group=1,
            n_tones=8,
            seed=10,
            epochs=1,
            batch_size=32,
            patience=1,
            device="cpu",
            bootstrap_draws=100,
        )

    assert observed["run_config"]["seed"] == 10
    assert not output.exists()


def test_runner_publishes_complete_hashed_predictions_and_group_reports(tmp_path):
    pytest.importorskip("torch")
    output = tmp_path / "pilot"
    manifest = run_experiment(
        output,
        num_groups=40,
        samples_per_group=1,
        n_tones=8,
        seed=10,
        epochs=1,
        batch_size=32,
        patience=1,
        device="cpu",
        bootstrap_draws=100,
    )

    assert manifest["claim_status"] == "NO_SCIENTIFIC_ADVANTAGE_CLAIM"
    assert manifest["schema"] == "loslab.observable_multipath_experiment.v3"
    replay = validate_run_replay_provenance(manifest["run_replay"])
    assert replay["state"] == "complete"
    assert replay["config_sha256"] == canonical_hash(manifest["run_config"])
    assert replay["inputs"][0]["sha256"] == manifest["dataset"]["source_sha256"]
    assert set(manifest["artifacts"]) == {
        "identifiability_off_predictions.json",
        "identifiability_on_predictions.json",
        "profile_likelihood_predictions.json",
        "identifiability_off_uncertainty_report.json",
        "identifiability_on_uncertainty_report.json",
        "profile_likelihood_uncertainty_report.json",
        "paired_test_comparisons.json",
        "profile_likelihood_delay_sets.json",
        "decoder_identifiability_off.npz",
        "decoder_identifiability_on.npz",
        "decoder_identifiability_off.contract.json",
        "decoder_identifiability_on.contract.json",
    }
    prediction = json.loads((output / "identifiability_on_predictions.json").read_text())
    assert prediction["schema"] == "loslab.scalar_regression_predictions.v1"
    assert {row["split"] for row in prediction["predictions"]} == {"calibration", "test"}
    report = json.loads((output / "identifiability_on_uncertainty_report.json").read_text())
    assert (
        report["input_provenance"]["dataset_source_sha256"] == prediction["dataset_source_sha256"]
    )
    assert report["input_provenance"]["calibration_and_test_are_disjoint_groups"] is True
    assert report["metrics"]["calibration_group_count"] == 6
    assert report["metrics"]["group_conformal_intervals"]["0.95"]["finite_sample_unbounded"]
    paired = json.loads((output / "paired_test_comparisons.json").read_text())
    assert len(paired["comparisons"]) == 3
    assert paired["comparisons"][0]["comparison_unit"] == "independent_group"
    summary_path = tmp_path / "summary.json"
    summary = summarize_run(output, summary_path)
    assert summary["schema"] == "loslab.observable_multipath_summary.v3"
    assert len(summary["verified_artifacts"]) == 12
    set_report = summary["profile_likelihood_delay_sets"]
    assert set_report["calibration"]["finite_sample_unbounded"]
    assert set_report["test"]["sample_coverage"] == 1.0
    assert set_report["test_multicomponent_sample_count"] == 0
    assert "rows" not in set_report["test"]
    assert "group_coverage" not in set_report["test"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        summarize_run(output, summary_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_experiment(output, num_groups=40, epochs=1)

    manifest_path = output / "run_manifest.json"
    tampered = json.loads(manifest_path.read_text())
    tampered["run_replay"]["inputs"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="run replay contract digest mismatch"):
        summarize_run(output, tmp_path / "tampered_summary.json")
