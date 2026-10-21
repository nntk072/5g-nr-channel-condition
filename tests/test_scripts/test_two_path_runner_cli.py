"""End-to-end artifact contract for the grouped two-path pilot runner."""

from __future__ import annotations

import json

import pytest
from scripts.run_observable_multipath_experiment import run_experiment
from scripts.summarize_observable_multipath_experiment import summarize_run


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
    assert set(manifest["artifacts"]) == {
        "identifiability_off_predictions.json",
        "identifiability_on_predictions.json",
        "profile_likelihood_predictions.json",
        "identifiability_off_uncertainty_report.json",
        "identifiability_on_uncertainty_report.json",
        "profile_likelihood_uncertainty_report.json",
        "paired_test_comparisons.json",
        "decoder_identifiability_off.npz",
        "decoder_identifiability_on.npz",
    }
    prediction = json.loads((output / "identifiability_on_predictions.json").read_text())
    assert prediction["schema"] == "loslab.scalar_regression_predictions.v1"
    assert {row["split"] for row in prediction["predictions"]} == {"calibration", "test"}
    report = json.loads((output / "identifiability_on_uncertainty_report.json").read_text())
    assert report["metrics"]["calibration_group_count"] == 6
    assert report["metrics"]["group_conformal_intervals"]["0.95"]["finite_sample_unbounded"]
    paired = json.loads((output / "paired_test_comparisons.json").read_text())
    assert len(paired["comparisons"]) == 3
    assert paired["comparisons"][0]["comparison_unit"] == "independent_group"
    summary_path = tmp_path / "summary.json"
    summary = summarize_run(output, summary_path)
    assert summary["schema"] == "loslab.observable_multipath_summary.v1"
    assert len(summary["verified_artifacts"]) == 9
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        summarize_run(output, summary_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_experiment(output, num_groups=40, epochs=1)
