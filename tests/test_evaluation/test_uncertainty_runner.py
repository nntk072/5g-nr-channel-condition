"""Input-schema and provenance tests for the grouped uncertainty runner."""

from __future__ import annotations

import json

import pytest
from scripts.evaluate_regression_uncertainty import evaluate_prediction_file


def test_runner_evaluates_frozen_group_disjoint_prediction_file(tmp_path):
    rows = []
    for index in range(20):
        rows.append(
            {
                "sample_id": f"cal-{index}",
                "group_id": f"cal-scene-{index}",
                "split": "calibration",
                "target": float(index),
                "prediction": float(index) + 0.25,
                "uncertainty_scale": 1.0,
            }
        )
    for index in range(4):
        rows.append(
            {
                "sample_id": f"test-{index}",
                "group_id": f"test-scene-{index // 2}",
                "split": "test",
                "target": float(index),
                "prediction": float(index) + 0.5,
                "uncertainty_scale": 1.0,
                "identifiability_stratum": "ambiguous" if index >= 2 else "resolved",
            }
        )
    source = tmp_path / "predictions.json"
    source.write_text(
        json.dumps(
            {
                "schema": "loslab.scalar_regression_predictions.v1",
                "target_name": "relative delay",
                "target_unit": "ns",
                "predictions": rows,
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_prediction_file(source, seed=7, bootstrap_draws=100)

    assert report["status"] == "DESCRIPTIVE_EVALUATION_NOT_A_RESEARCH_CLAIM"
    assert report["metrics"]["test_group_count"] == 2
    assert report["metrics"]["test_interval_metrics_by_stratum"]["0.9"].keys() == {
        "ambiguous",
        "resolved",
    }
    assert len(report["prediction_file_sha256"]) == 64


def test_runner_rejects_unknown_schema_duplicate_samples_and_missing_fields(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps({"schema": "wrong", "predictions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        evaluate_prediction_file(source)

    sample = {
        "sample_id": "same",
        "group_id": "scene",
        "split": "calibration",
        "target": 0.0,
        "prediction": 0.0,
        "uncertainty_scale": 1.0,
    }
    source.write_text(
        json.dumps(
            {
                "schema": "loslab.scalar_regression_predictions.v1",
                "predictions": [sample, sample, {**sample, "sample_id": "test", "split": "test"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_prediction_file(source)

    source.write_text(
        json.dumps(
            {
                "schema": "loslab.scalar_regression_predictions.v1",
                "predictions": [
                    sample,
                    {"sample_id": "test", "group_id": "test-scene", "split": "test"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing fields"):
        evaluate_prediction_file(source)
