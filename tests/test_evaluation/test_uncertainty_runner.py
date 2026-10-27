"""Input-schema and provenance tests for the grouped uncertainty runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import evaluation.uncertainty_io as uncertainty_io
import jsonschema
import pytest
from scripts.evaluate_regression_uncertainty import evaluate_prediction_file


def test_prediction_file_schema_accepts_runner_fixture_and_rejects_extra_fields(tmp_path):
    schema = json.loads(
        Path("evaluation/schema/scalar-regression-predictions.v1.schema.json").read_text()
    )
    payload = {
        "schema": "loslab.scalar_regression_predictions.v1",
        "predictions": [
            {
                "sample_id": "cal-1",
                "group_id": "scene-1",
                "split": "calibration",
                "target": 0.0,
                "prediction": 0.0,
                "uncertainty_scale": 1.0,
            }
        ],
    }
    jsonschema.Draft202012Validator(schema).validate(payload)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate({**payload, "extra": True})


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
    assert report["prediction_file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    schema = json.loads(
        Path("evaluation/schema/grouped-regression-uncertainty-report.v1.schema.json").read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(report)


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


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            '{"schema":"loslab.scalar_regression_predictions.v1",'
            '"schema":"loslab.scalar_regression_predictions.v1","predictions":[]}',
            "duplicate JSON field",
        ),
        (
            '{"schema":"loslab.scalar_regression_predictions.v1","predictions":[],"extra":1}',
            "unknown or invalid top-level fields",
        ),
        (
            '{"schema":"loslab.scalar_regression_predictions.v1","predictions":[NaN]}',
            "non-finite JSON value",
        ),
    ],
)
def test_runner_rejects_ambiguous_or_unknown_json(tmp_path, raw, message):
    source = tmp_path / "input.json"
    source.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        evaluate_prediction_file(source)


def test_runner_rejects_noncanonical_prediction_ids(tmp_path):
    source = tmp_path / "padded-ids.json"
    source.write_text(
        json.dumps(
            {
                "schema": "loslab.scalar_regression_predictions.v1",
                "predictions": [
                    {
                        "sample_id": " sample ",
                        "group_id": "scene",
                        "split": "calibration",
                        "target": 1.0,
                        "prediction": 1.0,
                        "uncertainty_scale": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical nonempty string"):
        evaluate_prediction_file(source)


def test_runner_rejects_nonstring_split_and_unknown_row_fields(tmp_path):
    source = tmp_path / "invalid-row.json"
    row = {
        "sample_id": "sample",
        "group_id": "scene",
        "split": [],
        "target": 1.0,
        "prediction": 1.0,
        "uncertainty_scale": 1.0,
    }
    source.write_text(
        json.dumps(
            {
                "schema": "loslab.scalar_regression_predictions.v1",
                "predictions": [row],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split must be exactly"):
        evaluate_prediction_file(source)

    row["split"] = "test"
    row["unrecognized"] = "value"
    source.write_text(
        json.dumps(
            {
                "schema": "loslab.scalar_regression_predictions.v1",
                "predictions": [row],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        evaluate_prediction_file(source)


def test_runner_bounds_and_stabilizes_prediction_file_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "large.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(uncertainty_io, "_MAX_PREDICTION_FILE_BYTES", 1)
    with pytest.raises(ValueError, match="exceeds 1 bytes"):
        evaluate_prediction_file(source)


def test_runner_rejects_symlink_prediction_files(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(ValueError, match="single-link regular file"):
        evaluate_prediction_file(alias)
