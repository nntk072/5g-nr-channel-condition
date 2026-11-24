"""Prediction-file contract for grouped regression uncertainty: parsing, file safety, caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import evaluation.scoring.uncertainty_io as uncertainty_io
import numpy as np
import pytest
from evaluation.scoring.uncertainty_io import (
    evaluate_prediction_file,
    evaluate_prediction_file_build_node,
)


def _rows(split: str, count: int, offset: float) -> list[dict]:
    rng = np.random.default_rng(int(offset * 100) + 3)
    rows = []
    for index in range(count):
        target = float(rng.normal())
        rows.append(
            {
                "sample_id": f"{split}-{index}",
                "group_id": f"{split}-g{index % 6}",
                "split": split,
                "target": target,
                "prediction": target + float(rng.normal(scale=0.2)) + offset,
                "uncertainty_scale": 0.3 + 0.05 * (index % 3),
            }
        )
    return rows


def _payload(**overrides) -> dict:
    payload = {
        "schema": "loslab.scalar_regression_predictions.v1",
        "method_name": "ridge",
        "experiment_id": "exp-1",
        "target_name": "delay",
        "target_unit": "s",
        "dataset_source_sha256": hashlib.sha256(b"dataset").hexdigest(),
        "seed": 5,
        "split_policy": "grouped holdout",
        "calibration_and_test_are_disjoint_groups": True,
        "research_status": "fixture",
        "identifiability_weight": 2.0,
        "group_unit": "site",
        "predictions": _rows("calibration", 36, 0.0) + _rows("test", 24, 0.1),
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_valid_prediction_file_reports_provenance_and_the_exact_file_digest(tmp_path) -> None:
    source = _write(tmp_path / "predictions.json", _payload())
    report = evaluate_prediction_file(source, seed=9, bootstrap_draws=100)
    assert report["schema"] == "loslab.grouped_regression_uncertainty_report.v1"
    assert report["status"] == "DESCRIPTIVE_EVALUATION_NOT_A_RESEARCH_CLAIM"
    assert report["method_name"] == "ridge"
    assert report["target_unit"] == "s"
    assert report["seed"] == 9
    assert report["prediction_file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = report["input_provenance"]
    assert provenance["group_unit"] == "site"
    assert provenance["identifiability_weight"] == pytest.approx(2.0)
    assert provenance["calibration_and_test_are_disjoint_groups"] is True
    assert provenance["seed"] == 5 and len(provenance["dataset_source_sha256"]) == 64
    assert "metrics" in report and report["metrics"]


def test_defaults_apply_when_optional_fields_are_absent(tmp_path) -> None:
    minimal = {
        "schema": "loslab.scalar_regression_predictions.v1",
        "predictions": _rows("calibration", 24, 0.0) + _rows("test", 18, 0.1),
    }
    report = evaluate_prediction_file(
        _write(tmp_path / "minimal.json", minimal), bootstrap_draws=100
    )
    assert report["method_name"] == "unspecified"
    assert report["experiment_id"] == "unspecified"
    assert report["target_name"] == "unspecified scalar target"
    assert report["input_provenance"]["group_unit"] == "caller_declared_group"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra_top": 1}, "unknown or invalid top-level"),
        ({"schema": "other"}, "unsupported prediction schema"),
        ({"dataset_source_sha256": "ABC"}, "lowercase SHA-256"),
        ({"seed": True}, "nonnegative integer"),
        ({"seed": -1}, "nonnegative integer"),
        ({"split_policy": "   "}, "canonical nonempty"),
        ({"calibration_and_test_are_disjoint_groups": False}, "disjoint groups"),
        ({"research_status": ""}, "canonical nonempty"),
        ({"identifiability_weight": True}, "finite and positive"),
        ({"identifiability_weight": 0.0}, "finite and positive"),
        ({"identifiability_weight": "2"}, "finite and positive"),
        ({"group_unit": "episode"}, "group_unit must be"),
        ({"predictions": []}, "nonempty list"),
        ({"predictions": ["not-an-object"]}, "each prediction must be an object"),
        ({"method_name": " padded "}, "method_name"),
    ],
)
def test_top_level_fields_are_validated(tmp_path, overrides, message) -> None:
    payload = _payload()
    if "extra_top" in overrides:
        payload["extra_top"] = overrides["extra_top"]
    else:
        payload.update(overrides)
    with pytest.raises(ValueError, match=message):
        evaluate_prediction_file(_write(tmp_path / "bad.json", payload), bootstrap_draws=100)


def _row_mutation(index: int, **changes) -> dict:
    payload = _payload()
    payload["predictions"][index].update(changes)
    return payload


@pytest.mark.parametrize(
    ("payload_factory", "message"),
    [
        (lambda: _row_mutation(0, sample_id=""), "canonical nonempty"),
        (lambda: _row_mutation(1, sample_id="calibration-0"), "duplicate sample_id"),
        (lambda: _row_mutation(0, split="validation"), "exactly 'calibration' or 'test'"),
        (lambda: _row_mutation(0, split=None), "exactly 'calibration' or 'test'"),
        (lambda: _row_mutation(0, group_id=" "), "group_id"),
        (lambda: _row_mutation(0, identifiability_stratum=""), "identifiability_stratum"),
        (lambda: _row_mutation(0, condition={"bandwidth_hz": 1.0}), "invalid condition descriptor"),
        (
            lambda: _row_mutation(
                0,
                condition={
                    "bandwidth_hz": True,
                    "snr_db": 1.0,
                    "separation_fraction_of_reciprocal_bandwidth": 0.5,
                },
            ),
            "must be finite numbers",
        ),
        (
            lambda: _row_mutation(
                0,
                condition={
                    "bandwidth_hz": 1e8,
                    "snr_db": float("inf"),
                    "separation_fraction_of_reciprocal_bandwidth": 0.5,
                },
            ),
            "non-finite JSON value",
        ),
        (lambda: _row_mutation(0, unexpected=1), "unknown fields"),
    ],
)
def test_prediction_rows_are_validated(tmp_path, payload_factory, message) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_prediction_file(
            _write(tmp_path / "rows.json", payload_factory()), bootstrap_draws=100
        )


def test_missing_row_fields_are_named(tmp_path) -> None:
    payload = _payload()
    del payload["predictions"][0]["prediction"]
    with pytest.raises(ValueError, match="missing fields"):
        evaluate_prediction_file(_write(tmp_path / "missing.json", payload), bootstrap_draws=100)


def test_both_calibration_and_test_partitions_are_required(tmp_path) -> None:
    payload = _payload(predictions=_rows("calibration", 24, 0.0))
    with pytest.raises(ValueError, match="both calibration and test"):
        evaluate_prediction_file(_write(tmp_path / "one.json", payload), bootstrap_draws=100)


def test_valid_optional_condition_and_stratum_are_accepted(tmp_path) -> None:
    payload = _payload()
    payload["predictions"][0]["condition"] = {
        "bandwidth_hz": 1e8,
        "snr_db": 10.0,
        "separation_fraction_of_reciprocal_bandwidth": 0.25,
    }
    payload["predictions"][-1]["identifiability_stratum"] = "resolved"
    report = evaluate_prediction_file(
        _write(tmp_path / "optional.json", payload), bootstrap_draws=100
    )
    assert report["metrics"]


def test_text_that_is_not_utf8_json_is_rejected(tmp_path) -> None:
    binary = tmp_path / "binary.json"
    binary.write_bytes(b"\xff\xfe\x00")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        evaluate_prediction_file(binary, bootstrap_draws=100)
    truncated = tmp_path / "truncated.json"
    truncated.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        evaluate_prediction_file(truncated, bootstrap_draws=100)


def test_non_finite_json_constants_and_duplicate_fields_are_rejected(tmp_path) -> None:
    constant = tmp_path / "constant.json"
    constant.write_text('{"schema": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON value"):
        evaluate_prediction_file(constant, bootstrap_draws=100)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": "a", "schema": "b"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        evaluate_prediction_file(duplicate, bootstrap_draws=100)


def test_source_file_must_be_a_single_link_regular_file(tmp_path) -> None:
    source = _write(tmp_path / "source.json", _payload())
    link = tmp_path / "linked.json"
    os.link(source, link)
    with pytest.raises(ValueError, match="single-link regular file"):
        evaluate_prediction_file(source, bootstrap_draws=100)
    alias = tmp_path / "alias.json"
    alias.symlink_to(tmp_path / "source.json")
    with pytest.raises(ValueError, match="single-link regular file"):
        evaluate_prediction_file(alias, bootstrap_draws=100)
    directory = tmp_path / "folder"
    directory.mkdir()
    with pytest.raises(ValueError, match="single-link regular file"):
        evaluate_prediction_file(directory, bootstrap_draws=100)


def test_oversized_prediction_files_are_refused(tmp_path, monkeypatch) -> None:
    source = _write(tmp_path / "big.json", _payload())
    monkeypatch.setattr(uncertainty_io, "_MAX_PREDICTION_FILE_BYTES", 16)
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_prediction_file(source, bootstrap_draws=100)


def test_platform_without_no_follow_open_is_refused(tmp_path, monkeypatch) -> None:
    source = _write(tmp_path / "platform.json", _payload())
    monkeypatch.delattr(uncertainty_io.os, "O_NOFOLLOW", raising=False)
    with pytest.raises(ValueError, match="no-symlink"):
        evaluate_prediction_file(source, bootstrap_draws=100)


def test_build_node_evaluates_once_and_reuses_the_cached_report(tmp_path) -> None:
    source = _write(tmp_path / "node.json", _payload())
    cache = tmp_path / "cache"
    report, first = evaluate_prediction_file_build_node(source, cache, seed=3, bootstrap_draws=100)
    assert report["schema"] == "loslab.grouped_regression_uncertainty_report.v1"
    assert first["cache_reused"] is False
    assert len(first["node_key"]) == 64 and len(first["artifact_sha256"]) == 64
    again_report, second = evaluate_prediction_file_build_node(
        source, cache, seed=3, bootstrap_draws=100
    )
    assert second["cache_reused"] is True
    assert second["node_key"] == first["node_key"]
    assert again_report == report
    other_seed, third = evaluate_prediction_file_build_node(
        source, cache, seed=4, bootstrap_draws=100
    )
    assert third["node_key"] != first["node_key"]
    assert other_seed["seed"] == 4
