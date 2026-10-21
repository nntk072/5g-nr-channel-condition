"""JSON input contract for grouped scalar-regression uncertainty evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from artifacts.provenance import package_source_sha256, sha256_file, versions

from evaluation.uncertainty import evaluate_grouped_regression_uncertainty


def evaluate_prediction_file(
    source: Path, *, seed: int = 42, bootstrap_draws: int = 2000
) -> dict[str, Any]:
    """Read a frozen calibration/test prediction file and return an audit report."""
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema") != "loslab.scalar_regression_predictions.v1":
        raise ValueError("unsupported prediction schema")
    rows = payload.get("predictions")
    if not isinstance(rows, list) or not rows:
        raise ValueError("predictions must be a nonempty list")
    sample_ids: set[str] = set()
    by_split: dict[str, list[dict[str, Any]]] = {"calibration": [], "test": []}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each prediction must be an object")
        sample_id = row.get("sample_id")
        split = row.get("split")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError("each prediction needs a nonempty sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if split not in by_split:
            raise ValueError("split must be exactly 'calibration' or 'test'")
        required = {"group_id", "target", "prediction", "uncertainty_scale"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"prediction {sample_id} is missing fields: {sorted(missing)}")
        if not isinstance(row["group_id"], str) or not row["group_id"].strip():
            raise ValueError(f"prediction {sample_id} needs a nonempty string group_id")
        if "identifiability_stratum" in row and (
            not isinstance(row["identifiability_stratum"], str)
            or not row["identifiability_stratum"].strip()
        ):
            raise ValueError(f"prediction {sample_id} has an invalid identifiability_stratum")
        by_split[split].append(row)
    if not by_split["calibration"] or not by_split["test"]:
        raise ValueError("both calibration and test predictions are required")

    calibration = by_split["calibration"]
    test = by_split["test"]
    report = evaluate_grouped_regression_uncertainty(
        [row["target"] for row in calibration],
        [row["prediction"] for row in calibration],
        [row["uncertainty_scale"] for row in calibration],
        [row["group_id"] for row in calibration],
        [row["target"] for row in test],
        [row["prediction"] for row in test],
        [row["uncertainty_scale"] for row in test],
        [row["group_id"] for row in test],
        test_stratum=[row.get("identifiability_stratum", "all") for row in test],
        seed=seed,
        bootstrap_draws=bootstrap_draws,
    )
    return {
        "schema": "loslab.grouped_regression_uncertainty_report.v1",
        "status": "DESCRIPTIVE_EVALUATION_NOT_A_RESEARCH_CLAIM",
        "method_name": str(payload.get("method_name", "unspecified")),
        "experiment_id": str(payload.get("experiment_id", "unspecified")),
        "target_name": str(payload.get("target_name", "unspecified scalar target")),
        "target_unit": str(payload.get("target_unit", "unspecified")),
        "prediction_file": str(source),
        "prediction_file_sha256": sha256_file(source),
        "seed": seed,
        "metrics": report,
        "runtime_versions": versions(),
        "package_source_sha256": package_source_sha256(),
    }
