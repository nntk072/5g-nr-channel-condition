"""JSON input contract for grouped scalar-regression uncertainty evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from artifacts.provenance import package_source_sha256, versions

from evaluation.uncertainty import evaluate_grouped_regression_uncertainty

_MAX_PREDICTION_FILE_BYTES = 64 * 1024 * 1024
_PREDICTION_FIELDS = {
    "schema",
    "predictions",
    "method_name",
    "experiment_id",
    "target_name",
    "target_unit",
}
_PREDICTION_ROW_FIELDS = {
    "sample_id",
    "group_id",
    "split",
    "target",
    "prediction",
    "uncertainty_scale",
    "identifiability_stratum",
}


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_prediction_snapshot(source: Path) -> bytes:
    """Read one bounded, stable, single-link file snapshot without symlinks."""
    path_before = os.lstat(source)
    if not stat.S_ISREG(path_before.st_mode) or path_before.st_nlink != 1:
        raise ValueError("prediction input must be a single-link regular file")
    if not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("platform lacks safe no-symlink prediction-file opening")
    descriptor = os.open(
        source,
        os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
    )
    try:
        before = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(path_before):
            raise ValueError("prediction input changed while opening")
        if before.st_size > _MAX_PREDICTION_FILE_BYTES:
            raise ValueError(f"prediction input exceeds {_MAX_PREDICTION_FILE_BYTES} bytes")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_PREDICTION_FILE_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > _MAX_PREDICTION_FILE_BYTES:
                raise ValueError(f"prediction input exceeds {_MAX_PREDICTION_FILE_BYTES} bytes")
        after = os.fstat(descriptor)
        path_after = os.lstat(source)
        if (
            size != before.st_size
            or _file_identity(after) != _file_identity(before)
            or _file_identity(path_after) != _file_identity(before)
        ):
            raise ValueError("prediction input changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"prediction input has duplicate JSON field {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"prediction input contains non-finite JSON value {value}")


def _canonical_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical nonempty string")
    return value


def evaluate_prediction_file(
    source: Path, *, seed: int = 42, bootstrap_draws: int = 2000
) -> dict[str, Any]:
    """Read a frozen calibration/test prediction file and return an audit report."""
    source_bytes = _read_prediction_snapshot(source)
    try:
        payload = json.loads(
            source_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("prediction input must be valid UTF-8 JSON") from error
    if not isinstance(payload, dict) or set(payload).difference(_PREDICTION_FIELDS):
        raise ValueError("prediction input has unknown or invalid top-level fields")
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
        sample_id = _canonical_text(sample_id, name="sample_id")
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if not isinstance(split, str) or split not in by_split:
            raise ValueError("split must be exactly 'calibration' or 'test'")
        required = {"sample_id", "group_id", "split", "target", "prediction", "uncertainty_scale"}
        missing = required.difference(row)
        if missing:
            raise ValueError(f"prediction {sample_id} is missing fields: {sorted(missing)}")
        if set(row).difference(_PREDICTION_ROW_FIELDS):
            raise ValueError(f"prediction {sample_id} has unknown fields")
        _canonical_text(row["group_id"], name=f"prediction {sample_id} group_id")
        if "identifiability_stratum" in row:
            _canonical_text(
                row["identifiability_stratum"],
                name=f"prediction {sample_id} identifiability_stratum",
            )
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
        "method_name": _canonical_text(
            payload.get("method_name", "unspecified"), name="method_name"
        ),
        "experiment_id": _canonical_text(
            payload.get("experiment_id", "unspecified"), name="experiment_id"
        ),
        "target_name": _canonical_text(
            payload.get("target_name", "unspecified scalar target"), name="target_name"
        ),
        "target_unit": _canonical_text(
            payload.get("target_unit", "unspecified"), name="target_unit"
        ),
        "prediction_file": str(source),
        "prediction_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": seed,
        "metrics": report,
        "runtime_versions": versions(),
        "package_source_sha256": package_source_sha256(),
    }
