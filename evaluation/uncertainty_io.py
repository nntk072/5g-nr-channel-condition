"""JSON input contract for grouped scalar-regression uncertainty evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

from artifacts.build_dag import BuildNodeSpec, resolve_build_artifact, run_build_node
from artifacts.provenance import package_source_sha256, sha256_file, versions

from evaluation.uncertainty import evaluate_grouped_regression_uncertainty

_MAX_PREDICTION_FILE_BYTES = 64 * 1024 * 1024
_PREDICTION_FIELDS = {
    "schema",
    "predictions",
    "method_name",
    "experiment_id",
    "target_name",
    "target_unit",
    "dataset_source_sha256",
    "seed",
    "split_policy",
    "calibration_and_test_are_disjoint_groups",
    "research_status",
    "identifiability_weight",
    "group_unit",
}
_PREDICTION_ROW_FIELDS = {
    "sample_id",
    "group_id",
    "split",
    "target",
    "prediction",
    "uncertainty_scale",
    "identifiability_stratum",
    "condition",
}
_UNCERTAINTY_BUILD_SCHEMA = {
    "type": "object",
    "required": ["operation", "seed", "bootstrap_draws"],
    "properties": {
        "operation": {"const": "grouped_regression_uncertainty_evaluation_v1"},
        "seed": {"type": "integer", "minimum": 0},
        "bootstrap_draws": {"type": "integer", "minimum": 100},
    },
    "additionalProperties": False,
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
    provenance: dict[str, Any] = {}
    if "dataset_source_sha256" in payload:
        digest = payload["dataset_source_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("dataset_source_sha256 must be a lowercase SHA-256 digest")
        provenance["dataset_source_sha256"] = digest
    if "seed" in payload:
        source_seed = payload["seed"]
        if isinstance(source_seed, bool) or not isinstance(source_seed, int) or source_seed < 0:
            raise ValueError("prediction seed must be a nonnegative integer")
        provenance["seed"] = source_seed
    if "split_policy" in payload:
        provenance["split_policy"] = _canonical_text(payload["split_policy"], name="split_policy")
    if "calibration_and_test_are_disjoint_groups" in payload:
        if payload["calibration_and_test_are_disjoint_groups"] is not True:
            raise ValueError("calibration and test predictions must use disjoint groups")
        provenance["calibration_and_test_are_disjoint_groups"] = True
    if "research_status" in payload:
        provenance["research_status"] = _canonical_text(
            payload["research_status"], name="research_status"
        )
    if "identifiability_weight" in payload:
        weight = payload["identifiability_weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ValueError("identifiability_weight must be finite and positive")
        if not math.isfinite(float(weight)) or float(weight) <= 0:
            raise ValueError("identifiability_weight must be finite and positive")
        provenance["identifiability_weight"] = float(weight)
    group_unit = payload.get("group_unit", "caller_declared_group")
    if not isinstance(group_unit, str) or group_unit not in {
        "caller_declared_group",
        "physical_episode",
        "site",
        "acquisition_unit",
    }:
        raise ValueError(
            "group_unit must be caller_declared_group, physical_episode, site, or acquisition_unit"
        )
    provenance["group_unit"] = group_unit
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
        if "condition" in row:
            condition = row["condition"]
            if not isinstance(condition, dict) or set(condition) != {
                "bandwidth_hz",
                "snr_db",
                "separation_fraction_of_reciprocal_bandwidth",
            }:
                raise ValueError(f"prediction {sample_id} has an invalid condition descriptor")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in condition.values()
            ):
                raise ValueError(f"prediction {sample_id} condition values must be finite numbers")
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
        group_unit=group_unit,
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
        "input_provenance": provenance,
        "prediction_file": str(source),
        "prediction_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "seed": seed,
        "metrics": report,
        "runtime_versions": versions(),
        "package_source_sha256": package_source_sha256(),
    }


def evaluate_prediction_file_build_node(
    source: Path,
    cache_root: str | Path,
    *,
    seed: int = 42,
    bootstrap_draws: int = 2000,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a frozen prediction file through an immutable build-DAG node."""
    input_path = Path(source).expanduser().resolve()
    input_digest = sha256_file(input_path)
    runtime = versions()
    spec = BuildNodeSpec(
        node_id="evaluation.grouped_regression_uncertainty",
        node_version="1.0.0",
        configuration={
            "operation": "grouped_regression_uncertainty_evaluation_v1",
            "seed": seed,
            "bootstrap_draws": bootstrap_draws,
        },
        configuration_schema=_UNCERTAINTY_BUILD_SCHEMA,
        source_revision=runtime.get("git_sha") or "unknown",
        source_bundle_sha256=package_source_sha256(),
        input_files={str(input_path): input_digest},
        dependencies={},
        outputs=("uncertainty-report.json",),
    )

    def build(staging: Path) -> None:
        report = evaluate_prediction_file(
            input_path, seed=seed, bootstrap_draws=bootstrap_draws
        )
        (staging / "uncertainty-report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    result = run_build_node(cache_root, spec, build)
    output_path = resolve_build_artifact(cache_root, result, "uncertainty-report.json")
    report_bytes = _read_prediction_snapshot(output_path)
    try:
        report = json.loads(report_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cached uncertainty report must be valid UTF-8 JSON") from error
    import jsonschema

    report_schema_path = (
        Path(__file__).resolve().parent / "schema/grouped-regression-uncertainty-report.v1.schema.json"
    )
    jsonschema.validate(report, json.loads(report_schema_path.read_text(encoding="utf-8")))
    manifest_bytes = _read_prediction_snapshot(result.manifest_path)
    return report, {
        "node_key": result.node_key,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "artifact_sha256": result.artifacts[0]["sha256"],
        "cache_reused": result.reused,
    }
