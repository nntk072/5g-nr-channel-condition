"""Measurement-driven, non-qualifying resource-budget preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_REQUEST_SCHEMA = _ROOT / "compute/schema/resource-budget-request.v1.schema.json"
_REPORT_SCHEMA = _ROOT / "compute/schema/resource-budget-report.v1.schema.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TIME_METRICS = (
    "solver_runtime_s",
    "receiver_runtime_s",
    "preprocessing_runtime_s",
    "downstream_runtime_s",
)
_STORAGE_METRICS = (
    "raw_path_bytes",
    "cfr_bytes",
    "waveform_bytes",
    "manifest_bytes",
    "other_artifact_bytes",
)
_COUNT_METRICS = ("gpu_peak_memory_bytes", *_STORAGE_METRICS)
_QUANTILES = (0.5, 0.9, 0.99)


def _load_schema(path: Path) -> dict[str, Any]:
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load resource-budget schema: {path}") from error
    import jsonschema

    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def _validate(value: Any, schema_path: Path, label: str) -> None:
    import jsonschema

    errors = list(jsonschema.Draft202012Validator(_load_schema(schema_path)).iter_errors(value))
    if errors:
        raise ValueError(f"{label} violates schema: " + "; ".join(e.message for e in errors[:10]))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _quantile(values: list[float | int], q: float) -> float | int:
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        ordered = sorted(values)
        return ordered[math.ceil(q * (len(ordered) - 1))]
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def _headroom_integer(value: int, planned_units: int, factor: float) -> int:
    return math.ceil(Decimal(value) * planned_units * Decimal(str(factor)))


def build_resource_budget_report(request: dict[str, Any], *, request_sha256: str) -> dict[str, Any]:
    """Project costs from measured pilot rows; never runs solver or receiver work.

    The estimate is explicitly conditional on the supplied pilot units being
    representative of each planned stratum. Runtime and cumulative storage are
    projected linearly from empirical per-unit quantiles, then multiplied by
    the caller's required 2x–3x headroom factor. Device peak memory is not
    multiplied by planned unit count.
    """

    _validate(request, _REQUEST_SCHEMA, "resource-budget request")
    if not isinstance(request_sha256, str) or not _SHA256.fullmatch(request_sha256):
        raise ValueError("request_sha256 must be lowercase SHA-256 hex")
    environment_sha256 = hashlib.sha256(_canonical_json(request["runtime_environment"])).hexdigest()
    if request["environment_sha256"] != environment_sha256:
        raise ValueError("environment_sha256 does not match the canonical runtime_environment")
    headroom = float(request["headroom_factor"])
    if len({row["stratum_id"] for row in request["strata"]}) != len(request["strata"]):
        raise ValueError("stratum IDs must be unique")
    strata_reports: list[dict[str, Any]] = []
    totals = {
        metric: {f"p{int(q * 100)}": (0.0 if metric in _TIME_METRICS else 0) for q in _QUANTILES}
        for metric in (*_TIME_METRICS, *_STORAGE_METRICS)
    }
    total_planned_units = 0
    peak_memory = 0

    for stratum in request["strata"]:
        observations = stratum["pilot_observations"]
        pairs = [(row["scene_id"], row["link_id"]) for row in observations]
        conditions = [row["condition_id"] for row in observations]
        measurement_records = [row["measurement_record_sha256"] for row in observations]
        if (
            len(set(pairs)) != len(pairs)
            or len(set(conditions)) != len(conditions)
            or len(set(measurement_records)) != len(measurement_records)
        ):
            raise ValueError(
                f"stratum {stratum['stratum_id']} pilot rows need distinct scene/link conditions"
            )
        if any(row["scene_family_id"] != stratum["family_id"] for row in observations):
            raise ValueError(
                f"stratum {stratum['stratum_id']} contains a pilot from a different scene family"
            )
        total_planned_units += stratum["planned_units"]
        distribution: dict[str, dict[str, float | int]] = {}
        for metric in (*_TIME_METRICS, *_COUNT_METRICS):
            sample = [row[metric] for row in observations]
            summary = {f"p{int(q * 100)}": _quantile(sample, q) for q in _QUANTILES}
            distribution[metric] = summary
            if metric == "gpu_peak_memory_bytes":
                peak_memory = max(peak_memory, _headroom_integer(summary["p99"], 1, headroom))
                continue
            total_by_quantile = totals[metric]
            if metric in _TIME_METRICS:
                for label in total_by_quantile:
                    total_by_quantile[label] += summary[label] * stratum["planned_units"] * headroom
            else:
                for label in total_by_quantile:
                    total_by_quantile[label] += _headroom_integer(
                        summary[label], stratum["planned_units"], headroom
                    )
        strata_reports.append(
            {
                "stratum_id": stratum["stratum_id"],
                "family_id": stratum["family_id"],
                "planned_units": stratum["planned_units"],
                "pilot_condition_count": len(observations),
                "empirical_per_unit_quantiles": distribution,
                "headroom_factor": headroom,
                "projected_totals": {
                    metric: {
                        f"p{int(q * 100)}": (
                            summary[f"p{int(q * 100)}"] * stratum["planned_units"] * headroom
                            if metric in _TIME_METRICS
                            else _headroom_integer(
                                summary[f"p{int(q * 100)}"],
                                stratum["planned_units"],
                                headroom,
                            )
                        )
                        for q in _QUANTILES
                    }
                    for metric, summary in distribution.items()
                    if metric != "gpu_peak_memory_bytes"
                },
            }
        )

    report = {
        "schema": "loslab.resource_budget_report.v1",
        "status": "MEASUREMENT_CONDITIONAL_NOT_QUALIFIED",
        "request_id": request["request_id"],
        "measurement_batch_id": request["measurement_batch_id"],
        "environment_sha256": request["environment_sha256"],
        "runtime_environment": request["runtime_environment"],
        "request_sha256": request_sha256,
        "request_schema_sha256": hashlib.sha256(_REQUEST_SCHEMA.read_bytes()).hexdigest(),
        "report_schema_sha256": hashlib.sha256(_REPORT_SCHEMA.read_bytes()).hexdigest(),
        "workflow_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cli_source_sha256": hashlib.sha256(
            (_ROOT / "loslab/__main__.py").read_bytes()
        ).hexdigest(),
        "estimation_method": "empirical_higher_quantiles_times_planned_units_times_headroom",
        "quantiles": [0.5, 0.9, 0.99],
        "headroom_factor": headroom,
        "pilot_condition_count": sum(row["pilot_condition_count"] for row in strata_reports),
        "planned_unit_count": total_planned_units,
        "strata": strata_reports,
        "projected_total_runtime_seconds": {metric: totals[metric] for metric in _TIME_METRICS},
        "projected_total_storage_bytes": {metric: totals[metric] for metric in _STORAGE_METRICS},
        "projected_peak_device_memory_bytes": peak_memory,
        "limitations": [
            "This preflight performs no RT, receiver, or model execution.",
            "Each projection assumes the measured pilot condition distribution represents its planned stratum and scales per-unit quantiles linearly.",
            "Empirical p99 with 10–100 pilot conditions is not a population guarantee or a confidence bound.",
            "Peak device memory is a per-process p99 estimate with headroom, not aggregate memory across concurrent workers.",
            "No monetary cost, throughput guarantee, capacity approval, qualification, or production authorization is emitted.",
        ],
    }
    _validate(report, _REPORT_SCHEMA, "resource-budget report")
    return report


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_stable_json(path: Path, *, maximum_bytes: int = 64 * 1024 * 1024) -> tuple[dict, bytes]:
    before_path = os.lstat(path)
    if not stat.S_ISREG(before_path.st_mode) or before_path.st_nlink != 1:
        raise ValueError("resource-budget request must be a single-link regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if before.st_size > maximum_bytes or (before.st_dev, before.st_ino) != (
            before_path.st_dev,
            before_path.st_ino,
        ):
            raise ValueError("resource-budget request exceeds its byte bound or changed on open")
        chunks = bytearray()
        while len(chunks) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(chunks)))
            if not chunk:
                break
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(path)
    finally:
        os.close(descriptor)

    def identity(row: os.stat_result) -> tuple[int, ...]:
        return (
            row.st_dev,
            row.st_ino,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
            row.st_nlink,
        )

    if (
        len(chunks) > maximum_bytes
        or identity(before) != identity(after)
        or identity(before) != identity(path_after)
    ):
        raise ValueError("resource-budget request changed during bounded read")
    raw = bytes(chunks)
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("resource-budget request must be strict UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("resource-budget request root must be an object")
    return value, raw


def _publish_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def run_resource_budget(request_path: Path, output_path: Path) -> dict[str, Any]:
    request, raw = _read_stable_json(request_path)
    report = build_resource_budget_report(request, request_sha256=hashlib.sha256(raw).hexdigest())
    _publish_exclusive(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = run_resource_budget(args.request, args.output)
    print(json.dumps({"status": report["status"], "request_id": report["request_id"]}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        print(f"resource budget failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
