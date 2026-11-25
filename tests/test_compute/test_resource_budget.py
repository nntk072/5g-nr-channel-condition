from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import compute.resource_budget as resource_budget
import pytest
from compute.resource_budget import (
    _headroom_integer,
    _load_schema,
    _read_stable_json,
    _validate,
    build_resource_budget_report,
    run_resource_budget,
)


def _request() -> dict:
    runtime_environment = {
        "platform": "linux",
        "machine": "x86_64",
        "python_version": "3.11.9",
        "numpy_version": "2.2.6",
        "sionna_version": "2.2.0",
        "torch_version": "2.9.0",
        "solver_backend": "llvm",
        "accelerator": "Tesla V100",
        "accelerator_driver": "580.95",
        "accelerator_runtime": "cuda-12.8",
    }
    observations = []
    for index in range(10):
        observations.append(
            {
                "condition_id": f"condition-{index}",
                "scene_family_id": "family-a",
                "scene_id": f"scene-{index}",
                "link_id": f"link-{index}",
                "configuration_sha256": f"{index + 1:064x}",
                "measurement_record_sha256": f"{index + 11:064x}",
                "solver_runtime_s": float(index + 1),
                "receiver_runtime_s": float(2 * (index + 1)),
                "preprocessing_runtime_s": 0.5,
                "downstream_runtime_s": 1.0,
                "gpu_peak_memory_bytes": 1000 * (index + 1),
                "raw_path_bytes": 100 * (index + 1),
                "cfr_bytes": 200 * (index + 1),
                "waveform_bytes": 300 * (index + 1),
                "manifest_bytes": 10,
                "other_artifact_bytes": 20,
            }
        )
    return {
        "schema": "loslab.resource_budget_request.v1",
        "request_id": "pilot-budget-01",
        "measurement_batch_id": "measured-pilot-batch-01",
        "environment_sha256": hashlib.sha256(
            json.dumps(
                runtime_environment,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "runtime_environment": runtime_environment,
        "headroom_factor": 2.0,
        "strata": [
            {
                "stratum_id": "family-a-los-siso",
                "family_id": "family-a",
                "planned_units": 100,
                "pilot_observations": observations,
            }
        ],
    }


def test_resource_budget_uses_measured_quantiles_and_separate_peak_memory():
    request = _request()
    report = build_resource_budget_report(
        request, request_sha256=hashlib.sha256(json.dumps(request).encode()).hexdigest()
    )

    assert report["status"] == "MEASUREMENT_CONDITIONAL_NOT_QUALIFIED"
    assert report["environment_sha256"] == request["environment_sha256"]
    assert len(report["workflow_source_sha256"]) == 64
    assert len(report["cli_source_sha256"]) == 64
    assert len(report["report_schema_sha256"]) == 64
    assert report["planned_unit_count"] == 100
    stratum = report["strata"][0]
    assert stratum["pilot_condition_count"] == 10
    assert stratum["empirical_per_unit_quantiles"]["solver_runtime_s"] == {
        "p50": 6.0,
        "p90": 10.0,
        "p99": 10.0,
    }
    assert report["projected_total_runtime_seconds"]["solver_runtime_s"]["p99"] == 2000
    assert report["projected_total_storage_bytes"]["raw_path_bytes"]["p99"] == 200000
    assert report["projected_peak_device_memory_bytes"] == 20000
    assert "population guarantee" in report["limitations"][2]


@pytest.mark.parametrize("mutation", ["duplicate_condition", "wrong_family", "nine_rows"])
def test_resource_budget_rejects_unrepresentative_or_undersized_pilot_roster(mutation):
    request = _request()
    rows = request["strata"][0]["pilot_observations"]
    if mutation == "duplicate_condition":
        rows[1]["condition_id"] = rows[0]["condition_id"]
    elif mutation == "wrong_family":
        rows[0]["scene_family_id"] = "family-b"
    else:
        request["strata"][0]["pilot_observations"] = rows[:9]

    with pytest.raises(ValueError):
        build_resource_budget_report(request, request_sha256="b" * 64)


def test_resource_budget_rejects_runtime_environment_digest_drift():
    request = _request()
    request["runtime_environment"]["accelerator"] = "different GPU"

    with pytest.raises(ValueError, match="does not match"):
        build_resource_budget_report(request, request_sha256="b" * 64)


def test_resource_budget_file_workflow_binds_exact_input_and_never_overwrites(tmp_path):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "report.json"
    request_bytes = json.dumps(_request(), indent=2).encode("utf-8")
    request_path.write_bytes(request_bytes)

    report = run_resource_budget(request_path, output_path)

    assert report["request_sha256"] == hashlib.sha256(request_bytes).hexdigest()
    assert json.loads(output_path.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError):
        run_resource_budget(request_path, output_path)


def test_resource_budget_cli_publishes_versioned_report(tmp_path):
    request_path = tmp_path / "request.json"
    output_path = tmp_path / "report.json"
    request_path.write_text(json.dumps(_request()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "compute.resource_budget",
            "--request",
            str(request_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["request_id"] == ("pilot-budget-01")


def test_resource_budget_keeps_large_storage_byte_arithmetic_exact():
    request = _request()
    for index, row in enumerate(request["strata"][0]["pilot_observations"]):
        row["raw_path_bytes"] = 2**60 + index

    report = build_resource_budget_report(request, request_sha256="c" * 64)

    assert report["projected_total_storage_bytes"]["raw_path_bytes"]["p99"] == (
        (2**60 + 9) * 100 * 2
    )


def test_resource_budget_rejects_duplicate_json_keys(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text('{"request_id":"a","request_id":"b"}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        run_resource_budget(request_path, tmp_path / "out.json")


def test_resource_budget_rejects_duplicate_stratum_ids():
    request = _request()
    duplicate = dict(request["strata"][0])
    duplicate["family_id"] = "family-b"
    request["strata"].append(duplicate)
    with pytest.raises(ValueError, match="stratum IDs must be unique"):
        build_resource_budget_report(request, request_sha256="b" * 64)


def test_resource_budget_validates_request_digest_and_integer_quantiles():
    request = _request()
    with pytest.raises(ValueError, match="request_sha256 must be lowercase"):
        build_resource_budget_report(request, request_sha256="not-a-digest")
    with pytest.raises(TypeError, match="must remain integral"):
        _headroom_integer(2.5, 10, 2.0)
    with pytest.raises(TypeError, match="must remain integral"):
        _headroom_integer(True, 10, 2.0)


@pytest.mark.parametrize("content", [b"\xff", b"not json", b"[]"])
def test_stable_json_rejects_invalid_encoding_json_and_root(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "request.json"
    path.write_bytes(content)
    message = "root must be an object" if content == b"[]" else "must be strict UTF-8 JSON"
    with pytest.raises(ValueError, match=message):
        _read_stable_json(path)


def test_stable_json_rejects_nonregular_and_oversized_request(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single-link regular file"):
        _read_stable_json(tmp_path)
    path = tmp_path / "large.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds its byte bound"):
        _read_stable_json(path, maximum_bytes=1)


def test_stable_json_rejects_changed_file_descriptor_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "request.json"
    path.write_text("{}", encoding="utf-8")
    real_fstat = os.fstat

    def changed_inode(descriptor: int) -> SimpleNamespace:
        actual = real_fstat(descriptor)
        return SimpleNamespace(
            st_size=actual.st_size,
            st_dev=actual.st_dev,
            st_ino=actual.st_ino + 1,
        )

    monkeypatch.setattr(resource_budget.os, "fstat", changed_inode)
    with pytest.raises(ValueError, match="changed on open"):
        _read_stable_json(path)


def test_stable_json_rejects_changed_file_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "request.json"
    path.write_text("{}", encoding="utf-8")
    real_fstat = os.fstat
    calls = 0

    def changed_mtime(descriptor: int) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        actual = real_fstat(descriptor)
        return SimpleNamespace(
            st_dev=actual.st_dev,
            st_ino=actual.st_ino,
            st_size=actual.st_size,
            st_mtime_ns=actual.st_mtime_ns + (calls == 2),
            st_ctime_ns=actual.st_ctime_ns,
            st_nlink=actual.st_nlink,
        )

    monkeypatch.setattr(resource_budget.os, "fstat", changed_mtime)
    with pytest.raises(ValueError, match="changed during bounded read"):
        _read_stable_json(path)


def test_stable_json_stops_after_reading_past_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "request.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(resource_budget.os, "read", lambda _descriptor, _size: b"xxx")
    with pytest.raises(ValueError, match="changed during bounded read"):
        _read_stable_json(path, maximum_bytes=2)


def test_schema_helpers_convert_read_and_validation_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot load resource-budget schema"):
        _load_schema(tmp_path / "missing.json")
    schema = Path(resource_budget._REQUEST_SCHEMA)
    with pytest.raises(ValueError, match="fixture violates schema"):
        _validate({}, schema, "fixture")


def test_main_prints_resource_budget_summary(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        resource_budget,
        "run_resource_budget",
        lambda _request, _output: {
            "status": "MEASUREMENT_CONDITIONAL_NOT_QUALIFIED",
            "request_id": "pilot-1",
        },
    )
    assert resource_budget.main(["--request", "request.json", "--output", "report.json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "MEASUREMENT_CONDITIONAL_NOT_QUALIFIED",
        "request_id": "pilot-1",
    }
