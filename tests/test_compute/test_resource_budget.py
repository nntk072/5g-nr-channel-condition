from __future__ import annotations

import hashlib
import json
import subprocess
import sys

import pytest
from compute.resource_budget import build_resource_budget_report, run_resource_budget


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
            "loslab",
            "resource-budget",
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
