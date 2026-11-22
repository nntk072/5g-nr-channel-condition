from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from agentops.contracts import ContractError
from agentops.resources import (
    ResourceUsageLedger,
    authorize_resource_class,
    validate_resource_authorization,
    validate_scheduler_argv,
)
from compute.resource_budget import build_resource_budget_report


def _budget_request() -> dict:
    runtime_environment = {
        "platform": "linux",
        "machine": "x86_64",
        "python_version": "3.12.16",
        "numpy_version": "2.2.6",
        "sionna_version": "2.2.0",
        "torch_version": "2.9.0",
        "solver_backend": "llvm",
        "accelerator": "V100",
        "accelerator_driver": "580.95",
        "accelerator_runtime": "cuda-12.8",
    }
    observations = [
        {
            "condition_id": f"condition-{index}",
            "scene_family_id": "family-a",
            "scene_id": f"scene-{index}",
            "link_id": f"link-{index}",
            "configuration_sha256": f"{index + 1:064x}",
            "measurement_record_sha256": f"{index + 11:064x}",
            "solver_runtime_s": float(index + 1),
            "receiver_runtime_s": float(index + 1),
            "preprocessing_runtime_s": 0.5,
            "downstream_runtime_s": 1.0,
            "gpu_peak_memory_bytes": 1024 * (index + 1),
            "raw_path_bytes": 100 * (index + 1),
            "cfr_bytes": 200 * (index + 1),
            "waveform_bytes": 300 * (index + 1),
            "manifest_bytes": 10,
            "other_artifact_bytes": 20,
        }
        for index in range(10)
    ]
    encoded_environment = json.dumps(
        runtime_environment,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": "loslab.resource_budget_request.v1",
        "request_id": "agentops-test-budget",
        "measurement_batch_id": "agentops-test-pilot",
        "environment_sha256": hashlib.sha256(encoded_environment).hexdigest(),
        "runtime_environment": runtime_environment,
        "headroom_factor": 2.0,
        "strata": [
            {
                "stratum_id": "family-a-los-siso",
                "family_id": "family-a",
                "planned_units": 10,
                "pilot_observations": observations,
            }
        ],
    }


def _budget_report(tmp_path: Path) -> tuple[Path, str]:
    request = _budget_request()
    report = build_resource_budget_report(request, request_sha256="d" * 64)
    path = tmp_path / "resource-budget-report.json"
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def _authorization(expiry: datetime, budget_digest: str) -> dict:
    return {
        "schema": "nr-agentops.resource-authorization.v1",
        "task_id": "TASK.ONE",
        "source_revision": "a" * 40,
        "task_card_sha256": "b" * 64,
        "resource_class": "native_smoke",
        "host": "halti",
        "selected_device": "cuda:0",
        "budget_report_sha256": budget_digest,
        "max_gpu_hours": 0.1,
        "ram_ceiling_bytes": 1024,
        "disk_ceiling_bytes": 2048,
        "allowed_scheduler_argv": [],
        "expires_at_utc": expiry.isoformat(),
        "approved_by": "human",
        "scientific_gate_authorized": False,
    }


def _inventory() -> dict:
    return {
        "host": "halti",
        "ram_bytes": 8192,
        "disk_free_bytes": 16384,
        "packages": {"sionna-no-rt": "2.2.0", "sionna-rt": "not installed"},
        "gpu_devices": [{"index": 0, "name": "V100", "memory_mib": 16384}],
    }


def _validate(auth: dict, report: Path, now: datetime) -> str:
    return validate_resource_authorization(
        auth,
        task_id="TASK.ONE",
        source_revision="a" * 40,
        task_card_sha256="b" * 64,
        resource_class=auth["resource_class"],
        host="halti",
        budget_report_path=report,
        now=now,
        inventory=_inventory(),
    )


def test_cpu_class_needs_no_native_authorization(tmp_path: Path) -> None:
    assert authorize_resource_class("cpu_test") is None
    with pytest.raises(ContractError, match="requires an exact"):
        authorize_resource_class("native_smoke")
    with pytest.raises(ContractError, match="do not accept"):
        authorize_resource_class("cpu_test", budget_report_path=tmp_path / "unused.json")


def test_resource_authorization_binds_host_expiry_and_budget_report(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    report, digest = _budget_report(tmp_path)
    auth = _authorization(now + timedelta(hours=1), digest)
    assert len(_validate(auth, report, now)) == 64

    with pytest.raises(ContractError, match="expired"):
        _validate(_authorization(now - timedelta(seconds=1), digest), report, now)

    wrong_digest = dict(auth)
    wrong_digest["budget_report_sha256"] = "e" * 64
    with pytest.raises(ContractError, match="budget-report digest"):
        _validate(wrong_digest, report, now)

    with pytest.raises(ContractError, match="measured budget report"):
        validate_resource_authorization(
            auth,
            task_id="TASK.ONE",
            source_revision="a" * 40,
            task_card_sha256="b" * 64,
            resource_class="native_smoke",
            host="halti",
            now=now,
            inventory=_inventory(),
        )


def test_resource_authorization_rejects_invalid_or_symlinked_budget_report(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"schema":"not-a-budget"}', encoding="utf-8")
    digest = hashlib.sha256(invalid.read_bytes()).hexdigest()
    auth = _authorization(now + timedelta(hours=1), digest)
    with pytest.raises(ContractError, match="violates schema"):
        _validate(auth, invalid, now)

    report, digest = _budget_report(tmp_path)
    link = tmp_path / "budget-link.json"
    link.symlink_to(report)
    auth = _authorization(now + timedelta(hours=1), digest)
    with pytest.raises(ContractError, match="single-link regular file"):
        _validate(auth, link, now)


def test_resource_authorization_rejects_oversubscription_and_unavailable_runtime(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    report, digest = _budget_report(tmp_path)
    auth = _authorization(now + timedelta(hours=1), digest)
    too_large = dict(auth)
    too_large["ram_ceiling_bytes"] = 16384
    with pytest.raises(ContractError, match="RAM capacity"):
        _validate(too_large, report, now)

    missing_runtime = _inventory()
    missing_runtime["packages"] = {"sionna-no-rt": "not installed", "sionna-rt": "not installed"}
    with pytest.raises(ContractError, match="Sionna 2.2"):
        validate_resource_authorization(
            auth,
            task_id="TASK.ONE",
            source_revision="a" * 40,
            task_card_sha256="b" * 64,
            resource_class="native_smoke",
            host="halti",
            budget_report_path=report,
            now=now,
            inventory=missing_runtime,
        )


def test_resource_authorization_rejects_unavailable_cuda_device(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    report, digest = _budget_report(tmp_path)
    auth = _authorization(now + timedelta(hours=1), digest)
    auth["selected_device"] = "cuda:2"
    with pytest.raises(ContractError, match="CUDA device is unavailable"):
        _validate(auth, report, now)


def test_resource_authorization_rejects_noncanonical_device_and_ungated_scientific_native(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    report, digest = _budget_report(tmp_path)
    auth = _authorization(now + timedelta(hours=1), digest)

    invalid_device = dict(auth)
    invalid_device["selected_device"] = "gpu0"
    with pytest.raises(ContractError, match="violates schema"):
        _validate(invalid_device, report, now)

    scientific = dict(auth)
    scientific["resource_class"] = "scientific_native"
    for flag in (False, True):
        # An AgentOps boolean never admits scientific_native, in either state.
        scientific["scientific_gate_authorized"] = flag
        with pytest.raises(ContractError, match="scientific_native remains denied"):
            validate_resource_authorization(
                scientific,
                task_id="TASK.ONE",
                source_revision="a" * 40,
                task_card_sha256="b" * 64,
                resource_class="scientific_native",
                host="halti",
                budget_report_path=report,
                now=now,
                inventory=_inventory(),
            )

    cpu_device = dict(auth)
    cpu_device["selected_device"] = "cpu"
    assert len(_validate(cpu_device, report, now)) == 64


def test_resource_authorization_rejects_wrong_host_and_unqualified_production(
    tmp_path: Path,
) -> None:
    now = datetime.now(timezone.utc)
    report, digest = _budget_report(tmp_path)
    auth = _authorization(now + timedelta(hours=1), digest)
    wrong_host_inventory = _inventory()
    wrong_host_inventory["host"] = "other-host"
    with pytest.raises(ContractError, match="inventory does not match"):
        validate_resource_authorization(
            auth,
            task_id="TASK.ONE",
            source_revision="a" * 40,
            task_card_sha256="b" * 64,
            resource_class="native_smoke",
            host="halti",
            budget_report_path=report,
            now=now,
            inventory=wrong_host_inventory,
        )

    production = dict(auth)
    production["resource_class"] = "production_d1"
    for gate_flag in (False, True):
        production["scientific_gate_authorized"] = gate_flag
        with pytest.raises(ContractError, match="existing D1 production gate is disabled"):
            validate_resource_authorization(
                production,
                task_id="TASK.ONE",
                source_revision="a" * 40,
                task_card_sha256="b" * 64,
                resource_class="production_d1",
                host="halti",
                budget_report_path=report,
                now=now,
                inventory=_inventory(),
            )


def test_scheduler_argv_must_match_authorized_argv_exactly(tmp_path: Path) -> None:
    _, digest = _budget_report(tmp_path)
    auth = _authorization(datetime.now(timezone.utc) + timedelta(hours=1), digest)
    auth["allowed_scheduler_argv"] = ["sbatch", "--partition=gpu", "run.sh"]

    assert validate_scheduler_argv(auth, ["sbatch", "--partition=gpu", "run.sh"]) == (
        "sbatch",
        "--partition=gpu",
        "run.sh",
    )
    with pytest.raises(ContractError, match="exactly match"):
        validate_scheduler_argv(auth, ["sbatch", "run.sh"])
    with pytest.raises(ContractError, match="nonempty"):
        validate_scheduler_argv(auth, [])


def test_resource_usage_ledger_enforces_cumulative_gpu_hour_ceiling(tmp_path: Path) -> None:
    ledger = ResourceUsageLedger(tmp_path / "usage")
    digest = "f" * 64
    ledger.reserve(digest, 1.0, reservation_id="first", requested_gpu_hours=0.6)
    assert ledger.remaining_gpu_hours(digest, 1.0) == pytest.approx(0.4)

    with pytest.raises(ContractError, match="exceed"):
        ledger.reserve(digest, 1.0, reservation_id="too-much", requested_gpu_hours=0.5)

    ledger.settle(
        digest,
        1.0,
        reservation_id="first",
        actual_gpu_hours=0.25,
        outcome="completed",
    )
    assert ledger.remaining_gpu_hours(digest, 1.0) == pytest.approx(0.75)

    ledger.reserve(digest, 1.0, reservation_id="second", requested_gpu_hours=0.75)
    ledger.settle(
        digest,
        1.0,
        reservation_id="second",
        actual_gpu_hours=0.5,
        outcome="failed",
    )
    assert ledger.remaining_gpu_hours(digest, 1.0) == pytest.approx(0.25)


def test_pending_gpu_reservation_survives_restart_and_prevents_duplicate_run(
    tmp_path: Path,
) -> None:
    root = tmp_path / "usage"
    first = ResourceUsageLedger(root)
    digest = "1" * 64
    first.reserve(digest, 0.2, reservation_id="live-run", requested_gpu_hours=0.2)

    restarted = ResourceUsageLedger(root)
    assert restarted.remaining_gpu_hours(digest, 0.2) == 0.0
    with pytest.raises(ContractError, match="ceiling"):
        restarted.reserve(digest, 0.2, reservation_id="duplicate", requested_gpu_hours=0.01)


def test_resource_usage_ledger_rejects_tampered_numeric_entries(tmp_path: Path) -> None:
    root = tmp_path / "usage"
    digest = "2" * 64
    ledger = ResourceUsageLedger(root)
    ledger.reserve(digest, 1.0, reservation_id="run", requested_gpu_hours=0.25)
    state_path = root / f"{digest}.json"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["entries"][0]["requested_gpu_hours"] = "nan"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ContractError, match="numeric GPU-hour"):
        ledger.remaining_gpu_hours(digest, 1.0)

    state["entries"][0]["requested_gpu_hours"] = 0.25
    state["entries"][0]["status"] = "failed"
    state["entries"][0]["actual_gpu_hours"] = -0.1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ContractError, match="nonnegative GPU-hour"):
        ledger.remaining_gpu_hours(digest, 1.0)


def test_resource_usage_ledger_stale_lock_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "usage"
    root.mkdir()
    digest = "3" * 64
    lock_path = root / f"{digest}.lock"
    lock_path.write_text("999999", encoding="ascii")

    ledger = ResourceUsageLedger(root)
    with pytest.raises(ContractError, match="ledger is locked"):
        ledger.reserve(digest, 1.0, reservation_id="blocked", requested_gpu_hours=0.1)
    assert lock_path.exists()
