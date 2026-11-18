from __future__ import annotations

import hashlib
import json

import pytest
from artifacts.manifest import run_manifest
from artifacts.run_replay import build_run_replay_provenance
from scripts.audit_interrupted_runs import audit_root, mark_interrupted_failed


def _running_manifest() -> dict:
    replay = build_run_replay_provenance(
        created_utc="2026-11-12T10:00:00+00:00",
        command_argv=["loslab", "experiment", "--seed", "7"],
        command_source="cli_argv",
        source_revision="a" * 40,
        package_source_sha256="b" * 64,
        config_sha256="c" * 64,
        inputs=[{"path": "input.json", "sha256": "d" * 64}],
        backend_identity={
            "requested_device": "cpu",
            "platform": "Linux-test",
            "dependencies": {"python": "3.12.0"},
        },
        rng_streams=[
            {
                "name": "numpy_seed_7",
                "library": "numpy",
                "algorithm": "default_rng",
                "seed_material": [7],
                "derived_seed": 7,
                "purpose": "test",
            }
        ],
    )
    return run_manifest("run-interrupted", {"seed": 7}, run_replay=replay)


def _running_capacity_manifest() -> dict:
    config = {"seed": 7}
    replay = build_run_replay_provenance(
        created_utc="2026-11-12T10:00:00+00:00",
        command_argv=["loslab", "capacity-sweep"],
        command_source="cli_argv",
        source_revision="a" * 40,
        package_source_sha256="b" * 64,
        config_sha256=hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        inputs=[{"path": "input.npz", "sha256": "d" * 64}],
        backend_identity={
            "requested_device": "cpu",
            "platform": "Linux-test",
            "dependencies": {"python": "3.12.0"},
        },
        rng_streams=[
            {
                "name": "capacity_seed_7",
                "library": "numpy",
                "algorithm": "default_rng",
                "seed_material": [7],
                "derived_seed": 7,
                "purpose": "test",
            }
        ],
    )
    return {
        "schema": "loslab.capacity_sweep.v2",
        "state": "running",
        "created_utc": "2026-11-12T10:00:00+00:00",
        "config": config,
        "config_sha256": replay["config_sha256"],
        "inputs": replay["inputs"],
        "run_replay": replay,
    }


def _running_observable_staging_receipt() -> dict:
    config = {"seed": 7}
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    replay = build_run_replay_provenance(
        created_utc="2026-11-12T10:00:00+00:00",
        command_argv=["loslab", "observable-multipath"],
        command_source="cli_argv",
        source_revision="a" * 40,
        package_source_sha256="b" * 64,
        config_sha256=config_sha256,
        inputs=[{"path": "generated:grouped_synthetic_two_path_dataset", "sha256": "d" * 64}],
        backend_identity={
            "requested_device": "cpu",
            "platform": "Linux-test",
            "dependencies": {"python": "3.12.0"},
        },
        rng_streams=[
            {
                "name": "dataset_and_training_seed",
                "library": "numpy/torch",
                "algorithm": "observable_two_path_seed_v1",
                "seed_material": [7],
                "derived_seed": 7,
                "purpose": "test",
            }
        ],
    )
    return {
        "schema": "loslab.observable_multipath_staging.v1",
        "state": "running",
        "created_utc": "2026-11-12T10:00:00+00:00",
        "output_directory_name": "pilot",
        "run_config": config,
        "run_config_sha256": config_sha256,
        "qualification_status": "NOT_QUALIFIED",
        "run_replay": replay,
    }


def _running_observable_preparation_receipt() -> dict:
    config = {"seed": 7}
    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "loslab.observable_multipath_staging.v2",
        "state": "running",
        "phase": "dataset_generation",
        "created_utc": "2026-11-12T10:00:00+00:00",
        "output_directory_name": "pilot",
        "run_config": config,
        "run_config_sha256": config_sha256,
        "qualification_status": "NOT_QUALIFIED",
        "source_identity": {"revision": "a" * 40, "package_source_sha256": "b" * 64},
        "runtime_identity": {
            "platform": "Linux-test",
            "dependencies": {"python": "3.12.0"},
            "requested_device": "cpu",
        },
        "command_argv": ["loslab", "observable-multipath"],
        "rng_streams": [
            {
                "name": "dataset_and_training_seed",
                "library": "numpy/torch",
                "algorithm": "observable_two_path_seed_v1",
                "seed_material": [7],
                "derived_seed": 7,
                "purpose": "test",
            }
        ],
    }


def test_audit_inventories_running_receipt_and_preserves_staging(tmp_path) -> None:
    output = tmp_path / "outputs" / "run-1"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(_running_manifest()), encoding="utf-8")
    staging = tmp_path / "outputs" / ".run-2.staging-deadbeef"
    staging.mkdir()
    (staging / "partial.bin").write_bytes(b"retain me")

    report = audit_root(tmp_path / "outputs")

    assert report["manifests"][0]["state"] == "running"
    assert report["manifests"][0]["recovery_supported"] is True
    assert report["staging_directories"] == [
        {"path": ".run-2.staging-deadbeef", "files": 1, "bytes": 9}
    ]
    assert (staging / "partial.bin").read_bytes() == b"retain me"


def test_audit_marks_capacity_sweep_recovery_support_only_after_schema_validation(tmp_path) -> None:
    output = tmp_path / "outputs" / "sweep"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(
        json.dumps(_running_capacity_manifest()), encoding="utf-8"
    )

    report = audit_root(tmp_path / "outputs")

    assert report["manifests"][0]["schema"] == "loslab.capacity_sweep.v2"
    assert report["manifests"][0]["state"] == "running"
    assert report["manifests"][0]["replay_state"] == "running"
    assert report["manifests"][0]["recovery_supported"] is True


def test_mark_interrupted_failed_requires_current_digest_and_preserves_outputs(tmp_path) -> None:
    output = tmp_path / "run-1"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(_running_manifest()), encoding="utf-8")
    before = manifest.read_bytes()
    digest = hashlib.sha256(before).hexdigest()
    artifact = output / "partial-output.bin"
    artifact.write_bytes(b"do not delete")

    result = mark_interrupted_failed(
        tmp_path,
        "run-1/manifest.json",
        expected_sha256=digest,
        operator_attestation="I verified this process is no longer active.",
    )

    recovered = json.loads(manifest.read_text(encoding="utf-8"))
    assert result["state"] == recovered["state"] == "failed"
    assert recovered["failure"]["error_type"] == "InterruptedRunRecovery"
    assert recovered["run_replay"]["state"] == "failed"
    assert recovered["run_replay"]["qualification_status"] == "NOT_QUALIFIED"
    assert artifact.read_bytes() == b"do not delete"


def test_capacity_sweep_recovery_validates_schema_and_transitions_both_states(tmp_path) -> None:
    output = tmp_path / "sweep"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(_running_capacity_manifest()), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    result = mark_interrupted_failed(
        tmp_path,
        "sweep/manifest.json",
        expected_sha256=digest,
        operator_attestation="I verified that the sweep process stopped.",
    )

    failed = json.loads(manifest.read_text(encoding="utf-8"))
    assert result["state"] == failed["state"] == failed["run_replay"]["state"] == "failed"
    assert failed["error"] == "I verified that the sweep process stopped."
    assert failed["failure"]["stage"] == "operator_recovery"


def test_observable_staging_receipt_recovery_keeps_hidden_partial_artifacts(tmp_path) -> None:
    staging = tmp_path / ".pilot.staging-interrupted"
    staging.mkdir()
    manifest = staging / "run_manifest.json"
    manifest.write_text(json.dumps(_running_observable_staging_receipt()), encoding="utf-8")
    partial = staging / "partial.bin"
    partial.write_bytes(b"preserve")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    report = audit_root(tmp_path)
    assert report["manifests"][0]["recovery_supported"] is True
    assert report["staging_directories"][0]["path"] == staging.name

    result = mark_interrupted_failed(
        tmp_path,
        f"{staging.name}/run_manifest.json",
        expected_sha256=digest,
        operator_attestation="I verified that the observable-multipath process stopped.",
    )

    failed = json.loads(manifest.read_text(encoding="utf-8"))
    assert result["state"] == failed["state"] == failed["run_replay"]["state"] == "failed"
    assert failed["qualification_status"] == "NOT_QUALIFIED"
    assert partial.read_bytes() == b"preserve"


def test_observable_preparation_recovery_does_not_invent_dataset_replay(tmp_path) -> None:
    staging = tmp_path / ".pilot.staging-generation"
    staging.mkdir()
    manifest = staging / "run_manifest.json"
    manifest.write_text(json.dumps(_running_observable_preparation_receipt()), encoding="utf-8")
    partial = staging / "partial-generation.bin"
    partial.write_bytes(b"retain")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    report = audit_root(tmp_path)
    entry = report["manifests"][0]
    assert entry["schema"] == "loslab.observable_multipath_staging.v2"
    assert entry["replay_state"] is None
    assert entry["recovery_supported"] is True

    mark_interrupted_failed(
        tmp_path,
        f"{staging.name}/run_manifest.json",
        expected_sha256=digest,
        operator_attestation="I verified that dataset generation stopped.",
    )

    failed = json.loads(manifest.read_text(encoding="utf-8"))
    assert failed["state"] == "failed"
    assert failed["phase"] == "dataset_generation"
    assert "run_replay" not in failed
    assert partial.read_bytes() == b"retain"


def test_observable_staging_audit_rejects_config_drift(tmp_path) -> None:
    staging = tmp_path / ".pilot.staging-drift"
    staging.mkdir()
    receipt = _running_observable_staging_receipt()
    receipt["run_config"]["seed"] = 8
    (staging / "run_manifest.json").write_text(json.dumps(receipt), encoding="utf-8")

    result = audit_root(tmp_path)["manifests"][0]

    assert result["recovery_supported"] is False
    assert "config digest mismatch" in result["error"]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"expected_sha256": "0" * 64, "operator_attestation": "process ended"}, "SHA-256 changed"),
        ({"expected_sha256": "", "operator_attestation": "process ended"}, "SHA-256 changed"),
        ({"expected_sha256": "f" * 64, "operator_attestation": " "}, "attestation"),
    ],
)
def test_recovery_fails_closed_without_digest_and_operator_attestation(
    tmp_path, kwargs, message
) -> None:
    output = tmp_path / "run-1"
    output.mkdir()
    (output / "manifest.json").write_text(json.dumps(_running_manifest()), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        mark_interrupted_failed(tmp_path, "run-1/manifest.json", **kwargs)


def test_recovery_does_not_terminalize_non_common_or_terminal_manifest(tmp_path) -> None:
    output = tmp_path / "run-1"
    output.mkdir()
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"schema": "other.v1", "state": "running"}), encoding="utf-8")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="supported, schema-valid running receipt"):
        mark_interrupted_failed(
            tmp_path,
            "run-1/manifest.json",
            expected_sha256=digest,
            operator_attestation="process ended",
        )
