from __future__ import annotations

from artifacts.manifest import mark_complete, mark_failed, run_manifest
from artifacts.run_replay import build_run_replay_provenance


def test_manifest_transitions_to_complete_without_mutating_running_record() -> None:
    running = run_manifest("run-001", {"seed": 42}, input_count=3)
    complete = mark_complete(running)

    assert running["state"] == "running"
    assert "completed_utc" not in running
    assert complete["state"] == "complete"
    assert complete["completed_utc"]
    assert complete["config_identity"] == running["config_identity"]
    assert complete["input_count"] == 3


def _replay_fixture():
    return build_run_replay_provenance(
        created_utc="2026-11-10T03:00:00+00:00",
        command_argv=["loslab", "experiment"],
        command_source="cli_argv",
        source_revision="a" * 40,
        package_source_sha256="b" * 64,
        config_sha256="c" * 64,
        inputs=[{"path": "fixture.npz", "sha256": "d" * 64}],
        backend_identity={
            "requested_device": "cpu",
            "platform": "Linux-test",
            "dependencies": {"python": "3.12.0"},
        },
        rng_streams=[
            {
                "name": "run_seed_11",
                "library": "numpy",
                "algorithm": "default_rng_v1",
                "seed_material": [11],
                "derived_seed": 11,
                "purpose": "fixture run",
            }
        ],
    )


def test_manifest_completion_transitions_nested_replay_contract() -> None:
    running = run_manifest("run-replay-complete", {"seed": 11}, run_replay=_replay_fixture())

    complete = mark_complete(running)

    assert running["run_replay"]["state"] == "running"
    assert complete["run_replay"]["state"] == "complete"
    assert complete["run_replay"]["state_changed_utc"] == complete["completed_utc"]
    assert complete["run_replay"]["failure"] is None


def test_manifest_failure_transitions_nested_replay_contract() -> None:
    running = run_manifest("run-replay-failed", {"seed": 11}, run_replay=_replay_fixture())

    failed = mark_failed(
        running,
        error_type="RuntimeError",
        error_message="training failed",
        stage="experiment",
    )

    assert running["state"] == "running"
    assert failed["state"] == "failed"
    assert failed["run_replay"]["state"] == "failed"
    assert failed["run_replay"]["state_changed_utc"] == failed["failed_utc"]
    assert failed["run_replay"]["failure"] == {
        "error_type": "RuntimeError",
        "message": "training failed",
        "stage": "experiment",
    }
