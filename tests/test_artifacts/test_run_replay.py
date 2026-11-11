from __future__ import annotations

import copy

import pytest
from artifacts.run_replay import (
    build_run_replay_provenance,
    transition_run_replay,
    validate_run_replay_provenance,
)


def _running(**overrides):
    kwargs = {
        "created_utc": "2026-11-10T03:00:00+00:00",
        "command_argv": ["loslab", "experiment", "--input", "fixture.npz"],
        "command_source": "cli_argv",
        "source_revision": "a" * 40,
        "package_source_sha256": "b" * 64,
        "config_sha256": "c" * 64,
        "inputs": [{"path": "fixture.npz", "sha256": "d" * 64}],
        "backend_identity": {
            "requested_device": "cpu",
            "platform": "Linux-test",
            "dependencies": {"numpy": "2.2.0", "python": "3.12.0"},
        },
        "rng_streams": [
            {
                "name": "run_seed_11",
                "library": "numpy",
                "algorithm": "default_rng_v1",
                "seed_material": [11],
                "derived_seed": 11,
                "purpose": "fixture run",
            }
        ],
    }
    kwargs.update(overrides)
    return build_run_replay_provenance(**kwargs)


def test_run_replay_contract_round_trips_and_binds_replay_identity() -> None:
    replay = _running()

    assert replay["schema"] == "loslab.run_replay_provenance.v1"
    assert replay["qualification_status"] == "NOT_QUALIFIED"
    assert replay["state"] == "running"
    assert replay["command"]["argv"] == ["loslab", "experiment", "--input", "fixture.npz"]
    assert replay["source"]["git_commit"] == "a" * 40
    assert replay["source"]["package_source_sha256"] == "b" * 64
    assert replay["config_sha256"] == "c" * 64
    assert replay["inputs"] == [{"path": "fixture.npz", "sha256": "d" * 64}]
    assert replay["backend_identity_sha256"]
    assert replay["rng_streams"][0]["name"] == "run_seed_11"
    assert replay["attempt"] == {"index": 0, "retry_of_manifest_sha256": None}
    assert validate_run_replay_provenance(replay) == replay


def test_run_replay_terminal_transitions_are_digest_bound() -> None:
    running = _running()

    complete = transition_run_replay(
        running, state="complete", changed_utc="2026-11-10T03:01:00+00:00"
    )
    failed = transition_run_replay(
        running,
        state="failed",
        changed_utc="2026-11-10T03:02:00+00:00",
        failure={"error_type": "RuntimeError", "message": "fixture failure", "stage": "train"},
    )

    assert running["state"] == "running"
    assert complete["state"] == "complete"
    assert complete["failure"] is None
    assert failed["state"] == "failed"
    assert failed["failure"] == {
        "error_type": "RuntimeError",
        "message": "fixture failure",
        "stage": "train",
    }
    assert (
        len({running["contract_sha256"], complete["contract_sha256"], failed["contract_sha256"]})
        == 3
    )
    with pytest.raises(ValueError, match="only a running"):
        transition_run_replay(complete, state="failed", failure=failed["failure"])


def test_run_replay_rejects_tampered_contract_and_backend_digest() -> None:
    replay = _running()
    tampered = copy.deepcopy(replay)
    tampered["config_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="contract digest mismatch"):
        validate_run_replay_provenance(tampered)

    tampered = copy.deepcopy(replay)
    tampered["backend_identity"]["requested_device"] = "cuda"
    with pytest.raises(ValueError, match="backend identity digest mismatch"):
        validate_run_replay_provenance(tampered)


def test_run_replay_retry_contract_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="attempt zero"):
        _running(retry_of_manifest_sha256="e" * 64)
    with pytest.raises(ValueError, match="retry attempts"):
        _running(attempt_index=1)

    retry = _running(attempt_index=1, retry_of_manifest_sha256="e" * 64)
    assert retry["attempt"] == {"index": 1, "retry_of_manifest_sha256": "e" * 64}


def test_run_replay_rejects_duplicate_rng_stream_names() -> None:
    stream = {
        "name": "duplicate",
        "library": "numpy",
        "algorithm": "default_rng_v1",
        "seed_material": [1],
        "derived_seed": 1,
        "purpose": "fixture",
    }
    with pytest.raises(ValueError, match="duplicate RNG stream name"):
        _running(rng_streams=[stream, stream])
