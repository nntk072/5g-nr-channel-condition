from __future__ import annotations

import copy
import hashlib

import pytest
from artifacts.run_replay import (
    _canonical_text,
    _digest,
    _normalise_backend,
    _normalise_inputs,
    _normalise_rng_streams,
    _seed,
    _source_identity,
    _utc,
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


@pytest.mark.parametrize(
    ("value", "label", "message"),
    [
        (None, "digest", "lowercase SHA-256"),
        ("A" * 64, "digest", "lowercase SHA-256"),
        ("f" * 63, "digest", "lowercase SHA-256"),
        (None, "name", "canonical nonempty string"),
        (" padded ", "name", "canonical nonempty string"),
        ("", "time", "ISO-8601 UTC timestamp"),
        ("not-a-time", "time", "ISO-8601 UTC timestamp"),
        ("2026-01-01T00:00:00", "time", "ISO-8601 UTC timestamp"),
        ("2026-01-01T00:00:00+01:00", "time", "ISO-8601 UTC timestamp"),
    ],
)
def test_run_replay_scalar_contract_helpers_reject_invalid_values(value, label, message):
    validator = _digest if label == "digest" else _canonical_text if label == "name" else _utc
    with pytest.raises(ValueError, match=message):
        validator(value, label)


@pytest.mark.parametrize("value", [True, 1.0, -1, 2**63])
def test_run_replay_seed_rejects_noncanonical_integer_values(value):
    with pytest.raises(ValueError, match="integer in"):
        _seed(value, "seed")


def test_run_replay_input_and_backend_normalizers_enforce_exact_contracts():
    with pytest.raises(ValueError, match="at least one artifact"):
        _normalise_inputs([])
    with pytest.raises(ValueError, match="exactly path and sha256"):
        _normalise_inputs([{"path": "input.bin"}])
    with pytest.raises(ValueError, match="canonical nonempty string"):
        _normalise_inputs([{"path": " ../input.bin", "sha256": "a" * 64}])

    with pytest.raises(ValueError, match="invalid field roster"):
        _normalise_backend({"requested_device": "cpu"})
    with pytest.raises(ValueError, match="nonempty mapping"):
        _normalise_backend({"requested_device": "cpu", "platform": "Linux", "dependencies": {}})
    with pytest.raises(ValueError, match="dependency name must be a canonical"):
        _normalise_backend(
            {
                "requested_device": "cpu",
                "platform": "Linux",
                "dependencies": {" numpy": "2"},
            }
        )


def test_run_replay_rng_and_source_identity_normalizers_reject_bad_rosters():
    with pytest.raises(ValueError, match="at least one stream"):
        _normalise_rng_streams([])
    with pytest.raises(ValueError, match="invalid field roster"):
        _normalise_rng_streams([{"name": "seed"}])
    stream = {
        "name": "seed",
        "library": "numpy",
        "algorithm": "default_rng_v1",
        "seed_material": [],
        "derived_seed": None,
        "purpose": "channel generation",
    }
    with pytest.raises(ValueError, match="seed_material must be a nonempty array"):
        _normalise_rng_streams([stream])
    with pytest.raises(ValueError, match="lowercase Git object ID"):
        _source_identity("not-a-git-id", "a" * 64)
    assert _source_identity("unavailable", "a" * 64)["git_commit_status"] == "unavailable"


def test_run_replay_builder_rejects_invalid_argv_attempts_and_retry_bindings():
    with pytest.raises(ValueError, match="argv sequence"):
        _running(command_argv="loslab")
    with pytest.raises(ValueError, match="only strings and may not be empty"):
        _running(command_argv=[])
    with pytest.raises(ValueError, match="only strings and may not be empty"):
        _running(command_argv=["loslab", 3])
    with pytest.raises(ValueError, match="non-negative integer"):
        _running(attempt_index=True)
    with pytest.raises(ValueError, match="attempt zero"):
        _running(retry_of_manifest_sha256="e" * 64)
    with pytest.raises(ValueError, match="retry attempts"):
        _running(attempt_index=2)


def _rehash(payload):
    from artifacts.run_replay import _canonical_json

    core = {key: value for key, value in payload.items() if key != "contract_sha256"}
    payload["contract_sha256"] = hashlib.sha256(_canonical_json(core)).hexdigest()
    return payload


def test_run_replay_validator_rejects_cross_field_inconsistencies():
    invalid_cases = []
    payload = copy.deepcopy(_running())
    payload["attempt"]["retry_of_manifest_sha256"] = "e" * 64
    invalid_cases.append((payload, "attempt zero may not declare"))
    payload = copy.deepcopy(_running(attempt_index=1, retry_of_manifest_sha256="e" * 64))
    payload["attempt"]["retry_of_manifest_sha256"] = None
    invalid_cases.append((payload, "retry attempts must bind"))
    payload = copy.deepcopy(_running())
    payload["source"]["git_commit_status"] = "unavailable"
    invalid_cases.append((payload, "requires git_commit=unavailable"))
    payload = copy.deepcopy(_running())
    payload["state"] = "failed"
    payload["failure"] = None
    invalid_cases.append((payload, "failure details must be present exactly"))
    payload = copy.deepcopy(_running())
    payload["state"] = "complete"
    payload["failure"] = {"error_type": "RuntimeError", "message": "x", "stage": "train"}
    invalid_cases.append((payload, "failure details must be present exactly"))

    for payload, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            validate_run_replay_provenance(_rehash(payload))


def test_run_replay_validator_accepts_explicitly_unavailable_source_identity():
    replay = _running(source_revision="unavailable")
    assert replay["source"]["git_commit"] == "unavailable"
    assert validate_run_replay_provenance(replay) == replay


def test_run_replay_rng_stream_allows_unresolved_derived_seed():
    stream = {
        "name": "platform-default",
        "library": "numpy",
        "algorithm": "default_rng_v1",
        "seed_material": [7],
        "derived_seed": None,
        "purpose": "external generator seed",
    }
    replay = _running(rng_streams=[stream])
    assert replay["rng_streams"][0]["derived_seed"] is None


def test_run_replay_validator_rejects_unsorted_rng_streams():
    stream_a = {
        "name": "a",
        "library": "numpy",
        "algorithm": "default_rng_v1",
        "seed_material": [1],
        "derived_seed": 1,
        "purpose": "first stream",
    }
    stream_b = {**stream_a, "name": "b", "seed_material": [2], "derived_seed": 2}
    payload = copy.deepcopy(_running(rng_streams=[stream_a, stream_b]))
    payload["rng_streams"] = list(reversed(payload["rng_streams"]))

    with pytest.raises(ValueError, match="sorted and unique"):
        validate_run_replay_provenance(_rehash(payload))


def test_run_replay_transition_rejects_invalid_terminal_failure_and_time_fields():
    running = _running()
    with pytest.raises(ValueError, match="terminal state"):
        transition_run_replay(running, state="running")
    with pytest.raises(ValueError, match="requires error_type, message, and stage"):
        transition_run_replay(running, state="failed", failure={"message": "missing fields"})
    with pytest.raises(ValueError, match="failure.message must be a string"):
        transition_run_replay(
            running,
            state="failed",
            failure={"error_type": "RuntimeError", "message": 4, "stage": "train"},
        )
    with pytest.raises(ValueError, match="may not carry failure details"):
        transition_run_replay(running, state="complete", failure={})
    with pytest.raises(ValueError, match="ISO-8601 UTC timestamp"):
        transition_run_replay(running, state="complete", changed_utc="not-a-time")

    assert transition_run_replay(running, state="complete")["state"] == "complete"
