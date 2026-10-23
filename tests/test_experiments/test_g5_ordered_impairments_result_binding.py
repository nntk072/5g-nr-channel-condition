"""Integrity checks for hash-bound ordered-impairment experiment results."""

import hashlib
import json
from pathlib import Path

import experiments.g5_ordered_impairments_receiver as experiment


def _minimal_probe_report() -> dict[str, object]:
    return {
        "schema": experiment.SCHEMA,
        "scope": "synthetic CPU fixture only",
        "config": {"payload_seeds": [1], "blocks_per_seed": 2},
        "acceptance": {
            "criteria_frozen_before_run": {"max_fixture_bler": 0.1},
            "observed": {"fixture_bler": 0.0},
            "pass": True,
        },
    }


def test_result_binds_source_protocol_and_observations(tmp_path: Path):
    result = experiment.build_hash_bound_result(
        _minimal_probe_report(), created_utc="2026-10-23T00:00:00+00:00"
    )

    assert experiment.verify_hash_bound_result(result)
    source = Path(experiment.__file__).resolve()
    assert result["provenance"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["provenance"]["source_matches_git_revision"] in (True, False, None)
    assert result["provenance"]["protocol"]["acceptance_criteria"] == {"max_fixture_bler": 0.1}
    assert result["interpretation"] == {
        "criterion_scope": "This verdict applies only to the declared deterministic synthetic CPU fixture.",
        "deployment_qualification": False,
        "g5_09_closed": False,
        "status": "conditional_fixture_pass",
    }

    destination = tmp_path / "g5-ordered-result.json"
    experiment.write_hash_bound_result(destination, result)
    stored = json.loads(destination.read_text(encoding="utf-8"))
    assert stored == result
    assert experiment.verify_hash_bound_result(stored)


def test_result_verifier_rejects_observation_or_protocol_tampering():
    result = experiment.build_hash_bound_result(
        _minimal_probe_report(), created_utc="2026-10-23T00:00:00+00:00"
    )
    result["experiment"]["acceptance"]["observed"]["fixture_bler"] = 0.5
    assert not experiment.verify_hash_bound_result(result)

    with __import__("pytest").raises(ValueError, match="invalid hash bindings"):
        experiment.write_hash_bound_result("unused.json", result)


def test_failed_fixture_keeps_conditional_interpretation():
    report = _minimal_probe_report()
    report["acceptance"]["pass"] = False
    result = experiment.build_hash_bound_result(report, created_utc="2026-10-23T00:00:00+00:00")

    assert result["interpretation"]["status"] == "conditional_fixture_fail"
    assert result["interpretation"]["deployment_qualification"] is False
    assert result["interpretation"]["g5_09_closed"] is False
