from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from jsonschema import ValidationError
from scripts.run_g4_cdl_directional_pattern_power import (
    CLUSTER_COUNTS,
    EXPECTED_CASES,
    TABLE_HASHES,
    audit_result,
    expected_case_configuration,
)


def _valid_report() -> dict[str, Any]:
    cases = []
    for cohort, profile, seed in sorted(EXPECTED_CASES):
        cluster_count = CLUSTER_COUNTS[profile]
        patterns, orientations, family_claim_count = expected_case_configuration(
            cohort, profile, seed
        )
        claims = [
            {
                "cluster_index": index,
                "reference": 1.0,
                "estimate": 1.0,
                "half_width": 0.001,
                "interval": [0.999, 1.001],
                "passed": True,
            }
            for index in range(cluster_count)
        ]
        cases.append(
            {
                "cohort": cohort,
                "profile": profile,
                "seed": seed,
                "patterns": patterns,
                "orientations_rad": orientations,
                "parameter_table_sha256": TABLE_HASHES[profile],
                "sample_count": 16384,
                "familywise_alpha": 0.01,
                "family_claim_count": family_claim_count,
                "total_expected_power": float(cluster_count),
                "maximum_half_width": 0.001,
                "maximum_allowed_half_width": 0.05 * cluster_count,
                "passed": True,
                "claims": claims,
                "test_nodeid": f"test::{cohort}::{profile}::{seed}",
                "test_outcome": "passed",
            }
        )
    return {
        "schema": "loslab.g4_cdl_directional_pattern_power_result.v1",
        "protocol": "G4_07_CDL_DIRECTIONAL_PATTERN_VECTOR_FIELD_PROTOCOL_2026-11-01",
        "run_date": "2026-11-01",
        "status": "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE",
        "seed_collision_audit": {
            "path": "G4_CDL_ANTENNA_PATTERN_VECTOR_FIELD_SEED_AUDIT_2026-11-01.json",
            "sha256": "a" * 64,
            "status": "CLEAR_WITHIN_SCANNED_JSON_CORPUS",
            "scanned_file_count": 2276,
        },
        "design": {
            "profiles": list("ABCDE"),
            "patterns": ["38.901", "38.901-handheld"],
            "samples_per_case": 16384,
            "one_end_seed_range": [2700000, 2700029],
            "both_ends_seed_range": [2701000, 2701039],
            "familywise_alpha": 0.01,
            "maximum_half_width_fraction_of_total_power": 0.05,
            "case_count": 70,
            "cluster_claim_count": 1358,
        },
        "runtime": {"sionna": "2.2.0", "torch": "2.14.1", "torch_cuda": "12.6", "device": "cpu"},
        "command": "python -m scripts.run_g4_cdl_directional_pattern_power",
        "provenance": {
            "git_revision": "a" * 40,
            "clean_worktree_at_run_start": True,
            "runner_source_sha256": "a" * 64,
            "test_source_sha256": "a" * 64,
            "protocol_sha256": "a" * 64,
            "schema_sha256": "a" * 64,
            "reference_sha256": {
                "G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json": "a" * 64,
                "G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json": "a" * 64,
            },
        },
        "outcome": {
            "pytest_exit_code": 0,
            "cases_observed": 70,
            "cases_passed": 70,
            "claims_passed": 1358,
            "claims_failed": 0,
        },
        "cases": cases,
        "scope": "synthetic audit fixture only",
    }


def test_audit_recomputes_complete_frozen_case_and_claim_roster() -> None:
    report = _valid_report()

    summary = audit_result(report)

    assert summary["case_count"] == 70
    assert summary["claim_count"] == 1358
    assert summary["failed_claim_count"] == 0


@pytest.mark.parametrize("mutation", ["interval", "seed", "pattern", "claim_status"])
def test_audit_rejects_tampered_claim_or_case_roster(mutation: str) -> None:
    report = deepcopy(_valid_report())
    if mutation == "interval":
        report["cases"][0]["claims"][0]["interval"][0] = 0.5
    elif mutation == "seed":
        report["cases"][0]["seed"] += 9000
    elif mutation == "pattern":
        report["cases"][0]["patterns"][0] = "38.901-handheld"
    else:
        report["cases"][0]["claims"][0]["passed"] = False

    with pytest.raises((ValueError, ValidationError)):
        audit_result(report)
