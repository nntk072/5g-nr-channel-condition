from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation.population_audit import (
    audit_expected_population,
    validate_expected_population_roster,
)


def _roster():
    return {
        "schema": "loslab.expected_population_roster.v1",
        "roster_id": "frozen-sampler-seed-19",
        "units": [
            {
                "unit_id": "sample-1",
                "family_id": "munich",
                "domain_id": "scene-a",
                "independent_group_id": "episode-1",
                "disposition": "included",
                "reason": None,
            },
            {
                "unit_id": "sample-2",
                "family_id": "munich",
                "domain_id": "scene-a",
                "independent_group_id": "episode-2",
                "disposition": "included",
                "reason": None,
            },
            {
                "unit_id": "sample-3",
                "family_id": "etoile",
                "domain_id": "scene-b",
                "independent_group_id": "episode-3",
                "disposition": "excluded",
                "reason": "outside_frozen_sampling_region",
            },
            {
                "unit_id": "sample-4",
                "family_id": "etoile",
                "domain_id": "scene-b",
                "independent_group_id": "episode-4",
                "disposition": "no_path",
                "reason": "solver_returned_empty_path_set",
            },
        ],
    }


def test_population_audit_separates_inclusion_from_no_path_and_exclusion():
    report = audit_expected_population(
        _roster(),
        observed_sample_ids=["sample-1", "sample-2"],
        observed_family_ids=["munich", "munich"],
        observed_domain_ids=["scene-a", "scene-a"],
        observed_group_ids=["episode-1", "episode-2"],
    )

    assert report["status"] == "ROSTER_MATCH"
    assert report["expected_unit_count"] == 4
    assert report["observed_row_count"] == 2
    assert report["expected_disposition_counts"] == {
        "empty": 0,
        "excluded": 1,
        "included": 2,
        "no_path": 1,
        "rejected": 0,
    }
    assert report["by_family"][0]["no_path_unit_count"] == 1
    assert len(report["roster_sha256"]) == 64
    assert len(report["observed_identity_roster_sha256"]) == 64
    schema = json.loads(
        Path("evaluation/schema/expected-population-audit.v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    import jsonschema

    jsonschema.Draft202012Validator(schema).validate(report)


def test_population_audit_reports_missing_duplicate_unrostered_and_identity_drift():
    report = audit_expected_population(
        _roster(),
        observed_sample_ids=["sample-1", "sample-1", "not-in-roster", "sample-3"],
        observed_family_ids=["munich", "munich", "other", "etoile"],
        observed_domain_ids=["scene-a", "scene-a", "scene-x", "scene-b"],
        observed_group_ids=["episode-1", "episode-1", "episode-x", "episode-3"],
    )

    assert report["status"] == "ROSTER_MISMATCH"
    assert report["included_missing_unit_ids"] == ["sample-2"]
    assert report["duplicate_observed_sample_ids"] == ["sample-1"]
    assert report["unrostered_observed_sample_ids"] == ["not-in-roster"]
    assert report["observed_for_nonincluded_unit_ids"] == ["sample-3"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda roster: roster["units"].append(dict(roster["units"][0])), "duplicate"),
        (lambda roster: roster["units"][2].update(reason=None), "require a canonical reason"),
        (lambda roster: roster["units"][0].update(extra=True), "field roster"),
    ],
)
def test_population_roster_rejects_duplicate_and_malformed_outcomes(mutate, message):
    roster = _roster()
    mutate(roster)
    with pytest.raises(ValueError, match=message):
        validate_expected_population_roster(roster)
