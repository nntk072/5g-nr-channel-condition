"""Expected population roster: frozen field, disposition and observed-alignment rules."""

from __future__ import annotations

import copy
from typing import Any

import pytest
from evaluation.audits.population_audit import (
    audit_expected_population,
    validate_expected_population_roster,
)
from tests.test_evaluation.test_population_audit import _roster


def _with_unit(**changes: Any) -> dict[str, Any]:
    roster = copy.deepcopy(_roster())
    roster["units"][0].update(changes)
    return roster


@pytest.mark.parametrize(
    ("roster", "message"),
    [
        pytest.param("not-a-mapping", "exactly schema, roster_id, and units", id="type"),
        pytest.param({"schema": "x"}, "exactly schema, roster_id, and units", id="fields"),
        pytest.param(
            {**_roster(), "schema": "other.v1"}, "population roster schema must be", id="schema"
        ),
        pytest.param(
            {**_roster(), "roster_id": " padded"}, "roster_id must be a canonical", id="roster-id"
        ),
        pytest.param(
            {**_roster(), "units": []}, "units must be a nonempty array", id="units-empty"
        ),
        pytest.param(
            {**_roster(), "units": ["not-a-unit"]}, "invalid field roster", id="unit-type"
        ),
        pytest.param(
            _with_unit(family_id=" munich"), "family_id must be canonical", id="unit-canonical"
        ),
        pytest.param(
            _with_unit(disposition="unknown"), "unsupported disposition", id="disposition"
        ),
        pytest.param(
            _with_unit(reason="should not be here"), "must have a null reason", id="included-reason"
        ),
        pytest.param(
            _with_unit(disposition="excluded", reason=None),
            "require a canonical reason",
            id="excluded-reason",
        ),
    ],
)
def test_expected_roster_rejects_malformed_contracts(roster, message) -> None:
    with pytest.raises(ValueError, match=message):
        validate_expected_population_roster(roster)


def test_duplicate_unit_identities_are_refused() -> None:
    roster = copy.deepcopy(_roster())
    roster["units"].append(copy.deepcopy(roster["units"][0]))
    with pytest.raises(ValueError, match="duplicate expected population unit ID"):
        validate_expected_population_roster(roster)


def test_observed_vectors_must_align_and_be_canonical() -> None:
    kwargs = {
        "observed_sample_ids": ["sample-1"],
        "observed_family_ids": ["munich"],
        "observed_domain_ids": ["scene-a"],
        "observed_group_ids": ["episode-1"],
    }
    with pytest.raises(ValueError, match="must align"):
        audit_expected_population(
            _roster(), **{**kwargs, "observed_family_ids": ["munich", "munich"]}
        )
    with pytest.raises(ValueError, match="must be canonical"):
        audit_expected_population(_roster(), **{**kwargs, "observed_domain_ids": [" scene-a"]})
