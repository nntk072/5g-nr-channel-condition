from __future__ import annotations

import copy

import pytest
from experiments.pilot_sampling import audit_pilot_outcomes, build_pilot_sample


def _digest(char: str) -> str:
    return char * 64


def _design(groups: int = 2) -> dict[str, object]:
    return {
        "schema": "umcd.pilot_design.v1",
        "sampling_seed": 192837,
        "prior_provenance_sha256": _digest("a"),
        "mandatory_channel_families": ["tdl-a"],
        "mandatory_antenna_topologies": ["siso"],
        "mandatory_difficulties": ["low", "high"],
        "required_prior_kinds": ["natural_prevalence", "enriched_stress"],
        "cells": [
            {
                "channel_family": "tdl-a",
                "antenna_topology": "siso",
                "difficulty": difficulty,
                "prior_kind": prior,
                "independent_groups": groups,
                "target_population_probability": (0.5 if prior == "natural_prevalence" else None),
            }
            for prior in ("natural_prevalence", "enriched_stress")
            for difficulty in ("low", "high")
        ],
    }


def _candidates() -> dict[str, object]:
    return {
        "schema": "umcd.pilot_candidate_roster.v1",
        "candidates": [
            {
                "candidate_id": f"{prior}-{difficulty}-{index}",
                "independent_group_id": f"{prior}-group-{index}",
                "channel_family": "tdl-a",
                "antenna_topology": "siso",
                "difficulty": difficulty,
                "prior_kind": prior,
                "source_record_sha256": _digest(str(index + 1)),
            }
            for prior in ("natural_prevalence", "enriched_stress")
            for difficulty in ("low", "high")
            for index in range(2)
        ],
    }


def test_pilot_sample_is_replayable_and_reports_diagnostic_outcomes_and_costs() -> None:
    design = _design()
    candidates = _candidates()
    first = build_pilot_sample(design, candidates)
    reordered = copy.deepcopy(candidates)
    reordered["candidates"].reverse()
    replay = build_pilot_sample(design, reordered)

    assert first["status"] == "SAMPLE_ROSTER_COMPLETE"
    assert first["selected_candidates"] == replay["selected_candidates"]
    assert first["qualification_status"] == "NOT_QUALIFIED"
    natural_cells = [cell for cell in first["cells"] if cell["prior_kind"] == "natural_prevalence"]
    enriched_cells = [cell for cell in first["cells"] if cell["prior_kind"] == "enriched_stress"]
    assert all(cell["sample_share_within_prior"] == 0.5 for cell in natural_cells)
    assert all(cell["design_importance_weight"] == 1.0 for cell in natural_cells)
    assert all(cell["target_population_probability"] is None for cell in enriched_cells)
    assert all(cell["design_importance_weight"] is None for cell in enriched_cells)
    outcomes = {
        "schema": "umcd.pilot_outcomes.v1",
        "outcomes": [
            {
                "candidate_id": selected["candidate_id"],
                "outcome": outcome,
                "runtime_seconds": float(index + 1),
                "storage_bytes": (index + 1) * 1024,
                "result_sha256": _digest(str(index + 1)),
            }
            for index, (selected, outcome) in enumerate(
                zip(
                    first["selected_candidates"],
                    (
                        "included",
                        "no_path",
                        "empty",
                        "rejected",
                        "failed",
                        "included",
                        "no_path",
                        "empty",
                    ),
                    strict=True,
                )
            )
        ],
    }

    report = audit_pilot_outcomes(first, outcomes)

    assert report["status"] == "COMPLETE_DIAGNOSTIC"
    assert report["outcome_counts"] == {
        "empty": 2,
        "failed": 1,
        "included": 2,
        "no_path": 2,
        "rejected": 1,
    }
    assert report["runtime_seconds_total"] == 36.0
    assert report["storage_bytes_total"] == 36864
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_pilot_sample_and_audit_preserve_shortfalls_and_not_run_outcomes() -> None:
    sample = build_pilot_sample(_design(groups=3), _candidates())
    report = audit_pilot_outcomes(sample, {"schema": "umcd.pilot_outcomes.v1", "outcomes": []})

    assert sample["status"] == "INCOMPLETE_DESIGN"
    assert all(cell["shortfall"] == 1 for cell in sample["cells"])
    assert report["status"] == "INCOMPLETE_DIAGNOSTIC"
    assert report["not_run_candidate_count"] == 8
    assert all(cell["not_run_count"] == 2 for cell in report["cells"])


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda design: design.update(sampling_seed=True), "sampling_seed"),
        (lambda design: design.update(mandatory_difficulties=["low"]), "mandatory_difficulties"),
        (lambda design: design["cells"][0].update(independent_groups=0), "independent_groups"),
        (lambda design: design["cells"][0].update(target_population_probability=0.4), "sum to one"),
    ],
)
def test_pilot_design_rejects_malformed_frozen_factors(mutate, message: str) -> None:
    design = _design()
    mutate(design)

    with pytest.raises(ValueError, match=message):
        build_pilot_sample(design, _candidates())


def test_pilot_audit_rejects_extra_or_duplicate_outcome_rows() -> None:
    sample = build_pilot_sample(_design(), _candidates())
    outcome = {
        "candidate_id": "unselected",
        "outcome": "failed",
        "runtime_seconds": 1.0,
        "storage_bytes": 0,
        "result_sha256": _digest("f"),
    }

    with pytest.raises(ValueError, match="outside the frozen sample"):
        audit_pilot_outcomes(sample, {"schema": "umcd.pilot_outcomes.v1", "outcomes": [outcome]})

    selected = sample["selected_candidates"][0]["candidate_id"]
    duplicate = [dict(outcome, candidate_id=selected), dict(outcome, candidate_id=selected)]
    with pytest.raises(ValueError, match="at most one row"):
        audit_pilot_outcomes(sample, {"schema": "umcd.pilot_outcomes.v1", "outcomes": duplicate})


def test_pilot_design_requires_full_difficulty_prior_cross_per_family_topology() -> None:
    design = _design()
    design["cells"] = [
        cell
        for cell in design["cells"]
        if not (cell["difficulty"] == "high" and cell["prior_kind"] == "enriched_stress")
    ]

    with pytest.raises(ValueError, match="must cross every mandatory difficulty and prior"):
        build_pilot_sample(design, _candidates())
