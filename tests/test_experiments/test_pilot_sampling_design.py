"""Pilot sampling: frozen design validation, deterministic independent-group selection, outcomes."""

from __future__ import annotations

import copy
import hashlib
import math

import pytest
from experiments.pilot_sampling import audit_pilot_outcomes, build_pilot_sample

FAMILIES = ["cdl", "tdl"]
TOPOLOGIES = ["single_panel", "dual_panel"]
DIFFICULTIES = ["low", "medium", "high"]
PRIORS = ["natural_prevalence", "enriched_stress"]
PRIOR_DIGEST = hashlib.sha256(b"prior").hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _design(groups: int = 2) -> dict:
    natural = [
        (family, topology, difficulty)
        for family in FAMILIES
        for topology in TOPOLOGIES
        for difficulty in DIFFICULTIES
    ]
    cells = []
    for family, topology, difficulty in natural:
        cells.append(
            {
                "channel_family": family,
                "antenna_topology": topology,
                "difficulty": difficulty,
                "prior_kind": "natural_prevalence",
                "independent_groups": groups,
                "target_population_probability": 1.0 / len(natural),
            }
        )
        cells.append(
            {
                "channel_family": family,
                "antenna_topology": topology,
                "difficulty": difficulty,
                "prior_kind": "enriched_stress",
                "independent_groups": groups,
                "target_population_probability": None,
            }
        )
    return {
        "schema": "umcd.pilot_design.v1",
        "sampling_seed": 20261101,
        "prior_provenance_sha256": PRIOR_DIGEST,
        "mandatory_channel_families": FAMILIES,
        "mandatory_antenna_topologies": TOPOLOGIES,
        "mandatory_difficulties": DIFFICULTIES,
        "required_prior_kinds": PRIORS,
        "cells": cells,
    }


def _candidates(groups_per_cell: int = 3, per_group: int = 2) -> dict:
    rows = []
    for cell in _design(groups_per_cell)["cells"]:
        for group in range(groups_per_cell):
            for member in range(per_group):
                label = "|".join(
                    [
                        cell["channel_family"],
                        cell["antenna_topology"],
                        cell["difficulty"],
                        cell["prior_kind"],
                        str(group),
                        str(member),
                    ]
                )
                rows.append(
                    {
                        "candidate_id": label,
                        "independent_group_id": f"{label.rsplit('|', 2)[0]}|g{group}",
                        "channel_family": cell["channel_family"],
                        "antenna_topology": cell["antenna_topology"],
                        "difficulty": cell["difficulty"],
                        "prior_kind": cell["prior_kind"],
                        "source_record_sha256": _digest(label),
                    }
                )
    return {"schema": "umcd.pilot_candidate_roster.v1", "candidates": rows}


def test_complete_roster_selects_each_requested_independent_group_once() -> None:
    sample = build_pilot_sample(_design(groups=2), _candidates(groups_per_cell=3))
    assert sample["status"] == "SAMPLE_ROSTER_COMPLETE"
    assert sample["qualification_status"] == "NOT_QUALIFIED"
    assert sample["selected_candidate_count"] == 2 * 24
    for cell in sample["cells"]:
        assert cell["shortfall"] == 0
        groups = [
            entry["independent_group_id"]
            for entry in sample["selected_candidates"]
            if entry["cell_id"] == cell["cell_id"]
        ]
        assert len(groups) == len(set(groups)) == 2


def test_selection_is_deterministic_and_seed_dependent() -> None:
    first = build_pilot_sample(_design(), _candidates())
    again = build_pilot_sample(_design(), _candidates())
    assert first["selected_candidates"] == again["selected_candidates"]
    reseeded = _design()
    reseeded["sampling_seed"] = 7
    other = build_pilot_sample(reseeded, _candidates())
    assert other["selected_candidates"] != first["selected_candidates"]


def test_shortfall_marks_the_design_incomplete() -> None:
    sample = build_pilot_sample(_design(groups=3), _candidates(groups_per_cell=2))
    assert sample["status"] == "INCOMPLETE_DESIGN"
    assert all(cell["shortfall"] == 1 for cell in sample["cells"])
    assert all(cell["available_independent_groups"] == 2 for cell in sample["cells"])


def test_importance_weights_follow_target_prevalence_and_are_absent_for_enriched_cells() -> None:
    sample = build_pilot_sample(_design(), _candidates())
    for cell in sample["cells"]:
        if cell["prior_kind"] == "natural_prevalence":
            share = cell["sample_share_within_prior"]
            assert share == pytest.approx(cell["selected_independent_groups"] / 24)
            assert cell["design_importance_weight"] == pytest.approx(
                cell["target_population_probability"] / share
            )
        else:
            assert cell["target_population_probability"] is None
            assert cell["design_importance_weight"] is None


def test_selected_candidates_are_unique_per_group_even_when_a_group_has_several_members() -> None:
    sample = build_pilot_sample(_design(groups=1), _candidates(groups_per_cell=1, per_group=4))
    ids = [entry["candidate_id"] for entry in sample["selected_candidates"]]
    assert len(ids) == len(set(ids)) == 24


def _mutate(document: dict, path: list, value) -> dict:
    copied = copy.deepcopy(document)
    node = copied
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return copied


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (["extra"], 1, "exactly the v1 fields"),
        (["schema"], "other", "unsupported pilot design schema"),
        (["sampling_seed"], True, "sampling_seed must be"),
        (["sampling_seed"], -1, "sampling_seed must be"),
        (["sampling_seed"], 2**128, "sampling_seed must be"),
        (["prior_provenance_sha256"], "xyz", "lowercase SHA-256"),
        (["mandatory_channel_families"], [], "nonempty list"),
        (["mandatory_channel_families"], ["cdl", "cdl"], "must be unique"),
        (["mandatory_antenna_topologies"], ["unknown"], "antenna-topology roster"),
        (["mandatory_difficulties"], ["low", "medium", "extreme"], "unsupported value"),
        (["mandatory_difficulties"], ["medium", "high"], "must include low and high"),
        (["required_prior_kinds"], ["natural_prevalence"], "natural and enriched"),
        (["cells"], [], "nonempty list"),
    ],
)
def test_design_validation_rejects_malformed_roster_fields(path, value, message) -> None:
    design = _mutate(_design(), path, value)
    if path == ["extra"]:
        design.pop("extra")
        design["unexpected"] = 1
    with pytest.raises(ValueError, match=message):
        build_pilot_sample(design, _candidates())


def test_design_validation_rejects_bad_cells() -> None:
    base = _design()
    bad_field = copy.deepcopy(base)
    bad_field["cells"][0]["unexpected"] = 1
    with pytest.raises(ValueError, match="each pilot cell"):
        build_pilot_sample(bad_field, _candidates())
    for path, value, message in [
        (["difficulty"], "extreme", "difficulty must be"),
        (["prior_kind"], "other", "prior_kind must be"),
        (["independent_groups"], 0, "positive integer"),
        (["independent_groups"], True, "positive integer"),
        (["target_population_probability"], 1.2, "natural-prevalence"),
    ]:
        cell_path = ["cells", 0] + path
        with pytest.raises(ValueError, match=message):
            build_pilot_sample(_mutate(base, cell_path, value), _candidates())


def test_design_rejects_enriched_cells_that_claim_a_population_probability() -> None:
    design = _design()
    enriched = next(cell for cell in design["cells"] if cell["prior_kind"] == "enriched_stress")
    enriched["target_population_probability"] = 0.1
    with pytest.raises(ValueError, match="must not claim a population probability"):
        build_pilot_sample(design, _candidates())


def test_design_rejects_duplicate_cells_and_unbalanced_natural_probabilities() -> None:
    duplicated = _design()
    duplicated["cells"].append(copy.deepcopy(duplicated["cells"][0]))
    with pytest.raises(ValueError, match="cells must be unique"):
        build_pilot_sample(duplicated, _candidates())
    unbalanced = _design()
    unbalanced["cells"][0]["target_population_probability"] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        build_pilot_sample(unbalanced, _candidates())


def test_design_rejects_rosters_that_do_not_match_the_cells() -> None:
    extra_family = _design()
    extra_family["mandatory_channel_families"] = FAMILIES + ["rician"]
    with pytest.raises(ValueError, match="channel-family roster"):
        build_pilot_sample(extra_family, _candidates())
    extra_topology = _design()
    extra_topology["mandatory_antenna_topologies"] = TOPOLOGIES + ["quad"]
    with pytest.raises(ValueError, match="antenna-topology roster"):
        build_pilot_sample(extra_topology, _candidates())
    extra_difficulty = _design()
    extra_difficulty["mandatory_difficulties"] = DIFFICULTIES + ["extreme"]
    with pytest.raises(ValueError, match="unsupported value"):
        build_pilot_sample(extra_difficulty, _candidates())


def test_design_requires_every_difficulty_and_prior_for_each_family_topology_pair() -> None:
    design = _design()
    design["cells"] = [
        cell
        for cell in design["cells"]
        if not (
            cell["channel_family"] == "cdl"
            and cell["antenna_topology"] == "dual_panel"
            and cell["difficulty"] == "high"
            and cell["prior_kind"] == "enriched_stress"
        )
    ]
    # Keep natural probabilities summing to one over the remaining natural cells.
    naturals = [cell for cell in design["cells"] if cell["prior_kind"] == "natural_prevalence"]
    for cell in naturals:
        cell["target_population_probability"] = 1.0 / len(naturals)
    with pytest.raises(ValueError, match="must cross every mandatory difficulty"):
        build_pilot_sample(design, _candidates())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c.update(extra=1), "exactly the v1 fields"),
        (lambda c: c.update(schema="other"), "unsupported pilot candidate roster"),
        (lambda c: c.update(candidates=[]), "must not be empty"),
        (lambda c: c["candidates"][0].update(extra=1), "each pilot candidate"),
        (lambda c: c["candidates"][0].update(difficulty="extreme"), "unsupported"),
        (lambda c: c["candidates"][0].update(source_record_sha256="zz"), "lowercase SHA-256"),
        (
            lambda c: c["candidates"][1].update(candidate_id=c["candidates"][0]["candidate_id"]),
            "must be unique",
        ),
    ],
)
def test_candidate_roster_validation(mutation, message) -> None:
    roster = copy.deepcopy(_candidates())
    mutation(roster)
    with pytest.raises(ValueError, match=message):
        build_pilot_sample(_design(), roster)


def test_candidate_roster_rejects_non_dict_roster() -> None:
    with pytest.raises(ValueError, match="exactly the v1 fields"):
        build_pilot_sample(_design(), ["not", "a", "roster"])


def _sample() -> dict:
    return build_pilot_sample(_design(), _candidates())


def _outcomes(sample: dict, *, outcome: str = "included", runtime: float = 2.0) -> dict:
    rows = []
    for index, entry in enumerate(sample["selected_candidates"]):
        rows.append(
            {
                "candidate_id": entry["candidate_id"],
                "outcome": outcome if index % 2 == 0 else "no_path",
                "runtime_seconds": runtime + index,
                "storage_bytes": 1000 + index,
                "result_sha256": _digest(entry["candidate_id"]),
            }
        )
    return {"schema": "umcd.pilot_outcomes.v1", "outcomes": rows}


def test_audit_reports_complete_cost_and_outcome_coverage() -> None:
    sample = _sample()
    report = audit_pilot_outcomes(sample, _outcomes(sample))
    assert report["status"] == "COMPLETE_DIAGNOSTIC"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["observed_candidate_count"] == len(sample["selected_candidates"])
    for cell in report["cells"]:
        assert cell["not_run_count"] == 0
        assert sum(cell["outcome_counts"].values()) == cell["observed_candidate_count"]
        runtime = cell["runtime_seconds"]
        assert runtime["p50"] <= runtime["p90"] <= runtime["p99"]
        assert (
            cell["storage_bytes"]["total"]
            == sum(1000 + i for i in range(cell["observed_candidate_count"]))
            or cell["storage_bytes"]["total"] > 0
        )


def test_audit_is_incomplete_when_candidates_are_not_run() -> None:
    sample = _sample()
    outcomes = _outcomes(sample)
    outcomes["outcomes"] = outcomes["outcomes"][:-1]
    report = audit_pilot_outcomes(sample, outcomes)
    assert report["status"] == "INCOMPLETE_DIAGNOSTIC"
    assert sum(cell["not_run_count"] for cell in report["cells"]) == 1


def test_audit_is_incomplete_for_an_incomplete_sample_design() -> None:
    sample = build_pilot_sample(_design(groups=3), _candidates(groups_per_cell=2))
    report = audit_pilot_outcomes(sample, _outcomes(sample))
    assert report["status"] == "INCOMPLETE_DIAGNOSTIC"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda o: o.update(extra=1), "exactly the v1 fields"),
        (lambda o: o.update(schema="other"), "unsupported pilot outcomes"),
        (lambda o: o["outcomes"][0].update(extra=1), "each pilot outcome"),
        (
            lambda o: o["outcomes"][1].update(candidate_id=o["outcomes"][0]["candidate_id"]),
            "at most one row",
        ),
        (lambda o: o["outcomes"][0].update(outcome="maybe"), "outcome value is unsupported"),
        (lambda o: o["outcomes"][0].update(runtime_seconds=True), "finite nonnegative"),
        (lambda o: o["outcomes"][0].update(runtime_seconds=-1.0), "finite nonnegative"),
        (lambda o: o["outcomes"][0].update(runtime_seconds=math.inf), "finite nonnegative"),
        (lambda o: o["outcomes"][0].update(runtime_seconds=10**400), "finite nonnegative"),
        (lambda o: o["outcomes"][0].update(storage_bytes=-5), "storage_bytes must be"),
        (lambda o: o["outcomes"][0].update(storage_bytes=2**63), "storage_bytes must be"),
        (lambda o: o["outcomes"][0].update(storage_bytes=True), "storage_bytes must be"),
        (lambda o: o["outcomes"][0].update(result_sha256="bad"), "lowercase SHA-256"),
    ],
)
def test_outcome_validation(mutation, message) -> None:
    sample = _sample()
    outcomes = _outcomes(sample)
    mutation(outcomes)
    with pytest.raises(ValueError, match=message):
        audit_pilot_outcomes(sample, outcomes)


def test_audit_rejects_outcomes_outside_the_frozen_sample() -> None:
    sample = _sample()
    outcomes = _outcomes(sample)
    outcomes["outcomes"].append(
        {
            "candidate_id": "not-selected",
            "outcome": "failed",
            "runtime_seconds": 0.0,
            "storage_bytes": 0,
            "result_sha256": _digest("x"),
        }
    )
    with pytest.raises(ValueError, match="outside the frozen sample"):
        audit_pilot_outcomes(sample, outcomes)


def test_audit_rejects_malformed_sample_reports() -> None:
    sample = _sample()
    broken = copy.deepcopy(sample)
    broken["selected_candidates"] = broken["selected_candidates"] + [
        broken["selected_candidates"][0]
    ]
    with pytest.raises(ValueError):
        audit_pilot_outcomes(broken, _outcomes(sample))
    broken_cells = copy.deepcopy(sample)
    broken_cells["cells"] = "not-a-list"
    with pytest.raises(ValueError):
        audit_pilot_outcomes(broken_cells, _outcomes(sample))
