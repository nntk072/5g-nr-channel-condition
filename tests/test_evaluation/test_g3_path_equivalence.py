from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest
from evaluation.g3_path_equivalence import (
    build_path_equivalence_classes,
    compare_diffuse_stochastic_path_distributions,
    compare_path_equivalence_classes,
    summarize_diffuse_stochastic_paths,
)


def _path(
    index: int,
    *,
    object_id: int = 7,
    primitive_id: int = 2,
    vertex_x: float = 1.0,
    delay_s: float = 1e-7,
    contribution: complex = 0.05 + 0.0j,
    kind: str = "specular",
) -> dict[str, Any]:
    return {
        "valid": True,
        "path_kind": kind,
        "path_index": index,
        "interaction_type_masks": [1],
        "object_ids": [object_id],
        "primitive_ids": [primitive_id],
        "interaction_vertices_m": [[vertex_x, 2.0, 3.0]],
        "absolute_delay_s": delay_s,
        "theta_t_rad": np.pi / 2,
        "phi_t_rad": 0.1,
        "theta_r_rad": np.pi / 2,
        "phi_r_rad": 0.2,
        "coefficient_real": contribution.real,
        "coefficient_imag": contribution.imag,
        "cfr_contribution_real": [contribution.real, contribution.real],
        "cfr_contribution_imag": [contribution.imag, contribution.imag],
    }


def _build(paths: list[dict[str, Any]]) -> dict[str, Any]:
    return build_path_equivalence_classes(
        paths,
        resolution_id="frozen-resolutions-v1",
        vertex_tolerance_m=0.01,
        delay_tolerance_s=1e-10,
        direction_tolerance_rad=0.01,
    )


def _compare(candidate: dict[str, Any], challenger: dict[str, Any]) -> dict[str, Any]:
    return compare_path_equivalence_classes(
        candidate,
        challenger,
        vertex_tolerance_m=0.01,
        delay_tolerance_s=1e-10,
        direction_tolerance_rad=0.01,
        candidate_full_cfr=np.asarray([0.1, 0.1], dtype=np.complex128),
        challenger_full_cfr=np.asarray([0.1, 0.1], dtype=np.complex128),
        h_floor=1e-12,
        transmit_power_per_re_w=1.0,
        noise_power_per_re_w=1e-6,
        resolution_id="frozen-resolutions-v1",
        candidate_configuration_id="candidate-v1",
        challenger_configuration_id="challenger-v1",
    )


def _diffuse_path(
    index: int, *, delay_s: float, power: float = 1.0, phi_t: float = 0.0
) -> dict[str, Any]:
    return {
        "valid": True,
        "path_kind": "diffuse_stochastic",
        "path_index": index,
        "absolute_delay_s": delay_s,
        "theta_t_rad": 0.8,
        "phi_t_rad": phi_t,
        "theta_r_rad": 1.2,
        "phi_r_rad": -phi_t,
        "coefficient_real": power**0.5,
        "coefficient_imag": 0.0,
    }


def _solver_budget() -> dict[str, Any]:
    return {
        "max_rays_per_link": 100_000,
        "max_interactions": 8,
        "max_episodes": 500,
        "max_wall_seconds": 3600.0,
        "overflow_policy": "record_censored_episode",
    }


def test_path_classes_preserve_raw_multiplicity_and_ignore_primitive_as_identity() -> None:
    candidate = _build([_path(0, primitive_id=2), _path(1, primitive_id=5)])
    challenger = _build([_path(9, primitive_id=99, contribution=0.1 + 0.0j)])

    comparison = _compare(candidate, challenger)

    assert candidate["raw_record_count"] == 2
    assert candidate["class_count"] == 1
    assert candidate["classes"][0]["raw_multiplicity"] == 2
    assert candidate["classes"][0]["raw_path_indices"] == [0, 1]
    assert comparison["matched_class_pair_count"] == 1
    assert comparison["class_coverage_status"] == "PASS"
    assert comparison["candidate_bidirectional_coverage"] == 1.0
    assert comparison["challenger_bidirectional_coverage"] == 1.0
    assert comparison["configuration"]["primitive_ids_are_identity"] is False
    assert comparison["admission_possible"] is False


def test_zero_interaction_multiplicity_forms_one_effective_class() -> None:
    first = _path(0, contribution=0.05 + 0.0j)
    second = _path(1, contribution=0.05 + 0.0j)
    for record in (first, second):
        record["interaction_type_masks"] = [0]
        record["object_ids"] = [4_294_967_295]
        record["primitive_ids"] = [4_294_967_295]
        record["interaction_vertices_m"] = [[0.0, 0.0, 0.0]]

    result = _build([first, second])

    assert result["class_count"] == 1
    assert result["classes"][0]["raw_multiplicity"] == 2
    assert result["classes"][0]["interaction_signature"] == []
    assert result["classes"][0]["aggregate_cfr_real"] == pytest.approx([0.1, 0.1])


def test_class_grouping_uses_all_declared_tolerances_and_prevents_chaining() -> None:
    paths = [
        _path(0, vertex_x=1.0),
        _path(1, vertex_x=1.009),
        _path(2, vertex_x=1.018),
        _path(3, delay_s=1.2e-7),
        _path(4, object_id=8),
        _path(5, primitive_id=9),
        {**_path(6), "phi_t_rad": 0.2},
    ]

    result = _build(paths)

    assert result["class_count"] == 5
    assert sorted(item["raw_multiplicity"] for item in result["classes"]) == [1, 1, 1, 1, 3]


def test_invalid_records_are_counted_and_diffuse_identity_matching_is_rejected() -> None:
    invalid = _path(0)
    invalid["valid"] = False
    result = _build([invalid])
    assert result["excluded_invalid_record_count"] == 1
    assert result["solver_valid_record_count"] == 0

    with pytest.raises(ValueError, match="diffuse stochastic paths"):
        _build([_path(1, kind="diffuse")])

    record_without_validity = _path(2)
    del record_without_validity["valid"]
    with pytest.raises(ValueError, match="explicitly declare a boolean valid mask"):
        _build([record_without_validity])


@pytest.mark.parametrize(
    "mutation",
    [
        {"interaction_vertices_m": [[float("nan"), 2.0, 3.0]]},
        {"interaction_vertices_m": [[True, 2.0, 3.0]]},
        {"interaction_vertices_m": [["1.0", 2.0, 3.0]]},
        {"interaction_type_masks": [1, 2]},
        {"cfr_contribution_real": [0.1]},
        {"cfr_contribution_real": ["0.05", 0.05]},
        {"cfr_contribution_imag": [True, 0.0]},
    ],
)
def test_malformed_valid_path_records_fail_closed(mutation: dict[str, Any]) -> None:
    record = _path(0)
    record.update(mutation)
    with pytest.raises(ValueError):
        _build([record])


def test_non_object_path_record_fails_with_contract_error() -> None:
    with pytest.raises(ValueError, match="path record must be an object"):
        _build([None])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aggregate_cfr_real", ["0.05", 0.05]),
        ("aggregate_cfr_imag", [True, 0.0]),
        ("power_weighted_delay_s", True),
        ("power_weighted_departure_direction", [1.0, "0", 0.0]),
        ("interaction_signature", [[True, 7]]),
    ],
)
def test_serialized_path_classes_reject_coercible_malformed_values(
    field: str, value: Any
) -> None:
    candidate = _build([_path(0)])
    challenger = _build([_path(1)])
    challenger["classes"][0][field] = value

    with pytest.raises(ValueError):
        _compare(candidate, challenger)


def test_bidirectional_coverage_lists_relevant_unmatched_classes() -> None:
    candidate = _build([_path(0)])
    challenger = _build([_path(4), _path(5, object_id=8)])
    comparison = _compare(candidate, challenger)

    assert comparison["class_coverage_status"] == "FAIL"
    assert comparison["candidate_bidirectional_coverage"] == 1.0
    assert comparison["challenger_bidirectional_coverage"] == 0.5
    assert len(comparison["challenger_only_relevant_classes"]) == 1
    assert comparison["challenger_only_relevant_classes"][0]["raw_path_indices"] == [5]


def test_empty_path_sets_are_not_reported_as_a_pass() -> None:
    empty = _build([])
    comparison = _compare(empty, empty)
    assert comparison["status"] == "NO_CLASSES_NOT_ASSESSABLE"
    assert comparison["coverage_criterion"] == 0.95
    assert comparison["configuration"]["candidate_configuration_id"] == "candidate-v1"
    assert comparison["configuration"]["challenger_configuration_id"] == "challenger-v1"
    assert comparison["class_coverage_status"] == "NOT_ASSESSABLE_NO_CLASSES"


def test_non_relevant_classes_do_not_create_a_coverage_pass() -> None:
    zero = _path(0, contribution=0.0 + 0.0j)
    candidate = _build([zero])
    challenger = _build([zero])
    comparison = compare_path_equivalence_classes(
        candidate,
        challenger,
        vertex_tolerance_m=0.01,
        delay_tolerance_s=1e-10,
        direction_tolerance_rad=0.01,
        candidate_full_cfr=np.zeros(2, dtype=np.complex128),
        challenger_full_cfr=np.zeros(2, dtype=np.complex128),
        h_floor=1e-12,
        transmit_power_per_re_w=1.0,
        noise_power_per_re_w=1e-6,
        resolution_id="frozen-resolutions-v1",
        candidate_configuration_id="candidate-v1",
        challenger_configuration_id="challenger-v1",
    )

    assert comparison["candidate_relevant_class_count"] == 0
    assert comparison["challenger_relevant_class_count"] == 0
    assert comparison["class_coverage_status"] == "UNRESOLVED_NO_RELEVANT_CLASSES"


def test_all_relevant_classes_missing_on_one_side_is_a_coverage_failure() -> None:
    irrelevant = _path(0, contribution=0.0 + 0.0j)
    relevant = _path(1, object_id=8, contribution=0.05 + 0.0j)
    candidate = _build([irrelevant])
    challenger = _build([relevant])
    comparison = _compare(candidate, challenger)

    assert comparison["candidate_relevant_class_count"] == 0
    assert comparison["challenger_relevant_class_count"] == 1
    assert comparison["candidate_bidirectional_coverage"] is None
    assert comparison["challenger_bidirectional_coverage"] == 0.0
    assert comparison["class_coverage_status"] == "FAIL"
    assert len(comparison["challenger_only_relevant_classes"]) == 1


def test_path_comparison_rejects_classes_built_under_different_resolutions() -> None:
    candidate = _build([_path(0)])
    challenger = _build([_path(1)])
    challenger["resolution_id"] = "different-resolution"

    with pytest.raises(ValueError, match="resolution_id does not match"):
        _compare(candidate, challenger)


def test_input_records_are_not_mutated() -> None:
    records = [_path(0), _path(1)]
    before = copy.deepcopy(records)
    _build(records)
    assert records == before


def test_diffuse_path_comparison_reports_distributions_without_ray_matching() -> None:
    candidate = [
        _diffuse_path(10, delay_s=1e-7, power=1.0, phi_t=np.pi - 0.02),
        _diffuse_path(11, delay_s=2e-7, power=2.0, phi_t=-np.pi + 0.02),
        _diffuse_path(12, delay_s=3e-7, power=1.0, phi_t=np.pi - 0.01),
    ]
    challenger = [
        _diffuse_path(90, delay_s=2e-7, power=1.0, phi_t=-np.pi + 0.02),
        _diffuse_path(91, delay_s=3e-7, power=2.0, phi_t=np.pi - 0.02),
        _diffuse_path(92, delay_s=4e-7, power=1.0, phi_t=-np.pi + 0.01),
    ]

    report = compare_diffuse_stochastic_path_distributions(
        candidate,
        challenger,
        solver_budget_provenance=_solver_budget(),
    )

    assert report["status"] == "DESCRIPTIVE_ONLY_NOT_QUALIFIED"
    assert report["ray_identity_assignment"] is False
    assert report["finite_solver_caps_are_ground_truth"] is False
    assert report["candidate"]["path_count"] == 3
    delay_deltas = report["candidate_minus_challenger_unweighted_quantile_deltas"]["delay_s"]
    assert all(value == pytest.approx(-1e-7) for value in delay_deltas.values())
    power_deltas = report["candidate_minus_challenger_normalized_power_quantile_deltas"]
    assert all(value == pytest.approx(0.0) for value in power_deltas.values())
    assert report["solver_budget_provenance"] == _solver_budget()
    assert report["solver_budget_sha256"]


def test_diffuse_path_summary_is_order_invariant_and_handles_empty_sets() -> None:
    records = [
        _diffuse_path(2, delay_s=3e-7, power=1.0),
        _diffuse_path(8, delay_s=1e-7, power=1.0),
    ]
    reordered = [dict(records[1], path_index=100), dict(records[0], path_index=101)]

    assert summarize_diffuse_stochastic_paths(records) == summarize_diffuse_stochastic_paths(
        reordered
    )
    report = compare_diffuse_stochastic_path_distributions(
        [],
        records,
        solver_budget_provenance=_solver_budget(),
    )
    assert report["candidate"]["status"] == "NO_DIFFUSE_PATHS"
    assert report["candidate_minus_challenger_unweighted_quantile_deltas"]["delay_s"] is None


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda budget: budget.update(max_rays_per_link=True), "max_rays_per_link"),
        (lambda budget: budget.update(overflow_policy=[]), "overflow_policy"),
        (lambda budget: budget.update(unfrozen_field=1), "five-field solver budget"),
    ],
)
def test_diffuse_distribution_requires_exact_valid_solver_budget(mutate, message) -> None:
    budget = _solver_budget()
    mutate(budget)
    with pytest.raises(ValueError, match=message):
        compare_diffuse_stochastic_path_distributions([], [], solver_budget_provenance=budget)


def test_diffuse_distribution_does_not_accept_deterministic_path_identity_records() -> None:
    deterministic = _path(0)
    with pytest.raises(ValueError, match="only diffuse_stochastic paths"):
        summarize_diffuse_stochastic_paths([deterministic])
