from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from datasets import path_evolution as evolution
from datasets.path_evolution import audit_path_evolution, path_evolution_audit_to_dict
from datasets.path_oracle import PathOracleTable, PathSolveStatus


def _table(
    frames: tuple[int, ...],
    offsets: tuple[int, ...],
    delays: tuple[float, ...],
    *,
    gains: tuple[complex, ...] | None = None,
    doppler: tuple[float, ...] | None = None,
    statuses: tuple[PathSolveStatus, ...] | None = None,
) -> PathOracleTable:
    path_count = len(delays)
    return PathOracleTable(
        link_ids=("link-a",) * len(frames),
        frame_indices=np.asarray(frames, dtype=np.int64),
        rx_port_indices=np.zeros(len(frames), dtype=np.int64),
        tx_port_indices=np.zeros(len(frames), dtype=np.int64),
        path_offsets=np.asarray(offsets, dtype=np.int64),
        solve_status=statuses or (PathSolveStatus.SOLVED,) * len(frames),
        delay_s=np.asarray(delays, dtype=np.float64),
        complex_gain=np.asarray(gains or (1.0 + 0.0j,) * path_count, dtype=np.complex128),
        interaction_order=np.zeros(path_count, dtype=np.int64),
        doppler_hz=None if doppler is None else np.asarray(doppler, dtype=np.float64),
    )


def _audit(table: PathOracleTable, times: dict[int, float], **kwargs):
    return audit_path_evolution(
        table,
        times,
        max_delay_change_s=2e-9,
        max_angle_change_rad=0.5,
        max_doppler_change_hz=100.0,
        **kwargs,
    )


def test_path_association_uses_descriptors_not_storage_row_order_and_checks_doppler_phase():
    phase = 2.0 * np.pi * 10.0 * 0.1
    table = _table(
        (0, 1),
        (0, 2, 4),
        (1e-8, 2e-8, 2.01e-8, 1.01e-8),
        gains=(1.0 + 0j, 0.5 + 0j, 0.5 * np.exp(1j * phase), np.exp(1j * phase)),
        doppler=(10.0,) * 4,
    )

    result = _audit(
        table,
        {0: 0.0, 1: 0.1},
        phase_gauge_by_frame={0: "coherent-chain-a", 1: "coherent-chain-a"},
    )

    assert result.schema == "umcd.path_evolution_audit.v1"
    transition = result.transitions[0]
    assert [
        (item.previous_path_row, item.current_path_row) for item in transition.associations
    ] == [
        (0, 3),
        (1, 2),
    ]
    assert all(
        abs(item.doppler_phase_residual_rad or 0.0) < 1e-12 for item in transition.associations
    )
    assert transition.births == ()
    assert transition.deaths == ()
    assert "not identities" in result.interpretation
    ungauged = _audit(table, {0: 0.0, 1: 0.1}).transitions[0]
    assert all(item.doppler_phase_residual_rad is None for item in ungauged.associations)


def test_path_association_reports_births_deaths_and_near_tied_ambiguity():
    birth_death = _table(
        (0, 1),
        (0, 2, 4),
        (1e-8, 2e-8, 1.05e-8, 4e-8),
    )
    transition = _audit(birth_death, {0: 0.0, 1: 0.01}).transitions[0]
    assert len(transition.associations) == 1
    assert transition.births == (3,)
    assert transition.deaths == (1,)

    ambiguous = _table(
        (0, 1),
        (0, 1, 3),
        (1e-8, 1e-8, 1.01e-8),
    )
    uncertain = _audit(ambiguous, {0: 0.0, 1: 0.01}).transitions[0]
    assert uncertain.status == "association_ambiguous"
    assert uncertain.ambiguous_previous_rows == (0,)
    assert uncertain.ambiguous_current_rows == (1, 2)
    assert uncertain.associations == ()
    assert uncertain.births == uncertain.deaths == ()


def test_path_evolution_censors_solver_failures_and_missing_groups():
    failed = _table(
        (0, 1),
        (0, 1, 1),
        (1e-8,),
        statuses=(PathSolveStatus.SOLVED, PathSolveStatus.UNCONVERGED),
    )
    assert _audit(failed, {0: 0.0, 1: 0.1}).transitions[0].status == "censored_solver_outcome"

    missing = _table((0, 2), (0, 1, 2), (1e-8, 1.1e-8))
    result = _audit(missing, {0: 0.0, 1: 0.05, 2: 0.1})
    assert [item.status for item in result.transitions] == [
        "censored_missing_group",
        "censored_missing_group",
    ]


def test_interaction_signature_changes_are_not_silently_matched():
    table = PathOracleTable(
        link_ids=("link-a", "link-a"),
        frame_indices=np.asarray([0, 1], dtype=np.int64),
        rx_port_indices=np.zeros(2, dtype=np.int64),
        tx_port_indices=np.zeros(2, dtype=np.int64),
        path_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        solve_status=(PathSolveStatus.SOLVED, PathSolveStatus.SOLVED),
        delay_s=np.asarray([1e-8, 1.01e-8]),
        complex_gain=np.ones(2, dtype=np.complex128),
        interaction_order=np.ones(2, dtype=np.int64),
        interaction_offsets=np.asarray([0, 1, 2], dtype=np.int64),
        interaction_types=np.asarray([1, 1], dtype=np.int64),
        object_ids=np.asarray([3, 4], dtype=np.int64),
        primitive_ids=np.asarray([0, 0], dtype=np.int64),
        interaction_vertices_m=np.zeros((2, 3), dtype=np.float64),
        interaction_vertex_frame="scene_local_m",
    )

    transition = _audit(table, {0: 0.0, 1: 0.01}).transitions[0]

    assert transition.associations == ()
    assert transition.deaths == (0,)
    assert transition.births == (1,)


def test_path_evolution_report_is_source_bound_and_schema_valid():
    jsonschema = pytest.importorskip("jsonschema")
    result = _audit(_table((0, 1), (0, 1, 2), (1e-8, 1.1e-8)), {0: 0.0, 1: 0.1})
    report = path_evolution_audit_to_dict(result)
    schema_path = (
        Path(__file__).parents[2]
        / "datasets"
        / "schema"
        / "umcd-path-evolution-audit.v1.schema.json"
    )

    jsonschema.Draft202012Validator(json.loads(schema_path.read_text())).validate(report)
    report_digest = report.pop("report_sha256")
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest() == report_digest
    assert len(result.source_path_table_sha256) == 64


def test_path_evolution_bounds_candidate_pairs_and_validates_time_contract():
    table = _table((0, 1), (0, 2, 4), (1e-8, 2e-8, 1.1e-8, 2.1e-8))
    with pytest.raises(ValueError, match="candidate-pair limit"):
        _audit(table, {0: 0.0, 1: 0.1}, maximum_candidate_pair_count=3)
    with pytest.raises(ValueError, match="strictly increasing"):
        _audit(table, {0: 0.1, 1: 0.1})
    with pytest.raises(TypeError, match="real numeric seconds"):
        _audit(table, {0: 0.0, 1: True})
    with pytest.raises(TypeError, match="real numeric seconds"):
        _audit(table, {0: 0.0, 1: np.complex128(0.1 + 0.0j)})


def test_path_evolution_rejects_invalid_roots_frames_gauges_and_limits():
    table = _table((0, 1), (0, 1, 2), (1e-8, 1.1e-8))
    with pytest.raises(TypeError, match="PathOracleTable"):
        _audit(object(), {0: 0.0, 1: 0.1})
    with pytest.raises(ValueError, match="nonempty frame-to-time mapping"):
        _audit(table, {})
    with pytest.raises(ValueError, match="frame keys"):
        _audit(table, {True: 0.0})
    with pytest.raises(ValueError, match="finite"):
        _audit(table, {0: 0.0, 1: float("inf")})
    with pytest.raises(ValueError, match="every path-table frame"):
        _audit(table, {0: 0.0})
    with pytest.raises(TypeError, match="frame-to-ID mapping"):
        _audit(table, {0: 0.0, 1: 0.1}, phase_gauge_by_frame=["gauge"])
    with pytest.raises(ValueError, match="declared time"):
        _audit(table, {0: 0.0, 1: 0.1}, phase_gauge_by_frame={2: "gauge"})
    with pytest.raises(ValueError, match="canonical strings"):
        _audit(table, {0: 0.0, 1: 0.1}, phase_gauge_by_frame={0: " padded "})
    for option in (
        {"max_delay_change_s": True},
        {"max_angle_change_rad": float("nan")},
        {"max_doppler_change_hz": 0.0},
        {"ambiguity_cost_margin": 1j},
    ):
        with pytest.raises((TypeError, ValueError)):
            _audit(table, {0: 0.0, 1: 0.1}, **option)
    with pytest.raises(TypeError, match="must be an integer"):
        _audit(table, {0: 0.0, 1: 0.1}, maximum_candidate_pair_count=True)
    with pytest.raises(ValueError, match="must be positive"):
        _audit(table, {0: 0.0, 1: 0.1}, maximum_candidate_pair_count=0)


def test_pair_cost_reports_gates_and_physical_descriptor_fallback():
    previous = evolution._PathDescriptor(0, 1.0, 1 + 0j, 0.0, (0.0, 0.0), None, ((1, 2, 3),))
    same = evolution._PathDescriptor(1, 1.1, 1 + 0j, 1.0, (0.1, 0.0), None, ((1, 2, 3),))
    assert (
        evolution._pair_cost(
            previous,
            same,
            max_delay_change_s=1.0,
            max_angle_change_rad=1.0,
            max_doppler_change_hz=2.0,
        )[1]
        == "delay_and_interaction_signature_angles_doppler"
    )
    assert (
        evolution._pair_cost(
            previous,
            evolution._PathDescriptor(1, 3.0, 1 + 0j, 1.0, (0.1, 0.0), None, ((1, 2, 3),)),
            max_delay_change_s=1.0,
            max_angle_change_rad=1.0,
            max_doppler_change_hz=2.0,
        )
        is None
    )
    assert (
        evolution._pair_cost(
            previous,
            evolution._PathDescriptor(1, 1.1, 1 + 0j, 1.0, (2.0, 0.0), None, ((1, 2, 3),)),
            max_delay_change_s=1.0,
            max_angle_change_rad=1.0,
            max_doppler_change_hz=2.0,
        )
        is None
    )
    assert (
        evolution._pair_cost(
            previous,
            evolution._PathDescriptor(1, 1.1, 1 + 0j, 4.0, (0.1, 0.0), None, ((1, 2, 3),)),
            max_delay_change_s=1.0,
            max_angle_change_rad=1.0,
            max_doppler_change_hz=2.0,
        )
        is None
    )
    fallback = evolution._PathDescriptor(1, 1.1, 1 + 0j, None, None, None, None)
    assert (
        evolution._pair_cost(
            previous,
            fallback,
            max_delay_change_s=1.0,
            max_angle_change_rad=1.0,
            max_doppler_change_hz=2.0,
        )[1]
        == "physical_descriptor_fallback"
    )


def test_path_evolution_report_requires_audit_value():
    with pytest.raises(TypeError, match="PathEvolutionAudit"):
        path_evolution_audit_to_dict(object())


def test_no_path_solver_status_is_preserved_as_a_transition_state():
    table = _table(
        (0, 1),
        (0, 1, 1),
        (1e-8,),
        statuses=(PathSolveStatus.SOLVED, PathSolveStatus.NO_PATH),
    )
    assert _audit(table, {0: 0.0, 1: 0.1}).transitions[0].status == "no_path_transition"


def test_path_descriptors_reject_incomplete_interaction_inventory():
    table = object.__new__(PathOracleTable)
    object.__setattr__(table, "path_offsets", np.asarray([0, 1]))
    object.__setattr__(table, "interaction_offsets", np.asarray([0, 0]))
    object.__setattr__(table, "interaction_types", None)
    object.__setattr__(table, "object_ids", None)
    object.__setattr__(table, "primitive_ids", None)
    with pytest.raises(ValueError, match="require interaction type/object/primitive arrays"):
        evolution._descriptors(table, 0)


def test_path_evolution_requires_positive_association_limits():
    table = _table((0, 1), (0, 1, 2), (1e-8, 1.1e-8))
    with pytest.raises(ValueError, match="max_delay_change_s must be finite and positive"):
        audit_path_evolution(table, {0: 0.0, 1: 0.1}, max_delay_change_s=0.0)


@pytest.mark.parametrize(
    ("delays", "status"),
    [
        ((1e-8, 1.02e-8, 1.01e-8), "association_ambiguous"),
        ((1e-8, 1.8e-8, 1.9e-8), "compared"),
    ],
)
def test_path_evolution_handles_current_path_cost_ties_and_separated_costs(delays, status):
    table = _table((0, 1), (0, 2, 3), delays)
    result = audit_path_evolution(
        table, {0: 0.0, 1: 0.1}, max_delay_change_s=1e-8, max_angle_change_rad=0.5
    )
    assert result.transitions[0].status == status


def test_path_evolution_keeps_best_of_separated_candidate_costs():
    table = _table((0, 1), (0, 1, 3), (1e-8, 1.1e-8, 1.9e-8))
    transition = audit_path_evolution(table, {0: 0.0, 1: 0.1}, max_delay_change_s=1e-8).transitions[
        0
    ]
    assert transition.status == "compared"
    assert len(transition.associations) == 1
    assert transition.associations[0].current_path_row == 1


def test_path_evolution_enforces_pair_budget_after_candidate_enumeration():
    table = _table((0, 1), (0, 2, 4), (1e-8, 2e-8, 1.1e-8, 2.1e-8))
    with pytest.raises(ValueError, match="candidate-pair limit"):
        _audit(table, {0: 0.0, 1: 0.1}, maximum_candidate_pair_count=1)
