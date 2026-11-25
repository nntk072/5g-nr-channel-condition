"""Path-equivalence records: solver validity, geometry, delays, coefficients and diffuse-path rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from evaluation.acceptance import path_equivalence
from evaluation.acceptance.path_equivalence import (
    build_path_equivalence_classes,
    compare_diffuse_stochastic_path_distributions,
    summarize_diffuse_stochastic_paths,
)
from tests.test_evaluation.test_path_equivalence import (
    _build,
    _diffuse_path,
    _path,
    _solver_budget,
)


def _mutated(**changes: Any) -> dict[str, Any]:
    record = _path(0)
    record.update(changes)
    return record


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        pytest.param({"interaction_type_masks": "x"}, "must be lists", id="masks-type"),
        pytest.param({"interaction_type_masks": [1.5]}, "must be integers", id="mask-float"),
        pytest.param({"interaction_type_masks": [-1]}, "must be nonnegative", id="mask-negative"),
        pytest.param(
            {"interaction_vertices_m": [[1.0, 2.0, 3.0], [4.0, 5.0]]},
            "only real numeric values",
            id="vertex-ragged",
        ),
        pytest.param(
            {"interaction_vertices_m": [[1.0, 2.0]]},
            "shape \\(interaction_slots, 3\\)",
            id="vertex-shape",
        ),
        pytest.param(
            {"interaction_vertices_m": [[np.nan, 2.0, 3.0]]}, "only finite values", id="vertex-nan"
        ),
        pytest.param({"theta_t_rad": "north"}, "must be numeric", id="theta-type"),
        pytest.param({"theta_t_rad": float("inf")}, "must be finite", id="theta-inf"),
        pytest.param({"absolute_delay_s": -1e-9}, "nonnegative", id="delay-negative"),
        pytest.param({"absolute_delay_s": True}, "nonnegative", id="delay-boolean"),
        pytest.param({"absolute_delay_s": float("nan")}, "nonnegative", id="delay-nan"),
        pytest.param(
            {"coefficient_real": True}, "finite real and imaginary", id="coefficient-boolean"
        ),
        pytest.param(
            {"coefficient_real": float("nan")}, "finite real and imaginary", id="coefficient-nan"
        ),
        pytest.param(
            {"coefficient_imag": float("inf")}, "finite real and imaginary", id="coefficient-inf"
        ),
    ],
)
def test_valid_path_records_reject_malformed_geometry_and_coefficients(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _build([_mutated(**changes)])


def test_path_index_must_be_a_nonnegative_integer() -> None:
    with pytest.raises(ValueError, match="path_index"):
        _build([_mutated(path_index=-1)])
    with pytest.raises(ValueError, match="duplicate path_index"):
        _build([_path(0), _path(0)])


def test_tolerances_must_be_finite_and_positive() -> None:
    with pytest.raises(ValueError, match="must be a finite positive number"):
        build_path_equivalence_classes(
            [_path(0)],
            resolution_id="r",
            vertex_tolerance_m=0.0,
            delay_tolerance_s=1e-10,
            direction_tolerance_rad=0.01,
        )
    with pytest.raises(ValueError, match="must be a finite positive number"):
        build_path_equivalence_classes(
            [_path(0)],
            resolution_id="r",
            vertex_tolerance_m=True,
            delay_tolerance_s=1e-10,
            direction_tolerance_rad=0.01,
        )
    with pytest.raises(ValueError, match="must not exceed pi"):
        build_path_equivalence_classes(
            [_path(0)],
            resolution_id="r",
            vertex_tolerance_m=0.01,
            delay_tolerance_s=1e-10,
            direction_tolerance_rad=4.0,
        )
    with pytest.raises(ValueError, match="resolution_id must be a nonempty canonical string"):
        build_path_equivalence_classes(
            [_path(0)],
            resolution_id=" ",
            vertex_tolerance_m=0.01,
            delay_tolerance_s=1e-10,
            direction_tolerance_rad=0.01,
        )


def test_cfr_contributions_must_share_one_frequency_grid() -> None:
    first = _path(0)
    second = _path(1)
    second["cfr_contribution_real"] = [0.05, 0.05, 0.05]
    second["cfr_contribution_imag"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="same frequency grid"):
        _build([first, second])


def test_array_conversion_failure_is_reported_as_invalid_geometry() -> None:
    class InvalidArray:
        def __array__(self, dtype=None, copy=None):
            raise ValueError("fixture rejects array conversion")

    with pytest.raises(ValueError, match="rectangular real-valued array"):
        _build([_mutated(interaction_vertices_m=InvalidArray())])


def test_diffuse_summary_rejects_malformed_diffuse_records() -> None:
    with pytest.raises(ValueError, match="sequence of path records"):
        summarize_diffuse_stochastic_paths("not-paths")
    with pytest.raises(ValueError, match="valid path objects"):
        summarize_diffuse_stochastic_paths(["not-a-path"])
    with pytest.raises(ValueError, match="delay must be finite and nonnegative"):
        summarize_diffuse_stochastic_paths([_diffuse_path(0, delay_s=-1.0)])
    with pytest.raises(ValueError, match="finite real and imaginary values"):
        summarize_diffuse_stochastic_paths([_diffuse_path(0, delay_s=1e-9, power=float("inf"))])
    with pytest.raises(ValueError, match="finite real angles"):
        summarize_diffuse_stochastic_paths(
            [{**_diffuse_path(0, delay_s=1e-9), "theta_t_rad": True}]
        )
    with pytest.raises(ValueError, match="finite zenith and azimuth angles"):
        summarize_diffuse_stochastic_paths([{**_diffuse_path(0, delay_s=1e-9), "theta_t_rad": 4.0}])
    with pytest.raises(ValueError, match="diffuse path power must be finite"):
        summarize_diffuse_stochastic_paths(
            [{**_diffuse_path(0, delay_s=1e-9), "coefficient_real": 1e308}]
        )


def test_diffuse_summary_reports_empty_and_circular_azimuth_support() -> None:
    empty = summarize_diffuse_stochastic_paths([])
    assert empty["status"] == "NO_DIFFUSE_PATHS"
    assert empty["unweighted_quantiles"]["delay_s"] is None
    paths = [
        _diffuse_path(0, delay_s=1e-9, phi_t=0.0),
        _diffuse_path(1, delay_s=1e-9, phi_t=np.pi),
    ]

    summary = summarize_diffuse_stochastic_paths(paths)

    assert summary is not None
    assert summary["unweighted_quantiles"]["tx_azimuth_offset_rad"] is None


def test_diffuse_summary_reports_quantiles_for_valid_paths() -> None:
    summary = summarize_diffuse_stochastic_paths(
        [_diffuse_path(0, delay_s=1e-9, power=1.0), _diffuse_path(1, delay_s=3e-9, power=2.0)]
    )
    assert summary is not None


def test_internal_path_metadata_relies_on_public_valid_mask_and_signature_invariants() -> None:
    valid = _path(0)
    assert _build([valid])["solver_valid_record_count"] == 1

    invalid = _path(1)
    invalid["valid"] = False
    assert _build([invalid])["excluded_invalid_record_count"] == 1


def test_path_class_builder_rejects_aggregate_coefficient_overflow() -> None:
    records = [_path(0, contribution=1e308 + 0j), _path(1, contribution=1e308 + 0j)]

    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="aggregate path-class coefficient must be finite"),
    ):
        _build(records)


def test_path_class_direction_average_uses_representative_on_cancellation(monkeypatch) -> None:
    directions = {
        (0, "t"): np.asarray([0.0, 0.0, 1.0]),
        (1, "t"): np.asarray([0.0, 0.0, -1.0]),
        (0, "r"): np.asarray([0.0, 0.0, 1.0]),
        (1, "r"): np.asarray([0.0, 0.0, -1.0]),
    }
    monkeypatch.setattr(
        path_equivalence,
        "_unit_direction",
        lambda record, endpoint: directions[(record["path_index"], endpoint)],
    )

    classes = build_path_equivalence_classes(
        [_path(0), _path(1)],
        resolution_id="r",
        vertex_tolerance_m=0.01,
        delay_tolerance_s=1e-10,
        direction_tolerance_rad=np.pi,
    )

    assert classes["class_count"] == 1
    assert classes["classes"][0]["power_weighted_departure_direction"] == [0.0, 0.0, 1.0]
    assert classes["classes"][0]["power_weighted_arrival_direction"] == [0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("aggregate_coefficient_real", "0.1", "aggregate coefficient"),
        ("aggregate_coefficient_imag", True, "aggregate coefficient"),
        ("aggregate_coefficient_real", float("inf"), "finite real components"),
    ],
)
def test_path_class_coefficient_fields_reject_untyped_and_nonfinite_values(
    field: str, value: Any, message: str
) -> None:
    candidate = _build([_path(0)])
    challenger = _build([_path(1)])
    challenger["classes"][0][field] = value
    from tests.test_evaluation.test_path_equivalence import _compare

    with pytest.raises(ValueError, match=message):
        _compare(candidate, challenger)


def test_compare_rejects_invalid_bundles_rosters_cfr_and_resolution() -> None:
    from tests.test_evaluation.test_path_equivalence import _compare

    valid = _build([_path(0)])
    with pytest.raises(ValueError, match="bundles must be objects"):
        _compare(None, valid)  # type: ignore[arg-type]

    invalid_roster = dict(valid, classes=())
    with pytest.raises(ValueError, match="rosters must be lists"):
        _compare(invalid_roster, valid)

    malformed_cfr = _build([_path(1)])
    malformed_cfr["classes"][0]["aggregate_cfr_real"] = [0.05, 0.05, 0.05]
    malformed_cfr["classes"][0]["aggregate_cfr_imag"] = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="match the declared full CFR grid"):
        _compare(valid, malformed_cfr)

    malformed_resolution = _build([_path(2)])
    malformed_resolution["resolution"] = dict(malformed_resolution["resolution"])
    malformed_resolution["resolution"]["delay_tolerance_s"] = 2e-10
    with pytest.raises(ValueError, match="resolutions do not match"):
        _compare(valid, malformed_resolution)

    malformed_builder = dict(valid, status="not-built-by-path-equivalence")
    with pytest.raises(ValueError, match="must come from the path-equivalence class builder"):
        _compare(malformed_builder, valid)


def test_compare_rejects_non_object_class_record_and_signature_vertex_mismatch() -> None:
    from tests.test_evaluation.test_path_equivalence import _compare

    valid = _build([_path(0)])
    malformed_record = dict(valid, classes=[None])
    with pytest.raises(ValueError, match="path class record must be an object"):
        _compare(malformed_record, valid)

    malformed_geometry = _build([_path(1)])
    malformed_geometry["classes"][0]["interaction_vertices_m"] = np.empty((0, 3))
    with pytest.raises(ValueError, match="vertices do not match its interaction signature"):
        _compare(malformed_geometry, valid)


@pytest.mark.parametrize(
    "candidate_full_cfr",
    [
        np.asarray([[0.1, 0.1]], dtype=np.complex128),
        np.asarray([0.1, 0.1]),
        np.asarray([complex(float("nan"), 0.0)] * 2),
    ],
    ids=["matrix", "real", "nan"],
)
def test_compare_rejects_invalid_full_cfr(candidate_full_cfr: np.ndarray) -> None:

    classes = _build([_path(0)])
    with pytest.raises(ValueError, match="full CFRs must be finite complex vectors"):
        path_equivalence.compare_path_equivalence_classes(
            classes,
            classes,
            vertex_tolerance_m=0.01,
            delay_tolerance_s=1e-10,
            direction_tolerance_rad=0.01,
            candidate_full_cfr=candidate_full_cfr,
            challenger_full_cfr=np.asarray([0.1, 0.1], dtype=np.complex128),
            h_floor=1e-12,
            transmit_power_per_re_w=1.0,
            noise_power_per_re_w=1e-6,
            resolution_id="frozen-resolutions-v1",
            candidate_configuration_id="candidate-v1",
            challenger_configuration_id="challenger-v1",
        )


def test_compare_validates_tolerances_and_configuration_ids() -> None:

    valid = _build([_path(0)])
    with pytest.raises(ValueError, match="direction_tolerance_rad must not exceed pi"):
        path_equivalence.compare_path_equivalence_classes(
            valid,
            valid,
            vertex_tolerance_m=0.01,
            delay_tolerance_s=1e-10,
            direction_tolerance_rad=4.0,
            candidate_full_cfr=np.asarray([0.1, 0.1], dtype=np.complex128),
            challenger_full_cfr=np.asarray([0.1, 0.1], dtype=np.complex128),
            h_floor=1e-12,
            transmit_power_per_re_w=1.0,
            noise_power_per_re_w=1e-6,
            resolution_id="frozen-resolutions-v1",
            candidate_configuration_id="candidate-v1",
            challenger_configuration_id="challenger-v1",
        )

    with pytest.raises(ValueError, match="candidate_configuration_id"):
        path_equivalence.compare_path_equivalence_classes(
            valid,
            valid,
            vertex_tolerance_m=0.01,
            delay_tolerance_s=1e-10,
            direction_tolerance_rad=0.01,
            candidate_full_cfr=np.asarray([0.1, 0.1], dtype=np.complex128),
            challenger_full_cfr=np.asarray([0.1, 0.1], dtype=np.complex128),
            h_floor=1e-12,
            transmit_power_per_re_w=1.0,
            noise_power_per_re_w=1e-6,
            resolution_id="frozen-resolutions-v1",
            candidate_configuration_id=" ",
            challenger_configuration_id="challenger-v1",
        )


def test_compare_coefficient_tolerance_and_irrelevant_matched_classes() -> None:

    zero_classes = _build([_path(0, contribution=0j)])
    result = path_equivalence.compare_path_equivalence_classes(
        zero_classes,
        zero_classes,
        vertex_tolerance_m=0.01,
        delay_tolerance_s=1e-10,
        direction_tolerance_rad=0.01,
        candidate_full_cfr=np.zeros(2, dtype=np.complex128),
        challenger_full_cfr=np.zeros(2, dtype=np.complex128),
        h_floor=1e-12,
        transmit_power_per_re_w=1.0,
        noise_power_per_re_w=1e-6,
        coefficient_relative_tolerance=0.0,
        resolution_id="frozen-resolutions-v1",
        candidate_configuration_id="candidate-v1",
        challenger_configuration_id="challenger-v1",
    )
    assert result["matched_class_pair_count"] == 1
    assert result["coefficient_criterion_status"] == "NOT_ASSESSABLE_NO_RELEVANT_MATCHED_CLASSES"

    with pytest.raises(ValueError, match="coefficient_relative_tolerance"):
        path_equivalence.compare_path_equivalence_classes(
            zero_classes,
            zero_classes,
            vertex_tolerance_m=0.01,
            delay_tolerance_s=1e-10,
            direction_tolerance_rad=0.01,
            candidate_full_cfr=np.zeros(2, dtype=np.complex128),
            challenger_full_cfr=np.zeros(2, dtype=np.complex128),
            h_floor=1e-12,
            transmit_power_per_re_w=1.0,
            noise_power_per_re_w=1e-6,
            coefficient_relative_tolerance=float("nan"),
            resolution_id="frozen-resolutions-v1",
            candidate_configuration_id="candidate-v1",
            challenger_configuration_id="challenger-v1",
        )


def test_diffuse_distribution_rejects_invalid_delay_and_wall_clock_budget() -> None:
    with pytest.raises(ValueError, match="delay must be finite and nonnegative"):
        summarize_diffuse_stochastic_paths([_diffuse_path(0, delay_s=True)])

    for value, message in [
        ("3600", "must be finite and at least 1e-6"),
        (0.0, "must be finite and positive"),
    ]:
        budget = _solver_budget()
        budget["max_wall_seconds"] = value
        with pytest.raises(ValueError, match=message):
            compare_diffuse_stochastic_path_distributions([], [], solver_budget_provenance=budget)


@pytest.mark.parametrize("coefficient", [True, "1.0"])
def test_diffuse_distribution_rejects_non_numeric_coefficients(coefficient: Any) -> None:
    path = _diffuse_path(0, delay_s=1e-9)
    path["coefficient_real"] = coefficient
    with pytest.raises(ValueError, match="diffuse path coefficients must be finite real"):
        summarize_diffuse_stochastic_paths([path])
