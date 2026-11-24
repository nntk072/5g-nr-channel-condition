"""Path-equivalence records: solver validity, geometry, delays, coefficients and diffuse-path rules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from evaluation.acceptance.path_equivalence import (
    build_path_equivalence_classes,
    summarize_diffuse_stochastic_paths,
)
from tests.test_evaluation.test_path_equivalence import _build, _diffuse_path, _path


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
        pytest.param({"absolute_delay_s": float("nan")}, "nonnegative", id="delay-nan"),
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


def test_diffuse_summary_rejects_malformed_diffuse_records() -> None:
    with pytest.raises(ValueError, match="sequence of path records"):
        summarize_diffuse_stochastic_paths("not-paths")
    with pytest.raises(ValueError, match="valid path objects"):
        summarize_diffuse_stochastic_paths(["not-a-path"])
    with pytest.raises(ValueError, match="delay must be finite and nonnegative"):
        summarize_diffuse_stochastic_paths([_diffuse_path(0, delay_s=-1.0)])
    with pytest.raises(ValueError, match="finite real and imaginary values"):
        summarize_diffuse_stochastic_paths([_diffuse_path(0, delay_s=1e-9, power=float("inf"))])


def test_diffuse_summary_reports_quantiles_for_valid_paths() -> None:
    summary = summarize_diffuse_stochastic_paths(
        [_diffuse_path(0, delay_s=1e-9, power=1.0), _diffuse_path(1, delay_s=3e-9, power=2.0)]
    )
    assert summary is not None
