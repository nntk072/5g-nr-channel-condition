"""Path-equivalence roster inputs: raw path records, CFR components and roster identity rules."""

from __future__ import annotations

from typing import Any

import pytest
from evaluation.acceptance.path_equivalence_assurance import evaluate_path_equivalence_roster
from tests.test_evaluation.test_path_equivalence_assurance import (
    _raw_configuration,
    _raw_record,
)

SCENARIO = [("site-family-a", "LOS", "episode-1")]


def _evaluate(
    records=None, *, expected=SCENARIO, configuration=None, **extra: Any
) -> dict[str, Any]:
    return evaluate_path_equivalence_roster(
        [_raw_record()] if records is None else records,
        expected_scenarios=expected,
        comparison_configuration=_raw_configuration() if configuration is None else configuration,
        **extra,
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        pytest.param(
            "vertex_tolerance_m", 0.0, "must be a finite positive number", id="vertex-zero"
        ),
        pytest.param(
            "delay_tolerance_s", float("nan"), "must be a finite positive number", id="delay-nan"
        ),
        pytest.param("h_floor", -1.0, "must be a finite positive number", id="floor-negative"),
        pytest.param("direction_tolerance_rad", 4.0, "must not exceed pi", id="direction"),
    ],
)
def test_comparison_configuration_tolerances_are_checked(key, value, message) -> None:
    configuration = {**_raw_configuration(), key: value}
    with pytest.raises(ValueError, match=message):
        _evaluate(configuration=configuration)


def test_raw_configuration_roster_is_fixed() -> None:
    configuration = {**_raw_configuration(), "extra_field": 1}
    with pytest.raises(ValueError, match="unsupported field roster"):
        _evaluate(configuration=configuration)


def test_coefficient_tolerance_must_be_finite_and_nonnegative() -> None:
    with pytest.raises(
        ValueError, match="coefficient_relative_tolerance must be finite and nonnegative"
    ):
        _evaluate(coefficient_relative_tolerance=-0.1)


def test_raw_records_must_be_nonempty_arrays_of_path_objects() -> None:
    record = _raw_record()
    record["candidate_paths"] = "not-a-list"
    with pytest.raises(ValueError, match="must be arrays"):
        _evaluate([record])
    record = _raw_record()
    record["candidate_paths"] = ["not-a-path"]
    with pytest.raises(ValueError, match="must contain path objects"):
        _evaluate([record])


def test_diffuse_raw_paths_are_not_matched_as_identities() -> None:
    record = _raw_record()
    record["challenger_paths"][0]["path_kind"] = "diffuse_stochastic"
    with pytest.raises(ValueError, match="compared as distributions, not identities"):
        _evaluate([record])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param(
            "candidate_full_cfr_real", [0.05], "must be equal nonempty arrays", id="length"
        ),
        pytest.param(
            "candidate_full_cfr_real", ["x", "y"], "must contain real numeric values", id="type"
        ),
        pytest.param(
            "candidate_full_cfr_imag", [0.0, float("inf")], "must be finite", id="nonfinite"
        ),
    ],
)
def test_full_cfr_components_are_numeric_and_finite(field, value, message) -> None:
    record = _raw_record()
    record[field] = value
    with pytest.raises(ValueError, match=message):
        _evaluate([record])


@pytest.mark.parametrize(
    ("expected", "message"),
    [
        pytest.param([], "must be a nonempty preregistered roster", id="empty-roster"),
        pytest.param(SCENARIO + SCENARIO, "duplicate expected scenario", id="duplicate"),
    ],
)
def test_expected_scenario_roster_is_nonempty_and_unique(expected, message) -> None:
    with pytest.raises(ValueError, match=message):
        _evaluate(expected=expected)


def test_roster_needs_records() -> None:
    with pytest.raises(ValueError, match="records must be nonempty"):
        _evaluate([])


def test_scenario_conditions_must_be_los_or_nlos() -> None:
    record = _raw_record()
    record["condition"] = "DAY"
    with pytest.raises(ValueError, match="expected scenarios must be"):
        _evaluate([record], expected=[("site-family-a", "DAY", "episode-1")])
