"""Sample-size design primitives: probability parsing, Hoeffding and exact-binomial designs."""

from __future__ import annotations

import pytest
from evaluation.sample_size import binomial
from evaluation.sample_size.binomial import (
    _exact_binomial_failure_design,
    _maximum_angular_units_for_pass_probability,
    _probability,
    _required_bounded_mean_units,
    _required_quantile_units,
)


@pytest.mark.parametrize("value", [True, "0.5", None])
def test_probability_must_be_numeric(value) -> None:
    with pytest.raises(ValueError, match="must be a finite probability"):
        _probability(value, "alpha")


@pytest.mark.parametrize("value", [0.0, 1.0, float("nan"), -0.1])
def test_probability_must_be_open_unit_interval(value) -> None:
    with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
        _probability(value, "alpha")


def test_quantile_design_requires_pass_probability_above_the_quantile() -> None:
    with pytest.raises(ValueError, match="must exceed the target quantile"):
        _required_quantile_units(quantile=0.9, planned_pass_probability=0.9, alpha=0.05, beta=0.2)
    units = _required_quantile_units(
        quantile=0.5, planned_pass_probability=0.9, alpha=0.05, beta=0.2
    )
    assert units > 0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"planned_mean": 2.0}, "planned_mean must lie within", id="mean-outside"),
        pytest.param({"lower_margin": 0.5, "upper_margin": 0.4}, "strictly ordered", id="margins"),
        pytest.param({"outcome_lower": 0.15}, "complete acceptance interval", id="range-low"),
        pytest.param({"outcome_upper": 0.85}, "complete acceptance interval", id="range-high"),
    ],
)
def test_bounded_mean_design_rejects_inconsistent_margins(overrides, message) -> None:
    arguments = {
        "planned_mean": 0.5,
        "lower_margin": 0.1,
        "upper_margin": 0.9,
        "outcome_lower": 0.0,
        "outcome_upper": 1.0,
        "alpha": 0.05,
        "beta": 0.2,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        _required_bounded_mean_units(**arguments)


def test_bounded_mean_design_returns_positive_units_inside_margins() -> None:
    units = _required_bounded_mean_units(
        planned_mean=0.5,
        lower_margin=0.1,
        upper_margin=0.9,
        outcome_lower=0.0,
        outcome_upper=1.0,
        alpha=0.05,
        beta=0.2,
    )
    assert units > 0


@pytest.mark.parametrize(
    ("planned", "minimum", "message"),
    [
        pytest.param(1.5, 0.8, "planned_unit_pass_probability must be in", id="planned-high"),
        pytest.param(
            float("nan"), 0.8, "planned_unit_pass_probability must be in", id="planned-nan"
        ),
        pytest.param(0.9, 1.0, "minimum_pass_probability must be in", id="minimum-one"),
    ],
)
def test_angular_maximum_design_validates_probabilities(planned, minimum, message) -> None:
    with pytest.raises(ValueError, match=message):
        _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=planned, minimum_pass_probability=minimum
        )


def test_angular_maximum_design_special_cases() -> None:
    assert (
        _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=1.0, minimum_pass_probability=0.8
        )
        is None
    )
    assert (
        _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=0.5, minimum_pass_probability=0.9
        )
        == 0
    )
    assert (
        _maximum_angular_units_for_pass_probability(
            planned_unit_pass_probability=0.9, minimum_pass_probability=0.5
        )
        >= 1
    )


def test_exact_binomial_design_finds_a_powered_minimal_design() -> None:
    design = _exact_binomial_failure_design(
        planned_failure_probability=0.02,
        per_comparison_alpha=0.05,
        per_comparison_power=0.8,
    )
    assert design["sample_size_found"] is True
    assert design["required_independent_outer_clusters_per_stratum"] > 0


def test_exact_binomial_design_can_report_no_feasible_search_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(binomial, "_EXACT_BINOMIAL_SEARCH_LIMIT", 3)
    design = _exact_binomial_failure_design(
        planned_failure_probability=0.4,
        per_comparison_alpha=0.001,
        per_comparison_power=0.99,
    )
    assert design is not None
