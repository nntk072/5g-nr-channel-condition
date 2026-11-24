"""Platt calibration and prior-shift declarations: each frozen contract is enforced."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from evaluation.scoring.prior_shift import validate_prior_shift_spec
from evaluation.scoring.probability_calibration import (
    apply_platt_calibrator,
    fit_group_balanced_platt_calibrator,
    invert_platt_calibrator,
)

GOOD_REPORT = {"schema": "loslab.group_platt_calibrator.v1", "slope": 1.5, "intercept": -0.2}


def _spec(**overrides):
    spec = {
        "schema": "loslab.prior_shift.v1",
        "training_prior_kind": "natural",
        "training_prior_id": "train-prior",
        "target_prior_kind": "natural",
        "target_prior_id": "target-prior",
        "target_positive_prevalence": 0.25,
        "target_prior_evidence_sha256": "a" * 64,
    }
    spec.update(overrides)
    return spec


def test_valid_prior_shift_declaration_is_returned_canonically() -> None:
    assert validate_prior_shift_spec(_spec())["target_positive_prevalence"] == 0.25


@pytest.mark.parametrize(
    ("spec", "error", "message"),
    [
        pytest.param(["not", "a", "mapping"], TypeError, "must be a mapping", id="not-mapping"),
        pytest.param({1: "x"}, ValueError, "field names must be strings", id="non-string-field"),
        pytest.param(
            {"schema": "x"},
            ValueError,
            "fields must match the versioned contract",
            id="missing-fields",
        ),
        pytest.param(
            _spec(schema="loslab.prior_shift.v9"),
            ValueError,
            "prior_shift schema must be",
            id="schema",
        ),
        pytest.param(
            _spec(training_prior_kind="other"),
            ValueError,
            "training_prior_kind must be",
            id="train-kind",
        ),
        pytest.param(
            _spec(target_prior_kind="enriched_stress"),
            ValueError,
            "target_prior_kind must be natural",
            id="target-kind",
        ),
        pytest.param(
            _spec(training_prior_id=" padded "),
            ValueError,
            "training_prior_id must be a nonempty canonical string",
            id="padded-id",
        ),
        pytest.param(
            _spec(target_prior_id=""),
            ValueError,
            "target_prior_id must be a nonempty canonical string",
            id="empty-id",
        ),
        pytest.param(
            _spec(target_positive_prevalence="0.2"),
            ValueError,
            "must be a real number",
            id="prevalence-string",
        ),
        pytest.param(
            _spec(target_positive_prevalence=True),
            ValueError,
            "must be a real number",
            id="prevalence-bool",
        ),
    ],
)
def test_prior_shift_declaration_is_checked_field_by_field(spec, error, message) -> None:
    with pytest.raises(error, match=message):
        validate_prior_shift_spec(spec)


def test_platt_application_validates_probabilities_and_report() -> None:
    with pytest.raises(ValueError, match="probabilities must be a finite one-dimensional vector"):
        apply_platt_calibrator(np.array([0.2, 1.5]), GOOD_REPORT)
    with pytest.raises(ValueError, match="unsupported Platt calibrator report"):
        apply_platt_calibrator(np.array([0.2, 0.7]), {"schema": "other"})
    bad = dict(GOOD_REPORT, slope=0.0)
    with pytest.raises(ValueError, match="finite with positive slope"):
        apply_platt_calibrator(np.array([0.2, 0.7]), bad)
    bad = dict(GOOD_REPORT, intercept=float("nan"))
    with pytest.raises(ValueError, match="finite with positive slope"):
        apply_platt_calibrator(np.array([0.2, 0.7]), bad)


def test_platt_inversion_validates_and_inverts_the_transform() -> None:
    raw = np.array([0.2, 0.7])
    calibrated = apply_platt_calibrator(raw, GOOD_REPORT)
    np.testing.assert_allclose(invert_platt_calibrator(calibrated, GOOD_REPORT), raw, atol=1e-9)
    with pytest.raises(ValueError, match="calibrated probabilities must be finite"):
        invert_platt_calibrator(np.array([1.5]), GOOD_REPORT)
    with pytest.raises(ValueError, match="unsupported Platt calibrator report"):
        invert_platt_calibrator(calibrated, {"schema": "other"})
    with pytest.raises(ValueError, match="finite with positive slope"):
        invert_platt_calibrator(calibrated, dict(GOOD_REPORT, slope=-1.0))


@pytest.mark.parametrize(
    ("raw", "labels", "groups", "message"),
    [
        pytest.param(
            [0.2, 1.5, 0.3, 0.4],
            [0, 1, 0, 1],
            ["a", "a", "b", "b"],
            "finite one-dimensional values in",
            id="raw-range",
        ),
        pytest.param([0.2, 0.3], [0, 1, 1], ["a", "b"], "equal-length vectors", id="length"),
        pytest.param([0.2, 0.3], [0, 1], ["a"], "equal-length vectors", id="group-length"),
        pytest.param([0.2, 0.3], [0, 0], ["a", "b"], "both binary classes", id="one-class"),
        pytest.param([0.2, 0.3], [0, 2], ["a", "b"], "both binary classes", id="non-binary"),
        pytest.param(
            [0.2, 0.3], [0, 1], [" a", "b"], "nonempty canonical strings", id="padded-group"
        ),
        pytest.param(
            [0.2, 0.3], [0, 1], ["a", "a"], "at least two independent groups", id="one-group"
        ),
    ],
)
def test_calibration_fit_rejects_inconsistent_inputs(raw, labels, groups, message) -> None:
    with pytest.raises(ValueError, match=message):
        fit_group_balanced_platt_calibrator(raw, labels, groups)


def test_calibration_fit_returns_a_positive_slope_report() -> None:
    rng = np.random.default_rng(1)
    raw = rng.uniform(0.05, 0.95, size=40)
    labels = (raw + rng.normal(0, 0.1, size=40) > 0.5).astype(int)
    labels[:2] = [0, 1]
    groups = [f"g{index // 4}" for index in range(40)]
    calibrated, report = fit_group_balanced_platt_calibrator(raw, labels, groups)
    assert report["slope"] > 0
    assert calibrated.shape == raw.shape
    assert copy.deepcopy(report)["schema"] == "loslab.group_platt_calibrator.v1"
