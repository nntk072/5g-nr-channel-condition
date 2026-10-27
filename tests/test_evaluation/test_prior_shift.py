from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from evaluation.prior_shift import (
    adjust_probabilities_between_priors,
    adjust_probabilities_for_prior_shift,
    validate_prior_shift_spec,
)
from evaluation.probability_calibration import (
    apply_platt_calibrator,
    fit_group_balanced_platt_calibrator,
    invert_platt_calibrator,
)


def _spec(**overrides):
    return {
        "schema": "loslab.prior_shift.v1",
        "training_prior_kind": "enriched_stress",
        "training_prior_id": "balanced-stress-v1",
        "target_prior_kind": "natural",
        "target_prior_id": "field-prevalence-study-v1",
        "target_positive_prevalence": 0.5,
        "target_prior_evidence_sha256": "a" * 64,
        **overrides,
    }


def test_prior_shift_estimates_equal_weight_per_group_and_adjusts_odds():
    probabilities, report = adjust_probabilities_for_prior_shift(
        np.array([0.5, 0.0, 1.0]),
        np.array([1, 0, 0, 0, 0]),
        np.array(["g1", "g1", "g2", "g2", "g2"]),
        _spec(),
    )

    assert probabilities == pytest.approx([0.75, 0.0, 1.0])
    assert report["calibration_group_weighted_positive_prevalence"] == pytest.approx(0.25)
    assert report["calibration_independent_group_count"] == 2
    assert report["fit_partition"] == "calibration"
    assert report["probability_evaluation_partitions"] == ["validation", "test"]
    assert report["calibration_labels_and_groups_sha256"]


def test_adjustment_is_identity_when_priors_match_and_monotone_otherwise():
    scores = np.linspace(0.0, 1.0, 11)
    assert adjust_probabilities_between_priors(scores, 0.3, 0.3) == pytest.approx(scores)
    shifted = adjust_probabilities_between_priors(scores, 0.2, 0.5)
    assert np.all(np.diff(shifted) >= 0)
    assert np.all((shifted >= 0) & (shifted <= 1))


def test_prior_shift_spec_is_strict_and_rejects_non_natural_target():
    with pytest.raises(ValueError, match="unknown"):
        validate_prior_shift_spec(_spec(extra=True))
    with pytest.raises(ValueError, match="target_prior_kind"):
        validate_prior_shift_spec(_spec(target_prior_kind="enriched_stress"))
    with pytest.raises(ValueError, match="strictly between"):
        validate_prior_shift_spec(_spec(target_positive_prevalence=1.0))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        validate_prior_shift_spec(_spec(target_prior_evidence_sha256="A" * 64))


@pytest.mark.parametrize(
    ("labels", "groups", "message"),
    [
        ([1, 1], ["g1", "g2"], "both classes"),
        ([0, 1], ["g1"], "equal-length"),
        ([0, 1], ["g1", " g2 "], "canonical strings"),
    ],
)
def test_adjustment_requires_valid_training_only_roster(labels, groups, message):
    with pytest.raises(ValueError, match=message):
        adjust_probabilities_for_prior_shift(
            np.array([0.5]), np.asarray(labels), np.asarray(groups), _spec()
        )


def test_prior_shift_declaration_and_report_match_versioned_schemas():
    jsonschema = pytest.importorskip("jsonschema")
    spec_schema = json.loads(
        Path("evaluation/schema/prior-shift.v1.schema.json").read_text(encoding="utf-8")
    )
    report_schema = json.loads(
        Path("evaluation/schema/prior-shift-adjustment.v1.schema.json").read_text(encoding="utf-8")
    )
    calibrator_schema = json.loads(
        Path("evaluation/schema/group-platt-calibrator.v1.schema.json").read_text(encoding="utf-8")
    )
    spec = validate_prior_shift_spec(_spec())
    _, report = adjust_probabilities_for_prior_shift(
        np.array([0.4]), np.array([0, 1]), np.array(["g0", "g1"]), spec
    )

    jsonschema.Draft202012Validator(spec_schema).validate(spec)
    jsonschema.Draft202012Validator(report_schema).validate(report)
    _, calibrator = fit_group_balanced_platt_calibrator(
        [0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1], ["g0", "g1", "g2", "g3"]
    )
    jsonschema.Draft202012Validator(calibrator_schema).validate(calibrator)


def test_group_balanced_platt_uses_separate_calibration_rows_and_inverts_thresholds():
    raw = np.array([0.1, 0.2, 0.3, 0.4, 0.7, 0.9])
    labels = np.array([0, 0, 1, 0, 1, 1])
    groups = np.array(["g0", "g0", "g1", "g2", "g2", "g3"])
    calibrated, report = fit_group_balanced_platt_calibrator(raw, labels, groups)

    assert report["fit_partition"] == "calibration"
    assert report["excluded_partitions"] == ["validation", "test"]
    assert report["calibration_independent_group_count"] == 4
    assert np.all(np.diff(calibrated) > 0)
    assert invert_platt_calibrator(calibrated, report) == pytest.approx(raw, abs=1e-6)
    assert apply_platt_calibrator(raw, report) == pytest.approx(calibrated)


def test_group_balanced_platt_rejects_single_class_or_single_group_calibration():
    with pytest.raises(ValueError, match="both binary classes"):
        fit_group_balanced_platt_calibrator([0.2, 0.3], [0, 0], ["g0", "g1"])
    with pytest.raises(ValueError, match="two independent groups"):
        fit_group_balanced_platt_calibrator([0.2, 0.8], [0, 1], ["g0", "g0"])
