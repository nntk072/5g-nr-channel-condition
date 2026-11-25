"""Group-level conformal uncertainty and selective-risk contracts."""

from __future__ import annotations

import evaluation.scoring.uncertainty as uncertainty
import numpy as np
import pytest
from evaluation.scoring.uncertainty import evaluate_grouped_regression_uncertainty


def _evaluate(*, calibration_groups: tuple[str, ...] = ("c1", "c2", "c3", "c4")):
    calibration_group = np.repeat(calibration_groups, 2)
    calibration_target = np.arange(calibration_group.size, dtype=float)
    calibration_prediction = calibration_target + np.tile([0.1, -0.2], len(calibration_groups))
    calibration_scale = np.ones(calibration_group.size)
    return evaluate_grouped_regression_uncertainty(
        calibration_target,
        calibration_prediction,
        calibration_scale,
        calibration_group,
        np.asarray([10.0, 10.1, 20.0, 22.0]),
        np.asarray([10.1, 10.2, 20.4, 22.8]),
        np.asarray([0.2, 0.4, 0.5, 1.0]),
        np.asarray(["test-a", "test-a", "test-b", "test-b"]),
        seed=9,
        bootstrap_draws=200,
    )


def test_group_conformal_uses_calibration_groups_and_reports_group_coverage():
    report = _evaluate()
    assert report["calibration_group_count"] == 4
    assert report["test_group_count"] == 2
    interval = report["group_conformal_intervals"]["0.5"]
    assert interval["scale_multiplier"] == pytest.approx(0.2)
    assert interval["mean_interval_width"] > 0
    assert 0 <= interval["test_group_simultaneous_coverage"] <= 1


def test_small_calibration_group_count_returns_unbounded_finite_sample_interval():
    report = _evaluate(calibration_groups=("only-one-calibration-scene",))
    interval = report["group_conformal_intervals"]["0.9"]
    assert interval["finite_sample_unbounded"] is True
    assert interval["scale_multiplier"] is None
    assert interval["mean_interval_width"] is None


def test_calibration_and_test_scene_overlap_is_rejected():
    with pytest.raises(ValueError, match="groups overlap"):
        evaluate_grouped_regression_uncertainty(
            [0, 1],
            [0, 1],
            [1, 1],
            ["scene-a", "scene-b"],
            [0],
            [0],
            [1],
            ["scene-a"],
        )


@pytest.mark.parametrize(
    "scale",
    [[0.0, 1.0], [1.0, float("nan")]],
)
def test_invalid_uncertainty_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="uncertainty"):
        evaluate_grouped_regression_uncertainty(
            [0, 1],
            [0, 1],
            scale,
            ["c1", "c2"],
            [0],
            [0],
            [1],
            ["t1"],
        )


def test_risk_coverage_accepts_low_uncertainty_first_and_bootstraps_groups():
    report = _evaluate()
    curve = report["risk_coverage"]
    assert [point["actual_coverage"] for point in curve] == [0.25, 0.5, 0.75, 1.0]
    assert curve[0]["mae"] <= curve[-1]["mae"]
    assert report["error_metrics"]["bootstrap_groups"] == 2
    assert report["error_metrics"]["bootstrap_draws"] == 200


def test_group_risk_coverage_selects_whole_groups_by_group_mean_uncertainty():
    report = evaluate_grouped_regression_uncertainty(
        [0.0, 1.0, 2.0, 3.0],
        [0.1, 0.9, 2.1, 2.9],
        [1.0, 1.0, 1.0, 1.0],
        ["cal-a", "cal-a", "cal-b", "cal-b"],
        [0.0, 0.0, 100.0],
        [0.0, 0.0, 1.0],
        [0.1, 0.1, 0.2],
        ["test-a", "test-a", "test-b"],
        risk_coverages=(0.5, 1.0),
        bootstrap_draws=100,
    )

    half, full = report["group_risk_coverage"]
    assert half["actual_group_coverage"] == 0.5
    assert half["accepted_groups"] == 1
    assert half["accepted_samples"] == 2
    assert half["group_macro_mae"] == 0.0
    assert full["actual_group_coverage"] == 1.0
    assert full["accepted_groups"] == 2
    assert full["accepted_samples"] == 3
    assert full["group_macro_mae"] == pytest.approx(49.5)


def test_interval_coverage_is_reported_by_identifiability_stratum():
    report = evaluate_grouped_regression_uncertainty(
        [0.0, 1.0, 2.0, 3.0],
        [0.1, 0.8, 2.2, 2.7],
        [1.0] * 4,
        ["c1", "c1", "c2", "c2"],
        [4.0, 5.0, 6.0, 7.0],
        [4.1, 5.1, 7.0, 9.0],
        [1.0] * 4,
        ["t1", "t1", "t2", "t2"],
        test_stratum=["resolved", "resolved", "ambiguous", "ambiguous"],
    )
    by_stratum = report["test_interval_metrics_by_stratum"]["0.5"]
    assert by_stratum["resolved"]["sample_count"] == 2
    assert by_stratum["ambiguous"]["group_count"] == 1
    assert 0 <= by_stratum["ambiguous"]["sample_coverage"] <= 1


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("calibration_target", [1 + 2j, 3 + 4j], TypeError, "real numeric"),
        ("test_prediction", ["1.0", "2.0"], TypeError, "real numeric"),
        ("calibration_scale", [True, False], TypeError, "real numeric"),
        ("test_group", [1, 2], ValueError, "canonical nonempty strings"),
        ("test_group", ["test-a", " padded"], ValueError, "canonical nonempty strings"),
        ("test_stratum", [1, 2], ValueError, "canonical nonempty strings"),
    ],
)
def test_grouped_uncertainty_rejects_coercive_input_domains(field, value, exception, message):
    inputs = {
        "calibration_target": [0.0, 1.0],
        "calibration_prediction": [0.1, 0.9],
        "calibration_scale": [1.0, 1.0],
        "calibration_group": ["cal-a", "cal-b"],
        "test_target": [2.0, 3.0],
        "test_prediction": [2.1, 2.9],
        "test_scale": [1.0, 1.0],
        "test_group": ["test-a", "test-b"],
        "bootstrap_draws": 100,
    }
    inputs[field] = value
    with pytest.raises(exception, match=message):
        evaluate_grouped_regression_uncertainty(**inputs)


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("coverage_levels", ("0.8",), TypeError, "real numeric"),
        ("coverage_levels", (True,), TypeError, "real numeric"),
        ("risk_coverages", (1 + 0j,), TypeError, "real numeric"),
    ],
)
def test_grouped_uncertainty_rejects_coercive_probability_grids(field, value, exception, message):
    inputs = {
        "calibration_target": [0.0, 1.0],
        "calibration_prediction": [0.1, 0.9],
        "calibration_scale": [1.0, 1.0],
        "calibration_group": ["cal-a", "cal-b"],
        "test_target": [2.0, 3.0],
        "test_prediction": [2.1, 2.9],
        "test_scale": [1.0, 1.0],
        "test_group": ["test-a", "test-b"],
        "bootstrap_draws": 100,
    }
    inputs[field] = value
    with pytest.raises(exception, match=message):
        evaluate_grouped_regression_uncertainty(**inputs)


def test_group_bootstrap_chunks_resampling_indices_at_declared_peak_limit(monkeypatch):
    monkeypatch.setattr(uncertainty, "_MAX_BOOTSTRAP_PEAK_INDEX_CELLS", 8)
    report = _evaluate()

    bootstrap = report["error_metrics"]
    assert bootstrap["bootstrap_resampling_cells"] == 400
    assert bootstrap["bootstrap_peak_index_cells"] <= 8
    per_group_rmse = np.asarray([0.1, np.sqrt((0.4**2 + 0.8**2) / 2)])
    indices = np.random.default_rng(9).integers(0, 2, size=(200, 2))
    expected_ci = np.quantile(per_group_rmse[indices].mean(axis=1), [0.025, 0.975])
    np.testing.assert_allclose(bootstrap["group_macro_rmse_ci95"], expected_ci)


def test_group_bootstrap_rejects_excessive_work_before_allocating_indices(monkeypatch):
    monkeypatch.setattr(uncertainty, "_MAX_BOOTSTRAP_RESAMPLING_CELLS", 3)
    with pytest.raises(ValueError, match="resampling work"):
        _evaluate()


def test_group_bootstrap_rejects_excessive_draw_count_before_allocation(monkeypatch):
    monkeypatch.setattr(uncertainty, "_MAX_BOOTSTRAP_DRAWS", 100)
    with pytest.raises(ValueError, match="draw ceiling"):
        _evaluate()


@pytest.mark.parametrize("value", [[], [[1.0]], [float("inf")]])
def test_real_vector_rejects_empty_multidimensional_and_nonfinite_inputs(value):
    with pytest.raises(ValueError):
        uncertainty._real_vector(value, name="probe")


def test_real_vector_rejects_values_outside_float64_range():
    with pytest.raises(TypeError, match="representable as float64"):
        uncertainty._real_vector([10**1000], name="probe")


@pytest.mark.parametrize(
    ("prediction", "scale", "groups", "message"),
    [
        ([0.0], [1.0, 1.0], ["a", "b"], "target, prediction, and uncertainty"),
        ([0.0, 1.0], [1.0, 1.0], ["a"], "groups must be nonempty and match"),
    ],
)
def test_vectors_reject_mismatched_prediction_scale_or_group_shapes(
    prediction, scale, groups, message
):
    with pytest.raises(ValueError, match=message):
        uncertainty._vectors([0.0, 1.0], prediction, scale, groups, name="probe")


@pytest.mark.parametrize("value", [[], [["a"]], [""]])
def test_canonical_string_vector_rejects_empty_or_noncanonical_shape(value):
    with pytest.raises(ValueError):
        uncertainty._canonical_string_vector(value, name="probe")


@pytest.mark.parametrize(
    ("value", "exception", "message"),
    [
        (None, TypeError, "must be a sequence"),
        ((), ValueError, "must be nonempty"),
        ((0.0,), ValueError, "finite and in"),
        ((1.0,), ValueError, "finite and in"),
        ((0.5, 0.5), ValueError, "must be unique"),
    ],
)
def test_probability_vector_rejects_invalid_bounds_and_duplicates(value, exception, message):
    with pytest.raises(exception, match=message):
        uncertainty._probability_vector(value, name="probe", upper_inclusive=False)


def test_conformal_order_statistic_validates_level_and_unbounded_rank():
    with pytest.raises(ValueError, match="coverage levels"):
        uncertainty._conformal_order_statistic(np.asarray([0.2]), 1.0)
    assert uncertainty._conformal_order_statistic(np.asarray([0.2]), 0.9) == float("inf")


def test_group_bootstrap_single_group_is_unassessable_without_draws():
    result = uncertainty._group_bootstrap_rmse(
        np.asarray([0.2, 0.4]), np.asarray(["one", "one"]), seed=0, draws=1
    )
    assert result["group_macro_rmse_ci95"] is None
    assert result["bootstrap_draws"] == 0


@pytest.mark.parametrize("draws", [True, 99])
def test_group_bootstrap_rejects_invalid_draw_count(draws):
    with pytest.raises(ValueError, match="integer >= 100"):
        uncertainty._group_bootstrap_rmse(
            np.asarray([0.2, 0.4]), np.asarray(["one", "two"]), seed=0, draws=draws
        )


def test_group_bootstrap_rejects_peak_index_limit(monkeypatch):
    monkeypatch.setattr(uncertainty, "_MAX_BOOTSTRAP_PEAK_INDEX_CELLS", 1)
    with pytest.raises(ValueError, match="peak ceiling"):
        _evaluate()


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("seed", True, ValueError, "seed must be"),
        ("seed", -1, ValueError, "seed must be"),
        ("group_unit", "unknown", ValueError, "group_unit must be"),
        ("bootstrap_draws", True, TypeError, "bootstrap_draws must be"),
        ("bootstrap_draws", 99, ValueError, "bootstrap_draws must be"),
        ("test_stratum", ["only-one"], ValueError, "test_stratum must"),
    ],
)
def test_grouped_uncertainty_rejects_invalid_evaluation_options(field, value, exception, message):
    options = {
        "calibration_target": [0.0, 1.0],
        "calibration_prediction": [0.0, 1.0],
        "calibration_scale": [1.0, 1.0],
        "calibration_group": ["cal-a", "cal-b"],
        "test_target": [2.0, 3.0],
        "test_prediction": [2.0, 3.0],
        "test_scale": [1.0, 1.0],
        "test_group": ["test-a", "test-b"],
        "bootstrap_draws": 100,
    }
    options[field] = value
    with pytest.raises(exception, match=message):
        evaluate_grouped_regression_uncertainty(**options)
