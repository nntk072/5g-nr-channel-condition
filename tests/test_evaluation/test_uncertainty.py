"""Group-level conformal uncertainty and selective-risk contracts."""

from __future__ import annotations

import evaluation.uncertainty as uncertainty
import numpy as np
import pytest
from evaluation.uncertainty import evaluate_grouped_regression_uncertainty


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
def test_grouped_uncertainty_rejects_coercive_probability_grids(
    field, value, exception, message
):
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
