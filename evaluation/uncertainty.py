"""Group-aware uncertainty calibration and selective regression evaluation.

Calibration treats an independent group (for example, a scene or measurement
session) as the exchangeable unit. This avoids pretending correlated samples
within one route are independent calibration examples.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np

_MAX_BOOTSTRAP_RESAMPLING_CELLS = 100_000_000
_MAX_BOOTSTRAP_PEAK_INDEX_CELLS = 2_000_000
_MAX_BOOTSTRAP_DRAWS = 1_000_000


def _real_vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or not raw.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional real vector")
    if raw.dtype.kind not in "iuf":
        if raw.dtype.kind != "O" or any(
            isinstance(item, (bool, np.bool_)) or not isinstance(item, Real)
            for item in raw.tolist()
        ):
            raise TypeError(f"{name} must contain real numeric values")
    try:
        result = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must contain real numeric values representable as float64") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values representable as float64")
    return result


def _canonical_string_vector(value: Sequence[str] | np.ndarray, *, name: str) -> np.ndarray:
    raw = np.asarray(value, dtype=object)
    if raw.ndim != 1 or not raw.size:
        raise ValueError(f"{name} must be a nonempty one-dimensional string vector")
    if any(
        not isinstance(item, str) or not item or item != item.strip() for item in raw.tolist()
    ):
        raise ValueError(f"{name} must contain canonical nonempty strings")
    return raw.astype(str)


def _probability_vector(
    values: Sequence[float], *, name: str, upper_inclusive: bool
) -> tuple[float, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of real numeric values") from error
    if not raw_values:
        raise ValueError(f"{name} must be nonempty")
    normalized: list[float] = []
    for value in raw_values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain real numeric values")
        number = float(value)
        valid_upper = number <= 1.0 if upper_inclusive else number < 1.0
        if not math.isfinite(number) or number <= 0 or not valid_upper:
            interval = "(0, 1]" if upper_inclusive else "(0, 1)"
            raise ValueError(f"{name} values must be finite and in {interval}")
        normalized.append(number)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} values must be unique")
    return tuple(normalized)


def _vectors(
    target: Sequence[float] | np.ndarray,
    prediction: Sequence[float] | np.ndarray,
    uncertainty_scale: Sequence[float] | np.ndarray,
    group: Sequence[str] | np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y = _real_vector(target, name=f"{name} target")
    mu = _real_vector(prediction, name=f"{name} prediction")
    scale = _real_vector(uncertainty_scale, name=f"{name} uncertainty")
    groups = _canonical_string_vector(group, name=f"{name} groups")
    if mu.shape != y.shape or scale.shape != y.shape:
        raise ValueError(f"{name} target, prediction, and uncertainty must be nonempty vectors")
    if groups.shape != y.shape:
        raise ValueError(f"{name} groups must be nonempty and match the vectors")
    if np.any(scale <= 0):
        raise ValueError(f"{name} uncertainty scales must be finite and positive")
    return y, mu, scale, groups


def _conformal_order_statistic(scores: np.ndarray, level: float) -> float:
    if not math.isfinite(level) or not 0.0 < level < 1.0:
        raise ValueError("coverage levels must be finite and in (0, 1)")
    ordered = np.sort(scores)
    rank = math.ceil((ordered.size + 1) * level)
    # Returning infinity is the exact finite-sample result when the requested
    # group coverage cannot be calibrated from the available groups.
    return math.inf if rank > ordered.size else float(ordered[rank - 1])


def _group_bootstrap_rmse(
    errors: np.ndarray, groups: np.ndarray, *, seed: int, draws: int
) -> dict[str, float | int | list[float] | None]:
    unique = np.unique(groups)
    per_group_rmse = np.asarray(
        [np.sqrt(np.mean(errors[groups == group] ** 2)) for group in unique], dtype=np.float64
    )
    if unique.size < 2:
        return {
            "group_macro_rmse_ci95": None,
            "bootstrap_groups": int(unique.size),
            "bootstrap_draws": 0,
            "bootstrap_resampling_cells": 0,
            "bootstrap_peak_index_cells": 0,
        }
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 100:
        raise ValueError("draws must be an integer >= 100")
    if draws > _MAX_BOOTSTRAP_DRAWS:
        raise ValueError(f"draws exceeds the {_MAX_BOOTSTRAP_DRAWS}-draw ceiling")
    if unique.size > _MAX_BOOTSTRAP_PEAK_INDEX_CELLS:
        raise ValueError(
            "group count exceeds the "
            f"{_MAX_BOOTSTRAP_PEAK_INDEX_CELLS}-cell bootstrap peak ceiling"
        )
    resampling_cells = int(draws * unique.size)
    if resampling_cells > _MAX_BOOTSTRAP_RESAMPLING_CELLS:
        raise ValueError(
            "group bootstrap resampling work exceeds the "
            f"{_MAX_BOOTSTRAP_RESAMPLING_CELLS}-cell ceiling"
        )
    rng = np.random.default_rng(seed)
    chunk_draws = max(1, _MAX_BOOTSTRAP_PEAK_INDEX_CELLS // unique.size)
    estimates = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, chunk_draws):
        stop = min(draws, start + chunk_draws)
        indices = rng.integers(0, unique.size, size=(stop - start, unique.size))
        estimates[start:stop] = per_group_rmse[indices].mean(axis=1)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "group_macro_rmse_ci95": [float(low), float(high)],
        "bootstrap_groups": int(unique.size),
        "bootstrap_draws": draws,
        "bootstrap_resampling_cells": resampling_cells,
        "bootstrap_peak_index_cells": min(draws, chunk_draws) * int(unique.size),
    }


def evaluate_grouped_regression_uncertainty(
    calibration_target: Sequence[float] | np.ndarray,
    calibration_prediction: Sequence[float] | np.ndarray,
    calibration_scale: Sequence[float] | np.ndarray,
    calibration_group: Sequence[str] | np.ndarray,
    test_target: Sequence[float] | np.ndarray,
    test_prediction: Sequence[float] | np.ndarray,
    test_scale: Sequence[float] | np.ndarray,
    test_group: Sequence[str] | np.ndarray,
    *,
    coverage_levels: Sequence[float] = (0.5, 0.8, 0.9, 0.95),
    risk_coverages: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
    test_stratum: Sequence[str] | np.ndarray | None = None,
    seed: int = 42,
    bootstrap_draws: int = 2000,
) -> dict[str, object]:
    """Calibrate scale on held-out groups and evaluate interval/selective risk.

    For each nominal level, split conformal calibration uses the maximum
    standardized absolute residual within each calibration group. The resulting
    interval is ``prediction +/- q * uncertainty_scale``. Finite-sample
    group-level validity requires exchangeable independent calibration/test
    groups; group-wise maxima can make intervals unbounded with few groups.
    """
    cal_y, cal_mu, cal_scale, cal_groups = _vectors(
        calibration_target,
        calibration_prediction,
        calibration_scale,
        calibration_group,
        name="calibration",
    )
    test_y, test_mu, test_uncertainty, test_groups = _vectors(
        test_target, test_prediction, test_scale, test_group, name="test"
    )
    overlap = set(cal_groups).intersection(test_groups)
    if overlap:
        raise ValueError(f"calibration and test groups overlap: {sorted(overlap)}")
    levels = _probability_vector(coverage_levels, name="coverage_levels", upper_inclusive=False)
    risks = _probability_vector(risk_coverages, name="risk_coverages", upper_inclusive=True)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(bootstrap_draws, bool) or not isinstance(bootstrap_draws, int):
        raise TypeError("bootstrap_draws must be an integer")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be >= 100")
    strata = (
        None
        if test_stratum is None
        else _canonical_string_vector(test_stratum, name="test_stratum")
    )
    if strata is not None and strata.shape != test_y.shape:
        raise ValueError("test_stratum must contain nonempty values matching test rows")

    cal_error = np.abs(cal_y - cal_mu) / cal_scale
    calibration_group_scores = np.asarray(
        [np.max(cal_error[cal_groups == group]) for group in np.unique(cal_groups)],
        dtype=np.float64,
    )
    error = test_y - test_mu
    abs_error = np.abs(error)
    group_rmse = {
        str(group): float(np.sqrt(np.mean(error[test_groups == group] ** 2)))
        for group in np.unique(test_groups)
    }
    intervals: dict[str, object] = {}
    stratum_intervals: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for level in levels:
        multiplier = _conformal_order_statistic(calibration_group_scores, level)
        if math.isinf(multiplier):
            sample_coverage = group_coverage = 1.0
            mean_width: float | None = None
            multiplier_value: float | None = None
        else:
            covered = abs_error <= multiplier * test_uncertainty
            sample_coverage = float(np.mean(covered))
            group_coverage = float(
                np.mean([np.all(covered[test_groups == group]) for group in np.unique(test_groups)])
            )
            mean_width = float(np.mean(2.0 * multiplier * test_uncertainty))
            multiplier_value = multiplier
        intervals[f"{level:.6g}"] = {
            "scale_multiplier": multiplier_value,
            "finite_sample_unbounded": math.isinf(multiplier),
            "test_sample_coverage": sample_coverage,
            "test_group_simultaneous_coverage": group_coverage,
            "mean_interval_width": mean_width,
        }
        if strata is not None:
            per_stratum: dict[str, dict[str, float | int | None]] = {}
            for stratum in np.unique(strata):
                take = strata == stratum
                if math.isinf(multiplier):
                    stratum_sample_coverage = stratum_group_coverage = 1.0
                    stratum_width = None
                else:
                    covered_stratum = abs_error[take] <= multiplier * test_uncertainty[take]
                    groups_in_stratum = np.unique(test_groups[take])
                    stratum_sample_coverage = float(np.mean(covered_stratum))
                    stratum_group_coverage = float(
                        np.mean(
                            [
                                np.all(covered_stratum[test_groups[take] == group])
                                for group in groups_in_stratum
                            ]
                        )
                    )
                    stratum_width = float(np.mean(2.0 * multiplier * test_uncertainty[take]))
                per_stratum[str(stratum)] = {
                    "sample_count": int(np.count_nonzero(take)),
                    "group_count": int(np.unique(test_groups[take]).size),
                    "sample_coverage": stratum_sample_coverage,
                    "group_simultaneous_coverage": stratum_group_coverage,
                    "mean_interval_width": stratum_width,
                }
            stratum_intervals[f"{level:.6g}"] = per_stratum

    # Selective prediction accepts the least uncertain observations first.
    # Stable sorting makes ties deterministic; results remain descriptive when
    # multiple samples belong to the same scene or trajectory.
    order = np.argsort(test_uncertainty, kind="stable")
    risk_curve: list[dict[str, float | int]] = []
    for fraction in risks:
        count = max(1, math.ceil(fraction * test_y.size))
        selected_error = error[order[:count]]
        risk_curve.append(
            {
                "requested_coverage": fraction,
                "actual_coverage": count / test_y.size,
                "accepted_samples": count,
                "mae": float(np.mean(np.abs(selected_error))),
                "rmse": float(np.sqrt(np.mean(selected_error**2))),
            }
        )

    bootstrap = _group_bootstrap_rmse(error, test_groups, seed=seed, draws=bootstrap_draws)
    return {
        "calibration_group_count": int(np.unique(cal_groups).size),
        "calibration_sample_count": int(cal_y.size),
        "test_group_count": int(np.unique(test_groups).size),
        "test_sample_count": int(test_y.size),
        "calibration_group_scores_max_standardized_error": calibration_group_scores.tolist(),
        "error_metrics": {
            "mae": float(np.mean(abs_error)),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "group_macro_rmse": float(np.mean(list(group_rmse.values()))),
            "group_rmse": group_rmse,
            **bootstrap,
        },
        "group_conformal_intervals": intervals,
        "test_interval_metrics_by_stratum": stratum_intervals,
        "risk_coverage": risk_curve,
        "resampling_unit": "independent test group",
        "interpretation": (
            "Conformal guarantees require exchangeable independent groups. With too few calibration "
            "groups, requested finite-sample intervals are unbounded; sample-level rows are not "
            "treated as independent calibration units. Risk-coverage is descriptive."
        ),
    }
