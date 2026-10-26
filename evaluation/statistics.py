"""Paired, predeclared-seed method comparisons adapted from Aalto thesis workflow."""

from __future__ import annotations

import itertools
import math

import numpy as np


def paired_seed_delta(
    rows: list[dict],
    control: str,
    treatment: str,
    *,
    metric: str = "auroc",
    seed: int = 1,
    draws: int = 5000,
) -> dict:
    """Compare matched seeds with identical train/val/test and dataset identity.

    Exact sign-flip test is used with <=20 seed pairs. Mean-difference bootstrap
    is descriptive and does not replace independent-scene inference.
    """
    by_method = {}
    for method in (control, treatment):
        matching = {}
        for r in rows:
            if r["method"] == method:
                s = int(r["seed"])
                if s in matching:
                    raise ValueError(f"duplicate seed {s}, method {method}")
                matching[s] = r
        by_method[method] = matching
    a, b = by_method[control], by_method[treatment]
    if set(a) != set(b) or not a:
        raise ValueError("Control/treatment seeds differ or empty")
    deltas = []
    for s in sorted(a):
        # Sizes/counts are necessary but not sufficient; exact split IDs should be
        # verified in saved prediction npz files during independent review.
        if a[s]["split"] != b[s]["split"]:
            raise ValueError(f"seed {s} split does not match across methods")
        deltas.append(float(b[s]["metrics"][metric] - a[s]["metrics"][metric]))
    differences = np.asarray(deltas, dtype=np.float64)
    if not np.all(np.isfinite(differences)):
        raise ValueError("Nonfinite metric delta")
    rng = np.random.default_rng(seed)
    boot = rng.choice(differences, (draws, len(differences)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    obs = abs(float(differences.mean()))
    if len(differences) <= 20:
        # exact randomization: sign flips are valid under a symmetric null
        # difference distribution; do not overinterpret with few seeds.
        abs_means = (
            abs(float(np.mean(differences * np.array(signs))))
            for signs in itertools.product([-1, 1], repeat=len(differences))
        )
        p = sum(v >= obs - 1e-12 for v in abs_means) / (2 ** len(differences))
        method_name = "exact paired sign-flip"
    else:
        signs = rng.choice([-1.0, 1.0], size=(20000, len(differences)))
        p = float(np.mean(np.abs(np.mean(signs * differences, axis=1)) >= obs - 1e-12))
        method_name = "Monte Carlo paired sign-flip (20000 draws)"
    return {
        "control": control,
        "treatment": treatment,
        "metric": metric,
        "paired_seeds": list(sorted(a)),
        "treatment_minus_control": list(map(float, differences)),
        "mean_delta": float(differences.mean()),
        "mean_delta_bootstrap_ci95": [float(lo), float(hi)],
        "two_sided_sign_flip_p_value": float(p),
        "test": method_name,
        "warning": (
            "Fewer than six seeds: even five consistent wins give p>=0.0625."
            if len(differences) < 6
            else "Seed-level inference cannot replace independent-scene inference."
        ),
    }


def paired_binary_error_comparison(
    control_errors: list[bool],
    treatment_errors: list[bool],
    *,
    seed: int = 1,
    draws: int = 5000,
    control_bits_per_success: float = 1.0,
    treatment_bits_per_success: float = 1.0,
) -> dict:
    """Compare paired per-codeword failures using a bootstrap and exact McNemar test.

    Pair order must be the same in both conditions. The result describes this
    paired simulation only; it does not replace independent-scene inference.
    """
    control = np.asarray(control_errors)
    treatment = np.asarray(treatment_errors)
    if control.ndim != 1 or treatment.ndim != 1:
        raise ValueError("paired error masks must be one-dimensional")
    if control.size == 0 or control.shape != treatment.shape:
        raise ValueError("paired error masks must be nonempty and have equal length")
    if not np.isin(control, [False, True]).all() or not np.isin(treatment, [False, True]).all():
        raise ValueError("paired error masks must contain only boolean outcomes")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    for name, value in (
        ("control_bits_per_success", control_bits_per_success),
        ("treatment_bits_per_success", treatment_bits_per_success),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    control = control.astype(bool, copy=False)
    treatment = treatment.astype(bool, copy=False)
    delta = treatment.astype(np.float64) - control.astype(np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(delta, size=(draws, delta.size), replace=True).mean(axis=1)
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
    control_goodput = (~control) * control_bits_per_success
    treatment_goodput = (~treatment) * treatment_bits_per_success
    goodput_delta = treatment_goodput - control_goodput
    goodput_bootstrap = rng.choice(goodput_delta, size=(draws, delta.size), replace=True).mean(
        axis=1
    )
    goodput_ci_low, goodput_ci_high = np.percentile(goodput_bootstrap, [2.5, 97.5])

    control_only = int(np.count_nonzero(control & ~treatment))
    treatment_only = int(np.count_nonzero(~control & treatment))
    discordant = control_only + treatment_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail_count = min(control_only, treatment_only)
        log_terms = [
            math.lgamma(discordant + 1)
            - math.lgamma(k + 1)
            - math.lgamma(discordant - k + 1)
            - discordant * math.log(2.0)
            for k in range(tail_count + 1)
        ]
        max_log = max(log_terms)
        one_tail = math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
        p_value = min(1.0, 2.0 * one_tail)

    return {
        "codewords": int(control.size),
        "control_errors": int(np.count_nonzero(control)),
        "treatment_errors": int(np.count_nonzero(treatment)),
        "both_error": int(np.count_nonzero(control & treatment)),
        "control_only_error": control_only,
        "treatment_only_error": treatment_only,
        "both_success": int(np.count_nonzero(~control & ~treatment)),
        "treatment_minus_control_bler": float(delta.mean()),
        "paired_bootstrap_ci95": [float(ci_low), float(ci_high)],
        "control_goodput_bits_per_codeword": float(control_goodput.mean()),
        "treatment_goodput_bits_per_codeword": float(treatment_goodput.mean()),
        "treatment_minus_control_goodput_bits_per_codeword": float(goodput_delta.mean()),
        "paired_goodput_bootstrap_ci95": [float(goodput_ci_low), float(goodput_ci_high)],
        "discordant_pairs": discordant,
        "mcnemar_exact_two_sided_p_value": p_value,
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "warning": "Paired codeword inference is conditional on this simulation; it is not scene-level inference.",
    }


def paired_cluster_binary_error_comparison(
    control_errors: list[bool],
    treatment_errors: list[bool],
    groups: list[str],
    *,
    seed: int = 1,
    draws: int = 5000,
) -> dict:
    """Compare paired binary outcomes by resampling independent clusters.

    The point estimate is the pooled treatment-minus-control error-rate
    difference. Each bootstrap replicate samples whole groups with replacement
    and retains all paired trials within each sampled group. This prevents
    repeated codewords or receiver views from being treated as independent
    evidence. The interval remains conditional on the supplied group population
    and does not establish generalization to unrepresented sites or families.
    """
    control = np.asarray(control_errors)
    treatment = np.asarray(treatment_errors)
    group_ids = np.asarray(groups)
    if control.ndim != 1 or treatment.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("paired error masks and groups must be one-dimensional")
    if control.size == 0 or control.shape != treatment.shape or control.shape != group_ids.shape:
        raise ValueError("paired error masks and groups must be nonempty and equally sized")
    if not all(isinstance(value, (bool, np.bool_)) for value in control.tolist()) or not all(
        isinstance(value, (bool, np.bool_)) for value in treatment.tolist()
    ):
        raise ValueError("paired error masks must contain only boolean outcomes")
    if not all(isinstance(group, str) and group.strip() for group in group_ids.tolist()):
        raise ValueError("group IDs must be non-empty strings")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")

    control = control.astype(bool, copy=False)
    treatment = treatment.astype(bool, copy=False)
    group_ids = group_ids.astype(str, copy=False)
    unique_groups, inverse = np.unique(group_ids, return_inverse=True)
    if unique_groups.size < 2:
        raise ValueError("at least two independent groups are required for cluster bootstrap")

    control_errors_by_group = np.bincount(inverse, weights=control.astype(float))
    treatment_errors_by_group = np.bincount(inverse, weights=treatment.astype(float))
    trials_by_group = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(seed)
    sampled_groups = rng.integers(0, unique_groups.size, size=(draws, unique_groups.size))
    sampled_trials = trials_by_group[sampled_groups].sum(axis=1)
    control_rates = control_errors_by_group[sampled_groups].sum(axis=1) / sampled_trials
    treatment_rates = treatment_errors_by_group[sampled_groups].sum(axis=1) / sampled_trials
    delta_bootstrap = treatment_rates - control_rates
    ci_low, ci_high = np.percentile(delta_bootstrap, [2.5, 97.5])

    control_rate = float(control.mean())
    treatment_rate = float(treatment.mean())
    return {
        "independent_groups": int(unique_groups.size),
        "paired_trials": int(control.size),
        "control_error_rate": control_rate,
        "treatment_error_rate": treatment_rate,
        "treatment_minus_control_error_rate": treatment_rate - control_rate,
        "cluster_bootstrap_ci95": [float(ci_low), float(ci_high)],
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "sampling_unit": "independent group; all paired trials within each sampled group",
        "warning": (
            "Cluster bootstrap inference is conditional on the supplied groups; "
            "it does not replace independent-site or external-data validation."
        ),
    }


def paired_cluster_binary_error_family_comparison(
    control_errors_by_point: list[list[bool]],
    treatment_errors_by_point: list[list[bool]],
    groups_by_point: list[list[str]],
    *,
    seed: int = 1,
    draws: int = 5000,
) -> list[dict]:
    """Compare paired BLER over an SNR family with shared cluster resampling.

    A single resampled cluster roster is used for every SNR in each bootstrap
    draw. The max absolute centered deviation then yields a simultaneous
    familywise band; pointwise percentile intervals are retained as diagnostics.
    """
    point_count = len(control_errors_by_point)
    if (
        point_count == 0
        or len(treatment_errors_by_point) != point_count
        or len(groups_by_point) != point_count
    ):
        raise ValueError("SNR family inputs must have the same nonzero point count")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")

    control_counts: list[np.ndarray] = []
    treatment_counts: list[np.ndarray] = []
    trial_counts: list[np.ndarray] = []
    point_estimates: list[float] = []
    expected_groups: set[str] | None = None
    for control_values, treatment_values, group_values in zip(
        control_errors_by_point, treatment_errors_by_point, groups_by_point, strict=True
    ):
        control = np.asarray(control_values)
        treatment = np.asarray(treatment_values)
        groups = np.asarray(group_values)
        if (
            control.ndim != 1
            or treatment.ndim != 1
            or groups.ndim != 1
            or control.size == 0
            or control.shape != treatment.shape
            or control.shape != groups.shape
        ):
            raise ValueError("each SNR point needs nonempty aligned outcomes and group IDs")
        if not all(isinstance(value, (bool, np.bool_)) for value in control.tolist()) or not all(
            isinstance(value, (bool, np.bool_)) for value in treatment.tolist()
        ):
            raise ValueError("error masks must contain only boolean outcomes")
        if not all(isinstance(value, str) and value.strip() for value in groups.tolist()):
            raise ValueError("group IDs must be non-empty strings")
        unique_groups = set(groups.tolist())
        if expected_groups is None:
            expected_groups = unique_groups
        elif unique_groups != expected_groups:
            raise ValueError("every SNR point must contain the same independent cluster roster")
        ordered_groups = sorted(unique_groups)
        group_indices = {group: index for index, group in enumerate(ordered_groups)}
        inverse = np.fromiter(
            (group_indices[group] for group in groups.tolist()),
            dtype=np.int64,
            count=groups.size,
        )
        control_bool = control.astype(bool, copy=False)
        treatment_bool = treatment.astype(bool, copy=False)
        size_by_group = np.bincount(inverse, minlength=len(ordered_groups)).astype(np.float64)
        control_counts.append(
            np.bincount(inverse, weights=control_bool.astype(np.float64), minlength=len(ordered_groups))
        )
        treatment_counts.append(
            np.bincount(inverse, weights=treatment_bool.astype(np.float64), minlength=len(ordered_groups))
        )
        trial_counts.append(size_by_group)
        point_estimates.append(float(treatment_bool.mean() - control_bool.mean()))

    assert expected_groups is not None
    if len(expected_groups) < 2:
        raise ValueError("at least two independent groups are required for cluster bootstrap")
    if len(expected_groups) * point_count > 5_000_000:
        raise ValueError("independent-group-by-SNR grid exceeds the 5,000,000-cell memory limit")
    if draws * point_count > 5_000_000:
        raise ValueError("bootstrap draw-by-SNR grid exceeds the 5,000,000-cell memory limit")

    control_matrix = np.stack(control_counts, axis=1)
    treatment_matrix = np.stack(treatment_counts, axis=1)
    trials_matrix = np.stack(trial_counts, axis=1)
    estimates = np.asarray(point_estimates, dtype=np.float64)
    rng = np.random.default_rng(seed)
    group_count = len(expected_groups)
    max_deviation = np.empty(draws, dtype=np.float64)
    pointwise_draws = np.empty((draws, point_count), dtype=np.float64)
    max_cells = 1_000_000
    chunk_size = max(1, min(draws, max_cells // group_count, max_cells // point_count))
    for start in range(0, draws, chunk_size):
        stop = min(draws, start + chunk_size)
        sampled = rng.integers(0, group_count, size=(stop - start, group_count))
        sampled_count = stop - start
        multiplicities = np.zeros((sampled_count, group_count), dtype=np.float64)
        np.add.at(
            multiplicities,
            (np.repeat(np.arange(sampled_count), group_count), sampled.ravel()),
            1.0,
        )
        control_rate = (multiplicities @ control_matrix) / (multiplicities @ trials_matrix)
        treatment_rate = (multiplicities @ treatment_matrix) / (multiplicities @ trials_matrix)
        deltas = treatment_rate - control_rate
        pointwise_draws[start:stop] = deltas
        max_deviation[start:stop] = np.max(np.abs(deltas - estimates), axis=1)

    critical_value = float(np.quantile(max_deviation, 0.95))
    output: list[dict] = []
    for point_index, estimate in enumerate(estimates.tolist()):
        pointwise_low, pointwise_high = np.quantile(
            pointwise_draws[:, point_index], [0.025, 0.975]
        )
        output.append(
            {
                "independent_groups": group_count,
                "paired_trials": int(trials_matrix[:, point_index].sum()),
                "treatment_minus_control_error_rate": estimate,
                "cluster_bootstrap_ci95": [float(pointwise_low), float(pointwise_high)],
                "familywise_cluster_bootstrap_ci95": [
                    max(-1.0, estimate - critical_value),
                    min(1.0, estimate + critical_value),
                ],
                "familywise_method": "shared-cluster max-absolute-centered-deviation bootstrap",
                "familywise_confidence_level": 0.95,
                "familywise_snr_points": point_count,
                "familywise_critical_deviation": critical_value,
                "bootstrap_seed": seed,
                "bootstrap_draws": draws,
                "sampling_unit": "independent batch seed; codewords nested",
                "warning": (
                    "Conditional on supplied synthetic batches; this does not establish "
                    "independent-scene, site, or measured-channel generalization."
                ),
            }
        )
    return output


def cluster_binary_error_rate_interval(
    errors: list[bool],
    groups: list[str],
    *,
    seed: int = 1,
    draws: int = 5000,
) -> dict:
    """Estimate a binary error rate and percentile interval by resampling clusters.

    The point estimate is pooled across the supplied trials. Each bootstrap draw
    samples whole groups with replacement and retains every trial in each group,
    so nested codewords are not treated as independent units. This interval is
    conditional on the supplied group population and cannot establish
    generalization to unrepresented scenes, sites, or channel families.
    """
    outcome = np.asarray(errors)
    group_ids = np.asarray(groups)
    if outcome.ndim != 1 or group_ids.ndim != 1:
        raise ValueError("error outcomes and groups must be one-dimensional")
    if outcome.size == 0 or outcome.shape != group_ids.shape:
        raise ValueError("error outcomes and groups must be nonempty and equally sized")
    if not all(isinstance(value, (bool, np.bool_)) for value in outcome.tolist()):
        raise ValueError("error outcomes must contain only boolean values")
    if not all(isinstance(group, str) and group.strip() for group in group_ids.tolist()):
        raise ValueError("group IDs must be non-empty strings")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")

    outcome = outcome.astype(bool, copy=False)
    group_ids = group_ids.astype(str, copy=False)
    unique_groups, inverse = np.unique(group_ids, return_inverse=True)
    result = {
        "error_rate": float(outcome.mean()),
        "independent_groups": int(unique_groups.size),
        "trials": int(outcome.size),
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "sampling_unit": "independent group; all nested trials remain together",
        "warning": (
            "Cluster bootstrap inference is conditional on the supplied groups; "
            "it does not replace independent-scene or external-data validation."
        ),
    }
    if unique_groups.size < 2:
        return {
            **result,
            "cluster_bootstrap_ci95": None,
            "status": "insufficient_independent_groups",
        }

    errors_by_group = np.bincount(inverse, weights=outcome.astype(np.float64))
    trials_by_group = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(seed)
    max_cluster_cells = 1_000_000
    draw_chunk_size = max(1, max_cluster_cells // int(unique_groups.size))
    rate_chunks: list[np.ndarray] = []
    for start in range(0, draws, draw_chunk_size):
        count = min(draw_chunk_size, draws - start)
        sampled_groups = rng.integers(0, unique_groups.size, size=(count, unique_groups.size))
        sampled_trials = trials_by_group[sampled_groups].sum(axis=1)
        rate_chunks.append(errors_by_group[sampled_groups].sum(axis=1) / sampled_trials)
    rates = np.concatenate(rate_chunks)
    ci_low, ci_high = np.percentile(rates, [2.5, 97.5])
    return {
        **result,
        "cluster_bootstrap_ci95": [float(ci_low), float(ci_high)],
        "status": "estimated",
    }


def paired_group_regression_comparison(
    target: list[float],
    control_prediction: list[float],
    treatment_prediction: list[float],
    group_id: list[str],
    *,
    control: str,
    treatment: str,
    seed: int = 1,
    draws: int = 5000,
) -> dict:
    """Compare scalar regression methods by paired independent-group bootstrap.

    Rows are paired within each group, then each group's RMSE/MAE contributes
    one equally weighted value. This prevents many correlated samples in one
    group from being treated as independent evidence.
    """
    y = np.asarray(target, dtype=np.float64)
    control_values = np.asarray(control_prediction, dtype=np.float64)
    treatment_values = np.asarray(treatment_prediction, dtype=np.float64)
    groups = np.asarray(group_id, dtype=str)
    if (
        y.ndim != 1
        or not y.size
        or control_values.shape != y.shape
        or treatment_values.shape != y.shape
        or groups.shape != y.shape
    ):
        raise ValueError("paired targets, predictions, and groups must be aligned nonempty vectors")
    if (
        not np.isfinite(y).all()
        or not np.isfinite(control_values).all()
        or not np.isfinite(treatment_values).all()
    ):
        raise ValueError("paired targets and predictions must be finite")
    if np.any(groups == ""):
        raise ValueError("group IDs must be nonempty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 100:
        raise ValueError("draws must be an integer >= 100")

    unique_groups = np.unique(groups)
    control_error = control_values - y
    treatment_error = treatment_values - y
    control_group_rmse = np.asarray(
        [np.sqrt(np.mean(control_error[groups == key] ** 2)) for key in unique_groups]
    )
    treatment_group_rmse = np.asarray(
        [np.sqrt(np.mean(treatment_error[groups == key] ** 2)) for key in unique_groups]
    )
    control_group_mae = np.asarray(
        [np.mean(np.abs(control_error[groups == key])) for key in unique_groups]
    )
    treatment_group_mae = np.asarray(
        [np.mean(np.abs(treatment_error[groups == key])) for key in unique_groups]
    )
    rmse_delta = treatment_group_rmse - control_group_rmse
    mae_delta = treatment_group_mae - control_group_mae
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, unique_groups.size, size=(draws, unique_groups.size))
    rmse_bootstrap = rmse_delta[indices].mean(axis=1)
    mae_bootstrap = mae_delta[indices].mean(axis=1)
    rmse_low, rmse_high = np.quantile(rmse_bootstrap, [0.025, 0.975])
    mae_low, mae_high = np.quantile(mae_bootstrap, [0.025, 0.975])
    return {
        "control": control,
        "treatment": treatment,
        "comparison_unit": "independent_group",
        "groups": int(unique_groups.size),
        "samples": int(y.size),
        "treatment_minus_control_group_macro_rmse": float(rmse_delta.mean()),
        "group_macro_rmse_delta_bootstrap_ci95": [float(rmse_low), float(rmse_high)],
        "treatment_minus_control_group_macro_mae": float(mae_delta.mean()),
        "group_macro_mae_delta_bootstrap_ci95": [float(mae_low), float(mae_high)],
        "bootstrap_seed": seed,
        "bootstrap_draws": draws,
        "warning": "Descriptive paired synthetic comparison; not independent-scene or measured-channel evidence.",
    }
