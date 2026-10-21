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
