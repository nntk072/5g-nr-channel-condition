"""Seeded noise-calibrated operating-characteristic study for NR PSS/SSS acquisition."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from integrations.sionna.nr_sync import acquire_nr_cell, make_nr_sync_burst
from scipy.stats import binomtest

from experiments.link_curves import wilson_score_interval_95

FRACTIONAL_TIMING_HYPOTHESES_SAMPLES = (-0.375, -0.1875, 0.0, 0.1875, 0.375)


def _measurement(
    capture: np.ndarray,
    *,
    sampling_frequency_hz: float,
    cfo_hypotheses_hz: tuple[float, ...],
) -> dict[str, Any]:
    detection = acquire_nr_cell(
        capture,
        fft_size=256,
        cyclic_prefix_length=32,
        sampling_frequency_hz=sampling_frequency_hz,
        cfo_hypotheses_hz=cfo_hypotheses_hz,
        fractional_timing_hypotheses_samples=FRACTIONAL_TIMING_HYPOTHESES_SAMPLES,
        cfo_guard_samples=12,
        minimum_pss_score=0.0,
        minimum_sss_score=0.0,
    )
    return {
        "pss_score": detection.pss_score,
        "sss_score": detection.sss_score,
        "physical_cell_id": detection.physical_cell_id,
        "sample_offset": detection.sample_offset,
        "fractional_sample_offset": detection.fractional_sample_offset,
        "cfo_hz": detection.cfo_hz,
    }


def _noise_captures(
    rng: np.random.Generator,
    count: int,
    *,
    capture_samples: int,
    sampling_frequency_hz: float,
    cfo_hypotheses_hz: tuple[float, ...],
) -> list[dict[str, float]]:
    measurements = []
    for _ in range(count):
        noise = (
            rng.standard_normal(capture_samples) + 1j * rng.standard_normal(capture_samples)
        ) / math.sqrt(2.0)
        result = _measurement(
            noise,
            sampling_frequency_hz=sampling_frequency_hz,
            cfo_hypotheses_hz=cfo_hypotheses_hz,
        )
        measurements.append({"pss_score": result["pss_score"], "sss_score": result["sss_score"]})
    return measurements


def _wilson_counts(successes: int, trials: int) -> dict[str, float]:
    lower, upper = wilson_score_interval_95(successes, trials)
    return {
        "rate": successes / trials,
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
    }


def compare_nr_sync_operating_characteristics(
    control: dict[str, Any],
    treatment: dict[str, Any],
    *,
    bootstrap_seed: int = 20261027,
    bootstrap_draws: int = 10_000,
) -> dict[str, Any]:
    """Compare same-seed acquisition trials and control family-wise error.

    Every operating point must contain the same randomized trial inputs in the
    same order. Exact paired binomial tests are applied to discordant binary
    outcomes, with Holm adjustment across the complete operating-point family.
    Bootstrap intervals summarize this simulation's paired trial mix only.
    """
    if control.get("schema") != "loslab.nr_sync_operating_characteristics.v1":
        raise ValueError("control has an unsupported NR sync study schema")
    if treatment.get("schema") != control.get("schema"):
        raise ValueError("control and treatment schemas differ")
    if control.get("seed") != treatment.get("seed"):
        raise ValueError("control and treatment seeds differ")
    for key in ("calibration_trials", "independent_noise_validation_trials"):
        if control.get("configuration", {}).get(key) != treatment.get("configuration", {}).get(key):
            raise ValueError(f"control and treatment {key} differ")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if (
        isinstance(bootstrap_draws, bool)
        or not isinstance(bootstrap_draws, int)
        or bootstrap_draws <= 0
    ):
        raise ValueError("bootstrap_draws must be a positive integer")

    before_points = control.get("signal_points")
    after_points = treatment.get("signal_points")
    if (
        not isinstance(before_points, list)
        or not before_points
        or not isinstance(after_points, list)
    ):
        raise ValueError("studies must contain nonempty signal_points lists")
    if len(before_points) != len(after_points):
        raise ValueError("control and treatment operating-point counts differ")

    rng = np.random.default_rng(bootstrap_seed)
    comparisons: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    for before, after in zip(before_points, after_points, strict=True):
        if (before.get("channel"), before.get("snr_db"), before.get("trials")) != (
            after.get("channel"),
            after.get("snr_db"),
            after.get("trials"),
        ):
            raise ValueError("control and treatment operating points do not align")
        before_trials = before.get("per_trial")
        after_trials = after.get("per_trial")
        if (
            not isinstance(before_trials, list)
            or not before_trials
            or not isinstance(after_trials, list)
            or len(before_trials) != len(after_trials)
        ):
            raise ValueError("paired operating points need equal nonempty per-trial records")
        identity_fields = (
            "nid1",
            "nid2",
            "true_timing_samples",
            "true_fractional_timing_samples",
            "true_cfo_hz",
        )
        if any(
            any(before_row.get(field) != after_row.get(field) for field in identity_fields)
            for before_row, after_row in zip(before_trials, after_trials, strict=True)
        ):
            raise ValueError("control and treatment trial inputs/order differ")

        before_success = np.asarray(
            [bool(row["correct_acquisition"]) for row in before_trials], dtype=bool
        )
        after_success = np.asarray(
            [bool(row["correct_acquisition"]) for row in after_trials], dtype=bool
        )
        improved = int(np.count_nonzero(~before_success & after_success))
        regressed = int(np.count_nonzero(before_success & ~after_success))
        discordant = improved + regressed
        p_value = (
            float(binomtest(min(improved, regressed), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        )
        raw_p_values.append(p_value)
        delta = after_success.astype(np.float64) - before_success.astype(np.float64)
        bootstrap = rng.choice(delta, size=(bootstrap_draws, delta.size), replace=True).mean(axis=1)
        timing_before = np.asarray(
            [bool(row["correct_timing"]) for row in before_trials], dtype=bool
        )
        timing_after = np.asarray([bool(row["correct_timing"]) for row in after_trials], dtype=bool)
        comparisons.append(
            {
                "channel": before["channel"],
                "snr_db": before["snr_db"],
                "trials": int(delta.size),
                "control_correct_acquisitions": int(before_success.sum()),
                "treatment_correct_acquisitions": int(after_success.sum()),
                "paired_acquisition_rate_change": float(delta.mean()),
                "paired_acquisition_rate_change_bootstrap_ci95": [
                    float(value) for value in np.percentile(bootstrap, [2.5, 97.5])
                ],
                "improved_trials": improved,
                "regressed_trials": regressed,
                "discordant_trials": discordant,
                "mcnemar_exact_two_sided_p_value": p_value,
                "control_correct_timing": int(timing_before.sum()),
                "treatment_correct_timing": int(timing_after.sum()),
                "paired_timing_rate_change": float(
                    timing_after.astype(np.float64).mean() - timing_before.astype(np.float64).mean()
                ),
            }
        )

    order = np.argsort(raw_p_values)
    adjusted = np.empty(len(raw_p_values), dtype=np.float64)
    running_max = 0.0
    count = len(raw_p_values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (count - rank) * raw_p_values[int(index)])
        adjusted[int(index)] = min(1.0, running_max)
    for row, value in zip(comparisons, adjusted, strict=True):
        row["mcnemar_holm_adjusted_p_value"] = float(value)

    before_noise = control.get("false_alarm_validation", {})
    after_noise = treatment.get("false_alarm_validation", {})
    before_mask = before_noise.get("validation_accepted")
    after_mask = after_noise.get("validation_accepted")
    if (
        not isinstance(before_mask, list)
        or not isinstance(after_mask, list)
        or len(before_mask) == 0
        or len(before_mask) != len(after_mask)
    ):
        raise ValueError("noise false-alarm validation masks must be paired and nonempty")
    before_mask_array = np.asarray(before_mask, dtype=bool)
    after_mask_array = np.asarray(after_mask, dtype=bool)
    false_alarm_improved = int(np.count_nonzero(before_mask_array & ~after_mask_array))
    false_alarm_regressed = int(np.count_nonzero(~before_mask_array & after_mask_array))
    return {
        "schema": "loslab.nr_sync_paired_comparison.v1",
        "evidence_grade": "PAIRED_SEEDED_CONTROLLED_SIMULATION",
        "seed": control["seed"],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": bootstrap_draws,
        "control_source_revision": control.get("source_revision"),
        "treatment_source_revision": treatment.get("source_revision"),
        "configuration": {
            "control": control.get("configuration"),
            "treatment": treatment.get("configuration"),
        },
        "operating_points": comparisons,
        "noise_false_alarm_comparison": {
            "trials": len(before_mask),
            "control_false_alarms": int(before_mask_array.sum()),
            "treatment_false_alarms": int(after_mask_array.sum()),
            "improved_noise_trials": false_alarm_improved,
            "regressed_noise_trials": false_alarm_regressed,
            "paired_exact_two_sided_p_value": float(
                binomtest(
                    min(false_alarm_improved, false_alarm_regressed),
                    false_alarm_improved + false_alarm_regressed,
                    0.5,
                ).pvalue
            )
            if false_alarm_improved + false_alarm_regressed
            else 1.0,
        },
        "limitations": [
            "The control and treatment are paired seeded synthetic trials, not independent scenes, devices, or measured NR captures.",
            "The bootstrap interval resamples trials within each configured channel/SNR point and does not quantify external validity.",
            "Holm adjustment covers the configured operating-point family only; no deployment-level synchronization claim follows.",
        ],
    }


def run_nr_sync_operating_characteristics(
    *,
    seed: int = 20261022,
    calibration_trials: int = 1024,
    validation_trials: int = 2048,
    signal_trials_per_point: int = 256,
    target_false_alarm_probability: float = 0.05,
    snr_db_values: Sequence[float] = (0.0, 6.0, 12.0, 18.0),
) -> dict[str, Any]:
    """Calibrate PSS/SSS score gates on noise, then independently evaluate.

    Signal checks cover a flat channel and a static 12-sample in-CP echo. SNR
    is the mean complex-sample power over the four-symbol transmitted burst,
    including its intentionally silent symbols, divided by AWGN variance.
    The acquisition grid, timing search, capture length, and channel families
    are fixed and recorded; the resulting rates do not certify a general NR
    synchronizer or a deployment false-alarm probability.
    """
    for name, value in (
        ("calibration_trials", calibration_trials),
        ("validation_trials", validation_trials),
        ("signal_trials_per_point", signal_trials_per_point),
        ("seed", seed),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < (0 if name == "seed" else 1):
            raise ValueError(f"{name} must be {'nonnegative' if name == 'seed' else 'positive'}")
    if (
        not math.isfinite(target_false_alarm_probability)
        or not 0.0 < target_false_alarm_probability < 1.0
    ):
        raise ValueError("target_false_alarm_probability must be finite and in (0, 1)")
    snr_values = tuple(float(value) for value in snr_db_values)
    if not snr_values or any(not math.isfinite(value) for value in snr_values):
        raise ValueError("snr_db_values must be a non-empty sequence of finite values")

    fft_size = 256
    cyclic_prefix_length = 32
    symbol_length = fft_size + cyclic_prefix_length
    sampling_frequency_hz = 7_680_000.0
    cfo_hypotheses_hz = (-3_000.0, 0.0, 3_000.0)
    maximum_timing_offset_samples = 32
    maximum_echo_delay_samples = 12
    cfo_accuracy_tolerance_hz = 250.0
    timing_accuracy_tolerance_samples = 0.1
    capture_samples = (
        maximum_timing_offset_samples
        + 4 * symbol_length
        + maximum_echo_delay_samples
        + cyclic_prefix_length
    )
    root = np.random.SeedSequence(seed)
    calibration_seed, validation_seed, signal_seed = root.spawn(3)
    calibration = _noise_captures(
        np.random.default_rng(calibration_seed),
        calibration_trials,
        capture_samples=capture_samples,
        sampling_frequency_hz=sampling_frequency_hz,
        cfo_hypotheses_hz=cfo_hypotheses_hz,
    )
    quantile = 1.0 - target_false_alarm_probability / 2.0
    minimum_pss_score = float(
        np.quantile([row["pss_score"] for row in calibration], quantile, method="higher")
    )
    minimum_sss_score = float(
        np.quantile([row["sss_score"] for row in calibration], quantile, method="higher")
    )

    validation = _noise_captures(
        np.random.default_rng(validation_seed),
        validation_trials,
        capture_samples=capture_samples,
        sampling_frequency_hz=sampling_frequency_hz,
        cfo_hypotheses_hz=cfo_hypotheses_hz,
    )
    false_alarms = [
        row["pss_score"] >= minimum_pss_score and row["sss_score"] >= minimum_sss_score
        for row in validation
    ]
    false_alarm_count = int(sum(false_alarms))
    false_alarm_metrics = _wilson_counts(false_alarm_count, validation_trials)

    rng = np.random.default_rng(signal_seed)
    signal_points: list[dict[str, Any]] = []
    for channel_name in ("flat", "two_tap_echo"):
        for snr_db in snr_values:
            trials: list[dict[str, Any]] = []
            for _ in range(signal_trials_per_point):
                nid1 = int(rng.integers(0, 336))
                nid2 = int(rng.integers(0, 3))
                true_fractional_timing = float(rng.uniform(-0.375, 0.375))
                burst = make_nr_sync_burst(
                    nid1,
                    nid2,
                    fft_size=fft_size,
                    cyclic_prefix_length=cyclic_prefix_length,
                    fractional_delay_samples=true_fractional_timing,
                )
                if channel_name == "two_tap_echo":
                    channel = np.zeros(maximum_echo_delay_samples + 1, dtype=np.complex128)
                    channel[0] = 1.0
                    channel[maximum_echo_delay_samples] = 0.9 + 0.1j
                    burst = np.convolve(burst, channel)
                timing_offset = int(rng.integers(0, maximum_timing_offset_samples + 1))
                cfo_hz = float(rng.choice((-3_500.0, 3_500.0)))
                gain = complex(np.exp(1j * rng.uniform(-np.pi, np.pi)))
                clean = np.zeros(capture_samples, dtype=np.complex128)
                clean[timing_offset : timing_offset + burst.size] = gain * burst
                sample_axis = np.arange(capture_samples, dtype=np.float64)
                clean *= np.exp(2j * np.pi * cfo_hz * sample_axis / sampling_frequency_hz)
                signal_power = float(np.mean(np.abs(clean) ** 2))
                noise_variance = signal_power / (10.0 ** (snr_db / 10.0))
                noise = math.sqrt(noise_variance / 2.0) * (
                    rng.standard_normal(capture_samples) + 1j * rng.standard_normal(capture_samples)
                )
                measured = _measurement(
                    clean + noise,
                    sampling_frequency_hz=sampling_frequency_hz,
                    cfo_hypotheses_hz=cfo_hypotheses_hz,
                )
                accepted = (
                    measured["pss_score"] >= minimum_pss_score
                    and measured["sss_score"] >= minimum_sss_score
                )
                correct_cell = measured["physical_cell_id"] == 3 * nid1 + nid2
                timing_error_samples = abs(
                    measured["sample_offset"]
                    + measured["fractional_sample_offset"]
                    - timing_offset
                    - true_fractional_timing
                )
                correct_timing = timing_error_samples <= timing_accuracy_tolerance_samples
                cfo_error_hz = abs(measured["cfo_hz"] - cfo_hz)
                correct_cfo = cfo_error_hz <= cfo_accuracy_tolerance_hz
                trials.append(
                    {
                        "nid1": nid1,
                        "nid2": nid2,
                        "true_timing_samples": timing_offset,
                        "true_fractional_timing_samples": true_fractional_timing,
                        "true_cfo_hz": cfo_hz,
                        "detected_cell_id": measured["physical_cell_id"],
                        "detected_timing_samples": measured["sample_offset"],
                        "detected_fractional_timing_samples": measured["fractional_sample_offset"],
                        "absolute_timing_error_samples": timing_error_samples,
                        "detected_cfo_hz": measured["cfo_hz"],
                        "absolute_cfo_error_hz": cfo_error_hz,
                        "pss_score": measured["pss_score"],
                        "sss_score": measured["sss_score"],
                        "accepted": accepted,
                        "correct_cell": correct_cell,
                        "correct_timing": correct_timing,
                        "correct_cfo": correct_cfo,
                        "correct_acquisition": accepted
                        and correct_cell
                        and correct_timing
                        and correct_cfo,
                    }
                )
            accepted_count = sum(bool(row["accepted"]) for row in trials)
            correct_count = sum(bool(row["correct_acquisition"]) for row in trials)
            signal_points.append(
                {
                    "channel": channel_name,
                    "snr_db": snr_db,
                    "trials": signal_trials_per_point,
                    "accepted": accepted_count,
                    "correct_acquisitions": correct_count,
                    "acquisition_probability": _wilson_counts(
                        correct_count, signal_trials_per_point
                    ),
                    "conditional_correct_given_accepted": (
                        _wilson_counts(correct_count, accepted_count) if accepted_count else None
                    ),
                    "wrong_cell_among_accepted": accepted_count
                    - sum(bool(row["correct_cell"]) and bool(row["accepted"]) for row in trials),
                    "wrong_timing_among_accepted": accepted_count
                    - sum(bool(row["correct_timing"]) and bool(row["accepted"]) for row in trials),
                    "wrong_cfo_among_accepted": accepted_count
                    - sum(bool(row["correct_cfo"]) and bool(row["accepted"]) for row in trials),
                    "per_trial": trials,
                }
            )

    return {
        "schema": "loslab.nr_sync_operating_characteristics.v1",
        "evidence_grade": "SEEDED_CONTROLLED_SIMULATION_NOT_FIELD_QUALIFICATION",
        "seed": seed,
        "configuration": {
            "calibration_trials": calibration_trials,
            "independent_noise_validation_trials": validation_trials,
            "signal_trials_per_channel_snr_point": signal_trials_per_point,
            "target_joint_false_alarm_probability": target_false_alarm_probability,
            "calibration_quantile_per_score": quantile,
            "minimum_pss_score": minimum_pss_score,
            "minimum_sss_score": minimum_sss_score,
            "finite_sample_union_bound_allocation_per_score": target_false_alarm_probability / 2.0,
            "fft_size": fft_size,
            "cyclic_prefix_length_samples": cyclic_prefix_length,
            "sampling_frequency_hz": sampling_frequency_hz,
            "cfo_hypotheses_hz": list(cfo_hypotheses_hz),
            "maximum_timing_offset_samples": maximum_timing_offset_samples,
            "fractional_timing_hypotheses_samples": list(FRACTIONAL_TIMING_HYPOTHESES_SAMPLES),
            "fractional_timing_refinement": {
                "method": "bounded scalar maximization of normalized PSS correlation",
                "bounds_samples": [-0.5, 0.5],
                "optimizer": "scipy.optimize.minimize_scalar(method=bounded)",
                "xatol_samples": 1e-5,
            },
            "timing_accuracy_tolerance_samples": timing_accuracy_tolerance_samples,
            "true_off_grid_cfo_hz": [-3_500.0, 3_500.0],
            "cfo_guard_samples": 12,
            "correct_cfo_tolerance_hz": cfo_accuracy_tolerance_hz,
            "capture_samples": capture_samples,
            "noise_model": "independent circular complex Gaussian; variance from unit-variance RNG draws",
            "snr_definition": "mean complex-sample signal power over the padded four-symbol burst divided by AWGN variance",
            "channels": {
                "flat": {"taps": [[0, 1.0, 0.0]]},
                "two_tap_echo": {"taps": [[0, 1.0, 0.0], [12, 0.9, 0.1]]},
            },
        },
        "false_alarm_validation": {
            "trials": validation_trials,
            "false_alarms": false_alarm_count,
            **false_alarm_metrics,
            "target_met_by_wilson_upper_bound": (
                false_alarm_metrics["wilson_95_upper"] <= target_false_alarm_probability
            ),
            "calibration_pss_scores": [row["pss_score"] for row in calibration],
            "calibration_sss_scores": [row["sss_score"] for row in calibration],
            "validation_pss_scores": [row["pss_score"] for row in validation],
            "validation_sss_scores": [row["sss_score"] for row in validation],
            "validation_accepted": false_alarms,
        },
        "signal_points": signal_points,
        "limitations": [
            "Only a four-symbol synthetic PSS/SSS burst and one configured numerology are evaluated; signal CFO is restricted to ±3.5 kHz, halfway between the outer ±3-kHz search bins.",
            "Threshold calibration and validation use white Gaussian noise, not interference, false neighboring cells, or realistic RF captures.",
            "Signal trials use randomized cell identity, integer plus fractional timing searched on a five-value grid, off-grid CFO, random global phase, and either a flat channel or one static 12-sample in-CP echo.",
            "The SNR convention includes the fixture's silent symbols and is not a standardized NR receiver sensitivity or link-budget measure.",
            "Fractional timing uses a five-point coarse grid followed by local bounded score refinement; CFO is CP-refined only within its unambiguous range. Mobility, PBCH, false-alarm-hour rates, and production thresholds are not validated.",
            "Confidence intervals describe these independent Monte Carlo trials only and do not quantify scene or device variability.",
        ],
    }
