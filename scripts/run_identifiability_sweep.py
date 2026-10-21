"""Reproducible local Fisher-information sweep for two-path CFR observations."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from propagation.identifiability import path_parameter_fisher_diagnostics


def build_sweep(
    *,
    tone_counts: Sequence[int],
    tone_spacing_hz: float,
    snr_db_values: Sequence[float],
    separation_fractions: Sequence[float],
) -> dict:
    """Evaluate local two-path CRLBs at fixed fractions of reciprocal bandwidth."""
    if not tone_counts or any(count < 2 for count in tone_counts):
        raise ValueError("tone_counts must contain integers >= 2")
    if not math.isfinite(tone_spacing_hz) or tone_spacing_hz <= 0:
        raise ValueError("tone_spacing_hz must be finite and positive")
    if not snr_db_values or any(not math.isfinite(value) for value in snr_db_values):
        raise ValueError("snr_db_values must contain finite values")
    if not separation_fractions or any(
        not math.isfinite(value) or value < 0 for value in separation_fractions
    ):
        raise ValueError("separation_fractions must contain finite nonnegative values")

    gains = np.asarray([1.0 + 0.2j, 0.7 - 0.4j], dtype=np.complex128)
    cases: list[dict] = []
    for tone_count in tone_counts:
        frequencies = (np.arange(tone_count, dtype=np.float64) - tone_count // 2) * tone_spacing_hz
        bandwidth_hz = float(np.ptp(frequencies))
        first_null_delay_s = 1.0 / bandwidth_hz
        for snr_db in snr_db_values:
            noise_variance = 10.0 ** (-snr_db / 10.0)
            for separation_fraction in separation_fractions:
                separation_s = separation_fraction * first_null_delay_s
                diagnostic = path_parameter_fisher_diagnostics(
                    gains,
                    np.asarray([0.0, separation_s]),
                    frequencies,
                    noise_variance=noise_variance,
                )
                covariance = diagnostic.crlb_covariance
                delay_std_ns = (
                    None
                    if covariance is None
                    else [
                        float(value)
                        for value in np.sqrt(np.maximum(np.diag(covariance)[4:], 0.0)) * 1e9
                    ]
                )
                condition = diagnostic.condition_number
                cases.append(
                    {
                        "tone_count": tone_count,
                        "tone_spacing_hz": tone_spacing_hz,
                        "bandwidth_hz": bandwidth_hz,
                        "snr_db": snr_db,
                        "noise_variance_per_complex_tone": noise_variance,
                        "separation_fraction_of_reciprocal_bandwidth": separation_fraction,
                        "path_separation_s": separation_s,
                        "rank": diagnostic.rank,
                        "parameter_count": 6,
                        "condition_number": condition if math.isfinite(condition) else None,
                        "crlb_delay_std_ns": delay_std_ns,
                    }
                )
    runtime = versions()
    return {
        "schema": "loslab.identifiability_sweep.v1",
        "evidence_grade": "ANALYTICAL_LOCAL_DIAGNOSTIC",
        "model": "H(f)=sum_l a_l exp(-j 2 pi f tau_l) + circular complex Gaussian noise",
        "path_gains_real_imag": [[1.0, 0.2], [0.7, -0.4]],
        "source_revision": runtime.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "script_sha256": sha256_file(__file__),
        "versions": runtime,
        "cases": cases,
        "limitations": [
            "Local Fisher information is not a global estimator guarantee or a delay-resolution threshold.",
            "No receiver, synchronization uncertainty, channel-estimation error, or scene variation is modeled.",
            "CRLB covariance is omitted when the supplied local parameterization is rank deficient.",
            "Noise variance is specified per complex tone and SNR is relative to unit variance.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tone-counts", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--tone-spacing-hz", type=float, default=30_000.0)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[0.0, 10.0, 20.0, 30.0])
    parser.add_argument(
        "--separation-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    try:
        result = build_sweep(
            tone_counts=args.tone_counts,
            tone_spacing_hz=args.tone_spacing_hz,
            snr_db_values=args.snr_db,
            separation_fractions=args.separation_fractions,
        )
    except ValueError as error:
        parser.error(str(error))
    atomic_json(args.output, result)
    print(f"Saved {len(result['cases'])} cases to {args.output}")


if __name__ == "__main__":
    main()
