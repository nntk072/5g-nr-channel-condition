"""Synthetic stress test for masked-CFR sparse path-order selection."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from models.multipath import fit_sparse_multipath
from scipy.stats import binomtest


def run_multipath_order_stress(
    *,
    seed: int = 20261022,
    trials_per_cell: int = 40,
    tone_count: int = 64,
    bandwidths_hz: tuple[float, ...] = (5e6, 20e6),
    snr_db_values: tuple[float, ...] = (0.0, 10.0, 20.0),
    path_orders: tuple[int, ...] = (0, 1, 2, 3),
    maximum_delay_s: float = 1e-6,
    delay_grid_size: int = 257,
) -> dict[str, Any]:
    """Evaluate BIC path-order selection under unknown order and noise.

    Each trial is an independently generated condition group. The estimator
    receives only noisy masked CFR, pilot locations, frequencies and delay
    support. Reported order accuracy has Wilson intervals; full-grid
    reconstruction is evaluated only after fitting and is explicitly an
    oracle-aided simulation metric. The sweep does not model diffuse paths,
    receiver estimation error, correlated observations, or scene transfer.
    """
    for name, value, minimum in (
        ("seed", seed, 0),
        ("trials_per_cell", trials_per_cell, 2),
        ("tone_count", tone_count, 8),
        ("delay_grid_size", delay_grid_size, 9),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
    if not bandwidths_hz or any(not math.isfinite(value) or value <= 0 for value in bandwidths_hz):
        raise ValueError("bandwidths_hz must contain finite positive values")
    if not snr_db_values or any(not math.isfinite(value) for value in snr_db_values):
        raise ValueError("snr_db_values must contain finite values")
    if not path_orders or any(
        isinstance(order, bool) or not isinstance(order, int) or not 0 <= order <= 3
        for order in path_orders
    ):
        raise ValueError("path_orders must contain integers in [0, 3]")
    if len(set(path_orders)) != len(path_orders):
        raise ValueError("path_orders must be unique")
    if not math.isfinite(maximum_delay_s) or maximum_delay_s <= 0:
        raise ValueError("maximum_delay_s must be finite and positive")

    rng = np.random.default_rng(seed)
    cases: list[dict[str, Any]] = []
    maximum_paths = max(path_orders)
    for bandwidth_hz in bandwidths_hz:
        frequencies = np.linspace(
            -bandwidth_hz / 2.0, bandwidth_hz / 2.0, tone_count, dtype=np.float64
        )
        for snr_db in snr_db_values:
            for true_order in path_orders:
                correct = 0
                predicted_orders: list[int] = []
                noise_ratios: list[float] = []
                nmse_values: list[float] = []
                for _ in range(trials_per_cell):
                    delays = np.sort(rng.uniform(0.0, maximum_delay_s, size=true_order))
                    gains = rng.uniform(0.5, 1.5, size=true_order) * np.exp(
                        1j * rng.uniform(-np.pi, np.pi, size=true_order)
                    )
                    if true_order:
                        truth = np.exp(-2j * np.pi * frequencies[:, None] * delays[None, :]) @ gains
                        signal_power = float(np.mean(np.abs(truth) ** 2))
                    else:
                        truth = np.zeros(tone_count, dtype=np.complex128)
                        signal_power = 1.0
                    noise_variance = signal_power * 10.0 ** (-snr_db / 10.0)
                    noise = np.sqrt(noise_variance / 2.0) * (
                        rng.standard_normal(tone_count) + 1j * rng.standard_normal(tone_count)
                    )
                    observed = truth + noise
                    pilot_mask = rng.random(tone_count) < 0.5
                    while np.count_nonzero(pilot_mask) < 4:
                        pilot_mask = rng.random(tone_count) < 0.5

                    fit = fit_sparse_multipath(
                        observed,
                        frequencies,
                        pilot_mask,
                        maximum_paths=maximum_paths,
                        maximum_delay_s=maximum_delay_s,
                        delay_grid_size=delay_grid_size,
                    )
                    predicted_orders.append(fit.model_order)
                    correct += fit.model_order == true_order
                    noise_ratios.append(fit.estimated_noise_variance / noise_variance)
                    if true_order:
                        nmse_values.append(
                            float(
                                np.mean(np.abs(fit.predicted_cfr - truth) ** 2)
                                / np.mean(np.abs(truth) ** 2)
                            )
                        )

                interval = binomtest(correct, trials_per_cell).proportion_ci(method="wilson")
                confusion = [
                    predicted_orders.count(candidate_order)
                    for candidate_order in range(maximum_paths + 1)
                ]
                cases.append(
                    {
                        "bandwidth_hz": float(bandwidth_hz),
                        "snr_db": float(snr_db),
                        "true_path_order": int(true_order),
                        "trials": trials_per_cell,
                        "order_accuracy": correct / trials_per_cell,
                        "order_accuracy_wilson_95": [
                            float(interval.low),
                            float(interval.high),
                        ],
                        "predicted_order_counts": confusion,
                        "median_estimated_to_true_noise_variance_ratio": float(
                            np.median(noise_ratios)
                        ),
                        "median_full_grid_nmse": (
                            float(np.median(nmse_values)) if nmse_values else None
                        ),
                    }
                )

    source_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    estimator_digest = hashlib.sha256(
        Path(fit_sparse_multipath.__code__.co_filename).read_bytes()
    ).hexdigest()
    repository_root = Path(__file__).resolve().parent.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "masked_cfr_unknown_path_order_and_noise_stress",
        "evidence_grade": "ANALYTICAL_SYNTHETIC_STRESS_ONLY",
        "code_revision": revision.stdout.strip() if revision.returncode == 0 else "unknown",
        "seed": seed,
        "trials_per_cell": trials_per_cell,
        "tone_count": tone_count,
        "bandwidths_hz": list(bandwidths_hz),
        "snr_db_values": list(snr_db_values),
        "path_orders": list(path_orders),
        "maximum_delay_s": maximum_delay_s,
        "delay_grid_size": delay_grid_size,
        "estimator": "delay-grid OMP + continuous least-squares + unknown-noise BIC",
        "estimator_source_sha256": estimator_digest,
        "experiment_source_sha256": source_digest,
        "limitations": [
            "No diffuse residual or receiver channel-estimation error is modeled.",
            "Each trial is synthetic and does not establish transfer to RT scenes or measurements.",
            "BIC consistency is not guaranteed for coherent, unresolved, or misspecified paths.",
            "Full-grid reconstruction uses hidden synthetic CFR only for post-fit scoring.",
        ],
        "cases": cases,
    }
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
    report["result_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return report
