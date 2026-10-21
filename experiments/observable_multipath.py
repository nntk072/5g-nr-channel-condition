"""Grouped analytical two-path training and matched profile-likelihood baseline.

This module is a controlled software experiment. Its generated groups are
independent synthetic conditions, not RT scenes or measured radio sessions.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True, slots=True)
class TwoPathDataset:
    """Receiver-only arrays plus separately held oracle targets and split IDs."""

    observed_cfr: np.ndarray
    pilot_mask: np.ndarray
    frequencies_hz: np.ndarray
    noise_variance: np.ndarray
    relative_delay_s: np.ndarray
    gains: np.ndarray
    maximum_relative_delay_s: float
    group_id: np.ndarray
    sample_id: np.ndarray
    split: np.ndarray
    stratum: np.ndarray
    condition: tuple[dict[str, float], ...]
    source_sha256: str

    def indices(self, split: str) -> np.ndarray:
        """Return stable row indices for a named split."""
        if split not in {"train", "validation", "calibration", "test"}:
            raise ValueError(f"unknown split: {split}")
        return np.flatnonzero(self.split == split)


def _validate_generation_args(
    num_groups: int, samples_per_group: int, n_tones: int, seed: int
) -> None:
    for name, value, minimum in (
        ("num_groups", num_groups, 40),
        ("samples_per_group", samples_per_group, 1),
        ("n_tones", n_tones, 8),
        ("seed", seed, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum}")


def generate_grouped_two_path_dataset(
    *,
    num_groups: int = 800,
    samples_per_group: int = 4,
    n_tones: int = 64,
    seed: int = 20261031,
    maximum_relative_delay_s: float = 1e-6,
) -> TwoPathDataset:
    """Generate grouped pilot observations and assign groups to four splits.

    Bandwidth, SNR, and delay-separation strata are fixed per group, while
    complex gains, pilot masks, and noise vary by sample. Whole groups are
    assigned to train/validation/calibration/test (60/10/15/15 percent), so
    correlated rows never cross a split. Condition groups are sampled from the
    same declared mixture; this is not an out-of-distribution transfer test.
    """
    _validate_generation_args(num_groups, samples_per_group, n_tones, seed)
    if not math.isfinite(maximum_relative_delay_s) or maximum_relative_delay_s <= 0:
        raise ValueError("maximum_relative_delay_s must be finite and positive")
    rng = np.random.default_rng(seed)
    split_names = np.asarray(["train", "validation", "calibration", "test"])
    split_counts = np.floor(np.asarray([0.60, 0.10, 0.15, 0.15]) * num_groups).astype(int)
    split_counts[0] += num_groups - int(split_counts.sum())
    shuffled_groups = rng.permutation(num_groups)
    group_splits = np.empty(num_groups, dtype="U11")
    cursor = 0
    for split_name, count in zip(split_names, split_counts, strict=True):
        group_splits[shuffled_groups[cursor : cursor + count]] = split_name
        cursor += int(count)

    total = num_groups * samples_per_group
    observed = np.empty((total, n_tones), dtype=np.complex64)
    mask = np.empty((total, n_tones), dtype=np.bool_)
    frequencies = np.empty((total, n_tones), dtype=np.float32)
    noise_variance = np.empty(total, dtype=np.float32)
    delay = np.empty(total, dtype=np.float32)
    gains = np.empty((total, 2), dtype=np.complex64)
    groups = np.empty(total, dtype="U32")
    ids = np.empty(total, dtype="U48")
    splits = np.empty(total, dtype="U11")
    strata = np.empty(total, dtype="U16")
    conditions: list[dict[str, float]] = []
    bandwidth_choices_hz = np.asarray([3.84e6, 7.68e6, 15.36e6])
    snr_choices_db = np.asarray([0.0, 6.0, 12.0, 18.0, 24.0, 30.0])
    separation_choices = np.asarray([0.05, 0.1, 0.25, 0.5, 1.0])
    row = 0

    for group_index in range(num_groups):
        bandwidth_hz = float(rng.choice(bandwidth_choices_hz))
        snr_db = float(rng.choice(snr_choices_db))
        separation_fraction = float(rng.choice(separation_choices))
        group_id = f"condition-group-{group_index:06d}"
        conditions.append(
            {
                "bandwidth_hz": bandwidth_hz,
                "snr_db": snr_db,
                "separation_fraction_of_reciprocal_bandwidth": separation_fraction,
            }
        )
        group_delay = min(separation_fraction / bandwidth_hz, maximum_relative_delay_s * 0.98)
        frequency_grid = np.linspace(
            -bandwidth_hz / 2.0, bandwidth_hz / 2.0, n_tones, dtype=np.float64
        )
        group_split = str(group_splits[group_index])
        for sample_index in range(samples_per_group):
            gain_amplitude = rng.uniform(0.5, 1.5, size=2)
            gain_phase = rng.uniform(-np.pi, np.pi, size=2)
            sample_gains = gain_amplitude * np.exp(1j * gain_phase)
            noiseless = sample_gains[0] + sample_gains[1] * np.exp(
                -2j * np.pi * frequency_grid * group_delay
            )
            signal_power = float(np.mean(np.abs(noiseless) ** 2))
            variance = signal_power * 10.0 ** (-snr_db / 10.0)
            noise = np.sqrt(variance / 2.0) * (
                rng.standard_normal(n_tones) + 1j * rng.standard_normal(n_tones)
            )
            pilot_mask = rng.random(n_tones) < 0.5
            while np.count_nonzero(pilot_mask) < 2:
                pilot_mask = rng.random(n_tones) < 0.5

            observed[row] = (noiseless + noise).astype(np.complex64)
            mask[row] = pilot_mask
            frequencies[row] = frequency_grid.astype(np.float32)
            noise_variance[row] = variance
            delay[row] = group_delay
            gains[row] = sample_gains.astype(np.complex64)
            groups[row] = group_id
            ids[row] = f"{group_id}-sample-{sample_index:03d}"
            splits[row] = group_split
            strata[row] = (
                "close"
                if separation_fraction < 0.1
                else "marginal"
                if separation_fraction < 0.5
                else "separated"
            )
            row += 1

    hashs = hashlib.sha256()
    for array in (observed, mask, frequencies, noise_variance, delay, gains, groups, splits):
        hashs.update(np.ascontiguousarray(array).tobytes())
    hashs.update(str(seed).encode())
    hashs.update(str(maximum_relative_delay_s).encode())
    return TwoPathDataset(
        observed_cfr=observed,
        pilot_mask=mask,
        frequencies_hz=frequencies,
        noise_variance=noise_variance,
        relative_delay_s=delay,
        gains=gains,
        maximum_relative_delay_s=float(maximum_relative_delay_s),
        group_id=groups,
        sample_id=ids,
        split=splits,
        stratum=strata,
        condition=tuple(conditions),
        source_sha256=hashs.hexdigest(),
    )


def profile_likelihood_delay(
    observed_cfr: np.ndarray,
    pilot_mask: np.ndarray,
    frequencies_hz: np.ndarray,
    maximum_delay_s: float,
    noise_variance: float,
    *,
    grid_points: int = 257,
) -> tuple[float, float]:
    """Fit a two-path complex least-squares profile and return delay and local scale.

    Complex gains are eliminated analytically at each delay. A bounded local
    refinement is seeded from a uniform global delay grid. The returned local
    scale is a curvature approximation; group calibration must be applied
    before interpreting it as an interval.
    """
    y = np.asarray(observed_cfr, dtype=np.complex128)
    f = np.asarray(frequencies_hz, dtype=np.float64)
    selected = np.asarray(pilot_mask, dtype=bool)
    if y.ndim != 1 or f.shape != y.shape or selected.shape != y.shape:
        raise ValueError("observations, frequencies, and mask must be matching vectors")
    if np.count_nonzero(selected) < 2:
        raise ValueError("at least two pilot tones are required")
    if not np.isfinite(y.real).all() or not np.isfinite(y.imag).all() or not np.isfinite(f).all():
        raise ValueError("observations and frequencies must be finite")
    if not math.isfinite(maximum_delay_s) or maximum_delay_s <= 0:
        raise ValueError("maximum_delay_s must be finite and positive")
    if not math.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise_variance must be finite and positive")
    if isinstance(grid_points, bool) or not isinstance(grid_points, int) or grid_points < 9:
        raise ValueError("grid_points must be an integer >= 9")
    y = y[selected]
    f = f[selected]
    count = y.size
    y_energy = float(np.vdot(y, y).real)

    def residual(delay_s: float) -> float:
        delayed = np.exp(-2j * np.pi * f * delay_s)
        cross = np.sum(delayed)
        y0 = np.sum(y)
        y1 = np.vdot(delayed, y)
        determinant = float(count * count - abs(cross) ** 2)
        if determinant <= np.finfo(np.float64).eps * count * count:
            return max(0.0, y_energy - abs(y0) ** 2 / count)
        gain0 = (count * y0 - cross * y1) / determinant
        gain1 = (count * y1 - np.conj(cross) * y0) / determinant
        explained = np.conj(gain0) * y0 + np.conj(gain1) * y1
        return max(0.0, y_energy - float(explained.real))

    grid = np.linspace(0.0, maximum_delay_s, grid_points)
    costs = np.asarray([residual(float(value)) for value in grid])
    best = int(np.argmin(costs))
    left = float(grid[max(0, best - 1)])
    right = float(grid[min(grid_points - 1, best + 1)])
    if left == right:
        estimate = float(grid[best])
    else:
        refined = minimize_scalar(residual, bounds=(left, right), method="bounded")
        estimate = float(refined.x) if refined.success else float(grid[best])
    curvature_step = maximum_delay_s / (grid_points - 1)
    if estimate <= curvature_step or estimate >= maximum_delay_s - curvature_step:
        return estimate, maximum_delay_s
    curvature = (
        residual(estimate + curvature_step)
        - 2.0 * residual(estimate)
        + residual(estimate - curvature_step)
    ) / curvature_step**2
    scale = math.sqrt(2.0 * noise_variance / curvature) if curvature > 0 else maximum_delay_s
    return estimate, float(min(max(scale, maximum_delay_s * 1e-6), maximum_delay_s))


def _profiled_delay_residual_power(
    observed: np.ndarray, frequencies_hz: np.ndarray, delay_s: float
) -> float:
    """Minimize residual power over unknown complex gains at one delay."""
    delayed = np.exp(-2j * np.pi * frequencies_hz * delay_s)
    count = observed.size
    energy = float(np.vdot(observed, observed).real)
    cross = np.sum(delayed)
    y0 = np.sum(observed)
    y1 = np.vdot(delayed, observed)
    determinant = float(count**2 - abs(cross) ** 2)
    if determinant <= np.finfo(np.float64).eps * count**2:
        return float(max(0.0, energy - abs(y0) ** 2 / count))
    gain0 = (count * y0 - cross * y1) / determinant
    gain1 = (count * y1 - np.conj(cross) * y0) / determinant
    explained = np.conj(gain0) * y0 + np.conj(gain1) * y1
    return float(max(0.0, energy - explained.real))


def profile_likelihood_deviance_grid(
    observed_cfr: np.ndarray,
    pilot_mask: np.ndarray,
    frequencies_hz: np.ndarray,
    maximum_delay_s: float,
    noise_variance: float,
    *,
    grid_points: int = 257,
) -> tuple[np.ndarray, np.ndarray]:
    """Return profile likelihood-ratio deviance over bounded relative delays.

    At each candidate delay, unknown complex path gains are eliminated by
    least squares. For circular complex Gaussian noise with variance
    ``E[|n|²]``, the returned deviance is ``2 * (RSS(delay)-RSS_min) / σ²``.
    The grid is a bounded numerical approximation; it is not a continuous
    confidence region and is specific to the fixed two-path/known-noise model.
    """
    y = np.asarray(observed_cfr, dtype=np.complex128)
    f = np.asarray(frequencies_hz, dtype=np.float64)
    mask = np.asarray(pilot_mask, dtype=bool)
    if y.ndim != 1 or f.shape != y.shape or mask.shape != y.shape:
        raise ValueError("observations, frequencies, and mask must be matching vectors")
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two pilot tones are required")
    if not np.isfinite(y.real).all() or not np.isfinite(y.imag).all() or not np.isfinite(f).all():
        raise ValueError("observations and frequencies must be finite")
    if not math.isfinite(maximum_delay_s) or maximum_delay_s <= 0:
        raise ValueError("maximum_delay_s must be finite and positive")
    if not math.isfinite(noise_variance) or noise_variance <= 0:
        raise ValueError("noise_variance must be finite and positive")
    if isinstance(grid_points, bool) or not isinstance(grid_points, int) or grid_points < 17:
        raise ValueError("grid_points must be an integer >= 17")
    y = y[mask]
    f = f[mask]
    grid = np.linspace(0.0, maximum_delay_s, grid_points)
    residual_power = np.asarray(
        [_profiled_delay_residual_power(y, f, float(delay)) for delay in grid]
    )
    deviance = 2.0 * (residual_power - float(np.min(residual_power))) / noise_variance
    return grid, np.maximum(deviance, 0.0)


def _profile_set_components(
    grid: np.ndarray, deviance: np.ndarray, threshold: float
) -> list[list[float]]:
    """Convert a sampled likelihood sublevel set to interpolated intervals."""
    if math.isinf(threshold):
        return [[float(grid[0]), float(grid[-1])]]
    included = deviance <= threshold
    components: list[list[float]] = []
    index = 0
    while index < included.size:
        if not included[index]:
            index += 1
            continue
        start = index
        while index + 1 < included.size and included[index + 1]:
            index += 1
        end = index
        if start == 0:
            lower = float(grid[0])
        else:
            x0, x1 = float(grid[start - 1]), float(grid[start])
            y0, y1 = float(deviance[start - 1]), float(deviance[start])
            lower = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
        if end == included.size - 1:
            upper = float(grid[-1])
        else:
            x0, x1 = float(grid[end]), float(grid[end + 1])
            y0, y1 = float(deviance[end]), float(deviance[end + 1])
            upper = x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)
        components.append([lower, upper])
        index += 1
    return components


def evaluate_profile_likelihood_sets(
    dataset: TwoPathDataset,
    *,
    coverage_level: float = 0.95,
    grid_points: int = 257,
) -> dict[str, object]:
    """Calibrate and evaluate group-simultaneous profile-likelihood delay sets.

    Calibration uses the maximum true-delay deviance within each independent
    calibration group and the split-conformal finite-sample order statistic.
    Test sets may contain disconnected delay intervals. The guarantee requires
    exchangeable independent condition groups and the declared fixed-order,
    known-noise analytical model; it does not transfer to RT scenes by itself.
    """
    if not math.isfinite(coverage_level) or not 0.0 < coverage_level < 1.0:
        raise ValueError("coverage_level must be finite and in (0, 1)")
    calibration = dataset.indices("calibration")
    test = dataset.indices("test")
    if not calibration.size or not test.size:
        raise ValueError("calibration and test splits must both be nonempty")
    calibration_groups = dataset.group_id[calibration].astype(str)
    test_groups = dataset.group_id[test].astype(str)
    overlap = set(calibration_groups).intersection(test_groups)
    if overlap:
        raise ValueError(f"calibration and test groups overlap: {sorted(overlap)}")

    group_scores: dict[str, float] = {}
    for sample_index, group in zip(calibration, calibration_groups, strict=True):
        grid, deviance = profile_likelihood_deviance_grid(
            dataset.observed_cfr[sample_index],
            dataset.pilot_mask[sample_index],
            dataset.frequencies_hz[sample_index],
            dataset.maximum_relative_delay_s,
            float(dataset.noise_variance[sample_index]),
            grid_points=grid_points,
        )
        target = float(dataset.relative_delay_s[sample_index])
        score = float(np.interp(target, grid, deviance))
        group_scores[str(group)] = max(group_scores.get(str(group), 0.0), score)

    ordered_scores = np.sort(np.asarray(list(group_scores.values()), dtype=np.float64))
    rank = math.ceil((ordered_scores.size + 1) * coverage_level)
    unbounded = rank > ordered_scores.size
    threshold = math.inf if unbounded else float(ordered_scores[rank - 1])

    rows: list[dict[str, object]] = []
    row_coverage: list[bool] = []
    row_measure: list[float] = []
    row_component_count: list[int] = []
    row_groups: list[str] = []
    row_strata: list[str] = []
    for sample_index, group in zip(test, test_groups, strict=True):
        grid, deviance = profile_likelihood_deviance_grid(
            dataset.observed_cfr[sample_index],
            dataset.pilot_mask[sample_index],
            dataset.frequencies_hz[sample_index],
            dataset.maximum_relative_delay_s,
            float(dataset.noise_variance[sample_index]),
            grid_points=grid_points,
        )
        components = _profile_set_components(grid, deviance, threshold)
        target = float(dataset.relative_delay_s[sample_index])
        covered = any(lower <= target <= upper for lower, upper in components)
        measure = float(sum(upper - lower for lower, upper in components))
        stratum = str(dataset.stratum[sample_index])
        row_coverage.append(covered)
        row_measure.append(measure)
        row_component_count.append(len(components))
        row_groups.append(str(group))
        row_strata.append(stratum)
        rows.append(
            {
                "sample_id": str(dataset.sample_id[sample_index]),
                "group_id": str(group),
                "stratum": stratum,
                "target_delay_s": target,
                "covered": covered,
                "components_s": components,
                "component_count": len(components),
                "total_set_measure_s": measure,
            }
        )

    test_group_coverage = {
        group: bool(
            all(
                covered
                for row_group, covered in zip(row_groups, row_coverage, strict=True)
                if row_group == group
            )
        )
        for group in sorted(set(row_groups))
    }
    by_stratum: dict[str, dict[str, float | int]] = {}
    for stratum in sorted(set(row_strata)):
        selected_indices = [index for index, value in enumerate(row_strata) if value == stratum]
        by_stratum[stratum] = {
            "sample_count": len(selected_indices),
            "sample_coverage": float(np.mean([row_coverage[index] for index in selected_indices])),
            "mean_total_set_measure_s": float(
                np.mean([row_measure[index] for index in selected_indices])
            ),
            "mean_component_count": float(
                np.mean([row_component_count[index] for index in selected_indices])
            ),
        }
    return {
        "schema": "loslab.profile_likelihood_delay_sets.v1",
        "status": "DESCRIPTIVE_ANALYTICAL_PROFILE_SET_EVALUATION",
        "model": "fixed_two_path_complex_gain_profile; known independent complex Gaussian noise",
        "calibration": {
            "coverage_level": coverage_level,
            "group_count": int(ordered_scores.size),
            "sample_count": int(calibration.size),
            "group_max_true_delay_deviance": ordered_scores.tolist(),
            "finite_sample_unbounded": unbounded,
            "deviance_threshold": None if unbounded else threshold,
            "order_statistic_rank_one_based": rank,
        },
        "test": {
            "group_count": int(len(test_group_coverage)),
            "sample_count": len(rows),
            "sample_coverage": float(np.mean(row_coverage)),
            "group_simultaneous_coverage": float(np.mean(list(test_group_coverage.values()))),
            "mean_total_set_measure_s": float(np.mean(row_measure)),
            "mean_component_count": float(np.mean(row_component_count)),
            "group_coverage": test_group_coverage,
            "by_stratum_descriptive": by_stratum,
            "rows": rows,
        },
        "grid_points": grid_points,
        "maximum_relative_delay_s": dataset.maximum_relative_delay_s,
        "interpretation": (
            "Intervals approximate the profile-likelihood sublevel set on a bounded grid and may be "
            "disconnected. Group coverage requires exchangeable independent groups; stratum results "
            "are descriptive. The fixed two-path and known-noise assumptions limit transfer."
        ),
    }


def fit_two_path_decoder(
    dataset: TwoPathDataset,
    *,
    seed: int = 42,
    epochs: int = 120,
    batch_size: int = 128,
    patience: int = 15,
    device: str = "auto",
    identifiability_weight: float = 0.0,
) -> tuple[Any, dict[str, object]]:
    """Train the differentiable decoder using train groups and select on validation."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the analytical decoder experiment") from exc
    from models.observable_multipath import (
        build_two_path_uncertainty_decoder,
        complex_pilot_gaussian_nll,
        heteroscedastic_delay_nll,
        identifiability_scale_floor_loss,
        render_two_path_cfr,
    )

    for name, value, minimum in (
        ("epochs", epochs, 1),
        ("batch_size", batch_size, 1),
        ("patience", patience, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(identifiability_weight) or identifiability_weight < 0:
        raise ValueError("identifiability_weight must be finite and nonnegative")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_idx = dataset.indices("train")
    val_idx = dataset.indices("validation")
    if not train_idx.size or not val_idx.size:
        raise ValueError("training and validation splits must be nonempty")
    model = build_two_path_uncertainty_decoder(
        dataset.observed_cfr.shape[1], dataset.maximum_relative_delay_s
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def tensors(indices: np.ndarray) -> tuple[Any, ...]:
        return (
            torch.as_tensor(dataset.observed_cfr[indices], device=device),
            torch.as_tensor(dataset.pilot_mask[indices], device=device),
            torch.as_tensor(dataset.frequencies_hz[indices], device=device),
            torch.as_tensor(dataset.noise_variance[indices], device=device),
            torch.as_tensor(dataset.relative_delay_s[indices], device=device),
        )

    train = tensors(train_idx)
    validation = tensors(val_idx)

    def objective(batch: tuple[Any, ...]) -> Any:
        observed, masks, frequencies, variance, target_delay = batch
        output = model(observed, masks, frequencies, variance)
        rendered = render_two_path_cfr(frequencies, output["gains"], output["relative_delay_s"])
        reconstruction = complex_pilot_gaussian_nll(observed, rendered, variance, masks)
        normalized_target = target_delay / dataset.maximum_relative_delay_s
        normalized_mean = output["relative_delay_s"] / dataset.maximum_relative_delay_s
        normalized_scale = output["relative_delay_std_s"] / dataset.maximum_relative_delay_s
        delay_nll = heteroscedastic_delay_nll(
            normalized_target, normalized_mean, torch.log(normalized_scale)
        )
        if identifiability_weight:
            information_floor = identifiability_scale_floor_loss(
                output["relative_delay_std_s"],
                output["gains"],
                output["relative_delay_s"],
                frequencies,
                variance,
                masks,
                dataset.maximum_relative_delay_s,
            )
        else:
            information_floor = 0.0
        return reconstruction + delay_nll + identifiability_weight * information_floor

    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_idx.size, device=device)
        train_loss_sum = 0.0
        seen = 0
        for start in range(0, train_idx.size, batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch = tuple(value[batch_indices] for value in train)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(batch)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite training objective at epoch {epoch + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            count = len(batch_indices)
            train_loss_sum += float(loss.detach()) * count
            seen += count

        model.eval()
        with torch.no_grad():
            val_loss = float(objective(validation).detach())
        train_loss = train_loss_sum / seen
        history.append(
            {"epoch": float(epoch + 1), "train_loss": train_loss, "validation_loss": val_loss}
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "device": device,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "identifiability_weight": identifiability_weight,
        "history": history,
        "train_group_count": int(np.unique(dataset.group_id[train_idx]).size),
        "validation_group_count": int(np.unique(dataset.group_id[val_idx]).size),
    }


def predict_decoder(
    model: Any, dataset: TwoPathDataset, indices: np.ndarray, *, device: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return delay, positive uncertainty scale, and CFR reconstruction arrays."""
    import torch
    from models.observable_multipath import render_two_path_cfr

    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= dataset.observed_cfr.shape[0]):
        raise ValueError("indices must be a valid one-dimensional index array")
    with torch.no_grad():
        observed = torch.as_tensor(dataset.observed_cfr[indices], device=device)
        masks = torch.as_tensor(dataset.pilot_mask[indices], device=device)
        frequencies = torch.as_tensor(dataset.frequencies_hz[indices], device=device)
        variance = torch.as_tensor(dataset.noise_variance[indices], device=device)
        output = model(observed, masks, frequencies, variance)
        reconstruction = render_two_path_cfr(
            frequencies, output["gains"], output["relative_delay_s"]
        )
    return (
        output["relative_delay_s"].detach().cpu().numpy(),
        output["relative_delay_std_s"].detach().cpu().numpy(),
        reconstruction.detach().cpu().numpy(),
    )


def dataset_source_sha256(dataset: TwoPathDataset) -> str:
    """Return the deterministic content identity of generated observable and oracle arrays."""
    digest = hashlib.sha256()
    for array in (
        dataset.observed_cfr,
        dataset.pilot_mask,
        dataset.frequencies_hz,
        dataset.noise_variance,
        dataset.relative_delay_s,
        dataset.gains,
        dataset.group_id,
        dataset.split,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()
