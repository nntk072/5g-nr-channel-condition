"""Family-wise DKW checks for fixed-TDL first-tap Rician fading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ncx2


@dataclass(frozen=True, slots=True)
class RicianMarginalResult:
    """Normalized-power CDF distance and simultaneous DKW upper bound."""

    model: str
    draw_count: int
    diffuse_power: float
    specular_power: float
    empirical_ks_distance: float
    confidence_radius: float
    simultaneous_upper_bound: float
    passed: bool


def rician_normalized_power_cdf(
    normalized_power: NDArray[np.generic], *, diffuse_power: float, specular_power: float
) -> NDArray[np.float64]:
    """Evaluate the Rician power CDF from independently referenced components.

    ``diffuse_power`` and ``specular_power`` are linear mean powers for one
    resolved tap. The input is normalized by their sum. A circular complex
    Gaussian diffuse component has per-quadrature variance ``diffuse_power/2``.
    """
    values = np.asarray(normalized_power)
    if values.dtype.kind not in "iuf" or values.ndim != 1 or values.size == 0:
        raise ValueError("normalized_power must be a nonempty real vector")
    values = values.astype(np.float64, copy=False)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("normalized_power must contain finite nonnegative values")
    diffuse = _positive_power(diffuse_power, "diffuse_power")
    specular = _nonnegative_power(specular_power, "specular_power")
    if diffuse <= 0.0:
        raise ValueError("diffuse_power must be positive for a Rician marginal")
    scale = diffuse + specular
    if not math.isfinite(scale):
        raise ValueError("total first-tap power must be finite")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = np.asarray(
            ncx2.cdf(
                2.0 * values * scale / diffuse,
                df=2,
                nc=2.0 * specular / diffuse,
            ),
            dtype=np.float64,
        )
    if not np.isfinite(result).all():
        raise ValueError("Rician CDF parameters exceed the supported numeric range")
    return result


def score_tdl_rician_marginal_family(
    samples_by_model: Mapping[str, NDArray[np.generic]],
    reference_powers: Mapping[str, tuple[float, float]],
    *,
    alpha: float = 0.01,
    maximum_simultaneous_cdf_error: float = 0.05,
) -> tuple[RicianMarginalResult, ...]:
    """Score first-tap powers against exact ETSI-derived Rician marginals.

    The two-sided DKW radius is Bonferroni-adjusted over the complete model
    family. The sample axis must represent independent channel draws; taps or
    time samples within one draw must not be pooled as additional replicates.
    """
    if not isinstance(samples_by_model, Mapping) or not samples_by_model:
        raise ValueError("samples_by_model must be a nonempty model mapping")
    if not isinstance(reference_powers, Mapping) or set(samples_by_model) != set(reference_powers):
        raise ValueError("sample and independent-reference model rosters must match exactly")
    models = tuple(samples_by_model)
    if any(not isinstance(model, str) or not model or model != model.strip() for model in models):
        raise ValueError("model identities must be nonempty canonical strings")
    alpha_value = _open_unit_interval(alpha, "alpha")
    maximum_error = _open_unit_interval(
        maximum_simultaneous_cdf_error, "maximum_simultaneous_cdf_error"
    )
    family_size = len(samples_by_model)
    results: list[RicianMarginalResult] = []
    for model in sorted(models):
        samples = np.asarray(samples_by_model[model])
        if samples.dtype.kind != "c" or samples.ndim != 1 or samples.size == 0:
            raise ValueError(f"{model} first-tap samples must be a nonempty complex vector")
        if not np.isfinite(samples).all():
            raise ValueError(f"{model} first-tap samples must be finite")
        powers = reference_powers[model]
        if not isinstance(powers, tuple) or len(powers) != 2:
            raise ValueError("reference powers must be (diffuse_power, specular_power) tuples")
        diffuse = _positive_power(powers[0], "diffuse_power")
        specular = _nonnegative_power(powers[1], "specular_power")
        total = diffuse + specular
        if not math.isfinite(total):
            raise ValueError("total first-tap power must be finite")
        normalized_power = np.square(np.abs(samples.astype(np.complex128, copy=False))) / total
        if not np.isfinite(normalized_power).all():
            raise ValueError(f"{model} normalized first-tap powers must be finite")
        ordered = np.sort(normalized_power)
        expected_cdf = rician_normalized_power_cdf(
            ordered, diffuse_power=diffuse, specular_power=specular
        )
        count = len(ordered)
        upper_empirical = np.arange(1, count + 1, dtype=np.float64) / count
        lower_empirical = np.arange(0, count, dtype=np.float64) / count
        distance = float(
            max(
                np.max(upper_empirical - expected_cdf),
                np.max(expected_cdf - lower_empirical),
            )
        )
        radius = math.sqrt(math.log(2.0 * family_size / alpha_value) / (2.0 * count))
        upper = distance + radius
        results.append(
            RicianMarginalResult(
                model=model,
                draw_count=count,
                diffuse_power=diffuse,
                specular_power=specular,
                empirical_ks_distance=distance,
                confidence_radius=radius,
                simultaneous_upper_bound=upper,
                passed=upper <= maximum_error,
            )
        )
    return tuple(results)


def _positive_power(value: object, name: str) -> float:
    result = _real_power(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_power(value: object, name: str) -> float:
    result = _real_power(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _real_power(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _open_unit_interval(value: object, name: str) -> float:
    result = _real_power(value, name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return result
