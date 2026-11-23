"""Closed-form reference values and finite-number helpers for recomputed waveform diagnostics."""

from __future__ import annotations

import math
from typing import Any, TypeGuard

import numpy as np


def _same_float(actual: Any, expected: Any) -> bool:
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and not isinstance(expected, bool)
        and isinstance(expected, (int, float))
        and math.isfinite(float(actual))
        and math.isfinite(float(expected))
        and math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12)
    )


def _is_finite_nonnegative(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _is_finite_number(value: Any) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _iq_imbalance_reference(
    values: np.ndarray, *, gain_imbalance_db: float, phase_imbalance_rad: float
) -> np.ndarray:
    i_gain = 10.0 ** (gain_imbalance_db / 40.0)
    q_gain = 10.0 ** (-gain_imbalance_db / 40.0)
    return (
        i_gain * values.real
        - q_gain * math.sin(phase_imbalance_rad) * values.imag
        + 1j * q_gain * math.cos(phase_imbalance_rad) * values.imag
    )


def _path_sum_reference(
    values: np.ndarray, coefficients: list[list[float]], delays: list[int]
) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.complex128)
    for coefficient_pair, delay in zip(coefficients, delays, strict=True):
        coefficient = complex(float(coefficient_pair[0]), float(coefficient_pair[1]))
        if delay == 0:
            result += coefficient * values
        else:
            result[..., delay:] += coefficient * values[..., :-delay]
    return result


def _positive_integer_delay_reference(values: np.ndarray, delay: int) -> np.ndarray:
    result = np.zeros_like(values)
    result[..., delay:] = values[..., :-delay]
    return result
