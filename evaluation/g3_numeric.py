"""Strict numeric input parsers shared by G3 planning and evaluators."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def finite_real_number(value: Any, name: str) -> float:
    """Return a finite real scalar without coercing booleans or strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def finite_real_array(values: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    """Parse a numeric array without accepting boolean/string/complex coercions."""
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a finite real array")
    try:
        raw = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real array") from exc
    if ndim is not None and raw.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional finite real array")
    parsed = [finite_real_number(item, f"{name} values") for item in raw.flat]
    return np.asarray(parsed, dtype=np.float64).reshape(raw.shape)


def finite_real_vector(values: Any, name: str) -> np.ndarray:
    """Parse a one-dimensional numeric vector without accepting coercions."""
    return finite_real_array(values, name, ndim=1)
