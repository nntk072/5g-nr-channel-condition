"""Supervised fitting entry points for the standard baselines."""

from __future__ import annotations

import numpy as np
from models.classical import make_estimator


def fit_sklearn(x: np.ndarray, y: np.ndarray, seed: int = 42, method: str = "mlp"):
    """Fit a named scikit-learn baseline."""
    return make_estimator(method, seed).fit(x, y)
