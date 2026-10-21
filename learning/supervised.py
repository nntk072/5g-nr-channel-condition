"""Supervised fitting entry points for the standard baselines."""

from __future__ import annotations

import numpy as np
from models.classical import make_estimator


def fit_sklearn(x: np.ndarray, y: np.ndarray, seed: int = 42, method: str = "mlp"):
    """Fit a named scikit-learn baseline."""
    estimator = make_estimator(method, seed)
    if method == "mlp":
        # Preserve sklearn's effective clipping of the configured 64-row
        # minibatch after its 10% early-stopping split on small pilots.
        early_stopping_train_size = len(y) - int(np.ceil(0.1 * len(y)))
        batch_size = max(1, min(64, early_stopping_train_size))
        estimator.named_steps["mlpclassifier"].set_params(batch_size=batch_size)
    return estimator.fit(x, y)
