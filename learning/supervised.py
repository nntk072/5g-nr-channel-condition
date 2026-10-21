"""Supervised fitting entry points for the standard baselines."""

from __future__ import annotations

import numpy as np
from models.classical import make_estimator


def fit_sklearn(
    x: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
    method: str = "mlp",
    capacity: str = "medium",
):
    """Fit a named scikit-learn baseline."""
    estimator = make_estimator(method, seed, capacity)
    if method == "mlp":
        # Preserve sklearn's effective clipping of the configured 64-row
        # minibatch after its 10% early-stopping split on small pilots.
        early_stopping_train_size = len(y) - int(np.ceil(0.1 * len(y)))
        batch_size = max(1, min(64, early_stopping_train_size))
        estimator.named_steps["mlpclassifier"].set_params(batch_size=batch_size)
    if method == "xgb":
        labels = np.asarray(y)
        if not np.isin(labels, (0, 1)).all() or set(np.unique(labels)) != {0, 1}:
            raise ValueError("XGBoost binary baseline requires both labels 0 and 1")
        counts = np.bincount(labels.astype(np.int64), minlength=2)
        estimator.set_params(scale_pos_weight=float(counts[0] / counts[1]))
    return estimator.fit(x, y)
