from __future__ import annotations

import pytest
from learning.supervised import fit_sklearn
from models.classical import make_estimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


@pytest.mark.parametrize("method", ["rf", "mlp", "logreg"])
def test_factory_builds_supported_baselines(method) -> None:
    estimator = make_estimator(method, seed=21)
    if method == "rf":
        assert isinstance(estimator, RandomForestClassifier)
    else:
        assert isinstance(estimator, Pipeline)
        expected = MLPClassifier if method == "mlp" else LogisticRegression
        assert isinstance(estimator.steps[-1][1], expected)


def test_factory_rejects_unknown_baseline() -> None:
    with pytest.raises(ValueError, match="Unknown estimator"):
        make_estimator("unknown")


def test_small_mlp_fit_uses_effective_clipped_batch_size() -> None:
    import numpy as np

    features = np.arange(36, dtype=np.float32).reshape(12, 3)
    labels = np.tile([0, 1], 6)
    fitted = fit_sklearn(features, labels, seed=21, method="mlp")

    expected_training_rows = len(labels) - int(np.ceil(0.1 * len(labels)))
    assert fitted.named_steps["mlpclassifier"].batch_size == expected_training_rows
