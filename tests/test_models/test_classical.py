from __future__ import annotations

import pytest
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
