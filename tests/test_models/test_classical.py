from __future__ import annotations

import pytest
from learning.supervised import fit_sklearn
from models.classical import make_estimator
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


@pytest.mark.parametrize("method", ["rf", "histgb", "mlp", "logreg"])
def test_factory_builds_supported_baselines(method) -> None:
    estimator = make_estimator(method, seed=21)
    if method == "rf":
        assert isinstance(estimator, RandomForestClassifier)
    elif method == "histgb":
        assert isinstance(estimator, HistGradientBoostingClassifier)
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


def test_histogram_gradient_boosting_trains_and_returns_probabilities() -> None:
    import numpy as np
    from sklearn.datasets import make_classification

    features, labels = make_classification(
        n_samples=80, n_features=6, n_informative=4, random_state=21
    )
    fitted = fit_sklearn(features, labels, seed=21, method="histgb")
    probabilities = fitted.predict_proba(features)

    assert probabilities.shape == (80, 2)
    assert np.isfinite(probabilities).all()


def test_xgboost_trains_with_binary_class_balance_and_returns_probabilities() -> None:
    pytest.importorskip("xgboost")
    import numpy as np

    rng = np.random.default_rng(91)
    features = rng.normal(size=(48, 6)).astype(np.float32)
    labels = np.asarray([0] * 32 + [1] * 16)
    features[:, 0] += labels * 1.5
    fitted = fit_sklearn(features, labels, seed=29, method="xgb")
    probabilities = fitted.predict_proba(features)

    assert fitted.scale_pos_weight == pytest.approx(2.0)
    assert probabilities.shape == (48, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_xgboost_rejects_nonbinary_or_single_class_targets() -> None:
    pytest.importorskip("xgboost")
    import numpy as np

    features = np.ones((8, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="both labels 0 and 1"):
        fit_sklearn(features, np.zeros(8, dtype=np.int64), method="xgb")
