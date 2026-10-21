"""Estimator construction for the classical research baselines."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_estimator(method: str, seed: int = 42, capacity: str = "medium"):
    """Create an unfitted baseline estimator with fixed research defaults."""
    from models.capacity import capacity_spec

    if method == "mlp":
        hidden_layer_sizes = capacity_spec(capacity).mlp_widths
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation="relu",
                alpha=1e-3,
                batch_size=64,
                early_stopping=True,
                n_iter_no_change=18,
                max_iter=300,
                random_state=seed,
            ),
        )
    if method == "rf":
        return RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if method == "histgb":
        return HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    if method == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise ImportError(
                "XGBoost baseline requires the optional 'boosting' dependency extra"
            ) from error
        return XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
        )
    if method == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
    raise ValueError(f"Unknown estimator {method}")
