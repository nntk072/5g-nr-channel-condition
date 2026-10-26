"""Profile-disjoint evaluation for standardized TDL/CDL proxy labels."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from copy import deepcopy
from time import perf_counter
from typing import Any, Sequence

import numpy as np
from datasets.channel import ChannelDataset
from evaluation.metrics import threshold_from_validation
from models.classical import make_estimator
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASELINE_CONFIGURATION: dict[str, dict[str, Any]] = {
    "logreg": {
        "estimator": "StandardScaler + LogisticRegression",
        "max_iter": 1000,
        "class_weight": "balanced",
    },
    "mlp": {
        "estimator": "StandardScaler + MLPClassifier",
        "hidden_layer_sizes": [128, 64, 32],
        "activation": "relu",
        "alpha": 1e-3,
        "batch_size": 64,
        "max_epochs": 300,
        "validation_patience_epochs": 18,
        "validation_min_delta": 1e-4,
    },
    "rf": {
        "estimator": "RandomForestClassifier",
        "n_estimators": 250,
        "min_samples_leaf": 3,
        "class_weight": "balanced_subsample",
        "n_jobs": 1,
    },
}


def normalized_cfr_features(h: np.ndarray) -> np.ndarray:
    """Return global-gain/phase normalized complex CFR samples as real features."""
    channel = np.asarray(h)
    if channel.ndim != 3 or not np.iscomplexobj(channel) or not np.isfinite(channel).all():
        raise ValueError("h must be a finite complex [samples, links, tones] array")
    power = np.mean(np.abs(channel) ** 2, axis=(1, 2), keepdims=True)
    if np.any(power <= np.finfo(np.float64).tiny):
        raise ValueError("each channel sample must have positive mean power")
    normalized = channel / np.sqrt(power)
    flat = normalized.reshape(len(normalized), -1)
    anchor = flat[np.arange(len(flat)), np.argmax(np.abs(flat), axis=1)]
    normalized = normalized * np.exp(-1j * np.angle(anchor))[:, None, None]
    flat = normalized.reshape(len(normalized), -1)
    return np.concatenate((flat.real, flat.imag), axis=1).astype(np.float32)


def _profile_rows(data: ChannelDataset) -> dict[str, np.ndarray]:
    if data.label_semantics != "profile_proxy":
        raise ValueError(
            "profile-disjoint evaluation requires dataset-level profile_proxy label semantics"
        )
    rows: dict[str, list[int]] = {}
    group_profile: dict[str, str] = {}
    for index, (group, metadata, label, family) in enumerate(
        zip(data.group, data.metadata, data.y, data.domain, strict=True)
    ):
        profile = str(metadata.get("profile", "")).upper()
        family_name = str(metadata.get("family", "")).lower()
        if metadata.get("label_semantics") != "PROFILE_PROXY":
            raise ValueError("all rows must use PROFILE_PROXY label semantics")
        if family_name not in {"tdl", "cdl"} or str(family).lower() != family_name:
            raise ValueError("metadata family must match a TDL/CDL dataset domain")
        if profile not in {"A", "B", "C", "D", "E"}:
            raise ValueError(f"unknown standardized profile: {profile!r}")
        expected_label = int(profile in "DE")
        if int(label) != expected_label:
            raise ValueError("profile-proxy label does not match the declared A/B/C vs D/E rule")
        group_name = str(group)
        previous = group_profile.setdefault(group_name, profile)
        if previous != profile:
            raise ValueError(f"link group crosses standardized profiles: {group_name}")
        rows.setdefault(profile, []).append(index)
    if set(rows) != {"A", "B", "C", "D", "E"}:
        raise ValueError("the dataset must contain each standardized profile A through E")
    return {profile: np.asarray(indices, dtype=np.int64) for profile, indices in rows.items()}


def _metadata_only_profile_control(data: ChannelDataset) -> dict[str, Any]:
    """Measure the label-definition shortcut using profile metadata alone.

    This control is intentionally an oracle for the PROFILE_PROXY target: the
    target is defined by whether a standardized profile is A/B/C or D/E. A
    perfect score demonstrates metadata recoverability, not geometric LOS
    inference or receiver-observable performance.
    """
    prediction = np.fromiter(
        (int(str(metadata["profile"]).upper() in "DE") for metadata in data.metadata),
        dtype=np.int64,
        count=len(data.metadata),
    )
    truth = np.asarray(data.y, dtype=np.int64)
    true_negative = int(np.sum((truth == 0) & (prediction == 0)))
    false_positive = int(np.sum((truth == 0) & (prediction == 1)))
    false_negative = int(np.sum((truth == 1) & (prediction == 0)))
    true_positive = int(np.sum((truth == 1) & (prediction == 1)))
    return {
        "feature_fields": ["metadata.profile"],
        "n_samples": int(len(truth)),
        "accuracy": float(np.mean(prediction == truth)),
        "confusion_matrix_labels_0_1": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
        "label_semantics": "PROFILE_PROXY: A/B/C=0, D/E=1",
        "physical_geometric_los_inference": False,
        "interpretation": (
            "This recovers the profile-proxy label definition from simulator metadata; "
            "it is not geometric LOS detection or receiver-observable performance."
        ),
    }


def _group_validation_split(
    y: np.ndarray, groups: np.ndarray, *, seed: int, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    if len(np.unique(groups)) < 4:
        raise ValueError("at least four training link groups are needed for validation")
    for offset in range(100):
        train, validation = next(
            GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed + offset).split(
                np.zeros(len(y)), y, groups
            )
        )
        if len(np.unique(y[train])) == len(np.unique(y[validation])) == 2:
            if set(groups[train]).isdisjoint(groups[validation]):
                return train, validation
    raise ValueError("could not make a two-class, group-disjoint inner split")


def _fit(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    if method == "mlp":
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        x_validation_scaled = scaler.transform(x_validation)
        estimator = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=1e-3,
            batch_size=min(64, len(y)),
            early_stopping=False,
            max_iter=1,
            random_state=seed,
            tol=0.0,
        )
        best_loss = float("inf")
        best_estimator: MLPClassifier | None = None
        epochs_without_improvement = 0
        epochs = 0
        for epoch in range(300):
            if epoch == 0:
                estimator.partial_fit(x_scaled, y, classes=np.asarray([0, 1]))
            else:
                estimator.partial_fit(x_scaled, y)
            epochs = epoch + 1
            validation_probability = estimator.predict_proba(x_validation_scaled)
            current_loss = float(log_loss(y_validation, validation_probability, labels=[0, 1]))
            if current_loss < best_loss - 1e-4:
                best_loss = current_loss
                best_estimator = deepcopy(estimator)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if epochs_without_improvement >= 18:
                break
        if best_estimator is None:
            raise RuntimeError("MLP did not produce a finite group-validation checkpoint")
        return make_pipeline(scaler, best_estimator), {
            "epochs_run": epochs,
            "best_validation_log_loss": best_loss,
            "early_stopping": "group-disjoint validation log loss; min_delta=1e-4; patience=18",
        }

    estimator = make_estimator(method, seed)
    if method == "rf":
        estimator.set_params(n_jobs=1)
    return estimator.fit(x, y), {"early_stopping": "not applicable"}


def _model_complexity(estimator: Any) -> dict[str, int]:
    """Describe fitted model size without treating incomparable units as equivalent."""
    import numpy as np

    components = getattr(estimator, "steps", [("estimator", estimator)])
    learned_parameter_count = 0
    trees: list[Any] = []
    for _, component in components:
        for attribute in ("coefs_", "intercepts_"):
            arrays = getattr(component, attribute, ())
            learned_parameter_count += sum(int(np.size(array)) for array in arrays)
        for attribute in ("coef_", "intercept_"):
            array = getattr(component, attribute, None)
            if array is not None:
                learned_parameter_count += int(np.size(array))
        trees.extend(getattr(component, "estimators_", ()))
    tree_nodes = [int(tree.tree_.node_count) for tree in trees]
    return {
        "learned_parameter_count": learned_parameter_count,
        "tree_count": len(trees),
        "tree_node_count": sum(tree_nodes),
        "tree_leaf_count": sum(int(tree.tree_.n_leaves) for tree in trees),
        "maximum_tree_depth": max((int(tree.tree_.max_depth) for tree in trees), default=0),
        "serialized_model_bytes": len(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)),
    }


def run_profile_holdout(
    data: ChannelDataset,
    *,
    methods: Sequence[str] = ("logreg", "mlp", "rf"),
    seed: int = 20261021,
    validation_fraction: float = 0.2,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Leave out each complete profile, tuning only the decision threshold in-group.

    Each outer test fold contains one profile and therefore only one proxy class.
    The output reports within-profile recall and macro averages over profiles;
    it intentionally does not pool scores from separately fitted folds into an
    AUROC or claim generalization to geometric LOS/NLOS. A metadata-only control
    reports how directly the declared profile metadata recovers the proxy label.
    """
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be a non-empty sequence without duplicates")
    unknown = set(methods) - {"logreg", "mlp", "rf"}
    if unknown:
        raise ValueError(f"unsupported methods: {sorted(unknown)}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    profile_indices = _profile_rows(data)
    model_inputs = data.model_inputs()
    features = normalized_cfr_features(model_inputs.observed_cfr)
    profile_names = tuple("ABCDE")
    scores: dict[str, np.ndarray] = {
        method: np.full(len(data.y), np.nan, dtype=np.float64) for method in methods
    }
    thresholds: dict[str, np.ndarray] = {
        method: np.full(len(data.y), np.nan, dtype=np.float64) for method in methods
    }
    fold_records: list[dict[str, Any]] = []

    for profile_index, heldout_profile in enumerate(profile_names):
        test = profile_indices[heldout_profile]
        train_all = np.concatenate(
            [profile_indices[name] for name in profile_names if name != heldout_profile]
        )
        train_local, validation_local = _group_validation_split(
            data.y[train_all],
            data.group[train_all],
            seed=seed + profile_index * 1000,
            fraction=validation_fraction,
        )
        train = train_all[train_local]
        validation = train_all[validation_local]
        train_groups = set(data.group[train].tolist())
        validation_groups = set(data.group[validation].tolist())
        test_groups = set(data.group[test].tolist())
        if not train_groups.isdisjoint(validation_groups | test_groups):
            raise ValueError("training groups overlap validation/test groups")
        if not validation_groups.isdisjoint(test_groups):
            raise ValueError("validation groups overlap held-out profile groups")
        if len(np.unique(data.y[train])) != 2 or len(np.unique(data.y[validation])) != 2:
            raise ValueError("each training and validation fold must contain both classes")

        for method_index, method in enumerate(methods):
            model_seed = seed + profile_index * 1000 + method_index
            if progress is not None:
                progress(f"profile={heldout_profile} method={method}: fitting")
            fit_started = perf_counter()
            estimator, training = _fit(
                method,
                features[train],
                data.y[train],
                features[validation],
                data.y[validation],
                model_seed,
            )
            fit_seconds = perf_counter() - fit_started
            validation_prediction_started = perf_counter()
            validation_scores = estimator.predict_proba(features[validation])[:, 1]
            validation_prediction_seconds = perf_counter() - validation_prediction_started
            threshold = threshold_from_validation(data.y[validation], validation_scores)
            test_prediction_started = perf_counter()
            test_scores = estimator.predict_proba(features[test])[:, 1]
            test_prediction_seconds = perf_counter() - test_prediction_started
            scores[method][test] = test_scores
            thresholds[method][test] = threshold
            test_class = int(data.y[test[0]])
            predictions = test_scores >= threshold
            correct = predictions if test_class == 1 else ~predictions
            fold_records.append(
                {
                    "profile": heldout_profile,
                    "method": method,
                    "label": test_class,
                    "n_test": int(len(test)),
                    "test_groups": int(len(test_groups)),
                    "train_profiles": sorted(
                        set(str(data.metadata[i]["profile"]).upper() for i in train)
                    ),
                    "train_groups": int(len(train_groups)),
                    "validation_groups": int(len(validation_groups)),
                    "train_indices_sha256": _index_hash(train),
                    "validation_indices_sha256": _index_hash(validation),
                    "test_indices_sha256": _index_hash(test),
                    "threshold_from_validation": float(threshold),
                    "training": training,
                    "resource_use": {
                        "fit_wall_seconds": fit_seconds,
                        "validation_prediction_wall_seconds": validation_prediction_seconds,
                        "heldout_prediction_wall_seconds": test_prediction_seconds,
                        "training_samples": int(len(train)),
                        "validation_samples": int(len(validation)),
                        "heldout_samples": int(len(test)),
                        "input_features": int(features.shape[1]),
                        "training_input_bytes": int(features[train].nbytes),
                        **_model_complexity(estimator),
                    },
                    "profile_recall_at_validation_threshold": float(np.mean(correct)),
                    "mean_test_positive_score": float(np.mean(test_scores)),
                }
            )

    summary: dict[str, Any] = {}
    for method in methods:
        per_profile = [row for row in fold_records if row["method"] == method]
        class_recall = {
            str(label): float(
                np.mean(
                    [
                        row["profile_recall_at_validation_threshold"]
                        for row in per_profile
                        if row["label"] == label
                    ]
                )
            )
            for label in (0, 1)
        }
        summary[method] = {
            "profile_macro_recall": float(
                np.mean([row["profile_recall_at_validation_threshold"] for row in per_profile])
            ),
            "class_macro_profile_recall": float(np.mean(list(class_recall.values()))),
            "profile_recall_by_class": class_recall,
            "per_profile": {
                row["profile"]: row["profile_recall_at_validation_threshold"] for row in per_profile
            },
        }

    if any(not np.isfinite(scores[method]).all() for method in methods):
        raise RuntimeError("outer folds did not produce one prediction for every sample")
    return {
        "schema": "loslab.profile_proxy_leave_one_profile_out.v1",
        "label_semantics": "PROFILE_PROXY: A/B/C=0, D/E=1",
        "metadata_only_profile_control": _metadata_only_profile_control(data),
        "feature_semantics": "global-gain-and-phase-normalized CFR real/imaginary samples",
        "validation_fraction_of_training_groups": validation_fraction,
        "seed": seed,
        "mlp_early_stopping": "group-disjoint outer-fold training validation, not random rows",
        "methods": list(methods),
        "baseline_configuration": {method: BASELINE_CONFIGURATION[method] for method in methods},
        "folds": fold_records,
        "summary": summary,
        "predictions": scores,
        "thresholds": thresholds,
    }


def _index_hash(indices: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
