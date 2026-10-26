"""Leakage and feature-contract checks for profile-disjoint evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from experiments.profile_holdout import (
    BASELINE_CONFIGURATION,
    _model_complexity,
    normalized_cfr_features,
    run_profile_holdout,
)
from sklearn.ensemble import RandomForestClassifier


def _small_profile_dataset() -> ChannelDataset:
    rng = np.random.default_rng(73)
    channels = []
    labels = []
    groups = []
    domains = []
    sources = []
    metadata = []
    for family in ("tdl", "cdl"):
        for profile in "ABCDE":
            label = int(profile in "DE")
            for link in range(20):
                # The profile and family fields are deliberately metadata only.
                channel = rng.normal(size=16) + 1j * rng.normal(size=16)
                channels.append(channel[None, :])
                labels.append(label)
                groups.append(f"{family}_{profile}_{link}")
                domains.append(family)
                sources.append(family)
                metadata.append(
                    {
                        "family": family,
                        "profile": profile,
                        "label_semantics": "PROFILE_PROXY",
                    }
                )
    return ChannelDataset(
        h=np.asarray(channels, dtype=np.complex64),
        y=np.asarray(labels),
        group=np.asarray(groups),
        domain=np.asarray(domains),
        source=np.asarray(sources),
        metadata=tuple(metadata),
        label_semantics="profile_proxy",
    )


def test_normalized_cfr_features_ignore_common_complex_gain():
    rng = np.random.default_rng(41)
    channel = rng.normal(size=(8, 2, 16)) + 1j * rng.normal(size=(8, 2, 16))
    gain = np.asarray([0.3 + 0.7j, -1.2 + 0.4j, 2.0j, -0.6] * 2)

    base = normalized_cfr_features(channel)
    transformed = normalized_cfr_features(channel * gain[:, None, None])

    np.testing.assert_allclose(base, transformed, atol=2e-7, rtol=2e-7)


def test_profile_holdout_keeps_outer_profiles_and_validation_groups_disjoint():
    result = run_profile_holdout(
        _small_profile_dataset(), methods=("logreg", "mlp"), seed=23, validation_fraction=0.25
    )

    assert len(result["folds"]) == 10
    assert set(result["summary"]["logreg"]["per_profile"]) == set("ABCDE")
    metadata_control = result["metadata_only_profile_control"]
    assert metadata_control["feature_fields"] == ["metadata.profile"]
    assert metadata_control["accuracy"] == 1.0
    assert metadata_control["confusion_matrix_labels_0_1"] == [[120, 0], [0, 80]]
    assert metadata_control["label_semantics"] == "PROFILE_PROXY: A/B/C=0, D/E=1"
    assert metadata_control["physical_geometric_los_inference"] is False
    assert "not geometric LOS detection" in metadata_control["interpretation"]
    for method in ("logreg", "mlp"):
        assert np.isfinite(result["predictions"][method]).all()
        assert np.isfinite(result["thresholds"][method]).all()
    for row in result["folds"]:
        assert row["profile"] not in row["train_profiles"]
        assert row["train_groups"] > 0
        assert row["validation_groups"] > 0
        assert row["test_groups"] == 40
        assert 0.0 <= row["profile_recall_at_validation_threshold"] <= 1.0
        resources = row["resource_use"]
        assert resources["fit_wall_seconds"] >= 0.0
        assert resources["heldout_prediction_wall_seconds"] >= 0.0
        assert resources["training_samples"] > 0
        assert resources["input_features"] == 32
        assert resources["training_input_bytes"] == (
            resources["training_samples"] * resources["input_features"] * 4
        )
        assert resources["learned_parameter_count"] > 0
        assert resources["serialized_model_bytes"] > 0


def test_model_complexity_reports_random_forest_structure():
    model = RandomForestClassifier(n_estimators=3, max_depth=2, random_state=41)
    model.fit(np.arange(40, dtype=np.float32).reshape(20, 2), np.tile([0, 1], 10))

    complexity = _model_complexity(model)

    assert complexity["learned_parameter_count"] == 0
    assert complexity["tree_count"] == 3
    assert complexity["tree_node_count"] >= complexity["tree_leaf_count"]
    assert complexity["tree_leaf_count"] >= complexity["tree_count"]
    assert complexity["maximum_tree_depth"] <= 2
    assert complexity["serialized_model_bytes"] > 0


def test_profile_holdout_rejects_inconsistent_proxy_labels():
    data = _small_profile_dataset()
    wrong = data.select(np.arange(len(data.y)))
    labels = wrong.y.copy()
    labels[0] = 1
    corrupted = ChannelDataset(
        wrong.h,
        labels,
        wrong.group,
        wrong.domain,
        wrong.source,
        wrong.metadata,
    )

    with pytest.raises(ValueError, match="does not match"):
        run_profile_holdout(corrupted, methods=("logreg",))


def test_profile_holdout_rejects_conflicting_dataset_level_label_semantics():
    data = _small_profile_dataset()
    mislabeled = ChannelDataset(
        data.h,
        data.y,
        data.group,
        data.domain,
        data.source,
        data.metadata,
        label_semantics="geometric_direct_path_oracle",
    )

    with pytest.raises(ValueError, match="dataset-level profile_proxy"):
        run_profile_holdout(mislabeled, methods=("logreg",))


def test_random_forest_baseline_configuration_is_single_worker():
    assert BASELINE_CONFIGURATION["rf"]["n_estimators"] == 250
    assert BASELINE_CONFIGURATION["rf"]["n_jobs"] == 1
