"""Leakage and feature-contract checks for profile-disjoint evaluation."""

from __future__ import annotations

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from experiments.profile_holdout import (
    BASELINE_CONFIGURATION,
    normalized_cfr_features,
    run_profile_holdout,
)


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
    for method in ("logreg", "mlp"):
        assert np.isfinite(result["predictions"][method]).all()
        assert np.isfinite(result["thresholds"][method]).all()
    for row in result["folds"]:
        assert row["profile"] not in row["train_profiles"]
        assert row["train_groups"] > 0
        assert row["validation_groups"] > 0
        assert row["test_groups"] == 40
        assert 0.0 <= row["profile_recall_at_validation_threshold"] <= 1.0


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


def test_random_forest_baseline_configuration_is_single_worker():
    assert BASELINE_CONFIGURATION["rf"]["n_estimators"] == 250
    assert BASELINE_CONFIGURATION["rf"]["n_jobs"] == 1
