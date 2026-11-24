"""Control-feature builders: leakage controls fail closed on malformed rosters and seeds."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.runner.validation import (
    _metadata_only_features,
    _oracle_label_leak_features,
    _permute_training_labels,
    _phase_randomized_real_imag_cfr,
    _single_metadata_features,
    _source_only_features,
)

TRAIN = np.arange(4)


def _cfr(samples: int = 4) -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.normal(size=(samples, 2, 8)) + 1j * rng.normal(size=(samples, 2, 8))


def test_phase_randomization_preserves_magnitudes_and_validates_seed() -> None:
    cfr = _cfr()
    randomized = _phase_randomized_real_imag_cfr(cfr, seed=5)
    assert randomized.shape[0] == cfr.shape[0]
    with pytest.raises(ValueError, match="complex \\[samples, links, subcarriers\\] CFR"):
        _phase_randomized_real_imag_cfr(np.ones((4, 8)), seed=5)
    bad = _cfr()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _phase_randomized_real_imag_cfr(bad, seed=5)
    with pytest.raises(ValueError, match="phase_randomization_seed"):
        _phase_randomized_real_imag_cfr(cfr, seed=-1)
    with pytest.raises(ValueError, match="phase_randomization_seed"):
        _phase_randomized_real_imag_cfr(cfr, seed=True)


def test_label_permutation_requires_binary_integer_labels_and_seeds() -> None:
    labels = np.asarray([0, 1, 0, 1, 1])
    permuted, record = _permute_training_labels(labels, run_seed=1, base_seed=2)
    assert sorted(permuted.tolist()) == sorted(labels.tolist())
    with pytest.raises(ValueError, match="at least two binary integer"):
        _permute_training_labels(np.asarray([0.0, 1.0]), run_seed=1, base_seed=2)
    with pytest.raises(ValueError, match="at least two binary integer"):
        _permute_training_labels(np.asarray([True, False]), run_seed=1, base_seed=2)
    with pytest.raises(ValueError, match="run_seed must be an integer"):
        _permute_training_labels(labels, run_seed=-1, base_seed=2)
    with pytest.raises(ValueError, match="base_seed must be an integer"):
        _permute_training_labels(labels, run_seed=1, base_seed=2**63)


def test_source_only_control_is_a_train_vocabulary_one_hot() -> None:
    sources = np.asarray(["sim", "sim", "rt", "rt"])
    features, record = _source_only_features(sources, train_indices=TRAIN)
    assert features.shape[0] == 4
    with pytest.raises(ValueError, match="nonempty 1D source roster"):
        _source_only_features(np.asarray([]), train_indices=TRAIN)
    with pytest.raises(ValueError, match="canonical string identifiers"):
        _source_only_features(np.asarray(["sim", " rt"]), train_indices=TRAIN)
    with pytest.raises(ValueError, match="training indices are invalid"):
        _source_only_features(sources, train_indices=np.asarray([9]))
    with pytest.raises(ValueError, match="training indices are invalid"):
        _source_only_features(sources, train_indices=np.asarray([], dtype=np.int64))


def test_metadata_only_control_requires_matching_canonical_rosters() -> None:
    group = np.asarray(["g0", "g0", "g1", "g1"])
    domain = np.asarray(["d0", "d1", "d0", "d1"])
    source = np.asarray(["s0", "s0", "s1", "s1"])
    features, _ = _metadata_only_features(
        group_values=group, domain_values=domain, source_values=source, train_indices=TRAIN
    )
    assert features.shape[0] == 4
    with pytest.raises(ValueError, match="equal nonempty 1D identifier rosters"):
        _metadata_only_features(
            group_values=group[:2], domain_values=domain, source_values=source, train_indices=TRAIN
        )
    with pytest.raises(ValueError, match="group IDs must be canonical"):
        _metadata_only_features(
            group_values=np.asarray(["", "g0", "g1", "g1"]),
            domain_values=domain,
            source_values=source,
            train_indices=TRAIN,
        )
    with pytest.raises(ValueError, match="max_feature_cells must be a positive integer"):
        _metadata_only_features(
            group_values=group,
            domain_values=domain,
            source_values=source,
            train_indices=TRAIN,
            max_feature_cells=0,
        )


def test_single_metadata_control_names_its_field() -> None:
    with pytest.raises(ValueError, match="must be group or domain"):
        _single_metadata_features(
            np.asarray(["a", "b"]), field="source", train_indices=np.asarray([0, 1])
        )
    with pytest.raises(ValueError, match="canonical nonempty string IDs"):
        _single_metadata_features(
            np.asarray(["a", " "]), field="group", train_indices=np.asarray([0, 1])
        )
    with pytest.raises(ValueError, match="training indices are invalid"):
        _single_metadata_features(
            np.asarray(["a", "b"]), field="group", train_indices=np.asarray([7])
        )
    with pytest.raises(ValueError, match="max_feature_cells must be a positive integer"):
        _single_metadata_features(
            np.asarray(["a", "b"]),
            field="group",
            train_indices=np.asarray([0, 1]),
            max_feature_cells=0,
        )


def test_oracle_label_leak_requires_binary_integer_targets() -> None:
    features, _ = _oracle_label_leak_features(np.asarray([0, 1, 1, 0]))
    assert features.shape[0] == 4
    with pytest.raises(ValueError, match="nonempty binary integer target vector"):
        _oracle_label_leak_features(np.asarray([0.0, 1.0]))
    with pytest.raises(ValueError, match="nonempty binary integer target vector"):
        _oracle_label_leak_features(np.asarray([], dtype=np.int64))
