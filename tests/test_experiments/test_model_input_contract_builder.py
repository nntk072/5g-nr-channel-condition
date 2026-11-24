"""Model-input artifact contract: the builder rejects every unsupported input combination."""

from __future__ import annotations

import numpy as np
import pytest
from experiments.runner.contracts import build_model_input_artifact_contract

_ROWS = 12


def _partitions():
    return {
        "train": np.arange(0, 6),
        "validation": np.arange(6, 9),
        "test": np.arange(9, 12),
    }


def _split():
    return {
        "mode": "fixture",
        "registry_sha256": None,
        "indices_sha256": "c" * 64,
        "train_rows": 6,
        "validation_rows": 3,
        "test_rows": 3,
    }


def _build(**overrides):
    kwargs = {
        "method": "cnn",
        "method_input": np.random.default_rng(0).normal(size=(_ROWS, 6)),
        "include_phase": False,
        "config_sha256": "a" * 64,
        "source_revision": "rev",
        "input_file_digests": [{"path": "a.npz", "sha256": "b" * 64}],
        "split": _split(),
        "model_file_sha256": "d" * 64,
        "partition_indices": _partitions(),
    }
    kwargs.update(overrides)
    return build_model_input_artifact_contract(**kwargs)


def test_a_plain_method_yields_a_schema_bound_contract() -> None:
    contract = _build()
    assert contract["split"]["train_rows"] == 6
    assert set(contract) >= {"schema", "model_sha256", "feature_contract", "training_input_files"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "not-a-method"}, "unsupported model-input artifact method"),
        ({"method_input": np.ones(_ROWS)}, "batched feature array"),
        ({"method_input": np.full((_ROWS, 6), np.nan)}, "finite numeric values"),
        ({"method_input": np.array([["a"] * 6] * _ROWS)}, "finite numeric values"),
        (
            {"partition_indices": {"train": np.arange(6), "test": np.arange(9, 12)}},
            "must include train, validation, and test",
        ),
        (
            {"partition_indices": {**_partitions(), "bogus": np.arange(1)}},
            "must include train, validation, and test",
        ),
        (
            {"partition_indices": {**_partitions(), "train": np.arange(6, dtype=float)}},
            "partition indices are invalid",
        ),
        (
            {"partition_indices": {**_partitions(), "train": np.array([True] * 6)}},
            "partition indices are invalid",
        ),
        (
            {"partition_indices": {**_partitions(), "train": np.array([-1, 0])}},
            "partition indices are invalid",
        ),
        (
            {"partition_indices": {**_partitions(), "train": np.array([_ROWS])}},
            "partition indices are invalid",
        ),
        (
            {"partition_indices": {**_partitions(), "train": np.zeros((2, 2), dtype=int)}},
            "partition indices are invalid",
        ),
        (
            {"partition_indices": {**_partitions(), "validation": np.array([0, 1, 2])}},
            "mutually disjoint",
        ),
    ],
)
def test_input_shape_and_partition_rules(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _build(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "label_permuted"}, "label-permuted"),
        (
            {
                "method": "label_permuted",
                "target_values": np.zeros(_ROWS),
                "label_permutation_seed": 1,
            },
            "label-permuted",
        ),
        ({"method": "oracle_label_leak"}, "oracle-label"),
        ({"method": "row_shuffled"}, "row-shuffled"),
        (
            {
                "method": "row_shuffled",
                "source_method_input": np.zeros((_ROWS, 6)),
                "feature_permutation_seed": 1,
            },
            "row-shuffled",
        ),
        (
            {"feature_permutation_seed": 1, "feature_permutation_run_seed": 2},
            "only valid for row_shuffled",
        ),
        ({"method": "source_only"}, "source-only"),
        ({"method": "source_only", "source_values": np.arange(_ROWS)}, "source-only"),
        ({"source_values": np.arange(_ROWS)}, "only valid for the source_only method"),
        ({"method": "metadata_only"}, "metadata-only"),
        ({"metadata_values": {"group": np.arange(_ROWS)}}, "metadata identifiers are only valid"),
        (
            {
                "method": "group_only",
                "metadata_values": {"group": np.arange(_ROWS), "domain": np.arange(_ROWS)},
            },
            "group_only contracts require only",
        ),
        (
            {"method": "domain_only", "metadata_values": {"group": np.arange(_ROWS)}},
            "domain_only contracts require only",
        ),
    ],
)
def test_method_specific_inputs_are_required_and_exclusive(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _build(**overrides)
