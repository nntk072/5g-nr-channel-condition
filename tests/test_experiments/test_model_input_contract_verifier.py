"""Model-input artifact contract verifier: sidecar bounds, schema and replayed inputs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from artifacts.provenance import canonical_hash
from experiments.runner.contracts import (
    build_model_input_artifact_contract,
    verify_model_input_artifact_contract,
)
from experiments.runner.validation import (
    _metadata_only_features,
    _oracle_label_leak_features,
    _permute_training_feature_rows,
    _single_metadata_features,
    _source_only_features,
)
from tests.test_experiments.test_model_input_contract_builder import (
    _ROWS,
    _partitions,
    _split,
)

MODEL_BYTES = b"synthetic model bytes for contract verification"


def _matrix() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(_ROWS, 6))


def _route_features(method: str, extra: dict[str, Any], partitions) -> np.ndarray | None:
    """The replayed feature matrix that a route's declared inputs must produce."""
    if method == "oracle_label_leak":
        features, _ = _oracle_label_leak_features(extra["target_values"])
        return features
    if method == "source_only":
        features, _ = _source_only_features(
            extra["source_values"], train_indices=partitions["train"]
        )
        return features
    if method == "row_shuffled":
        source = extra["source_method_input"]
        shuffled, _ = _permute_training_feature_rows(
            source[partitions["train"]],
            run_seed=extra["feature_permutation_run_seed"],
            base_seed=extra["feature_permutation_seed"],
        )
        features = np.array(source, copy=True)
        features[partitions["train"]] = shuffled
        return features
    if method == "metadata_only":
        features, _ = _metadata_only_features(
            group_values=extra["metadata_values"]["group"],
            domain_values=extra["metadata_values"]["domain"],
            source_values=extra["metadata_values"]["source"],
            train_indices=partitions["train"],
        )
        return features
    if method in {"group_only", "domain_only"}:
        field = "group" if method == "group_only" else "domain"
        features, _ = _single_metadata_features(
            extra["metadata_values"][field], field=field, train_indices=partitions["train"]
        )
        return features
    return None


def _bind(tmp_path: Path, method: str = "cnn", **overrides: Any):
    """Write a model and a sidecar whose contract binds that model's exact bytes."""
    model = tmp_path / "model.bin"
    model.write_bytes(MODEL_BYTES)
    route_input = _route_features(method, overrides, _partitions())
    kwargs: dict[str, Any] = {
        "method": method,
        "method_input": _matrix() if route_input is None else route_input,
        "include_phase": False,
        "config_sha256": "a" * 64,
        "source_revision": "rev",
        "input_file_digests": [{"path": "a.npz", "sha256": "b" * 64}],
        "split": _split(),
        "model_file_sha256": hashlib.sha256(MODEL_BYTES).hexdigest(),
        "partition_indices": _partitions(),
    }
    kwargs.update(overrides)
    contract = build_model_input_artifact_contract(**kwargs)
    sidecar = tmp_path / "model.info.json"
    sidecar.write_text(json.dumps(_wrapper(method, contract, kwargs)), encoding="utf-8")
    return sidecar, model, contract, kwargs


def _wrapper(method: str, contract: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """The sidecar wrapper fields the verifier cross-checks against the feature contract."""
    sklearn_methods = {
        "rf",
        "histgb",
        "xgb",
        "mlp",
        "raw_mlp",
        "power_only",
        "phase_randomized",
        "label_permuted",
        "row_shuffled",
        "source_only",
        "metadata_only",
        "group_only",
        "domain_only",
        "oracle_label_leak",
        "logreg",
        "delay_k_proxy",
        "sparse_multipath",
    }
    return {
        "format": "loslab.sklearn.v1"
        if method in sklearn_methods
        else "loslab.model_input_artifact.v1",
        "method": method,
        "include_phase": kwargs["include_phase"],
        "threshold": 0.5,
        "model_input_artifact_contract": contract,
    }


def _verify(sidecar: Path, model: Path, kwargs: dict[str, Any], **extra: Any) -> dict[str, Any]:
    arguments = {
        "method_input": kwargs["method_input"],
        "partition_indices": kwargs["partition_indices"],
    }
    arguments.update(extra)
    return verify_model_input_artifact_contract(sidecar, model, **arguments)


def test_plain_contract_verifies_against_its_exact_inputs(tmp_path: Path) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)

    report = _verify(sidecar, model, kwargs)

    assert report is not None


@pytest.mark.parametrize(
    ("partitions", "message"),
    [
        (None, "requires all split indices"),
        (
            {"train": np.zeros((1, 1)), "validation": np.array([6]), "test": np.array([9])},
            "indices are malformed",
        ),
        (
            {"train": np.array([0.0]), "validation": np.array([6]), "test": np.array([9])},
            "indices are invalid",
        ),
        (
            {
                "train": np.array([0, 1, 2, 3, 4, 5]),
                "validation": np.array([6]),
                "test": np.array([9]),
            },
            "differs from its frozen contract",
        ),
    ],
)
def test_feature_matrix_replay_requires_exact_well_formed_partition_indices(
    tmp_path: Path, partitions, message: str
) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)

    with pytest.raises(ValueError, match=message):
        verify_model_input_artifact_contract(
            sidecar,
            model,
            method_input=kwargs["method_input"],
            partition_indices=partitions,
        )


def test_contract_feature_digest_must_match_sidecar_contents(tmp_path: Path) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["model_input_artifact_contract"]["feature_contract_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match its contents"):
        _verify(sidecar, model, kwargs)


def test_unexpected_benchmark_cohort_is_rejected(tmp_path: Path) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["benchmark_cohort"] = {}
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected benchmark cohort provenance"):
        _verify(sidecar, model, kwargs)


@pytest.mark.parametrize(
    ("cohort", "partitions", "message"),
    [
        ({}, {"test": np.array([9.0])}, "one-dimensional integer array"),
        (
            {"selected_test_indices_sha256": "0" * 64},
            {"test": np.array([9])},
            "does not bind the model test indices",
        ),
        (
            {
                "selected_test_indices_sha256": canonical_hash([9]),
                "registry_sha256": "different",
            },
            {"test": np.array([9])},
            "registry does not match",
        ),
    ],
)
def test_benchmark_cohort_replay_rejects_malformed_test_roster(
    tmp_path: Path, cohort: dict[str, Any], partitions: dict[str, np.ndarray], message: str
) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["benchmark_cohort"] = cohort
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    indices = {**kwargs["partition_indices"], **partitions}

    with pytest.raises(ValueError, match=message):
        verify_model_input_artifact_contract(
            sidecar, model, partition_indices=indices, benchmark_cohort=cohort
        )


def test_sidecar_must_be_a_single_link_regular_file(tmp_path: Path) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)
    link = tmp_path / "linked.info.json"
    link.symlink_to(sidecar)
    with pytest.raises(ValueError, match="single-link regular file"):
        _verify(link, model, kwargs)


def test_sidecar_size_is_bounded(tmp_path: Path) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)
    sidecar.write_text(" " * 1_048_577, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds its safe file contract"):
        _verify(sidecar, model, kwargs)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        pytest.param(b"\xff\xfe", "readable UTF-8 JSON", id="not-utf8"),
        pytest.param(b"[]", "lacks its contract", id="root-array"),
        pytest.param(b'{"other": {}}', "lacks its contract", id="missing-contract"),
        pytest.param(
            b'{"model_input_artifact_contract": {"schema": "x"}}',
            "contract schema is unsupported",
            id="unknown-schema",
        ),
    ],
)
def test_sidecar_shape_is_checked(tmp_path: Path, payload: bytes, message: str) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)
    sidecar.write_bytes(payload)
    with pytest.raises(ValueError, match=message):
        _verify(sidecar, model, kwargs)


def test_contract_must_match_its_schema(tmp_path: Path) -> None:
    sidecar, model, contract, kwargs = _bind(tmp_path)
    broken = dict(contract)
    broken.pop("split")
    sidecar.write_text(json.dumps(_wrapper("cnn", broken, kwargs)), encoding="utf-8")
    with pytest.raises(ValueError, match="violates schema|fields or schema are invalid"):
        _verify(sidecar, model, kwargs)


def test_model_bytes_must_match_the_contract(tmp_path: Path) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)
    model.write_bytes(MODEL_BYTES + b"!")
    with pytest.raises(ValueError, match="model"):
        _verify(sidecar, model, kwargs)


def test_sidecar_refuses_platform_without_no_follow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar, model, _, kwargs = _bind(tmp_path)
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with pytest.raises(
        ValueError, match="platform lacks safe model input artifact sidecar opening"
    ):
        _verify(sidecar, model, kwargs)


def _rows_vector(seed: int) -> np.ndarray:
    """Binary integer labels with both classes present in every partition."""
    labels = np.array([0, 1] * (_ROWS // 2), dtype=np.int64)
    return np.roll(labels, seed)


def _ids(prefix: str, count: int) -> np.ndarray:
    return np.array([f"{prefix}{index % count}" for index in range(_ROWS)])


ROUTE_CASES = [
    pytest.param(
        {
            "method": "label_permuted",
            "target_values": _rows_vector(1),
            "label_permutation_seed": 3,
            "label_permutation_run_seed": 4,
        },
        {"target_values": _rows_vector(1)},
        id="label-permuted",
    ),
    pytest.param(
        {
            "method": "oracle_label_leak",
            "target_values": _rows_vector(2),
        },
        {"target_values": _rows_vector(2)},
        id="oracle-label-leak",
    ),
    pytest.param(
        {
            "method": "metadata_only",
            "metadata_values": {
                "group": _ids("g", 3),
                "domain": _ids("d", 2),
                "source": _ids("s", 4),
            },
        },
        {
            "metadata_values": {
                "group": _ids("g", 3),
                "domain": _ids("d", 2),
                "source": _ids("s", 4),
            }
        },
        id="metadata-only",
    ),
    pytest.param(
        {"method": "group_only", "metadata_values": {"group": _ids("g", 3)}},
        {"metadata_values": {"group": _ids("g", 3)}},
        id="group-only",
    ),
    pytest.param(
        {"method": "domain_only", "metadata_values": {"domain": _ids("d", 2)}},
        {"metadata_values": {"domain": _ids("d", 2)}},
        id="domain-only",
    ),
]


@pytest.mark.parametrize(("build_extra", "verify_extra"), ROUTE_CASES)
def test_replayed_route_verifies_its_declared_inputs(
    tmp_path: Path, build_extra: dict[str, Any], verify_extra: dict[str, Any]
) -> None:
    sidecar, model, _, kwargs = _bind(
        tmp_path,
        method=build_extra["method"],
        **{key: value for key, value in build_extra.items() if key != "method"},
    )

    report = _verify(sidecar, model, kwargs, **verify_extra)

    assert report is not None


@pytest.mark.parametrize(
    ("build_extra", "message"),
    [
        (
            {
                "method": "label_permuted",
                "target_values": _rows_vector(1),
                "label_permutation_seed": 3,
                "label_permutation_run_seed": 4,
            },
            "requires its target values",
        ),
        (
            {
                "method": "row_shuffled",
                "source_method_input": _matrix(),
                "feature_permutation_seed": 3,
                "feature_permutation_run_seed": 4,
            },
            "requires source features",
        ),
        (
            {"method": "source_only", "source_values": _ids("s", 3)},
            "requires source identifiers",
        ),
        (
            {
                "method": "metadata_only",
                "metadata_values": {
                    "group": _ids("g", 3),
                    "domain": _ids("d", 2),
                    "source": _ids("s", 4),
                },
            },
            "requires metadata ID rosters",
        ),
        (
            {"method": "group_only", "metadata_values": {"group": _ids("g", 3)}},
            "requires only its ID roster",
        ),
        (
            {"method": "domain_only", "metadata_values": {"domain": _ids("d", 2)}},
            "requires only its ID roster",
        ),
        (
            {"method": "oracle_label_leak", "target_values": _rows_vector(2)},
            "requires the exact target roster",
        ),
    ],
)
def test_replayed_route_requires_its_verification_inputs(
    tmp_path: Path, build_extra: dict[str, Any], message: str
) -> None:
    method = build_extra["method"]
    sidecar, model, _, kwargs = _bind(
        tmp_path,
        method=method,
        **{key: value for key, value in build_extra.items() if key != "method"},
    )

    with pytest.raises(ValueError, match=message):
        _verify(sidecar, model, kwargs)


def test_label_permutation_replay_rejects_other_targets(tmp_path: Path) -> None:
    sidecar, model, _, kwargs = _bind(
        tmp_path,
        method="label_permuted",
        target_values=_rows_vector(1),
        label_permutation_seed=3,
        label_permutation_run_seed=4,
    )
    with pytest.raises(ValueError):
        _verify(sidecar, model, kwargs, target_values=1 - _rows_vector(1))
