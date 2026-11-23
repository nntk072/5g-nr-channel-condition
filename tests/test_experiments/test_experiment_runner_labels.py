"""Experiment-runner label semantics and metadata vocabularies."""

from __future__ import annotations

import json

import numpy as np
import pytest
from datasets.channel import ChannelDataset
from datasets.synthetic import debug_multipath
from experiments.runner import (
    _label_semantics_by_source,
    _resolved_label_semantics,
    run_experiment,
    verify_model_input_artifact_contract,
)


def test_label_semantics_are_derived_from_each_dataset_source() -> None:
    from datasets.channel import ChannelDataset

    data = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["nlos", "los"]),
        domain=np.asarray(["scene-a", "profile-b"]),
        source=np.asarray(["sionna_rt", "tdl"]),
        metadata=(
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
            {"label_semantics": "PROFILE_PROXY"},
        ),
    )

    assert (
        _label_semantics_by_source(data)
        == "sionna_rt=GEOMETRIC_DIRECT_PATH_ORACLE; tdl=PROFILE_PROXY"
    )


def test_uniform_legacy_label_metadata_resolves_to_canonical_semantics() -> None:
    from datasets.channel import ChannelDataset

    legacy_rt = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["nlos", "los"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray(["sionna_rt", "sionna_rt"]),
        metadata=(
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
        ),
    )

    assert _resolved_label_semantics(legacy_rt) == "geometric_direct_path_oracle"


def test_unspecified_legacy_label_metadata_is_not_treated_as_a_task() -> None:
    import pytest
    from datasets.channel import ChannelDataset

    unknown = ChannelDataset(
        h=np.ones((2, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1]),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["scene", "scene"]),
        source=np.asarray(["old_source", "old_source"]),
        metadata=({"label_semantics": "UNSPECIFIED"}, {"label_semantics": "unknown"}),
    )

    with pytest.raises(ValueError, match="one explicit, consistent label meaning"):
        _resolved_label_semantics(unknown)


def test_experiment_rejects_mixed_legacy_label_semantics_before_training(tmp_path) -> None:
    import pytest
    from datasets.channel import ChannelDataset

    mixed = ChannelDataset(
        h=np.ones((4, 1, 8), dtype=np.complex64),
        y=np.asarray([0, 1, 0, 1]),
        group=np.asarray(["g0", "g1", "g2", "g3"]),
        domain=np.asarray(["d0", "d1", "d2", "d3"]),
        source=np.asarray(["rt", "rt", "tdl", "tdl"]),
        metadata=(
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
            {"geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE"},
            {"label_semantics": "PROFILE_PROXY"},
            {"label_semantics": "PROFILE_PROXY"},
        ),
    )
    source = tmp_path / "mixed-legacy.npz"
    mixed.save(source)

    with pytest.raises(ValueError, match="one explicit, consistent label meaning"):
        run_experiment([source], tmp_path / "runs" / "must-not-train")

    assert not (tmp_path / "runs" / "must-not-train").exists()


def test_run_experiment_preserves_precomputed_dmrs_observation_semantics(tmp_path) -> None:
    from datasets.channel import ChannelDataset

    generated = debug_multipath(n_scenes=4, links_per_scene=12, seed=29)
    source = tmp_path / "dmrs-observations.npz"
    observed = ChannelDataset(
        h=generated.h,
        y=generated.y,
        group=generated.group,
        domain=generated.domain,
        source=generated.source,
        metadata=tuple(
            {**metadata, "receiver_observation": "Sionna PUSCH DMRS LS channel estimate"}
            for metadata in generated.metadata
        ),
        generation_report=generated.generation_report,
    )
    observed.save(source)
    output = tmp_path / "runs" / "precomputed"

    run_experiment(
        [source],
        output,
        methods=("logreg",),
        seeds=(11,),
        bootstrap=100,
        observation="precomputed",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observation_semantics"] == "Sionna PUSCH DMRS LS channel estimate"
    config = json.loads((output / "config" / "resolved.json").read_text(encoding="utf-8"))
    assert config["values"]["observation"] == "precomputed"


def test_training_label_permutation_is_seeded_and_preserves_class_counts() -> None:
    from experiments.runner import _permute_training_labels

    labels = np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    original = labels.copy()
    first, contract = _permute_training_labels(labels, run_seed=31, base_seed=413)
    replay, replay_contract = _permute_training_labels(labels, run_seed=31, base_seed=413)
    other_seed, _ = _permute_training_labels(labels, run_seed=32, base_seed=413)

    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(first, replay)
    assert contract == replay_contract
    assert np.count_nonzero(first == 1) == np.count_nonzero(labels == 1)
    assert not np.array_equal(first, other_seed)
    assert contract["permuted_targets_sha256"]

    with pytest.raises(ValueError, match="binary integer training labels"):
        _permute_training_labels(np.asarray([0.0, 1.0]), run_seed=31, base_seed=413)


def test_run_experiment_routes_permuted_labels_on_shared_split(tmp_path, monkeypatch) -> None:
    import experiments.runner as runner

    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=8, links_per_scene=16, seed=83).save(source)
    original_fit = runner.experiment_runner.fit_sklearn
    captures = []

    def capture_fit(values, labels, *, seed, method, capacity):
        captures.append((values.copy(), labels.copy(), method))
        return original_fit(values, labels, seed=seed, method="logreg", capacity=capacity)

    monkeypatch.setattr(runner.experiment_runner, "fit_sklearn", capture_fit)
    output = tmp_path / "runs" / "label-permuted"
    run_experiment(
        [source],
        output,
        methods=("label_permuted", "raw_mlp"),
        seeds=(37,),
        label_permutation_seed=419,
        bootstrap=10,
    )

    results = json.loads((output / "results.json").read_text(encoding="utf-8"))
    control = next(row for row in results if row["method"] == "label_permuted")
    raw = next(row for row in results if row["method"] == "raw_mlp")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    sidecar_path = output / "models" / "label_permuted_seed37_modelinfo.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    contract = sidecar["model_input_artifact_contract"]
    feature_contract = contract["feature_contract"]
    prediction = np.load(output / "predictions" / "label_permuted_seed37.npz")
    dataset = ChannelDataset.load(source)
    train_indices = prediction["train_idx"]
    expected_labels, expected_transform = runner._permute_training_labels(
        dataset.y[train_indices], run_seed=37, base_seed=419
    )

    assert [method for _values, _labels, method in captures] == ["mlp", "mlp"]
    assert not np.iscomplexobj(captures[0][0])
    np.testing.assert_array_equal(captures[0][1], expected_labels)
    np.testing.assert_array_equal(captures[1][1], dataset.y[train_indices])
    assert control["split"]["indices_sha256"] == raw["split"]["indices_sha256"]
    assert "label_permuted_versus_raw_mlp" in summary["paired_method_comparisons"]
    assert contract["schema"] == "loslab.model_input_artifact_contract.v5"
    assert feature_contract["supervision_transform"] == expected_transform
    assert (
        verify_model_input_artifact_contract(
            sidecar_path,
            output / "models" / "label_permuted_seed37.joblib",
            method_input=runner._flatten_real_imag_cfr(dataset.h),
            partition_indices={
                "train": prediction["train_idx"],
                "validation": prediction["val_idx"],
                "test": prediction["test_idx"],
            },
            target_values=dataset.y,
        )["status"]
        == "PASS_MODEL_INPUT_ARTIFACT_BINDING"
    )
    assert control["input_representation"].startswith("raw flattened")


def test_source_only_vocabulary_fits_training_rows_and_zeros_unknowns() -> None:
    from experiments.runner import _source_only_features

    sources = np.asarray(["alpha", "beta", "future", "future"])
    features, transform = _source_only_features(
        sources, train_indices=np.asarray([0, 1], dtype=np.int64)
    )

    assert transform["vocabulary"] == ["alpha", "beta"]
    np.testing.assert_array_equal(features[:2], np.eye(2, dtype=np.float32))
    np.testing.assert_array_equal(features[2:], np.zeros((2, 2), dtype=np.float32))
    assert transform["unknown_category_behavior"] == "all_zero_vector"


def test_metadata_only_vocabulary_is_train_fitted_bounded_and_namespaced() -> None:
    from experiments.runner import _metadata_only_features

    groups = np.asarray(["g1", "g2", "future", "future"])
    domains = np.asarray(["d1", "d1", "d2", "d2"])
    sources = np.asarray(["s1", "s2", "s3", "s3"])
    features, transform = _metadata_only_features(
        group_values=groups,
        domain_values=domains,
        source_values=sources,
        train_indices=np.asarray([0, 1], dtype=np.int64),
    )

    assert features.shape == (4, 5)
    assert transform["fields"] == ["domain", "group", "source"]
    assert transform["unknown_category_behavior"] == "all_zero_vector_for_unknown_field_value"
    assert np.all(features[2:, 1:] == 0.0)
    with pytest.raises(ValueError, match="max_feature_cells"):
        _metadata_only_features(
            group_values=groups,
            domain_values=domains,
            source_values=sources,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            max_feature_cells=1,
        )


@pytest.mark.parametrize(("field", "method"), [("group", "group_only"), ("domain", "domain_only")])
def test_single_metadata_vocabulary_isolated_train_fitted_and_bounded(field, method) -> None:
    from experiments.runner import _single_metadata_features

    values = np.asarray(["seen-a", "seen-b", "held-out", "held-out"])
    features, transform = _single_metadata_features(
        values, field=field, train_indices=np.asarray([0, 1], dtype=np.int64)
    )
    assert features.shape == (4, 2)
    assert transform["field"] == field
    assert transform["vocabulary"] == ["seen-a", "seen-b"]
    assert transform["unknown_category_behavior"] == "all_zero_vector"
    np.testing.assert_array_equal(features[2:], np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="max_feature_cells"):
        _single_metadata_features(
            values,
            field=field,
            train_indices=np.asarray([0, 1], dtype=np.int64),
            max_feature_cells=1,
        )


def test_oracle_label_leak_control_is_explicit_and_replayable(tmp_path):
    source = tmp_path / "debug-oracle.npz"
    output = tmp_path / "runs" / "oracle-label-leak"
    debug_multipath(n_scenes=8, links_per_scene=12, n_subcarriers=16).save(source)

    run_experiment([source], output, methods=("oracle_label_leak",), seeds=(43,), bootstrap=20)

    dataset = ChannelDataset.load(source)
    model_path = output / "models" / "oracle_label_leak_seed43.joblib"
    sidecar_path = output / "models" / "oracle_label_leak_seed43_modelinfo.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    contract = sidecar["model_input_artifact_contract"]
    feature_contract = contract["feature_contract"]
    prediction = np.load(output / "predictions" / "oracle_label_leak_seed43.npz")
    partitions = {
        "train": prediction["train_idx"],
        "validation": prediction["val_idx"],
        "test": prediction["test_idx"],
    }

    assert contract["schema"] == "loslab.model_input_artifact_contract.v9"
    assert feature_contract["allowed_source_fields"] == ["data.y"]
    assert feature_contract["intentional_label_leakage"]["status"] == "INTENTIONAL"
    assert feature_contract["intentional_label_leakage"]["deployment_eligible"] is False
    assert (
        verify_model_input_artifact_contract(
            sidecar_path,
            model_path,
            method_input=np.eye(2, dtype=np.float32)[dataset.y],
            partition_indices=partitions,
            target_values=dataset.y,
        )["feature_matrix_verified"]
        is True
    )

    changed_targets = dataset.y.copy()
    changed_targets[partitions["test"][0]] = 1 - changed_targets[partitions["test"][0]]
    with pytest.raises(ValueError, match="does not replay from source targets"):
        verify_model_input_artifact_contract(
            sidecar_path,
            model_path,
            method_input=np.eye(2, dtype=np.float32)[dataset.y],
            partition_indices=partitions,
            target_values=changed_targets,
        )
