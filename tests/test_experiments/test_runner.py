from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from datasets.synthetic import debug_multipath
from experiments.runner import (
    _label_semantics_by_source,
    _resolved_label_semantics,
    _validation_threshold_selection_record,
    run_experiment,
    verify_model_input_artifact_contract,
    verify_validation_threshold_selection_record,
)


def test_validation_threshold_record_binds_only_selection_partition_data() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    scores = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
    groups = np.asarray(["a", "a", "b", "b"])
    indices = np.asarray([2, 3, 6, 7], dtype=np.int64)

    record = _validation_threshold_selection_record(labels, scores, groups, indices)
    repeated = _validation_threshold_selection_record(labels, scores, groups, indices)
    changed = _validation_threshold_selection_record(
        labels, scores + np.asarray([0.0, 0.0, 0.0, 0.01]), groups, indices
    )

    assert record["selection_partition"] == "validation"
    assert record["test_labels_used_for_selection"] is False
    assert record["validation_row_count"] == 4
    assert record["validation_group_count"] == 2
    assert record["validation_evidence_sha256"] == repeated["validation_evidence_sha256"]
    assert record["validation_evidence_sha256"] != changed["validation_evidence_sha256"]
    assert (
        verify_validation_threshold_selection_record(record, labels, scores, groups, indices)
        == record
    )

    tampered = dict(record, test_labels_used_for_selection=0)
    with pytest.raises(ValueError, match="validation-only recomputation"):
        verify_validation_threshold_selection_record(tampered, labels, scores, groups, indices)


def test_run_experiment_uses_verified_signed_split_lock(tmp_path) -> None:
    from datasets.channel import ChannelDataset
    from datasets.split_policy import (
        SplitParentType,
        SplitPolicy,
        SplitSubject,
        freeze_split_policy_registry,
        make_split_policy_registry,
    )

    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        import pytest

        pytest.skip("OpenSSH is unavailable")
    signer = "split-lock-runner-test"
    namespace = "umcd-split-policy-v2@github.com"
    private_key = tmp_path / "key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-C", signer, "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    keyring = tmp_path / "allowed_signers"
    keyring.write_text(f'{signer} namespaces="{namespace}" {public_key}\n', encoding="utf-8")
    subjects = [
        SplitSubject(f"parent-{index:03d}", SplitParentType.NETWORK_EPISODE, "test")
        for index in range(80)
    ]
    registry = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed="runner-signed-lock-test")
    )
    lock = tmp_path / "split-registry.json"
    freeze_split_policy_registry(lock, registry)
    subprocess.run(
        [ssh_keygen, "-Y", "sign", "-n", namespace, "-f", str(private_key), str(lock)],
        check=True,
        capture_output=True,
        text=True,
    )
    y = np.random.default_rng(54).integers(0, 2, size=len(subjects))
    h = np.ones((len(subjects), 1, 16), dtype=np.complex64)
    h[:, 0, :8] *= 1 + y[:, None]
    data = ChannelDataset(
        h=h,
        y=y,
        group=np.asarray([subject.subject_id for subject in subjects]),
        domain=np.full(len(subjects), "scene-a"),
        source=np.full(len(subjects), "sionna_rt"),
        metadata=tuple(
            {
                "geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE",
                "split_subject_id": subject.subject_id,
            }
            for subject in subjects
        ),
    )
    source = tmp_path / "locked.npz"
    data.save(source)
    output = tmp_path / "runs" / "locked"

    run_experiment(
        [source],
        output,
        methods=("logreg",),
        seeds=(11,),
        bootstrap=100,
        split_lock_path=lock,
        split_signature_path=Path(f"{lock}.sig"),
        trusted_split_signers_path=keyring,
        split_signer_id=signer,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))[0]
    assert manifest["split_policy"]["mode"] == "signed_locked"
    assert manifest["split_policy"]["signer_id"] == signer
    assert result["split"]["registry_sha256"] == registry.to_dict()["registry_sha256"]
    assert result["split"]["mode"] == "signed_locked"


def test_locked_split_indices_resolves_child_rows_through_parent_bindings() -> None:
    from datasets.channel import ChannelDataset
    from datasets.split_policy import (
        SplitChildBinding,
        SplitParentType,
        SplitPolicy,
        SplitSubject,
        bind_split_children,
        make_split_policy_registry,
    )
    from experiments.runner import _locked_split_indices

    parents = [
        SplitSubject(f"parent-{index:03d}", SplitParentType.NETWORK_EPISODE, "test")
        for index in range(80)
    ]
    bindings = [
        SplitChildBinding(f"episode-view-{index:03d}", parent.subject_id, "receiver_view")
        for index, parent in enumerate(parents)
    ]
    registry = bind_split_children(
        make_split_policy_registry(
            parents, policy=SplitPolicy.release_v2(seed="runner-child-binding-test")
        ),
        bindings,
    )

    subjects = [identity for parent in parents for identity in (parent.subject_id,)] + [
        binding.child_id for binding in bindings
    ]
    parent_by_row = [parent.subject_id for parent in parents] + [
        binding.parent_subject_id for binding in bindings
    ]
    labels = np.random.default_rng(73).integers(0, 2, size=len(subjects))
    channels = np.ones((len(subjects), 1, 16), dtype=np.complex64)
    channels[:, 0, :8] *= 1 + labels[:, None]
    data = ChannelDataset(
        h=channels,
        y=labels,
        group=np.asarray(parent_by_row),
        domain=np.full(len(subjects), "scene-a"),
        source=np.full(len(subjects), "sionna_rt"),
        metadata=tuple({"split_subject_id": identity} for identity in subjects),
    )

    train, validation, test, report = _locked_split_indices(data, registry)

    resolved = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    assert set(report["parent_counts"]) >= {"train", "validation", "test"}
    expected_indices = [
        index
        for index, parent in enumerate(parent_by_row)
        if registry.split_for_parent(parent) != "calibration"
    ]
    assert sorted(np.concatenate(tuple(resolved.values())).tolist()) == expected_indices
    assert report["calibration_rows_reserved"] == len(subjects) - len(expected_indices)
    for partition, indices in resolved.items():
        assert all(
            registry.split_for_parent(parent_by_row[index]) == partition for index in indices
        )

    padded_metadata = list(data.metadata)
    padded_metadata[0] = {"split_subject_id": f" {subjects[0]}"}
    padded_data = ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=tuple(padded_metadata),
    )
    with pytest.raises(ValueError, match="canonical split_subject_id"):
        _locked_split_indices(padded_data, registry)


def test_run_experiment_writes_reproducible_complete_artifacts(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=7).save(source)
    output = tmp_path / "runs" / "smoke"

    summary = run_experiment(
        [source],
        output,
        methods=("logreg",),
        seeds=(11,),
        bootstrap=100,
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert manifest["label_semantics"] == "synthetic_debug_direct_component"
    assert manifest["label_semantics_by_source"] == "SYNTHETIC_DEBUG_CONSTRUCTED"
    assert manifest["inputs"][0]["sha256"]
    assert (output / "COMPLETE").is_file()
    assert (output / "config" / "resolved.json").is_file()
    assert (output / "models" / "logreg_seed11.joblib").is_file()
    model_info = json.loads(
        (output / "models" / "logreg_seed11_modelinfo.json").read_text(encoding="utf-8")
    )
    model_contract = model_info["model_input_artifact_contract"]
    assert model_contract["schema"] == "loslab.model_input_artifact_contract.v1"
    assert model_contract["model_sha256"] == hashlib.sha256(
        (output / "models" / "logreg_seed11.joblib").read_bytes()
    ).hexdigest()
    assert model_contract["feature_contract"]["allowed_source_fields"] == ["observed_cfr"]
    assert "data.y" in model_contract["feature_contract"]["excluded_truth_and_metadata_fields"]
    assert model_contract["feature_contract"]["feature_transform"] == "logmag_pdp_moments_v1"
    assert verify_model_input_artifact_contract(
        output / "models" / "logreg_seed11_modelinfo.json",
        output / "models" / "logreg_seed11.joblib",
    )["status"] == "PASS_MODEL_INPUT_ARTIFACT_BINDING"
    model_info_path = output / "models" / "logreg_seed11_modelinfo.json"
    model_path = output / "models" / "logreg_seed11.joblib"
    symlink_sidecar = tmp_path / "modelinfo-symlink.json"
    symlink_sidecar.symlink_to(model_info_path)
    with pytest.raises(ValueError, match="single-link regular file"):
        verify_model_input_artifact_contract(symlink_sidecar, model_path)
    hardlink_sidecar = tmp_path / "modelinfo-hardlink.json"
    hardlink_sidecar.hardlink_to(model_info_path)
    with pytest.raises(ValueError, match="single-link regular file"):
        verify_model_input_artifact_contract(hardlink_sidecar, model_path)
    hardlink_sidecar.unlink()
    model_symlink = tmp_path / "model-symlink.joblib"
    model_symlink.symlink_to(model_path)
    with pytest.raises(ValueError, match="single-link regular file"):
        verify_model_input_artifact_contract(model_info_path, model_symlink)
    model_hardlink = tmp_path / "model-hardlink.joblib"
    model_hardlink.hardlink_to(model_path)
    with pytest.raises(ValueError, match="single-link regular file"):
        verify_model_input_artifact_contract(model_info_path, model_hardlink)
    model_hardlink.unlink()
    original_sidecar = model_info_path.read_bytes()
    inconsistent_sidecar = json.loads(original_sidecar)
    inconsistent_sidecar["method"] = "cnn"
    model_info_path.write_text(json.dumps(inconsistent_sidecar), encoding="utf-8")
    with pytest.raises(ValueError, match="wrapper disagrees"):
        verify_model_input_artifact_contract(model_info_path, model_path)
    model_info_path.write_bytes(original_sidecar)
    assert (output / "evaluation" / "summary.json").is_file()

    prediction = np.load(output / "predictions" / "logreg_seed11.npz")
    train = set(prediction["train_idx"].tolist())
    validation = set(prediction["val_idx"].tolist())
    test = set(prediction["test_idx"].tolist())
    assert not train & validation
    assert not train & test
    assert not validation & test
    assert summary["aggregate_across_fixed_splits"]["logreg"]["seeds"] == 1
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="model bytes do not match"):
        verify_model_input_artifact_contract(
            model_info_path, model_path
        )


def test_run_experiment_refuses_to_overwrite_completed_run(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=7).save(source)
    output = tmp_path / "runs" / "smoke"
    run_experiment([source], output, methods=("logreg",), seeds=(11,), bootstrap=10)

    import pytest

    with pytest.raises(FileExistsError, match="immutable"):
        run_experiment([source], output, methods=("logreg",), seeds=(11,), bootstrap=10)


def test_run_experiment_reports_paired_classical_vs_mlp_deltas(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=12, links_per_scene=24, seed=19).save(source)

    summary = run_experiment(
        [source],
        tmp_path / "runs" / "paired",
        methods=("rf", "mlp"),
        seeds=(11, 23, 47),
        bootstrap=100,
    )

    comparison = summary["paired_method_comparisons"]["mlp_versus_rf"]
    assert comparison["paired_seeds"] == [11, 23, 47]
    assert len(comparison["treatment_minus_control"]) == 3
    assert comparison["test"] == "exact paired sign-flip"
    assert comparison["warning"].startswith("Fewer than six seeds")

    results = json.loads((tmp_path / "runs" / "paired" / "results.json").read_text())
    by_seed_and_method = {(row["seed"], row["method"]): row for row in results}
    for seed in (11, 23, 47):
        assert (
            by_seed_and_method[(seed, "rf")]["split"]["indices_sha256"]
            == by_seed_and_method[(seed, "mlp")]["split"]["indices_sha256"]
        )
    mlp_compute = by_seed_and_method[(11, "mlp")]["compute"]
    assert mlp_compute["training_seconds"] >= 0.0
    assert mlp_compute["validation_prediction_seconds"] >= 0.0
    assert mlp_compute["test_prediction_seconds"] >= 0.0
    assert mlp_compute["learned_parameter_count"] > 0
    assert mlp_compute["peak_cuda_memory_bytes"] is None


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


def test_run_experiment_supports_histogram_gradient_boosting(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=8, links_per_scene=16, seed=43).save(source)

    summary = run_experiment(
        [source],
        tmp_path / "runs" / "histgb",
        methods=("histgb",),
        seeds=(13,),
        bootstrap=100,
    )

    assert summary["aggregate_across_fixed_splits"]["histgb"]["seeds"] == 1
    assert (tmp_path / "runs" / "histgb" / "models" / "histgb_seed13.joblib").is_file()


def test_run_experiment_routes_raw_observations_to_real_cnn(tmp_path, monkeypatch) -> None:
    import experiments.runner as runner

    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=8, links_per_scene=16, seed=53).save(source)
    observed_inputs = []

    class Predictor:
        def predict_proba(self, values):
            assert np.iscomplexobj(values)
            observed_inputs.append(values.shape)
            score = np.linspace(0.1, 0.9, len(values))
            return np.column_stack((1.0 - score, score))

        def save(self, path, *, threshold, include_phase):
            assert include_phase is True
            assert 0.0 <= threshold <= 1.0
            path.write_bytes(b"test checkpoint")

    def fake_fit(train, labels, validation, validation_labels, **kwargs):
        assert np.iscomplexobj(train)
        assert np.iscomplexobj(validation)
        assert labels.ndim == validation_labels.ndim == 1
        return Predictor(), {"epoch_count": 1, "trainable_parameters": 1}

    monkeypatch.setattr(runner, "fit_real_cnn", fake_fit)
    output = tmp_path / "runs" / "cnn"
    result = run_experiment([source], output, methods=("cnn",), seeds=(13,), bootstrap=10)

    assert result["aggregate_across_fixed_splits"]["cnn"]["seeds"] == 1
    assert (output / "models" / "cnn_seed13.pt").read_bytes() == b"test checkpoint"
    model_info = json.loads(
        (output / "models" / "cnn_seed13_modelinfo.json").read_text(encoding="utf-8")
    )
    assert model_info["model_input_artifact_contract"]["feature_contract"][
        "feature_transform"
    ] == "complex_cfr_grid_v1"
    assert observed_inputs
    config = json.loads((output / "config" / "resolved.json").read_text(encoding="utf-8"))
    assert "real/imaginary" in config["values"]["cnn_input_representation"]


def test_run_experiment_routes_raw_observations_to_complex_cnn(tmp_path, monkeypatch) -> None:
    import experiments.runner as runner

    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=8, links_per_scene=16, seed=59).save(source)
    captured = []

    class Predictor:
        def predict_proba(self, values):
            assert np.iscomplexobj(values)
            captured.append(values.shape)
            score = np.linspace(0.1, 0.9, len(values))
            return np.column_stack((1.0 - score, score))

        def save(self, path, *, threshold):
            assert 0.0 <= threshold <= 1.0
            path.write_bytes(b"complex checkpoint")

    def fake_fit(train, labels, validation, validation_labels, **kwargs):
        assert np.iscomplexobj(train) and np.iscomplexobj(validation)
        assert labels.ndim == validation_labels.ndim == 1
        return Predictor(), {"epoch_count": 1, "trainable_parameters": 1}

    monkeypatch.setattr(runner, "fit_complex_cnn", fake_fit)
    output = tmp_path / "runs" / "cvnn"
    result = run_experiment([source], output, methods=("cvnn",), seeds=(17,), bootstrap=10)

    assert result["aggregate_across_fixed_splits"]["cvnn"]["seeds"] == 1
    assert (output / "models" / "cvnn_seed17.pt").read_bytes() == b"complex checkpoint"
    assert captured


def test_run_experiment_routes_flattened_real_imag_to_raw_mlp(tmp_path, monkeypatch) -> None:
    import experiments.runner as runner

    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=8, links_per_scene=16, seed=71).save(source)
    original_fit = runner.fit_sklearn
    observed_shapes = []

    def capture_fit(values, labels, *, seed, method, capacity):
        observed_shapes.append(values.shape[1])
        assert not np.iscomplexobj(values)
        assert method == "mlp"
        return original_fit(values, labels, seed=seed, method="logreg", capacity=capacity)

    monkeypatch.setattr(runner, "fit_sklearn", capture_fit)
    output = tmp_path / "runs" / "raw-mlp"
    run_experiment([source], output, methods=("raw_mlp",), seeds=(19,), bootstrap=10)
    data = np.load(source)
    expected_features = 2 * data["h"].shape[1] * data["h"].shape[2]

    assert observed_shapes == [expected_features]
    prediction = np.load(output / "predictions" / "raw_mlp_seed19.npz")
    assert prediction["p_test"].ndim == 1
