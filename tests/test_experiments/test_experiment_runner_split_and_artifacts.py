"""Experiment-runner split locks, benchmark cohorts, failure receipts and artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import experiments.runner as runner_module
import experiments.runner.experiment_runner as experiment_runner_impl
import numpy as np
import pytest
from artifacts.provenance import canonical_hash
from datasets.channel import ChannelDataset
from datasets.synthetic import debug_multipath
from experiments.runner import (
    _validation_threshold_selection_record,
    run_experiment,
    verify_declared_partition_support,
    verify_model_input_artifact_contract,
    verify_validation_threshold_selection_record,
)
from propagation.physics import observable_features


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"phase_randomization_seed": True}, "phase_randomization_seed"),
        ({"label_permutation_seed": -1}, "label_permutation_seed"),
        ({"row_shuffle_seed": 2**63}, "row_shuffle_seed"),
        ({"low_confidence_margin": 0.5}, "low_confidence_margin"),
        ({"benchmark_manifest_path": "manifest.json"}, "benchmark cohort mode requires"),
        (
            {
                "benchmark_manifest_path": "manifest.json",
                "benchmark_declaration_path": "declaration.json",
                "benchmark_declaration_signature_path": "declaration.sig",
                "benchmark_declaration_allowed_signers_path": "allowed_signers",
                "benchmark_declaration_signer_id": "reviewer",
                "benchmark_dataset_names": ("dataset",),
                "benchmark_test_cohort": "test",
                "build_cache_root": "cache",
            },
            "direct dataset snapshots, not build cache",
        ),
        (
            {
                "benchmark_manifest_path": "manifest.json",
                "benchmark_declaration_path": "declaration.json",
                "benchmark_declaration_signature_path": "declaration.sig",
                "benchmark_declaration_allowed_signers_path": "allowed_signers",
                "benchmark_declaration_signer_id": "reviewer",
                "benchmark_dataset_names": ("dataset",),
                "benchmark_test_cohort": "test",
            },
            "verified signed split lock",
        ),
        (
            {
                "benchmark_manifest_path": "manifest.json",
                "benchmark_declaration_path": "declaration.json",
                "benchmark_declaration_signature_path": "declaration.sig",
                "benchmark_declaration_allowed_signers_path": "allowed_signers",
                "benchmark_declaration_signer_id": "reviewer",
                "benchmark_dataset_names": ("first", "extra"),
                "benchmark_test_cohort": "test",
                "split_lock_path": "split.json",
                "split_signature_path": "split.sig",
                "trusted_split_signers_path": "split_signers",
                "split_signer_id": "split-reviewer",
            },
            "dataset names must uniquely match ordered dataset paths",
        ),
        ({"split_lock_path": "split.json"}, "signed split mode requires"),
        (
            {
                "split_lock_path": "split.json",
                "split_signature_path": "split.sig",
                "trusted_split_signers_path": "allowed_signers",
                "split_signer_id": "reviewer",
                "holdout": "held-out",
            },
            "holdout cannot be combined with a signed split lock",
        ),
    ],
)
def test_experiment_runner_rejects_invalid_preflight_combinations(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    source = tmp_path / "dataset.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=123).save(source)
    with pytest.raises(ValueError, match=message):
        experiment_runner_impl._run_experiment_impl((source,), tmp_path / "run", **overrides)


def test_experiment_runner_rejects_symlink_output_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "run-link"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        experiment_runner_impl._run_experiment_impl((tmp_path / "unused.npz",), output)


def test_experiment_runner_rejects_unknown_observation_route(tmp_path: Path) -> None:
    source = tmp_path / "dataset.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=321).save(source)

    with pytest.raises(ValueError, match="observation must be ideal_cfr, comb_ls, or precomputed"):
        experiment_runner_impl._run_experiment_impl(
            (source,), tmp_path / "run", observation="unknown", methods=("logreg",)
        )


def test_experiment_runner_requires_the_full_model_trio_for_matched_capacity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=654).save(source)

    with pytest.raises(ValueError, match="requires raw_mlp, cnn, and cvnn"):
        experiment_runner_impl._run_experiment_impl(
            (source,), tmp_path / "run", capacity="matched", methods=("cnn",)
        )


def test_experiment_runner_rejects_unknown_model_methods(tmp_path: Path) -> None:
    source = tmp_path / "dataset.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=987).save(source)

    with pytest.raises(ValueError, match="Unknown method unknown"):
        experiment_runner_impl._run_experiment_impl(
            (source,),
            tmp_path / "run",
            methods=("unknown",),
            seeds=(11,),
            bootstrap=2,
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
    assert result["split"]["physical_independence_status"] == "not_established_by_runner"
    assert (
        result["split"]["partition_support_sha256"]
        == manifest["split_policy"]["partition_support_sha256"]
    )
    assert manifest["split_policy"]["partition_support"]["qualification_status"] == (
        "NOT_QUALIFIED"
    )


@pytest.mark.parametrize("mutate_snapshot", [False, True])
def test_run_experiment_scores_selected_signed_benchmark_cohort(
    tmp_path, monkeypatch, mutate_snapshot
) -> None:
    from datasets.split_policy import (
        SplitParentType,
        SplitPolicy,
        SplitSubject,
        freeze_split_policy_registry,
        make_split_policy_registry,
        verify_openssh_signed_artifact,
        verify_signed_split_policy_registry_attestation,
    )
    from evaluation.benchmark_partitions import materialize_benchmark_manifest

    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    signer = "signed-benchmark-runner-test"
    private_key = tmp_path / "key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-C", signer, "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    registry_namespace = "umcd-split-policy-v2@github.com"
    declaration_namespace = "umcd-benchmark-cohort-v1@github.com"
    registry_keyring = tmp_path / "registry_allowed_signers"
    registry_keyring.write_text(
        f'{signer} namespaces="{registry_namespace}" {public_key}\n', encoding="utf-8"
    )
    declaration_keyring = tmp_path / "declaration_allowed_signers"
    declaration_keyring.write_text(
        f'{signer} namespaces="{declaration_namespace}" {public_key}\n', encoding="utf-8"
    )

    subjects = [
        SplitSubject(f"parent-{index:03d}", SplitParentType.NETWORK_EPISODE, "test")
        for index in range(80)
    ]
    registry = make_split_policy_registry(
        subjects, policy=SplitPolicy.release_v2(seed="signed-benchmark-runner-test")
    )
    lock = tmp_path / "split-registry.json"
    freeze_split_policy_registry(lock, registry)
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            registry_namespace,
            "-f",
            str(private_key),
            str(lock),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    labels = np.arange(len(subjects), dtype=np.uint8) % 2
    data = ChannelDataset(
        h=np.broadcast_to((1 + labels[:, None, None]).astype(np.complex64), (80, 1, 16)).copy(),
        y=labels,
        group=np.asarray([subject.subject_id for subject in subjects]),
        domain=np.full(len(subjects), "scene-a"),
        source=np.full(len(subjects), "sionna_rt"),
        metadata=tuple(
            {
                "geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE",
                "split_subject_id": subject.subject_id,
                "split_source_family": subject.source_family,
                "split_parent_type": subject.parent_type.value,
            }
            for subject in subjects
        ),
    )
    source = tmp_path / "locked.npz"
    data.save(source)
    registry_verified, registry_artifact = verify_signed_split_policy_registry_attestation(
        lock,
        Path(f"{lock}.sig"),
        trusted_allowed_signers_path=registry_keyring,
        signer_id=signer,
    )
    test_parents = [
        parent
        for parent in sorted(registry_verified.subjects)
        if registry_verified.split_for_parent(parent) == "test"
    ]
    positive = next(parent for parent in test_parents if labels[int(parent[-3:])] == 1)
    negative = next(parent for parent in test_parents if labels[int(parent[-3:])] == 0)
    held_out = [positive, negative]
    remaining = [parent for parent in test_parents if parent not in held_out]
    site_array = remaining[:2]
    external = remaining[2:]
    declaration = {
        "schema": "umcd.benchmark_cohort_declaration.v1",
        "schema_version": 1,
        "test_cohorts": {
            "site_held_out": sorted(held_out),
            "site_array_held_out": sorted(site_array),
            "real_external": sorted(external),
        },
        "site_ids": {parent.subject_id: f"site-{parent.subject_id}" for parent in subjects},
        "array_ids": {parent.subject_id: f"array-{parent.subject_id}" for parent in subjects},
        "external_evidence": {
            parent: {
                "measurement_inventory_sha256": "1" * 64,
                "measurement_protocol_sha256": "2" * 64,
                "rights_review_sha256": "3" * 64,
            }
            for parent in external
        },
    }
    declaration_path = tmp_path / "cohort-declaration.json"
    declaration_path.write_text(json.dumps(declaration, sort_keys=True), encoding="utf-8")
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            declaration_namespace,
            "-f",
            str(private_key),
            str(declaration_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    declaration_artifact = verify_openssh_signed_artifact(
        declaration_path,
        Path(f"{declaration_path}.sig"),
        trusted_allowed_signers_path=declaration_keyring,
        signer_id=signer,
        namespace=declaration_namespace,
    )

    def attestation(artifact):
        return {
            "artifact_sha256": artifact.artifact_sha256,
            "signature_sha256": artifact.signature_sha256,
            "allowed_signers_sha256": artifact.allowed_signers_sha256,
            "signer_id": artifact.signer_id,
            "namespace": artifact.namespace,
        }

    benchmark_manifest = materialize_benchmark_manifest(
        registry_verified,
        declaration,
        [("locked", hashlib.sha256(source.read_bytes()).hexdigest(), data)],
        registry_attestation=attestation(registry_artifact),
        declaration_attestation=attestation(declaration_artifact),
    )
    manifest_path = tmp_path / "benchmark-manifest.json"
    manifest_path.write_text(json.dumps(benchmark_manifest, sort_keys=True), encoding="utf-8")
    output = tmp_path / "runs" / "benchmark"

    if mutate_snapshot:
        stable_input_bytes = experiment_runner_impl._stable_input_bytes

        def mutate_after_snapshot(path):
            payload = stable_input_bytes(path)
            Path(path).write_bytes(Path(path).read_bytes() + b"changed after snapshot")
            return payload

        monkeypatch.setattr(experiment_runner_impl, "_stable_input_bytes", mutate_after_snapshot)

    run_arguments = {
        "split_lock_path": lock,
        "split_signature_path": Path(f"{lock}.sig"),
        "trusted_split_signers_path": registry_keyring,
        "split_signer_id": signer,
        "benchmark_manifest_path": manifest_path,
        "benchmark_declaration_path": declaration_path,
        "benchmark_declaration_signature_path": Path(f"{declaration_path}.sig"),
        "benchmark_declaration_allowed_signers_path": declaration_keyring,
        "benchmark_declaration_signer_id": signer,
        "benchmark_dataset_names": ("locked",),
        "benchmark_test_cohort": "site_held_out",
    }
    if mutate_snapshot:
        with pytest.raises(ValueError, match="changed after its verified snapshot"):
            run_experiment(
                [source], output, methods=("logreg",), seeds=(11,), bootstrap=100, **run_arguments
            )
        assert (output / "benchmark-cohort-preflight.json").is_file()
        assert not (output / "manifest.json").exists()
        return

    run_experiment(
        [source], output, methods=("logreg",), seeds=(11,), bootstrap=100, **run_arguments
    )

    saved_manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))[0]
    assert saved_manifest["split_policy"]["benchmark_cohort"]["selected_test_cohort"] == (
        "site_held_out"
    )
    assert result["split"]["test_size"] == len(held_out)
    expected_test_indices = sorted(int(parent[-3:]) for parent in held_out)
    assert result["split"]["benchmark_cohort"]["selected_test_indices_sha256"] == canonical_hash(
        expected_test_indices
    )
    assert result["model_input_artifact_contract_verification"]["benchmark_cohort"] == {
        "selected_test_cohort": "site_held_out",
        "benchmark_manifest_sha256": benchmark_manifest["manifest_sha256"],
        "selected_test_indices_sha256": canonical_hash(expected_test_indices),
    }
    model_info_path = output / "models" / "logreg_seed11_modelinfo.json"
    original_sidecar = model_info_path.read_bytes()
    tampered_sidecar = json.loads(original_sidecar)
    tampered_sidecar["benchmark_cohort"]["manifest_sha256"] = "f" * 64
    model_info_path.write_text(json.dumps(tampered_sidecar), encoding="utf-8")
    with np.load(output / "predictions" / "logreg_seed11.npz", allow_pickle=False) as prediction:
        partition_indices = {
            "train": prediction["train_idx"],
            "validation": prediction["val_idx"],
            "test": prediction["test_idx"],
        }
    with pytest.raises(ValueError, match="provenance does not match replay"):
        runner_module.verify_model_input_artifact_contract(
            model_info_path,
            output / "models" / "logreg_seed11.joblib",
            partition_indices=partition_indices,
            benchmark_cohort=result["split"]["benchmark_cohort"],
        )
    model_info_path.write_bytes(original_sidecar)


def test_benchmark_preflight_failure_receipt_is_exclusive(tmp_path) -> None:
    output = tmp_path / "runs" / "preflight-failure"
    with pytest.raises(ValueError, match="No datasets supplied"):
        run_experiment(
            [],
            output,
            benchmark_manifest_path=tmp_path / "missing-manifest.json",
            benchmark_test_cohort="site_held_out",
        )
    receipt_path = output / "benchmark-cohort-preflight.json"
    first = receipt_path.read_bytes()
    receipt = json.loads(first)
    assert receipt == {
        "schema": "loslab.benchmark_cohort_preflight.v1",
        "status": "FAILED_BEFORE_RUN_MANIFEST",
        "qualification_status": "NOT_QUALIFIED",
        "selected_test_cohort": "site_held_out",
        "error_type": "ValueError",
        "error_message": "No datasets supplied",
    }

    with pytest.raises(ValueError, match="No datasets supplied"):
        run_experiment(
            [],
            output,
            benchmark_manifest_path=tmp_path / "other-manifest.json",
            benchmark_test_cohort="site_array_held_out",
        )
    assert receipt_path.read_bytes() == first

    protected_output = tmp_path / "runs" / "existing-user-data"
    protected_output.mkdir()
    existing_manifest = protected_output / "manifest.json"
    existing_manifest.write_text("preserve this user file", encoding="utf-8")
    with pytest.raises(ValueError, match="No datasets supplied"):
        run_experiment(
            [],
            protected_output,
            benchmark_manifest_path=tmp_path / "missing-manifest.json",
            benchmark_test_cohort="site_held_out",
        )
    assert existing_manifest.read_text(encoding="utf-8") == "preserve this user file"
    assert not (protected_output / "benchmark-cohort-preflight.json").exists()


def test_unsupported_cohort_report_names_reasons_and_external_dependencies() -> None:
    report = runner_module._unsupported_benchmark_test_cohorts(
        {
            "site_held_out": {
                "rows": 0,
                "independent_components": 0,
            },
            "site_array_held_out": {
                "rows": 4,
                "independent_components": 2,
            },
            "real_external": {
                "rows": 0,
                "independent_components": 0,
            },
        }
    )
    assert [item["cohort"] for item in report] == ["site_held_out", "real_external"]
    assert report[0]["reason_codes"] == ["NO_ROWS", "NO_INDEPENDENT_COMPONENTS"]
    assert (
        "independent physical-unit validation and trusted signer custody"
        in report[0]["external_dependencies"]
    )
    assert "rights review and measurement validity evidence" in report[1]["external_dependencies"]


def test_locked_split_indices_resolves_child_rows_through_parent_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datasets.channel import ChannelDataset
    from datasets.split_policy import (
        SplitChildBinding,
        SplitParentType,
        SplitPolicy,
        SplitSubject,
        bind_split_children,
        make_split_policy_registry,
    )
    from experiments.runner import _kish_effective_group_count, _locked_split_indices

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

    train, validation, test, calibration, report = _locked_split_indices(data, registry)

    resolved = {
        "train": train,
        "validation": validation,
        "test": test,
        "calibration": calibration,
    }
    assert set(report["parent_counts"]) >= {"train", "validation", "test"}
    expected_indices = list(range(len(subjects)))
    assert sorted(np.concatenate(tuple(resolved.values())).tolist()) == expected_indices
    assert report["calibration_rows_reserved"] == len(calibration)
    support = report["partition_support"]
    assert support["status"] == "PASS_INPUT_SCOPED"
    assert support["qualification_status"] == "NOT_QUALIFIED"
    assert support["physical_independence_status"] == "not_established_by_runner"
    assert report["partition_support_sha256"] == canonical_hash(support)
    for partition, indices in resolved.items():
        assert all(
            registry.split_for_parent(parent_by_row[index]) == partition for index in indices
        )
        partition_support = support["partition_support"][partition]
        assert partition_support["row_count"] == len(indices)
        assert partition_support["class_counts"]["0"] + partition_support["class_counts"][
            "1"
        ] == len(indices)
        assert partition_support["class_availability"] == {
            label: partition_support["class_counts"][label] > 0 for label in ("0", "1")
        }
        assert partition_support["parent_count"] == len({parent_by_row[index] for index in indices})
        assert partition_support["effective_declared_group_count_kish"] == pytest.approx(
            _kish_effective_group_count(
                [
                    sum(data.group[index] == group_id for index in indices)
                    for group_id in set(data.group[indices].tolist())
                ]
            )
        )

    assert _kish_effective_group_count([3, 1]) == pytest.approx(1.6)
    assert _kish_effective_group_count([]) == 0.0

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

    absent_metadata = list(data.metadata)
    absent_metadata[0] = {"split_subject_id": "absent-parent"}
    absent_data = ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=data.source,
        metadata=tuple(absent_metadata),
    )
    with pytest.raises(ValueError, match="absent from signed lock"):
        _locked_split_indices(absent_data, registry)

    partial_data = ChannelDataset(
        h=data.h[:2],
        y=data.y[:2],
        group=data.group[:2],
        domain=data.domain[:2],
        source=data.source[:2],
        metadata=data.metadata[:2],
    )
    with pytest.raises(ValueError, match="does not cover every signed split parent"):
        _locked_split_indices(partial_data, registry)

    padded_source = np.asarray(data.source, dtype=object)
    padded_source[0] = " padded "
    invalid_source_data = ChannelDataset(
        h=data.h,
        y=data.y,
        group=data.group,
        domain=data.domain,
        source=padded_source,
        metadata=data.metadata,
    )
    with pytest.raises(ValueError, match="canonical source and domain IDs"):
        _locked_split_indices(invalid_source_data, registry)

    split_by_parent = {
        parent.subject_id: registry.split_for_parent(parent.subject_id) for parent in parents
    }
    different_partitions = next(
        (first, second)
        for first in parents
        for second in parents
        if split_by_parent[first.subject_id] != split_by_parent[second.subject_id]
    )
    crossing_groups = np.asarray(data.group, dtype=object)
    first_row = subjects.index(different_partitions[0].subject_id)
    second_row = subjects.index(different_partitions[1].subject_id)
    crossing_groups[first_row] = "crossing-partitions"
    crossing_groups[second_row] = "crossing-partitions"
    crossing_group_data = ChannelDataset(
        h=data.h,
        y=data.y,
        group=crossing_groups,
        domain=data.domain,
        source=data.source,
        metadata=data.metadata,
    )
    with pytest.raises(ValueError, match="group crosses signed split partitions"):
        _locked_split_indices(crossing_group_data, registry)

    with monkeypatch.context() as patched:
        patched.setattr(type(registry), "split_for_parent", lambda _self, _subject: "other")
        with pytest.raises(ValueError, match="unsupported locked split partition"):
            _locked_split_indices(data, registry)

    monkeypatch.setattr(type(registry), "split_for_parent", lambda _self, _subject: "train")
    with pytest.raises(ValueError, match="must resolve non-empty train, validation, and test"):
        _locked_split_indices(data, registry)


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
        prior_shift={
            "schema": "loslab.prior_shift.v1",
            "training_prior_kind": "natural",
            "training_prior_id": "debug-observed-training-prior-v1",
            "target_prior_kind": "natural",
            "target_prior_id": "debug-natural-prior-sensitivity-v1",
            "target_positive_prevalence": 0.25,
            "target_prior_evidence_sha256": "a" * 64,
        },
        low_confidence_margin=0.05,
        command_argv=(
            "loslab",
            "experiment",
            "--input",
            str(source),
            "--output",
            str(output),
        ),
        command_source="test_cli_argv",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "complete"
    assert manifest["label_semantics"] == "synthetic_debug_direct_component"
    assert manifest["label_semantics_by_source"] == "SYNTHETIC_DEBUG_CONSTRUCTED"
    assert manifest["inputs"][0]["sha256"]
    replay = manifest["run_replay"]
    assert replay["schema"] == "loslab.run_replay_provenance.v1"
    assert replay["qualification_status"] == "NOT_QUALIFIED"
    assert replay["state"] == "complete"
    assert replay["config_sha256"] == manifest["config_sha256"]
    assert replay["inputs"] == manifest["inputs"]
    assert replay["source"]["package_source_sha256"] == manifest["package_source_sha256"]
    assert replay["source"]["git_commit"] == manifest["source_revision"]
    assert replay["command"]["source"] == "test_cli_argv"
    assert replay["command"]["argv"] == [
        "loslab",
        "experiment",
        "--input",
        str(source),
        "--output",
        str(output),
    ]
    assert replay["backend_identity"]["requested_device"] == "cpu"
    assert {stream["name"] for stream in replay["rng_streams"]} >= {
        "run_seed_11",
        "calibration_partition_seed_11",
        "phase_randomization_control",
    }
    assert replay["attempt"] == {"index": 0, "retry_of_manifest_sha256": None}
    assert (output / "COMPLETE").is_file()
    assert (output / "config" / "resolved.json").is_file()
    assert (output / "models" / "logreg_seed11.joblib").is_file()

    model_info = json.loads(
        (output / "models" / "logreg_seed11_modelinfo.json").read_text(encoding="utf-8")
    )
    model_contract = model_info["model_input_artifact_contract"]
    assert model_contract["schema"] == "loslab.model_input_artifact_contract.v2"
    assert (
        model_contract["model_sha256"]
        == hashlib.sha256((output / "models" / "logreg_seed11.joblib").read_bytes()).hexdigest()
    )
    assert model_contract["feature_contract"]["allowed_source_fields"] == ["observed_cfr"]
    assert "data.y" in model_contract["feature_contract"]["excluded_truth_and_metadata_fields"]
    assert model_contract["feature_contract"]["feature_transform"] == "logmag_pdp_moments_v1"
    assert len(model_contract["feature_contract"]["feature_transform_implementation_sha256"]) == 64
    assert model_contract["feature_contract"]["feature_matrix"]["sample_count"] == len(
        ChannelDataset.load(source).y
    )
    assert model_contract["feature_contract"]["partition_feature_matrices"].keys() == {
        "train",
        "validation",
        "test",
        "calibration",
    }
    assert (
        verify_model_input_artifact_contract(
            output / "models" / "logreg_seed11_modelinfo.json",
            output / "models" / "logreg_seed11.joblib",
        )["status"]
        == "PASS_MODEL_INPUT_ARTIFACT_BINDING"
    )
    model_info_path = output / "models" / "logreg_seed11_modelinfo.json"
    model_path = output / "models" / "logreg_seed11.joblib"
    data = ChannelDataset.load(source)
    features = observable_features(data.model_inputs().observed_cfr)
    with np.load(output / "predictions" / "logreg_seed11.npz") as prediction:
        partitions = {
            "train": prediction["train_idx"],
            "validation": prediction["val_idx"],
            "test": prediction["test_idx"],
            "calibration": prediction["calibration_idx"],
        }
    matrix_audit = verify_model_input_artifact_contract(
        model_info_path,
        model_path,
        method_input=features,
        partition_indices=partitions,
    )
    assert matrix_audit["feature_matrix_verified"] is True
    report_row = json.loads((output / "results.json").read_text(encoding="utf-8"))[0]
    assert report_row["model_input_feature_matrix_verified"] is True
    partition_support = report_row["split"]["declared_partition_support"]
    assert report_row["split"]["declared_partition_support_sha256"] == canonical_hash(
        partition_support
    )
    assert set(partition_support) == {"train", "validation", "calibration", "test"}
    support_replay = verify_declared_partition_support(
        data,
        partitions,
        partition_support,
        report_row["split"]["declared_partition_support_sha256"],
    )
    assert support_replay["status"] == "PASS_INPUT_SCOPED_REPLAY"
    assert support_replay["qualification_status"] == "NOT_QUALIFIED"
    with pytest.raises(ValueError, match="must name train, validation, calibration, and test"):
        verify_declared_partition_support(
            data,
            {**partitions, "unexpected": partitions["test"]},
            partition_support,
            report_row["split"]["declared_partition_support_sha256"],
        )
    malformed_partitions = dict(partitions)
    malformed_partitions["train"] = partitions["train"].astype(float)
    with pytest.raises(ValueError, match="one-dimensional integer array"):
        verify_declared_partition_support(
            data,
            malformed_partitions,
            partition_support,
            report_row["split"]["declared_partition_support_sha256"],
        )
    duplicate_partitions = dict(partitions)
    duplicate_partitions["train"] = np.append(partitions["train"], partitions["train"][0])
    with pytest.raises(ValueError, match="unique, in range, and disjoint"):
        verify_declared_partition_support(
            data,
            duplicate_partitions,
            partition_support,
            report_row["split"]["declared_partition_support_sha256"],
        )
    incomplete_partitions = dict(partitions)
    incomplete_partitions["test"] = partitions["test"][:-1]
    with pytest.raises(ValueError, match="cover every source dataset row exactly once"):
        verify_declared_partition_support(
            data,
            incomplete_partitions,
            partition_support,
            report_row["split"]["declared_partition_support_sha256"],
        )
    tampered_support = json.loads(json.dumps(partition_support))
    tampered_support["test"]["declared_group_count"] += 1
    with pytest.raises(ValueError, match="does not replay"):
        verify_declared_partition_support(
            data,
            partitions,
            tampered_support,
            report_row["split"]["declared_partition_support_sha256"],
        )
    for partition, indices in partitions.items():
        support = partition_support[partition]
        assert support["row_count"] == len(indices)
        assert support["physical_independence_status"] == "not_established_by_runner"
        assert support["effective_declared_group_count_kish"] <= support["declared_group_count"]
    support_cells = report_row["class_support"]["cells"]
    assert report_row["class_support"]["schema"] == "loslab.class_support.v1"
    assert {cell["class_label"] for cell in support_cells} == {0, 1}
    assert all(cell["independent_group_count"] >= 1 for cell in support_cells)
    assert report_row["prior_shift"]["fit_partition"] == "calibration"
    assert report_row["prior_shift"]["target_prior_id"] == "debug-natural-prior-sensitivity-v1"
    assert report_row["prior_shift"]["target_prior_evidence_sha256"] == "a" * 64
    assert report_row["low_confidence_audit"]["threshold_source"] == "validation"
    assert report_row["low_confidence_audit"]["margin"] == pytest.approx(0.05)
    assert report_row["model_output_threshold"] == pytest.approx(model_info["threshold"])
    assert model_info["prior_shift"] == report_row["prior_shift"]
    with np.load(output / "predictions" / "logreg_seed11.npz") as prediction:
        assert "test_source" in prediction.files
        assert len(prediction["test_source"]) == len(prediction["y_test"])
        assert "p_test_raw" in prediction.files
        assert "test_low_confidence" in prediction.files
        assert not np.allclose(prediction["p_test"], prediction["p_test_raw"])
    changed_features = features.copy()
    changed_features[0, 0] += 1.0
    with pytest.raises(ValueError, match="feature matrix differs"):
        verify_model_input_artifact_contract(
            model_info_path,
            model_path,
            method_input=changed_features,
            partition_indices=partitions,
        )
    overlapping_partitions = dict(partitions)
    overlapping_partitions["validation"] = np.concatenate(
        (partitions["validation"], partitions["train"][:1])
    )
    with pytest.raises(ValueError, match="partitions overlap"):
        verify_model_input_artifact_contract(
            model_info_path,
            model_path,
            method_input=features,
            partition_indices=overlapping_partitions,
        )
    legacy_info = json.loads(model_info_path.read_text(encoding="utf-8"))
    legacy_contract = legacy_info["model_input_artifact_contract"]
    legacy_contract["schema"] = "loslab.model_input_artifact_contract.v1"
    legacy_feature = legacy_contract["feature_contract"]
    legacy_feature["schema"] = "loslab.model_input_feature_contract.v1"
    legacy_feature.pop("feature_transform_implementation_sha256")
    legacy_feature.pop("feature_matrix")
    legacy_feature.pop("partition_feature_matrices")
    legacy_contract["split"].pop("calibration_rows", None)
    legacy_contract["feature_contract_sha256"] = canonical_hash(legacy_feature)
    legacy_path = tmp_path / "legacy-modelinfo.json"
    legacy_path.write_text(json.dumps(legacy_info), encoding="utf-8")
    assert verify_model_input_artifact_contract(legacy_path, model_path)["status"] == (
        "PASS_MODEL_INPUT_ARTIFACT_BINDING"
    )
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
        verify_model_input_artifact_contract(model_info_path, model_path)


def test_run_experiment_persists_failure_after_manifest_start(tmp_path, monkeypatch) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=17).save(source)
    output = tmp_path / "runs" / "failed"

    def fail_metrics(*args, **kwargs):
        raise RuntimeError("injected post-start failure")

    monkeypatch.setattr(runner_module.experiment_runner, "binary_metrics", fail_metrics)
    with pytest.raises(RuntimeError, match="injected post-start failure"):
        run_experiment([source], output, methods=("logreg",), seeds=(11,), bootstrap=20)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["state"] == "failed"
    assert manifest["failure"]["stage"] == "experiment"
    assert manifest["run_replay"]["state"] == "failed"
    assert manifest["run_replay"]["failure"]["message"] == "injected post-start failure"


def test_run_experiment_records_wrapper_failure_when_replay_transition_fails(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=18).save(source)
    output = tmp_path / "runs" / "replay-failure"
    monkeypatch.setattr(
        runner_module.experiment_runner,
        "binary_metrics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("injected model failure")),
    )
    monkeypatch.setattr(
        runner_module.experiment_runner,
        "mark_failed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("manifest unavailable")),
    )

    with pytest.raises(RuntimeError, match="failure replay provenance could not be recorded"):
        run_experiment([source], output, methods=("logreg",), seeds=(11,), bootstrap=20)


def test_run_experiment_writes_benchmark_preflight_receipt_before_manifest(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=19).save(source)
    output = tmp_path / "runs" / "benchmark-preflight"

    with pytest.raises(ValueError, match="verified signed split lock"):
        run_experiment(
            [source],
            output,
            methods=("logreg",),
            seeds=(11,),
            bootstrap=20,
            benchmark_manifest_path=tmp_path / "manifest.json",
            benchmark_declaration_path=tmp_path / "declaration.json",
            benchmark_declaration_signature_path=tmp_path / "declaration.sig",
            benchmark_declaration_allowed_signers_path=tmp_path / "allowed-signers",
            benchmark_declaration_signer_id="reviewer",
            benchmark_dataset_names=("debug",),
            benchmark_test_cohort="site_held_out",
        )

    receipt = json.loads((output / "benchmark-cohort-preflight.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "FAILED_BEFORE_RUN_MANIFEST"
    assert not (output / "manifest.json").exists()


def test_run_experiment_supports_comb_ls_observations(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=20).save(source)
    output = tmp_path / "runs" / "comb-ls"

    run_experiment(
        [source],
        output,
        methods=("logreg",),
        seeds=(11,),
        bootstrap=20,
        observation="comb_ls",
    )

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observation_semantics"] == "generic noisy comb pilot estimate, NOT 3GPP DMRS"


def test_run_experiment_takes_coral_device_and_checkpoint_routes(tmp_path, monkeypatch) -> None:
    import torch

    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=21).save(source)

    class Predictor:
        def predict_proba(self, values):
            scores = np.linspace(0.1, 0.9, len(values), dtype=np.float64)
            return np.column_stack((1.0 - scores, scores))

        def save(self, path, *, threshold, include_phase):
            Path(path).write_text(f"{threshold}:{include_phase}", encoding="utf-8")

    monkeypatch.setattr(experiment_runner_impl, "_sync_torch_device", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(
        experiment_runner_impl, "fit_coral_mlp", lambda *_args, **_kwargs: (Predictor(), {})
    )

    output = tmp_path / "runs" / "coral"
    run_experiment([source], output, methods=("coral",), seeds=(11,), bootstrap=10, device="cuda:0")

    assert (output / "models" / "coral_seed11.pt").read_text(encoding="utf-8")


def test_run_experiment_records_prior_shift_report_drift(tmp_path, monkeypatch) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=22).save(source)
    original = experiment_runner_impl.adjust_probabilities_for_prior_shift
    calls = 0

    def drifting_report(*args, **kwargs):
        nonlocal calls
        calls += 1
        probabilities, report = original(*args, **kwargs)
        if calls == 2:
            report = {**report, "injected_partition": "test"}
        return probabilities, report

    monkeypatch.setattr(
        experiment_runner_impl, "adjust_probabilities_for_prior_shift", drifting_report
    )
    prior_shift = {
        "schema": "loslab.prior_shift.v1",
        "training_prior_kind": "natural",
        "training_prior_id": "debug-training-prior",
        "target_prior_kind": "natural",
        "target_prior_id": "debug-target-prior",
        "target_positive_prevalence": 0.25,
        "target_prior_evidence_sha256": "a" * 64,
    }

    with pytest.raises(RuntimeError, match="train-only prior-shift report changed"):
        run_experiment(
            [source],
            tmp_path / "runs" / "prior-shift-drift",
            methods=("logreg",),
            seeds=(11,),
            bootstrap=10,
            prior_shift=prior_shift,
        )


def test_run_experiment_uses_locked_calibration_partition(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    source = tmp_path / "debug.npz"
    data = debug_multipath(n_scenes=4, links_per_scene=12, seed=24)
    data = ChannelDataset(
        h=data.h,
        y=data.y,
        group=np.asarray([f"independent-{index:03d}" for index in range(len(data.y))]),
        domain=data.domain,
        source=data.source,
        metadata=data.metadata,
    )
    data.save(source)
    partitions = (
        np.arange(0, 24, dtype=np.int64),
        np.arange(24, 32, dtype=np.int64),
        np.arange(32, 40, dtype=np.int64),
        np.arange(40, 48, dtype=np.int64),
        {"mode": "signed_locked", "registry_sha256": "c" * 64},
    )
    monkeypatch.setattr(
        experiment_runner_impl,
        "verify_signed_split_policy_registry_attestation",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(
                artifact_sha256="a" * 64,
                signature_sha256="b" * 64,
                allowed_signers_sha256="c" * 64,
            ),
        ),
    )
    monkeypatch.setattr(experiment_runner_impl, "_locked_split_indices", lambda *_args: partitions)
    prior_shift = {
        "schema": "loslab.prior_shift.v1",
        "training_prior_kind": "natural",
        "training_prior_id": "locked-training-prior",
        "target_prior_kind": "natural",
        "target_prior_id": "locked-target-prior",
        "target_positive_prevalence": 0.25,
        "target_prior_evidence_sha256": "d" * 64,
    }

    run_experiment(
        [source],
        tmp_path / "runs" / "locked-calibration",
        methods=("logreg",),
        seeds=(11,),
        bootstrap=10,
        split_lock_path=tmp_path / "lock.json",
        split_signature_path=tmp_path / "lock.sig",
        trusted_split_signers_path=tmp_path / "trusted-signers",
        split_signer_id="test-signer",
        prior_shift=prior_shift,
    )


def test_run_experiment_rejects_invalid_class_support_payload(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=23).save(source)
    monkeypatch.setattr(
        experiment_runner_impl,
        "audit_class_support",
        lambda **_kwargs: SimpleNamespace(
            cells=[
                SimpleNamespace(
                    split_id="test",
                    family_id="synthetic_debug",
                    class_label=0,
                    record_count=1,
                    independent_group_count=1,
                    status="impossible",
                    unexpected_field=True,
                )
            ]
        ),
    )

    with pytest.raises(ValueError, match="class support report violates schema"):
        run_experiment(
            [source],
            tmp_path / "runs" / "bad-class-support",
            methods=("logreg",),
            seeds=(11,),
            bootstrap=10,
        )


def test_run_experiment_preserves_existing_manifest_after_failure(tmp_path, monkeypatch) -> None:
    output = tmp_path / "runs" / "existing"
    output.mkdir(parents=True)
    (output / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        experiment_runner_impl,
        "_run_experiment_impl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("preflight failure")),
    )

    with pytest.raises(ValueError, match="preflight failure"):
        run_experiment(output=output)

    assert (output / "manifest.json").read_text(encoding="utf-8") == "{}"


def test_run_experiment_propagates_failure_without_output_root(monkeypatch) -> None:
    monkeypatch.setattr(
        experiment_runner_impl,
        "_run_experiment_impl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("missing output")),
    )

    with pytest.raises(ValueError, match="missing output"):
        run_experiment()


def test_run_experiment_skips_manifest_receipt_for_symlink_output(tmp_path, monkeypatch) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output-link"
    output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        experiment_runner_impl,
        "_run_experiment_impl",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("linked output")),
    )

    with pytest.raises(ValueError, match="linked output"):
        run_experiment(output=output)

    assert not (target / "benchmark-cohort-preflight.json").exists()


def test_run_experiment_preserves_nonrunning_manifest_on_failure(tmp_path, monkeypatch) -> None:
    output = tmp_path / "runs" / "nonrunning"

    def fail_after_persisting_terminal_manifest(*_args, **_kwargs):
        output.mkdir(parents=True)
        (output / "manifest.json").write_text(
            json.dumps({"run_replay": {"state": "complete"}}), encoding="utf-8"
        )
        raise ValueError("terminal manifest already present")

    monkeypatch.setattr(
        experiment_runner_impl, "_run_experiment_impl", fail_after_persisting_terminal_manifest
    )

    with pytest.raises(ValueError, match="terminal manifest already present"):
        run_experiment(["unused.npz"], output)

    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == {
        "run_replay": {"state": "complete"}
    }


def test_run_experiment_materializes_input_dataset_through_build_dag(tmp_path) -> None:
    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=4, links_per_scene=12, seed=73).save(source)
    cache = tmp_path / "build-cache"

    first_output = tmp_path / "runs" / "cache-first"
    run_experiment(
        [source],
        first_output,
        methods=("logreg",),
        seeds=(17,),
        bootstrap=20,
        build_cache_root=cache,
    )
    first_manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    first_contract = json.loads(
        (first_output / "models/logreg_seed17_modelinfo.json").read_text(encoding="utf-8")
    )["model_input_artifact_contract"]

    second_output = tmp_path / "runs" / "cache-second"
    run_experiment(
        [source],
        second_output,
        methods=("logreg",),
        seeds=(17,),
        bootstrap=20,
        build_cache_root=cache,
    )
    second_manifest = json.loads((second_output / "manifest.json").read_text(encoding="utf-8"))
    second_contract = json.loads(
        (second_output / "models/logreg_seed17_modelinfo.json").read_text(encoding="utf-8")
    )["model_input_artifact_contract"]

    assert first_manifest["dataset_build_node"]["cache_reused"] is False
    assert second_manifest["dataset_build_node"]["cache_reused"] is True
    assert (
        first_manifest["dataset_build_node"]["node_key"]
        == second_manifest["dataset_build_node"]["node_key"]
    )
    assert (
        first_manifest["dataset_build_node"]["artifact_sha256"]
        == second_manifest["dataset_build_node"]["artifact_sha256"]
    )
    assert (
        first_contract["feature_contract"]["feature_matrix"]
        == second_contract["feature_contract"]["feature_matrix"]
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


def test_parameter_matched_neural_trio_uses_common_split_and_records_budget(
    tmp_path, monkeypatch
) -> None:
    from models.capacity import matched_capacity_names

    source = tmp_path / "debug.npz"
    debug_multipath(n_scenes=8, links_per_scene=8, n_subcarriers=16, seed=89).save(source)
    profile = matched_capacity_names(links=1, subcarriers=16)
    seen_capacities = {}

    class Predictor:
        def predict_proba(self, values):
            scores = np.linspace(0.1, 0.9, len(values), dtype=np.float64)
            return np.column_stack((1.0 - scores, scores))

        def save(self, path, **kwargs):
            path.write_bytes(b"synthetic parameter-matching predictor")

    original_fit_sklearn = runner_module.experiment_runner.fit_sklearn

    def capture_sklearn(x, y, *, seed, method, capacity):
        seen_capacities["raw_mlp"] = capacity
        return original_fit_sklearn(x, y, seed=seed, method=method, capacity=capacity)

    def fake_real_cnn(x, y, x_validation, y_validation, *, seed, device, capacity):
        seen_capacities["cnn"] = capacity
        return Predictor(), {"trainable_parameters": profile["real_cnn_parameters"]}

    def fake_complex_cnn(x, y, x_validation, y_validation, *, seed, device, capacity):
        seen_capacities["cvnn"] = capacity
        return Predictor(), {"trainable_parameters": profile["complex_cnn_parameters"]}

    monkeypatch.setattr(runner_module.experiment_runner, "fit_sklearn", capture_sklearn)
    monkeypatch.setattr(runner_module.experiment_runner, "fit_real_cnn", fake_real_cnn)
    monkeypatch.setattr(runner_module.experiment_runner, "fit_complex_cnn", fake_complex_cnn)
    output = tmp_path / "runs" / "matched"
    run_experiment(
        [source],
        output,
        methods=("raw_mlp", "cnn", "cvnn"),
        seeds=(23,),
        bootstrap=100,
        capacity="matched",
    )

    assert seen_capacities == {
        "raw_mlp": profile["mlp_capacity"],
        "cnn": profile["real_cnn_capacity"],
        "cvnn": profile["complex_cnn_capacity"],
    }
    rows = json.loads((output / "results.json").read_text(encoding="utf-8"))
    assert len({row["split"]["indices_sha256"] for row in rows}) == 1
    for row in rows:
        match = row["compute"]["parameter_match"]
        assert match["target_parameters"] == profile["target_parameters"]
        assert match["relative_error"] <= 0.1
