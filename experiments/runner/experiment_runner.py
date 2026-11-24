"""Run entry point for a configured experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from artifacts.layout import RunLayout
from artifacts.manifest import mark_complete, mark_failed, run_manifest
from artifacts.provenance import (
    atomic_json,
    canonical_hash,
    package_source_sha256,
    sha256_file,
    versions,
)
from artifacts.run_replay import build_run_replay_provenance
from configuration.loader import resolve_config
from datasets.channel import ChannelDataset
from datasets.split_policy import (
    verify_signed_split_policy_registry_attestation,
)
from evaluation.scoring.class_support import audit_class_support
from evaluation.scoring.low_confidence import build_low_confidence_audit
from evaluation.scoring.metrics import (
    binary_metrics,
    block_bootstrap_auc,
    group_split,
    holdout_domain,
    in_domain_split,
)
from evaluation.scoring.prior_shift import (
    adjust_probabilities_between_priors,
    adjust_probabilities_for_prior_shift,
    validate_prior_shift_spec,
)
from evaluation.scoring.probability_calibration import (
    apply_platt_calibrator,
    fit_group_balanced_platt_calibrator,
    invert_platt_calibrator,
)
from evaluation.scoring.statistics import paired_seed_delta
from learning import fit_complex_cnn, fit_coral_mlp, fit_real_cnn, fit_sklearn
from models.capacity import capacity_spec, matched_capacity_names
from propagation.physics import delay_k_proxy_features, observable_features
from receivers.pilots import comb_ls_estimate

from .contracts import (
    build_model_input_artifact_contract,
    verify_model_input_artifact_contract,
)
from .features import (
    _MODEL_FEATURE_ROUTES,
    _VALIDATION_THRESHOLD_POLICY,
    _flatten_real_imag_cfr,
    _identity_complex_cfr,
    _label_semantics_by_source,
    _learned_parameter_count,
    _log_power_per_link_subcarrier,
    _precomputed_observation_semantics,
    _run_replay_rng_streams,
    _sparse_multipath_features,
    _stable_input_bytes,
    _stable_single_link_sha256,
    _sync_torch_device,
    _validate_evaluation_payload,
    _validate_npz_snapshot,
)
from .validation import (
    _declared_partition_support,
    _locked_split_indices,
    _materialize_channel_dataset_build_node,
    _metadata_only_features,
    _oracle_label_leak_features,
    _permute_training_feature_rows,
    _permute_training_labels,
    _phase_randomized_real_imag_cfr,
    _resolve_benchmark_test_cohort,
    _resolved_label_semantics,
    _single_metadata_features,
    _source_only_features,
    _validation_threshold_selection_record,
    _write_benchmark_preflight_failure_receipt,
)


def _run_experiment_impl(
    paths: Sequence[str | Path],
    output: str | Path,
    *,
    holdout: str | None = None,
    methods: Sequence[str] = ("rf", "histgb", "mlp", "raw_mlp", "coral", "coral0"),
    seeds: Sequence[int] = (11, 23, 47, 59, 71, 83, 97, 101, 109, 127),
    bootstrap: int = 200,
    include_phase: bool = False,
    observation: str = "ideal_cfr",
    pilot_spacing: int = 4,
    pilot_snr_db: float = 12.0,
    observation_seed: int = 2026,
    phase_randomization_seed: int = 2027,
    label_permutation_seed: int = 2028,
    row_shuffle_seed: int = 2029,
    device: str = "cpu",
    coral_penalty: float = 0.1,
    capacity: str = "medium",
    split_lock_path: str | Path | None = None,
    split_signature_path: str | Path | None = None,
    trusted_split_signers_path: str | Path | None = None,
    split_signer_id: str | None = None,
    benchmark_manifest_path: str | Path | None = None,
    benchmark_declaration_path: str | Path | None = None,
    benchmark_declaration_signature_path: str | Path | None = None,
    benchmark_declaration_allowed_signers_path: str | Path | None = None,
    benchmark_declaration_signer_id: str | None = None,
    benchmark_dataset_names: Sequence[str] | None = None,
    benchmark_test_cohort: str | None = None,
    build_cache_root: str | Path | None = None,
    prior_shift: dict[str, Any] | None = None,
    low_confidence_margin: float | None = None,
    command_argv: Sequence[str] | None = None,
    command_source: str | None = None,
    attempt_index: int = 0,
    retry_of_manifest_sha256: str | None = None,
) -> dict:
    if not paths:
        raise ValueError("No datasets supplied")
    output_path = Path(output).expanduser()
    if output_path.is_symlink():
        raise ValueError(f"experiment output root must not be a symlink: {output_path}")
    if (
        isinstance(phase_randomization_seed, bool)
        or not isinstance(phase_randomization_seed, (int, np.integer))
        or not 0 <= phase_randomization_seed < 2**63
    ):
        raise ValueError("phase_randomization_seed must be an integer in [0, 2**63)")
    if (
        isinstance(label_permutation_seed, bool)
        or not isinstance(label_permutation_seed, (int, np.integer))
        or not 0 <= label_permutation_seed < 2**63
    ):
        raise ValueError("label_permutation_seed must be an integer in [0, 2**63)")
    if (
        isinstance(row_shuffle_seed, bool)
        or not isinstance(row_shuffle_seed, (int, np.integer))
        or not 0 <= row_shuffle_seed < 2**63
    ):
        raise ValueError("row_shuffle_seed must be an integer in [0, 2**63)")
    resolved_prior_shift = None if prior_shift is None else validate_prior_shift_spec(prior_shift)
    if resolved_prior_shift is not None:
        _validate_evaluation_payload(resolved_prior_shift, "prior-shift.v1.schema.json")
    if low_confidence_margin is not None and (
        isinstance(low_confidence_margin, bool)
        or not isinstance(low_confidence_margin, (int, float))
        or not np.isfinite(low_confidence_margin)
        or not 0 <= low_confidence_margin < 0.5
    ):
        raise ValueError("low_confidence_margin must be finite and in [0, 0.5)")
    capacity_spec(capacity)
    benchmark_args = (
        benchmark_manifest_path,
        benchmark_declaration_path,
        benchmark_declaration_signature_path,
        benchmark_declaration_allowed_signers_path,
        benchmark_declaration_signer_id,
        benchmark_dataset_names,
        benchmark_test_cohort,
    )
    benchmark_enabled = any(value is not None for value in benchmark_args)
    if benchmark_enabled and not all(value is not None for value in benchmark_args):
        raise ValueError(
            "benchmark cohort mode requires manifest, signed declaration inputs, dataset names, "
            "and a selected test cohort"
        )
    if benchmark_enabled and build_cache_root is not None:
        raise ValueError("benchmark cohort mode requires direct dataset snapshots, not build cache")
    if benchmark_enabled and not all(
        value is not None
        for value in (
            split_lock_path,
            split_signature_path,
            trusted_split_signers_path,
            split_signer_id,
        )
    ):
        raise ValueError("benchmark cohort mode requires a verified signed split lock")
    dataset_build_node = None
    benchmark_datasets: list[tuple[str, str, ChannelDataset]] | None = None
    input_snapshot_digests: dict[str, str] = {}
    if benchmark_enabled:
        names = list(benchmark_dataset_names or ())
        if (
            len(names) != len(paths)
            or len(set(names)) != len(names)
            or any(
                not isinstance(name, str) or not name.strip() or name != name.strip()
                for name in names
            )
        ):
            raise ValueError("benchmark dataset names must uniquely match ordered dataset paths")
        benchmark_datasets = []
        for name, path in zip(names, paths, strict=True):
            payload = _stable_input_bytes(path)
            _validate_npz_snapshot(
                payload,
                label=f"benchmark ChannelDataset {name!r}",
                max_uncompressed_bytes=8 * 1024**3,
            )
            digest = hashlib.sha256(payload).hexdigest()
            dataset = ChannelDataset.load_bytes(payload)
            benchmark_datasets.append((name, digest, dataset))
            input_snapshot_digests[str(Path(path).expanduser().resolve())] = digest
        data = ChannelDataset.concatenate(
            [dataset for _name, _digest, dataset in benchmark_datasets]
        )
    elif build_cache_root is None:
        data = ChannelDataset.concatenate([ChannelDataset.load(p) for p in paths])
    else:
        data, dataset_build_node = _materialize_channel_dataset_build_node(paths, build_cache_root)
    signed_split_args = (
        split_lock_path,
        split_signature_path,
        trusted_split_signers_path,
        split_signer_id,
    )
    if any(value is not None for value in signed_split_args) and not all(
        value is not None for value in signed_split_args
    ):
        raise ValueError(
            "signed split mode requires lock, signature, trusted keyring, and signer ID"
        )
    locked_indices = None
    benchmark_test_indices: np.ndarray | None = None
    benchmark_provenance: dict[str, Any] | None = None
    split_provenance: dict[str, Any] = {"mode": "unlocked_pilot"}
    if all(value is not None for value in signed_split_args):
        if holdout is not None:
            raise ValueError("holdout cannot be combined with a signed split lock")
        lock_path = Path(split_lock_path)  # type: ignore[arg-type]
        signature_path = Path(split_signature_path)  # type: ignore[arg-type]
        signers_path = Path(trusted_split_signers_path)  # type: ignore[arg-type]
        registry, registry_attestation = verify_signed_split_policy_registry_attestation(
            lock_path,
            signature_path,
            trusted_allowed_signers_path=signers_path,
            signer_id=str(split_signer_id),
        )
        locked_indices = _locked_split_indices(data, registry)
        split_provenance = {
            **locked_indices[4],
            "signer_id": split_signer_id,
            "lock_file_sha256": registry_attestation.artifact_sha256,
            "signature_file_sha256": registry_attestation.signature_sha256,
            "trusted_keyring_sha256": registry_attestation.allowed_signers_sha256,
        }
        if benchmark_enabled:
            assert benchmark_datasets is not None
            benchmark_test_indices, benchmark_provenance = _resolve_benchmark_test_cohort(
                paths=paths,
                dataset_names=benchmark_dataset_names or (),
                datasets=benchmark_datasets,
                registry=registry,
                registry_attestation=registry_attestation,
                manifest_path=benchmark_manifest_path,  # type: ignore[arg-type]
                declaration_path=benchmark_declaration_path,  # type: ignore[arg-type]
                declaration_signature_path=benchmark_declaration_signature_path,  # type: ignore[arg-type]
                declaration_allowed_signers_path=benchmark_declaration_allowed_signers_path,  # type: ignore[arg-type]
                declaration_signer_id=str(benchmark_declaration_signer_id),
                test_cohort=str(benchmark_test_cohort),
                locked_indices=locked_indices,
            )
            split_provenance["benchmark_cohort"] = benchmark_provenance
    resolved_label_semantics = _resolved_label_semantics(data)
    model_inputs = data.model_inputs()
    if observation == "ideal_cfr":
        observable = model_inputs.observed_cfr
    elif observation == "comb_ls":
        observable = comb_ls_estimate(
            model_inputs.observed_cfr,
            spacing=pilot_spacing,
            snr_db=pilot_snr_db,
            seed=observation_seed,
        )
    elif observation == "precomputed":
        observable = model_inputs.observed_cfr
    else:
        raise ValueError("observation must be ideal_cfr, comb_ls, or precomputed")
    x = observable_features(observable, include_phase=include_phase)
    raw_real_imag = _flatten_real_imag_cfr(observable)
    power_only = _log_power_per_link_subcarrier(observable)
    if capacity == "matched":
        if not {"raw_mlp", "cnn", "cvnn"} <= set(methods):
            raise ValueError("capacity='matched' requires raw_mlp, cnn, and cvnn in one run")
        matched_capacities = matched_capacity_names(
            links=int(observable.shape[1]), subcarriers=int(observable.shape[2])
        )
        model_capacities = {
            method: str(
                matched_capacities[
                    "mlp_capacity"
                    if method == "raw_mlp"
                    else "real_cnn_capacity"
                    if method == "cnn"
                    else "complex_cnn_capacity"
                ]
            )
            if method in {"raw_mlp", "cnn", "cvnn"}
            else "medium"
            for method in methods
        }
    else:
        matched_capacities = None
        model_capacities = {method: capacity for method in methods}
    delay_k_proxy = delay_k_proxy_features(observable) if "delay_k_proxy" in methods else None
    sparse_multipath = (
        _sparse_multipath_features(observable) if "sparse_multipath" in methods else None
    )
    phase_randomized = _phase_randomized_real_imag_cfr(
        observable, seed=int(phase_randomization_seed)
    )
    layout = RunLayout.at(output)
    layout.ensure()
    output = layout.root
    config = {
        "holdout": holdout,
        "methods": list(methods),
        "seeds": list(seeds),
        "bootstrap": bootstrap,
        "include_phase": include_phase,
        "device": device,
        "capacity": capacity,
        "model_capacities": model_capacities,
        "parameter_match": matched_capacities,
        "coral_penalty": coral_penalty,
        "observation": observation,
        "pilot_spacing": pilot_spacing,
        "pilot_snr_db": pilot_snr_db,
        "observation_seed": observation_seed,
        "phase_randomization_seed": int(phase_randomization_seed),
        "label_permutation_seed": int(label_permutation_seed),
        "row_shuffle_seed": int(row_shuffle_seed),
        "feature_transform_by_method": {
            method: _MODEL_FEATURE_ROUTES.get(method, ("observed_cfr", "logmag_pdp_moments_v1"))[1]
            for method in methods
        },
        "raw_mlp_input_representation": "flattened real/imaginary CFR channels",
        "power_only_input_representation": (
            "per-link/per-subcarrier channel power in dB with a -120 dB numerical floor"
        ),
        "phase_randomized_input_representation": (
            "flattened real/imaginary CFR after independent seeded phase per sample, link, and subcarrier"
        ),
        "label_permuted_supervision": (
            "training labels only are permuted with a deterministic seed derived from the run and label-permutation seeds; "
            "validation, calibration, and test labels remain unchanged"
        ),
        "row_shuffled_training_features": (
            "training observed-CFR feature rows are deterministically permuted while their labels and all evaluation partitions remain unchanged"
        ),
        "source_only_control": (
            "one-hot source identifiers with a vocabulary fit only on training rows; unseen evaluation identifiers map to all zeros"
        ),
        "metadata_only_control": (
            "bounded training-vocabulary one-hot group/domain/source identifiers; this is an intentional metadata-leakage diagnostic, not a receiver feature"
        ),
        "oracle_label_leak_control": (
            "true binary target is intentionally supplied as a model input for evaluator-ceiling/falsification only; never deployment-eligible"
        ),
        "cnn_input_representation": "per-link real/imaginary channels over ordered subcarriers",
        "cvnn_input_representation": (
            "complex convolution, phase-equivariant encoding, magnitude-invariant classifier"
        ),
        "split_mode": split_provenance["mode"],
        "benchmark_test_cohort": (
            None if benchmark_provenance is None else benchmark_provenance["selected_test_cohort"]
        ),
        "threshold_selection_policy": _VALIDATION_THRESHOLD_POLICY,
        "prior_shift": resolved_prior_shift,
        "low_confidence_margin": low_confidence_margin,
        "dataset_materialization": (
            "content_addressed_build_node_v1"
            if dataset_build_node is not None
            else "direct_ordered_concatenation"
        ),
    }
    dependency_versions = versions()
    config_sha256 = canonical_hash(config)
    input_records = []
    for path in paths:
        resolved_path = str(Path(path).expanduser().resolve())
        digest = sha256_file(path)
        expected_digest = input_snapshot_digests.get(resolved_path)
        if expected_digest is not None and digest != expected_digest:
            raise ValueError(f"benchmark dataset changed after its verified snapshot: {path}")
        input_records.append({"path": str(path), "sha256": digest})
    package_source_digest = package_source_sha256()
    manifest = run_manifest(
        output.name,
        config,
        legacy_schema="loslab.run.v1",
        config_sha256=config_sha256,
        inputs=input_records,
        dependencies=dependency_versions,
        source_revision=dependency_versions.get("git_sha"),
        package_source_sha256=package_source_digest,
        n_samples=len(data.y),
        source_counts={str(s): int(np.sum(data.source == s)) for s in np.unique(data.source)},
        domain_counts={str(s): int(np.sum(data.domain == s)) for s in np.unique(data.domain)},
        label_semantics=resolved_label_semantics,
        label_semantics_by_source=_label_semantics_by_source(data),
        observation_semantics={
            "ideal_cfr": "ideal channel, diagnostic upper bound",
            "comb_ls": "generic noisy comb pilot estimate, NOT 3GPP DMRS",
            "precomputed": _precomputed_observation_semantics(data),
        }[observation],
        study_status=(
            "DEBUG_ONLY" if "synthetic_debug" in data.source else "SIMULATION_ONLY_UNVALIDATED"
        ),
        split_policy=split_provenance,
        dataset_build_node=dataset_build_node,
    )
    replay_argv = list(command_argv) if command_argv is not None else list(sys.argv)
    replay_command_source = command_source or (
        "caller_supplied" if command_argv is not None else "process_argv_snapshot"
    )
    manifest["run_replay"] = build_run_replay_provenance(
        created_utc=manifest["created_utc"],
        command_argv=replay_argv,
        command_source=replay_command_source,
        source_revision=str(dependency_versions.get("git_sha", "unavailable")),
        package_source_sha256=package_source_digest,
        config_sha256=config_sha256,
        inputs=input_records,
        backend_identity={
            "requested_device": device,
            "platform": platform.platform(),
            "dependencies": dependency_versions,
        },
        rng_streams=_run_replay_rng_streams(config),
        attempt_index=attempt_index,
        retry_of_manifest_sha256=retry_of_manifest_sha256,
    )
    atomic_json(layout.config / "resolved.json", resolve_config(config))
    atomic_json(output / "manifest.json", manifest)
    results: list[dict[str, Any]] = []
    for seed in seeds:
        if locked_indices is not None:
            train, val, test = locked_indices[:3]
            if benchmark_test_indices is not None:
                test = benchmark_test_indices
            calibration = locked_indices[3]
        else:
            train, val, test = (
                holdout_domain(data, holdout, seed=seed)
                if holdout
                else in_domain_split(data, seed=seed)
            )
            calibration = np.empty(0, dtype=np.int64)
        if resolved_prior_shift is not None:
            if len(calibration) == 0:
                train_pool = train
                fit_rows, calibration_rows = group_split(
                    data.y[train_pool],
                    data.group[train_pool],
                    test_fraction=0.2,
                    seed=seed + 907,
                )
                train = train_pool[fit_rows]
                calibration = train_pool[calibration_rows]
        else:
            calibration = np.empty(0, dtype=np.int64)
        assert not (set(data.group[train]) & set(data.group[val]))
        assert not (set(data.group[test]) & (set(data.group[train]) | set(data.group[val])))
        assert not (set(data.group[calibration]) & (set(data.group[train]) | set(data.group[val])))
        assert not (set(data.group[calibration]) & set(data.group[test]))
        partition_support = {
            name: _declared_partition_support(data, indices)
            for name, indices in (
                ("train", train),
                ("validation", val),
                ("calibration", calibration),
                ("test", test),
            )
        }
        partition_support_sha256 = canonical_hash(partition_support)
        for method in methods:
            method_input = (
                _identity_complex_cfr(observable)
                if method in ("cnn", "cvnn")
                else raw_real_imag
                if method in ("raw_mlp", "label_permuted", "row_shuffled")
                else power_only
                if method == "power_only"
                else phase_randomized
                if method == "phase_randomized"
                else delay_k_proxy
                if method == "delay_k_proxy"
                else sparse_multipath
                if method == "sparse_multipath"
                else _metadata_only_features(
                    group_values=data.group,
                    domain_values=data.domain,
                    source_values=data.source,
                    train_indices=train,
                )[0]
                if method == "metadata_only"
                else _single_metadata_features(
                    data.group if method == "group_only" else data.domain,
                    field="group" if method == "group_only" else "domain",
                    train_indices=train,
                )[0]
                if method in {"group_only", "domain_only"}
                else _oracle_label_leak_features(data.y)[0]
                if method == "oracle_label_leak"
                else x
            )
            if not isinstance(method_input, np.ndarray):
                raise RuntimeError(f"model input is unavailable for method {method}")
            if method == "label_permuted":
                training_targets, _ = _permute_training_labels(
                    data.y[train], run_seed=int(seed), base_seed=int(label_permutation_seed)
                )
            else:
                training_targets = data.y[train]
            if method == "row_shuffled":
                shuffled_features, _ = _permute_training_feature_rows(
                    method_input[train], run_seed=int(seed), base_seed=int(row_shuffle_seed)
                )
                method_input = np.array(method_input, copy=True)
                method_input[train] = shuffled_features
            if method == "source_only":
                method_input, _ = _source_only_features(data.source, train_indices=train)
            uses_torch = method in ("cnn", "cvnn", "coral", "coral0")
            _sync_torch_device(device, uses_torch)
            if uses_torch and device.startswith("cuda"):
                import torch

                torch.cuda.reset_peak_memory_stats(device)
            fit_started = perf_counter()
            if method in (
                "rf",
                "histgb",
                "xgb",
                "mlp",
                "raw_mlp",
                "label_permuted",
                "row_shuffled",
                "source_only",
                "metadata_only",
                "group_only",
                "domain_only",
                "power_only",
                "phase_randomized",
                "delay_k_proxy",
                "sparse_multipath",
                "oracle_label_leak",
                "observable_features",
                "logreg",
            ):
                clf = fit_sklearn(
                    method_input[train],
                    training_targets,
                    seed=seed,
                    method="mlp"
                    if method == "raw_mlp"
                    else "mlp"
                    if method == "label_permuted"
                    else "mlp"
                    if method == "row_shuffled"
                    else "logreg"
                    if method
                    in (
                        "power_only",
                        "source_only",
                        "metadata_only",
                        "group_only",
                        "domain_only",
                        "delay_k_proxy",
                        "sparse_multipath",
                        "observable_features",
                    )
                    else "logreg"
                    if method == "oracle_label_leak"
                    else "mlp"
                    if method == "phase_randomized"
                    else method,
                    capacity=model_capacities[method],
                )
                extra: dict[str, Any] = {}
            elif method == "cnn":
                clf, extra = fit_real_cnn(
                    observable[train],
                    training_targets,
                    observable[val],
                    data.y[val],
                    seed=seed,
                    device=device,
                    capacity=model_capacities[method],
                )
            elif method == "cvnn":
                clf, extra = fit_complex_cnn(
                    observable[train],
                    data.y[train],
                    observable[val],
                    data.y[val],
                    seed=seed,
                    device=device,
                    capacity=model_capacities[method],
                )
            elif method in ("coral", "coral0"):
                clf, extra = fit_coral_mlp(
                    x[train],
                    data.y[train],
                    data.domain[train],
                    x[val],
                    data.y[val],
                    seed=seed,
                    penalty=coral_penalty if method == "coral" else 0.0,
                    device=device,
                    capacity=model_capacities[method],
                )
            else:
                raise ValueError(f"Unknown method {method}")
            _sync_torch_device(device, uses_torch)
            training_seconds = perf_counter() - fit_started
            parameter_count = _learned_parameter_count(clf, extra)
            calibration_scores_raw = np.empty(0, dtype=np.float64)
            calibration_scores_calibrated = np.empty(0, dtype=np.float64)
            calibrator_report = None
            validation_started = perf_counter()
            val_scores_raw = clf.predict_proba(method_input[val])[:, 1]
            _sync_torch_device(device, uses_torch)
            validation_prediction_seconds = perf_counter() - validation_started
            prior_shift_report = None
            if resolved_prior_shift is None:
                val_scores = val_scores_raw
                val_model_calibrated = val_scores_raw
            else:
                calibration_scores_raw = clf.predict_proba(method_input[calibration])[:, 1]
                calibration_scores_calibrated, calibrator_report = (
                    fit_group_balanced_platt_calibrator(
                        calibration_scores_raw,
                        data.y[calibration],
                        data.group[calibration],
                    )
                )
                _validate_evaluation_payload(
                    calibrator_report, "group-platt-calibrator.v1.schema.json"
                )
                assert calibrator_report is not None
                val_calibrated = apply_platt_calibrator(val_scores_raw, calibrator_report)
                val_model_calibrated = val_calibrated
                val_scores, prior_shift_report = adjust_probabilities_for_prior_shift(
                    val_calibrated,
                    data.y[calibration],
                    data.group[calibration],
                    resolved_prior_shift,
                )
                _validate_evaluation_payload(
                    prior_shift_report, "prior-shift-adjustment.v1.schema.json"
                )
            threshold_selection = _validation_threshold_selection_record(
                data.y[val],
                val_scores,
                data.group[val],
                val,
            )
            threshold = threshold_selection["threshold"]
            if prior_shift_report is None:
                model_output_threshold = threshold
            else:
                assert resolved_prior_shift is not None
                assert calibrator_report is not None
                model_output_threshold = float(
                    invert_platt_calibrator(
                        adjust_probabilities_between_priors(
                            np.asarray([threshold]),
                            source_positive_prevalence=(
                                resolved_prior_shift["target_positive_prevalence"]
                            ),
                            target_positive_prevalence=(
                                prior_shift_report["calibration_group_weighted_positive_prevalence"]
                            ),
                        ),
                        calibrator_report,
                    )[0]
                )
            checkpoint = (
                layout.models / f"{method}_seed{seed}.pt"
                if method.startswith("coral") or method in ("cnn", "cvnn")
                else layout.models / f"{method}_seed{seed}.joblib"
            )
            if method.startswith("coral"):
                clf.save(checkpoint, threshold=model_output_threshold, include_phase=include_phase)
            elif method == "cnn":
                clf.save(checkpoint, threshold=model_output_threshold, include_phase=True)
            elif method == "cvnn":
                clf.save(checkpoint, threshold=model_output_threshold)
            else:
                import joblib

                joblib.dump(clf, checkpoint)
            split_contract = {
                "mode": split_provenance["mode"],
                "registry_sha256": split_provenance.get("registry_sha256"),
                "train_rows": int(len(train)),
                "validation_rows": int(len(val)),
                "test_rows": int(len(test)),
            }
            partition_indices = {"train": train, "validation": val, "test": test}
            indices_payload = {
                "train_idx": train.tolist(),
                "validation_idx": val.tolist(),
                "test_idx": test.tolist(),
            }
            if len(calibration):
                partition_indices["calibration"] = calibration
                split_contract["calibration_rows"] = int(len(calibration))
                indices_payload["calibration_idx"] = calibration.tolist()
            split_contract["indices_sha256"] = canonical_hash(indices_payload)
            model_input_contract = build_model_input_artifact_contract(
                method=method,
                method_input=method_input,
                include_phase=include_phase,
                config_sha256=canonical_hash(config),
                source_revision=dependency_versions.get("git_sha"),
                input_file_digests=manifest["inputs"],
                split=split_contract,
                model_file_sha256=_stable_single_link_sha256(checkpoint),
                partition_indices=partition_indices,
                target_values=data.y if method in {"label_permuted", "oracle_label_leak"} else None,
                label_permutation_seed=(
                    int(label_permutation_seed) if method == "label_permuted" else None
                ),
                label_permutation_run_seed=(int(seed) if method == "label_permuted" else None),
                source_method_input=(raw_real_imag if method == "row_shuffled" else None),
                feature_permutation_seed=(
                    int(row_shuffle_seed) if method == "row_shuffled" else None
                ),
                feature_permutation_run_seed=(int(seed) if method == "row_shuffled" else None),
                source_values=(data.source if method == "source_only" else None),
                metadata_values=(
                    {"group": data.group, "domain": data.domain, "source": data.source}
                    if method == "metadata_only"
                    else {"group": data.group}
                    if method == "group_only"
                    else {"domain": data.domain}
                    if method == "domain_only"
                    else None
                ),
            )
            model_info_path = layout.models / f"{method}_seed{seed}_modelinfo.json"
            atomic_json(
                model_info_path,
                {
                    "format": (
                        "loslab.sklearn.v1"
                        if method
                        in {
                            "rf",
                            "histgb",
                            "xgb",
                            "mlp",
                            "raw_mlp",
                            "label_permuted",
                            "row_shuffled",
                            "source_only",
                            "metadata_only",
                            "group_only",
                            "domain_only",
                            "oracle_label_leak",
                            "power_only",
                            "phase_randomized",
                            "delay_k_proxy",
                            "sparse_multipath",
                            "logreg",
                        }
                        else "loslab.model_input_artifact.v1"
                    ),
                    "threshold": model_output_threshold,
                    "threshold_probability_space": "raw_model_output",
                    "calibrated_threshold": threshold,
                    "probability_calibration": calibrator_report,
                    "prior_shift": prior_shift_report,
                    "include_phase": include_phase,
                    "method": method,
                    "model_input_artifact_contract": model_input_contract,
                    "benchmark_cohort": benchmark_provenance,
                },
            )
            model_input_contract_verification = verify_model_input_artifact_contract(
                model_info_path,
                checkpoint,
                method_input=method_input,
                partition_indices=partition_indices,
                target_values=data.y if method in {"label_permuted", "oracle_label_leak"} else None,
                source_method_input=raw_real_imag if method == "row_shuffled" else None,
                source_values=data.source if method == "source_only" else None,
                metadata_values=(
                    {"group": data.group, "domain": data.domain, "source": data.source}
                    if method == "metadata_only"
                    else {"group": data.group}
                    if method == "group_only"
                    else {"domain": data.domain}
                    if method == "domain_only"
                    else None
                ),
                benchmark_cohort=benchmark_provenance,
            )
            test_prediction_started = perf_counter()
            test_scores_raw = clf.predict_proba(method_input[test])[:, 1]
            _sync_torch_device(device, uses_torch)
            test_prediction_seconds = perf_counter() - test_prediction_started
            test_calibrated = test_scores_raw
            if resolved_prior_shift is None:
                prob = test_scores_raw
            else:
                assert calibrator_report is not None
                test_calibrated = apply_platt_calibrator(test_scores_raw, calibrator_report)
                prob, test_prior_shift_report = adjust_probabilities_for_prior_shift(
                    test_calibrated,
                    data.y[calibration],
                    data.group[calibration],
                    resolved_prior_shift,
                )
                if test_prior_shift_report != prior_shift_report:
                    raise RuntimeError("train-only prior-shift report changed across partitions")
            metrics = binary_metrics(data.y[test], prob, threshold=threshold)
            source_families = np.asarray(data.source, dtype=str)
            test_families = source_families[test]
            low_confidence_mask = np.zeros(len(test), dtype=bool)
            low_confidence_report = None
            if low_confidence_margin is not None:
                low_confidence_mask, low_confidence_report = build_low_confidence_audit(
                    prob,
                    data.group[test],
                    test_families,
                    data.domain[test],
                    threshold=threshold,
                    margin=float(low_confidence_margin),
                )
                _validate_evaluation_payload(
                    low_confidence_report, "low-confidence-audit.v1.schema.json"
                )
            expected_test_strata = tuple(
                ("test", family_id) for family_id in sorted(set(source_families.tolist()))
            )
            class_support = audit_class_support(
                labels=data.y[test],
                independent_group_ids=data.group[test],
                split_ids=np.full(len(test), "test", dtype=object),
                family_ids=test_families,
                expected_strata=expected_test_strata,
                classes=(0, 1),
            )
            class_support_payload = {
                "schema": "loslab.class_support.v1",
                "independence_identity_source": "ChannelDataset.group",
                "cells": [
                    {
                        "split_id": cell.split_id,
                        "family_id": cell.family_id,
                        "class_label": cell.class_label,
                        "record_count": cell.record_count,
                        "independent_group_count": cell.independent_group_count,
                        "status": cell.status,
                    }
                    for cell in class_support.cells
                ],
            }
            import jsonschema

            class_support_schema_path = (
                Path(__file__).resolve().parents[2]
                / "evaluation/schema/class_support.v1.schema.json"
            )
            class_support_schema = json.loads(class_support_schema_path.read_text(encoding="utf-8"))
            jsonschema.Draft202012Validator.check_schema(class_support_schema)
            class_support_errors = list(
                jsonschema.Draft202012Validator(class_support_schema).iter_errors(
                    class_support_payload
                )
            )
            if class_support_errors:
                raise ValueError(
                    "class support report violates schema: "
                    + "; ".join(error.message for error in class_support_errors[:10])
                )
            ci = block_bootstrap_auc(
                data.y[test], prob, data.group[test], seed=seed, draws=bootstrap
            )
            result = {
                "seed": int(seed),
                "method": method,
                "capacity": model_capacities[method],
                "input_representation": (
                    "complex observation split into real/imaginary channels"
                    if method == "cnn"
                    else "complex convolution with global-phase-invariant magnitude head"
                    if method == "cvnn"
                    else "flattened raw real/imaginary channel observation"
                    if method == "raw_mlp"
                    else "per-link/per-subcarrier channel power in dB"
                    if method == "power_only"
                    else "phase-randomized flattened real/imaginary CFR"
                    if method == "phase_randomized"
                    else "normalized delay spread and dominant-to-remainder PDP proxy"
                    if method == "delay_k_proxy"
                    else "bounded normalized-delay OMP/BIC fit features from receiver-visible CFR"
                    if method == "sparse_multipath"
                    else "raw flattened real/imaginary CFR trained on deterministically permuted training labels"
                    if method == "label_permuted"
                    else "row-shuffled flattened real/imaginary CFR"
                    if method == "row_shuffled"
                    else "training-vocabulary one-hot source identifiers"
                    if method == "source_only"
                    else "training-vocabulary one-hot group/domain/source metadata leakage control"
                    if method == "metadata_only"
                    else "engineered observable features"
                ),
                "metrics": metrics,
                "prior_shift": prior_shift_report,
                "probability_calibration": calibrator_report,
                "low_confidence_audit": low_confidence_report,
                "model_output_threshold": model_output_threshold,
                "class_support": class_support_payload,
                "confidence": ci,
                "threshold_selection": threshold_selection,
                "split": {
                    "mode": split_provenance["mode"],
                    "registry_sha256": split_provenance.get("registry_sha256"),
                    "benchmark_cohort": benchmark_provenance,
                    "declared_partition_support": partition_support,
                    "declared_partition_support_sha256": partition_support_sha256,
                    "partition_support_sha256": split_provenance.get("partition_support_sha256"),
                    "physical_independence_status": (
                        split_provenance.get("partition_support", {}).get(
                            "physical_independence_status"
                        )
                        if isinstance(split_provenance.get("partition_support"), dict)
                        else "not_applicable_unlocked_pilot"
                    ),
                    "indices_sha256": canonical_hash(
                        {
                            "train_idx": train.tolist(),
                            "validation_idx": val.tolist(),
                            "test_idx": test.tolist(),
                            **(
                                {"calibration_idx": calibration.tolist()}
                                if len(calibration)
                                else {}
                            ),
                        }
                    ),
                    "train_size": len(train),
                    "calibration_size": len(calibration),
                    "val_size": len(val),
                    "test_size": len(test),
                    "train_groups": len(set(data.group[train])),
                    "calibration_groups": len(set(data.group[calibration])),
                    "validation_groups": len(set(data.group[val])),
                    "test_groups": len(set(data.group[test])),
                    "train_domains": sorted(set(data.domain[train].tolist())),
                    "test_domains": sorted(set(data.domain[test].tolist())),
                },
                "training": extra,
                "compute": {
                    "training_seconds": float(training_seconds),
                    "validation_prediction_seconds": float(validation_prediction_seconds),
                    "test_prediction_seconds": float(test_prediction_seconds),
                    "validation_prediction_samples": int(len(val)),
                    "test_prediction_samples": int(len(test)),
                    "learned_parameter_count": parameter_count,
                    "parameter_match": (
                        {
                            "target_parameters": int(matched_capacities["target_parameters"]),
                            "actual_parameters": parameter_count,
                            "relative_error": (
                                abs(
                                    int(parameter_count)
                                    - int(matched_capacities["target_parameters"])
                                )
                                / int(matched_capacities["target_parameters"])
                                if parameter_count is not None
                                else None
                            ),
                            "input_shape": [int(observable.shape[1]), int(observable.shape[2])],
                        }
                        if matched_capacities is not None and method in {"raw_mlp", "cnn", "cvnn"}
                        else None
                    ),
                    "peak_cuda_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if uses_torch and device.startswith("cuda")
                        else None
                    ),
                },
                "checkpoint_path": str(checkpoint),
                "model_sha256": model_input_contract["model_sha256"],
                "model_input_artifact_contract_sha256": canonical_hash(model_input_contract),
                "model_input_artifact_contract_status": model_input_contract_verification["status"],
                "model_input_feature_matrix_verified": model_input_contract_verification[
                    "feature_matrix_verified"
                ],
                "model_input_artifact_contract_verification": model_input_contract_verification,
            }
            results.append(result)
            # Keep split indices and probabilities for independently reproducing every metric.
            np.savez_compressed(
                layout.predictions / f"{method}_seed{seed}.npz",
                train_idx=train,
                calibration_idx=calibration,
                val_idx=val,
                test_idx=test,
                y_test=data.y[test],
                y_val=data.y[val],
                test_group=np.asarray(data.group[test], dtype=str),
                test_domain=np.asarray(data.domain[test], dtype=str),
                test_source=test_families,
                p_test=prob,
                p_val=val_scores,
                p_test_raw=test_scores_raw,
                p_val_raw=val_scores_raw,
                p_test_calibrated=test_calibrated,
                p_val_calibrated=val_model_calibrated,
                p_calibration_raw=calibration_scores_raw,
                p_calibration_calibrated=calibration_scores_calibrated,
                test_low_confidence=low_confidence_mask,
                threshold=np.float64(threshold),
            )
        atomic_json(output / "results.json", results)
        atomic_json(layout.evaluation / "results.json", results)
    aggregated = {}
    for method in methods:
        all_scores = [row["metrics"]["auroc"] for row in results if row["method"] == method]
        seq = [score for score in all_scores if score is not None]
        aggregated[method] = {
            "seeds": len(all_scores),
            "auroc_defined_seeds": len(seq),
            "mean_test_auroc": float(np.mean(seq)) if seq else None,
            "std_test_auroc": float(np.std(seq, ddof=1)) if len(seq) > 1 else None,
        }
    comparisons = {}
    method_pairs = (
        ("rf", "mlp"),
        ("histgb", "mlp"),
        ("raw_mlp", "mlp"),
        ("raw_mlp", "rf"),
        ("raw_mlp", "histgb"),
        ("power_only", "mlp"),
        ("delay_k_proxy", "mlp"),
        ("delay_k_proxy", "rf"),
        ("delay_k_proxy", "raw_mlp"),
        ("sparse_multipath", "mlp"),
        ("sparse_multipath", "rf"),
        ("sparse_multipath", "raw_mlp"),
        ("power_only", "rf"),
        ("power_only", "raw_mlp"),
        ("raw_mlp", "phase_randomized"),
        ("raw_mlp", "label_permuted"),
        ("raw_mlp", "row_shuffled"),
        ("raw_mlp", "source_only"),
        ("raw_mlp", "metadata_only"),
        ("raw_mlp", "group_only"),
        ("raw_mlp", "domain_only"),
        ("oracle_label_leak", "raw_mlp"),
        ("oracle_label_leak", "mlp"),
        ("cnn", "raw_mlp"),
        ("cvnn", "raw_mlp"),
        ("xgb", "mlp"),
        ("xgb", "rf"),
        ("cnn", "mlp"),
        ("cvnn", "mlp"),
        ("cvnn", "cnn"),
        ("rf", "cvnn"),
        ("histgb", "cvnn"),
        ("rf", "cnn"),
        ("histgb", "cnn"),
        ("logreg", "mlp"),
        ("mlp", "coral"),
        ("coral0", "coral"),
        ("mlp", "coral0"),
    )
    for control, treatment in method_pairs:
        if control in methods and treatment in methods:
            comparisons[f"{treatment}_versus_{control}"] = paired_seed_delta(
                results, control, treatment, draws=5000
            )
    summary = {
        "schema": "loslab.summary.v1",
        "evidence_grade": manifest["study_status"],
        "paired_method_comparisons": comparisons,
        "aggregate_across_fixed_splits": aggregated,
        "warning": "Do not report synthetic-debug results or profile-proxy labels as real LOS measurements.",
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(layout.evaluation / "summary.json", summary)
    complete_manifest = mark_complete(manifest)
    atomic_json(layout.manifest, complete_manifest)
    atomic_json(
        layout.complete,
        {
            "run_id": output.name,
            "manifest_schema": complete_manifest["schema"],
            "config_identity": complete_manifest["config_identity"],
        },
    )
    return summary


def run_experiment(*args: Any, **kwargs: Any) -> dict:
    """Run an experiment and persist a failed terminal receipt when one exists.

    Failures before the common runner has written its initial manifest have no
    complete replay contract to transition. Process interruption likewise
    leaves the durable manifest in ``running`` state for explicit recovery.
    """
    output = args[1] if len(args) >= 2 else kwargs.get("output")
    existing_manifest_before = False
    if output is not None:
        output_before = Path(output).expanduser()
        if not output_before.is_symlink():
            existing_manifest_before = (output_before / "manifest.json").exists()
    try:
        return _run_experiment_impl(*args, **kwargs)
    except Exception as error:
        if output is not None:
            output_candidate = Path(output).expanduser()
            if output_candidate.is_symlink() or existing_manifest_before:
                raise
            manifest_path = output_candidate.absolute() / "manifest.json"
            manifest_written = False
            if manifest_path.is_file() and not manifest_path.is_symlink():
                try:
                    persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
                    replay = persisted.get("run_replay")
                    if isinstance(replay, dict) and replay.get("state") == "running":
                        failed = mark_failed(
                            persisted,
                            error_type=type(error).__name__,
                            error_message=str(error),
                            stage="experiment",
                        )
                        atomic_json(manifest_path, failed)
                        manifest_written = True
                except Exception as provenance_error:
                    raise RuntimeError(
                        "experiment failed and failure replay provenance could not be recorded: "
                        f"{provenance_error}"
                    ) from error
            benchmark_requested = any(
                kwargs.get(name) is not None
                for name in (
                    "benchmark_manifest_path",
                    "benchmark_declaration_path",
                    "benchmark_test_cohort",
                )
            )
            if benchmark_requested and not manifest_written and not manifest_path.exists():
                _write_benchmark_preflight_failure_receipt(
                    output,
                    error,
                    selected_test_cohort=kwargs.get("benchmark_test_cohort"),
                )
        raise
