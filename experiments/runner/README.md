# experiments.runner

Experiment runner: feature construction, label and validation logic, model-input contracts, and the run entry point.

## Purpose

This package is one module split into concept modules. It keeps the public names of the original module, so callers import from `experiments.runner` unchanged.

## Modules

| Module | Contents |
| --- | --- |
| `contracts.py` | Verification and build of model-input and validation-threshold contracts |
| `experiment_runner.py` | Run entry point for a configured experiment |
| `features.py` | Feature construction and channel-representation helpers for the experiment runner |
| `validation.py` | Label routing, permutation controls, and validation helpers for the experiment runner |

## Public API (43 names)

Re-exported from `__init__.py` through `__all__`:

`_CHANNEL_DATASET_BUILD_SCHEMA`, `_MODEL_FEATURE_ROUTES`, `_MODEL_METHODS`, `_VALIDATION_THRESHOLD_POLICY`, `_array_sha256`, `_flatten_real_imag_cfr`, `_identity_complex_cfr`, `_kish_effective_group_count`, `_label_semantics_by_source`, `_learned_parameter_count`, `_log_power_per_link_subcarrier`, `_normalized_label_semantics`, `_precomputed_observation_semantics`, `_run_replay_rng_streams`, `_sparse_multipath_features`, `_stable_input_bytes`, `_stable_single_link_sha256`, `_sync_torch_device`, `_unsupported_benchmark_test_cohorts`, `_validate_evaluation_payload`, `_validate_json_schema`, `_validate_npz_snapshot`, `_declared_partition_support`, `_locked_split_indices`, `_materialize_channel_dataset_build_node`, `_metadata_only_features`, `_oracle_label_leak_features`, `_permute_training_feature_rows`, `_permute_training_labels`, `_phase_randomized_real_imag_cfr`, `_resolve_benchmark_test_cohort`, `_resolved_label_semantics`, `_single_metadata_features`, `_source_only_features`, `_validation_threshold_selection_record`, `_write_benchmark_preflight_failure_receipt`, `_feature_transform_implementation_sha256`, `build_model_input_artifact_contract`, `verify_declared_partition_support`, `verify_model_input_artifact_contract`, `verify_validation_threshold_selection_record`, `_run_experiment_impl`, `run_experiment`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.

