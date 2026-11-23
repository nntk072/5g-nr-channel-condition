# experiments.audit_ordered_impairments_result

Audit of ordered-impairment result files, split by concept.

## Purpose

The audit is organised by concept: constants, hashing, preregistration checks, evidence-file reads, numeric references, waveform recomputation, replay, and the two audit entry points. Callers import from `experiments.audit_ordered_impairments_result`; the names are re-exported from `__init__.py`.

## Modules

| Module | Contents |
| --- | --- |
| `__init__.py` | Public re-exports |
| `__main__.py` | Command-line entry point |
| `constants.py` | Schema identifiers, byte budgets, frozen lookup tables |
| `hashing.py` | Canonical JSON and SHA-256 helpers |
| `preregistration.py` | Preregistration order, reviewer signature, execution-context checks |
| `evidence_files.py` | Bounded evidence reads and preflight checks |
| `references.py` | Closed-form references and finite-number helpers |
| `waveform.py` | Waveform diagnostic recomputation and cluster bootstrap |
| `replay.py` | Deterministic replay of a recorded experiment |
| `independent.py` | Independent-channel and SNR-curve audit checks |
| `file.py` | Audit of one result file and the command-line interface |

## Public API (40 names)

Re-exported from `__init__.py` through `__all__`:

`REPOSITORY_ROOT`, `_BLOCK_BOUND_REPORT_SCHEMAS`, `_CFO_CHANNEL_TAIL_SCHEMAS`, `_CUDA_CURVE_BASE_VERSIONS`, `_CUDA_DEVICE_REPORT_SCHEMAS`, `_EXACT_SEED_ROSTER_CURVE_SCHEMAS`, `_SNR_ACCEPTANCE_SIGNATURE_NAMESPACE`, `_HEX`, `_MAX_RESULT_BYTES`, `_MAX_WAVEFORM_EVIDENCE_BYTES`, `_MAX_WAVEFORM_EVIDENCE_UNCOMPRESSED_BYTES`, `_PER_CHANNEL_WAVEFORM_DIAGNOSTIC_SCHEMAS`, `_RESULT_SCHEMA`, `_canonical_json_bytes`, `_cluster_bootstrap_interval`, `_curve_report_schema_status`, `_estimator_source_checks`, `_declared_preregistration_order_matches`, `_execution_context_matches`, `_inference_seed_roster_consistent`, `_iq_imbalance_reference`, `_is_finite_nonnegative`, `_is_finite_number`, `_is_sha256`, `_path_sum_reference`, `_positive_integer_delay_reference`, `_preflight_waveform_archive`, `_read_bounded_evidence_file`, `_recompute_waveform_diagnostics`, `_reject_constant`, `_replay_experiment`, `_run_receiver_probe`, `_same_float`, `_sha256`, `_unique_object`, `_verify_preregistration_signature`, `_audit_independent_snr_curve`, `_audit_result`, `audit_result_file`, `main`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.
- Files that retained evidence records by path are listed in `docs/reference/digest-rebindings.json`.

