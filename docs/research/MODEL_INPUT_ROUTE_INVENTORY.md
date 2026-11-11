# Model input route inventory — working audit

This is a source-derived inventory for `PKG08-017`, not a leakage-clearance or
qualification claim. Active means referenced by current package or command-line
code outside `archive/` and `notebooks/legacy_2025/`. Synthetic and metadata
controls are called out so they are not mistaken for receiver inputs.

| Route | Producer and consumer | Input transform / supervision | Artifact binding and state |
|---|---|---|---|
| Canonical classification runner | `ChannelDataset.model_inputs()` → `experiments/runner.py` → `learning.fit_*` / `fit_sklearn` | Method-specific routes in `_MODEL_FEATURE_ROUTES`: CFR grid, flattened real/imaginary CFR, power-only, phase-randomized, engineered observable features, train-only group/domain/source vocabularies, and explicit label-leak control. Labels remain separate except in the named falsification control. | Versioned model-input contract binds exact feature and partition matrices, split indices, configuration, source files, and model bytes; replay validation runs before publication. Tests: `tests/test_experiments/test_runner.py`. Metadata/oracle controls are diagnostics, not deployable inputs. |
| Profile holdout | `datasets/channel.py` → `experiments/profile_holdout.py` → sklearn estimators; CLI in `scripts/run_profile_holdout_baselines.py` | `normalized_cfr_features` transforms audited CFR; profile metadata is evaluated in a separate proxy-only control. | Contract binds transform implementation, full feature matrix and fold partitions. Score report is emitted; no fitted checkpoint is retained by this route. Tests: `tests/test_experiments/test_profile_holdout.py`. |
| Analytical two-path decoder | `generate_grouped_two_path_dataset()` → `TwoPathModelInputs` → fit/predict/profile-likelihood in `experiments/observable_multipath.py`; CLI in `scripts/run_observable_multipath_experiment.py` | Receiver inputs are observed CFR, pilot mask, frequencies and known noise variance; relative delay and complex gains remain targets/oracle. Whole groups define train/validation/calibration/test. | Decoder NPZ checkpoints have schema-versioned sidecars binding exact input, target, split arrays, source dataset and checkpoint bytes. Loader verifies schema and checkpoint digest, and can replay all bound arrays when given the source dataset; unbound legacy/raw checkpoints are marked or rejected when required. Synthetic analytical pilot only. Tests: `tests/test_experiments/test_observable_multipath_inputs.py`, `test_two_path_experiment_contract.py`, and `tests/test_models/test_observable_multipath.py`. |
| CAEZ magnitude forecast | CAEZ NPZ summary → `scripts/evaluate_caez_magnitude_forecast.py` → Ridge | Current and history-window summary features predict future magnitude; target rows are held separately and chronological pairs exclude unavailable history. | Report contract binds input/target/pair indices, source archive hashes, model identity and script transform digest. No licensed archive was read for software verification. Tests: `tests/test_scripts/test_audit_caez_npz.py`. |
| CAEZ cross-environment forecast | Source CAEZ NPZ → Ridge; target archive is evaluation-only in `scripts/evaluate_caez_cross_environment.py` | Source-environment current/history summaries predict future magnitude; target-environment current/history summaries are evaluation inputs. | Same versioned report contract binds both matrices, targets, row pairs, both source archives, model and transform digest. No licensed archive was read. Tests: `tests/test_scripts/test_evaluate_caez_cross_environment.py`. |
| Probability calibration | Canonical runner model scores → `evaluation/probability_calibration.py` | Calibration scores, labels and groups fit a group-balanced Platt map; it consumes model outputs, not raw channel features. | Calibration parameters and provenance are embedded in the parent result; it is not a separate channel-input producer. Tests are part of `tests/test_experiments/test_runner.py`. |
| Sparse classical multipath estimator | `models/multipath.py::fit_sparse_multipath`, exercised by `experiments/multipath_order_stress.py` | Only masked observed CFR, frequency coordinates and pilot mask enter fitting; simulated path count/truth is used after fitting for the explicitly oracle-aided stress metric. | Per-observation fit result, no trained checkpoint. The estimator validates its vectors and Boolean mask directly rather than using a dataset adapter. This is a classical estimator diagnostic, not a receiver-qualified model. |
| Local-MLE identifiability diagnostics | `experiments/identifiability_monte_carlo.py` and `experiments/identifiability_matched_estimators.py` | Synthetic observed CFR and frequency arrays enter optimization; the local MLE receives truth-centered initialization by design, as stated in source limitations. | Diagnostic result reports, no retained inference model. Privileged initialization makes this a local optimizer/Fisher-information check, not a deployable path estimator. |
| CNN/CVNN and CORAL checkpoint APIs | `learning/real_cnn.py`, `learning/complex_cnn.py`, and `learning/coral.py`, called by canonical runner | Real/imaginary or complex CFR, or engineered real features, with train-only normalization; labels are passed separately. | Canonical runner saves these through its exact model-input contract and replay verifier. Raw checkpoint payloads now carry `artifact_contract_status=external_input_contract_required`, preserved by their loaders; a standalone save is explicitly unbound until accompanied by a verified external input/split contract. Direct external callers are not found in the repository. |
| Historical thesis implementations | `archive/legacy_2025/` and `notebooks/legacy_2025/` | Superseded scripts, notebooks, and prototype models. | Archive READMEs state these are historical and active packages must not import from the archive. This establishes repository classification, not exhaustive proof that no external consumer uses them. |

## Remaining audit work

- Finish transform and artifact inventory for model-like producers not found by
  source-level fit/save searches, including scripts that call library wrappers.
- Inspect diagnostic estimators and prediction consumers beyond fit/save searches
  to confirm generated features and oracle-only metrics remain separated.
- Reconcile all generated command entry points and saved-artifact loaders against
  this matrix, and verify that each retained artifact reaches the canonical
  replay verifier or remains explicitly marked as requiring an external
  contract.
- Reconcile this inventory against archived and generated entry points, retain
  the exact source search/verifier, and have the independent package crosswalk
  confirm complete coverage.

Until those items close, `PKG08-017` remains partial. No scientific acceptance,
production RT generation, or deployment claim follows from these software
contracts.
