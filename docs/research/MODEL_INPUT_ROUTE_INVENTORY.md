# Model input route inventory — working audit

This is a source-derived inventory for `PKG08-017`, not a leakage-clearance or
qualification claim. Active means referenced by current package or command-line
code outside `archive/` and `notebooks/legacy_2025/`. Synthetic and metadata
controls are called out so they are not mistaken for receiver inputs.

| Route | Producer and consumer | Input transform / supervision | Artifact binding and state |
|---|---|---|---|
| Canonical classification runner | `ChannelDataset.model_inputs()` → `experiments/runner.py` → `learning.fit_*` / `fit_sklearn` | Method-specific routes in `_MODEL_FEATURE_ROUTES`: CFR grid, flattened real/imaginary CFR, power-only, phase-randomized, engineered observable features, train-only group/domain/source vocabularies, and explicit label-leak control. Labels remain separate except in the named falsification control. | Versioned model-input contract binds exact feature and partition matrices, split indices, configuration, source files, and model bytes; replay validation runs before publication. Tests: `tests/test_experiments/test_runner.py`. Metadata/oracle controls are diagnostics, not deployable inputs. |
| Profile holdout | `datasets/channel.py` → `experiments/profile_holdout.py` → sklearn estimators; CLI in `scripts/run_profile_holdout_baselines.py` | `normalized_cfr_features` transforms audited CFR; profile metadata is evaluated in a separate proxy-only control. | Contract binds transform implementation, full feature matrix and fold partitions. Score report is emitted; no fitted checkpoint is retained by this route. Tests: `tests/test_experiments/test_profile_holdout.py`. |
| Analytical two-path decoder | `generate_grouped_two_path_dataset()` → `TwoPathModelInputs` → fit/predict/profile-likelihood in `experiments/observable_multipath.py`; CLI in `scripts/run_observable_multipath_experiment.py` | Receiver inputs are observed CFR, pilot mask, frequencies and known noise variance; relative delay and complex gains remain targets/oracle. Whole groups define train/validation/calibration/test. | Decoder NPZ checkpoints now have schema-versioned sidecars binding exact input, target, split arrays, source dataset and checkpoint bytes. Synthetic analytical pilot only. Tests: `tests/test_experiments/test_observable_multipath_inputs.py` and `test_two_path_experiment_contract.py`. |
| CAEZ magnitude forecast | CAEZ NPZ summary → `scripts/evaluate_caez_magnitude_forecast.py` → Ridge | Current and history-window summary features predict future magnitude; target rows are held separately and chronological pairs exclude unavailable history. | Report contract binds input/target/pair indices, source archive hashes, model identity and script transform digest. No licensed archive was read for software verification. Tests: `tests/test_scripts/test_audit_caez_npz.py`. |
| CAEZ cross-environment forecast | Source CAEZ NPZ → Ridge; target archive is evaluation-only in `scripts/evaluate_caez_cross_environment.py` | Source-environment current/history summaries predict future magnitude; target-environment current/history summaries are evaluation inputs. | Same versioned report contract binds both matrices, targets, row pairs, both source archives, model and transform digest. No licensed archive was read. Tests: `tests/test_scripts/test_evaluate_caez_cross_environment.py`. |
| Probability calibration | Canonical runner model scores → `evaluation/probability_calibration.py` | Calibration scores, labels and groups fit a group-balanced Platt map; it consumes model outputs, not raw channel features. | Calibration parameters and provenance are embedded in the parent result; it is not a separate channel-input producer. Tests are part of `tests/test_experiments/test_runner.py`. |
| CNN/CVNN model APIs | `learning/real_cnn.py` and `learning/complex_cnn.py`, called by canonical runner | Real/imaginary or complex CFR with train-only normalization; labels are passed separately. | Canonical runner saves these through the runner's model-input artifact contract. The standalone `RealCNNPredictor.save` and `ComplexCNNPredictor.save` APIs do not independently carry input/split contracts; direct external callers are not shown in this repository and remain an uncovered interface boundary pending explicit support/deprecation disposition. |
| Historical thesis implementations | `archive/legacy_2025/` and `notebooks/legacy_2025/` | Superseded scripts, notebooks, and prototype models. | Archive READMEs state these are historical and active packages must not import from the archive. This establishes repository classification, not exhaustive proof that no external consumer uses them. |

## Remaining audit work

- Resolve whether standalone CNN/CVNN save APIs are supported interfaces; if so,
  bind their exact inputs, targets, split and source contract, otherwise make the
  historical/runner-only scope explicit at the API boundary.
- Finish transform and artifact inventory for model-like producers not found by
  source-level fit/save searches, including scripts that call library wrappers.
- Reconcile this inventory against archived and generated entry points, retain
  the exact source search/verifier, and have the independent package crosswalk
  confirm complete coverage.

Until those items close, `PKG08-017` remains partial. No scientific acceptance,
production RT generation, or deployment claim follows from these software
contracts.
