# Model input route inventory — working audit

This is a source-derived inventory for `PKG08-017`, not a leakage-clearance or
qualification claim. Active means referenced by current package or command-line
code outside `archive/` and `notebooks/legacy_2025/`. Synthetic and metadata
controls are called out so they are not mistaken for receiver inputs.

| Route | Producer and consumer | Input transform / supervision | Artifact binding and state |
|---|---|---|---|
| Canonical classification runner | `ChannelDataset.model_inputs()` → `experiments/runner.py` → `learning.fit_*` / `fit_sklearn`; entry points are `loslab experiment`, `experiments/capacity_sweep.py`, and `loslab.__main__` | Method-specific routes in `_MODEL_FEATURE_ROUTES`: CFR grid, flattened real/imaginary CFR, power-only, phase-randomized, engineered observable features, train-only group/domain/source vocabularies, and explicit label-leak control. Labels remain separate except in the named falsification control. | Versioned model-input contract binds exact feature and partition matrices, split indices, configuration, source files, and model bytes; replay validation runs before publication. Tests: `tests/test_experiments/test_runner.py`. Metadata/oracle controls are diagnostics, not deployable inputs. |
| Profile holdout | `datasets/channel.py` → `experiments/profile_holdout.py` → sklearn estimators; CLI in `scripts/run_profile_holdout_baselines.py` | `normalized_cfr_features` transforms audited CFR; profile metadata is evaluated in a separate proxy-only control. | Contract binds transform implementation, full feature matrix and fold partitions. Score report is emitted; no fitted checkpoint is retained by this route. Tests: `tests/test_experiments/test_profile_holdout.py`. |
| Analytical two-path decoder | `generate_grouped_two_path_dataset()` → `TwoPathModelInputs` → fit/predict/profile-likelihood in `experiments/observable_multipath.py`; CLI in `scripts/run_observable_multipath_experiment.py` | Receiver inputs are observed CFR, pilot mask, frequencies and known noise variance; relative delay and complex gains remain targets/oracle. Whole groups define train/validation/calibration/test. | Decoder NPZ checkpoints have schema-versioned sidecars binding exact input, target, split arrays, source dataset and checkpoint bytes. Loader verifies schema and checkpoint digest, and can replay all bound arrays when given the source dataset; unbound legacy/raw checkpoints are marked or rejected when required. Synthetic analytical pilot only. Tests: `tests/test_experiments/test_observable_multipath_inputs.py`, `test_two_path_experiment_contract.py`, and `tests/test_models/test_observable_multipath.py`. |
| CAEZ magnitude forecast | CAEZ NPZ summary → `scripts/evaluate_caez_magnitude_forecast.py` → Ridge | Current and history-window summary features predict future magnitude; target rows are held separately and chronological pairs exclude unavailable history. | Report contract binds input/target/pair indices, source archive hashes, model identity and script transform digest. No licensed archive was read for software verification. Tests: `tests/test_scripts/test_audit_caez_npz.py`. |
| CAEZ cross-environment forecast | Source CAEZ NPZ → Ridge; target archive is evaluation-only in `scripts/evaluate_caez_cross_environment.py` | Source-environment current/history summaries predict future magnitude; target-environment current/history summaries are evaluation inputs. | Same versioned report contract binds both matrices, targets, row pairs, both source archives, model and transform digest. No licensed archive was read. Tests: `tests/test_scripts/test_evaluate_caez_cross_environment.py`. |
| Probability calibration | Canonical runner model scores → `evaluation/probability_calibration.py` | Calibration scores, labels and groups fit a group-balanced Platt map; it consumes model outputs, not raw channel features. | Calibration parameters and provenance are embedded in the parent result; it is not a separate channel-input producer. Tests are part of `tests/test_experiments/test_runner.py`. |
| Sparse classical multipath estimator | `models/multipath.py::fit_sparse_multipath`, exercised by `experiments/multipath_order_stress.py` | Only masked observed CFR, frequency coordinates and pilot mask enter fitting; simulated path count/truth is used after fitting for the explicitly oracle-aided stress metric. | Per-observation fit result, no trained checkpoint. The estimator validates its vectors and Boolean mask directly rather than using a dataset adapter. This is a classical estimator diagnostic, not a receiver-qualified model. |
| Local-MLE identifiability diagnostics | `experiments/identifiability_monte_carlo.py` and `experiments/identifiability_matched_estimators.py` | Synthetic observed CFR and frequency arrays enter optimization; the local MLE receives truth-centered initialization by design, as stated in source limitations. | Diagnostic result reports, no retained inference model. Privileged initialization makes this a local optimizer/Fisher-information check, not a deployable path estimator. |
| CNN/CVNN and CORAL checkpoint APIs | `learning/real_cnn.py`, `learning/complex_cnn.py`, and `learning/coral.py`, called by canonical runner | Real/imaginary or complex CFR, or engineered real features, with train-only normalization; labels are passed separately. | Canonical runner saves these through its exact model-input contract and replay verifier. Raw checkpoint payloads now carry `artifact_contract_status=external_input_contract_required`, preserved by their loaders; a standalone save is explicitly unbound until accompanied by a verified external input/split contract. Direct external callers are not found in the repository. |
| Historical thesis implementations | `archive/legacy_2025/` and `notebooks/legacy_2025/` | Superseded scripts, notebooks, and prototype models. | Archive READMEs state these are historical and active packages must not import from the archive. This establishes repository classification, not exhaustive proof that no external consumer uses them. |

## Reproducible source enumeration

The source pass used these repository-root searches, excluding tests and the
historical archive from active-route matches:

```bash
rg -n '(optimizer\s*=|\.backward\(|partial_fit\(|\.fit\(|fit_[a-zA-Z0-9_]+\()' \
  --glob '*.py' --glob '!tests/**' --glob '!archive/**' --glob '!notebooks/**'
rg -n '(torch\.save|joblib\.dump|pickle\.dump|np\.savez.*checkpoint|state_dict\(\)|load_.*checkpoint|load_.*cnn)' \
  --glob '*.py' --glob '!tests/**' --glob '!archive/**' --glob '!notebooks/**'
rg -n '(run_experiment\(|run_profile_holdout\(|fit_real_cnn\(|fit_complex_cnn\(|fit_coral_mlp\(|fit_two_path_decoder\(|regressor\.fit|model\.fit|estimator\.fit)' \
  --glob '*.py' --glob '!tests/**' --glob '!archive/**' --glob '!notebooks/**'
rg -n '(from archive|import archive|legacy_2025)' \
  --glob '*.py' --glob '!archive/**' --glob '!notebooks/**'
```

The active matches resolve to the routes above plus sparse multipath and
truth-initialized local-MLE diagnostics; no other fit/save call sites were
found. The archived 2025 tree contains 69 files across `archive/` and
`notebooks/legacy_2025/`; its READMEs and the research plan classify it as
read-only historical reproduction. No active Python import from those trees
was found. This is repository-local evidence and does not establish that no
external consumer exists.

## Repository-local audit closure (2026-11-12)

Repeated the source enumeration against the current worktree, including
prediction, transform, checkpoint-load, and active archive-import searches.
The active matches resolve to the routes in the matrix; diagnostic/local-MLE
paths remain explicitly synthetic, privileged, or oracle-aided. The three
standalone real-CNN, complex-CNN, and CORAL loader tests pass and confirm raw
checkpoints retain `external_input_contract_required`; canonical runner
artifacts bind and replay their exact sidecars. The two-path CLI writes model
contracts before atomic publication. Direct canonical-runner artifact
publication and two-path checkpoint binding regressions each pass. The archive and historical-notebook trees
contain 69 files, their README files classify them as historical, and no active
Python import from those trees was found.

Exact source checks:

```bash
rg -n '(optimizer\s*=|\.backward\(|partial_fit\(|\.fit\(|fit_[A-Za-z0-9_]+\()' \
  --glob '*.py' --glob '!tests/**' --glob '!archive/**' --glob '!notebooks/**' .
rg -n '(\.predict\(|predict_proba\(|decision_function\(|transform\()' \
  --glob '*.py' --glob '!tests/**' --glob '!archive/**' --glob '!notebooks/**' .
rg -n '(torch\.save|joblib\.dump|pickle\.dump|state_dict\(\)|load_.*checkpoint|load_.*cnn|torch\.load|joblib\.load|pickle\.load)' \
  --glob '*.py' --glob '!tests/**' --glob '!archive/**' --glob '!notebooks/**' .
rg -n '(from archive|import archive|legacy_2025)' \
  --glob '*.py' --glob '!archive/**' --glob '!notebooks/**' .
```

This closes the repository-local enumeration and leakage-boundary audit only.
It cannot enumerate external callers or certify measured CAEZ archives. No
scientific acceptance, production RT generation, or deployment claim follows.
Focused verification commands:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --locked python -m pytest \
  -o addopts='' -q --confcutdir=tests/test_learning \
  tests/test_learning/test_real_cnn.py tests/test_learning/test_complex_cnn.py \
  tests/test_learning/test_coral.py -k 'load or checkpoint'  # 3 passed
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --locked python -m pytest \
  -o addopts='' -q --confcutdir=tests/test_experiments \
  tests/test_two_path_experiment_contract.py \
  -k 'checkpoint_contract_binds_inputs_targets_splits'  # 1 passed
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --locked python -m pytest \
  -o addopts='' -q --confcutdir=tests/test_experiments \
  tests/test_runner.py -k 'writes_reproducible_complete_artifacts'  # 1 passed
```
