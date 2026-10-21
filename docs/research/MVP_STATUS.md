# MVP implementation status

**Status: simulation-only software MVP complete.** The latest implementation
commit is `a73a341`; hosted CI run
[39694168089](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39694168089)
passed all four jobs. The local full suite passes (423 passed, 16 skipped, 80.29%
coverage), Ruff and formatting pass across the maintained tree, mypy passes on
135 source files, and lock/build checks pass. This is an MVP implementation
and evaluation workflow, not a validated channel-condition benchmark.

The different model targets and channel-family roles are summarized in
[`OUTPUT_TASKS_AND_CHANNELS.md`](OUTPUT_TASKS_AND_CHANNELS.md) to keep
geometric LOS/NLOS, standardized-profile proxy labels, multipath estimation,
and measured CSI-magnitude forecasting separate.

## MVP boundary

The minimum viable research software is a reproducible, simulation-only 5G NR
channel-condition workflow. It must:

1. Generate geometric RT channel samples with explicit direct-path label
   semantics, scene identity, and solver/runtime provenance.
2. Feed those channels through the pinned Sionna PUSCH/DMRS observation path,
   recording its resource-grid, SNR, noise, synchronization, and normalization
   assumptions.
3. Train and evaluate multiple receiver-input baselines on identical samples,
   with a complete scene family held out and no scene/path/label metadata passed
   as model features.
4. Save a reproducible run manifest, predictions, metrics, split identity, and
   artifact hashes, and provide a tested command sequence.
5. Pass portable CI and at least one native Sionna run on a supported Linux
   backend. Runtime detection probes CUDA/OptiX and uses it only after a native
   trace succeeds; otherwise it selects the supported LLVM CPU backend.

The MVP is deliberately simulation-only. It does not require a best benchmark
score, measured-radio data, a novelty claim, long-horizon forecasting, beam
utility, or paper-submission readiness. It must report negative or near-chance
results honestly.

## Current evidence

| Requirement | State | Evidence and limitation |
|---|---|---|
| Package, CLI, dependency lock, lint, typing, and tests | Complete locally and hosted CI | Pinned GPU-enabled suite: 423 passed, 16 skipped, 80.29% coverage. Ruff check/format pass across maintained source, test, and script directories; mypy passes on 135 source files; `uv lock --check` and `uv build` pass. Hosted CI [39694168089](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39694168089) passed all four jobs. |
| Environment-aware RT backend | Implemented and locally verified | Mitsuba 3.9.1 / Dr.Jit 1.5.0 LLVM variant traces natively in this WSL Linux environment. Automatic selection rejects this host's failed CUDA/OptiX probe and falls back to LLVM. No separate LLVM-project download is needed here; OptiX remains an optional runtime path for a later supported host. |
| Geometric RT data and provenance | MVP data path exercised across six scene families | Six balanced datasets (40 links each) carry geometric direct-path labels and scene/runtime provenance. Scene asset identities are recorded, but geometry/material realism is not independently qualified. |
| PUSCH/DMRS observation adapter | Implemented and exercised across all six scenes | `loslab nr-observe` produced 240 observations of shape `[240, 1, 72]`, preserving group IDs and canonical label semantics. Configuration: 12 dB, seed 20261022, ideal synchronization. The adapter records its link-path-loss normalization assumption. |
| Shared classification/evaluation workflow | Exercised across six leave-one-scene-family-out folds | Logistic regression, random forest, and raw-I/Q MLP used the same precomputed DMRS observations. Each family was held out once (40 samples, 4–20 groups); the other five supplied train/validation groups. Seven fixed seeds and per-sample predictions are retained. |
| Classical multipath and learning baselines | Implemented | Grouped classifiers and the unknown-order sparse CFR estimator are available. The multipath-order stress report is analytical/synthetic and is not geometric RT evidence. |
| Locked independent-scene MVP evaluation | Completed at implementation-smoke scale | All six scene families were held out once with seven seeds and matched inputs; fold/split auditing and per-scene metrics are in [`mvp-six-scene-dmrs-losso-seed20261022.json`](results/mvp-six-scene-dmrs-losso-seed20261022.json). Scene-macro AUROCs were 0.560 (logistic), 0.554 (RF), and 0.782 (raw MLP). Six selected scenes and 40 links/scene remain too small for benchmark claims; held-out fold sizes range from 4 to 20 spatial groups. The earlier three-seed Etoile-only report remains available for exact fixed-fold replay. |
| Linux reproduction and release instructions | Complete for the MVP scope | The six-scene generation script, `nr-observe`, and `experiment` commands form the reproduction route. Hosted Linux CI passed from clean checkouts; native Sionna RT was exercised locally on LLVM. |
| Measured-data extension beyond MVP | CAEZ indoor CSI audited and one bounded forecast pilot completed | The user accepted the CAEZ license. The local preprocessed file passes CRC/SHA/schema checks and is kept Git-ignored. A one-second full-tensor magnitude forecast favors persistence over linear trend. A separate Ridge regressor improves a compact per-antenna mean-magnitude dB target over persistence on the same final chronological block; it has no LOS/NLOS labels and is not cross-environment evidence. See `docs/research/results/caez-indoor-audit.json` and `caez-indoor-magnitude-forecast.json`. |

## MVP exit checklist

- [x] Generate balanced datasets from six compatible scene families with
  immutable sample, scene, solver, and asset identity provenance. Scene/material
  realism qualification remains outside this simulation-only MVP.
- [x] Run `loslab nr-observe` over all six datasets with one receiver
  configuration and recorded observation semantics on Linux/LLVM plus CUDA PHY.
- [x] Compare logistic regression, random forest, and raw-I/Q MLP on identical
  DMRS observations; save per-sample predictions and seed-level uncertainty.
  Model-level calibration is explicitly unimplemented for this MVP.
- [x] Lock `rt_etoile` out of training and validation; the run manifest confirms
  it is the sole test domain and records the split indices hash.
- [x] Record the generation, observation, and evaluation commands and hashes in
  the committed evidence report.
- [x] Pass the clean hosted Linux workflow on the pushed numerical-fix commit
  (CI run 39684428037; all four jobs passed).
- [x] Review the report for simulation-only wording and retain weak/negative
  results without changing the locked evaluation.

The six-scene run is an implementation-scale MVP check, not a stable benchmark.
The software MVP gate is complete. The broader scientific gates in
`MASTER_RESEARCH_PLAN.md`—independent measurements, qualified materials,
receiver impairments, broad standards conformance, model novelty, forecasting,
and downstream utility—remain open and are not treated as complete.
