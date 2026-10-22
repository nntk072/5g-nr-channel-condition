# MVP implementation status

**Status: simulation-only software MVP complete; broader research plan remains open.** The absolute-delay/path-partition implementation is committed as `2b03354`; its complete local suite passed (481 passed, 3 skipped, 88.12% coverage), and hosted CI run [39710231241](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39710231241) passed all four jobs. Ruff check/format, mypy (139 configured source files), lock check, and package build passed for that commit. The later native odd/even-grid and multipath coefficient-reconstruction regressions pass locally; hosted CI for that follow-up is pending. These are software and bounded physics gates, not validation of a general channel-condition benchmark.

**Physics-audit correction (2026-10-22):** `integrations/sionna/adapter.py::rt_cfr`
previously normalized each RT link's first path delay independently. It now
preserves the absolute Sionna path-delay reference and records this channel
contract in generation-report schema v2. Existing datasets, DMRS observations,
and model runs retain their original hashes and are legacy relative-delay-view
artifacts; do not use them as absolute-delay canonical records. Re-generate the
canonical RT data and rerun affected downstream comparisons after the remaining
Phase 0 path-conservation gates pass. Native pinned-Sionna regression checks,
adapter contract tests, and the complete suite have passed; canonical
regeneration remains pending.

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
| Package, CLI, dependency lock, lint, typing, and tests | Complete locally and hosted CI | Pinned GPU-enabled suite: 425 passed, 16 skipped, 80.41% coverage. Ruff check/format pass across maintained source, test, and script directories; mypy passes on 137 configured source files; uv lock --check and uv build pass. Hosted CI [39697604971](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39697604971) passed all jobs. |
| Environment-aware RT backend | Implemented and locally verified | Mitsuba 3.9.1 / Dr.Jit 1.5.0 LLVM variant traces natively in this WSL Linux environment. Automatic selection rejects this host's failed CUDA/OptiX probe and falls back to LLVM. No separate LLVM-project download is needed here; OptiX remains an optional runtime path for a later supported host. |
| Geometric RT data and provenance | Historical six-scene pilot; canonical regeneration pending | Six balanced datasets (40 links each) carry geometric direct-path labels and scene/runtime provenance. These pre-correction channel arrays use independently normalized path delays; preserve them as legacy relative-delay-view experiments and regenerate before absolute-delay or canonical-channel claims. Scene asset identities are recorded, but geometry/material realism is not independently qualified. |
| Absolute-delay RT CFR and path partition | Adapter corrected; focused native validation passes | Pinned Sionna 2.2 LLVM tests verify absolute free-space delay/phase, same-Paths direct-plus-indirect CFR conservation for SISO and explicit 2×3 MIMO arrays, and versioned generation semantics. Full-suite result: 481 passed, 3 skipped, 88.12% coverage. Hosted CI run [39710231241](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39710231241) passed all four jobs for commit `2b03354`; odd/even-bin and manual two-path coefficient/delay regressions were added afterward and pass locally. Follow-up hosted CI and canonical dataset regeneration remain pending. See [`PHYSICS_VERIFICATION_STATUS.md`](PHYSICS_VERIFICATION_STATUS.md). |
| PUSCH/DMRS observation adapter | Implemented and exercised across all six scenes | `loslab nr-observe` produced 240 observations of shape `[240, 1, 72]`, preserving group IDs and canonical label semantics. Configuration: 12 dB, seed 20261022, ideal synchronization. The adapter records its link-path-loss normalization assumption. |
| Shared classification/evaluation workflow | Exercised across six leave-one-scene-family-out folds | Logistic regression, random forest, and raw-I/Q MLP used the same precomputed DMRS observations. Each family was held out once (40 samples, 4–20 groups); the other five supplied train/validation groups. Seven fixed seeds and per-sample predictions are retained. |
| Classical multipath and learning baselines | Implemented | Grouped classifiers and the unknown-order sparse CFR estimator are available. The multipath-order stress report is analytical/synthetic and is not geometric RT evidence. |
| Locked independent-scene MVP evaluation | Completed at implementation-smoke scale | All six scene families were held out once with seven seeds and matched inputs; fold/split auditing and per-scene metrics are in [`mvp-six-scene-dmrs-losso-seed20261022.json`](results/mvp-six-scene-dmrs-losso-seed20261022.json). Scene-macro AUROCs were 0.560 (logistic), 0.554 (RF), and 0.782 (raw MLP). Six selected scenes and 40 links/scene remain too small for benchmark claims; held-out fold sizes range from 4 to 20 spatial groups. The earlier three-seed Etoile-only report remains available for exact fixed-fold replay. |
| Linux reproduction and release instructions | Complete for the MVP scope | The six-scene generation script, `nr-observe`, and `experiment` commands form the reproduction route. Hosted Linux CI passed from clean checkouts; native Sionna RT was exercised locally on LLVM. |
| Measured-data extension beyond MVP | Three-campaign CAEZ exploratory magnitude-forecast pilot completed | The user accepted the license. Indoor/outdoor/hallway archives pass CRC/schema/timestamp-order checks and remain Git-ignored. Source-trained Ridge beats persistence on the compact one-second mean-magnitude target indoors and outdoors, but loses substantially in hallway; the separate full-tensor indoor baseline favors persistence over trend. One tail per campaign, campaign/device confounding, unresolved published-vs-file sample-count discrepancies, and no LOS/NLOS truth prevent broad transfer claims. See docs/research/results/caez-cross-environment-summary-forecast.json and MEASURED_DATA_CANDIDATES.md. |

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
