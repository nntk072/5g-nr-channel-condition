# MVP implementation status

**Current workspace validation (Unified plan edition 3.251 / Master plan edition 3.247; 2026-10-23):** The full local suite passes with **1,204 passed, 3 skipped, and 85.85% coverage** (2m39s). CI-scoped Ruff check and format pass across **323 maintained files**, configured mypy passes across **188 source files**, lock validation passes, and the package build produces both wheel and source distribution. Hosted CI has not run on this local revision. These checks verify software behavior, not completion of the scientific dataset or qualification gates.

**Bounded additions recorded in the current plan editions:** G3.03 now has a fail-closed, pilot-only evaluator whose acquisition-unit independence and split-manifest authenticity remain externally unverified; it cannot produce a qualification pass. G4.07 adds one dual-VH CDL-A XPR power check; G5.14 exports 15-kHz FR1 Case-A SSB candidate timing only; G7.02 hashes versioned target metadata, G7.06 tests compressed-object tamper detection, G7.10 flags cross-split geometry/CFR near duplicates for review, and G7.14 reports target coverage from an explicit split/family inventory. These are component evidence; the broader plan gates remain open.

**Status: simulation-only software MVP complete; broader research plan remains open.** The absolute-delay/path-partition implementation is committed as `2b03354`; its complete local suite passed (481 passed, 3 skipped, 88.12% coverage), and hosted CI run [39710231241](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39710231241) passed all four jobs. Follow-up commits `00928a3`, `dac545c`, and `70421a6` added native frequency-bin, path-coefficient, Friis, empty-path, and array-mask checks; hosted CI runs [39711191556](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39711191556), [39711520188](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39711520188), and [39712795569](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39712795569) pass all four software jobs. Commit `1a1de6c` adds all V/VH polarization pairs for synthetic/explicit 2×3 MIMO partition tests; all 30 focused native/mock tests pass locally, and hosted CI [39713206686](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39713206686) passes all four software jobs. The complete local suite passes at 88.11% coverage. Hosted CI does not install Sionna RT, so native cases are locally verified. Ruff, format, mypy (139 configured source files), lock check, and package build pass. These are software and bounded physics gates, not validation of a general channel-condition benchmark.

**Current build addition (2026-10-22):** the new
`integrations/sionna/standardized_pusch.py` path connects immutable TDL/CDL
draws to Sionna's PUSCH transmitter and DMRS LS estimator, keeps noisy receiver
data apart from perfect CFR/profile-proxy targets, and writes records through
the hash-verified NR store. `scripts/build_standardized_pusch_dataset.py`
builds a reproducible local sample; a four-record TDL-A dataset was generated
at `outputs/pusch-tdla-provenance-demo` in this workspace. The native test slice passes
16 tests, Ruff and mypy pass for the touched integration modules, and `uv build`
produces both wheel and source distribution. This is an ideal static
frequency-domain SISO/SIMO fixture: it does not add impairments, ICI/ISI,
absolute power calibration, geometric LOS labels, or temporal propagation.
The [3D Urban43 figure](figures/rt-urban43-spatial-holdout-3d-20261022.png)
visibly shows 16 building meshes, three TX sites, and 120 audited RX locations;
its labels certify geometric direct-path visibility only. The extended
[box-scene depth sweep](figures/rt-box-los-nlos-convergence-depth20-100k-20261022.png)
still changes at max depth 20, so the RT solver-depth convergence gate remains
open and this result does not establish a production ray-tracing configuration.

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
| Package, CLI, dependency lock, lint, typing, and tests | Historical hosted CI passed at the recorded revision; current workspace checks are listed above | Hosted CI [39723486090](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39723486090) passed all four jobs for the UMCD schema/store revision. The current uncommitted workspace has not been pushed through hosted CI; CI does not install Sionna RT or measure production-scale storage performance. |
| Environment-aware RT backend | Implemented and locally verified | Mitsuba 3.9.1 / Dr.Jit 1.5.0 LLVM variant traces natively in this WSL Linux environment. Automatic selection rejects this host's failed CUDA/OptiX probe and falls back to LLVM. No separate LLVM-project download is needed here; OptiX remains an optional runtime path for a later supported host. |
| Geometric RT data and provenance | Historical six-scene pilot; canonical regeneration pending | Six balanced datasets (40 links each) carry geometric direct-path labels and scene/runtime provenance. These pre-correction channel arrays use independently normalized path delays; preserve them as legacy relative-delay-view experiments and regenerate before absolute-delay or canonical-channel claims. Scene asset identities are recorded, but geometry/material realism is not independently qualified. |
| UMCD parent/link/path/channel/target schema | Initial typed schema and immutable bundle implemented; release foundation incomplete | `datasets/network.py`, `path_oracle.py`, `channel_view.py`, `targets.py`, and `episode_store.py` define validated episode/network, explicit array port weights, ragged absolute-delay paths, named-axis per-link CFR, reasoned target masks, and separate content-addressed storage layers. Focused dataset tests pass. Receiver-view linkage/access policy, time-varying poses, full solver/material provenance, split locks, schema migration and scalable columnar/chunked storage remain open. This initial NPZ object backend is not presented as production-scale UMCD storage. |
| Absolute-delay RT CFR and path partition | Adapter corrected; focused native validation passes | Pinned Sionna 2.2 LLVM tests verify absolute free-space delay/phase, Friis gain/phase, same-Paths direct-plus-indirect CFR conservation for synthetic and explicit 2×3 arrays across all four V/VH polarization pairs, zero-path behavior, versioned generation semantics, blocked-link multi-interaction conservation, and end-to-end slab Fresnel coefficients for V/V (TM) and H/H (TE) against independent analytic values. The current full local suite passes at 86.09% coverage (1,085 passed, 3 skipped); hosted CI [39723486090](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39723486090) is for an earlier committed revision, and native RT remains local-only. Canonical dataset regeneration remains pending. See [`PHYSICS_VERIFICATION_STATUS.md`](PHYSICS_VERIFICATION_STATUS.md). |
| PUSCH/DMRS observation adapter | Two explicit power modes implemented; absolute-power mode locally verified | `loslab nr-observe` produced 240 normalized-SNR diagnostic observations of shape `[240, 1, 72]`, preserving group IDs and label semantics. A new opt-in mode applies explicit watts per active resource element and fixed `k_B T B F` noise, preserving channel-gain differences; a CUDA Sionna PHY regression verifies half gain stays half. Both modes retain ideal synchronization and no inter-symbol channel variation. |
| Shared classification/evaluation workflow | Exercised across six leave-one-scene-family-out folds | Logistic regression, random forest, and raw-I/Q MLP used the same precomputed DMRS observations. Each family was held out once (40 samples, 4–20 groups); the other five supplied train/validation groups. Seven fixed seeds and per-sample predictions are retained. |
| Joint TX/RX spatial holdout pilot | Audited on one synthetic City43 scene | Three held-out transmitter domains pair with disjoint 25 m receiver disks (30/30/110 m geometric gaps), 40 balanced samples per domain, and 7–11 spatial groups per test fold. The audit confirms all 120 receivers are inside their declared disks and no cross-domain pair is within 1 m. TX and RX move together, so this is not a TX-only or independent-scene generalization result; metrics are descriptive smoke evidence. See [`rt-urban43-joint-location-holdout-20261022.json`](results/rt-urban43-joint-location-holdout-20261022.json) and the [3D scene figure](figures/rt-urban43-spatial-holdout-20261022.png). |
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
