# MVP implementation status

**Split Policy v2 durability update (2026-10-24):** Immutable registry creation now persists the file and directory entries on supported POSIX filesystems: the file is fsynced before publication, then the containing directory is fsynced after the canonical hard link is created and after the temporary alias is removed. The regression asserts this order and a final link count of one. All **49 Split Policy v2 tests** pass; changed-path Ruff, formatting, and mypy checks pass. This is crash-durability hardening, not trusted signing or proof of acquisition-unit independence.

**G4 Core-1 evidence reconciliation (2026-10-24):** Re-ran all 110 focused native CPU cases: 50 TDL/CDL profile-topology covariance cells, 50 profile-topology path-axis reconstruction cases, and 10 joint CDL-A–E TX/RX covariance cases for 2×2 and 4×4 arrays. All pass their declared family-wise interval and precision checks, so the mandatory named Core-1 matrix is verified. This does not certify arbitrary configurations, all polarization/time behavior, geometric RT, or external physical accuracy; broader G4 validity remains open.

**G3 joint-family planning update (2026-10-24):** Fixed a multiplicity gap in the data-free acquisition-unit planner: its exact-binomial complex-CSI claim had been allocated separately from the five distributional claims. The v3 planner now Bonferroni-controls all six claims across all strata and power-allocates the same family; its lower-bound count combines the CSI and distributional requirements before checking angular compatibility. For one documented hypothetical stratum, the CSI component requires 293 clusters and the combined six-claim Hoeffding lower bound is 10,583. The angular pass-probability assumption, bootstrap-assurance alignment, independent/representative measurement units, and material measurements remain unresolved, so this is not a campaign size or qualification. All **33 G3 planner tests** and G3 evaluator/package tests pass. The complete suite passes **1,616 tests, 3 skipped, at 84.69% coverage**; CI-scoped Ruff/format and mypy (203 source files) pass.

**G5.09 independent-channel receiver sensitivity update (2026-10-24):** Added a separate, preregistered CPU PUSCH study over 64 independent seeded two-path channel draws, pairing static and 750-Hz echo-Doppler outcomes on the same payloads and waveform noise within each draw. Each channel contributes one equally weighted BLER estimate from eight codewords; the paired effect interval uses a two-sided 99.5% percentile bootstrap over channel draws. Baseline is 0/512 CRC failures; because the bootstrap is degenerate at zero, its conservative one-sided exact bound instead uses the prevalence of channel draws with any failure: 0/64, giving a 99.5% Clopper-Pearson upper bound of 0.0795 on mean BLER. Mobile BLER is 202/512 (0.395); the paired mean increase is 0.395 with a 99.5% interval [0.242, 0.553]. The frozen baseline and effect criteria pass for this declared synthetic distribution. The full suite passes **1,616 tests, 3 skipped, at 84.69% coverage**; CI-scoped Ruff/format pass across 367 Python files, mypy passes across 203 source files, and `uv lock --check` plus `uv build` pass. This is conditional simulation evidence for a static-versus-time-varying comparison; it does not isolate intra-symbol ICI, qualify a deployment channel population, or close G5.09. The result is [`g5-independent-channel-pusch-20261024.json`](results/g5-independent-channel-pusch-20261024.json); reproduce with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run --no-sync --locked python experiments/g5_independent_channel_receiver.py`.

**G5.14 blind acquisition-to-PBCH update (2026-10-24):** A capture-level CPU chain now uses PSS/SSS to recover PCI/timing/CFO, tests all eight candidate/half-frame combinations for Lmax=4, then DM-RS-equalizes and CRC-decodes PBCH. In the seeded 24-dB capture (23-sample offset, 2.5-kHz CFO, 5-sample two-tap echo), it returns the exact candidate, half-frame, SFN and MIB with valid CRC; CFO error is below 100 Hz. The focused PBCH and NR-sync suites pass **143 tests**. The full repository suite was subsequently rerun with the independent-channel G5 study and passed as reported above. This remains one controlled capture, not broad synchronizer/BLER evidence; RT production remains gated.

**G5.14 known-occasion receiver update (2026-10-24):** The CPU PBCH chain now demodulates a four-symbol OFDM SSB, interpolates per-symbol channel estimates from PBCH-DM-RS, equalizes data REs, creates LLRs, decodes BCH and returns typed MIB fields. A flat case and all 16 seeded 24-dB, CP-contained two-tap AWGN trials pass CRC and exact MIB recovery. The focused PBCH suite passes **121 tests**; Ruff, format, and mypy pass. Synchronization/candidate search, standard power/waveform qualification, mobility, multiple antennas and broader receiver claims remain open. The preceding full repository suite passed 1,611 tests with 3 skipped and 85.06% coverage before the blind acquisition wrapper; production RT remains gated.

**G5.14 MIB interpretation update (2026-10-24):** A fixed-schema 24-bit BCCH-BCH MIB serializer/parser now connects typed RRC MIB fields to the ideal PBCH chain. The integrated test caught and fixed SFN placement after the leading MIB CHOICE bit; payload validation and recovery now use bits 1–6. The focused PBCH suite passes 119 tests; targeted Ruff, format, and mypy pass. General ASN.1/PER extensions, messageClassExtension, noisy reception, channel estimation, power scaling, OFDM, and receiver qualification remain open. Full repository tests were not rerun for this slice; RT production remains gated.

**G4 angular-statistic update (2026-10-24):** The CDL-A-E delay/cos(AoA) cluster-moment suite passes five profile cases, including hash-pinned CDL-D/E composed LOS/diffuse zero-delay references with 8,192 independent CPU draws per profile and a shared 99% family-wise rule. Ruff, format, and whitespace checks pass. This is bounded standardized-model consistency evidence; no full-suite rerun has occurred for this change, and broad G4/RT qualification remains open.

**Status snapshot:** 2026-10-24; see the current execution and gate crosswalks in the [Master Research Plan](MASTER_RESEARCH_PLAN.md) and [Unified Multipath Superdataset Plan](UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md). This page distinguishes software-MVP completion from the incomplete scientific dataset/research program.

**Current focused validation after commit `271b5c3` (2026-10-24):** The PBCH/SSB/NR-sync suites pass **166 tests**. This includes 0/3-dB relative PSS-to-SSS EPRE profile tests that verify the 3-dB linear-power ratio and preserve non-PSS RE values; the Sionna 2.2.0 OFDM waveform cross-check and candidate-index acquisition cases also pass. Ruff check/format and targeted mypy pass on the changed paths. The full repository suite, full-tree checks, lock/build checks, and hosted CI have not been rerun for this commit. Absolute block-power calibration and broad G5.14 qualification remain open.

**Previous focused validation after commit `2c6cd43` (2026-10-24):** The PBCH and NR-sync suites passed **159 tests**. The unscaled SSB OFDM modulator/demodulator matched Sionna 2.2.0 at FFT sizes 256/512/1024 with unequal per-symbol CP and round-tripped active grid values; existing known-occasion and blind acquisition cases passed. Ruff check/format and targeted mypy passed. This prior snapshot is retained for chronology.

**Previous focused validation after commit `245612f` (2026-10-24):** The PBCH and NR-sync suites passed **156 tests**. In addition to the high-index known-occasion decodes at candidate counts 10/20/64, blind acquisition recovered the highest SSB candidate at counts 8/10/20/64 from the payload's candidate-index prefix; the existing Lmax=4 timing/CFO/multipath case remained passing. Ruff check and formatting passed on the changed Python files, and targeted mypy passed on the changed acquisition module. This prior snapshot is retained for chronology.

**Research status: the Unified research dataset/program remains incomplete. G3 is not qualified, and production RT generation remains gated.**

**Previous local validation snapshot (2026-10-23; superseded by the current tranche below):** The integrated suite passed with 1,290 passed and 3 skipped; Ruff, formatting, mypy, lock and build passed. That snapshot predates the current G3 package verifier and D1 generation gate changes and is not validation of this current checkout.

**Last full-suite validation before PBCH payload recovery (2026-10-24):** That checkpoint collected **1,580 tests**; **1,577 passed and 3 skipped**, with **84.94% coverage** against the configured 75% minimum. The CI-scoped Ruff check passed; formatting passed for **359 maintained Python files**; mypy passed on **199 source files**; `uv lock --check` and `uv build` passed. Hosted CI has not run for that revision, and standard CI does not install native Sionna RT. These software checks do not establish scientific qualification.

**Previous focused validation after commit `669c9c8` (2026-10-24):** The PBCH and NR-sync suites passed **152 tests**, including independent scalar Gold-sequence checks and noiseless known-occasion receiver decodes at candidate indices 9, 19, and 63 for counts 10, 20, and 64. Ruff check and format passed on the six changed Python files; mypy passed on the three changed source modules. This prior snapshot is retained for chronology and does not validate blind acquisition at those counts.

**Previous focused validation (2026-10-24, before candidate-count extension):** The G3.03 bootstrap-assurance planner, G4 CDL-A–E joint TX/RX covariance, and preceding G5 implementation sets passed (**91 passed**); the separate Split Policy v2 suite passed (**48 passed**). The PBCH-only suite passed **122 tests**, including fixed-schema MIB field checks and an ideal payload-to-BCH-to-QPSK-to-decoded-MIB receiver round trip. The preceding project-wide suite passed 1,611 tests with 3 skipped and 85.06% coverage before the PSS/SSS-to-PBCH integration. This prior snapshot is retained for chronology; it does not validate the current commit.

**Edition 3.263 implementation update (2026-10-24):** G3.03 now has an exact-byte proposal package verifier and v0.2 schema; verifier success remains `NOT_QUALIFIED`. Production D1 generation fails closed without a trusted approval-signature keyring; explicitly requested pilots are bounded by derived work units and confined to `/tmp`. Added focused G4 covariance/topology matrices, one narrow G5 DM-RS mapping case, G6 timestamp-aligned trajectory targets, and G7 corpus-audit tooling. These are component tests and tools, not broad scientific gate passes. Hosted CI has not run on this revision.

**Bounded additions recorded in the current plan editions:** G3.03 has a fail-closed, pilot-only evaluator plus data-free Hoeffding, exact-binomial, and explicit-IID angular-maximum planning. A v0.1 acquisition JSON Schema and fictional fixture define a review proposal only and are incomplete without campaign/instrument records, the preregistered split/strata manifest, scored prediction/interval rows and evaluator-output binding; scenario-conditional bootstrap assurance is implemented, but defensible scenario distributions, the complete multi-estimand design, external verification of acquisition-unit independence/provenance, and representative sampling remain unresolved, so no qualification pass is possible. G4 adds pinned CDL-A/B/C/D spatial covariance and CDL-D/E LOS/diffuse temporal references plus fixed TDL-A30/D30 FR2 checks; the independent hash-pinned joint TX/RX covariance reference now passes for CDL-A–E on both 2×2 and 4×4 square ULAs under one 1,360-component 99% family-wise rule, with 16,384 independent draws per case; broader profile/configuration qualification remains open. G5.09's 64-channel static/symbol-static/continuous probe shows that static-to-mobile sensitivity passes at both 750 and 1,500 Hz, but neither isolated intra-symbol effect meets its 0.10 BLER-increase criterion: paired differences are 0.0137 [−0.0098, 0.0449] at 750 Hz and −0.0020 [−0.0098, 0] at 1,500 Hz under a five-claim 99% family-wise design. These conditional synthetic results do not establish deployment impact, and G5.09 remains open. The separate DM-RS noise-variance shape fix is also conditional receiver evidence. G5.14 now includes PBCH payload assembly/interleaving/scrambling and receiver-side serialized-MIB/timing/index-prefix recovery, CRC24C and N=512 Polar coding with E=864 repetition, physical-channel scrambling, QPSK, SSB grid mapping, and CRC-aided decoding from physically descrambled LLRs. general ASN.1/PER extensions and messageClassExtension decoding, channel-affected physical reception, demodulation/channel estimation, power scaling, OFDM, timing/beam validation, PRS/SRS, and end-to-end conformance remain open. Split Policy v2 now compares file identity, type, link count, size, mtime, and ctime across open and read, rejecting observable same-inode rewrites as well as symlink/hard-link aliases without changing assignments. G7.13 stages metadata in temporary SQLite with deterministic ordering, and focused regressions verify cleanup after early close and reader failure. Its traced-Python synthetic growth is not an RSS bound; SQLite-native memory, ZIP central-directory metadata and corpus-scale behavior remain outside the estimate. G7.02/G7.06/G7.10/G7.14 components remain scoped as stated in their gates. Production RT generation remains gated.

**Status: simulation-only, six-scene implementation-smoke software MVP complete; the Unified research dataset/program remains incomplete, G3 is not qualified, and production RT generation is gated.** The absolute-delay/path-partition implementation is committed as `2b03354`; its complete local suite passed (481 passed, 3 skipped, 88.12% coverage), and hosted CI run [39710231241](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39710231241) passed all four jobs. Follow-up commits `00928a3`, `dac545c`, and `70421a6` added native frequency-bin, path-coefficient, Friis, empty-path, and array-mask checks; hosted CI runs [39711191556](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39711191556), [39711520188](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39711520188), and [39712795569](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39712795569) pass all four software jobs. Commit `1a1de6c` adds all V/VH polarization pairs for synthetic/explicit 2×3 MIMO partition tests; all 30 focused native/mock tests pass locally, and hosted CI [39713206686](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39713206686) passes all four software jobs. The complete local suite passes at 88.11% coverage. Hosted CI does not install Sionna RT, so native cases are locally verified. Ruff, format, mypy (139 configured source files), lock check, and package build pass. These are software and bounded physics gates, not validation of a general channel-condition benchmark.

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

The software MVP is complete only at a reproducible, simulation-only,
six-scene implementation-smoke scope. It is not completion of the Unified
research dataset/program: G3 qualification is open and production RT generation
remains gated. This MVP scope covers a workflow that can:

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
| Absolute-delay RT CFR and path partition | Adapter corrected; focused native validation passes | Pinned Sionna 2.2 LLVM tests verify absolute free-space delay/phase, Friis gain/phase, same-Paths direct-plus-indirect CFR conservation for synthetic and explicit 2×3 arrays across all four V/VH polarization pairs, zero-path behavior, versioned generation semantics, blocked-link multi-interaction conservation, and end-to-end slab Fresnel coefficients for V/V (TM) and H/H (TE) against independent analytic values. The an earlier full local suite passed with 1,085 tests and 3 skips (historical snapshot); hosted CI [39723486090](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39723486090) is for an earlier committed revision, and native RT remains local-only. Canonical dataset regeneration remains pending. See [`PHYSICS_VERIFICATION_STATUS.md`](PHYSICS_VERIFICATION_STATUS.md). |
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
