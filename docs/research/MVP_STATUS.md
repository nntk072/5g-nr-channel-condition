# MVP implementation status

**Status: not complete yet.** The core packages and an end-to-end software path
exist, but the project has not passed the MVP release gate below. A green unit
suite or a working single-scene smoke run is not sufficient evidence of a
complete channel-condition MVP.

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
   artifact hashes, and provide a tested command sequence from a clean checkout.
5. Pass portable CI and at least one native Sionna run on a supported Linux
   backend. OptiX is preferred when its native trace probe succeeds; LLVM is the
   supported CPU fallback. The backend is not selected based only on CUDA
   visibility.

The MVP is deliberately simulation-only. It does not require a best benchmark
score, measured-radio data, a novelty claim, long-horizon forecasting, beam
utility, or paper-submission readiness. It must report negative or near-chance
results honestly.

## Current evidence

| Requirement | State | Evidence and limitation |
|---|---|---|
| Package, CLI, dependency lock, lint, typing, and tests | Implemented; local consolidated checks pass, hosted CI is pending the final push | 409 passed, 16 skipped, 80.41% coverage; Ruff/format, mypy, lock, and build pass locally. |
| Environment-aware RT backend | Implemented and locally verified | This WSL host's CUDA/OptiX trace probe fails; LLVM traces a native direct path. |
| Geometric RT data and provenance | Implemented; multi-scene MVP run still required | Cached generation, label/report contracts, RT-to-NR trajectory and scene pilots exist. Existing pilot scenes/materials are not independently qualified real environments. |
| PUSCH/DMRS observation adapter | Implemented and locally verified | `loslab nr-observe` converts saved RT CFR into DMRS LS estimates, preserves group IDs and canonical label semantics, and refuses output overwrite. It assumes ideal synchronization and normalizes away link path loss. |
| Shared classification/evaluation workflow | Implemented and locally exercised | One 40-link saved RT pilot passed through CUDA DMRS generation and the experiment runner. Its one-seed, one-scene AUROC was 0.50; this is only a plumbing smoke and cannot validate scene transfer. |
| Classical multipath and learning baselines | Implemented; matched multi-scene evaluation remains open | Grouped classifiers and the new unknown-order sparse CFR baseline are available. The latter's analytical stress report is synthetic, not RT evidence. |
| Locked independent-scene MVP evaluation | Not complete | Need multiple generated/qualified scene families, one frozen scene-family test set, and the same DMRS inputs for all baselines. |
| Clean Linux reproduction and release instructions | Partial | Quickstart now describes the RT→DMRS→experiment path; a clean-environment run and complete multi-scene command sequence remain to be recorded. |

## MVP exit checklist

- [ ] Create or select at least three compatible RT scene families, audit their
  geometry/material assets, and generate balanced training/development and
  locked held-out-scene datasets with immutable manifests.
- [ ] Run `loslab nr-observe` over those datasets on a supported Linux host;
  verify identical receiver configuration and recorded observation semantics.
- [ ] Run at least one classical method and two matched neural/feature controls
  on the same scene-group split; save per-sample predictions and uncertainty or
  explicitly mark it unimplemented for this MVP.
- [ ] Confirm that the held-out scene is excluded from training, validation,
  threshold selection, calibration, and model selection.
- [ ] Reproduce the workflow from a clean Linux environment with `uv sync`,
  native dependency setup, the exact commands, and output hashes recorded.
- [ ] Run the full portable checks and hosted CI on the final pushed commit.
- [ ] Review the generated report for correct simulation-only wording and retain
  weak/negative results without changing the locked evaluation.

The current single-scene smoke does **not** satisfy these exit criteria. The
remaining scientific gates in `MASTER_RESEARCH_PLAN.md`—independent measurements,
qualified materials, receiver impairments, broad standards conformance, model
novelty, forecasting, and downstream utility—remain beyond this MVP and are not
quietly treated as complete.
