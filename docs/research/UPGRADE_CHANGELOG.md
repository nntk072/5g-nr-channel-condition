# Research upgrade changelog — 2026-10-21

## Inherited from original materials

- Kept **all** 2025 thesis notebooks, original modules, figures and the original
  Tampere thesis PDF unchanged in the full codebase archive.
- Reviewed the user's separate Aalto_MS_Thesis engineering implementation;
  adapted its canonical hashes, atomic manifests, matched-seed comparison and
  independent verification discipline. Did **not** copy irrelevant RL trading
  models, reward functions, datasets or experiment results.

## Added (new 2026 research package)

- `loslab/dataset.py`: validated channel datasets, no-pickle NPZ, group ID,
  domain and explicit label semantic provenance.
- `loslab/physics.py`: exact path-to-frequency Fourier sum, channel features
  and diagnostic receiver estimation error.
- `loslab/pilots.py`: QPSK comb pilot/LS-interpolated diagnostic observables.
- `loslab/sionna2.py`: Sionna 2.2 TDL/CDL and RT adapters, geometry oracle
  and direct + indirect path propagation with power decomposition.
- `loslab/geometry.py`: configurable 3D city generator, materials, vehicles,
  closed meshes, geo-block sampling exclusion and scene manifests.
- `loslab/models.py`: standardized classical/RF and MLP baselines,
  equal-backbone CORAL regularization and checkpoint serialization.
- `loslab/evaluation.py`: group- and domain-disjoint splits, calibrated
  threshold from validation, AUROC/AUPRC/Brier/ECE and block-bootstrap CI.
- `loslab/statistics.py`: matched-seed tests and paired mean deltas.
- `loslab/provenance.py`: environment versions, file/config/source hashes,
  atomic result recording.
- `loslab/experiment.py` and `loslab/__main__.py`: reproducible experiment
  driver/CLI; every result includes split indices, model and probabilities.
- `tests_research/`: 47 lightweight tests including mocked RT contract and
  actual end-to-end synthetic debug model fit/checkpoint reload.
- `examples/research_scenes/`: three sample controlled urban scenes (XML/OBJ).
- `docs/research/`: quickstart, legacy audit, research program and changelog.

## Explicitly not completed

- Native `sionna==2.2.0` import/execution of realistic 3D propagation on the
  target machine (Dr.Jit/Mitsuba and CUDA environment are unavailable here).
- Physically validated rich urban/factory/campus assets; calibrated RF materials.
- 5G NR TS 38.211-compliant receiver/PUSCH DMRS/PRS signal processing.
- New, demonstrably original machine-learning method (CORAL is established).
- Real measured channels, official measurements, full experimental statistics
  and top-tier peer review. No real-domain accuracy claims are included.

## Physics-contract correction — 2026-10-22

- `integrations/sionna/adapter.py::rt_cfr` now preserves absolute Sionna path
  delays (`normalize_delays=False`) and unnormalized channel power
  (`normalize=False`). Pinned Sionna 2.2 source showed the previous delay
  normalization also changed the baseband carrier-phase reference and erased
  absolute delay slope.
- Direct and indirect components are now coefficient-masked from the same
  `Paths` object using valid-path and zero-interaction masks. A depth-zero solve
  still supplies the geometric label and cross-check; its CFR is no longer
  subtracted from the full-solve CFR.
- RT generation-report schema is now `loslab.raytraced_generation_report.v2`
  and declares channel semantics as `loslab.rt_absolute_channel.v1`, including
  absolute delay reference, unnormalized power, frequency reference, phase
  convention, and same-path partition.
- Existing generated NPZs, observations, models and reports were left unchanged.
  Their recorded source hashes identify the earlier normalized-delay contract;
  regenerate them after the physics gates pass before making canonical absolute-
  delay or new downstream-comparison claims.
- Focused mock and native LLVM tests verify adapter parity with the pinned CFR,
  delay-normalization sensitivity, same-path SISO and explicit 2×3 MIMO
  conservation, and generation integration. Locally: 481 passed, 3 skipped,
  88.12% coverage; focused tests passed; Ruff lint/format, mypy (139 files),
  lock check, package build and diff checks pass. Hosted CI run 39710231241
  passed all four jobs for commit 2b03354. Canonical dataset regeneration
  remains pending.

## Phase 0 follow-up — 2026-10-22

- Native LLVM RT-to-NR tests now cover even and odd FFT sizes (24 and 25) and
  compare bin coordinates and CFR against the explicit CIR/delay Fourier sum.
- A native two-path regression reconstructs CFR independently from raw
  `Paths.a` and `Paths.tau`, including the carrier-phase conversion, and checks
  it against the same-path full CFR within tolerance for pinned single-
  precision path coefficients.
- All 21 focused adapter/mock/native integration tests pass locally, as do
  Ruff lint/format and mypy. Hosted CI run 39711191556 passes all four software
  jobs for commit 00928a3; the hosted matrix does not install Sionna RT, so the
  native cases remain locally verified. Canonical dataset regeneration remains
  gated on the remaining physics checks.

## Friis power check — 2026-10-22

- The native 10 m isotropic direct-path test now checks the coefficient
  magnitude against `λ/(4πd)` and phase against the pinned Friis convention,
  plus the existing `τ=d/c` assertion.
- All 21 focused adapter/mock/native integration tests, Ruff lint/format, and
  mypy pass locally. The newly added assertion is pending hosted CI; broader
  receiver link-budget, polarization, and material qualification remain open.
