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
  88.12% coverage; 21 focused tests passed; Ruff lint/format, mypy (139 files),
  lock check, package build and diff checks pass. Hosted CI and canonical
  dataset regeneration remain pending.
