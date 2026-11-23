# Research plan topic: architecture

This note is built from the archived master research plan at git tag
`docs/master-research-plan-archive`, file `docs/research/MASTER_RESEARCH_PLAN.md`. Each section below is copied
verbatim; the source line range is given before it. The archived plan is
not restored to the tree. Rebuild this note with
`python scripts/build_research_topics.py`.

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 990-1072 -->
## 1. Current-repository audit and exact technical debt

### 1.1 What already exists (retain and test)

| Existing component | Path | Reliable starting point | Limit / action |
|---|---|---|---|
| Provenance and atomic manifests | `artifacts/` | canonical identities, hashes, outputs | extend for physical schema, target version, calibration, scene asset SHA256, data licenses |
| Parameter loading | `configuration/`, `config/` | config canonicalization | introduce strict unit-aware schema and version compatibility; reject unknown keys |
| Complex channel dataset | `datasets/channel.py` | separate complex `h`, labels, groups, provenance | currently essentially `[N, flattened_links, F]`, binary `y`; redesign for `[N,T,Rx,Tx,Pol,Sym,Subcarrier]`, masks and multiple targets |
| Synthetic debug data | `datasets/synthetic.py` | fast smoke tests | **debug only**, never reported as research evidence |
| Sionna PHY/RT adapters | `integrations/sionna/adapter.py` | version-gated, profiles and RT direct-path oracle | native 2.2 not tested; validate all shapes, per-link label semantics, temporal evolution and phase reference |
| Urban XML/OBJ generation | `propagation/geometry.py`; `scenes/generated/research_scenes/` | reproducible procedural scenes | basic buildings/vehicles are **not** calibrated physical city models; fix geometry/material authenticity |
| CFR transformations | `propagation/physics.py` | Fourier sum and descriptive features | current IFFT PDP moments are *sample-bin observables*, not automatically physical path parameters; cyclic alias and pilot effects matter |
| Pilot approximation | `receivers/pilots.py` | debug LS/interpolation stress | not 3GPP DMRS/PRS; its *per-link received-power-relative noise* masks absolute link-budget consequences |
| MLP / RF / CORAL | `models/`, `learning/` | useful baselines | not a newly discovered algorithm; add precise model registries and matched-param/compute controls |
| Group/domain evaluation | `evaluation/`, `experiments/` | seeds, held-out domain, saved predictions | add nested scene/trajectory/hardware hierarchy, cross-domain test lock, real CI and statistical power |
| Archived 2025 work | `archive/legacy_2025/`, `notebooks/legacy_2025/` | historical reproduction | keep read-only; do not treat historical reported metrics as reproduced code output |
| CI and packaging | pyproject.toml, .github/workflows/ci.yml, uv.lock | CI version matrix runs CPU core tests; CPU-neural job owns coverage enforcement; local configured suite: 166 pass, 80.97% coverage | CI remains CPU-portable; hosted native RT integration awaits a compatible runner. |

### 1.2 Specific correctness hazards discovered by audit

**Critical—must fix before serious experiments:**

- **Mixed label semantics:** `ChannelDataset.y` remains a binary compatibility field, now accompanied by required `label_semantics` provenance. Standardized TDL/CDL, geometric RT, and synthetic debug generators identify their distinct meanings; selection preserves that meaning and concatenation rejects mixed semantics. Rich per-target masks/source/confidence and distinct observable-direct/blockage targets are still open. Never calculate one universal binary accuracy over incompatible semantics.
- **Timing and decomposition (adapter defects corrected; broader gate open):** `rt_cfr()` now explicitly uses `normalize_delays=False` and records the absolute-delay, unnormalized-power channel contract in generation-report schema v2. Pinned Sionna 2.2 source and a native 10 m free-space test verify the absolute delay and baseband-frequency phase slope. Direct components are now masked from the **same** `Paths` object using its validity and zero-interaction masks; the native two-ray fixture verifies coherent full/direct/indirect CFR conservation. Existing pre-correction datasets and model/DMRS results remain immutable legacy artifacts under their old normalized-delay convention; regenerate canonical artifacts and affected downstream comparisons before using them for absolute-delay or canonical-channel claims. A synchronized receiver-time observable, MIMO component partition, and broader path/gain reconstruction are still open.
- **Delay/power target contamination:** Sionna PHY TDL/CDL scaled `delay_spread` is a *generation parameter*. It must not be mistaken for the noisy receiver-estimated RMS delay spread. Distinguish `ds_generation_parameter_s`, `ds_path_oracle_s`, and `ds_receiver_estimate_s`.
- **Current geometry labeling:** depth-zero, LOS-only ray solve is a useful diagnostic, but test its behavior for penetration/refraction, antenna orientation, endpoints inside walls, thin obstacles, 3D intersections and boundary cases. “Valid direct solver path” is a **specific semantic**, not universal building opacity.
- **CFR shape:** `rt_cfr()` flattens tensor axes and keeps one time snapshot; flattening is scientifically hazardous for AoA/AoD, MIMO topology, per-antenna phase and future prediction. Keep named axes before model input.
- **No realistic measured noise model:** noise variance in `comb_ls_estimate()` is set relative to each sample’s received signal power. This yields a controlled relative-SNR experiment, **not** fixed transmit power/noise figure/thermal-noise model. Run both explicitly labelled conditions.
- **Dynamic predictions impossible from a single snapshot:** current generators take one PHY time step and one RT CFR step. Forecasting, Doppler, blockage-onset and temporal uncertainty require contiguous timestamped trajectories, not arbitrary shuffled positions or isolated links.
- **Receiver realism:** 128 subcarriers × 30 kHz = 3.84 MHz synthetic grid in `GridConfig`; not automatically the 2025 thesis’s 16 PRBs × 12 × 15 kHz = 2.88 MHz nominal active allocation. Explicitly record bandwidth, number of used REs, PRBs, FFT size, guard subcarriers, CP and actual time/frequency pilot positions; never quietly pretend these are the same experiment.
- **Class balancing/selection:** rejecting direct-only LOS links and sampling until `per_class` is achieved makes an intentionally conditioned benchmark; report retention/rejection, original base rates and reweighted deployment prevalence. Do not silently drop no-coverage links or near-zero-power NLOS as “invalid.”
- **Training/test separation:** current group IDs are helpful but do not prove multi-scene independence, avoid near-boundary spatial autocorrelation or prevent revisiting the same street trajectory under a different ID. Assign groups **at trajectory, block, scene, asset-family and measurement-session levels**, then audit hashes and distances.
- **Profile/geometry confounding:** when standardized A–C are NLOS and D–E LOS, profile recognition can trivially explain a high result. Audit profile-only oracle and hold out entire profile families; primary geometric target must be measured in RT/real domains.
- **Finite CFR vs physical paths:** discrete IFFT PDP peaks are resolution-limited and periodic with subcarrier spacing. Interpret path count and first delay only relative to bandwidth, sampling and windowing; preserve masks and reject ambiguous labels.
- **Confidence intervals:** resampling correlated rays as independent samples produces falsely tight intervals. The primary unit for external validity is **scene/site/session**, then **trajectory** within scene; explain the nested resampling hierarchy.
- **Stochastic solver reproducibility:** `PathSolver` seed alone is not enough for deterministic output; official 2.2 documentation requires `PathSolver(deterministic=True)` where exact reproducibility matters, then verify backend/device equivalence rather than presuming bitwise reproducibility.
- **MIMO comparability:** the adapter currently uses 1×1 panels for CDL in its standardized generator. New spatial claims require nontrivial arrays, polarization, orientation/element spacing and matched aperture across RT/CDL.
- **Source-control provenance:** compare exact code checkout, package versions and active repo contents; historical Aalto paths and upstream source clones mentioned in docs are not proof that those checkouts exist in a new runtime.

### 1.3 Baseline correction from original thesis

The 2025 thesis Table 6.3 reports **overall** LOS/NLOS classification percentages: MLP **75.67% SISO / 77.99% MIMO**; ResNet **95.15% / 96.93%**; VGG **92.66% / 94.44%**. These are **historical thesis reports** under that evaluation, not validated results of the upgraded repository. The claim that “CNN contributes little; MLP carries everything” is a valuable **ablation hypothesis**, not yet a demonstrated finding. Split the old network into convolutional encoder, dense fusion, downstream head, training conditions and input differences; reproduce each under controlled conditions.

### 1.4 Repository target refactor (additive, not a rewrite)

Maintain existing domain ownership and extend with:

```text
config/
  physics/ channel_targets.yaml, identifiability.yaml, units.yaml
  receivers/ nr_pusch_dmrs.yaml, ideal_csi.yaml, impairments.yaml
  datasets/ analytical.yaml, tr38901.yaml, rt_static.yaml, rt_dynamic.yaml, real.yaml
  scenes/ split_registry.yaml, materials.yaml, source_licenses.yaml
  models/ mlp.yaml, cvnn.yaml, paths.yaml, proposed.yaml
  experiments/ E00_*.yaml ... E19_*.yaml, paper_v1_frozen.yaml
integrations/sionna/
  version_gate.py  phy.py  rt.py  native_contracts.py  linklevel.py
propagation/
  ray_path_schema.py  units.py  labels.py  targets.py  interventions.py
receivers/
  nr_pusch.py  dmrs.py  synchronization.py  impairments.py  masks.py
models/
  real_mlp.py  real_cnn.py  complex_baseline.py
  path_set.py  physics_decoder.py  uncertainty.py  proposed.py
learning/
  losses.py  physical_consistency.py  equivariance.py  matching.py  training.py
forecast/
  dynamic_dataset.py  channel_forecast.py  blockage.py  beam.py
evaluation/
  independent_splits.py  inference_contract.py  prediction_schema.py
  stat_plan.py  calibration.py  ood.py  plots.py  tables.py
validation/
  invariants.py  oracle_tests.py  sionna_native.py  physics_checks.py
  leakage_scan.py  material_audit.py  scene_visual_qa.py
paper/
  preregistration.md  claims_matrix.md  results_index.md  citations.bib
```

No generated data should be committed merely for convenience. Publish a versioned *dataset specification* and dataset manifests; store bulky data by checksum/object store. Use an immutable `paper_v1` lock for the eventual reported results.

---

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 1189-1271 -->
## 4. Sionna RT 2.2 / Sionna PHY 2.2 implementation contract

**Verified against official Sionna 2.2 online docs as of 2026-10-21; native execution still required.** Treat code snippets below as illustrative interfaces until live integration tests pass. Version-lock the actual install and write a compatibility matrix rather than trusting `main` HEAD.

### 4.1 Version matrix and standards traceability

| Layer | Freeze requirement | Official reference | Gate |
|---|---|---|---|
| Python / Linux | Python ≥3.11; Ubuntu 24.04 strongly preferred, compatible driver + PyTorch | Sionna installation | import and GPU/CPU smoke |
| `sionna` | v2.2.0 **as initial pinned candidate**, then entire dependency lock | Sionna docs/2.2.0 | exact installed version and smoke |
| `sionna-rt` | record separate RT package build/commit, Mitsuba and Dr.Jit | Sionna RT 2.2 PathSolver/Paths | GPU native path contracts |
| PyTorch | ≥2.9 per official 2.2 install, tested exact CUDA/wheel/SM pair | Sionna installation / PyTorch selector | run tensor/device/complex kernels |
| 3GPP TR 38.901 | **standard document** v19.5.0 available Sep 2026; **Sionna table selector** `spec_version="19.2"` for available TDL/CDL | ETSI and Sionna model APIs | record both, do not equate them |
| 3GPP TS 38.211 / 38.212 / 38.214 | exact version/release for NR numerology, RS, coding and scheduling | ETSI specs + 2.2 PUSCH tutorial | gold-vector resource grid and decoder validation |
| Geometry / materials | Mitsuba-compatible XML/OBJ, material coefficients and mesh hashes | Sionna scene/material docs | visual render + physical checks |
| Experiments | exact code commit, config hash, scene CRC/SHA, model weights, dataset manifest | internal reproducibility contracts | no untracked experiment evidence |

**Release note:** the newest ETSI 38.901 directory discovered as of this review is `19.05.00_60`, but Sionna 2.2 model constructors currently advertise only `"16.1"` and `"19.2"` choices for the scalable TDL/CDL tables. A manuscript must say exactly which numeric table Sionna implements, and may separately cite the later document for context. Verify whether parameter values changed instead of silently using a newer document number.

### 4.2 Canonical RT physics interface

Official `sionna.rt.PathSolver` in 2.2 supports flags for `los`, `specular_reflection`, `diffuse_reflection`, `refraction`, `diffraction`, `edge_diffraction`, `diffraction_lit_region`, and controls such as `max_depth`, `samples_per_src`, `max_num_paths_per_src`, `synthetic_array`, `seed`. `PathSolver(deterministic=True)` is needed for the solver's documented reproducibility guarantee. **The default `diffuse_reflection=False`, `diffraction=False`; enabling a flag is not equivalent to a validated material model.**

**Canonical tests:**

- Direct-only clear free-space path: one valid physical direct mechanism, timing approximately geometric distance/`c`, expected free-space behavior within modeled antenna conventions.
- Direct + ground/wall specular: ensure multiple paths with correct delay ordering/geometry, and a **coherent** CFR sum of complex contributions.
- Pure obstruction: geometric direct oracle false, but indirect reflections/diffractions can still be present.
- Thin/penetrable material: geometric clear path may be false even if refracted/transmitted energy exists; keep labels distinct.
- Scattering: default ITU material scattering coefficient may be zero. For a diffuse test **explicitly set and record** scattering roughness/pattern/coefficients; enabling `diffuse_reflection=True` alone does not guarantee visible diffuse components.
- Diffraction: compare enabled/disabled edge effects and a wedge/corner oracle; first-order modeling is not universally valid for all obstacles/frequencies.
- Multi-bounce: increase `max_depth` until per-link path gain/moments converge to a declared tolerance for a stratified scene subset.
- Sampling: sweep `samples_per_src` and seeds; assess path coverage and Monte Carlo variance; do not confuse zero discovered rays with physical absence.
- Antennas: validate with `synthetic_array=True/False` for near-field or large-aperture cases. Synthetic arrays use planar phase approximations, not independently traced paths per antenna.
- Material coefficients and scene frequency: validate permittivity, conductivity, thickness, scattering, polarization and frequency limits; log every override.
- Native output: inspect `paths.valid`, `paths.interactions`, `paths.a`, `paths.tau`, `paths.doppler`, `paths.theta_r/phi_r/theta_t/phi_t` shapes and dtypes with SISO **and MIMO**; respect the result masks rather than treating padded entries as zero-power valid paths.
- CIR/CFR contract: compare `paths.cfr(..., normalize_delays=False, normalize=False, out_type="numpy")` and manually summed `paths.cir(...)` using the correct baseband convention. Some interfaces return real/imag components separately: introspect actual return signature instead of treating tuple handling as proof of correctness.
- Phase reference: test that full/direct/indirect channel comparisons use the same **absolute** delay and carrier reference. Never subtract independently first-path-normalized CFRs to infer a physically meaningful indirect channel without alignment.
- Direction reciprocity: confirm `reverse_direction` + orientation/polarization and antenna indexing when using a downlink ray trace to emulate an uplink physical channel.

### 4.3 Paths APIs that matter (Sionna 2.2)

- `paths.cir(sampling_frequency=..., num_time_steps=..., normalize_delays=..., out_type=...)`: amplitudes and delays; frequency-dependent carrier conversion is internally defined.
- `paths.cfr(frequencies=..., num_time_steps=..., normalize_delays=..., normalize=..., out_type=...)`: per-frequency complex CFR from paths; **frequencies here are baseband offsets in Hz**.
- `paths.taps(bandwidth=..., l_min=..., l_max=..., normalize_delays=..., out_type=...)`: finite-bandwidth discrete taps with a documented sinc-shaped interpolation assumption; compare this with pilot-estimator output, not raw geometric delta functions directly.
- `paths.interactions`, `paths.valid`, `paths.tau`, `paths.doppler`, `paths.a` and angular fields provide **private oracle supervision/audits**. Interaction ordering, masks and dimensionality must be explicitly checked.
- `RadioMapSolver` produces coverage/path-gain maps and can sample receiver locations, but its path-gain averaging/non-coherent map semantics are **not** equivalent to a coherent instantaneous CFR; do not use a radio map as an I/Q training target.

### 4.4 Critical time-evolution distinction

From the 2.2 Mobility tutorial: **`Paths.cir/cfr(num_time_steps>1)` with fixed geometry and path Dopplers is suitable for short movement segments** (the tutorial discusses displacement not exceeding a few wavelengths), **not** long-term occlusion or appearance/disappearance of new paths. For event prediction, move the UE/vehicles/people over timestamps and **re-run `PathSolver` at appropriately sampled positions**, then reconcile path identity/appearance across snapshots. Use the Doppler shortcut *only inside a separately validated coherence interval*.

### 4.5 Correct receiver configuration

Sionna PHY 2.2 includes NR `PUSCHConfig`, `PUSCHDMRSConfig`, `PUSCHPilotPattern`, `PUSCHLSChannelEstimator`, `PUSCHLMMSEChannelEstimator`, `PUSCHTransmitter`, and `PUSCHReceiver`. This supports an initial **PUSCH-DMRS** route to actual 5G NR pilot observations, not an unrestricted claim of full 5G standard coverage. Downlink PRS and positioning-specific requirements **must be independently checked** against TS 38.211 and available library coverage; implement/reference-test anything absent rather than relabeling a generic comb as PRS.

**Reference starting profile from original thesis:** 3.5 GHz, 15 kHz SCS, 16 PRBs, 14 OFDM symbols per slot, DMRS type 1, configured DMRS symbols [2, 11], MIMO as documented in the thesis. **Reproduce before changing**, checking any indexing/conformance difference. All additional grid/BW/MIMO cases must be named as *new 2026 experimental configurations*.

**NR receiver acceptance sequence:**

1. Serialize TX grid and DMRS masks; verify symbol indices, RE occupancy, CDM groups, scrambling IDs and pilot energy against a trusted Sionna output and relevant specification.
2. Generate OFDM waveform, channel effects, AWGN and optionally RF hardware impairments; identify whether operating at time-domain samples, exact OFDM CFR or linear frequency-domain approximation.
3. Recover pilot-domain LS CSI, LMMSE CSI and error variances; preserve a pilot mask and time/frequency coordinates.
4. Verify ideal vs estimated CSI NMSE as SNR grows, interpolation endpoints, CP length, subcarrier ordering, pilot power and antenna/stream isolation.
5. Add residual CFO, phase noise/CPE, timing-offset, ADC quantization, channel aging, interference and noisy AGC **one at a time**, then factorial stress subset.
6. For absolute timing, define synchronization acquisition, bias, clock offset, calibration and clock reference; if absent, disable absolute ToA task in that mode.
7. Run a known receiver decoding test and report EVM/BER/BLER to demonstrate that the synthetic observations correspond to a plausible communication link.
8. Guarantee models never receive `H_true`, `paths.tau` or receiver impairment parameters that are **not observable or known at inference**.

### 4.6 Architecture of execution environments

Create **separate** dependency groups and CI jobs:

- `unit-cpu`: pure Python numerical/ML schema/tests; no Sionna import; fast PR gate.
- `phy-cpu`: pinned PyTorch + Sionna PHY TDL/CDL and NR resource-grid tests on a supported CPU runtime.
- `rt-native`: pinned Sionna RT, Mitsuba, Dr.Jit + supported GPU/driver or validated LLVM backend; scene/path integration and deterministic replay.
- `research-gpu`: full train/inference runs and profiling, pinned CUDA/cuDNN/package state.

Do **not** assume an NVIDIA V100 supports any arbitrary newest PyTorch CUDA wheel. Verify compute capability, wheel, driver, CUDA runtime, Dr.Jit supported target and a native kernel test; PyTorch working does not prove RT GPU kernels work. WSL2, Triton/HPC and personal laptop each need their own checked environment manifest.

Required environment doctor JSON: Python, platform/kernel, CUDA device/compute capability/VRAM, `torch.__version__`, CUDA runtime, driver, `sionna` + `sionna-rt` versions, `mitsuba`, `drjit`, active Mitsuba variant, CPU/GPU code paths, deterministic flag, ability to load scene, direct-ray test, and a checked hash of both 3GPP/profile resources and source code.

---

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 2344-2416 -->
## Appendix B. Example configuration and command contracts (designed, not currently implemented)

Do **not** paste the following YAML into an existing runner without writing and validating the schema; they are *proposed target interfaces*. Unknown fields, physical unit omissions and inconsistent array shapes should raise errors.

```yaml
schema_version: "2.0-proposed"
experiment_id: "E08_scene_ood_static"
status: "design_only"
randomness:
  scene_seed: 43001
  propagation_seed: 43002
  receiver_seed: 43003
  model_seed: 43004
physical:
  carrier_hz: 3500000000.0
  tr38901_document_version: "19.5.0"
  sionna_profile_selector: "19.2"
  path_delay_origin: "geometric_absolute_oracle_only"
  power_normalization: false
  rt:
    engine: "sionna-rt"
    version: "2.2.0"
    deterministic_solver: true
    max_depth: null  # choose and validate in convergence preflight
    samples_per_src: null  # choose and validate in convergence preflight
    mechanisms:
      direct: true
      specular: true
      diffuse: true
      refraction: true
      diffraction: true
receiver:
  scheme: "NR_PUSCH_DMRS"
  carrier_hz: 3500000000.0
  subcarrier_spacing_hz: 15000.0
  allocated_prbs: 16
  ofdm_symbols_per_slot: 14
  dmrs_type: 1
  dmrs_symbol_positions: [2, 11]  # hypothesis from legacy setup; validate indexing
  observed_quantity: "H_hat_from_LS_with_mask"
  absolute_time_reference_available: false
  normalization: "train_statistics_only"
labels:
  geometric_los: "ray_segment_oracle_with_unknown"
  observable_direct: "receiver_threshold_versioned"
  rms_delay_spread: "path_moment_relative_excess"
  first_path_delay: "relative_resolvable_only"
  path_count: "resolvable_count_with_validity_mask"
splits:
  policy: "held_out_scene_and_asset_family"
  target_test_access: "locked"
  temporal_disjointness: true
  intervention_pairs_same_partition: true
model:
  model_id: "registered_baseline_or_proposed"
  oracle_input_forbidden: true
  input_contract: "receiver_observation_v2"
evaluation:
  unit_of_inference: "site_scene"
  report_unknown_label_fraction: true
  primary_metrics_pre_registered: false
outputs:
  artifact_root: "outputs/research_v2/"
  immutable_manifest: true
```

**Deliberate unresolved choices:** `max_depth`, `samples_per_src`, calibration policy and primary metrics remain null/false until numerical pilot checks. Filling defaults merely to produce a runnable YAML before physical validation would create false confidence.

### 12.1 Proposed repeatable CLI contract

The **existing** CLI should not be assumed to implement these names. Use this as the design target:

```bash

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 2464-2511 -->
## Appendix C. Computational plan, storage, caching and failure recovery

### 13.1 Hardware/environment tiers

| Tier | Suitable work | Hard constraints |
|---|---|---|
| Laptop CPU/limited VRAM | schemas, unit tests, analytic toy channels, numerical signal-processing sanity, small MLP | no implicit claim to full Sionna RT GPU speed; bound memory and use small batches |
| Workstation modern NVIDIA GPU | native RT/PHY debugging, modest dataset generation, model training | validate current Dr.Jit/Mitsuba backend and correct PyTorch CUDA build, separate runtime manifests |
| HPC GPU node (including older cards) | large but well-defined physical sampling / multi-seed training | perform a **small native RT kernel test first**; compatibility/driver cannot be inferred from `nvidia-smi` alone |
| CPU LLVM backend | native RT correctness fallback when supported | document approximate throughput and numerical backend differences; don't rely on GPU-only code paths |

Specific machines, GPU quotas and node types are **not guaranteed** by this plan. If hardware lacks required RT kernels, change runtime or use validated CPU/other backend; do not downgrade code correctness checks or fabricate Ray Tracing outputs.

### 13.2 Scale estimation before production

For complex CSI stored as `complex64`, **raw data bytes** (excluding targets/masks/metadata) per sample approximately equal:

\[
8\times T \times N_{\rm Rx} \times N_{\rm TxStreams}
\times N_{\rm Symbols} \times N_{\rm Subcarriers}.
\]

Example **storage arithmetic only, not chosen experiment settings**: `T=10, Rx=4, TxStreams=1, symbols=14, subcarriers=192` requires `8×10×4×1×14×192 = 860,160 bytes ≈ 0.82 MiB` per stored dense complex sample; **one million such samples ≈ 0.86 TB** before masks, labels, path or waveforms. Therefore avoid storing dense full-resolution CSI for every stochastic realization unless essential.

- Use sequence/window subset and sparse pilot storage when valid; preserve a manageable number of full-oracle validation records.
- Precompute immutable physical ray oracle datasets once; derive several receiver SNR/impairment views with deterministic child seeds and parent IDs. Never put different noise realizations of the same ray geometry in disjoint splits.
- Precompute analytic quantities that can be reused with immutable labeling definitions; invalidate caches when changing antenna/frequency/material/phase conventions.
- Divide RT generation by **scene-level shards**, not random ray samples. Prevent temporal trajectory fragments from crossing worker shards with different split assignments.
- Record RT solver runtime/memory, path yield/failure rate, average output bytes and shard count on a pilot set; calculate planned disk hours and cost before launch.
- Keep a write-ahead manifest: shard `planned → started → verified → committed`, with content hash/atomic rename. Detect and quarantine partial/crashed shards.
- Do not let auto-resume skip failed corner cases. Explicitly track `FAILED`, `RETRY_LIMIT`, `UNSUPPORTED`, `ZERO_COVERAGE` and permit statistical reporting of sampling bias.

### 13.3 Runtime and ML resource budget

Use a *measurement-driven* budget, not invented GPU-hour promises:

1. Benchmark SISO and MIMO native RT on 10–100 representative **distinct scene/link conditions** with varying path budgets and antennas.
2. Measure median, p90, p99 trace time, GPU memory, valid-path count, artifact bytes and failure frequency; extrapolate with 2×–3× headroom rather than assuming linear scaling.
3. Measure receiver OFDM generation/decoding and preprocessing costs separately.
4. Measure MLP, real CNN, CVNN and structured-path models at three sizes on identical input tensors; log effective throughput and power if available.
5. Fix exploration budget per family; stop only on preregistered resource/futility criteria or documented physical-invalid cases—not because the result is unfavorable.

### 13.4 Reproducibility and reproducible *non*-determinism

Capture exact code tree SHA, package wheels and versions, CUDA driver/runtime, accelerator, solver backend, scene/material version, `deterministic=True` where supported, seeds for geometry/RT/receiver/model/augmentation, preprocessing transform, target schema, split ID, model weights, optimizer/scheduler, random state and all failed resume attempts. Exact bitwise agreement across GPU architectures is not promised: define **numerical and statistical** tolerances and save representative native gold outputs.

---
