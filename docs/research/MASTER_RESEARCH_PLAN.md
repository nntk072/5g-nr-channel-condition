# MASTER RESEARCH & ENGINEERING PLAN
## Physically Grounded, Identifiability-Aware Multipath State Learning for 5G NR
### From LOS/NLOS classification to channel-state reconstruction, uncertainty, and forecasting

**Document edition:** 3.11 — 2026-10-21 (paired multi-delay CP sweep with Holm correction)
**Status:** execution specification and scientific decision record; **not** a completed study, not experimental evidence, and not a claim of top-tier acceptance.
**Repository inspected:** live checkout /home/nguyenl37/5g-nr-channel-condition, branch main, initial implementation commit ffe199c80c1caf40dea91a406fe02124b3ff05ec (built from baseline cc43a5544036578a6bb57ce52b66436d8acb6c1b); local reference /home/nguyenl37/sionna at Sionna v2.2.0, commit 2f0a3820269b53fd022780a6f2f5f720f7ceb9b4, RT submodule commit 15b5ee036917a4c2a9b6420e05570b77717e2fcc.
**Legacy work:** Long Nguyen, *5G Physical Layer Channel Conditions Classification Using Machine Learning Techniques*, Tampere University, July 2025, embedded in `docs/thesis/NguyenLong.pdf`.
**Scope:** living research and engineering execution specification. Recent revisions add a normalized local Fisher-information diagnostic for joint complex path-gain and excess-delay parameters, bounded CP-based integer timing acquisition/correction for a known-origin capture, and reproducible frequency-domain and sample-domain TDL-A PUSCH BLER curves, plus a paired 3-versus-30-m/s receiver stress comparison. The new waveform link includes time-varying fading, sampled channel memory, and the resulting ICI/ISI within a CP-bounded configuration. Prior revisions add a reproducible three-scene leave-one-scene-out RT pilot and explicit class-balanced scene-group audit to residual CFO acquisition/correction and a two-tap sample-domain PUSCH decoder smoke test. Earlier revisions added deterministic CPU/LLVM RT generation, controlled RT material sensitivity, GPU PUSCH receiver, CPU/LLVM RT-to-NR bridge, content-addressed records, leakage-audited splits, analytical fixtures, and TDL-to-PUSCH generation. These remain bounded simulation and implementation checks; they establish neither model advantage nor standards-complete synchronization or realistic NR BLER curves.
**Verification performed during this revision:** the full pinned Sionna/CUDA test suite passes at 86.80% coverage. Ruff lint and format checks pass across the maintained source, test, and script directories (104 Python files); mypy passes on 62 source files. With CUDA hidden, the complete CPU suite passes at 76.91% coverage against the 75% gate. A repository-root Ruff scan also reaches archived 2025 code and legacy notebooks, which are outside the maintained lint scope. `uv lock --check` and `uv build` pass. GitHub Actions run 39583322560 passed all jobs for code commit af2d337; run 39583778271 passed for the first artifact commit; run 39584231041 passed after the Holm test additions. The GitHub CPU version matrix omits optional PyTorch and therefore runs core tests without applying the coverage threshold; the dedicated CPU-neural job installs PyTorch and enforces pytest's configured coverage threshold. The project-owned PUSCH DMRS LS adapter runs on CUDA, recovers a known flat complex channel over the estimated grid (maximum error approximately 4.3e-8), and lowers estimation NMSE as AWGN variance decreases. A native PUSCH transmitter → RayleighBlockFading → OFDMChannel → DMRS estimator loop recovers Sionna's channel truth without noise and has the expected AWGN error trend. Native direct-path and LOS-plus-specular-reflection RT scenes map to the NR grid on LLVM CPU and match absolute-delay CIR frequency responses, including coherent two-ray summation. A controlled simple-reflector check changes the surface from matched εr=1 to εr=4, observes one versus two valid paths and a changed NR-grid CFR; this validates material sensitivity in the pinned synthetic geometry, not material realism or general solver accuracy. NR records round-trip complex CSI, variance, masks, targets and provenance through pickle-free content-addressed NPZ files; scene-group split assignments are deterministic, audited and immutable once locked. A projected-coordinate audit reports cross-split neighbors inside a meter buffer and requires exact sample-set coverage. CFR fingerprints report cross-split near-duplicates invariant to common complex gain/phase, and trajectory windows report positive-duration cross-split overlap; the fingerprint utility is intentionally pilot-scale and quadratic. Seeded 1–3 path analytical fixtures reproduce the closed-form CFR and explicitly separate receiver inputs from synthetic path oracle. The Sionna v2.2.0 TDL adapter produces deterministic batched CUDA tap-to-CFR outputs for pinned profile-table selector 19.2, matches the explicit frequency-response equation for TDL-A and TDL-D, and restores the shared Sionna RNG state. Native batches of 4,096 realizations for each TDL-A–E profile match Sionna's nominal path delays and mean tap powers (10% Monte Carlo tolerance); this is implementation consistency, not a full standards certification. The native TDL→PUSCH batch path generates transmitter grids, TDL channel truth, AWGN observations, and per-sample NRChannelRecords with truth excluded from model inputs; the native test round-trips a GPU-generated target record through hash-verified pickle-free storage. Edition 3.2's CP correlation estimates per-waveform residual CFO (within its stated unambiguous range) and the inverse rotation restores injected PUSCH OFDM grids; a static two-tap sample-domain PUSCH path decodes at high SNR and exhibits the expected BER/BLER degradation at higher AWGN. This is a receiver smoke test, not a mobile/fading channel model or statistically meaningful BLER curve. Edition 3.5 acquires and corrects a bounded nonnegative integer sample delay on a known-origin PUSCH capture using CP repetition, validated on deterministic batched OFDM and the native Sionna PUSCH waveform; it is not blind NR frame synchronization or fractional timing recovery. Edition 3.3's three-scene leave-one-scene-out pilot, metrics and limitations are recorded below in P5. Edition 3.7 adds a native Sionna `TimeChannel` plus waveform-domain `PUSCHReceiver` path and a 1,024-codeword-per-SNR TDL-A curve with matched bits/channel/noise, sample-domain channel truth, and a checked 13-sample channel memory equal to the minimum CP. The sampled waveform mean power is recorded for SNR provenance. This remains a simulation with ideal synchronization and one CP-bounded link configuration. The Fisher diagnostic uses centered frequencies and delay coordinates scaled by inverse frequency span; its CRLB is returned in physical delay units and withheld for rank-deficient models. Tests cover rank deficiency, close-path conditioning, noise-variance scaling, invalid inputs, and agreement with the closed-form single-path delay CRLB for an unknown complex gain. These checks validate local formulas and numerical behavior, not empirical model performance or delay resolution. Both TDL-A PUSCH curves are recorded in section P5 with raw artifact hashes and confidence intervals. Production-scale diverse RT data, full TDL/CDL distributional conformance, blind/fractional timing synchronization, broader sample-domain link configurations, realistic faded-link decoding curves, and scientific model behavior remain open. Sionna RT CUDA initialization fails in WSL because working OptiX symbols are unavailable. These are software/physics checks, not research results.
**Scientific objective:** build a model that infers what is **observable and physically defensible** about a wireless channel, predicts what changes next, and knows when uncertainty is unavoidable.

> **Non-negotiable integrity rule:** A perfect result cannot be guaranteed. This plan targets *maximally defensible and reproducible evidence*. No expected accuracy, error reduction, “S-tier” result, or simulation-to-real success is to be invented, assumed, or reverse-engineered from desired conclusions. The project must support publishing an honest negative result or reducing the paper scope when a hypothesis fails.

---

## 0. Executive decisions — read before implementing

1. **Paper proposition:** *Identifiability-Aware, Symmetry-Consistent Multipath State Estimation and Forecasting from Sparse Complex 5G NR Observations*. The **paper claim is conditional** on experimentally establishing a novel advantage. Do **not** claim the name or general architecture is automatically new.
2. **Core scientific method, to be invented and tested:** a **receiver-observation-only**, symmetry-controlled latent multipath representation with a measurement-consistency decoder, physical/statistical consistency constraints, and **observable-information-aware uncertainty**. Extend to counterfactual/forecasting only if a scientifically faithful intervention or time-resolved dynamic dataset can be built.
3. **Primary result targets:** (i) accurate **recoverable multipath channel state** from sparse and noisy pilots; (ii) **calibrated uncertainty/identifiability** under difficult bandwidth, SNR, timing, and array regimes; and (iii) **transfer to independent RT scenes and preferably measured 5G channels**. LOS/NLOS is one secondary head and a diagnostic.
4. **Primary baseline:** robust parameter-matched **MLP** and analytical signal-processing estimators. The 2025 thesis reported CNN/ResNet performance above the original MLP for LOS/NLOS on its particular evaluation; a retrospective feeling that “the MLP did the work” must be tested with a *matched-input, matched-budget module ablation*, not assumed true.
5. **Data hierarchy:** (a) analytical sanity channels, (b) 3GPP TDL/CDL-A–E and spatially consistent UMi/UMa/RMa/InH/InF, (c) controlled Sionna RT scenes, (d) varied independent high-fidelity RT scenes, (e) held-out real measurements *where legally and technically possible*. **Never conflate profile A/B/C vs D/E with geometric LOS truth.**
6. **Observation hierarchy:** ideal simulated path oracle is for **targets and auditing only**, ideal complex CFR is an **upper-bound input**, imperfect OFDM/pilot CSI is the **main input**, and recorded I/Q or calibrated measured CSI is the **independent external test**. Never feed RT `Paths` metadata, position, profile IDs, scene name or measured ground-truth propagation labels into a supposedly receiver-only model.
7. **Execution order:** correctness gates → physically credible data → matched baselines → one new mechanism → independent transfer → dynamic/forecasting extension → downstream 5G impact → paper. Do not expand to all tasks before validating a minimal hypothesis.
8. **Venue strategy:** wireless communications first (e.g., IEEE TWC or a strong ICC/Globecom contribution), general ML venue only if the new learning principle generalizes beyond the wireless application. *“S tier” is not a universal formal rating.*

### 0.1 The single main claim to pre-register

> **H-main:** Under bandwidth-, SNR-, antenna-, and domain-limited observations, the proposed identifiability-aware physics-consistency method achieves better **joint estimation quality and uncertainty reliability on unseen environments** than classical estimators and parameter-/compute-matched MLP/CNN/CVNN baselines, **without relying on privileged simulation labels at inference**.

**Primary endpoint (pre-register after pilot studies):** a *hierarchical evaluation* based on held-out **scene-level channel reconstruction NMSE** as a measurable common output **plus** an identifiability-aware uncertainty metric (coverage under prespecified predictive sets/intervals), with a **prespecified small set** of associated physical-parameter endpoints (e.g., excess-delay-spread MAE and resolvable first-path delay MAE). Multiplicity correction required; do not merge unlike metrics into an arbitrary score. If reconstructing masked/held-out observations is too easy or not physically diagnostic, promote cross-environment parameter error to primary **before the test is unlocked**, documenting the change.

**Gate for a genuinely new method:** propose a mechanism that has testable mathematical properties beyond known complex convolutions, equivariance, CORAL, ordinary multitask heads, or generic path-token transformers; provide a theorem/proposition where possible, rigorous simulation falsification tests otherwise, and compare a same-backbone model **with that mechanism off**.

### 0.2 Live WSL status and environment-specific execution decision

This live snapshot supersedes older statements that the GPU or native Sionna is unavailable.

| Item | Verified state | Consequence |
|---|---|---|
| Project | /home/nguyenl37/5g-nr-channel-condition, initial receiver and plan commit ffe199c80c1caf40dea91a406fe02124b3ff05ec | This checkout supersedes the uploaded ZIP and old directory name. |
| Quality gates | Local dev: 165 passed, 7 optional Sionna/RT skips, 79.14% coverage; Ruff lint/format (96-file configured source/test set), mypy (59 source files), lock check and package build pass; full configured Sionna/CUDA suite: 185 passed | CPU matrix runs core tests without an optional-PyTorch coverage threshold; neural job installs CPU PyTorch and enforces configured coverage. Native RT remains local-only. |
| Sionna reference | `/home/nguyenl37/sionna` tag v2.2.0, commit `2f0a3820269b53fd022780a6f2f5f720f7ceb9b4`; initialized RT submodule v2.2.0, commit `15b5ee036917a4c2a9b6420e05570b77717e2fcc` | Both checkouts were clean. Pin source review and experiments to these revisions; keep the clone external. |
| GPU | WSL2 NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU, capability 12.0, about 6 GiB VRAM; driver 596.58, CUDA 13.2 | Suitable for PHY/model prototypes and small batches. Profile memory and stream data before scaling. |
| Isolated GPU environment | `/home/nguyenl37/.venvs/sionna220-gpu`, Python 3.12.3, PyTorch 2.11.0+cu128, Sionna and RT built from the pinned source | The project `.venv` remains CPU/development-only. |
| PHY smoke | CUDA flat-channel PUSCH DMRS golden vector, controlled AWGN trend, transmitter → RayleighBlockFading → OFDMChannel → estimator loop, frequency-grid and two-tap sample-domain decoder trends, OFDM round trip, CP-based CFO acquisition/correction and bounded known-origin integer timing acquisition/correction pass | Confirms only these NR PHY slices. CP CFO assumes a known frame boundary and static offset; CP timing assumes a first-symbol capture with a bounded nonnegative delay. Blind/fractional synchronization and statistically meaningful faded waveform BLER remain open. |
| RT smoke | Upstream `test_los_friis_sanity` and project-native direct/reflection checks pass using Mitsuba LLVM CPU. Two built-in scenes generated 40 geometry-labeled links each, and a seven-seed box-to-canyon baseline pilot completed. CUDA variant fails during Mitsuba initialization: WSL provides an OptiX loader stub but lacks `optixQueryFunctionTable`. | CPU/LLVM is viable for bounded physical-data pilots; it is not evidence of scalable RT throughput or material fidelity. Keep CUDA PHY/training separate. `torch.cuda.is_available()` does not prove Sionna RT is GPU-ready. |
| Migration residue | Cache-only tests_research/ and stale los_channel_thesis.egg-info / wireless_channel_condition.egg-info were inventoried and removed; current package egg-info remains build-generated | P0 cleanup residue is complete; preserve archive/legacy_2025/ and notebooks. |

The upstream WSL OptiX workaround requires extracting matching driver files and copying components into the protected Windows WSL system library directory, followed by WSL shutdown. Do not make that host-wide driver change as routine setup. Current plan: CUDA PHY/model work, small CPU/LLVM RT physical checks, CPU CI, and optional GPU tests. No WSL RT throughput claim or massive GPU RT campaign until native RT GPU execution is verified. For scale/performance claims, use validated native Linux/OptiX infrastructure or revise the data strategy.

**Local source evidence:** PHY contracts are in `/home/nguyenl37/sionna/src/sionna/phy/nr/pusch_channel_estimation.py`, `pusch_transmitter.py`, and `pusch_dmrs_config.py`; profile version selection is in `src/sionna/phy/channel/tr38901/tdl.py` and `cdl.py` (default table selector `19.2`). RT controls are in `ext/sionna-rt/src/sionna/rt/path_solvers/path_solver.py`; `path_solvers/paths.py` defines CIR/CFR/taps, whose `normalize_delays` defaults true. The upstream PUSCH test reference is `test/unit/nr/test_channel_estimation.py`; the upstream RT test reference is `test/unit/test_path_solver.py::test_los_friis_sanity`.

### 0.3 Revised implementation state and next gates

| Gate | Current state | Next evidence |
|---|---|---|
| P0 repo baseline/cleanup | Core gates pass; cache-only tests_research/ and stale egg-info removed; research plan, clean CI and packaging are maintained | Keep implementation revisions committed and pushed; preserve archival thesis assets unchanged. |
| P1 CUDA PHY | PUSCH/Rayleigh, TDL→PUSCH, flat-channel DMRS and impairment/decoder sanity tests pass in the pinned CUDA environment | Record reproducible environment/runtime recipe and peak-memory profile; broaden receiver configs and waveform BER/BLER. |
| P1 RT physics | Direct/two-ray/material checks and deterministic geometry-label smoke pass on LLVM CPU; two built-in scenes generated pilot data. RT CUDA remains unavailable under current WSL OptiX setup. | Add blocked/outage, diffuse multipath, solver convergence, path-overlay, material qualification and trajectory gold cases; no GPU RT performance claim. |
| P2 NR receiver | SionnaPUSCHLSReceiver and framework-independent NR contracts pass; CUDA tests cover flat DMRS recovery, AWGN trend, Rayleigh loop, OFDM round-trip, CP CFO and bounded known-origin integer timing acquisition/correction, timing-impairment sensitivity, frequency-grid BER/BLER ordering, static two-tap sample-domain checks, sample-domain TDL-A curves, paired mobility and DMRS comparisons, and a fixed-payload matched multi-delay CP sweep over 1-20 samples at one allocation/SNR | Blind/fractional synchronization, broader CP-margin coverage, antenna/numerology breadth, and broader fading/receiver configurations remain needed before receiver-capability claims. Extend DMRS evidence beyond the single TDL-A point before general receiver recommendations. |
| P3 RT-to-NR and observation schema | Paths.cfr-to-OFDM-grid bridge, hash-addressed NR records, immutable scene-group splits, spatial/CFR/temporal leakage audits, native GPU TDL→PUSCH records, and small CPU RT-generated NR-grid records pass | Broaden schema to multi-target/masked trajectories; validate audits on larger diverse scenes and freeze pre-registered independent-scene splits. |
| P4 analytical / standardized tier | Seeded 1–3 path fixtures, finite-bandwidth steering and joint Fisher diagnostics; native Sionna v2.2.0 TDL→CFR/PUSCH A–E checks; TDL-A tail smoke; reproducible TDL-A PUSCH BER/BLER curve; residual synchronization and decoder sanity tests | Extend analytic nuisance sweeps; validate TDL/CDL distributions, version drift and standard conformance; maintain profile-proxy labels separately. |
| P4+ science | Exploratory TDL-to-CDL profile-proxy and three-scene geometric RT leave-one-scene-out pilots now exist. Across RT folds, all baselines are near chance; no method has a consistent advantage, and three scene families do not support reliable scene-level inference. | Build a qualified larger scene dataset, register endpoints/splits, implement a falsifiable method and matched ablations, then evaluate across more independent scene families. |

The project-owned receiver tests verify exact flat-channel PUSCH recovery, lower estimation NMSE at lower AWGN variance, and a Rayleigh transmitter/channel/receiver loop against native channel truth. RT-grid tests additionally verify a native direct RT path and coherent LOS-plus-specular two-ray response.

---

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

- **Mixed label semantics:** `ChannelDataset.y` currently represents geometric direct-path status **or** an inherent TDL/CDL profile proxy. Create distinct `label_geometric_los`, `label_profile_proxy`, `label_observable_direct`, `label_blockage_state`, each with masks/source/confidence. Never calculate one universal binary accuracy over incompatible semantics.
- **Timing reference:** current `rt_cfr()` calls `paths.cfr(..., normalize_delays=True)`. This explicitly shifts first arrivals to zero. It is fine for certain *relative* channel analyses but destroys absolute-ToA information. Propagate both **absolute-delay oracle** and **physically synchronized receiver-time observable**; do not claim absolute ToA can be identified from an arbitrary common timing shift.
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

## 2. Core scientific definitions and target semantics

### 2.1 Observation model and its scope

For frequency bin `k`, OFDM symbol `t`, receive antenna `r`, and transmit antenna/stream `s`, a useful idealized form is

\[
H_{r,s}(f_k,t)=\sum_{p=1}^{P} g_p(t)\, A_{r,s}(\theta_p,\phi_p,f_k)\,\exp[-j2\pi f_k\tau_p(t)].
\]

This is a **schematic narrowband-per-subcarrier, finite-ray model**; real data can contain pulse-shaping, wideband array effects, diffuse continua, hardware distortion, CFO, clock offset, interference, and other terms. In Sionna, `Paths.a` is a passband-related path quantity and `Paths.cir()` returns the baseband-equivalent representation; do **not** multiply by carrier phase a second time. Verify Sionna’s exact CIR and CFR conventions with canonical test cases.

An observed pilot estimate is instead

\[
\hat H=\mathcal E_{\mathrm{RX}}\big(\mathcal M_{\mathrm{pilots}}[H;\eta],\mathcal S_{\mathrm{NR}},\xi\big),
\]

where `M` is the transmit/receive observation process, `E_RX` the estimator, `S_NR` the actual 5G NR resource-grid configuration, and `eta/xi` noise, synchronization and hardware parameters. The learning model sees `\hat H`, masks, documented receiver uncertainty and **only those known calibration parameters a real receiver could have**.

### 2.2 LOS does not mean “one ray”

Define **geometric clear LOS** as an unobstructed direct geometric segment according to the agreed scene collision/visibility model. A LOS link may include one direct contribution plus many specular, diffuse, refracted and diffracted contributions. A geometrically NLOS link may still carry substantial received power through other interactions. Keep separate:

- `geometric_los ∈ {0,1,unknown}`: **direct clear-path existence**;
- `observable_direct ∈ {0,1,indeterminate}`: whether a direct component is resolvable above the specified sensor threshold;
- `penetration_available`: whether power can pass through an obstructing material;
- `link_coverage`: usable signal or outage/no-path according to a *declared* receiver threshold;
- `dominant_mechanism`: {direct, specular, diffuse, refraction, diffraction, mixture, unknown}; avoid single-label exclusivity if mechanism contributions overlap;
- `blockage_state`: time-dependent visibility with predeclared boundary criteria.

Never use a 3GPP `TDL-E` profile ID as a measured true scene LOS label. The standardized profiles are valuable **controlled pretraining sources**, not independent geometric reality.

### 2.3 Timing, array and gauge freedoms

Suppose the measured CFR is multiplied by `exp(j phi) exp(-j 2π f Δτ)`. **Global phase** `phi` is commonly a nuisance for LOS and power/delay-spread properties; `Δτ` is a timing reference change. A physics-consistent estimator should make LOS and spread **invariant** to appropriate reference shifts and make a **relative-delay output transform equivariantly** as its declared reference changes. But absolute ToA **cannot** be recovered from observations whose timing origin is fundamentally unknown. The following are separate experiments:

1. **Synchronized absolute ToA:** known transmitter/receiver timing/position reference or a validated synchronizer.
2. **Relative first-path delay:** relative to a declared frame/reference point.
3. **Excess delay spread:** independent of common delay shift where the full delay support is represented without circular aliasing.
4. **MIMO phase differences:** meaningful only after array calibration and shared oscillator/phase-reference assumptions are stated; antenna-dependent phase offsets can mimic angles.
5. **AoA/AoD observability:** AoA requires a suitable receiver array; AoD inference needs beam/Tx-array illumination knowledge and sufficient richness. One SISO CFR must never be used to claim an identifiable azimuth.
6. **Doppler:** requires time sampling or coherence over multiple OFDM symbols/slots; a single independent fading sample is insufficient.
7. **Two-way sensing / ISAC:** not implicitly equivalent to one-way communication CSI; define a distinct observation model if pursued.

### 2.4 Resolution, aliasing, censoring and near-indistinguishable channels

- Time resolution from occupied bandwidth `B` is of the order of `1/B`, subject to waveform, SNR, signal processing and super-resolution priors; it is **not** a hard universal lower bound on all parametric estimators.
- Uniformly sampled frequency bins with spacing `Δf` make delay phase periodic with period `1/Δf` in the ideal discrete model. Label paths near/beyond the unambiguous window separately.
- Finite aperture and angular sampling limit spatial resolution; spatial grating lobes and polarization matter.
- Doppler resolution depends on observation duration and sampling cadence; short-trajectory `Paths.cir(num_time_steps>1)` Doppler evolution does **not** simulate new appearing/disappearing paths.
- Close paths may be mathematically non-identifiable under finite pilots/noise. Report **uncertainty or a merged resolvable cluster**, rather than forcing the exact RT oracle path count.
- Where geometry lacks a returned ray because of finite tracing budget, that is a **solver-censoring possibility**, not immediately a proof that the physical path does not exist.

---

## 3. Full output inventory: what the system should infer

The list below is comprehensive by design, **not** a demand to train 20 outputs immediately. Each task must have a valid oracle, observable input and independent measurement definition before it enters the main paper.

| Target ID | Target (units) | Ground truth and label source | Observability / conditions | Main metrics | Priority |
|---|---|---|---|---|---|
| T01 | Physical geometric LOS (`0/1/unknown`) | ray/segment oracle and documented scene obstruction | can be unresolvable from CSI; probabilistic decision | AUROC, AUPRC, balanced accuracy, Brier, selective risk | Core-secondary |
| T02 | Direct-path detectability / weak-LOS probability | direct power relative to receiver noise, thresholded with audited criterion | needs sensor model and observation threshold | precision/recall by direct-to-indirect ratio; calibration | Core |
| T03 | Outage / no-reception | received power, noise/interference and decision threshold | receiver-dependent | detection rate, false alarm, Brier | Core |
| T04 | RMS excess delay spread (seconds/ns) | oracle path-power-weighted delay variance **and** receiver-resolved reference | finite bandwidth, truncation/window bias | MAE(ns), RMSE(ns), median/p90 and coverage | Core |
| T05 | Mean excess delay (seconds/ns) | oracle moments under explicitly referenced support | depends on timing reference but invariant to additive shift after excess reference | MAE, relative error where safe | Core-secondary |
| T06 | Relative first-detectable-path delay (ns) | matched receiver first-detectable path; separate from ray-min `tau` | resolvability and power threshold important | MAE, p90, CDF vs SNR/BW | Core |
| T07 | Absolute propagation ToA (ns or meters) | synchronized geometry, clocks, radio calibration | **not inferable** if clock/timing origin unknown | bias, MAE, p90 localization-equivalent distance | Conditional |
| T08 | Strongest path delay and relative power | oracle path set or receiver-resolved cluster set | coherent cross-terms require careful choice | delay MAE; relative-power error dB | Core-secondary |
| T09 | Ricean / direct-to-diffuse power ratio | declared specular-vs-diffuse and/or direct-to-indirect oracle | definitions differ; direct vs total indirect **not automatically** Rice K | MAE(dB), censored/error distributions | Core-secondary |
| T10 | Resolvable MPC / cluster count | clustering of ideal paths under physical resolution; label includes ambiguity | bandwidth/aperture/noise dependent | count MAE, F1 for resolvable clusters | Core-secondary |
| T11 | Delay/PDP profile | ideal path support passed through **same receive filter/bandwidth** | only band-limited PDP is directly comparable | normalized error, Wasserstein delay distance, peak timing | Core |
| T12 | AoA/AoD, azimuth/elevation (radians/°) | per-path oracle with calibrated array and orientation | AoD harder with downlink unknown beam; angular aliasing | circular angle error, top-K angle recall | Expansion |
| T13 | RMS angular spreads | path-power-weighted **circular** moments with declared convention | calibrated arrays / array geometry | MAE(deg), robust circular metrics | Expansion |
| T14 | Doppler centroid / spread (Hz) | timestamped path velocities and receiver time series | multiple samples, known cadence | MAE(Hz), Doppler spectrum divergence | Expansion |
| T15 | Path-wise complex gain / phase | gauge-aligned and path-assigned oracle | phase gauge and permuted paths make naive MSE invalid | gauge-aligned NMSE, assignment-aware matching | Research |
| T16 | Missing resource-element CSI (complex) | held-out true complex CFR / re-simulated received pilots | must not leak target REs | NMSE, EVM, downstream receiver BLER | Core |
| T17 | Predicted future CSI/CIR (complex, horizon) | future time-consistent simulated/measured snapshots | exact trajectory and radio settings essential | horizon-NMSE, correlation, outage-conditioned NMSE | Expansion |
| T18 | Blockage onset/event time | dynamic object-occlusion event truth | dynamically recomputed geometry required | AUPRC, lead time, time-to-event error | Expansion |
| T19 | Beam ranking / best beam / beam failure | beamformed true channel and fixed codebook | known calibrated array and beams | top-K accuracy, regret, throughput gain, failure rate | Downstream |
| T20 | Position/orientation or localization bias | independent surveyed geometry, synchronization assumptions | can require multi-anchor/multi-snapshot setup | position error m, p90, coverage | Downstream-optional |
| T21 | Calibration, OOD and epistemic uncertainty | empirical outcome vs estimated uncertainty by target | calibration and OOD tests; no oracle at inference | NLL/Brier, ECE, PICP/MPIW, risk–coverage, OOD AUROC | Core |
| T22 | Mechanism / path-existence probabilities | annotated interactions, with ambiguous mixtures retained | mechanism often non-identifiable | macro-F1, expected detection calibration | Expansion |

**Mandatory task grouping for first paper:** T01/T02/T03 + T04/T06/T11 + T16 + T21. Add T08/T09/T10 if reliable; choose only **one** of the time-dependent task pairs (T17/T18 or T17/T19) as the flagship extension. T12–T15 and T20 require more expensive array/measurement validation and may become a follow-on paper rather than diluting the first.

### 3.1 Mathematical target definitions

For ideal path powers `w_p ≥ 0` and delays `τ_p`, with `W=Σ_p w_p >0`, define

\[
\mu_\tau=\sum_p w_p\tau_p/W,\qquad
\sigma_\tau=\sqrt{\sum_p w_p(\tau_p-\mu_\tau)^2/W}.
\]

This is an **ideal ray-power moment**, not automatically the same as the RMS spread of a measured/filtered band-limited PDP. For physical metrics specify per-antenna vs combined weighting and whether incoherent angular/polarization power addition is intended. Use an independent reference implementation and hand-computed toy test cases.

Define direct-to-indirect ratio only when individual direct/indirect powers have a physically consistent decomposition:

\[
R_{D/I}=10\log_{10}\frac{P_{\rm direct}}{P_{\rm indirect}},
\]

with clear limits for absent direct rays. Do not call every `R_D/I` a Ricean K-factor: K-factor definitions depend on which deterministic/specular and diffuse terms are used, and coherent overlapping components complicate estimates from total CFR alone.

A masked-RE reconstruction loss uses **held-out observable target samples**, with measured noise model and no training/test leakage:

\[
\mathcal L_{\rm recon}=\frac{\sum_{m\in\mathcal M_{\rm target}} |\hat H_m-H_m|^2}{\sum_{m\in\mathcal M_{\rm target}}|H_m|^2+\epsilon}.
\]

`H` can denote a noise-free simulated CFR only in simulation studies and must be distinguished from measured `\tilde H`. Normalize per-sample/per-group only when physically appropriate; channel amplitude should not be discarded for outage or link-budget tasks.

---

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
## 5. Data program — physically meaningful, stratified and auditable

### 5.1 Data tiers (do not treat all as the same benchmark)

| Tier | Data | Purpose | Required audits | Never claim |
|---|---|---|---|---|
| D0 | analytically solvable SISO/MIMO free space, two rays, known Doppler | mathematical unit tests, oracle validation | exact Fourier sum, path delay, phase, aperture, power | realistic field performance |
| D1 | Sionna PHY TDL/CDL A–E (`spec_version="19.2"`) | standardized controls, profile recognition and source pretraining | balanced profile+delay/SNR, antenna, speed, profile masks | geometric visibility truth independent of model design |
| D2 | scenario models UMi/UMa/RMa/InH/InF with actual LOS-state logic and spatial consistency | bridge from fixed-profile families to scenario distributions | scenario labels, geometry/statistical consistency, correlation lengths, blockers | equivalent to true detailed RT |
| D3 | controlled RT canonical scenes | demonstrate causal mechanisms and sensor resolution | geometry/oracle cross-check, path solver sweeps, direct+multiple indirect paths | complete physical city fidelity |
| D4 | independent high-fidelity urban/indoor/factory/campus RT scenes | primary held-out-scene generalization | material uncertainty, mesh QA, representative scene diversity | validated against the world without measurements |
| D5 | independently recorded, permission-cleared NR CSI or I/Q with surveyed ground truth | external transfer and simulator-bias audit | hardware calibration, clock/LOS labels, ethics/permissions | absolute LOS labels from subjective RSS alone |

The 3D scenes currently checked into the repository (`city41`, `city42`, `city43`) are useful *deterministic controlled fixtures*. They are **not** enough for strong external-validity claims: same procedural rule family and simple box meshes make held-out seeds much easier than an entirely independent city/asset pipeline.

### 5.2 Scene families and environments to build

**Controlled foundation:** free-space link; flat reflective ground; single wall, corner/wedge, corridor, hall, office partitions, two-wall transmission, moving metal screen, narrow street canyon, foliage proxy, rich indoor room, factory machinery line, parking vehicles, open plaza.

**External test families:** two or more **independent asset/geometry pipelines** (e.g., a measured/surveyed or authored building, independent procedural generator, Sionna built-ins such as Munich/Étoile). Across each, vary:

- geometry topology: streets/corners, buildings, material thicknesses, multilevel stories, vehicle density, indoor rooms, factory racks, irregular surfaces;
- materials: concrete, brick, glass, metal, wood, plasterboard, composite; explicit thickness/permittivity/conductivity/scattering parameters; frequency-valid ranges;
- EM mechanisms: direct, one-/multi-bounce specular, diffuse, refraction, diffraction; interaction mixtures and near-occlusion;
- UE and BS height, carrier, polarization, array geometry, element patterns, orientation, FoV, placement and steering;
- propagation distance, antenna gains, transmit power, thermal noise, interference and receiver sensitivity;
- receiver mobility and independent blocker/vehicle trajectories; include trajectories with clear LOS → partial shadow → blocked LOS → reappearance;
- a **no-detected-path / outage** category rather than dropping links.

**Important control:** Sionna’s built-in ITU material models can have `scattering_coefficient=0` by default. Scene realism depends on **material parameters**, not just turning the diffuse flag on. Log the enabled mechanism and actual fraction of paths/power it contributes.

### 5.3 Geometry and physical-consistency validation

For every scene version, generate an immutable `scene_manifest.json` with source/geometry license, XML/OBJ checksum, coordinate frame and units, vertices/faces/bounding boxes, transmitters/arrays, receivers, material assignments/thickness, frequency validity, scattering coefficients, and test outputs. Validate:

1. **Watertight/nonwatertight mesh policy** documented; normal orientation and face winding; no self-intersection that breaks physics unexpectedly.
2. TX/RX physically outside solid obstacles; adjustable clearance, no placements underneath ground/floors or inside vehicles.
3. TX/RX height and orientation in physically plausible ranges; min/max link distance; no overlapping antennas.
4. Direct-segment collision/ray oracle agreement on hand-reviewed LOS/NLOS examples; include wall grazing/epsilon tests.
5. Cross-process deterministic replay of canonical scene samples using identical package lock/precision/backends.
6. Path counts, path delay ordering, delay vs total traveled length, path power vs range, per-interaction oracle labels, diffraction/corner cases.
7. Coherent and incoherent power conventions: `|Σ a_p e^{-j2πfτ_p}|²` can differ from `Σ|a_p|²`; use correct convention for the target.
8. Material changes produce expected **trends** rather than assumed exact values (e.g., metal reflectivity, thickness and transmission); compare ray-level behavior with analytical Fresnel/free-space limiting cases.
9. Convergence with `samples_per_src`, `max_depth`, deterministic mode and raster/mesh precision. Report unresolved links and solver failures separately.
10. External visual QA: save rendered images from multiple angles of every final evaluation scene, ray overlays for a stratified set and screenshots of blockage events.

### 5.4 Link and trajectory generation protocol

**Static data:** sample geographic/geometry positions *before* looking at the label distribution; record full acceptance statistics. Do not make the training LOS ratio a hidden artifact of rejection sampling. For balanced diagnostic subsets, keep separate raw prevalence, balanced indices and inverse-probability weights.

**Dynamic data:** generate physically meaningful trajectories `p_UE(t)` and blocker motion `p_b(t)` at a declared cadence, velocity/acceleration/turn constraints, and time-stamped geometry hash. For transitions, re-trace at interval boundaries or every snapshot according to validated temporal resolution. Preserve entire sequences in one split; no overlap, shifted copies or mirrored scenes that leak across splits.

**Spatial exclusion:** assign super-groups by scene/asset family, sub-groups by street/block, link/trajectory, material seed, and BS site; hold out complete scenes, use a buffer in meters or a conservatively chosen estimated correlation distance across train/test boundaries, and perform a **nearest-neighbor geographic leakage audit**. One random `group_split()` is not a substitute.

**Obstruction labels:** A moving occluder can modify reflections as well as the direct ray. For a real counterfactual, re-run the **entire scene** with blocker inserted/removed and retain identifiers linking the two interventions; do not simply subtract the direct coefficient and call the resulting channel physically obstructed.

### 5.5 Minimum benchmark matrix: design examples, NOT final fixed sample counts

Construct the initial factorial design using levels below, then refine with compute/power pilot *without consulting the locked test result*. Numbers listed are proposal levels, not empirically optimal ranges:

| Axis | Example levels | Critical control |
|---|---|---|
| Carrier | 3.5 GHz (legacy anchor); 6 GHz; 28 GHz if assets/materials qualified | material/antenna availability; don't pretend one material is valid at every frequency |
| NR subcarrier spacing | 15 / 30 / 60 kHz only where selected numerology valid | same occupied bandwidth across models when comparing frequency effects |
| Occupied bandwidth | 2.88 MHz legacy active allocation; wider practical cases selected with explicit PRBs/FFT | pilot count and resolvability changes confound gains |
| Receiver noise | relative-SNR diagnostics and separate thermal-noise/link-budget simulation | record exact definition: SNR, Eb/N0, SINR are different |
| Mobility | static; pedestrian; vehicle; dynamic blocker | dynamic trajectories/cadence actually recomputed |
| Arrays | SISO; 2×4 or comparable calibrated MIMO; larger array *if justified* | preserve aperture, array/polarization geometry and source directions |
| Scenarios | LOS-direct-only; LOS-mixed; weak-direct LOS; pure NLOS-reflection; NLOS-transmission; NLOS-diffraction; outage | never map scenario mix to a single binary class without a label schema |
| Pilot density | validated NR DMRS patterns; controlled missing-RE stress subset | equal measurement budget in model comparison |
| Physical corruption | AWGN, CFO, timing offset, phase noise, quantization, channel aging, interference | single-factor and interaction tests; independent seeds |
| Scene types | urban, indoor, industrial, controlled analytical | separate topology family in at least one final held-out set |
| Train data | D1 only; D2 only; D3/D4 only; mixed D1–D4 | report exact domain mixing and labels |

**Design/cost rule:** the *full Cartesian product is computationally infeasible*. Use staged screening (orthogonal or Latin-hypercube sampling) followed by a smaller preregistered stress factorial. Publish the sampling policy and condition counts. Preserve at least several independent scenes per external comparison; no confidence interval based on a single scene can justify cross-scene generalization.

### 5.6 Dataset storage schema (must be testable)

Proposed canonical source sample dimensions:

```yaml
sample:
  id: immutable_unique_id
  source_type: {analytical, tdl, cdl, tr38901_scenario, rt, measurement}
  scene_id: immutable_scene_or_session_id
  asset_family_id: independent_scene_family
  trajectory_id: global_trajectory_or_link_id
  timestamp_s: float64_or_null
  tx_rx_geometry: oracle_private_metadata
  receiver_observation:
    complex_pilot_IQ: [time, rx_ant, tx_stream, ofdm_symbol, pilot_re]  # if available
    H_hat_complex: [time, rx_ant, tx_stream, ofdm_symbol, used_subcarrier]
    observed_re_mask: same_reference_grid
    H_hat_error_variance: same_reference_grid_or_broadcastable
    receiver_known_config: {carrier_hz, scs_hz, fft_size, cp, prbs, dmrs, array_geometry, calibrated_noise}
  simulation_privileged:
    H_true: ideal_complex_reference
    paths: {valid_mask, delays_s, amplitudes, angles_rad, doppler_hz, interaction_types}
    geometry: {los, blocker_objects, positions_m, materials}
  target_values:
    geometric_los: value_or_unknown
    observable_direct: value_or_indeterminate
    delay_spread_s: float_or_null
    path_set: list_or_empty
    future_H: array_or_null
    beam_labels: list_or_null
    uncertainty_flags: explicit_mask_per_target
  provenance:
    seed: integer
    code_commit: sha
    artifact_hashes: hashes
    api_lock: lock_id
    label_definition_version: semantic_version
    licensing: dataset_and_scene_terms
```

This is a **schema design**, not currently implemented serialization. Use Zarr/HDF5/Parquet + NumPy complex buffers as appropriate; confirm inter-process safety/immutability and float precision. No implicit flattening, no untyped `metadata_json` as the only semantics registry, no conversion of seconds to ns without explicit metadata. Raw/processed dataset manifests should have independent hashes; split indices saved as immutable artifacts.

### 5.7 Truth labels, soft labels and ambiguity policy

For each target create:

- `oracle_definition.md`: exact formula and whether it comes from geometry, path solver, receiver estimator or lab instrument.
- `label_scope`: per-path / per-antenna / per-link / per-trajectory; array-wide definitions cannot silently replace element-wise ones.
- `validity_mask`: estimated is possible / oracle unknown / censored / no coverage / insufficient information.
- `tolerance`: numerical/physical ambiguity thresholds determined from independent analytical checks, not optimized for model wins.
- `quality_flag`: solver convergence and real measurement calibration status.
- `label_version`: update triggers a new dataset and a new paper experiment ID.

For a path that is physically present but substantially below the noise floor, `geometric_los=True` and `observable_direct=indeterminate/false` may **both be correct**. This case is central to the proposed identifiability-aware mechanism; do not discard it as a contradiction.

### 5.8 Real measurements: highest-value gate

Prepare a separate `REAL_MEASUREMENT_PROTOCOL.md` with:

1. **Data permission:** own collection or an appropriately licensed public dataset, respecting any university/company/Nokia restrictions; user ownership of source code does not automatically grant rights to third-party measured data, scene assets, or proprietary device firmware.
2. **Hardware:** transmitter/receiver models, antenna arrays, RF front-end, clock/time synchronization, GPS/survey quality, position/orientation, frequency reference, trigger times, IQ dynamic range, AGC and calibration logs.
3. **Capture:** NR pilots or known probe waveform plus true/estimated channel, transmit power, noise floor, interference background, SNR/SINR estimate, RF impairments, polarization.
4. **Truth:** surveyed visibility and blocker video/map/manual inspection independent of CSI; timestamp alignment; LOS boundary/partial obstruction/penetration annotations; blind second audit on ambiguous cases.
5. **Sites/sessions:** distinct buildings/streets/days/hardware setups when possible; at least one held-out site untouched by tuning. Never mix recordings from one walk into both train and test.
6. **Privacy/security:** avoid releasing traceable personally identifying trajectories/metadata without approval; document anonymization and geographic redaction that do not corrupt labels.
7. **Transfer:** source-only zero-shot test first; if allowed, separate **unsupervised target calibration** and **few-shot labeled adaptation** with clear labels and separate evaluation splits.
8. **External reproducibility:** if raw data cannot be shared, publish de-identified aggregate evaluations, provenance hashes, data statement, access process and appropriately sanitized scripts.

**No real dataset = permissible simulation-heavy work, but change claim wording.** Never call Sionna-only validation “real-world robustness.”

---

## 6. The paper's actual methodological research program

### 6.1 Closest already-known ideas — novelty cannot be asserted by naming

Prior art already covers complex-valued neural networks, phase/co-domain symmetry, LOS + first-path ToA joint prediction, path-token wireless foundation models, and delay–Doppler–angle representations. For example, **MILCOM 2024** studied a CVNN for LOS classification and first-path ToA; **MultiPathFormer (Aug 2026 preprint)** learns path tokens and evaluates several wireless tasks; **AirFM-DDA (Apr 2026 preprint)** learns delay–Doppler–angle representations. See the references section. These are **research-overlap warnings**, not evidence that every proposed identifiability or counterfactual mechanism is anticipated. Before claiming novelty, map the closest 10–20 methods to exact mathematical components and evaluation settings.

### 6.2 The mechanistic contribution to attempt

Working name: **Observable Multipath State Inference (OMSI)**, used strictly as an *internal project label*. A proposed formulation:

1. **Equivariance-aware complex frontend** processes measured complex observations with explicit masks, antenna/frequency/time geometry and realistic nuisance transformations. Its purpose is to preserve relative phase and enforce **task-appropriate** gauge symmetries. Compare against real-valued `Re/Im` concatenation, magnitude/PDP, real CNN and conventional CV-CNN; do **not** assume CVNN wins.
2. **Structured path/cluster posterior** predicts a bounded or variable-cardinality **set** of resolvable components `\hat Z = {\hat \alpha,\hat\tau,\hat\theta,\hat\nu,q,\Sigma}_p` with existence probabilities and calibrated uncertainty. Distinguish finite coherent specular paths from diffuse/background residual; **do not force a finite ray model to explain all RF phenomena**.
3. **Measurement-consistent differentiable forward renderer** converts inferred components + residual into predicted complex **pilot observations** (not only ideal hidden CFR), honoring array response, time/frequency positions and known nuisance/calibration. Supervise withheld pilots and available physical targets, with standardized per-output units and confidence.
4. **Identifiability-aware loss/decision** uses estimated resolvability proxies (bandwidth, temporal aperture, angular aperture, SNR and pilot masks) to avoid penalizing models for distinguishing physically equivalent states. Candidate methods: uncertainty-weighted targets, equivalence-class losses, resolution-dependent optimal transport, likelihood-based posterior predictive calibration and **selective prediction**. Derive mathematical properties before calling it new.
5. **Counterfactual-consistency term** trains on *genuinely re-traced* scene interventions that vary obstruction while keeping all identified nonintervened scene controls fixed; compare against coefficient deletion augmentation and against plain augmentation without paired loss.
6. **Temporal dynamics head (conditional extension)** infers future measured CSI and blockage/beam utility from past observations plus documented UE/array motion. Avoid privileged future geometry at inference. Evaluate long-horizon updates where the actual scene is re-traced, not solely Doppler extrapolated.

A distinctive and falsifiable claim would be: **“Learning only representations consistent with what the receiver can physically distinguish improves transfer and predictive calibration under synthetic-to-real and novel-scene domain shift.”** Test whether identifiability control specifically accounts for the benefit; if not, simplify the model.

### 6.3 Mathematical contribution candidates (choose one primary and prove/test it)

**Candidate M1: invariance/equivariance with non-identifiable timing.** Define group action `G` of global phase, valid calibration offsets and common timing shifts, corresponding action on observations `g·X` and target quantity `ρ(g)Y`. Develop a representation `F` with rigorously specified invariance/equivariance for each task and show why absolute ToA is outside the identifiable quotient without synchronization. For stochastic/uncertain labels, compare probability distributions under transformed input.

**Candidate M2: observable-state equivalence classes.** Define two multipath states as `ε`-indistinguishable if, given pilot grid/array and noise covariance, they induce sufficiently close measurement distributions. Design a training loss/posterior that does not create false certainty about path count/delays within one equivalence class. Propose a *formal local identifiability criterion* via Jacobian singular values or Fisher information, **accounting for nuisance parameters and constraints**. Compare with classical CRLB and Bayesian bounds where regularity assumptions hold; do not report a bound unsupported by parameter identifiability.

**Candidate M3: intervention-consistent latent dynamics.** Formalize the causal variable as *scene modification* and the observation as *recomputed channel*. A representation should predict which channel statistics change after a blocker intervention while separating unmodified static reflectors, without assuming that reflection paths remain fixed. State assumptions and show failure cases.

**Research choice gate:** first develop **M2** on a two-path analytical setting with provable ambiguity/uncertainty, then add **M1** as the architectural symmetry prior. Consider **M3** only after Sionna dynamic RT and physical intervention validation. This avoids simultaneously selling multiple unrelated novelties.

### 6.4 Decoder and supervised objective (candidate, not finished invention)

For a proposed structured state `Z`, receiver/geometry-known parameters `u` and observation grid `M`, define a differentiable forward map `F_φ(Z,u,M)`. Let:

\[
\mathcal L = \lambda_{\rm pilot}\mathcal L_{\rm pilot-likelihood}
+ \lambda_{\rm hold}\mathcal L_{\rm heldout-pilot}
+ \lambda_{\rm task}\mathcal L_{\rm masked-targets}
+ \lambda_{\rm set}\mathcal L_{\rm identifiable-set}
+ \lambda_{\rm sym}\mathcal L_{\rm gauge}
+ \lambda_{\rm cal}\mathcal L_{\rm calibration}
+ \lambda_{\rm cf}\mathcal L_{\rm counterfactual}.
\]

**Do not activate every term blindly.** For each term define units, training stage, gradient flow, weighting policy, gradient-norm statistics and an **off** ablation. A task loss is applied **only where the target exists with trustworthy validity masks**; no missing labels silently set to zero. Avoid true-path supervision on target domains if claiming unsupervised or zero-shot domain transfer. Fit normalization/scalers exclusively on training data.

**Permutation symmetry:** path components are not ordered by a physically reliable identity across environments. Use optimal matching, symmetric set likelihood, or energy-based distribution matching; do not impose `p=1,2,...` supervised order unless the label semantics define and verify it. For diffuse clusters, compare *distributions/fields*, not an impossible exact one-ray correspondence.

**Phase alignment:** reconstruction losses must use correctly synchronized complex representations or explicitly marginalize nuisance phase/timing shifts. Do not allow the loss to “cheat” by phase aligning predicted and target observations at test time using hidden ground truth unless this alignment is a declared evaluation-only metric.

**Consistency:** derive RMS delay spread, mean delay, angular spread (if feasible) and filtered PDP from the same estimated state **using differentiable formulas**; compare against independent heads, and report whether the consistency constraint actually improves transferable targets.

**Uncertainty:** output predictive intervals/distributions with calibrated coverage; separate aleatoric sensor ambiguity, stochastic modeling error and epistemic shift where possible. Labels from stochastic RT sampling carry solver uncertainty—calibrate them too or report them separately.

### 6.5 MLP and CNN contribution audit: the decisive comparison

Use identical input tensors/masks, train-val splits, training steps, augmentations and precision for:

- C0: raw `Re/Im` flattened MLP, simple and capacity-scaled; time/array geometry fed identically;
- C1: expert features → MLP (PDP statistics, correlation, CSI distribution); verify feature computation matches real observations;
- C2: real-valued CNN on `Re/Im` channels;
- C3: complex-valued CNN with known, well-defined stable activation/normalization and matched capacity;
- C4: CNN encoder + MLP fusion;
- C5: **zero/identity/frozen/random CNN encoder** + same head (tests learned convolution gain);
- C6: permutation/phase-destroyed controls to test whether network exploits genuine physical structure;
- C7: proposed complex frontend, no identifiability or physics loss;
- C8: proposed frontend + physics decoder;
- C9: + identifiability-aware mechanism;
- C10: + symmetry and/or calibrated uncertainty;
- C11: + counterfactual or temporal extension, **only if justified**.

For every comparison disclose: learned parameter count, FLOPs/MACs where reliable, memory, wall-time, data volume, training steps, hyperparameter-search budget, inference latency and optimization curves. Run matched **small/medium/large capacity envelopes**, not a single cherry-picked parameter count. **If the strong MLP wins on external scenes**, simplify/remove convolution and document the finding; this is a valid scientific outcome.

### 6.6 Classical signal-processing and other state-of-art baselines

- LOS/direct evidence: Rician K/strongest-tap tests (with clear assumptions), delay-spread heuristic, kurtosis, correlation/cyclostationarity statistics, generalized likelihood-ratio tests if observation model supports them, calibrated logistic, Random Forest, XGBoost.
- Path inference: thresholded IFFT/periodogram, MUSIC/ESPRIT with valid array/snapshot conditions, SAGE/RiMAX or comparable high-resolution parameter estimation (only if implementations are available), sparse recovery/OMP, atomic-norm or parametric iterative least-squares where tractable.
- CSI estimation: pilot LS + linear/nearest/LMMSE interpolation, model-based time-frequency estimators, deep CSI denoising/transformer with **the same pilot budget**.
- Uncertainty: deep ensembles, MC dropout (with known limitations), heteroscedastic likelihood, post-hoc temperature/isotonic calibration, split-conformal intervals under explicit exchangeability assumptions.
- Domain generalization: ERM, group-DRO, CORAL, style/phase augmentation, adversarial invariance (if justified); training target distributions and access clearly separated.
- Multipath representation/foundation models: reproduce/port methods comparable in **input and task definition** to MultiPathFormer and AirFM-DDA where code/weights/license allow; never compare their published best number on different data as a fair head-to-head.
- Temporal: autoregressive CSI, Kalman/extended Kalman where physically meaningful, linear predictor, RNN/TCN/transformer with equal history and no future leakage.
- Downstream beam decisions: strongest previous beam, oracle perfect-CSI upper bound, conventional beam tracking/prediction; evaluate effect of model uncertainty/abstention.

**Baseline registry:** each method has exact paper link, code commit, data rights, input modality, oracle privileges, hyperparameter search, training-domain access, compute, and failure mode. Missing a baseline does not become “SOTA achieved”; disclose coverage of the literature.

---
## 7. Complete experiment catalogue and required evidence

Every experiment must produce an immutable `run_id`, input/scene/split/source hashes, config/label definition, exact environment, checkpoint(s), `predictions.parquet` or equivalent, scalar metrics and per-group metrics, plots, resource-cost measurements, failures, and a one-paragraph scientific conclusion. **No running experiment should modify the locked test cohort or replace its metric JSON.**

| ID / stage | Experiment, prespecified question | Minimum comparison | Primary outputs | Decision gate |
|---|---|---|---|---|
| E00 | Reproduce 2025 results on original data/preprocessing | thesis MLP, VGG, ResNet, kNN/RF | Table: historical vs rerun, seeds/config/CI; dependency migration notes | confirm which outputs are reproducible; quarantine irreproducible numbers |
| E01 | Verify analytical one/two/multipath channels | hand-derived H/tau/power, numerical Fourier/cir/taps | tolerance report, figures of CIR/CFR, exact/unit tests | fail hard if any invariant mismatch |
| E02 | Sionna PHY `TDL/CDL A–E` correctness | 16.1 vs 19.2 supported table selectors, documented shape/energy | profile-PDP plots, normalized RMS tests, shape, speed, K and antenna tests | fail if API assumptions false |
| E03 | RT scene/label oracle fidelity | analytical collision, LOS-only ray solve, inspected labels | confusion of geometric oracle vs hand-audit, boundary breakdown, ray render | do not generate main dataset until validated |
| E04 | RT solver convergence and physical realism | depth/path samples, deterministic & stochastic seeds | converged metric curves, invalid fraction, scene paths by mechanism | freeze ray budget at physics-converged level |
| E05 | NR receiver/pilot correctness | ideal CFR, pilot LS and LMMSE, NR PUSCH decoding | channel NMSE/EVM/BLER vs SNR, pilot energy, mask map | only validated receiver can feed main papers' receiver-realistic claims |
| E06 | Dataset bias/profile leakage | random TDL/CDL split, held-out profile, unknown RT scenes | AUROC collapse, profile-ID confusion, two label semantics | report proxy-vs-physical gap; never combine incompatible truth |
| E07 | Realized LOS complexity and balance | direct-only/LOS-mixed/NLOS mechanisms, noncoverage | conditional ratios, K/direct ratio histograms, sample acceptance | LOS-mixed must be present and tracked; no selection concealment |
| E08 | MLP vs CNN contribution | C0–C6 capacity envelopes | all-target scores, parameters/latency, module-off deltas | justify or remove convolution branch |
| E09 | Conventional channel-state inference | classical estimator vs matched MLP | delay spread, first-path, PDP, K, path count (with valid masks) | select measurable primary target |
| E10 | Held-out channel reconstruction from sparse pilots | LS/LMMSE, denoising MLP/CNN, proposed decoder-off/on | NMSE dB and linear, EVM, sensitivity vs observed pilot fraction | reject decoder if it only memorizes scene styles |
| E11 | Main identifiability-aware contribution | same backbone identifiability on/off | scene-level error + calibrated coverage + risk–coverage by identifiability | **principal novelty gate** |
| E12 | Phase/timing/array gauge tests | same model with/without symmetry constraints | per-transform prediction drift and correct equivariant target shift | disqualify theoretical symmetry claim if violations persist |
| E13 | Source-to-target generalization | TDL/CDL→RT, scenario models→RT, RT→unseen RT | complete source/target transfer matrix | no final domain-generalization claim without external scenes |
| E14 | Real measurement test | frozen source model vs real-data classical/MLP | actual measured target errors/CI, shift and calibration curves | if unavailable state simulation-only conclusion |
| E15 | Counterfactual blocked/unblocked scene pairs | coefficient-deletion toy vs geometry-retraced scenes | causal-effect error, target state consistency, paired deltas | no claim of physically causal counterfactual without re-tracing |
| E16 | Dynamic channel state / Doppler | short-term Doppler extrapolation vs full geometry updates | future CSI NMSE vs horizon, Doppler/velocity error | dynamic mechanism must beat naive persistence where appropriate |
| E17 | Blockage forecasting | full re-traced transitions, early warning | AUPRC, time-to-blockage MAE, false alarms/hour, lead time | ensure no future scene metadata at inference |
| E18 | Beam decision impact | static/last-beam, oracle, baseline forecast, proposed | top-K, normalized beam regret, beam-failure rate, throughput/BLER | must show operational benefit, not accuracy alone |
| E19 | Reliability + abstention | uncalibrated, post-hoc calibrated, proposed uncertainty | ECE, NLL, Brier, 90/95% coverage vs width, selective error | reject if interval calibration fails on held-out scenes |
| E20 | Robustness and ablation grid | all new mechanism toggles | failures under noise/BW/frequency/array/material/path count | mechanism must retain benefit across meaningful shifts |
| E21 | Computational practicality | MLP/classical, CVNN, proposed, path baseline | memory/params/FLOPs, train time, online latency, energy if measured | disclose trade-off; no “efficient” claim without benchmark |
| E22 | Data scale and hyperparameter sensitivity | 10%, 25%, 50%, 100% *training* data | learning curves, optimization instability, tuning sensitivity | general benefit not a single lucky budget |
| E23 | Cross-simulator / material-model validity | two scene pipelines or calibrated physical measurements | sensitivity to material and RT engine assumptions | distinguish simulator bias from method ability |
| E24 | Frozen final reproducibility / independent replication | fresh machine/split seed and untouched final data | identical dataset checksums, metric script replay, audit report | required for results release |

### 7.1 Detailed generalization split matrix

Evaluate at least these **distinct and transparently named** protocols; not all are equally strong evidence:

| Protocol | Train | Tune/validation | Locked test | What it proves and does not prove |
|---|---|---|---|---|
| P0: IID smoke | same source, disjoint links | same source | random examples from same generator | debugging, **not** scene transfer |
| P1: profile holdout | subset of A–E | disjoint source profiles | unseen TDL/CDL profiles | controls profile identity; still standardized proxy truth |
| P2: cross-family | TDL | CDL validation where allowed | CDL or vice versa | stochastic-family generalization, **not** geometric LOS |
| P3: synthetic→RT | TDL/CDL/3GPP scenario models | only source-domain/held-in dev scenes | independent Sionna RT scenes | zero-shot geometric test under source/target semantic shift |
| P4: unseen RT city | RT scenes/families A, B | RT development family C | RT families D, E | scene/geometry transfer; asset-family independence required |
| P5: material/frequency transfer | subset of material/frequency regimes | held-in validation | held-out material or carrier | robustness to independent EM assumptions |
| P6: dynamic | complete trajectories in training sites | separate complete trajectories/sites | novel trajectories, new blockers and sites | genuinely temporal prediction |
| P7: source→real | simulated only | source or unlabeled approved calibration | frozen measured site/session | most credible external transfer evidence |
| P8: few-shot real adaptation | sim + specified K measured labels | separate real dev | distinct unseen measured site | adaptation evidence, **NOT** zero-shot |

**One-shot protocol rules:** no test-set feature normalization fit, model selection, calibration, early stopping, augmentation choice, best SNR regime, class threshold or hyperparameters tuned on P4/P7 test labels. When unlabeled test-domain statistics are used, label clearly **transductive/domain-adapted**, not strict zero-shot. If a test dataset is viewed and revised against, demote it to development and obtain a genuinely fresh test.

### 7.2 Target-specific metrics (definitions must be frozen)

**Binary and multi-label:** AUROC (with both classes present), AUPRC with positive prevalence documented, balanced accuracy, macro-F1, sensitivity/specificity, false-alarm/detection rate at a validation-fixed threshold, confusion counts, Brier and calibrated NLL. When no positive case in one scene, report the raw counts and aggregate via a valid hierarchical method; do not compute undefined AUROC as zero or omit it without explanation.

**Continuous physical quantities:** MAE, median AE, RMSE, 90th/95th percentile AE, bias, stratified errors vs SNR/scene/BW/LOS, physically meaningful units. Report absolute and relative errors only where denominator safe; avoid percentage error around near-zero delay spread. Compute circular angular errors modulo 2π and distinguish azimuth/elevation.

**Path/cluster set:** cardinality count MAE, precision/recall under pre-registered resolution-aware delay/angle tolerance, assignment/optimal-transport cost, complex gain error only **after declared gauge/matching**, mixture likelihood when available, detection curves by path separation and power gap. Report **unresolvable oracle cases separately**.

**PDP/CSI:** complex CSI NMSE `||Hhat−H||² / max(||H||²,epsilon)` per independently defined sample and over globally pooled energy (both views), dB transform of the **aggregate specified statistic**—do not average dB when the intended result is linear-domain pooled NMSE. Also profile Wasserstein distance (with normalized nonnegative PDP) and EVM if appropriate. Keep noise/target reference conventions fixed.

**Uncertainty:** proper negative log-likelihood when probability density defined, CRPS for continuous distributions, interval coverage at 50/80/90/95%, mean width, interval score, calibration curves and expected calibration error (with bin definitions), selective risk–coverage/AURC, OOD detection AUROC/AUPRC, group-conditional coverage. Standard conformal guarantees do **not** automatically survive strong distribution shift; evaluate them empirically rather than asserting theoretical shifted-domain coverage.

**Forecasting:** NMSE versus horizon, trajectory-balanced MAE and correlation, interval coverage vs time, blockage AUPRC and event-time error, pre-event lead time at fixed false-alarm rate, receiver drop/outage prediction, beam top-K accuracy, rate/throughput, beam regret, BLER/packet errors, and computational/energy burden. Clearly distinguish inference with **past CSI only** from one supplied with speed or planned route known to the system.

### 7.3 Statistical analysis and sample-unit discipline

1. **Choose the primary endpoint(s) and smallest practically meaningful difference** before final training; these thresholds are research-policy decisions, not assumed performance gains.
2. Each model uses **paired datasets, paired seeds/splits and equal budgets**; random initialization seeds are not independent physical scenes.
3. Train at least multiple seeds (e.g., 5 exploratory, 10 confirmatory **if affordable**), but compute primary external uncertainty across **independent scenes/sites/sessions** and nested trajectories. Ten seeds on one city do **not** equal ten cities.
4. Report per-scene metrics, distribution across scenes, within-scene sample dispersion and across-seed variation. Use scene-level hierarchical bootstrap or appropriate mixed-effects analyses depending on endpoint/data structure; show small-N caveats.
5. Use pre-registered bootstrap CI or paired permutation tests with exchangeability units and paired differences. If scenes are too few for stable intervals, expand external data or limit the statistical claim; do not increase bootstrap replicates to manufacture independent evidence.
6. Control multiple comparisons (e.g., Holm for a small prespecified family of main contrasts), report adjusted p-values where hypothesis testing is valid, effect sizes with 95% CIs, and practical relevance—not p-values alone.
7. For AUPRC document prevalence changes; always report raw-world prevalence and any balanced diagnostic prevalence separately. Avoid comparing AUPRC across different prevalences as though directly comparable.
8. Pre-register what constitutes successful *task completion*, *scientific evidence* and *publication claim*. A design with no reliable ground truth can still have software tests but cannot support a parameter-estimation claim.
9. **Blinded final verification:** freeze architecture, thresholds, hyperparameters, splits, metrics and claim text before opening an independently held-out real/RT evaluation.
10. Distinguish training variability vs measurement variability vs scene-model variability; show uncertainty estimates that reflect the intended deployment population.

### 7.4 Controlled scientific stress/falsification tests

Each is a mandatory diagnostic for the claimed mechanism; expected direction is a hypothesis, **not a promised result**:

| Test | Perturbation | What should remain/change if method is correct |
|---|---|---|
| S01 | common global complex phase rotation | LOS/spread probabilities invariant; complex reconstruction equivariant to appropriate reference |
| S02 | known common time-reference shift | delay *differences/spreads* invariant; absolute delays shift if reference is meaningful |
| S03 | arbitrary per-antenna phase offsets | angle estimation should reflect calibration ambiguity; confidence should widen if offsets unknown |
| S04 | antenna element permutation with transformed geometry | angular output consistent under co-permutation; no phantom change |
| S05 | path-index permutation in oracle | set loss/reconstruction invariant |
| S06 | two paths with progressively smaller separation | resolution-aware path count uncertainty increases when distinction disappears |
| S07 | weak direct path progressively below noise | geometric oracle remains LOS; observable detectability and calibrated confidence change |
| S08 | add direct ray + multiple reflections | model not assume LOS=single path |
| S09 | remove/block direct ray while re-tracing full scene | correct physical channel changes beyond simple coefficient deletion |
| S10 | material permittivity/conductivity/thickness changes | channel response changes physically; performance uncertainty audited under OOD materials |
| S11 | scattering coefficient 0 → nonzero | true diffuse energy appears only when material and solver allow it |
| S12 | enable/disable diffraction near a corner | boundary/corner channel changes, no silent label flip semantics |
| S13 | double subcarrier spacing keeping used bandwidth control | delay ambiguity period changes; preserve stated resolution effects |
| S14 | change occupied bandwidth and pilot density | temporal resolution and estimation error curves follow verified measurement physics |
| S15 | zero noise with known simple channels | decoder approaches known truth up to proven gauge ambiguities |
| S16 | high noise/interference, CFO and I/Q imbalance | prediction confidence calibrated / graceful degradation, not fabricated precision |
| S17 | spectral nulls and very low link gain | no divide-by-zero and no signal-existence label leakage |
| S18 | ray solver budget/depth sweep | conclusions stable past physical convergence point |
| S19 | scene texture/filename/label metadata shuffle | predictions invariant because labels are not model inputs |
| S20 | duplicated/shuffled trajectories across splits | automated leakage scanner rejects dataset |
| S21 | train in one scene generator, test independent family | quantify domain gap rather than claim unseen seed equals unseen environment |
| S22 | true held-out receiver/hardware calibration | uncertainty performance degrades visibly/reportably under hardware shift |
| S23 | time reversal or shuffled future sequences | forecasting advantage should disappear/degrade if model uses genuine temporal structure |
| S24 | sample-label permutation negative control | learned predictive advantage disappears; no accidental metadata target leakage |
| S25 | constant or zero-conv encoder | honest CNN contribution ablation; no selection of favorite model post hoc |
| S26 | match SNR distributions independently of class | high LOS score should not be trivial SNR/received-power classifier |
| S27 | stratify link distance/position/antenna by class | test whether location-only shortcuts explain gains |
| S28 | repeat solver with independent random seeds/backend | measure stochastic variance in targets and outputs |
| S29 | cross-check numerology and DMRS mask | receiver model must fail correctly on invalid grid/overlapping pilots |
| S30 | clock offset without reference | absolute ToA predictor must be explicitly inapplicable or highly uncertain |

### 7.5 Realistic metrics success criteria — how to decide, without inventing outcomes

**Correctness gates:** all native physical invariants and observation contracts pass; questionable data never enters confirmatory evaluation.

**Scientific improvement gate:** new method yields a **positive paired improvement** on prespecified external scenes for the primary endpoint, with CI excluding a practically negligible effect threshold (when sample size supports such inference), while calibration and computational cost remain acceptable. If not, accept the null and revise/reduce scope. No mandatory “X% improvement” is invented here.

**Transfer gate:** demonstrate performance on **multiple genuinely distinct** scene/measurement sources, not only procedural random seeds. A zero-shot claim needs zero labeled target samples and no target test-driven hyperparameter selection.

**System gate:** if a forecasting/beam title is used, demonstrate system utility (e.g., improved beam decision outcomes under a fixed policy and feedback overhead), not only better surrogate CSI NMSE.

**Reproduction gate:** a separate environment regenerates published tables from immutable predictions and source hashes within documented numerical tolerance.

---

## 8. Results package — every required paper table, figure and data artifact

**Numbers are intentionally blank.** Populate tables only after validated experiments. Where a target is out of scope, mark `NOT_RUN/NOT_OBSERVABLE` with reason instead of adding fake values or zeroes.

### 8.1 Tables (suggested camera-ready result ledger)

| Table | Contents and required stratification | Source experiments |
|---|---|---|
| TBL01 | repo/thesis reproduction: reported vs rerun MLP/ResNet/VGG/RF for LOS/NLOS & 10-class, config, confidence | E00 |
| TBL02 | version/standards/scene/receiver compatibility and native test evidence | E01–E05 |
| TBL03 | datasets: independent scenes, path-mechanism counts, number of trajectories, geometry assets, real sessions, effective N | E04/E07 |
| TBL04 | link distributions: LOS-mixed/direct-only/NLOS mechanism/outage, ratio distributions, SNR/distance/bandwidth | E07 |
| TBL05 | LOS geometric vs observable direct probability: AUROC/AUPRC, calibration, prevalence, OOD | E06/E09/E13 |
| TBL06 | joint physical scalar outputs: delay spread, delay, direct ratio, dominant path power, path count (masked by observability) | E09/E11/E13 |
| TBL07 | CSI/PDP reconstruction: LS/LMMSE vs MLP/CNN/CVNN/proposed, ideal and realistic RX conditions | E05/E10 |
| TBL08 | main proposed innovation: mechanism-off/on paired held-out-scene improvements, uncertainty and inference cost | E11 |
| TBL09 | full A/B/C/D/E + multi-scene transfer matrix with grouped variance; physical vs proxy label separated | E06/E13 |
| TBL10 | phase/timing/array/path-index symmetry and identifiability near-limit tests | E12/E20 |
| TBL11 | external *real* held-out measurement results, dataset conditions and calibration; if absent mark clearly | E14 |
| TBL12 | actual re-traced blockage interventions and counterfactual state/power prediction | E15 |
| TBL13 | horizon-specific future CSI/blocked-path/Doppler results | E16/E17 |
| TBL14 | beam decisions: top-K, regret, beam failure, throughput/BLER vs feedback cost | E18 |
| TBL15 | 50/80/90/95% interval coverage/width, ECE, Brier, AURC under OOD scenes | E19 |
| TBL16 | architecture/loss/input/supervision ablations, MLP vs CNN-on/off, statistical power | E08/E20 |
| TBL17 | generalization stress (SNR, pilot density, material, antenna, carrier, clock) and failure rates | E20 |
| TBL18 | training/inference cost, parameters, GPU memory, energy if measured; per-model equivalent optimization budgets | E21 |
| TBL19 | sample efficiency and 5/10-seed variation; independent scene-specific CI | E22 |
| TBL20 | final reproducibility & validation audit, negative results and every excluded data fraction | E24 |

### 8.2 Figures — mandatory and conditional

| Figure | What it must show | Anti-misleading requirement |
|---|---|---|
| FIG01 | whole system: simulator/real receiver → complex pilot obs → latent state → physical decoder → heads/uncertainty → decisions | clear dashed boundary between inference-visible and oracle-private fields |
| FIG02 | original 2025 model vs new architecture, blocks and trainable parameters | show true MLP/CNN contribution and matched budgets |
| FIG03 | 3GPP TDL/CDL-A–E power delay profiles + scenario-model vs ray-traced examples | axes physical units and different label semantics clear |
| FIG04 | RT scene family grid and sample traced paths: LOS mixed, weak LOS, NLOS and outage | authentic renders from actual saved geometries, never decorative stock |
| FIG05 | complete solver convergence sweep (path powers/count/DS vs samples/depth) | include convergence failures and Monte Carlo spread |
| FIG06 | one channel: ray oracle delays/power, filtered CIR/PDP, pilot mask, noisy CSI, estimator output | same frequency/bandwidth/reference; avoid overlay mismatched grids |
| FIG07 | difficulty map: path separation × relative power × BW/SNR; estimator error & uncertainty | show physically unresolvable region, not only average |
| FIG08 | baseline LOS receiver ROC/PR curves with confidence and scene facets | annotate prevalence; thresholds from validation only |
| FIG09 | per-target scalar errors and CIs across scenes (DS, first path, K ratio, path count) | unavailable targets excluded with sample counts |
| FIG10 | CSI reconstruction NMSE vs SNR, pilot density and bandwidth | show both absolute and dB definitions; identical pilots |
| FIG11 | scene-transfer heatmap (train source ↔ test scene) | every cell independently defined; no leakage |
| FIG12 | main result: paired model-on/off improvement per held-out scene | show individual scenes and effect uncertainty, not only aggregate bar |
| FIG13 | gauge/phase/timing test: predicted/target transformation residual | time-shift equivariance only for defined tasks |
| FIG14 | learned path components and residual reconstruction vs oracle on curated analytical/RT cases | paths matched or represented as distributions; avoid overclaiming exact identity |
| FIG15 | 90/95% interval reliability, error vs predicted uncertainty, selective risk coverage | report accuracy of claimed coverage on shift |
| FIG16 | genuine counterfactual blocker insertion/leaveout: geometry changes → coherent response | not just deleting a direct path from old CIR |
| FIG17 | temporally resolved channel tracks and actual path birth/death | physical time and solver cadence correct |
| FIG18 | future CSI NMSE/beam regret/blockage event metrics vs prediction horizon | include persistence & classical tracker baselines |
| FIG19 | real dataset held-out error/calibration and qualitative examples | permitted data only; realistic failure examples included |
| FIG20 | Pareto frontier: fidelity/generalization/calibration vs compute or latency | no unsupported energy calculations |
| FIG21 | failure taxonomy montage with explanation (near wall, weak direct, cross-scene material, clock error) | include genuine failures, not only successes |
| FIG22 | dataset split map/trajectory overlap audit and label/selection distributions | no release of private geolocation without approval |

**Minimum conference-sized final selection:** FIG01, FIG04, FIG06, FIG07, FIG10, FIG11, FIG12, FIG15 (+ FIG18 only if forecasting is in title), TBL03, TBL06, TBL08, TBL09, TBL15 and TBL16. Full supplementary materials retain all diagnostics.

### 8.3 Machine-readable result payload (schema proposal)

```yaml
result:
  study_id: paper_v1
  experiment_id: E11
  run_id: immutable
  status: {PASS,FAIL,NOT_RUN,INVALID,NOT_OBSERVABLE}
  scientific_claim_status: {diagnostic,development,confirmatory,external}
  provenance:
    git_sha: exact
    code_hash: sha256
    dependency_lock_hash: sha256
    dataset_hashes: [sha256]
    scene_hashes: [sha256]
    metric_code_hash: sha256
    label_semantics_version: v1
    runtime_environment: measured
    seeds: [integers]
  split:
    train_scene_ids: [ids]
    validation_scene_ids: [ids]
    test_scene_ids: [ids]
    groups: {independent_scenes: count, trajectories: count, links: count}
    test_lock_hash: sha256
  model:
    algorithm_id: registered_name
    config_hash: sha256
    checkpoint_hash: sha256
    input_permission_profile: receiver_only
    params: integer
    train_compute: measured
    inference_latency: measured
  metrics:
    geometric_los: {auroc: null, auprc: null, brier: null}
    delay_spread_ns: {mae: null, rmse: null, ci95: null}
    relative_first_path_ns: {mae: null, ci95: null}
    csi_nmse_linear: null
    csi_nmse_db: null
    interval_coverage_90: null
    interval_width_90: null
    future_metrics: null
    per_scene: [{scene_id: ..., metrics: {}}]
  exclusions:
    solver_failures: integer
    oracle_unknown: integer
    out_of_coverage: integer
    target_unobservable: integer
  artifact_uris:
    predictions: immutable_path
    weights: immutable_path
    plots: [paths]
    logs: [paths]
    qa_report: immutable_path
```

Metrics `null` are **not** zero. Make the default result renderer fail if it attempts to present a missing metric as a number. All figures/tables generated programmatically from this result payload, never hand-typed from visual output.

### 8.4 Required “all results” folder contract

```text
outputs/studies/paper_v1/
  00_frozen_design/preregistration.md tests_lock.sha256 claims_matrix.md
  01_environment/environment_doctor.json dependency_lock.sha256
  02_assets/scenes_manifest.csv scene_renders/ labels_audit/
  03_data/raw_manifest.json curated_manifest.json splits/ provenance/
  04_native_validation/sionna_api_contract.json physics_qa.json receiver_qa.json
  05_baselines/E00...E10/ checkpoints/ predictions/ metrics/ logs/
  06_new_method/E11...E15/ checkpoints/ predictions/ metrics/ logs/
  07_forecasting/E16...E18/ checkpoints/ predictions/ metrics/ logs/
  08_reliability/E19...E23/ checkpoints/ predictions/ metrics/ logs/
  09_final_locked/E24/ final_test_manifest.json audit.json
  tables/TBL01.csv ... TBL20.csv
  figures/FIG01.pdf ... FIG22.pdf
  supplementary/all_scenarios.pdf statistical_analysis.md negative_results.md
  manuscript/manuscript.tex or manuscript.md, bibliography.bib
  reproduction/REPRODUCE.md artifact_inventory.json final_checksums.txt
```

These are *planned* assets, not files already present. Do not create meaningless empty placeholder figures as if they were results; for an unrun experiment provide a tracked to-do with explicit status.

---

## 9. Validation, CI and exhaustive failure-mode checklist

### 9.1 Automated contract tests by layer

**Version/API:** incompatible major/minor Sionna versions fail loudly, no implicit fallback to legacy TensorFlow RT; verify PyTorch tensor device/precision, Mitsuba variant, shape/channel ordering, polarizations, out types and actual optional method availability.

**Numerics/physics:** seconds/ns, Hz/GHz, W/dBm/dB, complex64/complex128, FFT sign convention, `2π`, centered/unshifted subcarrier indexing, CP and OFDM symbol duration, path carrier/baseband phase, duplicate/zero/negative delays, per-antenna/polarization axis, normalization, finite physical power, K-factor definition, velocity sign, scene refraction/reflection material constraints, path masks/padding, diffuse stochasticity, GPU/CPU precision tolerance.

**Pilot/receiver:** known pilot constellation normalization, pilot REs after guard/DC removal, null tones, DMRS power boost and symbol index, interpolation at edges, antenna/stream collisions, LMMSE noise covariance, no double AWGN, noise reference relative vs absolute, CFO sign/units, timing synchronization, AGC, RF bandwidth, quantization saturation, time/frequency sampling consistency, BER/EVM sanity.

**Targets:** path count excluding invalid padding, target masks, consistent geocentric vs local coordinates, direct oracle/collision oracle agreement, corrected phase/time gauge, no target computed from feature-only data with future samples, no exact-ray path loss from band-limited CSI, filter/window PDP vs path impulse differences, circular angle average, NLOS/outage code paths, physical class base rates.

**Learning:** deterministic split snapshots, fit scalers on training only, masks cannot encode labels by source or class, model cannot consume oracle fields, optimizer/checkpoint reload equality, multi-task loss not silently dominated by units, gradients finite, complex autograd tested, no `NaN`/`Inf`, no train/test callbacks, seeds per simulator/model independent, early-stopping from dev only, optimization budget fair, ablation turns off intended module without altering unrelated settings.

**Evaluation:** one unit test per metric on a hand-computed synthetic record, ROC/AUPRC undefined handling, prevalence reporting, confidence interval effective independent N, prediction alignment to IDs after merges, thresholds frozen, no best-seed-only reporting, data hash integrity, null-value handling, aggregation (mean of ratios vs ratio of means) documented, figure axes correct, no inconsistent dB conversion.

**Assets/license/security:** scene source provenance, no restricted location footage in artifact, no private secrets in config/logs, training data license and model weight redistribution allowed, human audit of scene renders, no duplicate split assets hidden under different names.

### 9.2 End-to-end reproducibility gates

| Level | Workload | Blocking requirement |
|---|---|---|
| L0 | pure numerical + schema unit tests | all pass; static lints; exact result hash/manifest checks |
| L1 | native free-space/two-ray Sionna smoke | pass analytical tolerance; version/API confirmed |
| L2 | ray geometry/material mechanism fixtures | ground truth audited and solver convergence quantified |
| L3 | end-to-end NR PUSCH waveform + receiver | pilot mask, estimator, synchronization acquisition/correction, and meaningful faded-link decoding curves validated |
| L4 | multi-scene pilot study | data QA and independent splits frozen |
| L5 | controlled main training and baseline evaluation | matched budgets, all predictions persisted |
| L6 | external held-out-scene and/or measured tests | no post-test tuning; statistics valid |
| L7 | second-environment reproducibility | generates every claimed figure/table from immutable results |

**Current observed status:** L0 passes locally: the full pinned Sionna/CUDA suite passes at 86.80% coverage, Ruff lint/format pass on 102 maintained Python files, mypy passes on 62 source files, and lock/build checks pass. Joint gain/delay local Fisher diagnostics have tests for rank, close-path conditioning, invalid inputs, noise scaling and agreement with the closed-form single-path CRLB with an unknown complex gain; this remains a numerical diagnostic, not a receiver or empirical resolution guarantee. GitHub Actions passed for prior source commit 26851a4 across Python 3.11/3.12 core, CPU-neural, and package-build jobs; commit 06f2a81 exposed a CPU-only coverage shortfall (73.19%, with all 175 tests passing and 7 GPU tests skipped). Follow-up commit b00e127 marks the CUDA-only adapter body as native-tested and adds CPU tests for result assembly; the CPU-emulated suite reaches 76.91%. Actions run 39582145663 passed all jobs for the coverage fix, and run 39582281995 passed all jobs for source commit 69d1df8. L1 direct/two-ray analytic checks pass. L2 has a controlled material-sensitivity fixture, but solver convergence and broad path-mechanism QA are incomplete. L3 has flat-channel DMRS and Rayleigh loops, frequency-grid BER/BLER sanity, cyclic-prefix CFO estimate/correction, bounded known-origin integer timing acquisition/correction, a static two-tap sample-domain PUSCH AWGN trend, and a time-varying sample-domain TDL-A waveform curve. The 1,024-codeword-per-SNR sample-domain run gives BLER 1.0, 1.0, 0.0225, 0.0010, and 0.0010 at 0, 6, 12, 18, and 24 dB; the last two points each have one block error and are statistically indistinguishable at this sample size. The configured 13-sample channel memory equals the shortest CP, so this run does not test channels longer than the CP. The frequency-domain TDL-A curve remains separately recorded. The paired 512-codeword fixed-allocation mobility stress run gives BLER 0.023 at 12 dB at 3 m/s versus 0.578 at 30 m/s; codeword-paired 95% bootstrap intervals for the treatment-minus-control deltas at 12/18/24 dB exclude zero, but inference remains conditional on this TDL-A simulator and single receiver configuration. Blind NR frame synchronization and fractional timing remain open. A deterministic matched fixed-payload CP-delay sweep now spans in-CP, near-boundary, boundary, and beyond-CP echoes for one configuration; broader CP-margin coverage, antenna/numerology breadth, and independently qualified link curves remain open. L4 now has a script-reproducible three-scene leave-one-scene-out pilot, but no frozen qualified dataset; all three held-out AUROCs are near chance and three scene families do not support reliable scene-level intervals. L5 has a matched seed/split evaluation pipeline and leave-one-scene-out baseline runs, but no registered primary method, compute-budget matching, or complete baseline table. L6 has no independent measured evidence, and L7 has not been performed. Native batches of 4,096 realizations for each TDL-A–E profile match Sionna's nominal delays and mean powers within the stated Monte Carlo tolerance; the 8,192-realization TDL-A tail check is an implementation sanity check, not 3GPP certification. The TDL-to-CDL profile-proxy pilot and three-scene RT pilot are reported above with hashes, split identities and limitations. Receiver-visible NR records have typed contracts and content-addressed storage; spatial, CFR-fingerprint and temporal-overlap audits exist, but the duplicate criteria have not been qualified against a production dataset. Sionna RT CUDA remains unavailable under WSL OptiX. Full CDL/tail validation, broader qualified geometry data, blind/fractional timing synchronization, broader sample-domain link characterization, target semantics, main-method ablations, external measurement and cold independent reproduction remain open.

### 9.3 Severity-1 failures (invalidate a claim, stop experiment)

- Geometric LOS treated as the same label as standardized profile type without explicit `proxy` semantics.
- Absolute delay reported despite normalized delay inputs and no timing reference.
- Classifier input includes unobservable path truth, source profile, scene label, future CSI, hidden geometry or test statistics.
- Cross-domain test includes same scene/trajectory or visually identical procedurally duplicated assets in both training and test.
- Data generator selects only detectable multipath and silently discards link outages; results then marketed as all-link performance.
- All dominant path terms normalized to unit power while claiming physical absolute power/K/coverage or outage performance.
- Native RT yields physically incorrect path count/timing/labels but training is run anyway.
- Tensor dimensions flattened/reshaped incorrectly, so antenna/spatial/channel ordering is not conserved.
- “Causal blocker” examples created by deleting one path with no scene re-trace and then advertised as real physical interventions.
- Claimed 5G-compliant channel estimator is the generic, nonstandard diagnostic comb LS.
- Paper cites `69 tests passed` as evidence of real-ray-tracing accuracy.
- Evaluation picks best results from the held-out target scene without a new independently held-out evaluation.
- The user or agent manually inserts imagined/placeholder accuracy, error or speed values into the manuscript.

### 9.4 Change-control policy for tiny implementation changes

Every change to units, formula convention, `normalize_delays`, antenna ordering, channel mask, target oracle, material defaults, Sionna flags, receiver pilot grid, split IDs, noise definitions or metrics aggregation triggers:

1. A changelog entry with rationale and expected affected results;
2. New semantic version for the relevant artifact contract;
3. Targeted tests and numerical golden/reference figures;
4. Regeneration of impacted datasets and/or predictions under new hashes;
5. A comparison against frozen baseline; investigation of unexpected differences;
6. Re-execution of affected scientific experiments and uncertainty analysis;
7. A paper claim-status update. **Never silently patch results in place.**

---

## 10. Implementation phases, dependencies, verification and stop/go decisions

A plan covering everything is not the same as performing everything simultaneously. Each phase produces an objectively inspectable artifact and fails closed when its gate is not met.

### Phase 0 — Freeze and audit (P0, immediate)

**Tasks:** archive exact ZIP hash; record repo tree and 69-test state; commit/tag clean snapshot; capture original thesis tables with accurate provenance; map old notebook preprocessing; inspect `uv.lock`; environment doctor; enumerate native Sionna 2.2 API and material uncertainty; license audit. Build `paper/preregistration.md` and `paper/claims_matrix.md` initially marked hypothesis-only.

**Outputs:** baseline fingerprint, dependency matrix, result-provenance rulebook, migrated historical reproducibility ticket list.

**Pass:** repeatable L0 tests; no assumptions about real RT or historical reproducibility remain implicit.

### Phase 1 — Physical truth/units and native Sionna contract (P0)

**Tasks:** implement canonical `PathState`, `ChannelObservation`, `TargetDefinition` schemas; test real `PathSolver`, path/cir/cfr/taps, units, polarization/shape, masks and oracle across SISO/MIMO. Write analytical one-/two-ray tests. Implement full geometric LOS vs detectable-direct vs outage semantics. Stop normalizing all delay oracle channels to first arrival if absolute timing planned.

**Outputs:** `native_contract_report.json`, physics fixtures and figures, oracle audit, `target_version=1`.

**Pass:** all golden physics tests and no runtime version mismatch, including at least one verified mixed LOS channel.

### Phase 2 — Standardized and RT data generator integrity (P0)

**Tasks:** TDL/CDL and UMi/UMa/... generator variants, procedural scene QA, independent scene family assets, convergence sweeps, material settings and environmental limits; geometry/scene manifests; temporal dataset skeleton. Explicitly support no-coverage and censored path states. Freeze per-scene sampling design.

**Outputs:** D0–D4 data manifests, distribution report, split registry, QA render gallery and generation reproducibility script.

**Pass:** no data source/label leakage, physical convergence within prespecified tolerance and scene labels independently audited.

### Phase 3 — NR receiver and timing calibration (P0)

**Tasks:** fully validated Sionna PHY 2.2 PUSCH/DMRS and LS/LMMSE pipeline; implement source-specific observation masks/error covariance and nonideal physical impairments; preserve real waveform sampling metadata. Enable reliable measured ToA only if clock/synchronization available.

**Verified subset (edition 3.0):** for a 2-PRB, 30-kHz Sionna PUSCH grid, native OFDM modulation/demodulation is an identity within numerical tolerance. Injected 0.05-subcarrier residual CFO and a +1-sample timing error (inside the 3-sample cyclic prefix) both increase DMRS-LS channel-estimate error; impairment parameters remain private simulation inputs. A separate frequency-grid PUSCH decoder test checks that BER and CRC-derived BLER decrease from complex AWGN variance 1.0 to 1e-3. This is a synchronization/decoder sanity check, not a complete time-domain faded-link curve or receiver-capability claim.

**Bounded timing acquisition (edition 3.5):** a CP-repetition detector scans a known-origin capture over a configured nonnegative integer-delay range and aggregates normalized CP-to-tail correlation across complete symbols. Deterministic batched tests recover the injected sample delay with residual CFO present. A native 2-PRB PUSCH waveform test acquires a two-sample delay and corrects it back to the original grid. The capture must include the first symbol; this does not implement blind frame detection in a continuous stream, PSS/SSS acquisition, fractional-delay estimation, or a standards-complete synchronizer. Multipath use requires a guard length consistent with the channel delay spread.

**Outputs:** L3 receiver test report, pilot masks, CSI NMSE/EVM/BER/BLER plots, receiver feature export.

**Pass:** receiver baseline passes numerical and link-level sanity across SNR/bandwidth/array settings.

### Phase 4 — Matched legacy + modern baselines (P0)

**Tasks:** reproduce 2025 results if feasible; train MLP/CNN/CVNN hybrid capacity grid; classical multipath estimators; test profile-vs-geometric leakage; freeze input standardization and strong single-task/multitask baselines. Save negative controls.

**Outputs:** E00/E06/E08–E10 complete baseline suite, honest “does CNN help?” answer with paired CIs.

**Pass:** at least one strong MLP and one high-quality classical estimator beat naive/degenerate controls, fair-compute study ready.

### Phase 5 — Invent and validate one novel mechanism (P0)

**Tasks:** start with M2 identifiability analysis on analytical two-path measurements; mathematically define resolvability/equivalence classes, build differentiable decoder and uncertainty; implement same-backbone on/off. Add symmetry M1 only if tested; make novel method independent of Sionna-private path truth at inference.

**Outputs:** E11/E12 tests, mathematical proposition with assumptions, paired held-out-scene result, attribution ablations.

**Pass:** well-calibrated uncertainty and meaningful held-out improvement **beyond capacity-only MLP**; otherwise pivot before adding more heads.

### Phase 6 — Cross-domain robustness (P0)

**Tasks:** train D1/D2/D3/D4 source variants; independently held-out scene family testing, material/frequency/device/receiver impairments, *zero-shot* vs adaptation protocols strictly separated. Run full statistical evaluation.

**Outputs:** transfer matrix, CIs, calibration plots, failure breakdown, claim-limits memo.

**Pass:** independent-scene evidence supports the stated claim; otherwise restrict claim to controlled simulation.

### Phase 7 — Dynamic RT and forecasting (P1; conditionally flagship)

**Tasks:** recreate dynamic scenes per frame, validate Doppler/time semantics, learn future CSI and blockage, compare persistence and physically based baselines. Pair counterfactual geometry interventions with statistical controls. Decide whether beam prediction adds meaningful system value.

**Outputs:** E15–E18, exact trajectories/interventions, time-to-event and horizon curves, beam decision results if built.

**Pass:** demonstrable downstream utility under a fixed intervention/feedback scenario, or move to separate future-work paper.

### Phase 8 — Independent measurement campaign (P1 but pivotal for journal strength)

**Tasks:** secure measurement rights, survey positions/LOS, record synchronized or explicitly unsynchronized NR signal observations, calibrate antennas/time, test source-only model untouched, assess domain bias and uncertainty. No target-domain leakage.

**Outputs:** real-world data card, measurement QA, E14 locked evaluation, reproducible anonymized statistics.

**Pass:** enough independently surveyed sites/sessions for a meaningful external evaluation; if not, paper must say simulation-based and avoid inflated claims.

### Phase 9 — Full sensitivity/cost/reproduction (P0 before submission)

**Tasks:** E19–E24, architecture ablations, failure/uncertainty cases, multi-seed scene bootstrap, memory/latency, evidence integrity checks, second host/system reproduction.

**Outputs:** TBL01–20 and FIG01–22 where relevant, raw metrics, generated supplementary results, final signed audit.

**Pass:** every manuscript claim maps to a specific immutable result and can be regenerated without privileged data access except documented private assets.

### Phase 10 — Manuscript and review-readiness (P0)

**Tasks:** related-work matrix, formal contribution statement, experimental protocol, definitions/limitations, statistical analysis, reproducibility checklist, ethical/data licenses, controlled reviewer-style attacks, publication venue fit, arXiv/supplement under policies.

**Outputs:** paper, bibliography, supplementary, code/data release artifacts, “what we did *not* prove” section.

**Pass:** a skeptical wireless+ML reviewer can identify data origins, observables, target definitions, split independence, fairness, and calibrated negative findings without asking for missing fundamentals.

### 10.1 Critical dependency graph

```text
Phase 0 repository+standards freeze
   └─> Phase 1 native RT/physical truth
          ├─> Phase 2 data / scene / label validation
          │      └─> Phase 4 baselines
          │             └─> Phase 5 main method
          │                     └─> Phase 6 cross-domain final tests
          │                              ├─> Phase 7 dynamic forecasting (conditional)
          │                              └─> Phase 8 real measurements (parallel, if feasible)
          └─> Phase 3 5G NR receiver validation ──> Phase 4+5+6
                                       all ──> Phase 9 statistics/reproduction ──> Phase 10 paper
```

**Hard block:** no full-scale model training until Phase 1 + Phase 3 have validated native source and receiver contracts. Debug algorithm work may occur in parallel on analytical data but remains `DEBUG_ONLY`/`DEVELOPMENT`.

### 10.2 Relative effort / hardware budget (planning estimate only)

| Workstream | Relative effort | Why it is expensive | Scale-up strategy |
|---|---|---|---|
| API/physics/test infrastructure | medium | many subtle numerical/semantic bugs | CPU analytical tests first, one canonical native scene |
| High-fidelity RT data | very high | geometry × ray sample budget × path depth × trajectories | benchmark links/s first, cache immutable paths, solve unique scenes in batches, sample design |
| Real NR receiver | high | waveform truth, MIMO/DMRS and synchronization details | build single-link SISO first, verify, then MIMO |
| Path inference model | high | complex autodiff + set matching + uncertainty | two-path toy → 3D small → large domains |
| Baselines/ablation/statistics | high | many fairness-preserving runs | pre-register compact candidate shortlist; parallel seeds |
| Measurement campaign | potentially very high | equipment, licensing, surveying, manpower | begin permission and collaborator search early |
| Paper/replication | high | re-run every figure with locked study artifacts | automatic plots and result schema from day one |

**Do not invent GPU-hour totals before a native benchmark.** Record representative: per-scene load time, rays/second, links/second, memory by solver depth, per-epoch batch time, checkpoint size, and bytes/link. Fit empirical cost model `C≈N_scenes×N_links/throughput + N_seeds×N_models×training_time`, add rerun safety margin explicitly after actual measurements. An H100 allocation, V100 node or local 6 GB VRAM is only usable if its kernel/toolchain/VRAM validation passes; CPU-only RT may require a separate LLVM variant and could be orders slower.

### 10.3 If constrained by time or compute: minimum viable strong paper

**Do not sacrifice the physical correctness gates.** Reduce scope, not integrity:

- D0 analytical + D1 standardized + 3+ independently credible RT scene groups and optionally D5 real;
- actual NR PUSCH-DMRS receiver, SISO and one calibrated MIMO regime;
- core tasks: LOS vs direct detectability, delay spread, relative first path, band-limited PDP/CSI reconstruction and uncertainty;
- strong classical + same-input MLP + real CNN + CVNN + proposed identifiability-off/on;
- cross-domain held-out scenes + calibrated uncertainty + fair-compute ablations;
- no long-horizon blockage or beam claim if temporal RT and real-world validation cannot be achieved.

This is a stronger paper than a nominal “foundation model” with dozens of unverified heads.

---
## 11. Risk register, mitigations and scientifically acceptable pivots

| Risk | Severity | Detection | Required mitigation / pivot |
|---|---|---|---|
| False novelty: multipath tokens, CVNN, ToA multitask or symmetry already published | Critical | living related-work/component matrix | define a precise **new mechanism** and evaluate against closest prior; adjust title if overlap discovered |
| TDL/CDL profile labels mistaken for physical LOS | Critical | explicit label scopes, profile-only oracle | primary geometric claims only from independently labeled RT/measurement domains |
| Absolute first-path delay lost through `normalize_delays=True` | Critical | fixed phase/time reference tests | preserve absolute oracle and explicit synchronized observable; otherwise use relative first path |
| Sionna 2.2 API actually differs from mocked contract | Critical | native smoke over all formats | fix versioned adapter and update golden tests **before** data generation |
| Ray material default prevents scattering or transmission case | High | measured mechanism-energy histograms | adjust physically justified material/pattern and validate; not flags alone |
| Incorrect phase/baseband convention | Critical | two-ray analytical golden test | verify `Paths.cir`/CFR and Fourier sign/carrier phase, recompute affected datasets |
| Sparse sampling misses weak paths, mislabels no reception | High | solver convergence and alternative samples/depth | report unknown/censored; increase solver budget only where supported |
| Scene geometry unrealistic yet reported as field accurate | High | visual/material/geometry QA + external comparison | call controlled synthetic; require measured/material-calibrated scenes for realism claim |
| MLP beats proposed CVNN | Medium scientifically, high for chosen marketing claim | matched ablation E08/E11 | discard unnecessary CNN; contribution should stand on identifiability/decoder rather than convolution |
| Novel method no better than classical/MLP | Critical to claim | E11 held-out paired CI | publish methodological limits/negative result or redesign with prespecified fresh test |
| Conditional LOS sampling inflates binary accuracy | High | raw prevalence and sample retention | evaluate representative distribution, report diagnostic balancing separately |
| Direct power or path counts unavailable from limited BW | High | analytic identifiability and receiver resolutions | predict broad posterior / resolvable clusters, not false exact oracle quantities |
| Counterfactual intervention only deletes coefficient | Critical | scene geometry intervention replay | re-trace blocked scene; otherwise call it a mathematical ablation |
| Forecasting uses only Doppler snapshot extrapolation | Critical to blockage claim | actual moving obstruction path birth/death | recompute geometry over time; narrow claim to short-coherence extrapolation if not done |
| Delay spread regression trivial from perfect oracle | High | ideal and realistic receiver comparison | use incomplete/noisy pilots, external held-out scenes, analytical estimation baselines |
| SNR noise definition wrong | Critical | thermal vs relative noise tests | preserve both named modes and record model/measurement context |
| Array alias / synthetic-array far-field validity | High | aperture/distance criteria and non-synthetic comparison | use proper geometry or restrict validity claim |
| MIMO phase calibration hidden | High | per-antenna nuisance and calibration tests | calibrate or widen uncertainty and disable absolute angular targets |
| Test leakage through near-identical geometry | Critical | scene-family & trajectory-hash/nearest-neighbor audit | replace contaminated test and invalidate old results |
| Multiple comparisons/best-seed cherry-picking | High | preregistered contrasts/power, all seeds persisted | fresh locked confirmatory test and adjusted inference |
| Real dataset cannot be legally shared/used | High | early written permission | simulation paper, truthful limitation, use public licensed alternative |
| Hardware runtime incompatible (V100/CUDA/Dr.Jit/WSL) | High | native smoke with exact environment manifest | separate compatible machine/backend; never silently switch physics model |
| Path matching unstable or approximate diffuse continuum | High | sensitivity/matching ablations | use distributional spectral objective and uncertainty, not hard per-ray labels |
| Model becomes too large/slow | Medium–high | Pareto and memory profiling | compress, prune outputs, keep MLP-classical fallback |
| Single data generator accounts for all scenes | High | independent asset-family splits | add independently authored/collected family; decrease scope until obtained |
| Timing/event labels shifted by clock drift | Critical to time claims | paired timestamp/clock QA | resynchronize from recorded signals, mask events where ambiguous |
| Negative results hidden | Critical to scientific ethics | required exclusions/failure artifacts | publish all prespecified comparisons, limitations and full sweep indices |

### 11.1 Pivot decisions based on outcomes

- **If no robust direct-path detection:** focus on **calibrated channel-state uncertainty** and explain physically unobservable LOS, rather than chase artificial binary accuracy.
- **If MLP dominates all CVNNs:** use MLP as the encoder and contribute a new identifiability-aware physical decoder/loss; prove CNN unnecessary in this observation regime.
- **If true per-ray count/angles do not generalize:** predict **band-limited PDP/dispersion and resolvable clusters** with uncertainty, not exact electromagnetic ray paths.
- **If standard profile pretraining harms RT transfer:** quantify negative transfer and design/test an independent approach; do not force standardized models into the pipeline.
- **If real data cannot be obtained:** target a rigorous synthetic method/benchmark publication with explicit external validity limits; no “field-validated” abstract.
- **If dynamic RT is too expensive:** make first paper static multipath identifiability/calibration; reserve forecasting as a separate project after data/cadence/solver validity is established.
- **If novelty is incremental after literature review:** position as a careful *benchmark / empirical scientific study* rather than misstate a methodological breakthrough.

---

## 12. Manuscript outline and claim-to-evidence traceability

**Working title (subject to results):**

> *Learning Only What the Radio Channel Reveals: Identifiability-Aware Multipath Inference from Sparse 5G NR Pilots Across Unseen Environments*

**Alternative title if dynamics succeed:**

> *Physics-Consistent Multipath State Inference and Blockage Forecasting Under Cross-Environment Wireless Channel Shift*

Avoid “foundation model” unless broad pretraining transfer, compute/data scale and multiple downstream tasks justify it. Avoid “phase-equivariant” in the title unless the symmetry class is precisely defined and validated. Avoid “real-time” unless actual supported-device latency meets a documented online inference constraint.

**Abstract template (fill only after experiments):**

> Estimating propagation state from sparse pilot measurements is fundamentally limited by [specified physical ambiguities]. We present [precisely original method] that [mechanism, not architecture buzzwords]. On [N independently held-out scenes and any measured sites], we compare against [strong classical/MLP/complex baselines] under [realistic 5G NR observation]. We observe [validated primary effect with CI and units] and [calibration/robustness result], and show [downstream benefit if proven]. We identify failures under [ambiguous regimes]. Source code, configurations, and reproducibility artifacts are [release statement].

**Section outline:**

1. Introduction: why channel profile recognition is not real propagation-state inference; research problem and evidence gap.
2. Standards, observation model and identifiability: 3GPP TDL/CDL, geometry-based RT, NR pilot CSI, physical label and timing definitions.
3. Related work: classical MPC estimation, ML LOS/ToA, complex-valued networks, symmetry, path foundation models, sim→real.
4. New method: identifiable state posterior, observation-consistent decoder, new loss, mathematical properties and limitations.
5. Experimental design: scene sources/materials/solver convergence, receiver realism, data splits, oracle audit, baseline fair-compute settings, preregistration/statistics.
6. Main experiments: direct/dispersion/CSI estimation, transfer, identifiability, uncertainty, ablations; with measured validation where available.
7. Conditional forecasting/beam section: only if supported by genuinely dynamic ray traces and receiver-level decision improvements.
8. Limitations: gauge freedoms, weak paths, finite solver/scene fidelity, external validity, standards/version dependence, compute and rights.
9. Conclusion: quantitatively supported results and concrete next steps.
10. Supplement: native API tests, scene-level counts, all hyperparameters/seeds, negative results, numerical derivations, additional plots.

### 12.1 Claim matrix — every strong sentence must resolve to artifacts

| Candidate claim | Minimum evidence | Invalid if |
|---|---|---|
| “Geometric LOS is harder than profile classification” | E06 profile-only vs E13 geometric RT with independent label semantics | same proxy y used for both |
| “CVNN extracts additional multipath structure” | E08 matched MLP/real CNN/CVNN with encoder off, equal inputs | compares different data, training or parameter budgets |
| “Identifiability-aware learning is original and improves transfer” | formal formulation + related-work novelty audit + E11 paired external scenes | only appends ordinary calibration/head |
| “Phase-equivariant” | proof/property + E12 numerical transform tests | arbitrary global rotation only used as train augmentation and miscalled exact equivariance |
| “Estimates ToA” | known clock/synchronization oracle and T07 validity | normalized first-arrival delay or no time origin |
| “Recovers physical path count” | parameter-resolution/uncertainty study E09/E11 with matched path oracle | exact RT rays assumed resolvable from insufficient pilots |
| “Realistic NR channel estimation” | validated NR 38.211 PUSCH/DMRS and decoding E05 | generic synthetic comb-pilot simulator only |
| “Generalizes to unseen environments” | multiple independent held-out scene families and scene-level intervals E13 | unseen positions in same procedural seed family only |
| “Works on real measurements” | E14 independent licensed/measured dataset with surveyed truth | Sionna RT data relabeled “real” |
| “Predicts blockage” | dynamic re-traced path changes, E17 event metrics | only Doppler extrapolation from static snapshot |
| “Improves radio performance” | E18 beam/BLER/throughput under equal pilots/overhead | only classifier AUROC improves |
| “Efficient/deployable” | target hardware latency/memory/energy E21 | reported FLOPs without runtime or hardware test |

---

## 13. Practical execution checklist (checkboxes are future work)

### Before any large training run

- [ ] Freeze repo/tag, original thesis results and ZIP SHA256.
- [ ] Record exact Sionna/PyTorch/Mitsuba/Dr.Jit and CUDA packages, material tables, TR/TS releases.
- [ ] Execute native PathSolver free-space, two-ray, multi-bounce, refraction, diffraction and diffuse smoke checks.
- [ ] Resolve `Paths.cir/cfr` units, complex tensor shape and `normalize_delays` policy.
- [ ] Implement correct physical LOS, detectability, outage and unknown labels with tests.
- [ ] Prove no metadata/oracle/scene/proximity leaks into receiver-only features.
- [ ] Validate solver sample/depth and synthetic/non-synthetic antenna behavior.
- [ ] Validate 5G NR DMRS resource grid, LS/LMMSE channel estimation and decoding.
- [ ] Verify actual channel response, noise/SINR and receiver interface with golden waveforms.
- [ ] Build independent scene families, QA images, source licenses and material masks.
- [ ] Save immutable grouped train/validation/test split manifests, buffers and hash checks.
- [ ] Pre-register one main novelty mechanism, primary endpoints, seed plan and minimum useful effect.
- [ ] Produce the strongest analytical estimator + MLP/CNN/CVNN matched controls.

### Before claiming scientific improvement

- [ ] Confirm model reconstructs physically valid complex pilots under mask and reference conventions.
- [ ] Show main new mechanism **on/off** under the same backbone/training/input constraints.
- [ ] Run all relevant gauge, physical ambiguity, weak-LOS and noise stress cases.
- [ ] Show per-scene metrics and uncertainty, not only micro-average over rays.
- [ ] Report unknown/censored/outage and selected examples; don't hide failures.
- [ ] Fit calibration, thresholds and hyperparameters only on allowed development data.
- [ ] Test independent scene family and frequency/material/receiver stress.
- [ ] Verify fairness: parameters, FLOPs/latency, tuning budget and number of examples.
- [ ] Document negative results, instability and sensitivity.
- [ ] If claiming real or temporal effects, supply independent real data or re-traced time sequences.

### Before paper submission

- [ ] Bibliography verified against original published PDFs/abstracts and 2026 prior art; no “first-ever” without systematic evidence.
- [ ] All result numbers programmatically derived from saved predictions and validated metrics.
- [ ] Figures have correct scientific units, masks, split names, error bars and labels.
- [ ] No fabricated number, fake ray plot, stock photo, convenient undocumented omission or tuned test result.
- [ ] All artifact/code/scene/data/calibration licenses and restrictions documented.
- [ ] A second environment reproduces key numeric tables and plots within declared tolerances.
- [ ] Manuscript separates standardized profile proxy, geometry truth, receiver-observable inference and measurement evidence.
- [ ] Manuscript states physical observability limits and exactly where model can be wrong.
- [ ] Reviewer checklist: strongest baselines, theory assumptions, independence, statistics, reproducibility, failed hypotheses.
- [ ] Freeze all output hashes, create an archival release, then submit to venue matching demonstrated contribution.

---

## 14. Source documentation and selected prior art

Sources below were checked or identified during preparation as of **2026-10-21**. **Official library/specification sources control technical behavior**; abstracts/preprints are pointers for novelty review, not grounds for claiming head-to-head performance. Full bibliographic verification for every new method is an explicit Phase 0/10 task.

### 14.1 Official Sionna 2.2 documentation — must-read in this order

1. **Installation and runtime:** https://nvlabs.github.io/sionna/v2.2.0/installation.html — 3 modules PHY/RT/SYS, Python 3.11+, PyTorch 2.9+, Mitsuba/Dr.Jit and LLVM CPU notes.
2. **RT PathSolver API:** https://nvlabs.github.io/sionna/rt/api/paths_solvers.html — interaction switches, deterministic mode, sampling, path depth and synthetic arrays.
3. **RT Paths API:** https://nvlabs.github.io/sionna/rt/api/paths.html — `a`, `tau`, `valid`, `interactions`, `doppler`, `cir`, `cfr`, `taps`, `normalize_delays` and return interfaces.
4. **RT Scenes:** https://nvlabs.github.io/sionna/rt/api/scene.html — `load_scene`, `Scene`, radio objects, `frequency`, `bandwidth` and material assignments.
5. **RT Scene Objects:** https://nvlabs.github.io/sionna/rt/api/scene_object.html — mesh loading/placement, orientation and velocity.
6. **RT Radio Materials:** https://nvlabs.github.io/sionna/rt/api/radio_materials.html — thickness, ITU permittivity/conductivity, scattering patterns and defaults.
7. **RT Mobility tutorial:** https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html — recomputed geometry vs local Doppler-based evolution.
8. **RT Scattering tutorial:** https://nvlabs.github.io/sionna/rt/tutorials/Scattering.html — diffuse surface physics and modeled path contributions.
9. **RT Scene Edit tutorial:** https://nvlabs.github.io/sionna/rt/tutorials/Scene-Edit.html — mesh manipulation, material changes, merge-shapes performance.
10. **RT Paths developer guide:** https://nvlabs.github.io/sionna/rt/developer/dev_understanding_paths.html — array/path semantics and mask structures.
11. **RT technical report:** https://nvlabs.github.io/sionna/rt/tech-report/S3.html — mathematical method, synthetic arrays, Doppler and propagation restrictions.
12. **RT Radio Maps:** https://nvlabs.github.io/sionna/rt/api/radio_map_solvers.html — useful for candidate receiver sampling/path-gain, not a substitute for coherent CIR.
13. **Sionna PHY TDL 2.2:** https://nvlabs.github.io/sionna/phy/api/channel/wireless/sionna.phy.channel.tr38901.TDL.html — 3GPP profile shapes, `spec_version`, scalable vs fixed profiles.
14. **Sionna PHY CDL 2.2:** https://nvlabs.github.io/sionna/phy/api/channel/wireless/sionna.phy.channel.tr38901.CDL.html — antenna arrays, propagation profiles, normalization, `spec_version`.
15. **Sionna PHY PUSCH 2.2:** https://nvlabs.github.io/sionna/phy/api/nr/pusch.html — `PUSCHConfig`, DMRS, LS/LMMSE estimators and complete receiver.
16. **NR PUSCH tutorial 2.2:** https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html — end-to-end PHY configuration and supported subset of NR functionality.
17. **Link-level RT+PHY tutorial:** https://nvlabs.github.io/sionna/v2.1.0/phy/tutorials/notebooks/Link_Level_Simulations_with_RT.html — design reference only; **adapt/test for 2.2**, never copy unverified API.
18. **Official release changelog / source:** https://github.com/NVlabs/sionna/releases and https://github.com/NVlabs/sionna-rt/releases — exact tag/commit and breaking changes, not floating `main`.

### 14.2 Standards

- 3GPP **TR 38.901** — channel models for 0.5–100 GHz. ETSI v19.5.0 directory (25 Sep 2026): https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.05.00_60/ . Record that current Sionna TDL/CDL `spec_version="19.2"` is a *separate implementation parameter*.
- 3GPP **TS 38.211** — NR physical channels and modulation (PUSCH DMRS, positioning RS where relevant).
- 3GPP **TS 38.212** — NR multiplexing/channel coding; **TS 38.214** — NR physical-layer data procedures, MCS/scheduling.
- Relevant antenna/measurement specs and ITU-R **P.2040** material model, exact applicable version to be frozen in standards manifest.

### 14.3 Publications / related directions (verify paper-specific details before citation)

- Long Nguyen (2025), *5G Physical Layer Channel Conditions Classification Using Machine Learning Techniques*, Tampere University; embedded in this repository at `docs/thesis/NguyenLong.pdf` and original public record: https://trepo.tuni.fi/bitstream/handle/10024/229296/NguyenLong.pdf?sequence=2 . **Reuse disclosure and citation mandatory.**
- *Machine-Learning-Based LOS Detection for 5G Signals with Applications in Airport Environments*, Sensors 2023: https://www.mdpi.com/1424-8220/23/3/1470 . Shows LOS detection with statistical 5G channels was already investigated.
- *Complex-Valued Neural Networks for Distributed Machine Learning Assisted RF Positioning*, MILCOM 2024, DOI `10.1109/MILCOM61039.2024.10773773` — joint LOS/first-path ToA; important direct overlap.
- *MultiPathFormer: Towards a Foundation Model for Multipath Wireless Propagation*, arXiv Aug 2026: https://arxiv.org/abs/2608.05076 . **Preprint**, path-token pretraining and multiple downstream tasks; cannot claim path-token novelty without distinction.
- *AirFM-DDA: Air-Interface Foundation Model in the Delay-Doppler-Angle Domain for AI-Native 6G*, arXiv Apr 2026: https://arxiv.org/abs/2605.00020 . **Preprint**, delay–Doppler–angle learning; compare input modalities before drawing conclusions.
- *Co-Domain Symmetry for Complex-Valued Deep Learning* (CVPR 2022), official proceedings: https://openaccess.thecvf.com/ . General complex symmetry/equivariance prior art; verify complete bibliographic details in Phase 0.
- Official 3GPP, Sionna, classical channel estimation and high-resolution parametric estimation references should be expanded in `paper/citations.bib`, rather than invented or assumed here.

---

## 15. Final project charter — the non-negotiable scientific standard

A good final system will not necessarily output a sharply resolved number for every channel quantity. It will:

1. Recover physically meaningful channel characteristics when its pilots/arrays/time window make them **observable**.
2. Reconstruct and predict the complex channel **within verified timing, phase, frequency, and antenna conventions**.
3. Produce probabilistic LOS, weak-direct, blockage and outage estimates that acknowledge the difference between **geometry** and **detectability**.
4. Estimate delay spread, relative path delays, PDP, resolvable cluster statistics, and—in suitable MIMO/time regimes—angles/Doppler, without asserting physically impossible resolution.
5. Calibrate its confidence under new scenes, materials, bandwidths and receiver distortions, and expose when a particular target is not identifiable.
6. Demonstrate a useful generalization advantage over **strong MLP** and classical estimators, or report honestly that such an advantage was not found.
7. For any forecasting claim, correctly model moving blockers and path birth/death with **time-updated geometry**, and preferably demonstrate a real radio decision benefit.
8. Generate every accepted paper number from independently reproducible and licensed research artifacts with an unbroken chain of provenance.

**Recommended core-paper focus:** **Identifiability-aware, physically consistent channel-state reconstruction and uncertainty under independent-scene domain shift.** Keep LOS/NLOS as one of the outputs. Add blockage/beam forecasting after this foundation is independently verified; a smaller correct paper is better than a large invalid one.

**Acceptance decision is made from evidence, never from ambition.** Until full RT contracts, an end-to-end synchronized NR receiver/link test, cross-environment datasets and a novel-method ablation exist, the current project remains promising research infrastructure, not an established state-of-the-art result.

---

**Plan specification complete; research execution remains open.** The following appendices provide lower-level implementation contracts, storage estimates, and an expanded exhibit registry. The current next engineering steps are RT-to-NR channel mapping, receiver timing/link validation, canonical schema/split QA and physical RT gold cases before full-scale learning.
## Appendix A. Implementation roadmap: files, prerequisites, acceptance gates and rollback

**Scope guard:** this document specifies implementation. It does not assert these future modules have been written. Preserve `archive/legacy_2025/` read-only and the user’s original experiment outputs. Never overwrite the submitted 2025 results or silently import code from a different Aalto repository without version/legal review.

### Phase P0 — freeze and record the reproducible baseline

**Deliverables**

1. Create tagged immutable baseline commit (`legacy-upgrade-base`) with source SHA256 of user-supplied ZIP, provenance, licenses and `docs/` evidence; do not treat a ZIP filename as a Git commit.
2. Capture pytest XML/JUnit, lint/type status, current dependency lock, CLI doctor report and environment. The original baseline snapshot recorded 84 passed, one optional Sionna native test skipped, and 82.13% coverage; the current edition-1.8 status is recorded in section 0.2. The CUDA known-flat PUSCH DMRS adapter test passes in the isolated Sionna 2.2 environment. These checks do not establish RT dataset validity or scientific claims.
3. Audit current decoder output against named schema; document every existing data access and object lifecycle. Identify code duplicated between `loslab/` compatibility facade and modules.
4. Establish an experiment registry in `paper/claims_matrix.md`: *claim → measurement → evidence → counterexample → uncertainty → status*.
5. Add data/label schema semantic versions to prevent old A/B/C/D/E labels from being silently treated as geometric LOS.

**Done when:** archived baseline remains executable, all existing lightweight tests pass, and any behavior difference is recorded. **Rollback:** restore tag; never corrupt the immutable archive.

### Phase P1 — native 2.2 compatibility and physical contracts (**hard stop until green**)

**Implementation order:** `integrations/sionna/version_gate.py` → `native_contracts.py` → `rt.py` and `phy.py` → `validation/sionna_native.py` → gold scenes/tests.

- Use explicit supported version range or exact lock for Sionna 2.2.0, Sionna RT 2.2.0, Mitsuba, Dr.Jit and PyTorch, maintaining distinct documented dependency responsibilities. Do not rely on `pip install -U` as a reproducible environment.
- Create API snapshots via *actual native calls* for TDL, CDL, `PanelArray`, `PathSolver`, `Paths.cir`, `Paths.cfr`, `Paths.taps`, `paths.valid` and intervention scene editing. Record shapes, axis meanings, units, dtype and complex packing.
- Fix old `normalize_delays=True` usage by a **task-explicit** option: `physical_absolute_delay` for oracle, `receiver_relative_timing` for unsynchronized observation; forbid misleading absolute time labels.
- Split stochastic standardized profile generation and RT geometry generation into different target-semantic dataclasses.
- Preserve coherent complex summation; ensure distinction between Sionna RT passband and baseband coefficients as documented.
- Implement direct-path ray-oracle tests in free space, via slab and behind blocker, with unknown/error states.
- Assert scene material assignments, normalized/unnormalized power behavior, solver convergence settings and random seeds.

**Done when:** V001–V018, V026 and relevant native contracts pass on the actual desired compute platform; captured outputs demonstrate both direct + indirect paths in LOS and legitimate NLOS multipath. **Fail/rollback:** if native version breaks, pin/revise within P1, never edit trained results or scientific labels to hide incompatibility.

### Phase P2 — independent receiver model and strict observation-only API

**Implementation order:** `receivers/nr_pusch.py` → `dmrs.py` → `masks.py` → `synchronization.py` → `impairments.py`; independent observation adapter with no geometry/RT imports.

- Start with **Sionna PHY 2.2 PUSCH DMRS**, because its official APIs include `PUSCHLSChannelEstimator` and `PUSCHLMMSEChannelEstimator`; a PRS-specific receiver is an optional **separate** implementation and validation project.
- Gold-test the legacy 3.5 GHz/15 kHz/16 PRB config before exploring larger bandwidths. Document whether original *allocation vs FFT* matches channel length.
- Implement `Observation` and `ReceiverKnownConfig` types that expose only actual receiver-known pilots, masks, estimated CSI, error variance, arrays and calibrated metadata. Do **not** pass a whole sample dictionary into `model.forward()`; it would make oracle leakage easy.
- Simulate channel, transmit symbols and receive noise with a consistent, audited noise/signal power convention; keep separate relative-SNR controls.
- Add one impairment per ablation; preserve impairment levels as private evaluation metadata unless the target deployment receiver can measure them.
- Compute BER/BLER on a closed loop before using synthetic CSI for a 5G-system claim.

**Done when:** real NR resource-grid/pilot tests pass, CSI estimator error decreases with appropriate SNR changes, and a receiver decode sanity check runs; inference signature rejects private targets/coordinates. **Rollback:** revert receiver changes while preserving pure numerical debug pilots, visibly marked `NON_STANDARD`.

### Phase P3 — canonical dataset, labels, splits and QA

**Implementation order:** `propagation/ray_path_schema.py` → `labels.py` → `targets.py` → `datasets/channel_v2.py` → `evaluation/independent_splits.py` → storage/manifest tooling.

- Define `Scene`, `Trajectory`, `Link`, `Frame`, `Sample`, `TargetMask`, `ObservationMask`, `OracleMetadata`, `MeasurementMetadata` and `DataLineage` schema types.
- Enforce typed frequency/time/angle/distance/power units; store units in schema and assert dimensional consistency in tests.
- Store the source of truth once; derived `.npz`/Zarr shards have parent content hashes. Split assignment is versioned **before** model runs.
- Add leak-resistant supergroup IDs: source asset/scene family/site/session, route, BS location, transmitter shape, mobility intervention pair, and source sample hash.
- Implement spatial duplicate detection, overlapping temporal window detection, profile-label probe and near-identical CFR fingerprint checks.
- Include no-ray/outage/solver-error/label-unknown records in manifest and denominators.
- Add standardized synthetic `oracle` fields only in labels/QA datasets not passed through inference dataloader.

**Done when:** schema round trip is lossless for complex CSI/masks/timing, split is scene-disjoint, and a deliberately leaked oracle feature is rejected automatically. **Rollback:** retain raw immutable shards; regenerate derived shards rather than patch them.

### Phase P4 — analytical and 3GPP stochastic dataset program

- Build 1-, 2-, 3-path analytical channels spanning delay separation, phase and amplitude, power, SNR and unknown synchronization; retain analytic solutions and ambiguity tests.
- Generate TDL/CDL-A–E at tracked `spec_version`; full labels include `profile_proxy` and delay-spread generation setting, not geometry truth.
- Add 3GPP **scenario-level** stochastic channels (where truly implemented and validated), including spatial consistency and prescribed LOS probability, as a **distinct data tier** from canned A–E profiles.
- Include path-based oracle distributions, channel realizations, and noisy NR receiver outputs with equal treatment of carrier/bandwidth/pilots.
- Reproduce old task inputs as a named `legacy_replication` dataset, never mix with new pilot/array configurations without labels.

**Done when:** standardized model statistics are validated against relevant TR 38.901 tables/ranges, class/profile shortcuts are quantified, and analytical ambiguity sweeps yield intuitive physical bounds. **Rollback:** fail data build and preserve source event; never clip erroneous physical numbers to look normal.

### Phase P5 — multi-scene Sionna RT data engine

- Use `scenes/` to register approved built-in, authored, surveyed and procedurally generated environments. Give each geometry family a disjoint split ID and rights ledger.
- Add authentic materials as a **separate controlled factor**; qualify each material’s parameters at each studied carrier, and record thickness/refraction/scattering.
- Generate mixed-LOS links, NLOS reflections/refractions/diffractions, weak direct components, edge/grazing cases and legitimate outage samples. Label separately by propagation mechanism and observer detectability.
- Run solver convergence sweeps for path depth, sample count, deterministic option, antennas and geometry precision. Do not assume a large path count implies higher realism.
- Produce scene visualizations and QA path overlays; save scene-relative velocities and full 3D placements; ensure antenna orientation matches array model.
- Establish raw scene and path-data caches keyed on the *entire* physical configuration + exact version hash; if ANY physical parameter changes, invalidate derived caches.
- Treat path metadata as private oracle; derive CFR/NR signal receiver observations in a separate stage.

**Done when:** hand-inspected stratified scene set has correct path mechanisms, convergence and direct-path labels; independent geometry-family test scenes are locked. **Rollback:** scene excluded from confirmatory release with reason-coded manifest; do not rerender a subset after seeing model performance.

**Edition 3.1 pilot evidence (not a phase completion):** `generate_raytraced` now selects a configurable Mitsuba variant before Sionna RT imports; its portable default is `llvm_ad_mono_polarized`, with the selected variant stored in each record. Each newly generated row now embeds its complete `RayScene`, `GridConfig`, and explicit solver settings, closing a provenance gap for future datasets. The CPU-native test generates four balanced records from `box_one_screen` and checks geometric direct-path labels and path metadata. A smoke collection under ignored `outputs/rt-cross-scene-pilot-20261021/` contains 40 links each from `box_one_screen` and `simple_street_canyon` (80 total; distinct domains and spatial groups). A held-out `rt_simple_street_canyon` run used generic noisy comb pilots (spacing 4, 12 dB), seven seeds, and RF/MLP/logistic regression. All methods share the same grouped train/validation/test indices per seed; the test partition contains 40 links from 19 groups. Mean AUROC was 0.511 RF, 0.559 MLP, and 0.544 logistic regression. Paired MLP-minus-RF was +0.047 (95% seed-bootstrap interval -0.002 to +0.096; exact sign-flip p=0.156); MLP-minus-logistic was +0.015 (-0.045 to +0.068; p=0.641). The run manifest is marked `SIMULATION_ONLY_UNVALIDATED`, labels `GEOMETRIC_DIRECT_PATH_ORACLE`, records both input hashes, and identifies the observation as generic pilot estimation rather than NR DMRS. The sample is too small and scene count too low for confirmatory inference. These built-in simple geometries, default material choices, and limited path sampling do not meet P5's scene diversity, material qualification, convergence, path-overlay review, or independent test-scene gate. Input hashes and all split indices/predictions are in ignored local outputs; the original pilot's complete generation config was not captured, so that dataset is not exactly regenerable. Future shared datasets should use the embedded config or a versioned recipe before execution.

**Edition 3.3 leave-one-scene-out pilot (not a phase completion):** `scripts/run_rt_cross_scene_pilot.sh` regenerates 40 balanced links each for `box_one_screen`, `simple_street_canyon`, and Sionna's `etoile` scene, then runs all three held-out-scene folds. Each dataset has 20 geometric direct-path and 20 NLOS labels; spatial group counts are 4, 11, and 21 respectively. Configured scene/grid/solver parameters are embedded in row metadata. The data are under ignored `outputs/rt-three-scene-pilot-20261021/`; input SHA256 values in scene order are `daa48df5e783dde5f07ee796b032b2036dc81fdcc1c6527b1e61aa77d70e98fc`, `314fc68a75d566245a9a77408c8df894c127725badc2d3b11f8aad2fdb1521a9`, and `c7888e419b8d608b7954911abfcfe1b8c86ef3fadf15f588637c5ad1dac14d61`. Each fold tests 40 links from the held-out scene; all 21 method/seed rows per fold share the same grouped split hash for a given seed. With generic noisy comb pilots (spacing 4, 12 dB), seven seeds, RF/MLP/logistic regression, and 200 link-block bootstrap draws, mean AUROC (RF/MLP/logistic) was 0.580/0.485/0.459 on box, 0.479/0.484/0.459 on canyon, and 0.490/0.470/0.479 on Etoile. Paired MLP-minus-RF seed deltas and 95% seed-bootstrap intervals were -0.096 [-0.200, 0.004] (exact sign-flip p=0.172), +0.0046 [-0.0339, 0.0400] (p=0.828), and -0.020 [-0.071, 0.031] (p=0.469), respectively. Performance is near chance and there is no consistent MLP advantage. Three held-out scene families are still too few for a reliable scene-level interval; seed inference is not scene inference. This is a simulator-bound smoke with 120 links, one NR grid/SNR regime, generic pilots rather than DMRS, simple built-in materials, and no convergence or visual path audit. CPU LLVM reruns reproduce labels and groups exactly and channels within small floating-point tolerance; raw NPZ file hashes may therefore differ slightly across reruns. Evidence grade remains `SIMULATION_ONLY_UNVALIDATED`.

**Edition 3.6 TDL-A PUSCH fading curve (not phase completion; simulation only):** `scripts/run_tdl_pusch_bler_curve.py` evaluates 1,024 transport blocks at each of 0, 6, 12, 18, and 24 dB with Sionna v2.2.0 TDL-A, 100 ns delay spread, 3.5 GHz carrier, 3 m/s, 2-PRB/30-kHz PUSCH, and the Sionna DMRS-based `PUSCHReceiver`. BLER was 0.9990 [95% Wilson 0.9945, 0.9998], 0.8213 [0.7966, 0.8435], 0.3730 [0.3440, 0.4031], 0.1025 [0.0854, 0.1226], and 0.0264 [0.0182, 0.0381]. BER was 0.3457, 0.2039, 0.0746, 0.0206, and 0.00449. The five points use common seeds; all 40 per-batch checks confirm identical transmitted bits and TDL channel truth across SNRs. Raw JSON is ignored under `outputs/tdl-pusch-bler-tdla-1024-20261021-26851a4.json`, SHA256 `6d473754cc2b17861a198cfdd844c1b16509716b68851645414ed766f9d3cd16`; it records code revision `26851a4cc9ae06eb5af0ea0fdd3946853ec935da` and source hash `55b16c945918b99f7f8de20f4254e04632b128646b2da7dfb4310ba6715c3fd2`. This is a configuration-conditional frequency-domain `OFDMChannel` curve with perfect synchronization and no sample-domain ICI/ISI, not a time-domain link result, standards certification, independent-scene inference, or external measurement.

**Edition 3.7 sample-domain TDL-A PUSCH curve (not phase completion; simulation only):** `scripts/run_tdl_pusch_time_bler_curve.py` runs 1,024 transport blocks per point through Sionna v2.2.0 `TimeChannel` and a waveform-input `PUSCHReceiver`, with TDL-A, 100 ns delay spread, 3.5 GHz, 3 m/s, 12-PRB/30-kHz PUSCH, and SNR values 0/6/12/18/24 dB. The sampled channel uses lags -6 through 7 (13 samples of memory); the shortest configured CP is 13 samples. The transmit waveform mean powers across the 16 batches ranged from 0.9967 to 1.0028, and the AWGN variance was set to 10^(-SNR/10). BLER was 1.0000 [95% Wilson 0.9963, 1.0000], 1.0000 [0.9963, 1.0000], 0.02246 [0.01501, 0.03348], 0.000977 [0.000172, 0.005511], and 0.000977 [0.000172, 0.005511]. BER was 0.29681, 0.15297, 0.001286, 0.0001669, and 0.0001724. The 18 and 24 dB points each contain one failed codeword; this finite-sample plateau does not establish a high-SNR error floor. Identical bits, channel realizations, and standardized noise samples are paired across SNR. Raw JSON is ignored at `outputs/tdl-pusch-time-bler-tdla-1024-20261021-rerun.json`, SHA256 `489d6892e3de7c877a52e2b11ef339ac08d341dc0615b360a986e0c436ed8aa6`; package source hash `c3f90afb34837d6c7cf1e16cf46ef5d8b5490c1b53964e8236fa4c157db3471e`; code revision `4ddb288ddf4ee430bb5386bf91163218419ed2d0`. This single CP-bounded simulation configuration uses ideal synchronization and is not an external measurement, an independent-scene result, a standards-conformance result, or a claim about performance beyond this model/configuration.


**Edition 3.8 paired TDL-A mobility stress (not phase completion; simulation only):** The `scripts/run_tdl_pusch_time_bler_curve.py` runner generated matched 512-codeword curves at 3 and 30 m/s with all other configured settings fixed: TDL-A, 100 ns, 3.5 GHz, 12-PRB/30-kHz PUSCH, seed 42000, 0/6/12/18/24 dB, sampled lags -6 through 7, and 13-sample minimum CP. The 30 m/s condition corresponds to 108 km/h and approximately 350 Hz maximum Doppler. The artifacts contain per-codeword CRC masks, identical transmitted-bit hashes and batch/noise seeds across speed, and different sampled channel realizations. At 3 m/s, BLER was 1.000, 1.000, 0.0234, 0, and 0.00195; at 30 m/s it was 1.000, 1.000, 0.578, 0.430, and 0.482. Paired treatment-minus-control BLER changes at 12/18/24 dB were +0.5547 [95% paired bootstrap 0.5117, 0.5977], +0.4297 [0.3867, 0.4727], and +0.4805 [0.4375, 0.5234]. Holm-adjusted exact McNemar p-values were 3.22e-85, 3.56e-66, and 7.07e-74. These comparisons are conditional on one seeded standardized simulation, one DMRS/receiver configuration and codeword-level pairing; they are not independent-scene inference, external validation, or a standards result. The receiver has ideal timing/CFO and this configuration retains channel memory equal to the shortest CP. Raw curves: `docs/research/results/tdl-pusch-time-bler-tdla-12prb-3mps-512.json` SHA256 `c358af4338eeb5c667d26f0658c3729ca1c339b9ee1322da6fa543f8fbfcb540` and `docs/research/results/tdl-pusch-time-bler-tdla-12prb-30mps-512.json` SHA256 `ff0fdcb5c5c89dc82e2cdbe67cd70cee10fccdac99630b9ad88cce5d4fa9481e`; paired analysis `docs/research/results/tdl-pusch-time-bler-tdla-mobility-paired-512.json` SHA256 `e78f6a00e44a2e16609de706e58d115d0f9c456baecdac8fd03aefb179eddb61`. Reproduce each link curve with the script README recipe and compare them with `scripts/compare_tdl_pusch_time_curves.py`.

### Phase P6 — fair baselines and result pipeline

- Restore old MLP/VGG/ResNet only as a historical reproduction on historical inputs. Train strong modern `Re/Im MLP`, feature-MLP, RF/XGB, classical estimation and a matched CVNN/real CNN on **new identical** observation splits.
- Use same data, augmentation, pilot mask, training updates and hyperparameter-search allocation across neural baselines. Plot capacity–performance and compute–performance Pareto, not one matched parameter count only.
- Implement unit-tested evaluation and probability calibration; save per-sample predictions to immutable artifacts.
- Train on source validation only, choosing a benchmark baseline by a registered rule before target test unlock.

**Done when:** E03–E08 reproducible tables exist, MLP module contribution is genuinely measured, and no architecture is advantaged by oracle input. **Rollback:** invalid comparison discarded and rerun from checked source, with failure recorded.

### Phase P7 — proposed method, **one novelty mechanism at a time**

- Implement an observation-aware complex/real frontend as an interchangeable adapter; start with MLP if it dominates.
- Implement differentiable receiver-to-observation forward renderer with path/cluster plus diffuse residual; verify analytical gradient and limiting cases.
- Implement uncertainty and equivalence-aware posterior with precise mathematical assumptions and a **dedicated two-path identifiability demonstration**. Investigate M2 first (Section 6.3).
- Add symmetry constraints only for tasks where the group action is physically correct; test explicit transformed predictions.
- Run E10–E14 *before* adding counterfactual/forecasting. If gains disappear under matched compute and external scenes, reduce complexity.

**Done when:** the new contribution has a falsifiable mathematical statement, a code-level off switch, a same-backbone ablation and a locked-target comparison plan. **Rollback:** preserve simpler baseline as primary model; publish a negative result when appropriate.

### Phase P8 — dynamic physics and forecasting (conditional)

- Create contiguous trajectories with timestamped BS/UE/blocker motions; move 3D geometry and **retrace** at a cadence validated for channel/occlusion changes.
- Verify Doppler short-window shortcut against exact retracing; use shortcuts only when error stays below a preregistered tolerance for the studied regime.
- Add discrete future horizons in **milliseconds** and receiver frames; validate interpolation/resampling and leakage between adjacent windows.
- Train simple persistence/AR/Kalman/TCN baselines first; only then train a structured latent dynamics head.
- For counterfactuals change actual blocker geometry/material/orientation and recompute rays. Control the interventions and inspect whether other reflected paths change.
- Export event uncertainty, path births/deaths and beam quality under a fixed observable-information contract.

**Done when:** E17–E19 pass geometry/time QA and forecasts beat an appropriate persistence/physics baseline at practically relevant horizons. **Rollback:** remove temporal claims, keep static estimation paper.

### Phase P9 — external independent measurement (high-value, not automatically available)

- Identify a legitimate measurement dataset or collect with approved SDR/5G equipment; record licensing, consent/privacy constraints, exact TX/RX radio configuration, clocks, antenna calibration and spatial survey.
- Define reliable geometric LOS labels or use unknown; if only measured strength is available, do **not** assert authoritative physical LOS.
- Make signal preprocessing reproducible: IQ capture → synchronization/CFO correction → RS extraction → LS/LMMSE CSI → model input; preserve instrument logs and uncertainty.
- Hold out complete locations and capture sessions; tune/calibrate only on allowed partitions and separately report *zero-shot*, *unlabeled adaptation* and *few-shot* performance.

**Done when:** independently reviewable measurement provenance and defensible labels exist; otherwise say exactly **simulation-only**, without implying verified real-world transfer.

### Phase P10 — downstream link/beam decisions and runtime feasibility (conditional)

- Convert predicted distributions to beam selection/abstention or channel estimator uncertainty weighting with a deterministic decision policy fixed from permitted validation data.
- Evaluate beam regret, BLER, SE, outage, failure rate, control/signaling cost and inference latency against same-information baselines; report false alarms and stale-prediction penalty.
- Quantization/deployment is an independent engineering study; never assume complex operators or Python ray-tracing code can run on real-time RAN hardware.

**Done when:** downstream system benefits exist under a measured cost envelope. **Rollback:** report state estimation only.

### Phase P11 — preregister, lock, reproduce and prepare the paper

- Publish or privately timestamp `paper/preregistration.md` *before* final target evaluation; lock endpoint definitions, test split IDs, baselines, seeds, allowed tuning and claimed novelty.
- Execute confirmatory runs once under written procedure; preserve failures/outliers in the result ledger. All post-lock changes require a version increment and disclosure.
- Generate tables/figures from machine-readable predictions; trace each figure through dataset hashes and code commit, avoid manually transcribed results.
- Execute independent cold-run reproduction on a second environment/host when feasible and run full audit checklist in Section 18.

**Done when:** every central paper claim points to a raw artifact and a replicating command, with limitations stated in an ablation and failure analysis.

---

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
# Proposed after implementation:
loslab doctor --native-sionna --report outputs/doctor.json
loslab validate physics --config config/physics/gold.yaml --strict
loslab validate receiver --config config/receivers/nr_pusch_dmrs.yaml --strict
loslab data build --config config/datasets/rt_static.yaml --manifest outputs/data_rt_manifest.json
loslab data splits --config config/scenes/split_registry.yaml --freeze
loslab experiment run --config config/experiments/E08_scene_ood_static.yaml
loslab experiment compare --experiment E08 --paired --cluster scene
loslab paper build --registry paper/results_index.md --verify-manifests
```

Every command should return a nonzero exit code on physics/schema/provenance failure, log the exact resolved configuration, and never overwrite an artifact unless a versioned explicit overwrite was requested. Deterministic filenames should contain a hash or unique run ID, **not only a mutable timestamp**.

### 12.2 Train/evaluate interface boundaries

```python
# Pseudocode / type design, not existing runnable code.
@dataclass(frozen=True)
class ReceiverObservation:
    pilot_y: ComplexTensor             # measured LS input or received pilot symbols
    pilot_x: ComplexTensor             # known reference pilot symbols
    pilot_mask: BoolTensor
    observed_grid: ComplexTensor       # only observed or estimated CSI
    known_grid_config: NRGridConfig
    known_array_config: ArrayConfig
    available_covariance: FloatTensor | None
    receiver_calibration: CalibrationState | None

@dataclass(frozen=True)
class PrivateOracle:
    physical_paths: PathSet
    geometric_los: BoolTensor | None
    geometry: GeometryTruth | None
    full_ideal_channel: ComplexTensor | None

# Trainer receives two separately scoped objects.
# Model.forward(...) must NOT accept PrivateOracle.
outputs = model.forward(observation)  # never model.forward(full_dataset_row)
loss = objective(outputs, oracle_targets, target_masks, heldout_observed_pilots)
```

In real Python, implement strict nested models or dataclasses with runtime validators and static types; the pseudocode types above are explanatory. Confirm that `heldout_observed_pilots` were not included in the input estimator if using them as an evaluation target.

---

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

## Appendix D. Results artifact contract and exhaustive paper exhibit inventory

**Result registry design:** every artifact is generated from prediction tables plus immutable metadata; hand-editing the numeric contents of a figure/table invalidates the artifact. A publication may use a subset of these, but the internal benchmark should retain them all, explicitly marking `NOT_RUN` or `NOT_APPLICABLE` rather than filling in optimistic numbers.

### 14.1 Primary paper tables

| Table ID | Content | Inputs / required columns |
|---|---|---|
| TAB01 | dataset and scene statistics | sites, distinct scenes/asset families, trajectories, LOS prevalence, mixed paths, weak direct, outage, bandwidth, carrier, label unknowns |
| TAB02 | API/physics validation | 2.2 package versions, solver options, delay/power tests, convergence, native pass/fail and precision |
| TAB03 | observation realism | NR DMRS config, pilot ratio, synchronization availability, SNR definition, LS/LMMSE CSI NMSE, EVM/BLER |
| TAB04 | source in-domain performance | model, parameters, FLOPs, train compute, LOS/AUPRC, delay MAE, CSI NMSE, posterior calibration |
| TAB05 | locked unseen-scene transfer (**primary**) | independent scene group count, primary error and paired CI, OOD degradation, calibration and compute |
| TAB06 | stochastic→RT transfer | source-only vs RT-adapted vs mixed training; identical target scene IDs; explicit target-label access |
| TAB07 | identifiable/ambiguous conditions | delay separation, BW, SNR, array aperture and estimator variance/coverage comparisons |
| TAB08 | component ablation | same backbone with/without complex encoder, MLP, forward decoder, ambiguity, symmetry, posterior, counterfactual |
| TAB09 | full multipath task suite | targets T01–Txx with validity count, errors, confidence, target reference and task enable status |
| TAB10 | physics sensitivity | scattering, diffraction, material, geometry, depth, samples/seed, antenna errors and effect on results |
| TAB11 | dynamic forecasts (if valid) | horizons, coherence, channel NMSE, future delay-spread error, blockage lead time, uncertainty |
| TAB12 | downstream system (if valid) | BLER/SE/beam regret/latency against same-information and oracle comparators |
| TAB13 | external real measurement (if valid) | device/session/site, known labels, zero-shot/adaptation protocol, uncertainty, negative transfer |
| TAB14 | computational/environment cost | CPU/GPU names, RT gen throughput, training time, inference ms, memory, preprocessing cost |
| TAB15 | limitations and failure classes | physical ambiguity, solver error, label unknown, receiver outage, profile shortcut, drift |

**Each table cell metadata:** mean/median as stated, CI, independent group count, frame count, training seeds, unit, validity mask, config hash and source run IDs. No table of a million frames pretending to have a million independent propagation sites.

### 14.2 Main and supplementary figure inventory

| Figure ID | Exhibit and scientific function |
|---|---|
| FIG01 | entire physical generation → receiver observation → proposed inference → tasks diagram with oracle boundary |
| FIG02 | representative 3D RT LOS-mixed, NLOS-reflection, NLOS-transmission, no-coverage scenes, plus rendered physical ray overlays |
| FIG03 | clean versus noisy CSI, pilot-mask, CFR magnitude/phase, CIR/PDP; annotate delay/phase origins |
| FIG04 | resolvability two-path experiment; error and posterior width vs \(B\Delta\tau\) and SNR |
| FIG05 | train-domain vs unseen-scene main comparison and **paired scene-level uncertainty**, MLP shown prominently |
| FIG06 | multipath state representation plus measured-pilot forward decoder and identifiable latent equivalence |
| FIG07 | ablation bars for proposed mechanism and parameter/compute-matched controls |
| FIG08 | LOS ROC/PR and calibration; prevalence-separated and weak-direct slices |
| FIG09 | RMS delay spread prediction scatter/residual/error CDF, with oracle vs receiver-resolved labels differentiated |
| FIG10 | first-path delay error by bandwidth/SNR/path separation; explicitly mask absolute timing when unobservable |
| FIG11 | PDP reconstruction / path-set qualitative overlay and *power-consistency* residual plot |
| FIG12 | AoA/AoD circular error and polar pattern/resolution slices, where arrays permit valid estimation |
| FIG13 | Doppler/delay–Doppler spectrum target and predicted, aliased/nonaliased controls |
| FIG14 | interval empirical coverage/width, risk–coverage, error versus uncertainty, and scene-level calibration |
| FIG15 | RT mechanisms/material/sampling/convergence robustness matrix, with uncertain solver regimes flagged |
| FIG16 | phase/timing invariance/error under controlled nuisance transformations; exact theoretical test |
| FIG17 | dynamic scene trajectory, LOS transition, actual retraced paths and Doppler shortcut divergence |
| FIG18 | forecast horizon curves for CSI/blockage/beam; baseline persistence and uncertainty shown |
| FIG19 | counterfactual paired intervention example: geometric change, altered path set, CSI change, model prediction |
| FIG20 | downstream beam/BLER/SE gains vs model inference latency and decision confidence, if measured |
| FIG21 | external measurement transfer and error domain-gap histograms, if independent recordings exist |
| FIG22 | compute/latency/memory Pareto frontier (models and receiver + preprocessing) |
| FIG23 | train/test site/trajectory maps, exclusion zones and similarity/duplicate audit (privacy-safe) |
| FIG24 | worst-case and representative error archetypes with physical explanations and uncertainty |

### 14.3 Results-index schema (never manually populate predictions)

```yaml
# paper/results_index.yaml — proposed output manifest only
- exhibit_id: TAB05
  title: "Held-out-scene multipath estimation"
  status: NOT_RUN
  experiment_ids: [E08, E11, E12, E14]
  primary: true
  required_predictions: ["scene_id", "sample_id", "y_true", "y_hat", "uncertainty"]
  result_file: null
  generation_script: "evaluation/tables.py"
  source_run_hashes: []
  independent_unit: "scene/site"
  units: "declared_per_endpoint"
  manual_numeric_edits_permitted: false
```

All figures/tables must have provenance plus generated caption notes for normalized/un-normalized labels, unknowns, group CIs and locked access. A graph with impressive curves but no traceable source model/data/config should not enter the draft manuscript.

---
## Appendix E. Last-mile red-team review and immutable release decision

### E.1 Sionna 2.2-specific implementation hazards to challenge manually

As verified against the **current official 2.2 pages on 2026-10-21**:

1. `PathSolver(deterministic=True)` is distinct from passing `seed=...` to its call; documentation says a fixed seed alone does **not** guarantee deterministic results.
2. `PathSolver.__call__` includes `max_depth`, `max_num_paths_per_src`, `samples_per_src`, `synthetic_array`, `los`, `specular_reflection`, `diffuse_reflection`, `refraction`, `diffraction`, `edge_diffraction`, `diffraction_lit_region` and `seed`. Defaults include `diffuse_reflection=False`, `diffraction=False`, `refraction=True`, `synthetic_array=True`; **store all explicitly**, do not let defaults drift.
3. `max_depth=0` is LOS-only candidate generation under the documented solver interface, but **label validity** still requires direct-line geometry/endpoint/wall-boundary testing. A missing sampled ray is not always proof of physical obstruction.
4. `Paths.cir()`, `Paths.cfr()` and `Paths.taps()` all default `normalize_delays=True`. Explicitly pass `False` in physical-delay oracle generation. `cfr()` and `taps()` support amplitude/energy normalization flags: leave these `False` for link budget studies.
5. `Paths.cfr(frequencies=...)` calculates from baseband path coefficients and frequency offsets. Never double-apply the carrier phase factor or silently use frequencies with the wrong absolute/baseband convention.
6. The official Paths API returns **real and imaginary components** in specified output forms; validate tuple packing and every tensor axis at runtime. Some `out_type` transformations can surprise code expecting a single native `complex64` tensor.
7. Synthetic arrays reduce geometric work by phase-shifting a center-antenna trace. They are a potentially poor substitute for per-element tracing when aperture, near field, blockers or large geometry offsets matter; verify full vs synthetic arrays before spatial super-resolution claims.
8. Refraction/transmission uses a thin-material/slab approximation with **no angular deviation in traversal** per the solver documentation. Do not call transmission through thick 3D walls full Maxwell/FDTD fidelity.
9. The documented solver currently supports first-order diffraction interactions; unsupported multi-edge/wedge effects are a **fidelity limitation**, not evidence of channel absence.
10. `Paths` short-time Doppler evolution **does not create or remove geometric ray trajectories as real blockers move**. For blockage onset, retrace after time-varying geometry and compare against the shortcut explicitly.
11. `PathSolver.loop_mode` may be `symbolic` or `evaluated`; the documentation says **symbolic is fast but does not support automatic differentiation**. If training proposes differentiating through **RT geometry**, inspect the exact AD path and benchmark evaluated mode. The proposed differentiable *measurement decoder* is a separate model component; do not conflate it with differentiating through the RT solver.
12. Sionna 2.2 install instructions require Python 3.11+ and PyTorch 2.9+ for **PHY/SYS**, while **RT** depends on Mitsuba/Dr.Jit; CPU RT needs the LLVM backend. Confirm the actual set of installed packages and compiled supported device backend, including `sionna-rt` if split-installed.
13. Sionna PHY's NR tutorial covers a **subset** of NR functions. Use documented PUSCH DMRS for the initial receiver; neither `PUSCHLSChannelEstimator` nor generic OFDM pilots imply a verified NR positioning PRS receiver.
14. `spec_version="19.2"` selects Sionna's available TDL/CDL profile resources; the contemporaneous published **TR 38.901 v19.5.0** is a *separately identified standard version*. Preserve both in manuscript/config/manifest.
15. `RadioMapSolver` gain/coverage maps may use different coherent/noncoherent aggregation from instantaneous CFR. Never turn path-gain map samples directly into a complex CSI/PDP truth source.
16. Random diffuse scattering adds stochastic solver variance and is material-model dependent. Numerical determinism cannot turn an inaccurate choice of scattering coefficients into scientific fidelity.
17. Dr.Jit/complex backprop kernels have precision/gradient support and potentially GPU backend assumptions. Check gradients with finite differences in analytic scenes, not only nonzero `.grad`.
18. The RT first-path oracle, baseband channel estimator and synchronized absolute ToA are three distinct interfaces. **A network cannot learn physically nonexistent clock information**, even if training labels contain geometric distance.
19. All public dataset images and meshes require appropriate attribution/license; screenshots of proprietary scenes or device-location details may not be publishable.
20. Research findings must survive Sionna patch/minor-version drift or explicitly be described as **version-bound results**.

### E.2 Black-box reviewer checks: can this paper be falsified?

Before submission, simulate reviewers who try to disprove the claimed advantage:

- **Reviewer 1 (communications):** Does a conventional LS/LMMSE + calibrated classical estimator solve the task already? Are propagation regimes, array assumptions, SNR, link budgets and path delays physically correct? Is the simulator using a simplified wall/scattering material model?
- **Reviewer 2 (machine learning):** Do MLP gains persist after matching inputs, augmentation, parameter/FLOP envelopes and tuning budget? Is the new mechanism actually new or a relabeled path-token network, standard equivariance, set matching, ordinary uncertainty weighting or CORAL?
- **Reviewer 3 (statistics):** Were primary endpoints and scene test splits locked? How many independent geometries/sites exist? Are CIs hierarchical and paired? Are hyperparameter choices independent of locked OOD results?
- **Reviewer 4 (inverse problems):** Which parameters are identifiable from exact pilot/time/array geometry? How is unknown oscillator/timing offset handled? Are uncertainty intervals actually calibrated in unresolved multipath cases? Can path ambiguity be proven or demonstrated analytically?
- **Reviewer 5 (reproducibility):** Can one rerun the path oracle, noisy receiver, model and figures from immutable hashes, valid licenses and documented hardware? Are failure/no-ray cases retained? Does native Sionna (not a mock) work?
- **Reviewer 6 (practicality):** Does better path parameter estimation yield useful beam/receiver/localization decisions, and is the improvement worth inference and calibration cost?

### E.3 Acceptance gates (actual outcomes to fill, not invented promises)

| Gate | Evidence that allows the next stage | If gate fails |
|---|---|---|
| G0 repo integrity | baseline tree hash, original scores accurately attributed, 69 local lightweight tests replicated under controlled environment | repair provenance/tests, no paper claims |
| G1 native physics | documented 2.2 RT/PHY calls work, gold physical cases and masks/units pass, LOS-mixed present | stop dataset generation, fix adapter/material/geometry |
| G2 receiver validity | exact NR PUSCH-DMRS config and estimator checked, known signal tests/BER/BLER and timing semantics pass | limit all claims to debug/noisy-CSI simulator |
| G3 data validity | scene-disjoint immutable splits, target definitions audited, no profile/geometry or temporal leakage | rebuild dataset and invalidate contaminated model runs |
| G4 strong baselines | same-information/compute/validation comparisons, historical MLP contribution question actually answered | rebuild comparisons; do not justify complex branch by aesthetic preference |
| G5 credible novelty | named, mathematically specified new mechanism and same-model off-ablation; physical observability respected | pivot to robust benchmark or specialized wireless contribution |
| G6 external evidence | prespecified held-out scene transfer; independent group CIs; calibration width/error; physics sensitivity | report limitation, investigate simulator bias and negative transfer |
| G7 added forecasting claim | re-traced trajectories and event results; persistence/AR benchmark passed | drop forecasting/causal claims, retain static paper |
| G8 system/measurement claim | actual measurable BLER/beam/real RF evidence with compatible side information and budgets | omit deployment/measurement claims |
| G9 release | cold rerun plus raw predictions/manifests; all tables auto-generated, data license and deviations disclosed | do not submit a scientifically unsupported version |

### E.4 Priority order when time or compute is constrained

**Mandatory paper core:** G0–G6, static observed multipath state, LOS/direct detectability, RMS delay spread, relative resolvable first-path, held-out-pilot reconstruction, path ambiguity uncertainty, MLP/analytical baselines, independent RT scenes, negative-result reporting.

**High-value additions:** AoA/angular spread with real arrays; Doppler with validated contiguous trajectories; real measured CSI; a decision-theoretic receiver use case. Add only after core passes.

**Deferred unless proven feasible:** large-scale wireless “foundation model,” exhaustive all-target training, absolute ToA without clock reference, semantic material inference from sparse CSI, scene-counterfactual causal claims without actual intervention, guaranteed universal OOD uncertainty, and production RAN deployment.

### E.5 Publication decision tree

- **New mechanism + mathematically justified ambiguity control + consistent held-out scene benefit + calibrated uncertainty:** pursue a strong IEEE wireless journal submission, including a robust method/physics section.
- **Same plus broad, non-wireless complex inverse-problem generality:** consider a competitive top-tier ML venue with additional datasets and theory; this is **not established** by 5G experiments alone.
- **Only a robust new benchmark/physics QA and honest MLP vs CNN negative finding:** publish as a reproducibility/benchmark paper or specialized conference paper; do not force algorithm novelty.
- **Only in-domain LOS accuracy and no receiver/physical validation:** do not present as an S-tier research contribution.
- **No reliable transfer or calibrated uncertainty:** publish limitations and method failures if scientifically informative; redirect algorithm design using those failures as diagnostic evidence.

### E.6 Definitive documentation snapshot and limitations

Primary official docs checked on 2026-10-21:

- NVIDIA Sionna 2.2 **Installation:** <https://nvlabs.github.io/sionna/v2.2.0/installation.html>.
- NVIDIA Sionna RT **PathSolver:** <https://nvlabs.github.io/sionna/rt/api/paths_solvers.html>.
- NVIDIA Sionna RT **Paths/CIR/CFR/Taps:** <https://nvlabs.github.io/sionna/rt/api/paths.html>.
- NVIDIA Sionna RT **Mobility:** <https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html>.
- NVIDIA Sionna RT **Scenes / SceneObject:** <https://nvlabs.github.io/sionna/rt/api/scene.html>, <https://nvlabs.github.io/sionna/rt/api/scene_object.html>.
- NVIDIA Sionna PHY **PUSCH and DMRS:** <https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/pusch.html>, <https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html>.
- ETSI publication **TR 38.901 v19.5.0, published 2026-09-25:** <https://portal.etsi.org/webapp/WorkProgram/Report_WorkItem.asp?WKI_ID=79559>.

**Verification status at this 1.7 update (2026-10-21):** native PHY kernels were exercised on CUDA with a known flat-channel PUSCH DMRS vector, controlled AWGN and a Rayleigh transmitter/channel/estimator loop against Sionna truth. Native RT direct and two-ray frequency responses were checked on LLVM CPU. Receiver-only NR records now separate observations from targets/provenance and validate axis/bin consistency. No new measurements, trained research models, held-out scene metrics or manuscript results were produced. RT CUDA remains blocked by the WSL OptiX loader; all scientific claims remain pending experimental verification.

---

**Execution status note — 2026-10-21:** This document is an active research execution plan, not a completed thesis or scientific study. Edition 3.9 adds a 512-codeword-per-SNR paired sample-domain PUSCH comparison of additional DMRS positions 0 and 3 at 30 m/s, alongside edition 3.8's paired 3-versus-30-m/s TDL-A mobility stress comparison and a 1,024-codeword-per-SNR sample-domain curve with time-varying fading and channel memory equal to the shortest CP. In the DMRS comparison, channel truth, batch seeds and noise seeds match, while payload size changes and decoded-bit goodput is reported; all results remain one ideal-synchronization TDL-A simulation, not scene-level evidence or a receiver recommendation. The RT baseline results remain near chance, inconclusive, and do not pass the independent-scene gate. Remaining priorities include broader analytical sweeps; TDL/CDL distributional validation; blind/fractional timing synchronization; sample-domain links with broader CP margins, mobility, and receiver settings; qualified diverse RT scenes with convergence and path-overlay audits; matched baselines across enough independent scene families; mechanism-aware target semantics; external measured-data feasibility; and implementation/ablation of the proposed identifiability-aware method. No scientific-performance or measurement claim is established.

**DMRS comparison evidence — 2026-10-21:** Raw control (`dmrs_additional_position=0`), treatment (`=3`) and paired comparison are archived in `docs/research/results/tdl-pusch-time-bler-tdla-12prb-30mps-dmrs{0,3}-512.json` and `tdl-pusch-time-bler-tdla-dmrs-paired-512.json`. At 12/18/24 dB, the respective BLERs were 0.635/0.475/0.533 (position 0) and 0.018/0/0 (position 3); mean decoded transport-block bits per codeword were 1472.6/2118.4/1882.1 and 3686.0/3752/3752. These are conditional paired codeword results from one 12-PRB, 30-kHz, TDL-A, 100-ns, 3.5-GHz simulation (512 codewords per SNR, 30 m/s), with ideal synchronization. The 3-position treatment lowers payload from 4032 to 3752 bits/codeword; no independent-scene uncertainty, antenna/numerology sweep, or standards-general conclusion follows. Results are descriptive simulation evidence only.

**Edition 3.10 fixed-payload CP-boundary diagnostic — 2026-10-21:** `scripts/run_pusch_cp_margin_experiment.py` uses one 12-PRB/30-kHz PUSCH allocation (4,032 bits/codeword), identical transmitted waveforms and standardized AWGN across two deterministic two-path conditions, and a 30-dB sample-domain SNR. The shortest CP is 13 samples; the echo delay is 11 samples inside CP (2 samples of margin) or 16 samples beyond CP (3-sample excess). Across 512 codewords, BLER was 10/512 = 0.0195 [95% Wilson 0.0106, 0.0356] inside CP and 512/512 = 1.0 [0.9926, 1.0] beyond CP. The paired BLER increase was 0.9805 [95% paired codeword bootstrap 0.9668, 0.9922], exact two-sided McNemar p=1.53e-151. Raw codeword masks, matched bits/noise hashes, and runtime provenance are in `docs/research/results/pusch-static-cp-margin-12prb-30khz-512.json`, SHA256 `8793af2c6259ef631e77444f15930a2f7c7b072761b6bd804bbb530b95dc68be`. This is a deliberately deterministic discrete two-path ISI diagnostic with zero prehistory at the finite burst boundary. It establishes only that this receiver/configuration is sensitive to this chosen CP violation; it is not a CP-margin curve, a TDL/CDL result, or evidence for general NR receiver performance.

**Edition 3.11 matched multi-delay CP sweep - 2026-10-21:** The committed runner evaluates echo delays 1, 4, 8, 11, 13, 16, and 20 samples against the same transmitted PUSCH codewords and standardized noise, with a 13-sample shortest CP. The 8-sample in-CP reference has five samples of remaining CP and 0/512 CRC failures (BLER 0; Wilson 95% upper bound 0.00745). BLER was 0/512 at delays 1 and 4, 10/512 (0.01953; Wilson 95% CI [0.01064, 0.03558]) at 11, 88/512 (0.17188; [0.14167, 0.20697]) at 13, and 512/512 (1.0; [0.99255, 1]) at delays 16 and 20. Paired BLER increases versus delay 8 were 0.01953 [paired codeword bootstrap 95% CI 0.0078125, 0.0332031] at delay 11, 0.171875 [0.140625, 0.205078] at delay 13, and 1.0 [1, 1] at delays 16 and 20. Exact two-sided McNemar p-values were Holm-adjusted over all six non-reference delays: 0.005859 at 11 samples, 2.585e-26 at 13, 8.950e-154 at 16 and 20; both 1- and 4-sample comparisons had adjusted p=1. Raw data and provenance are in docs/research/results/pusch-static-cp-margin-sweep-12prb-30khz-512.json, SHA256 6c8385eb2f0a3c0dd1fee6e950c7a9a93114514bb322944f297f21e00be3e1eb, generated with code commit c8ac8897aa655d8dd1426b479b1d98b6d7d0ec71, seed 82417, 512 codewords, and matched bits/noise hashes. This is one deterministic finite-burst two-path simulation with zero channel prehistory, one PUSCH configuration, and one SNR; paired inference is conditional on codewords, not independent scenes. It diagnoses this receiver's sensitivity to this selected echo-delay sweep and does not establish general NR performance or channel-condition classification results.
