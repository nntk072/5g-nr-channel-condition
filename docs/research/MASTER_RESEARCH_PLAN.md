# MASTER RESEARCH & ENGINEERING PLAN
## Physically Grounded, Identifiability-Aware Multipath State Learning for 5G NR
### From LOS/NLOS classification to channel-state reconstruction, uncertainty, and forecasting

**Document edition:** 3.84 — 2026-10-22 (add raw complex-observation MLP control)
**Status:** execution specification and scientific decision record; **not** a completed study, not experimental evidence, and not a claim of top-tier acceptance.
**Repository inspected:** live checkout /home/nguyenl37/5g-nr-channel-condition, branch main, initial implementation commit ffe199c80c1caf40dea91a406fe02124b3ff05ec (built from baseline cc43a5544036578a6bb57ce52b66436d8acb6c1b); local reference /home/nguyenl37/sionna at Sionna v2.2.0, commit 2f0a3820269b53fd022780a6f2f5f720f7ceb9b4, RT submodule commit 15b5ee036917a4c2a9b6420e05570b77717e2fcc.
**Legacy work:** Long Nguyen, *5G Physical Layer Channel Conditions Classification Using Machine Learning Techniques*, Tampere University, July 2025, embedded in `docs/thesis/NguyenLong.pdf`.
**Scope:** living research and engineering execution specification. Recent revisions add a normalized local Fisher-information diagnostic for joint complex path-gain and excess-delay parameters, bounded CP-based integer timing acquisition/correction for a known-origin capture, and reproducible frequency-domain and sample-domain TDL-A PUSCH BLER curves, plus a paired 3-versus-30-m/s receiver stress comparison. The new waveform link includes time-varying fading, sampled channel memory, and the resulting ICI/ISI within a CP-bounded configuration. Prior revisions add a reproducible three-scene leave-one-scene-out RT pilot and explicit class-balanced scene-group audit to residual CFO acquisition/correction and a two-tap sample-domain PUSCH decoder smoke test. Earlier revisions added deterministic CPU/LLVM RT generation, controlled RT material sensitivity, GPU PUSCH receiver, CPU/LLVM RT-to-NR bridge, content-addressed records, leakage-audited splits, analytical fixtures, and TDL-to-PUSCH generation. These remain bounded simulation and implementation checks; they establish neither model advantage nor standards-complete synchronization or realistic NR BLER curves.
**Historical verification record (edition 3.15):** the full pinned Sionna/CUDA test suite passes at 86.80% coverage. Ruff lint and format checks pass across the maintained source, test, and script directories (108 Python files); mypy passes on 62 source files. With CUDA hidden, the complete CPU suite passes at 76.91% coverage against the 75% gate. A repository-root Ruff scan also reaches archived 2025 code and legacy notebooks, which are outside the maintained lint scope. `uv lock --check` and `uv build` pass. GitHub Actions run 39583322560 passed all jobs for code commit af2d337; run 39583778271 passed for the first artifact commit; run 39584231041 passed after the Holm test additions; run 39584383462 passed for the matching CP sensitivity update; run 39585208208 passed for the analytical sweep implementation; run 39586268134 passed for the RT convergence runner; run 39586471715 passed for its evidence update. The GitHub CPU version matrix omits optional PyTorch and therefore runs core tests without applying the coverage threshold; the dedicated CPU-neural job installs PyTorch and enforces pytest's configured coverage threshold. The project-owned PUSCH DMRS LS adapter runs on CUDA, recovers a known flat complex channel over the estimated grid (maximum error approximately 4.3e-8), and lowers estimation NMSE as AWGN variance decreases. A native PUSCH transmitter → RayleighBlockFading → OFDMChannel → DMRS estimator loop recovers Sionna's channel truth without noise and has the expected AWGN error trend. Native direct-path and LOS-plus-specular-reflection RT scenes map to the NR grid on LLVM CPU and match absolute-delay CIR frequency responses, including coherent two-ray summation. A controlled simple-reflector check changes the surface from matched εr=1 to εr=4, observes one versus two valid paths and a changed NR-grid CFR; this validates material sensitivity in the pinned synthetic geometry, not material realism or general solver accuracy. NR records round-trip complex CSI, variance, masks, targets and provenance through pickle-free content-addressed NPZ files; scene-group split assignments are deterministic, audited and immutable once locked. A projected-coordinate audit reports cross-split neighbors inside a meter buffer and requires exact sample-set coverage. CFR fingerprints report cross-split near-duplicates invariant to common complex gain/phase, and trajectory windows report positive-duration cross-split overlap; the fingerprint utility is intentionally pilot-scale and quadratic. Seeded 1–3 path analytical fixtures reproduce the closed-form CFR and explicitly separate receiver inputs from synthetic path oracle. The Sionna v2.2.0 TDL adapter produces deterministic batched CUDA tap-to-CFR outputs for pinned profile-table selector 19.2, matches the explicit frequency-response equation for TDL-A and TDL-D, and restores the shared Sionna RNG state. Native batches of 4,096 realizations for each TDL-A–E profile match Sionna's nominal path delays and mean tap powers (10% Monte Carlo tolerance); this is implementation consistency, not a full standards certification. The native TDL→PUSCH batch path generates transmitter grids, TDL channel truth, AWGN observations, and per-sample NRChannelRecords with truth excluded from model inputs; the native test round-trips a GPU-generated target record through hash-verified pickle-free storage. Edition 3.2's CP correlation estimates per-waveform residual CFO (within its stated unambiguous range) and the inverse rotation restores injected PUSCH OFDM grids; a static two-tap sample-domain PUSCH path decodes at high SNR and exhibits the expected BER/BLER degradation at higher AWGN. This is a receiver smoke test, not a mobile/fading channel model or statistically meaningful BLER curve. Edition 3.5 acquires and corrects a bounded nonnegative integer sample delay on a known-origin PUSCH capture using CP repetition, validated on deterministic batched OFDM and the native Sionna PUSCH waveform; it is not blind NR frame synchronization or fractional timing recovery. Edition 3.3's three-scene leave-one-scene-out pilot, metrics and limitations are recorded below in P5. Edition 3.7 adds a native Sionna `TimeChannel` plus waveform-domain `PUSCHReceiver` path and a 1,024-codeword-per-SNR TDL-A curve with matched bits/channel/noise, sample-domain channel truth, and a checked 13-sample channel memory equal to the minimum CP. The sampled waveform mean power is recorded for SNR provenance. This remains a simulation with ideal synchronization and one CP-bounded link configuration. The Fisher diagnostic uses centered frequencies and delay coordinates scaled by inverse frequency span; its CRLB is returned in physical delay units and withheld for rank-deficient models. Tests cover rank deficiency, close-path conditioning, noise-variance scaling, invalid inputs, and agreement with the closed-form single-path delay CRLB for an unknown complex gain. These checks validate local formulas and numerical behavior, not empirical model performance or delay resolution. Both TDL-A PUSCH curves are recorded in section P5 with raw artifact hashes and confidence intervals. Production-scale diverse RT data, full TDL/CDL distributional conformance, blind/fractional timing synchronization, broader sample-domain link configurations, realistic faded-link decoding curves, and scientific model behavior remain open. Sionna RT CUDA initialization fails in WSL because working OptiX symbols are unavailable. These are software/physics checks, not research results.
**Scientific objective:** build a model that infers what is **observable and physically defensible** about a wireless channel, predicts what changes next, and knows when uncertainty is unavoidable.

**Historical verification snapshot — edition 3.18:** The complete pinned Sionna/CUDA test suite passes at 88.37% coverage; with CUDA hidden, the complete CPU suite passes at 77.27%, above the configured 75% gate. Ruff check and format pass across 116 maintained Python files, mypy passes on 63 source files, and `uv lock --check` plus `uv build` pass. GitHub Actions run 39593450490 passed for code commit `235ccc9a275ca1c38520ce94f3c010fe982e487a`. The earlier pasted CPU run (175 passed, 7 skipped, 73.19%) was from an earlier tree and failed the coverage threshold; the current complete CPU run passes. The native Sionna generator now derives deterministic seeds for each family/profile/delay-spread call and restores shared RNG state. Three native tests verify same-seed output and RNG restoration, CDL A–E proxy labeling, and invalid-input handling. A CUDA generation run produced 40,960 rows across TDL/CDL A–E, 4,096 rows per family/profile and 1,024 per family/profile/delay-spread stratum, at 3.5 GHz, 128 tones, 30 kHz spacing, 20 dB SNR, delay spreads 30/100/300/1000 ns, Sionna profile selector 19.2, seed 78021, on an NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU. Profile-proxy counts are 24,576 for A/B/C and 16,384 for D/E. The 39,040,864-byte NPZ is retained as a local ignored output; its SHA256 is `efd5705592f6bdb9aca25320efda60bab1453c474b2fdde1fa5056f86a081521`. The committed sidecar manifest is `docs/research/results/profile-proxy-tdl-cdl-a-to-e-4096-seed78021.json`, SHA256 `2a2cf4c0eda1dcaabdc4d5e7654e1e33133e642cef70c38eec0ece9b35609aaa`; it records code revision `2e8244e794e1d39be5f05b9c81ac45b0c055fe7c`, package-source SHA256 `3255cfaa6374d71691240a5d654ba00f7032b7df95d0c5a71fa71e511ca647ea`, and runner SHA256 `7e569af9db6b7463eecb592380e3e67f5d2dcdd0eb9c5b8e6ea7befd1cecfeaf`. This is simulated profile-proxy source data only, not geometric LOS/NLOS, measured evidence, or a model-performance result. CDL full conformance, physical-domain transfer, receiver imperfections, and broad independent scene validation remain open.

> **Non-negotiable integrity rule:** A perfect result cannot be guaranteed. This plan targets *maximally defensible and reproducible evidence*. No expected accuracy, error reduction, “S-tier” result, or simulation-to-real success is to be invented, assumed, or reverse-engineered from desired conclusions. The project must support publishing an honest negative result or reducing the paper scope when a hypothesis fails.

---

**Edition 3.19 verification addendum:** Native Sionna/CUDA tests now verify the CDL A–E table delays and Monte Carlo path powers at each of 30/100/300/1,000 ns, with 20 parameterized profile/spread cases. The complete pinned GPU suite passes (234 tests, 88.37% coverage); the CPU suite passes (189 passed, 11 skipped, 77.27%). Ruff check and format pass across 116 maintained Python files, mypy passes on 63 source files, and lock/build checks pass. Three additional 4,096-realization/profile CUDA reports at 30/300/1,000 ns are recorded in the edition 3.19 evidence entry below. These runs validate delay scaling and pinned simulator consistency only; speed/Doppler, array correlation, full standard conformance, geometric/RT validation, measured evidence, and the main method remain open. GitHub Actions run 39594888777 passed all jobs for code commit 470e98613810bf1e166b73b65e0dab55531c0496.

**Edition 3.20 verification addendum:** The pinned LLVM RT gold test at `tests/test_integrations/test_rt_blockage_diffuse_native.py` exercises Sionna's `box_one_screen` at 3.5 GHz with a 1x1 isotropic array, deterministic PathSolver seed 13, 2,000 samples/source, depth 3. A same-side pair has a direct oracle path; an across-screen pair has none but retains indirect paths. With scattering coefficient 0, the across-screen solve has no paths tagged diffuse; with coefficient 0.5 it has diffuse-tagged paths. The observed counts in this pinned run were same-side direct/full = 1/50, blocked direct/full = 0/31, and blocked plus scattering = 0/43 with 12 diffuse-tagged paths. The test asserts these mechanism/visibility relationships rather than exact Monte Carlo path counts. It is a controlled synthetic blockage/material smoke, not a full outage case, material calibration, standard certification, or general physical validation. Complete local suites pass: GPU 235 tests, 88.37% coverage; CPU 189 passed/12 optional skips, 77.27%. Ruff check/format pass across 117 maintained Python files; mypy passes 63 source files. GitHub Actions run 39595912339 passed all jobs for code commit `227f821b14a0f695fbdc93c2622f607d871ed273`; the hosted environment does not provide the native RT stack, so the LLVM RT gold case is validated locally.

**Edition 3.21 verification addendum:** Added a distinct solver-empty RT gold case: in Sionna's closed metal `box_one_screen`, a transmitter inside the enclosure and a receiver outside yield no depth-zero direct path and no valid full-solver paths at depth 3 with 2,000 samples/source, deterministic seed 13, reflection/refraction enabled, diffraction disabled. This separates no-path behavior from edition 3.20's blocked-direct case where indirect paths survive. The native RT test suite passes locally; complete GPU suite: 236 passed, 88.37% coverage; CPU suite: 189 passed, 12 optional skips, 77.27%. Ruff check/format pass across 117 maintained Python files, mypy passes 63 files, and GitHub Actions run 39596524674 passes all jobs for code commit `91969e2763e845da42f5c8a62c8356685d566a02`. Native Sionna RT is not installed in hosted CI, so the RT gold case remains locally validated on LLVM CPU. This synthetic solver-empty fixture does not validate field outage probabilities or receiver sensitivity thresholds.

**Latest verification addendum — edition 3.22:** `ChannelDataset` now stores a dataset-level JSON-validated `generation_report` inside its pickle-free NPZ, loads legacy NPZ files without that field, and records parent reports plus selected indices or component reports for dataset derivations. The RT generator report records scene/grid/solver configuration, software versions and package-source SHA256, candidate attempts, obstacle filtering, depth-zero labels, class-quota skips, empty full-solver outcomes, direct-only LOS rejection, CFR conversion failures, and accepted label counts. Runtime invariants require every started attempt and every full-solver call to be accounted for. Empty full-solver outcomes are kept in the report denominator but remain excluded as receiver observations; this avoids fabricating a zero-noise observation for an outage. Tests cover a forced empty-path mock, NPZ round-trip, legacy-file loading, and lineage. Full local GPU suite: 238 passed, 88.22% coverage; CPU: 191 passed, 12 optional skips, 77.23%. Ruff check/format pass on 117 maintained Python files and mypy on 63 files. The code-containing commit's GitHub Actions result will be added after push; source changes do not alter dependencies.

**Latest verification addendum — edition 3.22:** Dataset-level `generation_report` now preserves RT attempt denominators and rejection outcomes inside `ChannelDataset` NPZ files, including empty full-solver paths that are intentionally excluded from observations. Reports include scene/grid/solver configuration, software versions, package-source SHA256, and candidate/outcome counts; runtime invariants enforce accounting for all started candidates and returned full-solver calls. `select` persists exact selected indices and parent report; `concatenate` retains component reports. Legacy NPZ files remain readable. The generator's empty-path rejection is now measurable in the dataset-level report rather than disappearing from accepted rows. Full local GPU suite: 238 passed, 88.22% coverage; CPU suite: 191 passed, 12 optional skips, 77.23%. Ruff check/format pass on 117 maintained Python files and mypy on 63 files. GitHub Actions run 39598306545 passed all jobs for code commit `75a9be75cec05f89b949d71a312d3871defa2e96`.

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
| Quality gates | Latest complete locked CPU suite: 398 passed, 16 optional skips, 80.11% coverage; Ruff lint/format pass on 201 maintained Python files, mypy passes on 126 configured source files, and lock/build/diff checks pass | These are local checks; hosted Actions have not been polled for the latest commits. Native RT stays outside portable CI; backend capability probing and LLVM fallback are implemented. |
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
| P1 CUDA PHY | PUSCH/Rayleigh, TDL→PUSCH, flat-channel DMRS, impairment/decoder sanity, sample-domain TDL-A–E curves, exact same-channel 1-vs-2-RX diversity/correlation studies at 12-PRB/30-kHz and CP-contained 24-PRB/15-kHz, plus a phase-resolved CUDA memory profile pass in the pinned environment | Extend resource/receiver profiling to further delay spreads and link configurations; test broader correlation matrices and calibrated/measured arrays. Synthetic correlation is not array calibration. |
| P1 RT physics | Direct/two-ray/material checks, a deterministic geometry-label smoke, and matched three-seed path-sample/depth convergence sweeps on three built-in scenes pass on LLVM CPU; two built-in scenes generated pilot data. RT CUDA remains unavailable under current WSL OptiX setup. | Add blocked/outage and diffuse multipath cases; extend convergence across scenes, path-overlay, material qualification and trajectory gold cases; no GPU RT performance claim. |
| P2 NR receiver | SionnaPUSCHLSReceiver and framework-independent NR contracts pass; CUDA tests cover flat DMRS recovery, AWGN trend, Rayleigh loop, OFDM round-trip, CP CFO and bounded known-origin integer timing acquisition/correction, timing-impairment sensitivity, frequency-grid BER/BLER ordering, static two-tap sample-domain checks, sample-domain TDL-A–E curves, and same-waveform TDL-A diversity sweeps at 12-PRB/30-kHz and 24-PRB/15-kHz with ρ=0 and synthetic ρ=0.9; paired mobility/DMRS and fixed-payload multi-delay CP sweeps also pass. PSS/SSS acquisition has bounded continuous fractional-timing refinement with a same-trial two-tap improvement, but residual low-SNR failures remain. | Acquisition is not qualified under broad multipath, interference, channel evolution, or field conditions. Calibrated/real-array response, further numerologies/link configurations, and measured receiver evidence remain open before generalized receiver-capability claims. |
| P3 RT-to-NR and observation schema | Paths.cfr-to-OFDM-grid bridge, hash-addressed NR records, immutable scene-group splits, spatial/CFR/temporal leakage audits, native GPU TDL→PUSCH records, and six CPU/LLVM-retraced NR DMRS trajectories (24 timestamped observations) across three built-in and three procedural urban scenes pass. Retracing now reuses the trajectory seed across positions to control Monte Carlo ray-sampling jitter; a static-scene native test confirms identical CFR. The frozen scene-family split and receiver-visible CFR/temporal audit report zero cross-split near-duplicates and overlaps. Persistence and linear-complex one-step forecast baselines now run against the frozen split. | Expand to many independent qualified scenes and longer/more varied trajectories; make validation/test contain multiple independent scene families; qualify assets/materials, motion and outage policy; then run powered transfer/forecast evaluations. Current pilot cannot support generalization or field claims. |
| P4 analytical / standardized tier | Seeded 1–3 path fixtures, finite-bandwidth steering, joint Fisher diagnostics, 80-case two-path CRLB sweep, native Sionna v2.2.0 TDL→CFR/PUSCH A–E checks, sample-domain PUSCH A–E BLER curves, paired TDL-A receive-diversity and synthetic spatial-correlation sweeps, and residual synchronization/decoder checks | Extend analytic nuisance sweeps; validate TDL/CDL distributions, version drift and standard conformance; maintain profile-proxy labels separately. |
| P4+ science / P7 analytical M2 | The fixed-two-path CFR decoder has been trained and evaluated on 800 synthetic condition groups with identifiability-off/on ablations, profile-likelihood point estimates and group-calibrated set-valued inference. The on/off point-error difference is small and its paired interval includes zero; the profile-likelihood set achieves descriptive group coverage under the known-order/known-noise synthetic assumptions. This falsifies neither the mechanism nor establishes advantage, transfer, or a novel contribution. | Stress unknown model order, receiver/noise mismatch and condition shift; validate the equivalence decision rule; compare against more matched estimators; move to independently qualified RT scenes. |
| P6 fair-baseline implementation | RF, HistGB, XGBoost, logistic regression, engineered-feature MLP, raw flattened Re/Im MLP, CORAL-off/on, real CNN, and phase-invariant CVNN share leakage-protected grouped splits and saved predictions. Results record timing, learned parameter counts where defined, and optional CUDA peak allocation. Small/medium/large neural tiers and an immutable fixed-split capacity-sweep runner are implemented and tested. | Run the sweep on qualified identical observations; equalize/report training updates and tuning budget; add classical path estimators where appropriate; evaluate P6 E03–E08 with enough independent scenes. No comparative superiority is claimed. |

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

- **Mixed label semantics:** `ChannelDataset.y` remains a binary compatibility field, now accompanied by required `label_semantics` provenance. Standardized TDL/CDL, geometric RT, and synthetic debug generators identify their distinct meanings; selection preserves that meaning and concatenation rejects mixed semantics. Rich per-target masks/source/confidence and distinct observable-direct/blockage targets are still open. Never calculate one universal binary accuracy over incompatible semantics.
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

**Current observed status:** L0 passes locally: the latest full pinned Sionna/CUDA suite passes at 88.37% coverage, Ruff lint/format pass on 116 maintained Python files, mypy passes on 63 source files, and lock/build checks pass. With CUDA hidden, the latest full CPU suite passes at 77.27% against the 75% gate; GitHub Actions run 39590520407 passes on source commit 6fbd08e. Joint gain/delay local Fisher diagnostics have tests for rank, close-path conditioning, invalid inputs, noise scaling and agreement with the closed-form single-path CRLB with an unknown complex gain. An 80-case reproducible two-path sweep varies tone count, SNR, and separation; this remains a local numerical diagnostic, not a receiver or empirical resolution guarantee. GitHub Actions passed for prior source commit 26851a4 across Python 3.11/3.12 core, CPU-neural, and package-build jobs; commit 06f2a81 exposed a CPU-only coverage shortfall (73.19%, with all 175 tests passing and 7 GPU tests skipped). Follow-up commit b00e127 marks the CUDA-only adapter body as native-tested and adds CPU tests for result assembly; the CPU-emulated suite reaches 76.91%. Actions run 39582145663 passed all jobs for the coverage fix, and run 39582281995 passed all jobs for source commit 69d1df8. L1 direct/two-ray analytic checks pass. L2 has a controlled material-sensitivity fixture and matched three-seed solver-budget/depth sweeps on three built-in fixed links; convergence across broader geometries and path mechanisms remains incomplete. L3 has flat-channel DMRS and Rayleigh loops, frequency-grid BER/BLER sanity, cyclic-prefix CFO estimate/correction, bounded known-origin integer timing acquisition/correction, a static two-tap sample-domain PUSCH AWGN trend, and a time-varying sample-domain TDL-A waveform curve. The 1,024-codeword-per-SNR sample-domain run gives BLER 1.0, 1.0, 0.0225, 0.0010, and 0.0010 at 0, 6, 12, 18, and 24 dB; the last two points each have one block error and are statistically indistinguishable at this sample size. The configured 13-sample channel memory equals the shortest CP, so this run does not test channels longer than the CP. The frequency-domain TDL-A curve remains separately recorded. The paired 512-codeword fixed-allocation mobility stress run gives BLER 0.023 at 12 dB at 3 m/s versus 0.578 at 30 m/s; codeword-paired 95% bootstrap intervals for the treatment-minus-control deltas at 12/18/24 dB exclude zero, but inference remains conditional on this TDL-A simulator and single receiver configuration. Blind NR frame synchronization and fractional timing remain open. Matched fixed-payload CP-delay sweeps now cover three SNRs at 30 kHz and a second numerology at 30 dB; broader CP-margin coverage, antenna breadth, and independently qualified link curves remain open. L4 now has a script-reproducible three-scene leave-one-scene-out pilot, but no frozen qualified dataset; all three held-out AUROCs are near chance and three scene families do not support reliable scene-level intervals. L5 has a matched seed/split evaluation pipeline and leave-one-scene-out baseline runs, but no registered primary method, compute-budget matching, or complete baseline table. L6 has no independent measured evidence, and L7 has not been performed. Native batches of 4,096 realizations for each TDL-A–E profile match Sionna's nominal delays and mean powers within the stated Monte Carlo tolerance; the 8,192-realization TDL-A tail check is an implementation sanity check, not 3GPP certification. The TDL-to-CDL profile-proxy pilot and three-scene RT pilot are reported above with hashes, split identities and limitations. Receiver-visible NR records have typed contracts and content-addressed storage; spatial, CFR-fingerprint and temporal-overlap audits exist, but the duplicate criteria have not been qualified against a production dataset. Sionna RT CUDA remains unavailable under WSL OptiX. Full CDL/tail validation, broader qualified geometry data, blind/fractional timing synchronization, broader sample-domain link characterization, target semantics, main-method ablations, external measurement and cold independent reproduction remain open.

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

**Edition 3.12 CP-delay SNR/numerology sensitivity (simulation only) - 2026-10-21:** The same 512-codeword, seed-82417, matched-bits/AWGN sweep was rerun at 12 PRB/30 kHz and 20 dB and 40 dB, and at 12 PRB/15 kHz and 30 dB. At 20 dB/30 kHz, BLER for in-CP reference delay 8, echo delays 11 and 13, and beyond-CP delay 16 was 0/512, 0/512, 0/512, and 65/512 (0.127); delay 20 produced 512/512 errors. At 40 dB/30 kHz, the same conditions yielded 0/512, 19/512 (0.0371), 145/512 (0.2832), and 512/512, with delay 20 also 512/512. At 30 dB/15 kHz the minimum CP is 12 samples, the reference delay is 7 (5 samples of margin), and the beyond-CP condition is 15 (3-sample excess); BLER was 0/512, 11/512 (0.0215) at delay 11, 99/512 (0.1934) at delay 13, 510/512 at delay 15, and 512/512 at delay 20. In each run, paired exact McNemar tests were Holm-adjusted over six delay comparisons; adjusted p-values for key rising-delay comparisons were 2.71e-19 (16 samples at 20 dB/30 kHz), 1.14e-5 and 1.79e-43 (11 and 13 samples at 40 dB/30 kHz), and 0.00293, 1.26e-29, and 2.98e-153 (11, 13, and 15 samples at 30 dB/15 kHz). The 30-dB/30-kHz baseline is archived separately above. Raw JSON files and SHA256 values: docs/research/results/pusch-static-cp-margin-sweep-12prb-30khz-20db-512.json (3fb6bdc980b68ae6b4c271bf9c12f91fe79d3d3d3ad565b347c5c7497be340b9), docs/research/results/pusch-static-cp-margin-sweep-12prb-30khz-40db-512.json (8ab99c07c5f798307134259644e2204ea910d81396d3b630f100c6c4f16bd57a), and docs/research/results/pusch-static-cp-margin-sweep-12prb-15khz-30db-512.json (5d2733058b3784e0231cb838ecd537db64f4f6383ae23295ad160aeb87d613a0). Each artifact records code revision a75293b596b4889c73e186e80578bc83ec5eafae and package-source hash 42804eaf16e55a489356c4e5a1c8b90feae46e8bd5b9a26b75690a3d524b31d6. Multiplicity adjustment is within each configuration's six comparisons, not across the three configurations. These are parameter-sensitivity diagnostics for one deterministic finite-burst two-path model with zero prehistory, one seed and one allocation; they do not establish scene-level inference or general NR performance.

**Edition 3.13 two-path local identifiability sweep (analytical diagnostic only) - 2026-10-21:** The new script runs 80 cases over 256 or 1,024 uniformly spaced tones at 30 kHz, 0/10/20/30-dB per-complex-tone SNR, and ten path separations from zero to two reciprocal-bandwidth intervals. It records local Fisher rank/conditioning and each path's marginal delay CRLB standard deviation only when the full six-real-parameter local model is identifiable. At exactly coincident delays, rank was 4/6 and covariance was correctly withheld in all eight tone-count/SNR combinations. At 20 dB and separation 0.01/B, the two delay standard deviations were 58.17/73.58 ns for 256 tones (7.65-MHz span) and 7.29/9.22 ns for 1,024 tones (30.69-MHz span). At 1/B they were 0.493/0.624 ns and 0.0619/0.0783 ns, respectively. Changing from 256 to 1,024 tones changes both observation count and bandwidth, so this comparison cannot isolate bandwidth alone. The local bound assumes known frequencies, path count, and gains' operating point, circular Gaussian noise, and no synchronization or channel-estimation uncertainty; it is not a global resolution threshold or proof a receiver attains these bounds. The complete 80-case artifact is docs/research/results/identifiability-two-path-fisher-sweep-80.json, SHA256 4976875695b856149320f949f044a945d0a74b44ba5cc11c95cc25ef06e68e59, from code revision 9239741721e2e99c05e6fba83488c1f7c77f62ca; package-source SHA256 42804eaf16e55a489356c4e5a1c8b90feae46e8bd5b9a26b75690a3d524b31d6 and runner SHA256 9625a4ec6d54189665cc641eb437d4303bc523a03e5572de500551f4051e5182. This validates behavior of the stated local formula across a small controlled grid, not a learned model, estimator, or empirical NR channel.

**Edition 3.14 fixed-link Sionna RT convergence diagnostic - 2026-10-21:** scripts/run_rt_solver_convergence.py evaluates the built-in box_one_screen scene at one fixed SISO link (TX [-2, 0, 2.5] m; RX [0, 0, 2.5] m), 3.5 GHz, and a 256-tone, 30-kHz grid. The deterministic CPU/LLVM solver was run with seeds 11, 23, and 47, ray budgets 200/2,000/20,000/100,000 samples per source at max depth 3, and max depths 0/1/2/3 at the 100,000-sample budget. The 100,000-sample, depth-3 CFR is only a within-seed reference. At depth 3 and 200 samples, valid path counts were 8/9/10 and relative CFR L2 errors against the reference were 0.229/0.155/0.0965 for seeds 11/23/47. At 2,000, 20,000, and 100,000 samples, all seeds found 11 paths; relative CFR errors were below 1.1e-5 at 2,000 and 1.8e-5 at 20,000, with 100,000 as the zero-error reference. Sample-budget differences were not monotonic at this small numerical level. At 100,000 samples, depths 0 and 1 found only the direct path, depth 2 found two paths, and depth 3 found 11; depth-0/1 and depth-2 CFR relative errors versus depth 3 were approximately 0.571 and 0.517. Thus this single fixed link shows sensitivity to interaction depth and ray budget, but the highest setting is not independently certified ground truth and the result does not qualify convergence on other scenes, materials, receiver locations, or stochastic seeds. Scene XML/tree hashes, per-run path delays/powers, CFR hashes, timings, runtime versions, and config are archived in docs/research/results/rt-box-one-screen-convergence.json, SHA256 5d1095bc307b675d0a4bf51ee39926b66546330dbf4fda0fb1bc6bc96bbbe9e8, code revision 24bf9cffdbd23b3c8013b33cdf267549fec233a9, and runner hash 12bd62e6797c73514e1a2eaad765950291010e56b2a80eaad983ef5573734322. This CPU/LLVM implementation check establishes neither material realism nor GPU RT performance.

**Edition 3.15 three-scene Sionna RT convergence sweep (CPU/LLVM implementation diagnostic only) - 2026-10-21:** The committed runner was applied to one fixed SISO link in each of box_one_screen, simple_street_canyon, and etoile, with 3 seeds, budgets 200/2,000/20,000/100,000 samples per source at max depth 3, and max depths 0-3 at the 100,000-sample budget. All CFR comparisons are paired to the 100,000-sample, depth-3 result for the same seed; this is a reference, not certified truth. At depth 3, box_one_screen found 11 paths for all seeds by 2,000 samples, with relative CFR L2 error <=1.1e-5; the 200-sample error range was 0.096-0.229. Canyon found 14 paths for all seeds at 100,000 samples; at 20,000 samples it found 13-14 paths with error <=6.8e-4, at 2,000 it found 10-11 with error 0.0025-0.0066, and at 200 it found 6-7 with error 0.0265-0.0753. Etoile found 6-8 paths at 100,000 across seeds; 20,000-sample runs found only 4-6 paths and had error 0.105-0.291, while 200/2,000-sample error was about 0.285-0.292. This shows the sample budget that looked adequate for the box scene does not transfer to the tested Etoile link. At depth 100,000 samples, lowering max depth from 3 to 0/1/2 changed box path count from 11 to 1/1/2 with CFR error 0.571/0.571/0.517; canyon count from 14 to 1/4/8 with error 0.254/0.0449/0.0456; and Etoile count from 6-8 to 1/3/4 with error 1.80/0.286/0.308-0.361. Scene XML/tree hashes, per-run powers/delays/CFR hashes, timings, and runtime provenance are stored in docs/research/results/rt-box-one-screen-convergence-multiseed.json (SHA256 15870716123abacc69cbebc29aefb8d1d3ae8dc32ccb8cef9e2aff24402c0721), docs/research/results/rt-simple-street-canyon-convergence-multiseed.json (SHA256 e3e857e4844a48cc9e6e8e9fd77cdeb81566c978379d7c29aabdcaeaee649b55), and docs/research/results/rt-etoile-convergence-multiseed.json (SHA256 1db32ffd3f54ed951adbacc96ee63aa6cd23fe94fff8fc09c1c42b50e8943492). They use runner SHA256 12bd62e6797c73514e1a2eaad765950291010e56b2a80eaad983ef5573734322 and code revision 61a24c634e7376b3ef4979feb9cf16fed01170ff. Three fixed links do not qualify solver convergence across each scene's spatial extent, material choices, antenna arrays, stochastic settings, or held-out environments. The highest sample count is not ground truth; overlays and physical/material audits remain required.

**Edition 3.16 reproducible standardized profile-proxy dataset (simulation source data only) - 2026-10-21:** `scripts/run_profile_proxy_dataset.py` generated 40,960 noisy CFR rows on CUDA using Sionna 2.2.0's TR 38.901 TDL/CDL A–E profiles, selector `spec_version="19.2"`, seed 78021, 3.5 GHz carrier, 128 tones at 30 kHz, 20 dB SNR, and delay spreads 30/100/300/1000 ns. Counts are 4,096 per family/profile and 1,024 per family/profile/spread; class-proxy counts are 24,576 (A/B/C) and 16,384 (D/E). The NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU produced the 39,040,864-byte dataset; SHA256 `efd5705592f6bdb9aca25320efda60bab1453c474b2fdde1fa5056f86a081521`. Its provenance sidecar `docs/research/results/profile-proxy-tdl-cdl-a-to-e-4096-seed78021.json` has SHA256 `2a2cf4c0eda1dcaabdc4d5e7654e1e33133e642cef70c38eec0ece9b35609aaa`, identifies source revision `2e8244e794e1d39be5f05b9c81ac45b0c055fe7c`, package source `3255cfaa6374d71691240a5d654ba00f7032b7df95d0c5a71fa71e511ca647ea`, and runner `7e569af9db6b7463eecb592380e3e67f5d2dcdd0eb9c5b8e6ea7befd1cecfeaf`. The NPZ remains a local ignored output and can be regenerated with the recorded command and manifest. These labels represent standardized stochastic profile proxies only; they do not encode geometric LOS/NLOS truth, measurement validity, or classifier performance. Native repeated-seed tests also confirm the generator leaves Sionna's shared RNG state and PyTorch global RNG state unchanged. Latest local validation is 88.49% full GPU coverage and 76.74% CPU-emulated coverage against a 75% gate; Ruff check/format, mypy, lock validation, package build, and GitHub Actions run 39587703216 pass. The pasted earlier 73.19% CPU run is historical evidence from a prior commit and is superseded for current source by the passing 76.74% run. Full CDL standard conformance, receiver impairment sensitivity, independent geometric scene evaluation and measured-channel evaluation remain incomplete.
**Edition 3.17 profile-disjoint standardized profile-proxy baseline diagnostic (not physical LOS evidence) — 2026-10-21:** `experiments/profile_holdout.py` and `scripts/run_profile_holdout_baselines.py` implement leave-one-profile-out evaluation on the manifest-backed 40,960-row dataset from edition 3.16. Each outer fold holds out all 8,192 rows (both TDL/CDL, one profile) and keeps that profile out of its 26,214 training and 6,554 validation rows; inner splits are link-group-disjoint. Features are the same globally gain- and phase-normalized complex CFR real/imaginary vector for all methods; metadata and profile IDs are used only to define folds and validate proxy-label semantics. Logistic regression, a 128/64/32 MLP, and a 250-tree random forest use the same outer folds. Youden-J thresholds are chosen on validation only; the MLP uses group-disjoint validation log loss for early stopping (min delta 1e-4, patience 18), and the random forest is single-worker for exact repeatability. Descriptive class-macro profile recalls were 0.92510, 0.92757, and 0.92506 for logistic regression, MLP, and random forest; profile-macro recalls were 0.91978, 0.93159, and 0.92144. These small descriptive differences are not evidence of method superiority: there are only five profile types, each held-out fold has one label class, and no profile-level confidence interval or pooled AUROC is valid for this setup. Repeating the full run produced exactly equal score/threshold arrays and the same prediction artifact SHA256. Report: `docs/research/results/profile-holdout-baselines-seed20261021.json`, SHA256 `4fd6b110aa59bbe4bf1088efd5481e722cfbc4471c29638fbbdb8a91480020af`; pickle-free predictions: `docs/research/results/profile-holdout-baselines-seed20261021.json.predictions.npz`, SHA256 `5f83a9391a74a28a434d6dadbdca555e652aa12c768dd249c289a32957b241fd`. The report records input dataset SHA256 `efd5705592f6bdb9aca25320efda60bab1453c474b2fdde1fa5056f86a081521`, source revision `6fbd08e0f21740f2a3b412ec13f8a94d9fe46811`, package-source SHA256 `c616d37d580ec0634738bc233830d4b3d6406fee8e289bca1f5715de3ac1249a`, and runner SHA256 `ef8d8332580464c2b73b449cbcee9dea46c19da0f3f88afa920779d972113269`. This advances the controlled standardized-profile holdout diagnostic only. It establishes neither geometric LOS/NLOS recognition nor transfer to RT, measured channels, other profile sets, or a physical receiver. E06 still requires unknown RT/measured domains and explicit leakage/profile-ID audits; the main method and compute-matched ablations remain open.
**Edition 3.18 native CDL A–E profile-table and Monte Carlo check (implementation consistency only) — 2026-10-21:** The native test `tests/test_integrations/test_cdl_profile_tables_native.py` and runner `scripts/run_cdl_profile_validation.py` exercise CDL A–E on CUDA using Sionna 2.2.0 selector `spec_version="19.2"`, one omni single-polarized antenna per end, 3.5 GHz, static zero-speed links, 100-ns delay spread, 4,096 channel realizations/profile, and base seeds 31900–31904. The check sorts the resource-table delays as returned by the CIR generator, verifies exact scaled delay matches, derives expected normalized tap powers from the same versioned resource JSON, combines CDL-D/E specular LOS and zero-delay NLOS-cluster powers according to the pinned K factor, and compares Monte Carlo path powers with relative tolerance 10% plus absolute tolerance 0.005. Observed total powers for profiles A–E were 0.99827, 0.99994, 1.00455, 1.00409, and 0.99987; maximum absolute per-tap power errors were 0.002869, 0.002769, 0.002555, 0.004264, and 0.0003904. The provenance artifact is `docs/research/results/cdl-profile-validation-19.2-a-to-e-4096-seed31900.json`, SHA256 `1c29b597ae04fbd7b370143f1cca98e70efaa805e2090bef2d2d5b60c31db094`; source revision `235ccc9a275ca1c38520ce94f3c010fe982e487a`, package-source SHA256 `c616d37d580ec0634738bc233830d4b3d6406fee8e289bca1f5715de3ac1249a`, runner SHA256 `86cced3a56efd10f54fa26ffa417b8f3175aa9a97516392f2b2f57d3d67795c2`, and hashes of all five pinned resource files are recorded in the artifact. All five new native parameterized checks pass; the complete pinned GPU suite passes at 88.37% coverage and the CPU suite at 77.27%, with Ruff check/format over 116 maintained Python files, mypy over 63 source files, lock/build checks, and GitHub Actions run 39593450490 passing. This is a versioned simulator self-consistency and finite Monte Carlo check, not independent validation against the standard text, cross-version 16.1/19.2 equivalence, full CDL conformance, measured propagation, or geometric LOS truth. Broader delay-spread/speed/array cases, receiver impairment checks, and independent RT/measured validation remain open.

**Edition 3.19 CDL delay-spread sensitivity of the pinned A–E table check (implementation consistency only) — 2026-10-21:** The native A–E Monte Carlo test now covers 30, 100, 300, and 1,000 ns at 3.5 GHz, zero speed, 1x1 omni single-polarized arrays, 4,096 realizations/profile/spread, and Sionna 2.2.0 selector `19.2`; the common seed base is 31900. The three newly run CUDA batches at 30/300/1,000 ns all matched the same normalized profile powers as the 100-ns run (maximum absolute errors for A–E: 0.002869, 0.002769, 0.002555, 0.004264, and 0.0003904; total empirical powers: 0.998267, 0.999937, 1.004551, 1.004094, and 0.999869). The generated tap delays pass the profile-table check at each spread and scale with the selected delay-spread parameter. Provenance artifacts: `docs/research/results/cdl-profile-validation-19.2-a-to-e-4096-seed31900-30ns.json` SHA256 `cb994c00b61bd48bf9784f2821b497fc1ac23b9240aaccebc469129ebf68b489`; `...-300ns.json` SHA256 `21c8ddaee69bafdf5404587b1a9d6534dd36dc5d1ca59cf84dda39c55dcae8a2`; and `...-1000ns.json` SHA256 `03e753b8d4074831499718a66fdef64ebf10e003868bd18d6a20038c14229c14`. Each artifact records the pinned per-profile resource hashes, source revision, environment and full tap report. This adds delay-scaling coverage across four configured spreads; it does not add independent statistical evidence for the PDP because CDL normalized powers are unchanged by scalar delay-spread scaling. Speed/Doppler, array correlation, cross-version equivalence, independent standard-text validation, full CDL certification, measured propagation, and geometric LOS truth remain open.

Shared provenance for the three new spread reports: source revision `61350e8ca1f5f81d1a26bf0b8755b40dbfc70d17`, package-source SHA256 `c616d37d580ec0634738bc233830d4b3d6406fee8e289bca1f5715de3ac1249a`, runner SHA256 `86cced3a56efd10f54fa26ffa417b8f3175aa9a97516392f2b2f57d3d67795c2`, and Sionna 2.2.0 profile-table selector `19.2`. The three report files have the same pinned resource hashes as the edition 3.18 100-ns report.

**Edition 3.20 blocked-direct and diffuse-interaction RT gold case (synthetic LLVM implementation test only) — 2026-10-21:** `tests/test_integrations/test_rt_blockage_diffuse_native.py` loads Sionna's pinned `box_one_screen` scene with `merge_shapes=False`, 3.5 GHz, 1x1 isotropic vertical arrays, depth-zero geometric oracle and depth-3 full solver. With deterministic seed 13 and 2,000 samples/source, a same-side link at TX `[-2,-1,2.5]` m and RX `[-2,1,2.5]` m has one direct path and 50 total paths, including indirect paths. Across the screen, TX `[-2,0,2.5]` m to RX `[2,0,2.5]` m has zero direct paths but 31 full-solver indirect paths when screen scattering is zero. Setting the screen's Sionna scattering coefficient to 0.5 retains zero direct paths, yields 43 total paths, and tags 12 valid paths with `InteractionType.DIFFUSE`; the no-scattering case tags zero paths diffuse. The test checks the geometric direct-path label independently from total received path existence and inspects Sionna's interaction-type bits. It asserts qualitative path conditions rather than exact stochastic counts. Reproduce locally with `/home/nguyenl37/.venvs/sionna220-gpu/bin/python -m pytest --no-cov tests/test_integrations/test_rt_blockage_diffuse_native.py -q` in the pinned Sionna environment. Test source SHA256 `a26f5336b63f6902dcd05278a4e732971a3658389a96a07ffc8fca34e42d15f8`, source revision `227f821b14a0f695fbdc93c2622f607d871ed273`. This confirms a blocked-direct synthetic link with surviving indirect energy, not a complete outage, realistic material parameters, scene-family generalization, or measured propagation. Outage handling, broader material qualification, path-overlay review and independent scenes remain open.

**Edition 3.21 solver-empty enclosure RT gold case (synthetic LLVM implementation test only) — 2026-10-21:** The native test in `tests/test_integrations/test_rt_blockage_diffuse_native.py` reuses the closed metal box geometry with TX `[0,-1,2.5]` m inside and RX `[6,0,2.5]` m outside. Under the same pinned 3.5-GHz SISO setup, deterministic seed 13, 2,000 samples/source, max depth 3, enabled LOS/specular/diffuse/refraction and disabled diffraction, the depth-zero direct oracle returns zero paths and the full solver also returns zero valid paths. The new assertion complements the edition 3.20 case: the across-screen link is geometrically blocked but has indirect received paths, while the enclosure pair is solver-empty under its recorded interaction budget. Reproduce with `/home/nguyenl37/.venvs/sionna220-gpu/bin/python -m pytest --no-cov tests/test_integrations/test_rt_blockage_diffuse_native.py -q`. Updated test SHA256 `912f298a025ca25aeaecd30032feb2b7fefee0d5d2bc97eb9d956442bfc63f4d2`, source revision `91969e2763e845da42f5c8a62c8356685d566a02`. This is an empty-path outcome in one synthetic, closed geometry with Sionna's configured solver; it does not establish propagation probability of outage, a receiver noise/sensitivity threshold, material realism, or a real-world link budget. Outage-state storage, thresholded receiver observability, material qualification and independent measured scenes remain open.

**Edition 3.22 RT attempt-funnel reports and dataset lineage (data QA engineering):** `ChannelDataset` has an optional `generation_report` JSON object. It is validated as finite JSON, stored in the NPZ, and loaded compatibly when absent in legacy files. Dataset selection records the exact indices and parent report; concatenation keeps every component report. `generate_raytraced` reports scene/grid/solver configuration, environment versions, active package-source SHA256, attempt counts (candidate attempts, obstacle rejections, depth-zero label outcomes, class-quota skips), and full-solver outcomes (empty paths, direct-only LOS rejected by policy, CFR conversion failures, accepted rows). Runtime accounting invariants require `attempts_started == obstacle_rejections + depth_zero_oracle_calls` and full-solver calls to partition into empty-path, policy rejection, CFR failure, or accepted outcomes. A deterministic mock forces one empty full-solver result, and the native `box_one_screen` generator verifies report construction. Dataset save/load, legacy NPZ loading, selection, and concatenation tests pass. Code commit `75a9be75cec05f89b949d71a312d3871defa2e96`; GitHub Actions run 39598306545 passes. This report covers successful balanced generation runs; it does not yet serialize partial reports when a solver raises an exception, turn an empty solver outcome into a receiver observation, or define calibrated outage/censoring thresholds. Solver-error recovery, receiver-visible status semantics, and independent-scene denominators remain open.

**Edition 3.23 RT failure reports and current test evidence — 2026-10-21:** `RaytraceGenerationError` now preserves a structured generation report when the depth-zero LOS oracle or full PathSolver raises, or when the attempt budget cannot satisfy both class quotas. Reports distinguish attempts from successful solver calls and errors, identify the failing stage/attempt/type/message, state `solver_error` or `insufficient_samples`, and retain accepted class counts and the requested quota. Accounting invariants partition candidate attempts into obstacle rejections plus oracle attempts, oracle attempts into calls plus errors, full-solver attempts into calls plus errors, and returned full-solver calls into empty, policy-rejected, conversion-failed, or accepted outcomes. Deterministic mock coverage exercises oracle failure, full-solver failure, and exhausted quota. The exact CPU command `uv run --no-sync --locked python -m pytest` passes: 191 passed, 12 skipped, 77.37% coverage (75% required); the pasted 175-test/73.19% coverage output is from an older source snapshot and is not reproducible at the current HEAD. The full pinned Sionna GPU suite passes: 238 passed, 88.26% coverage. CI-scoped Ruff lint/format, mypy (63 source files), `uv lock --check`, and `uv build` pass locally. These checks validate software behavior, not physical outage semantics or scientific generalization; no solver-empty result is converted into a calibrated receiver-visible observation, and empirical scene validation, material calibration, and receiver thresholds remain open.


**Edition 3.24 matched sample-domain channel-memory stress test (conditional simulation evidence) — 2026-10-21:** The opt-in CP-overrun mode is fail-closed by default and was exercised using the native Sionna 2.2.0 waveform-domain TDL-A PUSCH link on CUDA. Three matched 512-codeword curves per SNR used 12 PRB, 30 kHz, 3.5 GHz, 3 m/s, one receive antenna, ideal synchronization, and 12/18/24 dB. The 100-ns control has 13 samples of channel memory against a 13-sample minimum CP; 300 ns has 14 samples (one over CP), and 1,000 ns has 17 (four over CP). Transmitted bits and generated noise prefixes are matched by codeword; channel truth intentionally changes with delay spread. Complex AWGN uses independently seeded per-codeword streams, and `tests/test_integrations/test_seeded_awgn.py` checks exact prefix equality across differing waveform lengths on CPU and CUDA.

The 100-ns control had 7/512, 2/512, and 2/512 CRC failures at 12/18/24 dB. At 300 ns the counts were 12/512, 0/512, and 0/512. Paired treatment-minus-control BLER differences were +0.00977 (95% paired bootstrap CI [-0.00195, 0.02344], Holm-adjusted exact McNemar p=0.680), -0.00391 ([-0.00977, 0], p=1), and -0.00391 ([-0.00977, 0], p=1), respectively. These comparisons are inconclusive at all three SNRs. At 1,000 ns the failure counts were 95/512, 4/512, and 10/512; paired differences were +0.17188 (CI [0.13672, 0.20703], Holm p=1.36e-21), +0.00391 ([-0.00391, 0.01172], p=0.625), and +0.01563 ([0.00391, 0.02930], p=0.0430). This is evidence of a substantial paired degradation at 12 dB and a small effect at 24 dB for this one simulated link/configuration; it is not evidence of scene-level performance, a receiver recommendation, or general NR behavior. Only three SNR points, one TDL profile, one allocation, one seed family, and one SISO configuration were evaluated; independent geometries, realistic receiver synchronization, and measurement validation remain outstanding.

Raw curves and paired reports are `docs/research/results/tdl-pusch-time-bler-tdla-12prb-30khz-100ns-control-512-seed51005.json` (SHA256 `c05fcf966af6474b38af31f2b54e6fd4fb08630ef74ce8c6e4299cacc660e547`), `...-300ns-cp-overrun-512-seed51005.json` (`a217389131b27c6877074574e2c8e084fb62a0cd29bc81e753778ce9e010042f`), `...-1000ns-cp-overrun-512-seed51005.json` (`3bde2c074e2d89389170e0e5b639ef26dd5dd4d9ec0f021431c162f0db5c836b`), `tdl-pusch-time-bler-tdla-100ns-vs-300ns-cp-overrun-512-seed51005.json` (`85189247c3ae3e75a4e664b94c17299a8b6ffee7d2b2ce7ff7dda90847e1bbd9`), and `...-100ns-vs-1000ns-cp-overrun-512-seed51005.json` (`3304f016e41fdbc701d92f134ea0c1febf5b581b501a2a736deea7c04bc6eeb1`). All inputs record source revision `6a45da530b9dc5c533805c890ec7a681e07c32aa` and package-source SHA256 `fd94c58598552823d9411efaad656822a9ad57fbe14830d228b3850d1f7cbcfb`. Runner SHA256 is `b0b332249ef1945110f5498080b9e2b264e55a2804d296d8facebe6cb9910d1a`, paired analysis script SHA256 `9c1550367b9086931e56e0f3ebd5a40340f594a0b9244dfcf8e786696afa5d8b`, and matched-noise test SHA256 `4173a5b12b1fba0e142771497a6c64223cbb658fe6d6fe5010fadb9a181dc961`. Current CPU suite: 200 passed, 13 skipped, 77.45% coverage (75% gate); the attached 175-test/73.19% report is from an older checkout and is superseded. The pinned GPU suite and CI passed on source commit `6a45da5` (GitHub Actions run 39602613248). This edition advances only sample-domain CP-memory stress validation. Blind/fractional synchronization, broader SNR/numerology/mobility/receiver settings, independent scenes, external measured data, primary-method implementation and ablations, and full publication gates remain unfinished.


**Edition 3.25 fractional-sample waveform impairment (receiver validation only) — 2026-10-21:** `integrations/sionna/ofdm_impairments.py::apply_fractional_timing_offset` now applies a scalar fractional delay in (-1, 1) samples using a finite Blackman-windowed sinc interpolator, with zero-valued samples outside the finite capture. The integer timing operator remains separate. A CPU complex-tone test at 0.071 cycles/sample and 0.375-sample delay agrees with the ideal interior phase response to 2e-5 absolute/relative tolerance; input and kernel bounds are tested. The native CUDA PUSCH impairment test injects a 0.35-sample delay into an OFDM waveform and confirms it changes the demodulated grid and DMRS channel estimate versus the ideal waveform. This validates impairment construction and receiver sensitivity for one configuration only. It does not implement blind frame acquisition, fractional-delay estimation/correction, multipath-aware synchronization, or a standards-complete NR synchronizer.

The full CPU suite passes with 202 passed, 13 skipped, and 77.62% coverage; the pinned CUDA/Sionna suite passes with 251 passed and 88.25% coverage. Targeted Ruff lint/format and mypy pass. GitHub Actions run 39604168013 passes all Python 3.11/3.12, CPU-neural, and package-build jobs for source commit `9cf3d42`. Source SHA256: impairment helper `5f851390ab4693b603153ea608f07a616f11ef1beb701c15a07140452492c5ce`; CPU tone and validation tests `ab7e55972f78c5a40e9570071b72fc6a53dab40fe3aea36c60b92a46c9726229`; native CUDA receiver test `765b1b3e770629dff60c93412129294cb6de07eb9bc1ebf59a14c6e563ac7908`. Remaining receiver work includes blind NR frame/PSS-SSS acquisition, fractional timing estimation and correction, multipath-aware synchronization, and broader antenna, numerology, and fading configurations.


**Edition 3.26 controlled blind NR PSS/SSS cell acquisition (sequence and receiver implementation check) — 2026-10-21:** The project now has `integrations/sionna/nr_sync.py`, a NumPy-based reference implementation of the TS 38.211 PSS/SSS length-127 sequences and their 127-RE SS/PBCH mapping (TS 38.211 V19.3.0, clauses 7.4.2.2–7.4.3.1: [official ETSI PDF](https://www.etsi.org/deliver/etsi_ts/138200_138299/138211/19.03.00_60/ts_138211v190300p.pdf)). It searches unknown integer sample timing, all three N_ID^(2) hypotheses, and a caller-provided finite CFO grid; SSS correlation resolves N_ID^(1) and the physical cell ID. Configurable normalized-score gates reject a deterministic noise-only capture. Tests check PSS m-sequence cyclic autocorrelation, uniqueness of all 336 SSS sequences for a fixed N_ID^(2), and one seeded 18-dB capture with N_ID^(1)=237, N_ID^(2)=2, 23-sample timing offset, and on-grid 3-kHz CFO; it recovers the correct PCI and timing with PSS/SSS scores above 0.9.

This is controlled acquisition, not a complete SS/PBCH transmitter or production cell-search implementation. Numerology and CFO hypotheses are supplied, timing resolution is integer samples, CFO is grid-quantized, SSS assumes a flat channel, and the threshold has not been calibrated for a target false-alarm/miss probability. Multipath, off-grid CFO, fractional timing estimation/correction, multiple SSB hypotheses, PBCH, mobility and receiver sensitivity remain unvalidated. The code does not support absolute propagation ToA claims.

The full CPU suite passes with 207 passed, 13 skipped, and 78.14% coverage; the pinned CUDA/Sionna suite passes with 256 passed and 88.26% coverage. Ruff lint/format and mypy pass, including 64 source files. GitHub Actions run 39605336083 passes all Python 3.11/3.12, CPU-neural, and package-build jobs for commit `f0386ba`. Source SHA256: implementation `c649aa5a2c3bb11d211dff6fafdac2a527a103492ba587cbf5daf55c1580b0aa`; tests `6d88508bf98c17769af79904497576a70375ff6d8e5789d1549107109d3ce2e0`. This advances one bounded slice of receiver synchronization; the remaining end-to-end research gates remain open.


**Edition 3.27 PSS-derived channel-aware SSS under one in-CP multipath case (controlled receiver check) — 2026-10-21:** The PSS/SSS acquisition now estimates the complex channel on the known PSS subcarriers and scores SSS candidates against the channel-shaped sequences, assuming the channel remains static between SSB symbols 0 and 2. A deterministic two-path capture has a direct tap of 1, an echo of 0.9+0.1j delayed by 12 samples within the 32-sample CP, 17 leading timing samples, an on-grid 3-kHz CFO, and seeded 12-dB AWGN. The detector recovers integer timing 17, CFO 3 kHz, and PCI 1006. In this one realization, the channel-aware SSS score is 0.9866; applying the existing flat-channel SSS score to the same corrected symbol gives 0.7396. This comparison demonstrates that the PSS-derived channel response helps match this selected static multipath capture; it is not a measured channel result, false-alarm/miss characterization, BLER result, or general synchronizer performance claim.

The finite CFO hypothesis set, configured numerology, integer timing resolution, static-channel assumption, one echo delay/gain, and one noise realization bound this evidence. Doppler/channel evolution between PSS and SSS, fractional timing, path delays beyond CP, off-grid CFO, SSB/PBCH completeness, and threshold calibration remain open. Code commit `d088dd8` passes the full CPU suite (208 passed, 13 skipped, 78.19% coverage) and pinned CUDA/Sionna suite (257 passed, 88.25%); Ruff lint/format and mypy over 64 source files pass. GitHub Actions run 39606314375 passes Python 3.11/3.12, CPU-neural, and package-build jobs. Source SHA256: `integrations/sionna/nr_sync.py` `eee02b5e11ffe079d4a0a61a3f12916b566c58b33b73efa17195b6a112552d51`; `tests/test_integrations/test_nr_sync.py` `496d89e9fe2ed59df158f050d04581edf2055111de734204bd76022738403453`. This closes only one static in-CP multipath implementation case; receiver qualification and the broader research gates remain open.


**Edition 3.28 profile-proxy baseline resource audit (one CPU configuration; descriptive, not a matched-compute claim) — 2026-10-21:** `experiments/profile_holdout.py` now records per outer fold fit wall time, validation and held-out prediction wall time, train/validation/test row counts, feature count and training-array bytes, fitted learned-parameter count, tree count/nodes/leaves/depth, and serialized estimator bytes. Tests assert finite nonnegative timings, exact input-buffer accounting, positive fitted sizes, and RandomForest tree structure. Commit `f6d9d69` was pushed to `origin/main`; source SHA256 `ddf18f119f23ee939b9ff9d82bf9e2bd9a1db90b8f5cd98a25929fdf4cb3c90b`, test SHA256 `ceeeacc5f5534f44f86a0f6d2bca04f34b3518a811ccfdd9b5f86bca1ccfde37`.

The instrumented runner was executed on the existing 40,960-row `SIMULATED_PROFILE_PROXY_NOT_GEOMETRIC_LOS` dataset, with 256 real-valued normalized-CFR features, 26,214 training rows in each group-disjoint outer fold, the same five held-out profiles and group-disjoint validation protocol, and no held-out-label threshold tuning. Mean profile macro recall was 0.9198 for logistic regression (257 fitted coefficients), 0.9316 for the MLP (43,265 learned weights), and 0.9214 for the 250-tree random forest (399,230–505,980 tree nodes across folds). Serialized model sizes were respectively 8,271 bytes, 706,547–706,574 bytes, and 32,012,777–40,552,777 bytes. CPU fit times were 1.3–5.5 seconds, 10.8–61.3 seconds, and 56.6–83.0 seconds per fold; held-out inference was 0.014–0.021, 0.010–0.030, and 0.149–0.292 seconds per fold. The broad run-to-run timing variation—especially the 61-second first MLP fold—shows these are host-load-sensitive descriptive measurements, not a controlled latency benchmark or equal-budget comparison. The small recall differences do not establish that one estimator is superior. The report is `docs/research/results/profile-holdout-resource-audit-f6d9d69.json` (source revision `f6d9d6918b2a718dc0a24b6506c01fc0c73188a9`, package-source SHA256 `d8a9bacc54fbed78865bdb2203e1d2349ca6d2068a6e5a349b9ee1816fbf3c59`, runner SHA256 `ef8d8332580464c2b73b449cbcee9dea46c19da0f3f88afa920779d972113269`); its prediction archive SHA256 is `5f83a9391a74a28a434d6dadbdca555e652aa12c768dd249c289a32957b241fd`.

At commit `f6d9d69`, the CPU suite passes with 209 passed, 13 skipped, and 78.37% coverage; the pinned Sionna/CUDA suite passes with 258 passed and 88.35% coverage. Ruff lint/format passes across 120 maintained source, script and test files; mypy passes over all 64 configured source files. The pasted 175-test/73.19%-coverage failure does not reproduce at this checkout and corresponds to an earlier source snapshot. The audit advances reporting and reproducibility only: the analytical profile proxy is still not geometric LOS, physical-scene transfer evidence, an equal-compute baseline comparison, or measured propagation data. Capacity-matched sweeps, controlled resource benchmarking, independent scene generalization, primary-method/off-ablation experiments, and the remaining preregistered research gates are open.


**Edition 3.29 noise-calibrated NR PSS/SSS acquisition operating-characteristic study (controlled receiver simulation) — 2026-10-21:** `acquire_nr_cell` now excludes PSS timing candidates too late for the capture to contain the required SSS at SS/PBCH symbol index 2. This prevents a late noise/template peak from becoming an invalid acquisition whose SSS lies beyond the capture. A regression test synthesizes that late PSS case and checks it is ignored; an additional test rejects captures shorter than the PSS-plus-SSS timing span. The full seeded study uses an integer timing search over 0–32 samples, a configured 256-point FFT/32-sample CP, 7.68-MHz sampling, CFO hypotheses {-3, 0, +3} kHz, AWGN, random physical cell IDs, and either a flat channel or a static two-tap channel with direct gain 1 and an echo `0.9+0.1j` delayed by 12 samples inside the CP. It independently calibrates PSS and SSS score thresholds on 1,024 noise-only captures at the 97.5th percentile for each marginal score (5% joint target with 2.5% allocated per marginal), then measures joint false accepts on 2,048 fresh noise captures. There were 3 false accepts (0.146%; Wilson 95% interval [0.050%, 0.430%]), below the 5% target at the measured upper interval bound for this capture/noise/search configuration.

Each of eight flat/multipath and 0/6/12/18-dB operating points has 256 seeded independent trials with full per-trial inputs, scores, cell, timing and CFO outcomes stored in `docs/research/results/nr-sync-operating-characteristics-seed20261022.json` (SHA256 `40d5e2a24c2302aad442c3fee4928ec64b757618b193f023139d5280f005db48`; source revision `802e05bce0bc7feb13567055c159dfcedcfa8a01`; package-source SHA256 `6f4fd58a7a89576830a791d311ead9c6f3c036af5832c643a25224422d4915bb`; runner SHA256 `e610c40cade39fd8c84184c9a049031e42c5419ee46911ac4e5f876341d4c8bf`). The flat-channel fixture produced 256/256 correct cell/timing/CFO acquisitions at every tested SNR. The 12-sample echo fixture produced 240/256 correct acquisitions at 0 dB (Wilson 95% interval 90.1–96.1%); all 256 crossed both calibrated score gates, with 13 timing errors and 3 CFO errors and no wrong-cell results. At 6/12/18 dB the echo fixture produced 256/256 correct acquisitions. This shows conditional behavior of this synthetic receiver/search setup only: the 0-dB in-CP echo exposes timing/CFO errors even though PSS/SSS thresholds pass. The four-symbol acquisition fixture has no PBCH, on-grid CFO and integer timing only; white noise omits interference, competing cells and RF impairments; one tap geometry does not represent diverse fading or measured NR channels. SNR is normalized to the captured padded four-symbol burst, including silent symbols. The binomial intervals cover these randomized trials, not devices, scenes, or false alarms per unit time. Fractional/off-grid CFO, temporal channel evolution, false alarms under structured interference, broader numerologies, PBCH, and field thresholds remain unvalidated.

At commit `802e05b`, the full CPU suite passes with 212 passed, 13 skipped, and 78.81% coverage; the pinned Sionna/CUDA suite passes with 261 passed and 88.51% coverage. Ruff lint/format passes across 123 maintained source, script and test files; mypy passes over 65 source files; `uv lock --check` and `uv build` pass. GitHub Actions run 39611331568 succeeds for the code commit. This advances controlled receiver operating-characteristic evidence, not complete synchronizer qualification or thesis completion; broader receiver/PHY validation, physical scenes, a registered method and ablations, independent measured evidence and the other open plan gates remain active.

**Edition 3.30 fractional timing and off-grid CFO operating-characteristic study (controlled receiver simulation; not qualification) — 2026-10-21:** The PSS/SSS detector was exercised with five fractional timing hypotheses (-0.375, -0.1875, 0, 0.1875, 0.375 samples), a coarse CFO grid at -3/0/+3 kHz, and true CFO selected at -3.5 or +3.5 kHz. Each flat and static two-tap channel/SNR point used 256 seeded trials; two-tap echo delay/gain were 12 samples and 0.9+0.1j. Timing correctness requires absolute error at most 0.1 sample and CFO correctness at most 250 Hz. Every signal point crossed both calibrated score gates, but combined correct cell/timing/CFO acquisition was only 52/256, 114/256, 205/256, and 254/256 in the flat channel at 0/6/12/18 dB, and 38/256, 69/256, 127/256, and 156/256 in the echo channel. Wrong-cell counts were zero; timing errors remained especially frequent for the two-tap channel (119, 91, 102, and 99 of 256 across those SNRs). Thus the prior noise false-alarm result does not establish acquisition accuracy: this receiver configuration is not qualified for fractional timing/off-grid CFO, and a high score can accompany an inaccurate estimate.

On 2,048 independent noise-only captures the calibrated gates accepted 3 (0.146%; Wilson 95% interval 0.050%-0.430%), below the configured 5% target by the measured upper bound for this finite capture/search setup. This does not cover false alarms per unit time, structured interference, other numerologies, channel evolution, or field thresholds. The seeded controlled-simulation artifact is `docs/research/results/nr-sync-operating-characteristics-fractional-seed20261022.json`, SHA256 `28ece24982a325f2988732e6bfd42d0375905aa46da107a5a4ff467306177733`; source revision `85dda8f10fb8bca66979523a4b4f3fefcada67e2`, package-source SHA256 `4cab9659bb5dabed5c9fdf4847d0b6a1964dcb79fdd41d6fd8f3fe37e535b7f6`, and runner SHA256 `e610c40cade39fd8c84184c9a049031e42c5419ee46911ac4e5f876341d4c8bf`. The full per-trial inputs and outcomes are retained for audit.

At source commit `85dda8f`, the exact CPU command `uv run --no-sync --locked python -m pytest` passes with 217 passed, 13 skipped, and 78.86% coverage; the pasted 175-test/73.19% report is from an earlier source snapshot and does not reproduce. The 75% coverage threshold remains unchanged. The pinned Sionna/CUDA suite passes with 266 passed and 88.48% coverage, Ruff lint/format and mypy pass, and GitHub Actions run 39613188358 succeeds. Remaining work includes diagnosing and improving multipath timing/CFO accuracy, expanding receiver/channel conditions and independent scenes, implementing and evaluating the primary method with matched baselines and ablations, and closing the external-data and publication gates. This study is a negative receiver result and does not complete the research plan.

**Edition 3.31 nuisance-aware local delay identifiability (analytical diagnostic; not an estimator result) — 2026-10-21:** The two-path CFR Fisher diagnostic now optionally includes an unknown common receiver timing offset and reports a covariance bound for relative-delay contrasts only when each contrast lies in the locally estimable subspace. In the new 160-case sweep (256/1,024 tones at 30 kHz; SNR 0/10/20/30 dB; separation 0 to 2 reciprocal-bandwidth intervals; relative path-gain phase 0/90 degrees), the seven-parameter model with common timing had no full absolute-delay covariance in any case. Relative-delay bounds were also withheld when the contrast was locally unidentifiable. At 20 dB and 0.01/B, the quadrature-phase relative-delay standard-deviation bound was 67.94 ns for 256 tones and 8.52 ns for 1,024 tones; at 1/B, bounds were 0.57–0.66 ns and 0.072–0.083 ns respectively. These tone-count comparisons change both observation count and total occupied span. At exactly coincident, co-phased paths the relative-delay contrast was locally unidentifiable; a finite local bound for the quadrature-phase coincident case does not resolve the global path-permutation/coalescence ambiguity. The bounds are local analytic diagnostics under known frequencies, gain magnitude/phase operating points and circular complex Gaussian noise; they do not show an estimator attains the bound, establish a global resolution threshold, or validate physical-path reconstruction.

The full sweep is `docs/research/results/identifiability-common-timing-phase-sweep-160.json`, SHA256 `b07d27aaa432e92d3a6dc6b13bc1d4ad206a0bede5036aa3b7aaa9a36bae2a1e`. It records source revision `3520279b127352593d9aa87f44c544fb3e419c3c`, package-source SHA256 `5f9364125610a910c5c1c45689e4d4f52af3d72ae2b52e768eaa9c54b5e839c5`, and runner SHA256 `d854c40ccf4a15c79aecab900439d4d8a435340ddf14af77e1494f9d2cfedec1`. The full CPU suite passes with 219 passed, 13 skipped, and 78.98% coverage; Ruff lint and format pass, and mypy passes over 65 source files. The main M2 gate remains open: validate estimable contrasts against an estimator/Monte Carlo study, define observation nuisance and path-label semantics, then compare the resulting identifiability-aware uncertainty method against matched classical and neural baselines on independent RT scenes. The analytical result alone establishes no research-method advantage.

**Edition 3.32 local maximum-likelihood check of relative-delay Fisher bounds (oracle-initialized controlled simulation) — 2026-10-21:** A seeded eight-condition study fitted two complex path gains and two path delays to complex CFR observations with iid circular AWGN and a randomized common receiver-time offset. It uses 256/1,024 centered tones at 30 kHz, 30-dB mean observed CFR SNR, separations 0.1/B and 1/B, and relative gain phases 0/90 degrees, with 128 trials per condition. All 1,024 fits converged and none reversed path order. At 1/B, empirical standard deviations of absolute relative-delay error were 0.97–1.09 times the nuisance-aware local CRLB across both tone counts and phases, with absolute empirical biases at most 0.020 ns. This is consistent with the local bound in these well-separated, high-SNR, correctly specified cases.

At 0.1/B, results depended strongly on gain phase. Quadrature-phase cases had empirical dispersion 0.89–0.96 times the local bound and small bias. Co-phased cases had dispersion 2.75–2.82 times the local bound and bias of 62.30 ns (256 tones) and 6.53 ns (1,024 tones), exceeding the respective true separations despite truth-centered initial values and optimizer convergence. This demonstrates that a local CRLB can be a poor description in a highly ambiguous regime; optimizer convergence alone is not evidence of a correct path estimate. The optimizer receives truth-centered gains and delays with small seeded perturbations, so this is a local-bound check, not a deployable estimator or receiver benchmark. It does not address global search, path-count selection, path permutations, channel-estimation error, scene transfer, or field performance.

The full per-trial error artifact is `docs/research/results/identifiability-relative-delay-local-mle-seed20261023.json`, SHA256 `24463e905233034cd27a5272623f89743d75b92df4bd94ec658b2756c2055574`. Source revision is `c395a13306dee9916435cc4a94db666da6f4af6b`; package-source SHA256 `4badf83bed073d68c5afcd09f9ee95d0d484f590a8d62d0a5875d52babe09416`; experiment module SHA256 `7a655e101bff2fe5c795b9cbe4c4aef189123ca8224f99037079e3c104f78955`; runner SHA256 `96cb249b6c12ebe77b62d8133714679a4f9e8666badc435eaf08d9ec615dc1b2`. A cold rerun reproduced the result byte-for-byte. Full CPU suite: 223 passed, 13 skipped, 79.40% coverage; pinned Sionna/CUDA suite: 272 passed, 88.70% coverage. Ruff lint/format, mypy over 66 source files, and package build pass. Next M2 work is a non-oracle initialized/global estimator comparison and path-label/uncertainty evaluation; the local analytical bound is not sufficient evidence for the proposed method.

**Edition 3.33 oracle-free OMP initialization versus local maximum-likelihood refinement (paired two-path analytical simulation) — 2026-10-21:** A fixed-path-count-two complex OMP delay-grid initializer was compared with the same nonlinear least-squares refinement started from OMP estimates and with the truth-centered local fit, on paired CFR/noise trials. The sweep used 256/1,024 tones at 30 kHz, 30-dB mean CFR SNR, separations 0.1/B and 1/B, two relative gain phases (0/90 degrees), and 128 trials per condition. The OMP search spans 0–2/B at 1/(32B) spacing. At 1/B, refined RMSE was within 0.001 ns of the corresponding truth-centered local RMSE in all cases, and both were close to the nuisance-aware CRLB (0.03–0.26 ns). At 0.1/B with quadrature-phase gains, refined RMSE was 2.84 ns (256 tones) and 0.33 ns (1,024 tones), close to the local bound (2.62 ns and 0.33 ns). For co-phased gains, refined RMSE was 93.47 ns and 11.85 ns, versus truth-centered local RMSE 85.53 ns and 10.37 ns; the local CRLBs were much smaller at 16.97 ns and 2.13 ns. OMP-refined optimization succeeded in 126/128 and 127/128 trials for those co-phased close-path cases, with 1 and 4 path-order reversals respectively. OMP grid estimates before refinement were poor for all 0.1/B conditions. Thus the coarse initializer plus the same local optimizer does not resolve the close, co-phased regime; path-gain phase and local identifiability materially affect the outcome. These are conditional synthetic two-path results, not scene or receiver performance.

The comparison gives OMP the path count (two) and a bounded, oversampled delay dictionary; it does not test model-order selection, diffuse energy, unknown numerology, channel-estimation error, or global search outside 0–2/B. The truth-centered local fit remains an oracle-start diagnostic, not a deployable baseline. Full paired per-trial errors are archived in `docs/research/results/identifiability-global-omp-local-mle-seed20261024.json`, SHA256 `a82a68ecb6dc0abb17103694e9699d1fe9c4a38d443ae13adac74122a26021f8`; source revision `bd5fd7d067966627ff72b01567d5cc24f21c0c6d`, package-source SHA256 `9963e9b969dd123af62a9803790999b483e404933c3ef66fc886be86e1ad4707`, experiment module SHA256 `0246862aeb20a4041dcfff34d76863ac1cb4cc19becbf9ccb6f56a6f7149ece3`, runner SHA256 `4dee50bcb61d7b127ff828300464c3cfdb1bf57451fac7d37bc1ba8413ea4d52`. A cold rerun reproduced it byte-for-byte. Current CPU suite: 225 passed, 13 skipped, 79.72% coverage; pinned Sionna/CUDA suite: 274 passed, 88.75% coverage. Ruff lint/format, mypy over 66 source files, and package build pass. The M2 method gate remains open; this baseline does not establish a learned-method advantage or independent-scene transfer.

**Edition 3.34 condition-calibrated profile-likelihood sets for two-path relative delay (analytical diagnostic; not scene or receiver performance) — 2026-10-21:** Added a grid profile likelihood that profiles unknown complex path gains and common timing over unordered two-path delays, then forms central normalized-profile-mass intervals and condition-specific split-conformal profile-likelihood-ratio sets. The sweep uses 256/1,024 centered tones at 30 kHz, 20/30-dB mean CFR SNR, separations 0.1/B and 1/B, relative gain phases 0/90 degrees, a 1/(128B) delay grid over 0–2/B, and 512 calibration plus 512 independent evaluation trials per condition (16,384 trials total). Noise variance and path count are known to this diagnostic; common timing is sampled uniformly from 0.1–0.3/B. Each held-out interval/set and each calibration nonconformity score is retained.

Across the 16 held-out conditions, central intervals formed from normalized profile-likelihood mass covered the true relative delay in 66.02–100% of trials for the nominal 95% level. Condition-specific split-conformal thresholds produced 94.14–100% empirical coverage on the union of selected delay-grid support cells; the lowest observed condition was 256 tones, 20 dB, 0.1/B separation, co-phased paths (482/512), with a Wilson 95% interval of 91.76–95.87%. The more ambiguous 0.1/B co-phased cases produced broad median 95% calibrated set spans: up to 226.72 ns (256 tones, 20 dB), 105.19 ns (256 tones, 30 dB), 52.31 ns (1,024 tones, 20 dB), and 11.20 ns (1,024 tones, 30 dB). Well-separated cases often yielded grid-cell-limited or sub-nanosecond spans. The ordinary profile-mass intervals therefore do not give reliable nominal coverage in the ambiguous regime; calibration trades that overconfidence for large sets where the measurement is weakly identifying.

This is a condition-wise calibration diagnostic, not a transferable uncertainty method: calibration and evaluation use the same known synthetic SNR, tone count, gain phase, and delay separation. The 95% split-conformal guarantee is marginal under exchangeability within that fixed simulator condition, not conditional for every finite calibration draw and not evidence of calibration on unseen scenes or hardware. Coverage for the conformal set uses finite grid cells centered at delay hypotheses, so its finest resolution is 1/(128B); it is not continuous-delay calibration. The delay grid, known model order, known noise variance, static specular model, and central-interval summary limit interpretation; disconnected likelihood regions are summarized by a span. No neural baseline, scene variation, channel-estimation error, or receiver impairment is included.

The full calibration/evaluation records are `docs/research/results/identifiability-profile-likelihood-seed20261025.json`, SHA256 `013ed373ad8dc5cd10ab8a83db8411a4c7cef9ae8392ba410c0b4cb2e4b51e61`; source revision `5ec93cf59ab3c10d5f483edbcd3ea3212b048450`, package-source SHA256 `bc993361b0bcd57bcb3546919712d82a6b75d783ad1358c66626ba69f2966038`, experiment module SHA256 `49658c4439daec2573e5ef8f617291be17f575a9c92d367f7721a5855eee278b`, and runner SHA256 `2ec9a8a3aacd648441f579207f6e551f33fda4f9b95e76f6de4160f595bd72c1`. A cold rerun reproduced the JSON byte-for-byte. Full CPU suite: 231 passed, 13 skipped, 80.47% coverage; pinned Sionna/CUDA suite: 280 passed, 88.91% coverage. Ruff lint/format across 130 files, mypy across 67 source files, lock check, and package build pass. The M2 method gate remains open: the next needed evidence is calibration across varying conditions and independent RT scenes, followed by matched classical/neural uncertainty baselines and ablations. This synthetic profile-likelihood diagnostic establishes neither a novel learned method nor external-scene transfer.

**Edition 3.35 factor-level holdout stress test of profile-likelihood uncertainty transfer (analytical two-path simulation; not scene transfer) — 2026-10-21:** The prior same-condition calibration experiment is extended with four held-out policies: exclude the exact SNR/separation/gain-phase cell, exclude the target SNR level, exclude the target separation level, and exclude the target relative gain-phase level. Every policy keeps tone count fixed and has a matched calibration control that uses the target condition's own calibration observations. The 16 target cells span 256/1,024 centered tones at 30 kHz, 20/30 dB, separation 0.1/1 reciprocal-bandwidth intervals, and 0/90-degree relative path-gain phase. Each cell contributes 512 calibration and 512 independent evaluation trials; each held-out policy pools calibration scores from the prescribed other cells at the same tone count. The experiment uses the 1/(128B) delay grid over 0–2/B, known two-path count and noise variance, and randomized common timing from 0.1–0.3/B.

Empirical 95% support-cell coverage across target cells ranged from 87.89–100% for exact-cell holdout, 89.26–100% for SNR-level holdout, 88.67–100% for separation-level holdout, and 88.09–100% for gain-phase-level holdout. Same-condition calibration ranged from 92.77–100%. For the most ambiguous co-phased 0.1/B case at 256 tones and 20 dB, the exact-cell, SNR-level, separation-level, gain-phase-level, and same-condition coverages were 87.89%, 89.26%, 88.67%, 91.80%, and 92.77%, respectively; their median set spans were 222.63–225.69 ns. This shows that the score calibration is sensitive to condition shift even within the same analytical channel family, while broad uncertainty sets remain necessary in the close-path case. These empirical rates do not establish a universal coverage guarantee.

The full per-calibration-score and held-out evaluation records are `docs/research/results/identifiability-cross-condition-calibration-seed20261026.json`, SHA256 `a68196aa1ea0f3daa013111e804ec4ccf1fa83cf268a8c9375ffb2b2b00990c1`; source revision `9150396ad075953796679769f98652174f5c41f7`, package-source SHA256 `7d3255992a96e517c814ad57d022e5006b01e56c8b8d4c31bfd5d728cadb3cdb`, experiment module SHA256 `d81e0f0dfff313d4d7580f55a760609414f57d03e7da2141fe56f774fb8a99cf`, and runner SHA256 `03b4f04b13e6f4f95c5a2817167929c6a9e1aa51f779cbb95bd7e6705fe23299`. A cold rerun reproduced the artifact byte-for-byte. Full CPU suite: 233 passed, 13 skipped, 80.70% coverage; pinned Sionna/CUDA suite: 282 passed, 88.86% coverage. Ruff lint/format across 133 files, mypy across 68 source files, lock check, and package build pass. The score distributions are not exchangeable under these holdouts, so all cross-condition rates are descriptive stress-test results; the same-condition conformal guarantee cannot be transferred. Tone-count transfer, independent RT geometries, measured channels, channel-estimation error, and matched learned/classical uncertainty methods remain open. The M2 method gate is still not met.

**Edition 3.36 six-scene RT leave-one-scene-out stress test (synthetic simulation; not physical validation) — 2026-10-21:** Added a reproducible runner that generated 40 links per scene (20 geometric direct-path and 20 blocked-direct-path labels) for Sionna's `box_one_screen`, `simple_street_canyon`, and `etoile`, plus three procedurally generated urban geometries (`city41`–`city43`). The generated urban assets use simplified procedural buildings/vehicles and placeholder ITU materials. Sionna RT ran with the LLVM CPU backend, depth 2 and 200 samples/source. Each fold holds out one of the six scene domains; the remaining five provide training data. The observation is a generic noisy comb-pilot estimate (spacing 4, 12 dB), not 5G NR DMRS. RF, MLP, and logistic regression were evaluated on seven paired seeds with a common grouped split per seed and 200 bootstrap draws.

Across the six held-out-scene folds, macro mean test AUROC was 0.500 for MLP, 0.468 for RF, and 0.513 for logistic regression. Per-scene MLP means were 0.509, 0.493, 0.576, 0.453, 0.386, and 0.582 in the order box, Etoile, street canyon, urban 00041, urban 00042, urban 00043. The direction varied by scene, and logistic regression had the strongest macro mean. These descriptive results do not establish model superiority: scenes are few and heterogeneous, each has only 40 links, the urban assets are synthetic, test partitions are small, and the fold training sets overlap. Paired-seed intervals and tests are conditional on each fixed scene and are not independent-scene inference. The labels are `GEOMETRIC_DIRECT_PATH_ORACLE`, not measured LOS/NLOS or a standards-defined channel-state label.

The rerun command is `SIONNA_PYTHON=/home/nguyenl37/.venvs/sionna220-gpu/bin/python bash scripts/run_rt_six_scene_holdout.sh outputs/rt-six-scene-holdout-20261021`; it refuses to overwrite an existing output directory. The six input NPZ files, six fold manifests, raw fold results, and summaries remain in the ignored local output directory. Their hashes, runner hash, per-fold metrics, and caveats are indexed in `docs/research/results/rt-six-scene-holdout-20261021.json`. Source revision for generation/evaluation is `25f9d4692d643b649ca5e1073a7e3cb1b98063f5`; dataset generation used Sionna 2.2.0 and LLVM CPU RT. This advances the cross-scene stress-test gate but does not close P5 or M2. Next evidence requires independent geometries with qualified/calibrated materials, visual path/label audits, matched NR receiver observations, larger test samples, and uncertainty-method baselines/ablations; measured or otherwise externally validated channels remain necessary for physical claims.
**Edition 3.37 high-budget Etoile solver-convergence diagnostic (negative result; single fixed link) — 2026-10-21:** Extended the existing deterministic LLVM sweep for Sionna's `etoile` at 3.5 GHz, 256 tones, 30 kHz spacing, fixed transmitter/receiver coordinates, max depth 3, and seeds 11/23/47 to 0.5M/1M/2M and then 2M/5M/10M samples per source. The highest-budget result is only a comparison reference, not ground truth. Repeating the same seed-11, 2M configuration in two fresh processes reproduced all four depth-specific CFR SHA256 values and path counts exactly, confirming repeatability for the fixed seed/configuration.

Across ray budgets, path counts at max depth 3 remained seed-sensitive: at 2M samples they were 9/8/8 for seeds 11/23/47; at 10M they were 8/8/9. Relative complex-CFR L2 error from 2M to each seed's 10M reference was 2.77%, 0.087%, and 2.76%; at 5M it was 2.78%, 0.081%, and 2.76%. Thus increasing the budget did not monotonically stabilize path count or CFR for two seeds. At 10M, depth 0/1/2 remained materially different from depth 3 (relative CFR errors about 1.81/0.29/0.36), so the interaction-depth choice also changes this link substantially. These are solver-sensitivity observations for one scene/link and one simulator version; no acceptance tolerance was preregistered, and even 10M is not a validated reference.

The compact report is `docs/research/results/rt-etoile-convergence-highbudget-20261021.json`; full per-run outputs are local and ignored under `outputs/rt-convergence-etoile-highbudget-{500k,2m,10m}.json`. The three original 100k multiseed reports for box, street canyon, and Etoile are also indexed there. The existing six-scene holdout runner used only 200 samples/source and depth 2; consequently its classifier metrics remain low-budget exploratory output distributions, not convergence-qualified channel evidence. Its geometric visibility labels are independent evidence about direct-path geometry, but its channel features and model scores must not be interpreted as robust to RT sampling budget. Do not use those metrics to pass P5 or support physical-transfer claims. Before future scene-model evaluation, predeclare tolerances, compare multiple budgets/depths/seeds per geometry family, and exclude or separately qualify settings that fail. Full scene/material qualification, path overlays, antenna/precision factors, and measured channels remain open.
**Edition 3.38 depth-matched six-scene RT resampling and held-out evaluation (simulation only; not phase completion) — 2026-10-21:** Parameterized `scripts/run_rt_six_scene_holdout.sh` with `SAMPLES_PER_SRC` (positive integer; default remains 200) and regenerated the same six-scene protocol at 100,000 samples/source, depth 2, 40 links per scene with 20 labels/class. Each fold again uses five training scene domains, one held-out domain, generic spacing-4 comb-pilot observations at 12 dB, RF/MLP/logistic regression and seven paired seeds. Group counts ranged from 4 to 20 across the scene files. The CPU/LLVM run produced all six datasets and all six completed folds; source data, manifests, and raw predictions remain in ignored local outputs.

Macro mean test AUROC was 0.586 MLP, 0.571 RF, and 0.555 logistic regression. Per-scene MLP-minus-RF AUROC was +0.116 on box (seed-bootstrap 95% interval +0.058 to +0.177), +0.021 canyon (-0.032 to +0.082), -0.020 Etoile (-0.074 to +0.026), +0.004 urban 00041 (-0.014 to +0.020), -0.064 urban 00042 (-0.116 to -0.011), and +0.031 urban 00043 (-0.005 to +0.067). These intervals and sign-flip tests summarize seed variation within a fold; six overlapping-training folds are not six independent scene replications. The macro-score differences from the 200-sample run (+0.086 MLP, +0.104 RF, +0.041 logistic regression) are descriptive, not paired causal effects: changing sample budget can change solver acceptance and therefore links and groups. Results remain simulation-only, with simplified urban assets, placeholder materials, geometric visibility labels and non-DMRS observations; they establish no model advantage.

Depth-matched max-depth-2 checks on one fixed receiver link per scene, using three seeds and 200/2k/20k/100k/1M rays/source, found the 200-sample setting under-resolved for canyon (5–6 paths versus 8 at 1M; up to 2.85% relative CFR L2 error), Etoile (3 versus 4 paths; about 16.17%), and urban 00041 (5 versus 6; about 11.26%). The selected links were much closer by 100k: maximum relative CFR L2 error against the 1M within-seed reference ranged from 0 to 0.084% across all six scenes, with matching path counts at 100k and 1M. This is encouraging for those six links but does not qualify all generated receiver positions; no numerical convergence threshold was preregistered, and 1M is not truth. This depth-matched result also narrows the edition 3.37 finding: its instability was at depth 3 in Etoile, whereas the selected depth-2 link was stable from 100k to 1M. The depth-3 uncertainty remains unresolved.

The resampled six-scene metrics and hashes are indexed in `docs/research/results/rt-six-scene-holdout-100k-20261021.json`; the fixed-link depth-2 sweep is `docs/research/results/rt-six-scene-depth2-budget-check-20261021.json`. Reproduce the six-scene run with `SAMPLES_PER_SRC=100000 SIONNA_PYTHON=/home/nguyenl37/.venvs/sionna220-gpu/bin/python bash scripts/run_rt_six_scene_holdout.sh outputs/rt-six-scene-holdout-100k-20261021`. P5 remains open: stratified multi-link budget checks, preregistered tolerances, path-overlay review, materials qualification, antenna/precision sweeps, and independent/measurement-backed evaluation are still required.

**Edition 3.39 all-link RT ray-budget audit and adaptive six-scene rerun (simulation only; not phase completion) — 2026-10-21:** Added `scripts/audit_rt_dataset_ray_budget.py` to replay the exact 240 stored receiver positions and solver seeds from the six 100k datasets at 200/20k/100k/1M samples per source with scene, depth 2, and grid held fixed. Compared with the largest budget, valid-path counts changed on 150/240 links at 200, 30/240 at 20k, and 14/240 at 100k. Thirteen of the 14 changes at 100k are in Etoile. Across all links the 100k maximum relative complex-CFR L2 error was 0.8844; the largest per-scene 95th percentile was 0.2667. Even without a path-count change, the 95th-percentile CFR error reached 0.2667 in Etoile. The six built-in scenes vary substantially: box, street canyon, and urban 00041/00042/00043 had no path-count changes at 100k, while Etoile had 13/40; some 100k maximum CFR errors were 0.0974 in canyon and 0.00569 in urban 00042. Thus the earlier single-link depth-2 check and its apparent 100k stability do not qualify the generated 100k dataset. At 200 samples, path counts changed on 150/240 links and the maximum CFR error was 4.8701. The 1M comparison is still a finite solver output, not truth; no convergence acceptance tolerance was preregistered.

Extended the same fixed-link replay on every held-out Etoile link to 1M/2M/5M/10M samples/source. Relative to 10M, 1M changed path counts on 4/40 links and had maximum CFR error 0.5379; 2M changed 1/40 and had maximum error 0.04475; 5M changed none and had maximum error 0.00274. Ten million remains only a comparison reference. Canyon and urban 00042 each had no path-count changes at 1M/2M/5M against their 10M reference, but CFR differences were nonmonotonic; urban 00042's maximum was 0.00569 at 1M and 5M and 0.000523 at 2M. These selected-scene audits provide no formal stopping rule and do not validate material or path-label realism.

To avoid carrying the plainly under-resolved per-scene inputs into another cross-scene stress test, the adaptive rerun used depth 2 and exploratory budgets of box 20k, canyon 1M, Etoile 5M, urban 00041 20k, urban 00042 100k, and urban 00043 20k samples/source. It regenerated 40 links per scene with 20 labels/class, then evaluated the six grouped leave-one-scene-out folds using the same seven seeds, generic spacing-4 comb-pilot observation at 12 dB, and RF/MLP/logistic-regression methods. The per-scene mean-AUROC values were MLP 0.560/0.493/0.616/0.700/0.515/0.609, RF 0.428/0.459/0.612/0.704/0.578/0.577, and logistic regression 0.572/0.515/0.595/0.650/0.506/0.563, in box/Etoile/canyon/urban 00041/00042/00043 order. The descriptive macro means across these six folds were 0.582 MLP, 0.560 RF, and 0.567 logistic regression. Performance varies by scene; the six scenes are not independent field replications, seven seeds are not additional scenes, folds have overlapping training sets, and simulation-only scores do not establish a model advantage. Because the per-scene budgets were selected from exploratory diagnostics without a preregistered acceptance threshold, this is a stress test, not a convergence-qualified model comparison. Inputs and fold artifacts remain in ignored `outputs/rt-six-scene-holdout-adaptive-20261021/`.

The compact, hash-indexed evidence is `docs/research/results/rt-six-scene-adaptive-budget-20261021.json` (SHA256 `67738f303ab6aa4b4c65966447152579c9d7f1f30ab39ee9807fd88cfc740ca3`); raw per-link audit JSON remains in ignored `outputs/`. The audit runner SHA256 is `f1d268ab650786362ce0d0490d489d7ca586d9504b2820a1f7e43c21186c4e3f`; adaptive runner SHA256 is `eef28a35c51c5846c21e4fee442e1068045d4873ec53662f4d404545d91974e3`. Sionna RT used the LLVM CPU backend because OptiX CUDA RT is unavailable in WSL; the Sionna PHY suite did use the GPU. Neither adaptive run nor the convergence audit used measured NR DMRS. P5 and M2 remain open pending preregistered solver tolerances, qualified materials and independently checked geometry/path labels, NR-receiver observations, antenna/precision sweeps, independent scenes and/or measurement-backed evaluation.

**Edition 3.39 software verification:** Added adapter-contract tests for tuple and ndarray RT CFR conversions, malformed/nonfinite responses, tensor-like path validity, and invalid diagnostic inputs. The full CPU suite passes (247 passed, 13 skipped, 80.93% coverage); the full pinned Sionna/GPU suite passes (296 passed, 89.08% coverage). Ruff lint and format pass across 135 maintained Python files, mypy passes across 68 source files, and `uv lock --check` plus `uv build` pass. The pasted 175-pass run at 73.19% was a failing run below the configured 75% gate; it is superseded by these later complete passing runs. GitHub Actions run 39629642092 passed all jobs for the code and artifact commit `bb34c2e9f059c6c91f1a60f04749b6756cccd664`.

**Edition 3.40 stratified RT path-overlay QA (visual simulator audit; not physical validation) — 2026-10-21:** Added `scripts/render_rt_path_qa.py` to select two low-to-high stored path-count quantiles per geometric label from each adaptive six-scene dataset. It reruns the exact selected link and its independent depth-zero solver oracle at the stored scene, link seed, depth, and per-scene ray budget, then renders top-view and elevation projections with Mitsuba object bounding boxes and the ordered Sionna interaction vertices. Across 24 selected links (4/scene), all 24 stored geometric labels matched deterministic depth-zero replay, and the presence/absence of a direct candidate in each full solve matched that oracle. Several NLOS samples retain indirect paths, including valid one-path examples; LOS examples show their direct component together with indirect paths. Six four-link contact sheets were visually inspected for endpoint/axis consistency and broad alignment between the plotted rays and scene-object bounds. These plots support human QA of the simulation outputs; they do not prove that geometry/materials are physically realistic or that individual paths are correct under surveyed ground truth.

The hash-indexed report is `docs/research/results/rt-six-scene-path-qa-adaptive-20261021.json` (SHA256 `3ebd24b626b31c23f432513ce273b589a01cfac8c8ab84a4c3ae40db3becc5e4`); its local ignored 24-image gallery and contact sheets are under `outputs/rt-path-qa-adaptive-20261021/`. The renderer script SHA256 is `6215e721f2f9e3a0f12435f326cfb03a3fb640c3b2c6ab108d0742ed8f4eb274`. The scene projections use axis-aligned object bounding boxes; overlapping boxes and large scene extents can hide topology, so independent RF/EM scene review, qualified materials, and surveyed path truth remain outstanding. Sionna RT ran on LLVM CPU because the WSL OptiX backend is unavailable. This visual sample is not a full 240-link audit, convergence tolerance, or evidence that P5/M2 passes; measured receiver observations and the independent geometry-family gates also remain open.

**Edition 3.40 software verification:** Added deterministic path extraction and stratified selection tests (10 new tests). The complete locked CPU suite passes (257 passed, 13 skipped, 80.93% coverage); the complete pinned Sionna/GPU suite passes (306 passed, 89.08% coverage). The pasted 175-pass CPU run at 73.19% coverage is an older failing run and is superseded by the passing run above. The 24-link overlay replay and six-scene contact-sheet review completed as described; plan, compact report, and renderer are hash-indexed. Ruff lint and format pass across 136 maintained Python files, mypy passes across 68 source files, uv lock --check and uv build pass, and git diff --check is clean. GitHub Actions run 39631545845 passed all four jobs on commit c0bb7a4a833cc009cd7f7e6f9afd2b52a458791c.

**Edition 3.41 continuous fractional timing refinement for blind PSS/SSS acquisition (controlled receiver simulation; not qualification) — 2026-10-21:** Replaced the fractional timing estimate's coarse-grid-only output with bounded scalar maximization of normalized PSS correlation over (-0.5, 0.5) samples after coarse N_ID^(2), integer timing, and CFO acquisition. Six deterministic offsets spanning both interval edges pass at 0.005-sample tolerance. A paired rerun reused the exact 1,024 calibration-noise captures, 2,048 independent validation-noise captures, and 2,048 signal trials from the edition 3.30 seed/protocol. The calibrated joint-noise gates still accepted 3/2,048 validation captures (0.146%; same finite-sample outcome as before).

Across the four flat-channel 0/6/12/18-dB points, correct cell/timing/CFO acquisitions changed from 52/114/205/254 of 256 to 59/119/206/254; continuous timing correctness changed from 229/247/254/256 to 255/256/256/256. For the 12-sample 0.9+0.1j echo, correct acquisitions changed from 38/69/127/156 to 54/86/190/250, and timing correctness from 137/165/154/157 to 178/209/230/251. No paired acquisition regressed on the flat points; echo-point improvements were 19/3, 27/10, 72/9, and 97/3 improved/regressed trials. Exact paired McNemar p-values with Holm adjustment across all eight signal points were 0.00513, 0.0382, 1.72e-12, and 2.10e-24 for the echo channel at 0/6/12/18 dB. Paired bootstrap 95% intervals for acquisition-rate change were [0.0273, 0.0977], [0.0234, 0.1133], [0.1836, 0.3086], and [0.3047, 0.4297]. The flat-channel gains did not pass the eight-point Holm-adjusted significance threshold. This is evidence for this seeded synthetic setup only; substantial low-SNR and residual multipath errors remain, and the result does not qualify field synchronization.

Full paired per-trial results are in `docs/research/results/nr-sync-operating-characteristics-continuous-seed20261027.json` (SHA256 `6ac0ee34a97795d6dda3b6e799ad87802d5f742463475753344b0664b7d94f64`) and the paired analysis in `docs/research/results/nr-sync-continuous-timing-paired-comparison-20261027.json` (SHA256 `e5328063e85f86b37723b4c723471ccbf37856fe37f9cefd9c6972868b12ac68`). Source revision is `8ade5e15cbc38780cee5269bd89cfcee5cb3e6b7`; package-source SHA256 `a0dd72d31e687ef03dfb1a8248aa7bdcd8927798390523de8405fb2fd53bf8d3`; receiver SHA256 `1f521ff4849649b2022121d4a67f8445b2df87dceb34f9416aca9dadbd37a64d`; comparison implementation SHA256 `22417f3b7754a4be8a78281e1ee9e0bf1cf703738dfcb616f5a8571900fbc98b`; paired-analysis script SHA256 `9d86eee4e912d995cc52f2a1fcded0885a0a41ace9dd6998754bfce8bb24ec7d`; and operating-characteristic runner SHA256 `e610c40cade39fd8c84184c9a049031e42c5419ee46911ac4e5f876341d4c8bf`. The estimator still uses a finite CFO hypothesis set and refines only the best coarse cell/timing/CFO candidate; the channel set is flat or one static two-tap echo, and the signal/noise convention is the padded synthetic burst. Structured interference, competing cells, other numerologies, mobility, PBCH, receiver RF impairments, deployment false-alarm rates, and measured captures remain untested.

The complete locked CPU suite passes (264 passed, 13 skipped, 81.04% coverage); the pinned Sionna/GPU suite passes (313 passed, 89.00% coverage). Ruff lint and format pass across 137 maintained Python files, mypy passes across 68 source files, and `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run 39634808862 passed all four jobs for code and evidence commit `833c7c0bf56d090ab725dc53a559b0b2fc145966`. This advances a bounded P2 receiver test and does not complete P2, P5, M2, or the research program.

**Edition 3.42 paired M2 uncertainty-set comparison (analytical two-path simulation; not method or scene validation) — 2026-10-21:** Added an auditable comparator for the same 8,192 held-out trials in the 16-cell profile-likelihood sweep. It evaluates nominal central 95% profile intervals and same-condition split-conformal 95% likelihood-ratio sets on identical per-trial observations, reporting paired coverage and delay-span changes with exact McNemar tests and Holm adjustment across the 16 condition cells. Pooled empirical coverage was 91.22% for the central intervals and 96.35% for the conformal sets. The most ambiguous co-phased 0.1/B conditions were 75.0% versus 94.1% (256 tones, 20 dB), 71.1% versus 96.7% (256 tones, 30 dB), 77.0% versus 96.3% (1,024 tones, 20 dB), and 66.0% versus 94.7% (1,024 tones, 30 dB); these paired coverage differences remained significant after family-wise adjustment. The conformal set's median hull span was not uniformly larger, and its hull may include disconnected low-likelihood gaps. These results support calibration of this finite-grid score under same-cell synthetic exchangeability; they do not show a Bayesian posterior, an independent estimator advantage, coverage under condition shift, or real-channel calibration.

The paired report is `docs/research/results/identifiability-interval-paired-comparison-seed20261028.json` (SHA256 `efd66be2bc20ac1ee5598536f476c9c293d73327357313d180f7dd3ebd81b010`), using input `docs/research/results/identifiability-profile-likelihood-seed20261025.json` (SHA256 `013ed373ad8dc5cd10ab8a83db8411a4c7cef9ae8392ba410c0b4cb2e4b51e61`). Comparator source revision is `a92a3a0f644549d344a8c94889f629aa651a3011`; comparator module SHA256 `75f05951c7f11de99bc4b699c5d32ae5ccbc286128383a6098ce1c4925347e8d`, runner SHA256 `71ae71c2842a145ed103d2a17934f06129f8eab682434bedc3b044a47c69a3e6`, and package-source SHA256 `5612a2c078abf0ea7aaff08a5fbe899ea269906d9ad0960a12df0aa462a7237d`. The upstream profile-likelihood observations were generated at revision `5ec93cf59ab3c10d5f483edbcd3ea3212b048450`; calibration and evaluation were separate seeded draws for each cell. This is a paired analysis of existing evidence, not new channel simulation. The next M2 gate remains a deployable estimator and uncertainty baseline on independent RT scenes with realistic observation nuisance; measured data, path-count uncertainty and model mismatch remain open.

**Edition 3.42 software verification:** The complete locked CPU suite passes (271 passed, 13 skipped, 81.15% coverage); the pinned Sionna/GPU suite passes (320 passed, 88.97% coverage). Ruff lint and format pass across 140 maintained Python files, mypy passes across 69 source files, and `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run 39635941838 passed all four jobs for code and evidence commit `7a1579d179fd7212ebc7343ca6b4fa1f8b92a7dc`.

**Edition 3.43 matched profile-set versus classical OMP uncertainty baseline (analytical two-path simulation; M2 remains open) — 2026-10-21:** Added paired nonparametric bootstrap 95% intervals to the same-condition profile-likelihood and two-path complex OMP comparison. Across 16 cells (256/1,024 tones, 20/30 dB SNR, 0.1/1.0 reciprocal-bandwidth path separation, 0/90-degree gain phase difference), each method used 256 independent per-cell calibration trials and was evaluated on the same 256 held-out observations: 4,096 evaluation trials total. Profile-set coverage was 96.53% and OMP split-conformal interval coverage was 94.97% (paired difference +1.56 percentage points). Exact paired McNemar p-values were Holm-adjusted over the 16 cells; only the 30-dB, 1/B, co-phased cells at both tone counts passed adjustment (p=0.000977 each), with within-cell paired-bootstrap coverage-difference intervals [3.13, 8.98] percentage points. The per-cell confidence intervals are not multiplicity-adjusted. Aggregate median profile hull span was 3.064 ns versus 2.970 ns for the OMP interval, so width is not uniformly improved; close co-phased low-separation cells often had much wider OMP intervals (e.g. 383.1 ns vs 228.8 ns at 256 tones/20 dB). OMP local refinement reported optimizer success on 4,031/4,096 evaluations and path-order reversal on 19; these are failure indicators for the finite-dictionary initialization/local-refinement baseline, not evidence that it is representative of all classical estimators.

The compact full-trial artifact is `docs/research/results/identifiability-matched-profile-vs-omp-seed20261029.json`, SHA256 `538480ad9ba27f533610947b12128d6456ac99004ad4d00d688288d3158e5a07`. It uses seed 20261029, 5,000 paired bootstrap draws per cell, source revision `04a50a6e40970f477051b425bddb341ca76922aa`, package-source SHA256 `138d966a450de4ae40bf6d3cc2a674711fc7cfc32c7f04cc25d9077186f738fa`, experiment-module SHA256 `e61addc4cdd52ceb0ad385ccdffd5bd722c830ef2271c84565289f25ccc2616e`, and runner SHA256 `5b2a7ebb602ffeb86750570064d8e3dc6a8031417e97406202bd4160381078cc`. This is a known-two-path, known-noise-variance analytical CFR experiment with exact-condition calibration. It does not establish cross-condition calibration, model-order robustness, diffuse/mismatched channels, independent RT-scene performance, a learned-method advantage, or measured-channel validity. M2 still requires deployable uncertainty baselines on independent, observation-realistic RT scenes and preferably measured 5G NR channel data.

**Edition 3.43 software verification:** The locked CPU suite passes (277 passed, 13 skipped, 81.59% coverage); the pinned Sionna/GPU suite passes (326 passed, 89.16% coverage). Ruff lint and formatting pass over 124 maintained files; mypy passes across 70 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run `39638087098` passed all four jobs for evidence/plan commit `2e75afffb9cbd0ec4d95c80ac9f1e580d18e8d4d`.

**Edition 3.44 matched cross-condition calibration-transfer audit (paired analytical stress test; M2 remains open) — 2026-10-21:** Reanalyzed the edition 3.43 paired trials using five calibration policies: leave-one-condition-out, hold out SNR, hold out path separation, hold out gain-phase level, and within-condition control. Each target's calibration excludes its evaluation condition; pooling stays within tone count, and OMP errors are normalized by the common reciprocal-bandwidth scale. Across 4,096 paired evaluation rows per policy, profile/OMP coverage was 95.36%/91.09% for leave-one-condition-out, 95.00%/92.60% for SNR holdout, 95.34%/56.40% for separation holdout, 95.56%/77.95% for gain-phase holdout, and 96.53%/94.97% for the within-condition control. OMP coverage ranged from 0% to 100% under separation holdout and 3.52% to 100% under phase holdout; profile coverage ranged from 87.89% to 100% in each shifted policy. One severe separation-transfer cell (256 tones, 20 dB, 0.1/B, co-phased gains) had 87.89% profile-set coverage versus 0% OMP interval coverage; its paired evaluation bootstrap 95% interval for the difference was [83.59, 91.80] percentage points. Across 80 paired policy-condition tests, 36 remained significant after Holm adjustment. The bootstrap intervals condition on observed calibration scores and do not capture calibration-sample uncertainty. These pooled scores show sensitivity of this finite OMP baseline to the tested synthetic condition shifts; they do not establish universal profile-method superiority or transfer to RT scenes or measured channels.

The compact analysis is `docs/research/results/identifiability-matched-transfer-seed20261030.json`, SHA256 `21c739a89cbd6b93dd9627126825713fba6dce549a03b6ee09ffb1796fb9bb57`; its paired input is `docs/research/results/identifiability-matched-profile-vs-omp-seed20261029.json` (SHA256 `538480ad9ba27f533610947b12128d6456ac99004ad4d00d688288d3158e5a07`). Analysis source revision `c4986c723b5bf6c0d527e81115de38fe0af0530d`, package-source SHA256 `4312f1c6c41e355ce8f2c7111929067fcb030d2ab40f29adf5dda2063e7bae0b`, analysis-module SHA256 `9fa95602247aecd9ac17bd7a14dbb393a5695d30834a791e8067f1a198cef34b`, and runner SHA256 `9430b822a697484b6bef9887f01c6e6fb025246291a4f4fc8e4e12d6b76c8e39`. No new channel draws were generated for this reanalysis; all limitations of the known-two-path, known-noise analytical input still apply.

**Edition 3.44 software verification:** The complete locked CPU suite passes (282 passed, 13 skipped, 81.87% coverage); the pinned Sionna/GPU suite passes (331 passed, 89.28% coverage). Ruff lint/format pass over 126 maintained files; mypy passes across 71 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run `39639082063` passed all four jobs for transfer-audit commit `d5f9ac8c6bfeee7f44b09f397a61433c01fde40a`.

**Edition 3.45 Sionna NR PUSCH/DMRS observations on six held-out RT scenes (simulation-only receiver-path study; external validation remains open) — 2026-10-21:** Added a reproducible 6-PRB, 30-kHz, 14-symbol PUSCH observation path using Sionna 2.2.0's PUSCH DMRS LS channel estimator on CUDA. It converts six existing RT CFR scene datasets to 240 per-link observations (40 per scene), interpolates the centered 128-tone/30-kHz CFR to the 72-tone PUSCH grid, normalizes mean channel power per link, and adds seeded 12-dB AWGN. The estimator uses ideal synchronization and nearest-neighbor interpolation. Its estimated CFR agrees with the interpolated RT CFR at median normalized complex correlation 0.9704 and median complex NMSE 0.0608; this checks the implementation under this setup, not receiver performance under synchronization error or field conditions.

Six leave-one-scene-out evaluations used RF, MLP, and logistic regression with seven seeds each. Mean test AUROC across fixed folds was 0.5411 (RF), 0.5174 (MLP), and 0.5201 (logistic). The same train/test split indices were verified against the earlier generic noisy-comb-pilot baseline for every method and seed. Across those folds, DMRS-minus-comb macro AUROC was -0.0185 (RF), -0.0647 (MLP), and -0.0468 (logistic); this is a descriptive comparison of distinct observation-generation paths on reused splits, not a controlled claim of receiver or model superiority. Scene-level performance varies materially, so pooled means do not establish generalization. Each six-fold evaluation shares overlapping training scenes; the seven seeds and six scenes do not turn this simulated corpus into independent field trials.

Evidence report `docs/research/results/rt-six-scene-dmrs-pusch-holdout-12db.json` has SHA256 `fcb1cd1f7b9b32ed64fa9a52923309cef8ca186f013a5f9f91692e3bfd66394c`; source revision `2da07f381126b7c17f323989cf6fadee4d1a63f4`, summarizer SHA256 `ed396390f93d7028a934afb9d847a539c722d7ea7d0ab1bd6365d168c55bd570`, package-source SHA256 `413ea92e420432779a7832b166af75ff0df8e260727d0eba359996f02f97670d`, and observation dataset SHA256 `82f28060b81372d8cb1006766e4e6634c78167af73a440a50926b65711a82a2e`. The compact report records source-artifact hashes, fold manifests/results, seed/method split-hash comparisons and receiver-agreement checks. Recreate it with `scripts/summarize_rt_dmrs_scene_holdout.py` from the ignored generated dataset/fold outputs.

The scenes are simplified/built-in Sionna RT scenes with geometric depth-zero direct-path oracle labels, not independently qualified materials or measured LOS/NLOS annotations. Per-link normalization removes path-loss evidence. Timing/CFO/CP error, mobility, interference, field data, and external scene/material qualification remain absent; M2, independent-scene validation and thesis-level evidence gates remain open. This phase adds an NR DMRS observation path, not a completed channel-condition classifier qualification.

**Edition 3.45 software verification:** `uv run --no-sync --locked python -m pytest` passes (288 passed, 14 skipped, 80.93% coverage; 75% gate unchanged). The pinned Sionna/CUDA suite passes (338 passed, 89.18% coverage). Ruff lint/format pass over 130 maintained source, script and test files; mypy passes over 72 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run 39641851734 passed all four jobs for evidence commit fdcd25b9c4360bc479312dad7b62aee699d0caca.

**Edition 3.46 matched sample-domain PUSCH curves across standardized TDL-A–E (single link configuration; simulation only) — 2026-10-21:** Extended the Sionna 2.2 waveform-domain PUSCH receiver study across TDL-A, B, C, D and E at 3.5 GHz, 100-ns delay spread, 3 m/s, 12 PRB, 30-kHz SCS, DMRS additional position 0, and 512 codewords per SNR at 0/6/12/18/24 dB (12,800 codewords total). Every profile used the same transmitted payload hashes, batch seeds and standardized AWGN streams; the profile-specific channel-truth hashes differed, and each profile reused its own channel realizations across SNR. All profiles had 13-sample discrete channel memory within the configured minimum 13-sample CP. Smoke checks for TDL-B–E passed before the full sweep.

Observed BLER error counts at 0/6/12/18/24 dB were: TDL-A 512/512/9/0/0; TDL-B 512/512/14/1/1; TDL-C 512/512/11/0/0; TDL-D 512/512/0/0/0; TDL-E 512/512/0/0/0. Thus all profiles saturated at 100% BLER at 0 and 6 dB in this coding/link configuration. At 12 dB the observed rates ranged from 0/512 for D/E to 14/512 for B; at 18 and 24 dB B had 1/512 errors and the other profiles had none. The report includes Wilson intervals and paired per-codeword comparisons against TDL-A. Exact McNemar tests were Holm-adjusted across the complete 20-comparison family (four profiles by five SNR points); no contrast passed 0.05, and the smallest adjusted p-value was 0.078125 (TDL-D/E at 12 dB). These are finite seeded simulation outcomes, not evidence that the profiles are equivalent or a universal ranking.

Compact hash-indexed report `docs/research/results/tdl-pusch-time-profile-sweep-512-seed63000.json`, SHA256 `5ae2da8cb5f3d15a112be41aee943642dfa0f2ad9416bc958fc9f47dbfa5133e`; source revision `7bad3a814391d921c9aa69e0758bf5f59a2c0aed`, package-source SHA256 `c5ecc59482c0c00ba1fbe8f6883776017bfb3954965814770374b430a779ddcc`, and aggregator SHA256 `59f56145d6fb8d562cfa468dba14398f46fde3911dbeb1658ca9494e0b9f3e2c`. It records hashes for all five raw curves, configuration, link outcomes and paired contrasts; raw per-codeword masks remain under ignored `outputs/tdl-pusch-profile-sweep-512-seed63000-20261021/`. Reproduce with `scripts/run_tdl_pusch_time_bler_curve.py` for each profile using seed 63000 and the saved configuration, then pass the five JSON files to `scripts/summarize_tdl_pusch_profile_sweep.py`.

The link uses Sionna's standardized stochastic TDL channels and its time-domain PUSCH receiver with ideal timing/frequency synchronization. It tests one numerology, bandwidth, DMRS pattern, speed, delay spread and coding configuration; the 100-ns setting is CP-contained here. It does not validate standard conformance, independent RT scene behavior, synchronization tolerance, material/geometry fidelity, or measured-channel performance. TDL profile labels are stochastic channel-model identities, not geometric LOS/NLOS ground truth. P2 and P4 validation, P5 scene qualification, M2, and external measurement remain open.

**Edition 3.46 software verification:** The locked CPU suite passes (290 passed, 14 skipped, 80.92% coverage); the pinned Sionna/CUDA suite passes (340 passed, 89.04% coverage). Ruff lint/format pass over 131 maintained source, script and test files; mypy passes over 72 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run 39643499913 passed all four jobs for profile-comparison implementation commit `7bad3a814391d921c9aa69e0758bf5f59a2c0aed`.

**Edition 3.47 paired 1-vs-2 receive-antenna diversity on the identical sampled TDL-A waveform (simulation only) — 2026-10-21:** Added a waveform-domain Sionna PUSCH/DMRS comparison that generates each two-antenna TDL-A realization once, then decodes the exact same noisy waveform with antenna 0 alone and with both antennas. This avoids the invalid comparison of separately generated one- and two-antenna channels, whose first-antenna channel draws differ even under equal random seeds. Across five SNR points, each condition used 1,024 codewords (5,120 paired codewords total), TDL-A, 100-ns delay spread, 3.5 GHz, 3 m/s, 12-PRB/30-kHz PUSCH, DMRS additional position 0, seed 65000, and a 13-sample channel memory within the 13-sample minimum CP. The per-codeword payload is shared; the one-RX baseline is exactly receive antenna index 0 from the same two-RX channel and noise draw. Sionna TDL was configured without spatial correlation matrices, so the two receive branches are uncorrelated; measured pre-noise branch powers averaged 1.00076 and 0.98716. No array calibration is applied.

At 6/8/10/12/14 dB, one-RX BLER error counts were 971/668/390/221/104 of 1,024; two-RX counts were 306/1/0/0/0. Paired two-RX-minus-one-RX BLER differences were -64.94, -65.14, -38.09, -21.58, and -10.16 percentage points, with paired codeword-bootstrap 95% intervals [-67.87,-61.91], [-68.07,-62.21], [-41.11,-35.16], [-24.12,-19.14], and [-12.11,-8.30] pp. Exact paired McNemar tests Holm-adjusted over these five SNR points all remained below 5% (adjusted p-values 4.11e-194, 1.63e-200, 2.38e-117, 1.19e-66, and 9.86e-32). This supports a receive-diversity effect for this one ideal-synchronization, uncorrelated Sionna TDL-A link setup; it does not validate correlated/measured arrays, other TDL/CDL profiles, numerologies, array calibration, field channels, or general receiver performance.

Full paired masks, per-batch payload/channel hashes, branch powers, Wilson intervals and paired bootstrap results are in `docs/research/results/tdl-pusch-rx-diversity-tdla-1024-seed65000.json` (SHA256 `3c70e1d90738fcef94ce2117c8adb1290b78c649d63589e08e38856f92733a00`; 192,583 bytes). It was generated at source revision `31ebd63cf125a17806aafbc30f497e4f207f77eb`, package-source SHA256 `3779eb9d440d339649601a522150edb5f9109fc3b3ef9722ae1181fde1ea4ff7`, by `scripts/run_tdl_pusch_rx_diversity_curve.py` (SHA256 `b470b7f508c477a953689f8c533bc2eb7d7860a4441b4235c3384955464ed1ec`). Reproduce with seed 65000, TDL-A, 100 ns, 3.5 GHz, 3 m/s, 12 PRB, 30 kHz, DMRS additional position 0, 1,024 frames, batch size 64, and SNR 6/8/10/12/14 dB.

**Edition 3.47 software verification:** The locked CPU suite passes (295 passed, 14 skipped, 79.55% coverage); the pinned Sionna/CUDA suite passes (347 passed, 89.02% coverage). Ruff lint/format pass over 132 maintained source, script and test files; mypy passes across 72 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run 39646983385 passed all four jobs for diversity implementation commit `31ebd63cf125a17806aafbc30f497e4f207f77eb`.

**Edition 3.48 paired receive-correlation sensitivity on the same TDL-A waveforms (synthetic simulation; 2026-10-21):** Added a correlation sensitivity condition to the paired 1-vs-2-RX waveform study. For each batch, Sionna first generates one uncorrelated two-branch TDL-A channel and received waveform; a real Cholesky transform then leaves branch 1 unchanged and constructs branch 2 as `0.9*branch1 + sqrt(1-0.9^2)*independent_branch2`, for both channel truth and the corresponding noiseless receive samples. The resulting receive covariance has unit diagonal and off-diagonal coefficient ρ=0.9. Both conditions use the same 1,024 payloads, latent channel draw, first-antenna channel, and seeded AWGN streams at each of 6/8/10/12/14 dB; the single-RX error masks are byte-for-byte-equivalent outcomes. The paired comparator rejects results whose payload, latent channel, first-antenna channel, noise seeds, or single-RX masks do not match.

The 1-RX baseline produced 971/668/390/221/104 block errors out of 1,024. The 2-RX uncorrelated control produced 306/1/0/0/0 errors; the 2-RX ρ=0.9 treatment produced 558/279/105/24/2. Paired treatment-minus-control BLER differences were +24.61/+27.15/+10.25/+2.34/+0.20 percentage points, with paired codeword-bootstrap 95% intervals [20.61,28.61]/[24.41,29.88]/[8.40,12.11]/[1.46,3.32]/[0.00,0.49] pp. Exact paired McNemar p-values with Holm adjustment over the five SNR points were 1.46e-30/1.45e-81/1.97e-31/2.38e-7/0.5; the 14-dB difference does not pass the adjusted 5% threshold. These observations quantify sensitivity to one controlled high-correlation transform in this simulation, not a general receive-array law or field performance.

The raw ρ=0.9 curve is `docs/research/results/tdl-pusch-rx-diversity-tdla-rho090-paired-1024-seed65000.json` (194,328 bytes, SHA256 `2ad7ebee904cdb2a26ea44be42baf127150ce95f820815effb3b18044e377acd`). The paired comparison is `docs/research/results/tdl-pusch-rx-correlation-tdla-rho090-paired-1024-seed65000.json` (7,276 bytes, SHA256 `a40605f0e23c3df2791a756c51c9597ad78d9b8687b14e5c6b74098576eb760b`). The treatment was regenerated from clean source revision `6baa9cbac105d83465e622a927a83ccc076f0223`, package-source SHA256 `c6e8492dc1dbab6796b7d6509b461346cda92e0ad95901f0f33126842fcd3dc4`. Reproduce the raw treatment with `scripts/run_tdl_pusch_rx_diversity_curve.py` using seed 65000, 1,024 frames, batch size 64, SNR 6/8/10/12/14 dB, TDL-A, 100 ns, 3.5 GHz, 3 m/s, 12 PRB, 30 kHz, DMRS additional position 0, and `--rx-correlation 0.9`; produce the comparison with `scripts/summarize_tdl_pusch_rx_correlation.py` and the preceding ρ=0 report.

**Edition 3.48 software verification:** The locked CPU suite passes (296 passed, 14 skipped, 79.39% coverage); the pinned Sionna/CUDA suite passes (349 passed, 88.92% coverage). Ruff check/format pass across the configured 155 maintained Python files, mypy passes across 72 source files, and `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run 39649554446 passed all four jobs for implementation commit `6baa9cbac105d83465e622a927a83ccc076f0223`.

**Edition 3.49 time-ordered multi-target NR trajectory schema and immutable storage — 2026-10-21:** Added `NRTrajectory`, which binds one-observation NR records to a trajectory ID and strictly increasing finite timestamps. It enforces unique sample IDs, one observation per step, and a consistent receiver configuration; target names are the union across steps, so labels may be missing at individual times. `target_series()` returns aligned values and masks with absent targets represented explicitly as `None`, while model-input sequences remain free of identifiers, targets, and provenance. `NRChannelRecord` now validates that each supplied target mask is boolean and broadcastable to its target. `write_nr_trajectory()` persists each record through the existing content-addressed pickle-free NR store, then publishes a hash-addressed ordered index and immutable trajectory-ID manifest; `read_nr_trajectory()` verifies the index hash and sample ordering before reconstructing the trajectory. Tests cover multi-target and missing-target steps, partially masked CFR values, timestamp/ID invariants, mask dtype/shape errors, ordered storage round-trip, idempotent writes, and payload tampering.

The schema makes trajectory grouping explicit for downstream group splits, but no new RT trajectories, temporal-overlap audit results, or independent-scene split qualification are claimed here. Those remain open until the larger diverse-scene dataset is generated and audited.

**Edition 3.49 software verification:** The locked CPU suite passes (302 passed, 14 skipped, 78.97% coverage); the pinned Sionna/CUDA suite passes (355 passed, 88.82% coverage), including the native tensor-mask record path. Ruff lint/format pass across the configured 155 maintained Python files; mypy passes across 72 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. CI will be recorded after the implementation and plan are pushed.

**Edition 3.50 Sionna PUSCH CUDA runtime and allocator profile — 2026-10-21:** Added `scripts/profile_tdl_pusch_gpu_memory.py` and measured synchronized phase runtime and PyTorch peak allocated/reserved bytes for sample-domain TDL-A PUSCH with 2 RX branches on an NVIDIA RTX PRO 500 Blackwell Laptop GPU (compute capability 12.0, 6,113 MiB reported device memory), Python 3.12.3, PyTorch 2.11.0+cu128, and Sionna 2.2.0. Configuration: 12 PRB, 30-kHz spacing, DMRS additional position 0, 100-ns TDL-A, 3.5 GHz, 3 m/s, 10-dB noise variance, ideal synchronization, batch sizes 1/8/16/32/64/128 codewords. Per phase the runner synchronizes CUDA, clears unused allocator cache, resets peak counters, times one execution, and records baseline/peak allocated and reserved bytes. Phases include waveform/channel generation, AWGN construction and addition, and one-/two-RX decoding.

The waveform/channel phase peaked at 53.84/324.98/635.68/1,257.07/2,500.63/4,987.74 MiB allocated for batch sizes 1/8/16/32/64/128; corresponding allocator-reserved peaks were 66/380/724/1,408/2,788/5,548 MiB. At batch 64 the two-RX decoder peaked at 291.43 MiB allocated and 360 MiB reserved; at batch 128 it peaked at 562.11/666 MiB. For this 6-GiB device, batch 64 leaves substantially more headroom than batch 128 and matches the existing research-run default. These are process-local allocator measurements; they exclude driver/display allocations and are not total-device memory guarantees. Runtime values are one pass per size with a cold first batch and warm-up effects, so they do not support throughput comparisons.

The hash-addressed profile is `docs/research/results/tdl-pusch-cuda-memory-profile-rtxpro500-20261021.json` (18,218 bytes, SHA256 `a3acae4e0b107676cb25a7d6a20e1b6c920f744160ac815f4035738e1085f2b8`). Source revision `ee70880f23e9ed0351aae5076842634e3614a09f`; package-source SHA256 `75a61651999fef47ae6cf7cc559401a45076b0310daa3469dde01171c41a8cb1`; profiler SHA256 `f5495392eeecbcde3ef17eb8fb95304ea6e08e9df22bd08cbff99f241ab05a41`. Reproduce with `scripts/profile_tdl_pusch_gpu_memory.py` in the pinned Sionna/CUDA environment and the recorded profile, numerology, seed 73000, and batch sizes.

**Edition 3.50 software verification:** The locked CPU suite passes (302 passed, 14 skipped, 78.97% coverage); the pinned Sionna/CUDA suite passes (355 passed, 88.82% coverage). Ruff lint/format pass across the configured 156 maintained Python files; mypy passes across 72 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions run 39651644238 passed all four jobs for profiler commit `ee70880f23e9ed0351aae5076842634e3614a09f`.

**Edition 3.51 paired 1-vs-2-RX and receive-correlation study at 24-PRB/15-kHz (simulation only; 2026-10-21):** Extended the sample-domain TDL-A PUSCH diversity sweep to a second numerology. The CP-contained configuration uses 24 PRB at 15 kHz, 4.32-MHz sample rate, DMRS additional position 0, 100-ns delay spread, 3.5 GHz, 3 m/s, 1,024 codewords per point, SNR 6/8/10/12/14 dB, batch size 64, and seed 74000. Sionna reports 13 samples of channel memory and a 23-sample minimum CP. Each correlation condition uses the same payloads, latent channel draws, RX0 channel samples and noise seeds; the ρ=0.9 branch is formed with the documented post-sampling Cholesky transform. The paired audit confirms identical one-RX outcome masks across ρ conditions.

At 6/8/10/12/14 dB, the 1-RX and uncorrelated 2-RX error counts were 969/695/394/208/112 and 302/1/0/0/0 of 1,024. The ρ=0.9 two-RX errors were 563/302/112/35/3. Paired ρ=0.9-minus-uncorrelated 2-RX BLER differences were +25.49/+29.39/+10.94/+3.42/+0.29 percentage points, with paired codeword-bootstrap 95% intervals [21.58,29.39]/[26.56,32.23]/[9.08,12.89]/[2.34,4.59]/[0.00,0.68] pp. Exact paired McNemar tests Holm-adjusted across the five points were 1.92e-34/1.87e-88/1.16e-33/1.16e-10/0.25; the 14-dB result is inconclusive after adjustment. These are conditional simulation observations for this TDL-A configuration and synthetic receive covariance, not standardized performance, array calibration, or field-channel evidence.

The uncorrelated raw curve is `docs/research/results/tdl-pusch-rx-diversity-tdla-24prb-15khz-1024-seed74000.json` (194,743 bytes, SHA256 `3e5c1641cb0e92d7a35000b10f49ee08049ae3849dfff45191a3d677ccb611cd`). The ρ=0.9 curve is `docs/research/results/tdl-pusch-rx-diversity-tdla-24prb-15khz-rho090-1024-seed74000.json` (194,241 bytes, SHA256 `37245e4f726e94714a10a8a5349837c6252a723dd1ec96a28dccd81fabf08268`). The paired comparison is `docs/research/results/tdl-pusch-rx-correlation-tdla-24prb-15khz-rho090-1024-seed74000.json` (7,326 bytes, SHA256 `3d816fbe69de71a443ce79b297ecc1c769aea29f7f10bbe2aa68f982b7788ef7`). Data source revision `94487fc8422f4b9fdc282fb819f5c38149bdc3bf`, package-source SHA256 `75a61651999fef47ae6cf7cc559401a45076b0310daa3469dde01171c41a8cb1`; curve-runner SHA256 `de21ec91fd5bbdcf7712732346e7f3f9d45fc9e41076a40504fcc1e04d8e4de1`, comparison-runner SHA256 `249908d7ebd2f377c860c3c7e4b32c31217622ecfd8fbd35cb985387ba62a515`. Reproduce both curves with the configuration above and `--rx-correlation 0` / `0.9`, then run `scripts/summarize_tdl_pusch_rx_correlation.py` on the two outputs.

**Edition 3.51 software verification:** Existing local suites pass (CPU 302 passed, 14 skipped, 78.97% coverage; pinned CUDA 355 passed, 88.82%); Ruff, mypy, lock/build, and CI run 39651988836 pass for profiler/evidence commit `94487fc8422f4b9fdc282fb819f5c38149bdc3bf`. The full 24-PRB/15-kHz CUDA sweep and hash-checked paired analysis were rerun locally with the pinned Sionna environment.

**Edition 3.52 timestamped RT-to-NR trajectory pilot — 2026-10-21:** Added `integrations/sionna/rt_trajectory.py` and generation/audit runners. Each route position is retraced with deterministic Sionna RT on Mitsuba LLVM CPU; a separate depth-zero solve supplies geometric direct-path truth, while the full path set is mapped with absolute RT delays onto the PUSCH frequency grid. One common seeded payload/noise batch is passed through the PUSCH DMRS estimator. The observation contains only the received grid, DMRS estimate, estimator variance and pilot mask; full CFR and geometric LOS are stored as masked targets, with the scene, solver, seeds, channel counts, scene asset hashes and runtime provenance in the record.

The committed configurations generate four samples at 0/0.1/0.2/0.3 seconds for each of six scenes: Sionna `box_one_screen`, `etoile`, and `simple_street_canyon`, plus the three deterministic procedural urban assets `urban_00041`–`urban_00043`. All use 3.5 GHz, 30-kHz PUSCH, 6-PRB BWP, 12-dB configured batch-mean AWGN SNR, maximum path depth 2, and 200 samples/source. The trajectory store is content-addressed and intentionally remains in ignored `outputs/`; six index hashes and all receiver/target record hashes are retained there. The immutable split groups the three procedural urban scenes together as one asset family. Its four independent groups map to train (Etoile and procedural urban family), validation (box scene) and test (street canyon), giving 16/4/4 samples. Thus validation and test each contain only one scene group and this is strictly a storage/QA pilot, not a meaningful model evaluation.

Audit report `docs/research/results/rt-nr-trajectories-pilot-six-scenes-761ff9f.json` is 5,229 bytes, SHA256 `21af130fa2849ce768251d1e21d4df6b44d49b8a808c64849c280b6193d598e0`; it records zero positive-duration temporal overlaps across splits and zero cross-split receiver-visible CFR near-duplicates at similarity threshold 0.9999. The frozen split lock `docs/research/locks/rt-nr-trajectory-scene-family-v1.json` has SHA256 `0e19c7a635aa85bb76dbaead6f37a961eb0ccc65acb423ee7981a735a70afc3d`. Data were generated at commit `761ff9f2a26209fa9863b6223d8dbe1e10c0ff20` with package-source SHA256 `8ef22a06fe9c4b6684a60e045a1ed1e38efe38d729c2846d2bf32124df948ff6`; generation runner SHA256 `5f8a94abe3390ca1b1a21e79d4573522513d401927edfc58bc876e0de92a67af`, audit runner SHA256 `032880da66f664102a292785d57bdbe4a49c72dd440817a5fceaddb97e337b10`, and RT trajectory adapter SHA256 `e83f736377b25d7c2d00dcf8b27f2d7a8bfe6f35cf933a530300c0ddbedfee1f`. Reproduce each store shard with `scripts/generate_rt_nr_trajectory.py config/datasets/rt_trajectories_pilot/<scene>.json <store>`, then run `scripts/audit_rt_nr_trajectories.py` with all six specs and the committed split lock.

Limitations: the scene assets are synthetic or built-in and not field-calibrated; trajectories are short constant-velocity paths made from separately retraced quasi-static snapshots, not a model of within-slot Doppler or path birth/death. The 0.5-ms audit windows denote one 30-kHz NR slot and do not establish forecast horizons. Outage samples are rejected until a zero-power receiver policy is declared. CUDA RT remains unavailable in WSL. Larger scene families, qualified materials/arrays, controlled blockage and outage behavior, long-duration dynamics, and independent-scene model evaluation remain open.

**Edition 3.52 software verification:** The locked CPU suite passes (311 passed, 15 optional skips, 78.06% coverage); the pinned Sionna/CUDA suite passes (365 passed). The native LLVM RT→NR trajectory round-trip passes on CPU. Ruff check/format pass across 161 maintained source, script and test files; mypy passes on 73 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions [run 39655751430](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39655751430) passed all four jobs for implementation commit `761ff9f2a26209fa9863b6223d8dbe1e10c0ff20`.

**Edition 3.53 common-random RT trajectories and E16 baseline pilot — 2026-10-21:** Corrected the trajectory generator to reuse one deterministic PathSolver seed across all positions in a route. A native LLVM static-scene test verifies that a receiver held fixed produces the same full CFR at both timestamps; this avoids interpreting changing finite-ray sampling draws as physical movement. Regenerated all six four-snapshot trajectories from code commit `39018cefe44e9068f80bd7719e2aa8297d030552`. The updated content-addressed trajectory audit is `docs/research/results/rt-nr-trajectories-pilot-six-scenes-39018ce.json` (5,229 bytes, SHA256 `a51c3c62828edbfdff6c7d5b516036567f1852d6a01a2dc139993abef230cbdd`). It again finds zero cross-split temporal overlaps and zero receiver-visible CFR near-duplicates at threshold 0.9999; the scene-family lock remains byte-identical.

Added a prespecified E16 diagnostic with two fixed, receiver-only predictors: persistence reuses the last DMRS estimate, and complex linear extrapolation uses the two latest DMRS estimates with timestamp-scaled slope. Full absolute-delay RT CFR is used only as the offline target. The report `docs/research/results/rt-nr-forecast-pilot-six-scenes-39018ce.json` (18,211 bytes, SHA256 `dfa368a369f9f0d7c0e408f31caa1e44c9033ea29d04c633466d54befd735928`) records 30 transitions across six trajectories and preserves split labels. On the single held-out street-canyon route, there are three persistence transitions and two matched transitions for linear extrapolation. Mean transition NMSE was 4.54 dB for persistence and 9.85 dB for linear extrapolation; on the same two target times, linear-minus-persistence was +4.34 dB. The complex linear baseline therefore did not beat persistence in this tiny held-out pilot. With only one independent test scene and two paired target times, this is a pipeline check and descriptive negative result, not inferential evidence that linear prediction is generally inferior or that a learned dynamic model would help.

The forecast evaluator SHA256 is `b441d63d8f8db735f9aa184e00642f8688f4aae904dc28d2067991c4231952d7`; its experiment module SHA256 is `d1ca3a54534b64b0d82ddf01c98033c31f2d281fee1b52db6a38405c2a3d6db5`. Data use Sionna 2.2.0 / Sionna RT 2.2.0, Mitsuba LLVM CPU, 3.5 GHz, 30-kHz PUSCH, 6 PRB, and 12-dB configured batch-mean AWGN. All generated datasets remain in ignored `outputs/`; manifests and the compact audit/forecast reports are committed. The code and evaluation remain limited to synthetic/built-in scenes, ideal receiver timing, short quasi-static snapshots and no measured channels.

**Edition 3.53 software verification:** The locked CPU suite passes (316 passed, 15 optional skips, 78.24% coverage); the pinned Sionna/CUDA suite passes (371 passed). Ruff check/format pass across 164 maintained source, script and test files; mypy passes on 74 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions [run 39657163580](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39657163580) passed all four jobs for implementation commit `39018cefe44e9068f80bd7719e2aa8297d030552`.

**Edition 3.54 grouped uncertainty-evaluation infrastructure — 2026-10-21:** Added `evaluation/uncertainty.py` and `scripts/evaluate_regression_uncertainty.py` for frozen scalar-regression predictions. Calibration uses the independent group—not correlated rows—as the conformal exchangeability unit and scores each calibration group by its maximum standardized absolute residual. The report provides finite-sample group-simultaneous and sample coverage, interval width, test-group macro-RMSE bootstrap intervals, uncertainty-ranked selective risk, and optional coverage stratification by a predeclared identifiability category. If the requested finite-sample coverage cannot be calibrated from the available number of groups, the interval is reported as unbounded. The JSON input schema and assumptions are documented in `docs/research/UNCERTAINTY_EVALUATION.md`.

This closes an evaluation-tooling gap for E11/E19, not those experiment gates. The predictor and uncertainty scale must be frozen before calibration; calibration and test scene families must be disjoint and exchangeable for the stated group-level guarantee. The evaluator does not prove exchangeability, scene generalization, or useful uncertainty, and no experimental model output has yet been assessed with this new runner. E11 still requires the primary identifiable-state method, matched on/off ablations, classical/neural comparisons, and adequately powered unseen-scene evaluation. E19 still requires held-out calibration/robustness evidence.

**Edition 3.54 software verification:** The complete CPU suite passes (325 passed, 15 skipped, 78.44% coverage); the pinned Sionna/CUDA suite passes (380 passed). Ruff lint and format pass across the maintained packages/tests, mypy passes across 75 source files, `uv lock --check`, `uv build`, and `git diff --check` pass. The new calibration runner and numerical contracts have dedicated tests; these results validate implementation behavior only and do not constitute scientific evidence.

**Edition 3.55 analytical M2 decoder foundation — 2026-10-21:** Added `models/observable_multipath.py` with a differentiable two-path CFR renderer, masked complex-Gaussian pilot likelihood, heteroscedastic relative-delay likelihood, and an MLP baseline that returns ordered relative delay, complex gains, and a positive delay-uncertainty scale. The renderer fixes the common-delay gauge explicitly: under arbitrary complex path gains, absolute common arrival time is not estimated from this pilot-only model. Tests verify the closed-form response, finite gradients through the renderer/decoder, pilot-mask behavior, delay bounds, and invalid-input handling. Implementation details and limitations are in `docs/research/OBSERVABLE_MULTIPATH_MODEL.md`.

This is not the planned primary method: no training loop or analytical dataset runner is wired to it yet; path count is fixed at two; no diffuse residual, existence posterior, phase/array nuisance, or measured data are modeled. The Gaussian scale is a model output only; useful uncertainty claims require the group-disjoint calibration pipeline above. M2/E11 remains open pending the equivalence-class decision rule, matched classical/MLP and identifiability-on/off training, powered held-out conditions/scenes, and reproducible evaluation.

**Edition 3.55 software verification:** The complete CPU suite passes (329 passed, 16 skipped, 78.46% coverage); the pinned Sionna/CUDA suite passes (385 passed), including a forward/backward run of the new decoder on CUDA. Ruff lint/format pass on 139 maintained package/test files plus the new evaluator script; mypy passes on 76 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. These checks validate the component and runner contracts, not a trained model or scientific result.

**Edition 3.56 runtime-adaptive RT and matched analytical decoder/identifiability pilot — 2026-10-21:** Added `integrations/sionna/backend.py` to resolve `auto`, explicit CUDA, explicit LLVM, or an exact Mitsuba variant. Automatic selection does not equate an installed CUDA variant with a usable RT device: it runs a deterministic free-space Sionna RT direct-path trace in an isolated child process, preferring `cuda_ad_mono_polarized` only when the native trace returns a valid path and otherwise recording the failure before selecting `llvm_ad_mono_polarized`. `RayScene` and new RT trajectory configs default to `auto`; explicit historical configs retain their recorded backend for replay. `scripts/check_rt_backend.py` exposes the same preflight. On this WSL host, Mitsuba lists the CUDA variant and PyTorch training is available on CUDA, but Sionna RT CUDA fails with `Could not initialize OptiX`; the native LLVM trace passes and the automatic selector reports `llvm_cpu`. This resolves the environment-selection requirement without making OptiX a development-machine prerequisite; an OptiX-capable Linux host will be selected automatically after its native probe passes.

Completed the missing analytical training/evaluation path for the decoder. Added a grouped, seeded two-path CFR generator with whole-condition-group train/validation/calibration/test splits; the model now accepts per-sample frequency grids and carries occupied bandwidth into its features so delay remains in physical seconds across bandwidth conditions. Added validation-selected Torch training, pickle-free NPZ checkpoint save/load, a variable-projection profile-likelihood baseline with local-curvature scale, group-conformal reports, and paired bootstrap comparisons resampling independent test groups. The matched identifiability-on arm adds a local relative-delay CRLB scale floor after projecting out unknown complex gains; the off arm shares architecture, initialization, data, splits and training budget. The experiment runner publishes its outputs atomically, verifies no overwrite, records hashes/configuration, and the summary script integrity-checks every run artifact before producing a compact committed result.

The default-seed analytical pilot used 800 independent synthetic condition groups, four samples/group, and 64 tones. Group counts were 480/80/120/120 and sample counts were 1,920/320/480/480 for train/validation/calibration/test. It trained on CUDA with 33/38 best epochs for the identifiability-off/on arms. Group-macro delay RMSE was 40.67 ns (off), 40.36 ns (on), and 95.92 ns (known-order/noise profile baseline). The paired on-minus-off group-bootstrap delta was -0.31 ns (95% CI [-1.60, 1.08] ns), which includes zero; this run does not show a clear benefit from the CRLB regularizer. Nominal 95% group-conformal coverage on test groups was 96.7%/93.3%/96.7%, with mean interval widths 423/380/9,974 ns for off/on/profile. The very broad classical interval and the finite-test calibration values are specific to this synthetic pilot and do not establish comparative superiority, valid transfer, or useful field uncertainty.

The compact verified summary is `docs/research/results/observable-multipath-analytical-pilot-seed20261031-v2.json` (SHA256 `9abbd4b22cd68b10a20950ece24611b951460b7ea13090a2ab4dda9015c8791b`). It hashes the ignored full prediction/checkpoint/report run under `outputs/observable-multipath-analytical-pilot-seed20261031-v4/`; run-manifest SHA256 is `c562e9e80e9c7e8f936f92b69169b2e987cbf464f511ac6dd4901b34b9168500`. Source HEAD at execution was `2148e316aab27e1b0cc992e0cdd0b7626f5830f4`; the modified source tree is identified by package-source SHA256 `c7c5f79edf8700f1ffa277236255691644f348e1d4967d5e2002bb41d984451a` and runner-source SHA256 `cf22b9db78caca079ceb4796b37308afb9cf00696cdf176bbdaef443243229d4`. Runtime used Python 3.12.3, PyTorch 2.11.0+cu128, Sionna/Sionna RT 2.2.0. Recreate with `scripts/run_observable_multipath_experiment.py <new-output-dir>` and summarize with `scripts/summarize_observable_multipath_experiment.py <run-dir> <new-summary.json>`; both runners refuse to overwrite existing outputs. This pilot still assumes a fixed, known two-path model order and known receiver noise variance, uses a single synthetic condition mixture, and has no RT/NR estimator noise, path-existence uncertainty, diffuse residual, measured data, or independent-scene transfer. E11/M2 remains open pending a defensible equivalence-class decision rule, better-matched classical/neural baselines, robustness and on/off attribution across independent RT scene families, and external evidence.

**Edition 3.56 software verification:** The complete locked CPU suite passes (342 passed, 16 skipped, 78.93% coverage); the pinned Sionna/CUDA suite passes (398 passed), including the CUDA neural fit/CRLB-scale ablation contracts and native RT trajectory coverage. Ruff lint and formatting pass across the configured 180 maintained Python files; mypy passes on 79 source files; `uv lock --check`, `uv build`, and `git diff --check` pass. The WSL `auto` backend preflight was also run against the native Sionna RT stack: CUDA/OptiX was rejected by the one-path probe and LLVM was selected successfully. These implementation checks and the analytical pilot do not close P5, P6, P7, P8, P9, E11, or the external-evidence gate.

**Edition 3.57 RT backend entry-point consistency:** Extended the isolated native backend selector to the RT solver-convergence, stored-dataset ray-budget audit, and path-QA render runners. New convergence runs default to `auto`, preferring CUDA/OptiX only after a successful native direct-path trace and falling back to LLVM on probe failure. Audit and render replay intentionally request the exact Mitsuba variant recorded in the dataset and fail with actionable native-probe details if that exact environment is unavailable. All three runners now include resolved backend/probe provenance in their reports and activate the selected Mitsuba variant before importing Sionna RT. Selections also record the Sionna RT, Mitsuba, and Dr.Jit package versions. This keeps runtime behavior consistent across generators and analysis tools while preserving historical replay semantics.

**Edition 3.57 software verification:** The complete locked CPU suite passes (342 passed, 16 skipped, 78.97% coverage); the pinned Sionna/CUDA suite passes (398 passed). Ruff lint and formatting pass across all 180 maintained Python files, mypy passes across 79 source files, `uv lock --check`, `uv build`, and `git diff --check` pass. The backend preflight under Sionna RT 2.2.0 / Mitsuba 3.9.1 / Dr.Jit 1.5.0 confirms this WSL host cannot initialize OptiX and selects a successfully traced LLVM backend. This confirms runtime fallback behavior only; it does not validate OptiX execution on a supported Linux host or close any scientific evidence gate.

**Edition 3.58 RT CLI default consistency:** Changed the primary `loslab rt-data` command to default to the same capability-probed `auto` selection as the Python API and trajectory generator. Exact recorded variants remain selectable for reproducing historical datasets. A CLI contract test verifies the default without importing or initializing native Sionna in the lightweight test environment.

**Edition 3.59 maintained-script CI coverage:** Added `scripts/` to the GitHub Actions Ruff lint/format inputs and to configured mypy targets. This brings all 33 research/evaluation scripts into the same static-analysis gate as the packages and tests. The expanded mypy run uncovered three pre-existing typing defects in experiment CLI output handling and a local-name collision; those are corrected with runtime schema validation and a distinct profile scale variable. Mypy now checks 112 source files, including scripts.

**Edition 3.59 software verification:** After the CLI default and script-typing changes, the complete locked CPU suite passes (343 passed, 16 skipped, 79.10% coverage); Ruff lint/format pass across 180 maintained files; mypy passes on 112 files; `uv lock --check` and `git diff --check` pass. The pinned Sionna/GPU suite remains green at 398 passed from edition 3.57; these follow-up changes only affect the lightweight RT CLI default and CI static-analysis scope.

**Edition 3.60 RT scene asset identity:** Added `integrations/sionna/scene_assets.py` to recursively hash local Mitsuba XML references, OBJ meshes, MTL libraries/textures, and an adjacent scene manifest. Its canonical asset-set digest is checkout-root independent; unresolved URI/data resources are explicit and mark the manifest incomplete. Static RT generation places the full inventory once in the dataset generation report and compact asset identity in sample metadata. RT trajectory records carry only the compact identity while the immutable generation sidecar stores the complete inventory. Native LLVM tests assert complete asset capture for both packaged `box_one_screen` and `simple_reflector` scenes. This improves cache/replay provenance but does not qualify geometry or material realism; arbitrary plugin resources and runtime-converted material state remain outside the file hash contract. Details are in `docs/research/RT_SCENE_ASSET_IDENTITY.md`.

**Edition 3.60 software verification:** Complete locked CPU suite: 345 passed, 16 skipped, 79.26% coverage. Complete pinned Sionna/CUDA environment suite: 401 passed, including native LLVM RT scene-asset provenance tests. Ruff lint/format pass across 182 maintained Python files; mypy passes on 113 source/script files; `uv lock --check`, `uv build`, and `git diff --check` pass. These verify implementation and the local LLVM backend only; no OptiX-capable CI host or field-calibrated scene asset has been used.

**Edition 3.61 analytical observability sets:** Added profile-likelihood deviance curves for the fixed two-path complex CFR model after eliminating unknown complex gains. Experiment schema v2 calibrates a group-level threshold using each independent calibration condition's maximum true-delay deviance and emits bounded, potentially disconnected delay intervals on test groups. When the finite-sample conformal rank exceeds the calibration-group count, the report correctly returns the entire declared delay support. This is a set-valued classical comparator and an observation-level ambiguity diagnostic, not a claim of novel equivalence-class method: the fixed path count, known noise, exchangeable synthetic groups, bounded grid, and absence of RT/receiver mismatch remain material limitations. Added tests for the noiseless deviance minimum, whole-group calibration, support-unbounded behavior via the runner, and preservation of disconnected components. Historical experiment manifest v1 remains readable by the summary tool.

**Edition 3.61 analytical set-valued pilot:** Reran the same seeded 800-condition-group experiment with the set-valued comparator enabled. The split remains 480/80/120/120 train/validation/calibration/test groups (1,920/320/480/480 samples). A 257-point delay grid and a group-max calibrated 95% deviance threshold of 23.6374 produced 95.83% test-group simultaneous coverage and 98.33% sample coverage across 120 held-out groups. Mean total set measure was 297.7 ns; 16/480 sample sets contained multiple disconnected components (up to six), demonstrating that the output can preserve delay ambiguity instead of collapsing every observation to one interval. Descriptive coverage by the generated close/marginal/separated strata was 95.45%/100%/98.25%; these are not separately calibrated guarantees. The scalar point-estimate macro RMSEs reproduce the prior run to rounding: 40.67 ns identifiability-off, 40.36 ns identifiability-on, and 95.92 ns profile point estimator; the on/off paired interval still includes zero. These are same-mixture synthetic diagnostics under known path count and noise variance, not evidence of method advantage, scene transfer, or measurement validity.

The compact integrity-checked summary is `docs/research/results/observable-multipath-analytical-pilot-seed20261031-v4.json` (summary schema v3, SHA256 `f5f214cd68023676cfe49f0d6f19353142dc1bf44a8bf088d8f2c2ab9c3398b4`). It verifies run manifest SHA256 `5fc700c881a38747ad3c65b4788b50e38b1285c28e06215efb578dc9b615d7c3` and all ten artifacts; the complete set rows remain in ignored `outputs/observable-multipath-analytical-pilot-seed20261031-v5/profile_likelihood_delay_sets.json` (SHA256 `18bca6a08a8752fe77a6377b9074f8144071c9630493d56faa71da7cd06eedb`). Run provenance records source HEAD `304132c5b6cc49edfbdabbde819e7a0dc3da4999`, package-source SHA256 `31654a0f12cacd1bd696afd251df7afa449f3c68f5eb12e392417b2692e65dfd`, and runner SHA256 `d09ec34ff70451d51685a3923aabba22c911210e1da9b7eeef35ca4ed43290cd`; runtime was Python 3.12.3, PyTorch 2.11.0+cu128, and Sionna/Sionna RT 2.2.0. M2/E11 remains open pending theoretical validation of the decision rule, mismatched/path-order and receiver-noise stress, fair baselines, and independent qualified RT scenes.

**Edition 3.61 software verification:** The complete locked CPU suite passes (348 passed, 16 skipped, 79.51% coverage); the full pinned Sionna/CUDA environment suite passes (404 passed). Ruff lint/format pass across 182 maintained Python files, mypy passes across 113 source/script files, and `uv lock --check`, `uv build`, and `git diff --check` pass. The CUDA suite exercises training; RT scene integration runs on LLVM because OptiX cannot initialize on this WSL host.

**Edition 3.62 compact evidence output:** Kept raw per-group calibration scores and all per-sample interval components in the hash-verified full run artifact, while reducing the committed summary from 236 KB to about 9.4 KB. Summary schema v3 retains the calibration threshold/rank, aggregate test and stratum coverage, component-count histogram, all run-artifact hashes, and no per-row predictions. Schema v1 summaries remain readable for historical runs.

**Edition 3.63 complete RT run identity:** Added `integrations/sionna/rt_identity.py` and applied it to static Sionna RT dataset reports and RT-to-NR trajectory sidecars. A cache key now binds the complete scene asset-set digest, physical/grid/route configuration, solver settings, seed, selected verified Mitsuba variant and native Sionna RT/Mitsuba/Dr.Jit versions, broader runtime versions, and package-source digest. Probe diagnostics and checkout paths do not affect identity. Unresolved scene resources produce an explicitly incomplete identity with no reusable key. Tests establish determinism, invalidation when each result-affecting input changes, incomplete-asset fail-closed behavior, and rejection of an unidentified backend. This supplies cache identity/provenance only; it does not implement a cache store or guarantee bitwise equivalence across different hardware. See `docs/research/RT_RUN_IDENTITY.md`. P5's cache/replay gate remains open until production cache behavior and broader independent scene/material qualification are exercised.

**Edition 3.63 software verification:** The complete locked CPU suite passes (352 passed, 16 skipped, 79.56% coverage). Six focused native Sionna RT trajectory/backend tests pass on Mitsuba LLVM CPU. Ruff lint and format pass for all 184 maintained source, script and test files; mypy passes on all 114 configured files; `uv lock --check`, `uv build`, and `git diff --check` pass. The complete pinned native GPU suite and hosted CI have not been rerun for this provenance-only change.

**Edition 3.64 M2 receiver-noise and path-count stress implementation:** Extended the analytical profile-likelihood sweep with independently calibrated conditions for assumed/true receiver-noise variance ratios 0.5/1/2, a two-path matched model or a deliberately omitted third path at 0.25 of the strongest-path amplitude, and randomized physical path ordering. The fixed estimand is declared as the delay between the two strongest paths; the fitted model remains two-path. Added CLI parameters, validation, machine-readable per-condition and per-trial records, a compact SHA256-linked summary, and a protocol note giving the exchangeability/order-statistic argument and its limitations. Tests verify calibration/evaluation score-set equivalence on-grid, path permutations, condition separation, summary integrity fields, and invalid settings.

**Edition 3.64 analytical stress pilot:** Ran six cells at 64 tones, 30-kHz spacing, 20-dB SNR, 0.5 reciprocal-bandwidth target separation, 90-degree two-strongest-path phase difference, 1/32 reciprocal-bandwidth grid, and 120 calibration plus 120 evaluation realizations per cell (seed 20261031). The 95% rank was 115/120 in every cell. Empirical held-out set coverage for assumed/true noise variance ratios 0.5/1/2 was respectively 95.0/89.17/95.0% for the matched two-path source and 94.17/91.67/91.67% with the third path; corresponding median set spans were 49.60/49.60/49.60 ns and 82.67/198.41/115.74 ns. These are conditional on one calibration split; the matched, correctly specified 1:1-noise cell's Wilson interval is 82.34–93.56%, which lies below nominal 95%. The split-conformal order-statistic guarantee is marginal over calibration and future draws, whereas this interval conditions on the realized threshold. This result is not presented as a coverage failure or success; it makes repeated independent calibration splits a required next diagnostic. Random path permutations were exercised in all cells. No result establishes robustness to arbitrary channel mismatch, nonstationary receiver noise, measured data, or RT scenes.

The compact summary `docs/research/results/identifiability-profile-likelihood-stress-seed20261031-v2.json` (SHA256 `00b81fe68d46b51a1daea8f01365f028dd372878af95f246e40da73f5cef4e4f`) binds the ignored raw report `outputs/identifiability-profile-stress-seed20261031-v2.json` (SHA256 `dd1f9b22ff07d51634c201d4b505391fdf4c99b21111f8a7ff30bc759b0cc285`). The summary retains tone count/spacing, SNR, delay grid/support, separation, phase, all six mismatch conditions, thresholds/ranks, and coverage/set-size results. The clean run used source revision `3cdeeaf72bd6b11b98a488ef2f7b970fbd5435e8`, package-source SHA256 `57f2cdf816d9cce2ead1cfaa2bfa3d9c5766b88399ae74dcd4b4d60d2e7fcbef`, and runner SHA256 `91276e086fa345afe14c94bdfc91221d83194348753fa6c014f33b586d2f6833`. Reproduce with the command and source in `scripts/run_identifiability_profile_likelihood.py`; the protocol is `docs/research/PROFILE_LIKELIHOOD_STRESS.md`.

**Edition 3.64 software verification:** Complete locked CPU suite: 358 passed, 16 skipped, 79.62% coverage. Ruff lint/format pass across 186 maintained Python files; mypy passes all 116 configured files; `uv lock --check`, `uv build`, and `git diff --check` pass. The pinned GPU suite and hosted CI were not rerun for this CPU analytical experiment.

**Edition 3.65 repeated calibration-split stress:** Added `experiments/identifiability_repeated_calibration.py` and `scripts/run_repeated_identifiability_calibration.py`. The runner repeats complete, separately calibrated and evaluated cases with 100 seeded splits, 120 calibration and 120 test realizations per split, then bootstraps the split-level coverage rates (20,000 draws). This preserves the shared-threshold dependence within each evaluation split. At the same six noise-scale/third-path cells as edition 3.64, mean coverage and split-bootstrap 95% intervals were: assumed/true noise ratio 0.5 with no third path, 95.58% [95.06, 96.08]; ratio 0.5 with third path, 95.16% [94.64, 95.66]; ratio 1.0 without/with third path, 94.99% [94.48, 95.49] / 94.84% [94.23, 95.43]; ratio 2.0 without/with third path, 95.36% [94.81, 95.89] / 94.83% [94.21, 95.42]. The 20-split exploratory matched-model estimate was 93.96%; the 100-split result demonstrates why one realized calibration threshold is insufficient for interpreting marginal coverage. These intervals are bootstrap summaries across synthetic splits, not external calibration guarantees; all results remain conditional on one analytical channel family, fixed SNR/separation/phase, fixed third-path amplitude, and on-grid delay support. The earlier held-out split's low conditional Wilson interval is retained as a valid per-split observation and not erased by the aggregate.

The compact 100-split report is `docs/research/results/identifiability-profile-repeated-calibration-seed20261101-v2.json` (SHA256 `7d9329f364c49401f47e73c0f8b4ebd99c8499074594e56a0d17c6a023c97e4c`), binding raw ignored output SHA256 `ce222ed8f8e90f1ae06995d12f4a115cb9015ebb7c4d8233f7a89a85fc87582b`. It retains per-condition bootstrap/coverage/span/threshold quantiles and seed range; every split's raw metrics remain in the ignored report. The compact summary is 8,825 bytes, replacing a 292,800-byte draft that repeated raw split records. Source revision `6a37cffdcf385745f08b629ffc96af0a8dee18b4`, package-source SHA256 `b3044ad3c4cd9443e8e2690a3fd5015e0eb812d2041d5c4c1bbd0dc642108c61`, repeated-calibration module SHA256 `a0ccf2613fcc8550d3a02f0444c613f1c85b335371a644332c9533114e1626f3`, and runner SHA256 `42066cd616263251d87d3d3155c8b14c641c8225d9cced8fcd3dc258922dc008` are recorded in the report. Exact reproduction arguments are listed in `docs/research/PROFILE_LIKELIHOOD_STRESS.md`.

**Edition 3.65 software verification:** Complete locked CPU suite passes (360 passed, 16 skipped, 79.67% coverage); Ruff lint/format pass across 188 maintained Python files; mypy passes on 118 configured files; `uv lock --check`, `uv build`, and `git diff --check` pass. The repeated-split experiment uses CPU NumPy/SciPy only; the pinned GPU suite and hosted CI were not rerun.

**Edition 3.66 environment-aware static RT cache implementation:** Replaced the previously identity-only P5 cache groundwork with a user-invokable, immutable cache for static Sionna RT datasets. `loslab rt-data --cache-dir PATH` now resolves and probes the available Mitsuba backend, hashes complete scene asset closure and the physical/grid/solver/seed/source/runtime inputs, reuses only matching verified entries, and otherwise generates and atomically publishes the dataset plus a hash/shape/report sidecar. Runtime identity includes OS/release/architecture, Python implementation, CUDA/Mitsuba/Dr.Jit environment selectors, and the visible NVIDIA device/driver fingerprint when available; it degrades deterministically to `unavailable` when `nvidia-smi` is absent. The exact runtime fingerprint is shared between cache preflight and generation. Incomplete scene manifests bypass cache reads and writes. Payload, sidecar, input-key, generation-report, sample-count, and channel-shape checks fail closed; conflicting outputs for the same identity are rejected. Trajectory records retain their run identities but are not yet wired to this static dataset cache. Cache reuse is a software optimization and does not establish cross-device bitwise equivalence.

**Edition 3.66 software verification:** The locked CPU suite passes (369 passed, 16 skipped, 79.91% coverage). Ruff lint and format checks pass across 190 maintained Python files; mypy passes all 119 configured source files; `uv lock --check`, `uv build`, and `git diff --check` pass. Six focused native Sionna backend/LLVM trajectory tests pass in the pinned environment. This WSL host does not provide usable OptiX, so the automatic native probe exercises the configured LLVM fallback; OptiX execution must be validated on the later Linux host that supports it. Hosted CI is intentionally deferred until the implementation batch is ready for a single full review.

**Edition 3.67 label-semantics contract:** Added a first-class `ChannelDataset.label_semantics` field and persisted it in the pickle-free NPZ format, with legacy artifacts loading as `unspecified_legacy`. The TDL/CDL generator labels its binary target `profile_proxy`, the RT generator labels the depth-zero direct-path target `geometric_direct_path_oracle`, and debug fixtures are marked `synthetic_debug_direct_component`. Dataset selection preserves the value, report/field disagreements are rejected, and concatenation rejects incompatible binary targets so a profile proxy cannot silently be scored as geometric LOS. This is a compatibility guard and does not yet provide separate masked geometric, observable-direct, blockage, or continuous path targets; richer target records remain in the typed NR data model and are not retrofitted into the legacy channel classifier.

**Edition 3.67 software verification:** Locked CPU suite passes (373 passed, 16 skipped, 79.94% coverage); Ruff lint/format pass across 190 maintained Python files; mypy passes all 119 configured source files; lock check, package build, and whitespace checks pass. Nine focused native Sionna backend, LLVM trajectory, and standardized TDL/CDL generator tests pass. OptiX remains unavailable on this WSL host and is not claimed as verified.

**Edition 3.68 RT environment diagnostics:** Added opt-in `python -m loslab doctor --probe-rt`. The default doctor command remains a fast version-only report; the opt-in performs the same isolated native probe used by generation and reports requested backend, resolved Mitsuba variant, native versions, probe outcome, and fallback reason in JSON. Unavailable or failed backends are returned as structured diagnostics, so a WSL OptiX failure can be distinguished from a successful LLVM fallback before users spend time generating data. The `rt-data` auto mode continues to select a verified OptiX backend where available and falls back to LLVM otherwise.

**Edition 3.68 verification and local capability result:** The pinned Sionna 2.2 environment ran `python -m loslab doctor --probe-rt`; CUDA variants were installed, but the native CUDA probe failed with `Could not initialize OptiX!`, after which the native direct-path LLVM probe succeeded with `llvm_ad_mono_polarized`. This verifies automatic fallback on this host, not OptiX. A real `box_one_screen` dataset run through `generate_raytraced_cached` produced four balanced geometric-RT samples and a second call loaded the hash-verified entry with identical generation provenance; this is a software integration smoke, not a new research dataset result. The complete locked CPU suite passes (375 passed, 16 skipped, 79.98% coverage); Ruff lint/format pass across 190 maintained Python files; mypy passes all 119 configured files; `uv lock --check`, `uv build`, and `git diff --check` pass. Focused CLI tests cover default version-only behavior, successful backend reporting, and structured backend-unavailable reporting.

**Edition 3.69 evaluation-label gate:** The experiment runner now resolves one canonical target meaning before creating a run directory or fitting any model. It uses the dataset-level semantic contract for new artifacts, checks recognized per-sample semantics against it, and safely recovers uniform legacy RT/profile/debug labels from row metadata. Mixed or unknown legacy labels stop before training, preventing geometric direct-path labels and TDL/CDL profile proxies from being pooled into a misleading binary score. Run manifests retain both the resolved task meaning and the original per-source semantic description. This protects the legacy binary experiment path; it does not add soft labels or extra target heads.

**Edition 3.69 focused verification:** The experiment-runner tests pass, including positive resolution of uniform legacy RT metadata and a negative check that mixed geometric/profile labels fail before any run directory or model is created. Ruff and mypy pass for the changed implementation. The next complete suite is deferred to the planned consolidated verification batch.

**Edition 3.70 RT target contract:** Each generated RT-to-NR record now carries explicit validity masks for both full-path CFR and geometric LOS truth; the LOS mask is a scalar boolean rather than an implicit absence of masking. Record provenance names each target's semantics and unit, documenting that `channel_cfr` is an absolute-delay full-path complex response on effective PUSCH subcarriers and `geometric_los` is depth-zero LOS-only solver visibility. These values remain private targets and are not added to `NRModelInputs`.

**Edition 3.70 focused verification:** The native LLVM RT trajectory generation/storage round-trip and NR record/trajectory contract tests pass (19 tests); mypy and Ruff pass for the changed implementation. Full-suite verification will be consolidated after the current implementation batch.

**Edition 3.71 legacy-label sentinel handling:** Legacy artifacts that carry `UNSPECIFIED`, `UNKNOWN`, or `UNKNOWN_LEGACY` in per-sample metadata are now treated as unresolved labels, not as a valid task category. A dataset-level declared semantic can still resolve rows lacking old metadata, while explicit contradictory row labels fail. Eight focused experiment-runner tests pass, including consistent old RT-label recovery and prevention of mixed/unknown-label runs before training.

**Edition 3.72 RT trajectory cache:** RT-to-NR generation now computes its complete run identity before scene loading or ray tracing. A matching trajectory-ID manifest loads only after verifying the generation sidecar, identity-key contents, content-addressed trajectory, and each record's embedded identity; this returns without constructing a PathSolver. Reusing a trajectory ID for changed physical/runtime inputs fails before tracing and directs the caller to use a new ID. Interrupted publication without a sidecar is regenerated. Incomplete scene manifests bypass identity reuse. The feature complements the static dataset NPZ cache; cache hits preserve receiver-only observations and private target arrays exactly.

**Edition 3.72 focused verification:** Native LLVM route generation and storage round-trip pass, followed by a second call with PathSolver replaced by a fail-fast stub to prove the cache avoids retracing. A changed-SNR run using the same trajectory ID is rejected before solver construction. All 19 focused native trajectory and NR record-contract tests pass, along with Ruff and mypy for the changed code.

**Edition 3.73 hardware identity:** Extended the NVIDIA runtime fingerprint with GPU index, UUID, PCI bus ID, model, driver, compute capability, and memory size. This distinguishes identical-card hosts/devices more safely when forming CUDA RT cache identities; the environment variables controlling device visibility/order remain included. The CPU/LLVM path remains usable when `nvidia-smi` is unavailable.

**Edition 3.74 typed target definitions:** Added `TargetDefinition(semantic, units, source)` to the typed NR record contract and its pickle-free content-addressed storage, with backward-compatible reads for records created before the field existed. A nonempty definition map must cover every declared target. RT trajectory records define the absolute-delay CFR and depth-zero geometric LOS oracle separately; TDL→PUSCH records define the perfect synchronized channel target and Sionna PHY source. Definitions remain outside the observation-only model input. Dataset NR contract tests pass (27), and native LLVM trajectory plus native TDL→PUSCH tests pass (9); Ruff and mypy pass for changed code.

**Edition 3.75 trajectory consistency:** `NRTrajectory` now rejects changes in the meaning/unit/source definition of a target across time steps, while still allowing legacy records with no typed definitions. A focused contract test exercises definition drift; dataset record tests, Ruff, and mypy pass.

**Edition 3.76 matched baseline expansion:** Added a portable scikit-learn histogram gradient-boosting classifier (`histgb`) to the estimator registry and experiment runner, including immutable model/prediction artifacts and a paired MLP-versus-HistGB seed comparison. It joins the default baseline set beside RF, MLP, CORAL-off, and CORAL; users may also select it alone. Model-factory/fit and end-to-end runner tests pass. This broadens P6 controls but does not yet establish equal tuning budgets, a capacity/compute Pareto study, XGBoost, a CNN/CVNN, or performance superiority.

**Edition 3.77 observation-only CNN baseline:** Added an optional real-valued 1D CNN that consumes the same complex receiver observation as per-link real and imaginary channels over ordered subcarriers. Channel-wise scaling is fitted only on training observations; validation cross-entropy controls early stopping and the decision threshold remains validation-only. Checkpoints use tensor-only loading, and predictions, split identities, and training diagnostics use the common experiment artifact path. The runner supports paired CNN/MLP, RF/CNN, and HistGB/CNN seed comparisons. CNN is opt-in (`--methods cnn`) because PyTorch remains an optional dependency; portable default CPU experiments continue to run without it. This is an implementation control only: architecture/compute matching, tuning-budget equality, CVNN controls, adequate independent scene counts, and scientific performance evidence remain open.

**Edition 3.77 software verification:** The complete locked CPU suite passes (387 passed, 16 optional skips, 79.66% coverage). Ruff lint and formatting pass across 193 maintained Python files; mypy passes across 121 configured source files; `uv lock --check`, `uv build`, and `git diff --check` pass. CNN tests cover a short CPU fit, safe checkpoint round-trip including the validation threshold, malformed labels/inputs, unavailable CUDA, and raw-complex routing in the experiment runner. No native Sionna code changed in this batch; previous LLVM backend verification remains recorded in edition 3.68. These checks establish software behavior, not scientific performance.

**Edition 3.78 phase-equivariant CVNN control:** Added an optional complex-valued convolutional baseline. Complex convolution uses paired real/imaginary kernels without additive complex bias; modReLU preserves global-phase equivariance, and a magnitude feature head makes binary scores invariant to a common phase rotation. Per-link RMS scaling is learned from training samples only. Validation-only early stopping and threshold selection, tensor-only checkpoints, run artifacts, and paired CVNN-versus-MLP/CNN/RF/HistGB comparisons use the shared runner. The baseline is opt-in because PyTorch remains optional. This tests a precise architectural symmetry on the supplied CFR input; it does not establish calibrated carrier-phase robustness, equal compute/parameter budgets, scientific superiority, or invariance to per-antenna phase errors.

**Edition 3.78 software verification:** The complete locked CPU suite passes (391 passed, 16 optional skips, 79.95% coverage). Ruff lint and format pass across 196 maintained Python files; mypy passes across 123 configured source files; `uv lock --check`, `uv build`, and `git diff --check` pass. CVNN tests verify common-phase-invariant logits, a short CPU training and safe checkpoint round trip, malformed-input rejection, and raw-complex runner routing. This establishes implementation behavior only; compute matching and empirical performance remain open.

**Edition 3.79 baseline resource reporting:** Each experiment result now records estimator-fit wall time, validation and test prediction wall times with sample counts, learned parameter count when available, and peak allocated CUDA memory for Torch methods on CUDA. Neural parameter counts come from the fitted model; sklearn MLP reports learned weights and biases, while estimators without a meaningful parameterization report `null`. Device synchronization brackets measured Torch operations. This makes resource cost reviewable in the immutable results artifact but does not enforce equal budgets or create capacity–performance Pareto sweeps; those remain P6 work.

**Edition 3.79 software verification:** The complete locked CPU suite passes (391 passed, 16 optional skips, 79.99% coverage). Ruff lint and formatting pass across 196 maintained Python files; mypy passes across 123 configured source files; `uv lock --check`, `uv build`, and `git diff --check` pass. Runner coverage verifies nonnegative timing fields, MLP learned-parameter accounting, and null CUDA-memory reporting on CPU. Measurements are machine/run specific and are not evidence of matched compute.

**Edition 3.80 capacity controls:** Added named `small`, `medium`, and `large` width presets shared by the sklearn MLP, CORAL MLP, real CNN, and phase-equivariant CVNN. `loslab experiment --capacity` and experiment configs select a tier; the resolved configuration records it and checkpoints preserve it, while older neural checkpoints load as `medium`. Non-neural classical estimators keep their own fixed hyperparameters. This enables reproducible neural capacity sweeps across separate immutable runs; it does not itself equalize FLOPs, optimization updates, search budgets, or classical-versus-neural costs.

**Edition 3.80 software verification:** The complete locked CPU suite passes (394 passed, 16 optional skips, 80.06% coverage). Ruff lint and formatting pass across 198 maintained Python files; mypy passes across 124 configured source files; `uv lock --check`, `uv build`, and `git diff --check` pass. Tests assert monotonic parameter counts across capacity presets, preset validation, sklearn MLP architecture selection, and capacity-preserving neural checkpoint round trips. This validates selectable capacity controls, not a completed capacity-performance sweep.

**Edition 3.81 immutable capacity sweep:** Added `scripts/run_capacity_sweep.py` and `experiments/capacity_sweep.py`. One invocation runs requested methods over small/medium/large tiers with identical input hashes, grouped seed splits, observation settings, and device; verifies split hashes across all child runs; and writes per-tier AUROC, parameter-count and training-time points plus child-manifest hashes. Child runs and the aggregate receive completion markers, and existing output paths are never overwritten. A failed run preserves inspectable child artifacts and marks the sweep failed. The report explicitly limits inference to the supplied dataset and states that timing is machine-specific; this runner does not claim budget matching or scene-level evidence.

**Edition 3.81 software verification:** The complete locked CPU suite passes (395 passed, 16 optional skips, 80.10% coverage). Ruff lint and formatting pass across 201 maintained Python files; mypy passes across 126 configured source files; `uv lock --check`, `uv build`, and `git diff --check` pass. A grouped MLP pilot exercises all capacity tiers, verifies identical split identity and increasing parameter counts, checks immutable report/completion markers, and rejects output reuse. The pilot is a software fixture, not research evidence.

**Edition 3.82 live-status correction:** Refreshed the top-level status table to match the implemented and recorded M2 analytical pilot and the P6 baseline/cost/capacity-sweep tooling. Earlier snapshot text incorrectly said the decoder had not been trained. The current state still leaves the external-scene, model-mismatch, qualified-measurement, matched-compute, and scientific-performance gates open; the refreshed status makes no completion claim for P5–P11.

**Edition 3.83 optional XGBoost control:** Added an XGBoost tree baseline with deterministic histogram training, single-worker execution, and training-partition-derived binary class weighting. It can be selected as `xgb` and is available to the capacity-sweep runner as a fixed classical comparator. The `boosting` extra uses the CPU-only `xgboost-cpu` distribution, avoiding the default GPU distribution's large NCCL dependency; the neural CI job installs this extra and runs its tests alongside PyTorch coverage. Portable core environments remain free of the optional dependency.

**Edition 3.83 software verification:** The complete locked CPU suite passes with the CPU-only XGBoost extra installed (397 passed, 16 optional skips, 80.11% coverage). Ruff and formatting pass across 201 maintained Python files; mypy passes across 126 configured source files; lock check, package build, and whitespace checks pass. XGBoost tests cover imbalance-derived class weights, finite normalized probabilities, invalid targets, and experiment-runner integration. Portable core installs remain separate from this optional extra.

**Edition 3.84 raw-observation MLP:** Added `raw_mlp`, which flattens the receiver-observed CFR's real and imaginary parts without using engineered PDP/magnitude features. It shares train-only sklearn standardization, the named capacity presets, grouped split IDs, predictions, and cost records. The default experiment and capacity-sweep method lists now include it; the existing `mlp` remains the engineered-feature control. Paired comparisons cover raw-vs-feature MLP and raw MLP against RF/HistGB/CNN/CVNN. This closes an input-modality mismatch in C0–C3's implementation; it does not equalize training budgets or establish performance.

**Edition 3.84 focused verification:** Runner tests verify the raw MLP receives flattened real/imaginary receiver values at the expected feature dimension and emits normal immutable artifacts. Full-suite and quality-gate results are pending for this edition.

**Edition 3.84 software verification:** The complete locked CPU suite passes (398 passed, 16 optional skips, 80.11% coverage). Ruff lint and formatting pass across 201 maintained Python files; mypy passes across 126 configured source files; `uv lock --check`, `uv build`, and `git diff --check` pass. The raw-input test verifies flattened Re/Im dimension and complete prediction artifacts. No performance claim is made.
