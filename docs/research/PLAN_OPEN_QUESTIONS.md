# Research plan topic: open questions and risks

This note is built from the archived master research plan at git tag
`docs/master-research-plan-archive`, file `docs/research/MASTER_RESEARCH_PLAN.md`. Each section below is copied
verbatim; the source line range is given before it. The archived plan is
not restored to the tree. Rebuild this note with
`python scripts/build_research_topics.py`.

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 914-989 -->
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

### 0.2 Historical WSL environment and verification snapshot (edition 3.247)

The table below is retained as historical evidence from the earlier 3.247 snapshot. Its environment details and test totals are not the current repository status; use §0.3 and docs/research/MVP_STATUS.md for the authoritative current ledger.

| Item | Verified state | Consequence |
|---|---|---|
| Project | /home/nguyenl37/5g-nr-channel-condition, initial receiver and plan commit ffe199c80c1caf40dea91a406fe02124b3ff05ec | This checkout supersedes the uploaded ZIP and old directory name. |
| Quality gates | Latest complete local suite: 478 passed, 3 skipped, 88.16% coverage; scoped Ruff check/format, mypy (138 files), lock check, build, and whitespace checks pass. The run includes all eight native RT tests after initializing Sionna RT before test collection. | Local CPU/LLVM and GPU-PHY checks pass. Hosted CI run [39704656804](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39704656804) passed all four jobs. Hosted CI does not install native Sionna RT; the native checks remain locally verified. |
| Sionna reference | `/home/nguyenl37/sionna` tag v2.2.0, commit `2f0a3820269b53fd022780a6f2f5f720f7ceb9b4`; initialized RT submodule v2.2.0, commit `15b5ee036917a4c2a9b6420e05570b77717e2fcc` | Both checkouts were clean. Pin source review and experiments to these revisions; keep the clone external. |
| GPU | WSL2 NVIDIA RTX PRO 500 Blackwell Generation Laptop GPU, capability 12.0, about 6 GiB VRAM; driver 596.58, CUDA 13.2 | Suitable for PHY/model prototypes and small batches. Profile memory and stream data before scaling. |
| Project GPU environment | Project `.venv`, Python 3.12.3, PyTorch 2.11.0+cu128 with CUDA available on the RTX PRO 500 Blackwell Generation Laptop GPU; Sionna/RT 2.2.0, Mitsuba 3.9.1, Dr.Jit 1.5.0 | CUDA is usable for PHY/training. Native RT tests force the verified LLVM CPU variant before Sionna import because the WSL OptiX runtime is unavailable. The lock's generic Torch dependency does not preserve the CUDA wheel across environment synchronization. |
| PHY smoke | CUDA flat-channel PUSCH DMRS golden vector, controlled AWGN trend, transmitter → RayleighBlockFading → OFDMChannel → estimator loop, frequency-grid and two-tap sample-domain decoder trends, OFDM round trip, CP-based CFO/integer timing, and seeded PSS/SSS cell search with continuous fractional-timing refinement pass | Confirms only these NR PHY slices. CP CFO assumes a known frame boundary and static offset; CP timing assumes a first-symbol capture with bounded nonnegative delay. PSS/SSS evidence is a synthetic four-symbol burst with white-noise calibration and flat/one-echo channels; low-SNR errors remain. It is not a field-qualified or standards-complete receiver. |
| RT smoke | Upstream `test_los_friis_sanity` and project-native direct/reflection checks pass using Mitsuba LLVM CPU. Two built-in scenes generated 40 geometry-labeled links each, and a seven-seed box-to-canyon baseline pilot completed. CUDA variant fails during Mitsuba initialization: WSL provides an OptiX loader stub but lacks `optixQueryFunctionTable`. | CPU/LLVM is viable for bounded physical-data pilots; it is not evidence of scalable RT throughput or material fidelity. Keep CUDA PHY/training separate. `torch.cuda.is_available()` does not prove Sionna RT is GPU-ready. |
| Migration residue | Cache-only tests_research/ and stale los_channel_thesis.egg-info / wireless_channel_condition.egg-info were inventoried and removed; current package egg-info remains build-generated | P0 cleanup residue is complete; preserve archive/legacy_2025/ and notebooks. |

The upstream WSL OptiX workaround requires extracting matching driver files and copying components into the protected Windows WSL system library directory, followed by WSL shutdown. Do not make that host-wide driver change as routine setup. Current plan: CUDA PHY/model work, small CPU/LLVM RT physical checks, CPU CI, and optional GPU tests. No WSL RT throughput claim or massive GPU RT campaign until native RT GPU execution is verified. For scale/performance claims, use validated native Linux/OptiX infrastructure or revise the data strategy.

**Local source evidence:** PHY contracts are in `/home/nguyenl37/sionna/src/sionna/phy/nr/pusch_channel_estimation.py`, `pusch_transmitter.py`, and `pusch_dmrs_config.py`; profile version selection is in `src/sionna/phy/channel/tr38901/tdl.py` and `cdl.py` (default table selector `19.2`). RT controls are in `ext/sionna-rt/src/sionna/rt/path_solvers/path_solver.py`; `path_solvers/paths.py` defines CIR/CFR/taps, whose `normalize_delays` defaults true. The upstream PUSCH test reference is `test/unit/nr/test_channel_estimation.py`; the upstream RT test reference is `test/unit/test_path_solver.py::test_los_friis_sanity`.

### 0.3 Authoritative execution and research-gate ledger

This ledger separates implementation completion from scientific qualification. The latest full repository checkpoint passed `uv lock --check`, CI-scoped Ruff, formatting for 414 maintained Python files, mypy across 225 source files, the core suite (2,226 passed, 3 skipped), `uv build`, and `git diff --check`; it predates the current consolidated worktree. Since that checkpoint, targeted evidence includes the independent TDL-A–E revision-2 replication/audit (80 fresh channel seeds per profile), the TDL-C confirmation, the TDL-C signed-lag DFT receiver confirmation on 80 disjoint seeds, G4 CDL angular-power/ray-coupling checks (5 profiles), the G4 temporal-family checks, G3 v5 all-evaluator planning integration plus receiver-boundary regression, and split-policy domain separation plus signed-lock verification in the corpus auditor. The DFT result's independent audit passes and its three claims hold only for the registered TDL-C configuration. Focused tests and applicable changed-path Ruff/format/mypy checks pass; the complete native TDL integration test module now type-checks after fixing its generic keyword-argument dictionaries. The complete suite, package build, and hosted CI have not been rerun against the current worktree. G3/G4/G5 results remain limited to their frozen fixtures; software checks do not close scientific or release gates. Detailed acceptance criteria remain in the Unified Multipath Superdataset Plan.

| Area | Authoritative status | Remaining evidence to close |
|---|---|---|
| Simulation-only software MVP | **Complete within its implementation-smoke boundary:** six RT scene families, DMRS observations, six leave-one-scene-family-out folds, matched classifiers, reproducibility records, and hosted CI. This is not a qualified dataset or a scientific performance result. | Preserve reproducibility and CI; qualify the underlying propagation, receiver, and evaluation evidence before making broader claims. |
| G0–G2 foundations | **Partial; gate sign-off open.** Components and bounded checks exist, while all 33 gate checklist items remain unchecked. | Complete the specified environment, geometry/LOS, path extraction, absolute-delay CFR, and conservation evidence against the frozen criteria. |
| G3 propagation and path truth | **Partial; all 12 checks open; production D1 RT generation remains gated.** City43 Phase-0 diagnostics are bounded evidence only. The hash-bound family v3 and v5 acquisition-design CLI cover material, complex CSI, delay/angular propagation, and NR receiver claims under one shared family; each evaluator's scenario-conditional bootstrap is reported separately by its physical sampling unit, and the combined G3 sample size remains null. The checked-in v5 example exercises all four designs, but its scenarios are synthetic/unverified. No target-band calibrated material evidence, externally justified independent acquisition units, or representative population assumptions have been established. | Meet frozen complex-CSI, delay/angular-statistics, path-truth, and NR-impact criteria; establish calibrated angle-resolved material evidence and independent acquisition units; externally justify scenarios and complete the separate powered designs before qualification runs. |
| G4 standardized channel references | **The mandatory named Core-1 matrix is verified:** the 50-cell TDL/CDL covariance family, 50 path-axis reconstruction cases, and CDL-A–E 1×4/4×1/2×2/4×4 joint covariance family pass their frozen native CPU checks. Additional bounded evidence covers CDL-A–E angular-sector power, pairwise ray-rank coupling, four-angle rank quartiles, and selected continuous-valued four-angle Fourier characteristics; CDL-D/E coherent LOS composition; and CDL-A–E temporal covariance at one speed/lag along x/y/z plus one oblique 3D direction. Scalable TDL temporal-covariance and two-sided Doppler-spectrum families now have retained frozen native passes: 570/570 covariance coordinates and 35/35 spectrum-CDF claims respectively, under their predeclared confidence rules. These remain pinned standards-model consistency checks: the full continuous joint angular law/orientation coverage, arbitrary arrays/patterns, all polarization/time configurations, geometric RT, and physical channel accuracy are unqualified. | Complete the remaining profile/configuration checks and broader G4 evidence; keep standards-profile reference evidence distinct from geometric-RT truth and measured validity. |
| G5 NR receiver and impairment qualification | **Partial (12/15 checks); G5.01, G5.09, and G5.14 remain open.** PBCH known-occasion reception covers candidate counts 4/8/10/20/64 and supports synthetic multi-chain MRC; blind acquisition has seeded single-chain cases at counts 4/8/10/20/64 plus one bounded two-chain timing/CFO/multipath case. A preregistered shared-CP synchronization confirmation on 64 synthetic independent channels × 8 blocks passes its declared fixture criteria. Independently audited TDL-A–E revision-2 results pass 15 bounded static-reference/mobile-BLER/goodput claims on 80 fresh seeds/profile, and a separate TDL-C confirmation passes the same class of claims on 64 seeds. Both validate only their named statistical-channel setups. The custom two-path Doppler/DM-RS confirmation remains `NOT_MET` (7/64 channel draws failed its static-reference criterion); historical receiver reports are not evidence for subsequently modified source. | Resolve G5.09 with a justified receiver/protocol design and fresh independent confirmation across its required operating envelope; complete standards-conformance evidence, broader multi-channel/topology/waveform/power/numerology qualification, G5.01 and G5.14, and independently validated PRS/SRS behavior. |
| G6–G8 dataset, evaluation, and external validity | **Foundations exist; all 35 gate checklist items remain unchecked.** Forecasting, release integrity, powered independent evaluation, and external measured evaluation are not qualified. | Complete trajectory/intervention evidence, corpus-scale integrity, powered held-out evaluation, and independently verified measured evaluation. |
| Qualified research result or release | **Not complete.** There is no production D1 RT corpus, completed qualified Unified superdataset, public qualified release, or validated main-paper result. | Close applicable gates and preregistered evaluation requirements before describing a result or dataset as qualified. |

### Historical implementation detail retained from edition 3.247 (superseded status snapshot)

| Gate | Recorded state at edition 3.247 | Next evidence |
|---|---|---|
| P0 repo baseline/cleanup | Core gates pass; cache-only tests_research/ and stale egg-info removed; research plan, clean CI and packaging are maintained | Keep implementation revisions committed and pushed; preserve archival thesis assets unchanged. |
| P1 CUDA PHY | PUSCH/Rayleigh, TDL→PUSCH, flat-channel DMRS, impairment/decoder sanity, sample-domain TDL-A–E curves, exact same-channel 1-vs-2-RX diversity/correlation studies at 12-PRB/30-kHz and CP-contained 24-PRB/15-kHz, plus a phase-resolved CUDA memory profile pass in the pinned environment | Extend resource/receiver profiling to further delay spreads and link configurations; test broader correlation matrices and calibrated/measured arrays. Synthetic correlation is not array calibration. |
| P1 RT physics | Direct/two-ray/material checks, a deterministic geometry-label smoke, matched three-seed path-sample/depth convergence sweeps on three built-in scenes, and a six-scene LOS-mixed/NLOS-indirect audit pass and matched six-scene retrace on LLVM CPU; all 38/278 empty full-solver outcomes were attributed to geometric-NLOS candidates (0/120 geometric-LOS calls), with 240 accepted records. Etoile (depth 1–3 × 500k/1M/5M) and urban_00043 (depth 1–3 × 20k/100k/500k) class-conditional sweeps with second-seed checks are recorded. Higher depth reduced NLOS noncoverage in both selected scenes; CFR budget sensitivity varied by scene. RT CUDA remains unavailable under current WSL OptiX setup. | Replicate depth/budget/seed sensitivity across the remaining scenes; add blocked/outage and diffuse multipath cases; extend path-overlay, material qualification and trajectory gold cases; no GPU RT performance claim. |
| P2 NR receiver | SionnaPUSCHLSReceiver and framework-independent NR contracts pass; CUDA tests cover flat DMRS recovery, AWGN trend, Rayleigh loop, OFDM round-trip, CP CFO and bounded known-origin integer timing acquisition/correction, timing-impairment sensitivity, frequency-grid BER/BLER ordering, static two-tap sample-domain checks, sample-domain TDL-A–E curves, and same-waveform TDL-A diversity sweeps at 12-PRB/30-kHz and 24-PRB/15-kHz with ρ=0 and synthetic ρ=0.9; paired mobility/DMRS and fixed-payload multi-delay CP sweeps also pass. PSS/SSS acquisition now includes continuous fractional-timing refinement and a noise-calibrated operating-characteristic study over flat and one static 12-sample echo channels (0/6/12/18 dB, 256 trials/point, 2,048 held-out noise captures). | Low-SNR and strong-echo errors remain; the finite four-symbol fixture does not cover interfering cells, evolving channels, realistic RF capture, broad timing/CFO ranges, or field conditions. Calibrated/real-array response, further numerologies/link configurations, and measured receiver evidence remain open before generalized receiver-capability claims. |
| P3 RT-to-NR and observation schema | Paths.cfr-to-OFDM-grid bridge, hash-addressed NR records, immutable scene-group splits, spatial/CFR/temporal leakage audits, native GPU TDL→PUSCH records, six CPU/LLVM-retraced NR DMRS trajectories (24 timestamped observations), and six-scene `[240,1,72]` Sionna PUSCH/DMRS observations pass. All six scene families have leave-one-scene-family-out folds with three matched methods and seven seeds; the split/hash auditor rejects missing folds, changed observations, inconsistent seeds/splits, and held-out-domain leakage. Indoor/outdoor/hallway CAEZ magnitude archives pass CRC/schema/timestamp-order audits, and a frozen indoor-trained forecast is evaluated on all three chronological tails with paired time-block intervals. CAEZ has no LOS/NLOS labels and cannot close the geometric LOS validation gate. | Improve power beyond 40 links per held-out family; extend trajectories and receiver conditions; independently qualify geometry/materials, motion, and outage policy; reconcile current ETH-page vs downloaded-file sample counts; then run powered transfer/forecast evaluations. The six selected synthetic scenes and one campaign tail per site do not close external/physical validation. CAEZ output is magnitude-only and not phase-sensitive. |
| P4 analytical / standardized tier | Seeded 1–3 path fixtures, finite-bandwidth steering, joint Fisher diagnostics, 80-case two-path CRLB sweep, native Sionna v2.2.0 TDL→CFR/PUSCH A–E checks, sample-domain PUSCH A–E BLER curves, paired TDL-A receive-diversity and synthetic spatial-correlation sweeps, and residual synchronization/decoder checks | Extend analytic nuisance sweeps; validate TDL/CDL distributions, version drift and standard conformance; maintain profile-proxy labels separately. |
| P4+ science / P7 analytical M2 | The fixed-two-path CFR decoder has been trained and evaluated on 800 synthetic condition groups with identifiability-off/on ablations, profile-likelihood point estimates and group-calibrated set-valued inference. A new OMP + continuous-delay refinement + unknown-noise BIC baseline tests path orders 0–3 on 960 independent analytical trials. It selected the exact order in 82.40% overall (227/240, 219/240, 195/240, and 150/240 for true orders 0–3); the three-path rate was 62.50%. This exposes model-order ambiguity in the tested synthetic setup; it does not validate physical path count, diffuse channels, RT transfer, or the proposed method. | Stress receiver-estimator error, misspecified/diffuse channels and condition shift; validate the equivalence decision rule; compare under matched inputs/compute; move to independently qualified RT scenes. |
| P6 fair-baseline implementation | RF, HistGB, XGBoost, logistic regression, engineered-feature MLP, raw flattened Re/Im MLP, CORAL-off/on, real CNN, phase-invariant CVNN, and an unknown-order sparse CFR estimator are available. Classifier results record timing, learned parameter counts where defined, and optional CUDA peak allocation. Small/medium/large neural tiers and an immutable fixed-split capacity-sweep runner are implemented and tested. A six-scene LOSO capacity experiment completed 756 fits (six methods × seven seeds × three tiers × six held-out scenes) on the same precomputed DMRS artifact, using CUDA for neural fits; per-scene split and artifact audits pass. | Repeat after geometry/material qualification; equalize/report training updates and tuning budgets; extend independent scenes and evaluate P6 E03–E08. Current results are exploratory; no comparative superiority is claimed. |
| MVP exit gate | Simulation-only software MVP complete: six RT scene families → DMRS observations → six leave-one-scene-out folds → matched classifiers; command/reproduction record and hosted CI are maintained in `docs/research/MVP_STATUS.md`. | Publication-level research remains open: independent measured evidence, qualified scene/material assets, robust receiver impairments, powered sample sizes and external scene families. |

The project-owned receiver tests verify exact flat-channel PUSCH recovery, lower estimation NMSE at lower AWGN variance, and a Rayleigh transmitter/channel/receiver loop against native channel truth. RT-grid tests additionally verify a native direct RT path and coherent LOS-plus-specular two-ray response.

---

**Edition 3.176 post-solver weak-path sensitivity audit:** Committed runner `f839c02` measures how per-link relative path-coefficient-power masks applied after PathSolver alter the absolute-delay coherent CFR, without changing canonical path truth. A three-seed blocked link in the synthetic City43 16-building scene has 335–349 valid paths. A −20 dB cutoff keeps 15–23 paths but causes 43.4–75.3% relative CFR L2 error; −40 dB keeps 99.85–99.88% of summed coefficient power yet causes 2.30–6.65% CFR error. The hash-bound report preserves paths, CFR vectors, settings and source hashes. This is one-link, S=0.5 synthetic sensitivity evidence, not receiver detectability or a justified production cutoff. G3.09 remains open pending broader scenes, physical threshold semantics and predeclared tolerances. See the [audit report](results/rt-city43-nlos-path-power-threshold-20261022.json), [City43 3D building scene](figures/rt-city43-los-nlos-capstress-depth3-s05-20k-scene-20261022.png), and physics-status edition 3.182.

**Edition 3.177 City43 interaction-depth evidence across three TX sites:** The matched 0–10 depth sweep covers six geometry-audited City43 links (one LOS and one NLOS at each of three TX sites), seeds 11/23/47, 20k rays/source, 1M candidate cap, and synthetic S=0.5 material overrides. At depth 10, none of the 18 link-seed runs saturates the cap. From depth 9 to 10, the finite-reference median CFR error is 15.4% LOS (range 0.007–45.3%) and 0.103% NLOS (range 0.018–0.882%); LOS RMS delay still shifts by up to 22.0%, while selected NLOS delay and angular statistics are nearly unchanged. Path counts still increase, and the depth-10 result is only a comparison reference. This adds site-position coverage but does not resolve G3.01: LOS is not stable and ray-budget convergence/material calibration remain unqualified. Evidence and limitations are in docs/research/PHYSICS_VERIFICATION_STATUS.md edition 3.183.

**Edition 3.178 paired 20k/100k City43 ray-budget audit:** Repeated the six-link, three-TX-site, three-seed depth-0–10 sweep at both 20,000 and 100,000 samples/source, storing matched full CFRs and candidate counts for 216 runs. No run fills the 1M path cap. At depth 10, path counts rise about 4.4–5.0× under the fivefold ray-budget increase. Against the finite 100k reference, 20k-budget relative CFR L2 error has a 0.76 LOS median (0.65–0.92 range) and 0.71 NLOS median (0.24–1.14 range). This large change in both classes shows that apparent 20k NLOS depth stability is not robust to sampling budget. Even at 100k, depth 9→10 LOS CFR changes reach 30%; NLOS are below 0.9% in these matched runs. These results are budget/depth sensitivity evidence, not convergence certification or ground truth; G3.01 and G3.02 remain open. See docs/research/PHYSICS_VERIFICATION_STATUS.md edition 3.184 and the paired raw report at docs/research/results/rt-city43-spatial-los-nlos-depth10-s05-20k-100k-20261022.json.

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 1979-2022 -->
## 11. Risk register, mitigations and scientifically acceptable pivots

| Risk | Severity | Detection | Required mitigation / pivot |
|---|---|---|---|
| False novelty: multipath tokens, CVNN, ToA multitask or symmetry already published | Critical | living related-work/component matrix | define a precise **new mechanism** and evaluate against closest prior; adjust title if overlap discovered |
| TDL/CDL profile labels mistaken for physical LOS | Critical | explicit label scopes, profile-only oracle | primary geometric claims only from independently labeled RT/measurement domains |
| Absolute first-path delay lost through `normalize_delays=True` | Critical; adapter corrected | native adapter-to-Sionna comparison at a known nonzero delay; same-Paths two-ray CFR conservation; full gain/path/array reconstruction remains pending | preserve absolute oracle in the versioned channel contract, keep old artifacts immutable, and separately implement/validate receiver-relative or synchronized observables |
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

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 2023-2070 -->
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

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 2071-2116 -->
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
