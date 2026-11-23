# Research plan topic: receivers, targets and methods

This note is built from the archived master research plan at git tag
`docs/master-research-plan-archive`, file `docs/research/MASTER_RESEARCH_PLAN.md`. Each section below is copied
verbatim; the source line range is given before it. The archived plan is
not restored to the tree. Rebuild this note with
`python scripts/build_research_topics.py`.

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 1073-1128 -->
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

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 1129-1188 -->
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

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 1420-1505 -->
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
