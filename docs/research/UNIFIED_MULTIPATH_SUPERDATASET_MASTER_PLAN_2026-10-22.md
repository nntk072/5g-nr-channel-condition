# UMCD-1: Unified Multipath Channel Dataset — Complete Design, Generation, Validation, and Research Plan

**Project:** 5G NR Channel Condition / Multipath State Intelligence
**Planning date:** 2026-10-22 (Europe/Helsinki)
**Status:** design specification; **not** a generated dataset, empirical result, simulator-accuracy certification, or promise of a top-tier publication.
**Implementation status (updated 2026-10-23):** Foundations and controlled pilot audits exist; G3 acceptance criteria are explicit, but the active release blocker remains G3 physics/observable qualification. The bounded City43 investigation localized the anomaly to the deterministic shoot-and-bounce count/reservation/replay region, with ImageMethod promoting default invalid slots into direct-like valid paths. The exact thread/hash/output-offset source remains unresolved, and this is not a confirmed Sionna defect. This is a targeted diagnosis, not qualification. No production ray setting is approved and RT production D1 generation remains gated. G0–G2 have component evidence but no plan-wide sign-off. Release Split Policy v2 independently reconstructs canonical connected components and assignments from frozen lineage/policy; standardized-draw opaque partition identities default to generation seed plus within-run draw index, independent of profile/target/source-family labels. Callers must provide explicit opaque per-draw IDs when independent streams reuse a seed/index; the registry rejects collisions rather than silently joining them. The contract records explicit child-to-parent bindings and SHA-256 lock digests with integrity-not-authorship semantics. The Phase-A hardening suite covers re-hashed noncanonical locks, child-binding conflicts, reused-seed/index collisions, and legacy three-way behavior. The dependency-ready G5 receiver foundation has passing native CPU evidence for resource-grid/FFT mapping, ideal decoding, LS/LMMSE, thermal noise, observation axes/masks, same-parent noise views, SNR provenance, and the model-input firewall; broader G5 conformance remains open. G4.07's standardized-profile statistical interval criteria now pass for TDL/CDL A–E delay/PDP, TDL A–E temporal correlation, and bounded TDL/CDL receive covariance configurations; arbitrary array/correlation combinations remain outside this evidence. G4.08 frequency, bandwidth, link, and array admission checks pass for the tested standardized TDL/CDL scope. G4.09 now emits and tests an explicit profile-metadata-only control for the A/B/C-versus-D/E proxy label; this only measures recovery of the label definition and is not geometric-LOS evidence. Other applicable G4 admission checks remain open. G7 remains open for corpus-wide schema, spatial/temporal/duplicate/target-coverage audits and external lock pinning. G3 qualification and production D1 RT generation remain separately gated. D3 mobility, broad network/polarization coverage, and measured D5 validation remain later releases.

**G5.12 foundation (2026-10-23):** Standardized TDL/CDL PUSCH generation now separates the transmitted-waveform seed from an optional receiver-noise seed. Each persisted noise view has a deterministic ID and identity digest bound to its standardized parent, receiver configuration, SNR, time index, and realized seeds; its provenance records the parent, noise/SNR definition, and complex versus per-real-dimension noise variance. A native CPU regression test verifies different-noise views retain identical CFR truth and waveform seed, receive different observation samples, replay deterministically, and survive immutable NR-record round trips. This closes only the tested standardized noise-view case; it does not qualify the full receiver gate or bind a complete corpus release split.

**G5.01 resource-grid mapping coverage (2026-10-23):** Native CPU checks cover Sionna PUSCH resource-grid FFT sizing, full-band effective-bin ordering, DMRS mask axes, and recorded numerology across 15/30/60/120-kHz subcarrier spacings and 2/6/12/24-RB allocations. A separate reference-vector test checks the full-slot, mapping-A, Type-1 DMRS symbol positions and frequency comb against TS 38.211 V19.1.0 Table 6.4.1.1.3-3 and Table 6.4.1.1.3-1 for additional positions 0–3. Independent Gold-sequence vectors now check DMRS values for the tested cell/scrambling identities; together with the frequency-selective DMRS estimator check, this catches mapping, sequence, and axis regressions in the covered configurations. G5.01 remains open for broader standard-conformance coverage, including transform precoding, frequency hopping, multiport, other DMRS types/mappings, and numerologies beyond the tested cases.

**G5.02/G5.03 receiver fixture (2026-10-23):** A native CPU test passes a frequency-selective, adjacent-bin-distinguishable channel through a Sionna PUSCH resource grid without noise and verifies the DMRS estimate at pilot REs against the exact full-band FFT-bin gains. It asserts the current supported 6-RB grid has no guard/DC removal and maps all 72 estimate bins to FFT bins 0–71. This is a fixed full-band frequency-domain fixture; it does not validate other numerologies/guard layouts, CP, waveform timing, or ISI/ICI, so G5.03 remains open.

**G5.10 analytical receiver fixtures (2026-10-23):** A native CPU Monte Carlo fixture applies proper complex AWGN of declared variance 0.05 to a flat known channel over 1,024 PUSCH draws. At one DMRS pilot RE, measured LS bias and variance are checked against the known channel and Sionna's analytical pilot-estimator variance (0.025 after CDM averaging). A separate native Sionna LMMSE fixture uses a 3×2 channel and non-diagonal noise covariance; equalized symbols and effective noise variance match an independently computed filter/error covariance, and the measured Monte Carlo error variance agrees within a predeclared 2.5% absolute ratio tolerance. This satisfies the analytical-fixture criterion for these cases; interpolation-wide LS statistics and other receiver/channel configurations remain unqualified.

**G5.06 idealized thermal-noise qualification (2026-10-23):** The RT-DMRS and standardized-PUSCH builders share a proper-complex-AWGN helper, with `E[|n|²]=variance` and each real component having half that variance. A deterministic CPU statistical test feeds exact `kTB·F` variance into the helper and checks complex power, real/imaginary variance, cross-antenna covariance, and seeded replay. A CPU no-path RT trajectory test measures the stored noise-only received grid against the same `kTB·F` value and checks properness plus adjacent-frequency and adjacent-symbol covariance. This qualifies the declared ideal independent complex-RE noise model for the tested receiver configurations; analog front-end filtering, interference, and correlated antenna noise are not modeled.

**G5.11 observation-mask schema (2026-10-23):** `NRObservation` now carries a boolean `received_grid_mask` on exactly the received-grid axes, exposes it through the inference-only model-input object, and persists it with NR records. Older/direct observations default to an all-observed mask; the estimator's DMRS `pilot_mask` remains a separate field. A storage round-trip test preserves a deliberately missing resource element. This validates the typed representation and persistence contract, not any physical outage/missingness-generation mechanism.

**G5.13 native PUSCH outcome fixtures (2026-10-23):** The matched Sionna PUSCH transmitter and `PUSCHReceiver` run on CPU in CI for both the frequency-domain AWGN reference and standardized TDL time-channel cases. The AWGN fixture evaluates one seeded payload/noise realization at two noise levels and confirms decoded BER/CRC worsen at lower SNR. The TDL fixture passes the adapter's DMRS LS estimate to the decoder and recovers the seeded payload with valid CRC for one CP-supported 24-RB case and one CP-exceeding 6-RB case at 100 dB. The latter is a single high-SNR diagnostic: CP exceedance does not guarantee a decoder failure, and this result cannot establish absence or size of receiver impact. Repeated CP-exceeding BLER, mobility, impairment sweeps, and a deployment operating curve remain open.

**G5.04 bounded CP/ISI fixtures (2026-10-23):** A native CPU test modulates a seeded 6-RB PUSCH grid, applies a static two-tap sample-spaced channel in the waveform domain, and compares demodulated resource elements against the diagonal frequency-domain response. At an echo delay one sample inside the configured CP, the normalized discrepancy is below `1e-6`; one sample beyond the CP, it increases by more than `1e-3`. The standardized PUSCH adapter now also applies TDL/CDL path coefficients and excess delays through Sionna's sinc-discretized `cir_to_time_channel` and `ApplyTimeChannel` by default. Against each sample's own diagonal CFR prediction, a 24-RB TDL/CDL fixture with tap span inside the CP has normalized residual below `1e-4`; a 6-RB TDL case whose finite sinc-tap span exceeds the CP has residual above `5e-4` and differs from the explicit ideal-CFR control. These are static, seeded CPU cases; they do not qualify mobility, other numerologies/delay strata, or receiver BLER/CRC under multipath. G5.03's direct-path perfect-channel/noiseless receiver criterion remains separate and these fixtures do not establish multipath decoding qualification.

**G5.03 perfect-channel/noiseless receiver fixture (2026-10-23):** A native CPU test round-trips a seeded 6-RB PUSCH grid through OFDM modulation and demodulation using its configured CP, then decodes it with perfect unity CSI. No noise is added to the waveform; a `1e-8` detector variance is used only to keep the LMMSE detector nonsingular. The recovered grid matches the transmitted grid, decoded payload bits match exactly, and all transport-block CRCs pass. This is an ideal direct-path receiver reference, not qualification under channel estimation error, multipath, impairments, or other numerologies.

**G5.05 NR symbol-clock/Doppler fixture (2026-10-23):** The normal-CP NR timing helper is tested against the 15-kHz slot CP pattern and slot-dependent timing, while an adapter contract test feeds its 30-kHz symbol-start times into explicit-time RT channel evaluation and checks the complex phase against `exp(j·2π·f_D·t)` for a 2.4-kHz path Doppler. The native CPU RT trajectory test additionally checks that generated symbol-channel targets record the same TS 38.211 symbol-start times and vary over the slot. This verifies the tested clock-to-Doppler contract; it does not qualify all numerologies, moving geometry within an OFDM symbol, or G3 RT stability.

**G5.08 physical-gain receiver fixture (2026-10-23):** The absolute-power RT-DMRS receiver test now runs on CPU. Two analytic channels with gains 1 and 0.5 use the same per-RE transmit power and `kTB·F` noise; the observed physical CFR retains the 0.5 gain ratio. The normalized per-link SNR diagnostic produces equal observations for those same channels and records that path-loss normalization was applied. This verifies the receiver builder's scale separation and thermal-noise accounting for the controlled fixture; it does not independently validate scene material/path-loss accuracy or every NR link-budget component.

**G5.09 sample-domain impairment primitives and standardized PUSCH views (2026-10-23):** CFO, caller-supplied oscillator phase trajectories, seeded Wiener phase noise, memoryless I/Q gain/phase imbalance, and integer/fractional timing helpers have CPU analytical checks. The declared phase-noise model is a Wiener random walk with independent increment variance `2π·linewidth_hz/Fs`; a statistical test checks its variance, phasor correlation, and exact seeded replay. A composition fixture verifies the chain `Tx I/Q → channel → CFO/phase trajectory → Rx I/Q → timing-window offset` against direct samplewise references. The standardized PUSCH path now defaults to a frozen, sinc-discretized TDL/CDL time channel, so path delays can produce CP spillover/ISI; `channel_mode="ideal_cfr"` remains an explicit circular-frequency diagnostic control. Native CPU tests check the received-grid residual against each sample's diagonal CFR for CP-supported and CP-exceeding paths, zero-impairment equivalence, a shared receiver oscillator across antennas, and seeded same-parent impairment views. Seeded high-SNR decoder fixtures recover exact payload/CRC for one CP-supported 24-RB TDL case and one CP-exceeding 6-RB TDL case using the adapter's DMRS LS estimate; the latter shows CP exceedance alone does not imply CRC failure for that realization, not that receiver impact is negligible. A paired 60-dB TDL fixture verifies combined impairments increase DMRS LS NMSE over the same parent draws. This uses a static channel snapshot over one PUSCH allocation and a finite Sionna sinc tap window. The oscillator model is single-linewidth Wiener noise without PLL shaping or measured PSD; repeated CP-exceeding BLER, broader impairment parameter coverage, and decoded mobility impact remain open.

**G5.15 NR input-firewall fixture (2026-10-23):** An adversarial receiver-record test attaches extreme future-oracle CSI, target masks/definitions, sample IDs, and provenance to the private training record, then verifies the inference payload exposes only the exact typed receiver-input fields and retains the estimator output rather than oracle CSI. This verifies the `NRObservation` adapter/schema boundary; it does not replace corpus-wide G7 audits or prove every downstream model loader uses only this adapter.
**Implementation status (2026-10-22, historical):** Foundations and controlled pilot audits exist; the active release blocker is Gate G3, especially depth and ray-budget stability (G3.01/G3.02). Matched City43 depth-10 sweeps remain highly budget-sensitive, including with scattering set to zero. A new 20-seed depth-3 City43 comparison covers six links across three TX sites at 100k/1M samples/source: with native S=0 materials it produces no diffuse paths and stable aggregate observables; with an uncalibrated S=0.5 override, diffuse-delay summaries have broad seed intervals, while LOS total RMS delay shows a small but resolved shift. These are finite synthetic sensitivity tests, not ground truth or material calibration; no production ray setting is approved. The 3D Urban43 pilot has 3 TX sites and 120 RX links; its held-out-location baselines remain exploratory. This is **not** a frozen, release-qualified UMCD corpus. G0–G2 have component evidence but no plan-wide sign-off. Next: validate budget/depth behavior across independent scene/material cases and resolve/bound G3 for the MVP; admit the selected standardized/NR channel and receiver paths under applicable G4/G5 checks; generate the multi-scene, multi-position D1 corpus; lock and audit splits/integrity under G7; then run independent baselines and falsification checks under G8. D3 mobility, broad network/polarization coverage, and measured D5 validation remain later releases.
**Existing repository evaluated:** supplied `5g-nr-channel-condition-main.zip` (571 ZIP entries; extracted source inspected), with comparison to the earlier `master_thesis-main.zip` and prior research plans. **Do not assume the ZIP represents the newest inaccessible GitHub commit.**
**Canonical simulator reference:** NVIDIA Sionna **2.2.0** source/documentation; `NVlabs/sionna` (PHY/SYS) and `NVlabs/sionna-rt` (RT), with pinned tags, hashes and dependency lock at execution. Versions and signatures must be verified at runtime.
**Goal:** **ONE versioned, queryable, scientifically defensible dataset** with heterogeneous provenance-preserving partitions, not one giant flat tensor or misleading universal label.

> **Core principle:** collect each expensive physical propagation event **once**, preserve its raw ground truth, then derive synchronized multi-antenna / time / carrier / receiver-observation views and task labels with explicit validity. Never mix generator-only knowledge into model inference inputs.
>
> **Absolute-result caveat:** a simulation dataset cannot guarantee absolute physical truth or universal model performance. It can offer reproducible numerical results, verified simulator-grounded labels, audited uncertainty, and transfer evidence. Real propagation requires measurements or externally validated calibrated models.

## Decision summary (read first)

**Edition 3.191 City43 depth-10 zero-scattering Phase-0 attribution (G3 diagnostic; not qualification):** On the preselected worst case (`center-los`, seed 23), the 100k/1M samples/source, depth-10 pair has 87.09% relative complex-CFR L2 difference and no candidate-cap saturation. Path-level CFR reconstruction closes to numerical precision. Of 597 high-budget versus 194 low-budget valid records, 440 occur only at the higher budget; a conservative geometry/order match finds 157 shared records whose coefficient, delay, endpoint direction, and interaction-vertex differences are tiny (matched coefficient relative-error p95 2.48e-5; delay p95 5.68e-5 ns; vertex displacement p95 0.286 mm). The dominant change is instead the growth from 9 to 75 zero-interaction records. Those records collapse to one coefficient, delay, and geometry fingerprint at each budget; 8/9 low and 74/75 high records have zero TX-angle fields. The newly present path terms account for 99.9965% of the CFR-delta norm, and the order-0 component is 97.93% of high-budget CFR mean power, indicating coherent accumulation of duplicate direct-like records in the returned path set. Higher interaction orders grow in count but carry negligible power in this zero-scattering case. This localizes the selected output discrepancy to path-record discovery/multiplicity plus coherent summation, not matched-path coefficient/geometry instability; it does **not** identify the internal Sionna stage responsible for those records or prove a general solver defect. The largest finite budget is not truth, City43 is one procedural scene family, and this is not a qualification experiment. G3 remains open; production D1 RT generation stays gated. See the [hash-bound raw report](docs/research/results/rt-city43-center-los-seed23-depth10-s0-path-attribution-100k-vs-1m-20261022.json.gz), [attribution audit](docs/research/results/rt-city43-center-los-seed23-depth10-s0-path-attribution-audit-20261022.json), and [reproducible auditor](scripts/audit_rt_path_budget_attribution.py).

**Edition 3.190 City41/City42 matched budget stress test (G3.02/G3.04 partial evidence):** Repeated the City43 six-link, three-site geometry and 20-seed 100k/1M samples/source depth-3 comparison on City41 and City42, two distinct seeded layouts from the same generator. Independent Mitsuba segment queries verified all six labels per city (three visible, three blocked). The 600-run reports use native ITU placeholder materials with scattering coefficients 0.0, contain no diffuse paths, and do not fill the 1M candidate cap. Across 60 matched link-seed pairs per class, median complex-CFR relative-L2 errors are 2.20e-6 LOS / 3.85e-5 NLOS (City41) and 9.13e-8 LOS / 6.94e-5 NLOS (City42). Tail behavior exposes localized NLOS budget sensitivity: City41 center-NLOS has 13.79% maximum error and an extra 370.209 ns path at 1M in 8/20 seeds; City42 west-/center-NLOS have extra late paths in 4/20 and 2/20 seeds and maximum errors 2.51%/3.48%. LOS path counts do not change. This shows City43 is not the only procedural layout with a budget-sensitive NLOS tail, but these are same-generator variants rather than independent sites, zero scattering leaves diffuse convergence untested, the largest budget is not truth, and no acceptance tolerance is set. G3.01/G3.02/G3.04 remain open. Provenance-bound raw reports, paired-seed audits, scene hashes and exact link geometry are recorded in `docs/research/results/rt-city41-spatial-budget-depth3-native-materials-6links-{100k-1m,depth3-distribution-audit}-20seeds-20261022.json` and the corresponding City42 files.

**Edition 3.189 paired complex-CFR distribution audit (G3.02 partial evidence):** Extended the paired-seed auditor to include the solver’s same-seed complex-CFR relative-L2 error against the 1M-sample/source, depth-3 reference. Across 20 seeds and all six City43 LOS/NLOS links, native zero-scattering materials yield median errors of 3.13e-7 LOS (95% seed-bootstrap interval [9.78e-8, 9.27e-7]) and 1.71e-5 NLOS ([3.49e-6, 3.44e-5]). With the uncalibrated S=0.5 overrides, the corresponding errors rise to 0.264 LOS ([0.236, 0.301]) and 0.638 NLOS ([0.541, 0.767]). This directly bounds one complex-channel observable at depth 3 for the selected finite budgets, but the 1M result is not ground truth and the result is specific to one scene and material settings. The compact S=0.5 artifact stores the per-run CFR error and hashes for the omitted path-delay arrays; complex CFR vectors themselves are not included. G3.02/G3.04 remain open pending independent-scene evidence and declared admission tolerances. Updated audits: `docs/research/results/rt-city43-spatial-budget-depth3-native-materials-6links-depth3-distribution-audit-20seeds-20261022.json` and `docs/research/results/rt-city43-spatial-budget-depth3-s05-depth3-distribution-audit-20seeds-20261022.json`.

**Edition 3.188 City43 20-seed native-material control (G3.02/G3.04 partial evidence):** Added a same-scene comparison of 100k versus 1M samples/source at depth 3 for the six west/center/east LOS/NLOS links, 20 matched seeds, and the same 16-building City43 scene. The native ITU concrete/glass/metal scattering coefficients are recorded as 0.0; the 600-run report contains zero diffuse paths and no cap-filled cases. Across 120 paired link-seed cases, non-diffuse path counts are unchanged; median total RMS-delay shifts are +0.000005 ns NLOS and −0.000002 ns LOS, with intervals spanning numerical zero. The matched S=0.5 report also has 120 pairs: NLOS diffuse RMS-delay median shift is −0.664 ns (95% seed-bootstrap interval [−12.757, +6.770] ns), and LOS is +0.419 ns ([−1.277, +1.198] ns), so this City43 depth-3 matrix does not resolve diffuse-delay budget stability. Total RMS-delay shifts are +0.183 ns NLOS (interval [−1.207, +0.982]) and −0.548 ns LOS ([−0.683, −0.203]). This is a finite solver-budget comparison, not a truth claim; the S=0.5 value is an uncalibrated intervention, seed intervals do not cover scene/material uncertainty, and no admission tolerance is applied. The committed S=0.5 summary retains all run-level observables and per-run hashes for omitted sorted path-delay arrays, bound to the original full-report hash. G3.02/G3.04 remain open. Reports: [native-material report](docs/research/results/rt-city43-spatial-budget-depth3-native-materials-6links-100k-1m-20seeds-20261022.json.gz), [native audit](docs/research/results/rt-city43-spatial-budget-depth3-native-materials-6links-depth3-distribution-audit-20seeds-20261022.json), [S=0.5 report](docs/research/results/rt-city43-spatial-budget-depth3-s05-100k-1m-20seeds-20261022-summary.json.gz), and [S=0.5 audit](docs/research/results/rt-city43-spatial-budget-depth3-s05-depth3-distribution-audit-20seeds-20261022.json).


**Edition 3.178 G3.02 ray-budget follow-up across three City43 TX sites:** Added a 100k-sample/source comparison for the same six audited links, three seeds and interaction depths 0–10. Although per-seed CFR changes from depth 9→10 at 100k are below 0.9% for the selected NLOS links, changing the budget from 20k to 100k changes their depth-10 CFR by median relative L2 error 0.71 (range 0.24–1.14). LOS budget error is median 0.76 (range 0.65–0.92), and valid path counts grow approximately 4.4–5.0×. Thus fixed-budget apparent NLOS depth stability does not establish stable output across solver sample budgets. The finite 100k result is not ground truth; both G3.01 and G3.02 remain open pending ray-budget-aware tolerances and broader scenes/materials. See docs/research/results/rt-city43-spatial-los-nlos-depth10-s05-20k-100k-20261022.json, the [paired depth/budget plot](docs/research/figures/rt-city43-spatial-los-nlos-depth10-s05-20k-100k-20261022.png), and the [3D building holdout scene](docs/research/figures/rt-urban43-spatial-holdout-3d-20261022.png).

**Edition 3.177 G3.01 multi-site interaction-depth follow-up:** Extended the convergence diagnostic from a single TX site to six audited LOS/NLOS links across all three fixed City43 TX sites. With three seeds and 20k samples/source, NLOS CFR/depth summaries are nearly unchanged from depth 9→10 for this selected path budget, but LOS matched CFR error has a 15.4% median and reaches 45.3%; weighted LOS RMS delay shifts as much as 22.0%. Path counts continue growing and depth 10 remains a finite, unqualified reference. This provides broader synthetic geometry evidence, not convergence or material validation. Keep G3.01 open until both LOS and NLOS target observables meet predeclared, ray-budget-aware tolerances across independent scene families. See docs/research/results/rt-city43-spatial-los-nlos-depth10-s05-20k-20261022.json, the [depth plot](docs/research/figures/rt-city43-spatial-los-nlos-depth10-s05-20k-20261022.png), and the [3D building holdout scene](docs/research/figures/rt-urban43-spatial-holdout-3d-20261022.png).

**Edition 3.176 G3.09 weak-tail sensitivity evidence:** A committed post-solver audit compared all valid-path absolute-delay CFR with hypothetical per-link relative coefficient-power cutoffs on the City43 blocked building link for three LLVM seeds. At −20 dB only 15–23 of 335–349 valid paths remain and coherent CFR error reaches 43.4–75.3%; at −40 dB 99.85–99.88% of summed coefficient power remains but CFR error reaches 2.30–6.65%. Therefore path-power sum alone is insufficient to validate discarded weak paths. The audit is a diagnostic only: canonical generation retains all solver-valid paths, and these coefficients are not a receiver detectability threshold. G3.09 remains open until tolerance criteria and broader scene/material/budget evidence are prespecified. See `docs/research/results/rt-city43-nlos-path-power-threshold-20261022.json`, `docs/research/PHYSICS_VERIFICATION_STATUS.md` edition 3.182, and the [3D building visualization](docs/research/figures/rt-city43-los-nlos-capstress-depth3-s05-20k-scene-20261022.png).

1. Create **UMCD-1** as a logical dataset comprised of **nine data families** under one data contract: analytical, standardized stochastic, calibrated geometric RT, dynamic RT, counterfactual RT, NR PHY/receiver, adverse-shift/stress, measured external, and held-out external/field validation. Families have different truth semantics and permissible tasks.
2. Use a **canonical atomic unit** `network_propagation_episode` = scene + geometry/material realization + multi-node topology + frequency-band configuration + full synchronized time interval. One network episode owns **many directed TX→RX child links** with potentially unequal antenna/port counts. Immutable link-path truth spawns multiple waveform, noise and receiver views. A snapshot, link, or pilot variant is not an independent sample when they share the parent site, scene, topology, route, or episode.
3. Store **path oracle / true channel / observable receiver measurement / model input / labels / provenance** as separate access-controlled logical layers. Keep variable-size paths and unequal axis shapes native; do not squeeze to `[N,F]` prematurely.
4. Support **SISO / SIMO / MISO / MIMO / massive MIMO, SU-/MU-MIMO, multi-BS/multi-UE, many active interferers, uplink/downlink, dual polarization and arrays of unequal sizes** through one explicit node–array–link graph; distinguish native Sionna support from constructed multi-link approximations. Prioritize a **physically credible urban/indoor RT benchmark** with LOS containing many reflected paths, NLOS with strong indirect components, intermittently blocked links, path disappearance, outages, accurate materials and antenna effects. Add outdoor-to-indoor/industrial/mobility only after scene QA.
5. Make **future trajectory, uncertainty and physical observability** first-class: moving UE and moving obstacles retraced; masks for absolute timing, angle, Doppler, path order and availability. Short-time Doppler extrapolation alone is **not** a blockage simulator.
6. Keep TDL/CDL-A/B/C vs D/E labels explicitly as **profile category**, not ground-truth geometric LOS. Use 3GPP scenario-aware families (UMi/UMa/RMa/InH/InF where supported/validated) in separate partitions rather than relabeling them as geometry-grounded RT.
7. Generate an **un-normalized absolute-delay physical channel** as primary, then optional normalized, common-gauge derived views. All power, path loss, ToA and direct/indirect decomposition calculations require consistent conventions.
8. Freeze split units and scene families **before generation**; hold out independent cities/buildings/material families, trajectories, receiver hardware and measured domains. Use paired observations only inside one split.
9. The first implementation milestone is a **native Sionna conformance and path-conservation gate**. Do not scale to millions of frames before this gate passes.
10. Treat every other model and label as a consumer of this corpus. The dataset design must outlive the first proposed CNN/MLP/physics-informed paper.

---

## 1. Objective, scope, promises and non-promises

### 1.1 Research supported by one dataset

The umbrella must support at least:

- Detect physical geometric direct-path visibility (LOS/NLOS), detectable LOS, weak-LOS, blocked, and outage as **different** concepts.
- Infer observable channel statistics: RMS/mean excess delay, coherence bandwidth estimates, PDP/CIR, K-like specular/diffuse ratios, path loss and shadow-fading residual where well defined.
- Estimate multipath sets: resolvable component delays/gains, directions, Doppler, number and uncertainty. Distinguish solver ray count, physical clusters, and resolvable model order.
- Reconstruct/noise-reduce/interpolate/extrapolate **complex CSI** across sparse subcarriers, symbols and antennas.
- Estimate location/range/relative timing/angles **when synchronization, geometry and arrays make them identifiable**; record gauge and reference clocks.
- Forecast future CSI, delay spread, dominant directions, LOS/detectability, blockage, path-appearance/disappearance and received power over specified horizons.
- Evaluate beam selection, beam failure, outage, BLER, positioning, and possibly scheduling based on genuinely receiver-accessible inputs and identical link assumptions.
- Study transfer between TDL/CDL, 3GPP scenario models, geometry RT, different RT solvers/backends, hardware, measured data, and distribution shifts.
- Study complex-valued phase symmetry, nuisance-invariant inference, counterfactual obstruction, physical reconstruction, and calibrated abstention.
- Permit optional ISAC/RCS targets only with an independent sensing-waveform/target model and truth contract; don't treat RCS as inherently observed from ordinary communication CSI.

### 1.2 Scope boundaries / scientific honesty

- One 'full' corpus **does not mean every record has every field**. A `caez` measured-magnitude example has **no geometric path oracle** and must never be assigned one. Masks are not a license to invent labels.
- Simulation path identity and material ground truth are **simulator truth**, not independently measured physical truth.
- A physically LOS channel can contain hundreds of indirect paths; a physically NLOS channel may still have a strong single reflected/diffracted component.
- The number of returned diffuse RT rays depends strongly on Monte Carlo sampling, so never label it a physical object-independent multipath count.
- Conditional labels must be interpreted with antenna pattern, link budget, synchronization, detection threshold, bandwidth and observation window.
- Absolute ToA and absolute carrier phase may be unidentifiable if timing and phase references are not supplied.
- Forecasting data needs real time order and causal inputs. Repeated snapshots of fixed channel are **not** temporal prediction data.
- No zero-path links are silently discarded. 'No valid simulated rays' can represent either a true physical outage **or** a solver miss; validate and label them separately.

### 1.3 Evidence ladder

`analytical invariant` → `standard model table conformance` → `native RT numerical convergence` → `diverse physically plausible scenes` → `NR receiver correctness` → `independent held-out scenes` → `measured transfer` → `cross-site/hardware repeatability`.

A machine-learning metric on stage 3 data cannot be presented as stage 7 evidence. Each manuscript figure must cite its evidence rung.

---

## 2. Audited starting point: what to reuse and what to change

This section describes code observed in the **uploaded archive**, not fresh remote GitHub commits. File paths are relative to `5g-nr-channel-condition-main/`.

| Existing module/artifact | Retain/reuse | Required extension / risk |
|---|---|---|
| `datasets/nr.py` (`NRObservation`, `NRChannelRecord`, `TargetDefinition`, `NRTrajectory`) | Explicit receiver-input vs target isolation and typed axes | Generalize to multi-link, multiple observation modalities, long sequences, per-target observability and provenance confidence; represent time/trajectory joins. |
| `datasets/nr_store.py` | Immutable hash-verified record manifests, pickle-free loads | Chunked columnar/object storage for massive variable-cardinality path tables and multidimensional arrays; schema migrations and partial/recovery transactions. |
| `integrations/sionna/rt_trajectory.py` | Retracing positions, timestamped DMRS records, run-identity cache | Full per-path snapshots and time-varying geometry; expose MIMO arrays; add world/object motion, nontrivial obstruction; preserve physical time references and empty-path outcomes. Default config currently small (4 steps, 6 PRBs, modest ray budget): debug only. |
| `integrations/sionna/rt_to_nr.py` | Bridges RT `Paths.cfr()` to NR resource-grid frequencies; defaults `normalize_delays=False` | End-to-end invariants: subcarrier ordering, absolute-delay phase, antenna axes, OFDM symbols vs physical clock, and waveform link budget. |
| `integrations/sionna/adapter.py` | Profile generation, Sionna wrapping, RT adapters | Legacy `rt_cfr` uses `normalize_delays=True`: prohibit this for canonical ToA or full-minus-direct decomposition; unify conversion and phase conventions. |
| `integrations/sionna/rt_dmrs.py` | NR PUSCH/DMRS observation bridge, masks | Explicit comment says per-link signal normalization removes path-loss information. Preserve separate absolute-power and normalized diagnostic views. |
| `integrations/sionna/backend.py` | CUDA→LLVM native probe isolated in child process | Pin backend in each manifest; compare numerical convergence; never silently substitute LLVM/CUDA for a frozen experiment. |
| `integrations/sionna/rt_identity.py`, `scene_assets.py`, `rt_cache.py` | Source/scene/runtime hashes, backend identity, immutable artifacts | Include runtime-edited radio-material state, dynamic object trajectories, calibration fits and external assets, not merely XML/OBJ files. |
| `evaluation/independent_splits.py`, `spatial_audit.py`, `duplicate_audit.py` | Group-disjoint split locks, physical buffer and correlation checks | Add scene-asset-parent, route/time adjacency, counterfactual family, material-realization, antenna/receiver capture and external-site lineage to split DAG. |
| `models/observable_multipath.py`, identifiability experiments | Two-path model, bounded delay, fixed-gauge uncertainty and classical comparisons | Build on it; do not call 2-path simulations a full variable-cardinality channel-grounding result. |
| `docs/research/OUTPUT_TASKS_AND_CHANNELS.md` | Critical truthful separation of output types | Expand to unified task/label registry and include strong negative/abstention and unobservable targets. |
| `docs/research/MEASURED_DATA_CANDIDATES.md`, `datasets/caez.py` | CAEZ-5G license/field audit and measured magnitude forecast | Magnitude-only official local archives **cannot** validate complex phase, AoA, absolute ToA, path count or geometric LOS; maintain separate measured view and licensing. |
| `scenes/generated/research_scenes`, six-scene configs, pilot scripts | Reproducible initial geometry families | Validate physics/material diversity, scene independence, spatial density, near-boundary labels and numeric convergence before promotion to benchmark. |
| `docs/research/results/*` | Existing exploratory outputs | Re-run/audit raw data hashes and versions; result JSON filenames in uploaded archive include dates beyond this planning date. **Filename date is not verification of execution/provenance.** No undocumented result promotion. |

**Immediate corrections:** (a) split **source profile category** from **geometric LOS**; (b) never subtract independently delay-normalized channels; (c) store true absolute power; (d) store outages and solver failures separately; (e) extend single-link/snapshot, small-PUSCH records to large, synchronized MIMO-time arrays; (f) preserve native path metadata *before* conversion to CFR/NR.

### 2.1 Sionna source facts verified against official 2.2 documentation

- Sionna 2.2 separates **Sionna RT**, **PHY**, **SYS**; NVIDIA RT source is https://github.com/NVlabs/sionna-rt and PHY source https://github.com/NVlabs/sionna.
- `PathSolver(deterministic=True)` accepts `max_depth`, `max_num_paths_per_src`, `samples_per_src`, `los`, `specular_reflection`, `diffuse_reflection`, `refraction`, `diffraction`, `edge_diffraction`, `diffraction_lit_region`, `synthetic_array`, `seed`. Fixed seed alone need not yield deterministic results without the deterministic setting.
- RT supports LoS, specular/diffuse reflection, transmission/refraction, and **first-order edge/wedge diffraction**. Higher-order diffraction is not established by the documented solver. Transmission uses a thin-slab/no-angular-deflection approximation that may be inaccurate for thick structures.
- RT synthetic arrays apply synthetic antenna phase shifts about a common traced location; validate against `synthetic_array=False` on representative array/near-field samples.
- `Paths.cfr()` and `Paths.cir()` default `normalize_delays=True`; `Paths.cfr()` defaults `normalize=False`. For physical timing/power, **set both explicitly** (`normalize_delays=False`, `normalize=False`) and test the underlying `Paths.a`/`Paths.tau` representation and carrier-phase conventions.
- Sionna RT supports path-wise delays, angles, Doppler and interaction metadata; record only native fields actually exposed by the installed pinned version, and derive unavailable fields explicitly with a documented algorithm rather than inventing them.
- Short-distance/time motion can be approximated using path Doppler; path births/deaths and blockade changes require **moving geometry and re-solving** over time.
- Diffuse scattering requires **nonzero configured scattering coefficients**, and sampled path counts can increase with `samples_per_src`; raw count is not a stable physical target.
- Sionna PHY 2.2 has TDL/CDL and additional 3GPP scenario models, and NR PUSCH transmitter/receiver and LS/LMMSE estimators. TDL/CDL `spec_version='19.2'` is the API table selector documented for profiles. Record ETSI/3GPP technical-spec document revision independently.
- Native Sionna RT performance and API behavior **must still be tested on the target machine**. This document was prepared from source/docs and inspected project code, not a fresh native simulator run.

**Canonical sources:**

- https://nvlabs.github.io/sionna/rt/api/paths_solvers.html
- https://nvlabs.github.io/sionna/rt/api/paths.html
- https://nvlabs.github.io/sionna/rt/developer/dev_understanding_paths.html
- https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html
- https://nvlabs.github.io/sionna/rt/tutorials/Scattering.html
- https://nvlabs.github.io/sionna/rt/developer/dev_custom_radio_materials.html
- https://nvlabs.github.io/sionna/v2.2.0/phy/api/nr/pusch.html
- https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/tdl.html
- https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/cdl.html
- https://github.com/NVlabs/sionna-rt/releases/tag/v2.2.0
- https://github.com/NVlabs/sionna/tree/main/src/sionna/phy

---

## 3. One *logical* superdataset, not a physically incorrect homogeneous one

### 3.1 Parent entity and linked views

```text
UMCD-1
  registry + split locks + vocabulary + version + provenance ledger
  ├─ scene catalog (geometry, material priors, calibration, assets, coordinate frames)
  ├─ episode catalog (scene realization × link × TX/RX × route × time interval)
  │    ├─ G0 geometry/visibility oracle (restricted)
  │    ├─ G1 raw RT paths and interactions (restricted)
  │    ├─ G2 physically scaled true channel H(f,t,rx,tx,pol) (restricted during inference)
  │    ├─ G3 NR PHY transmitted/received waveform & DMRS/PRS + RF impairments
  │    ├─ G4 receiver-only CSI/quality/noise/pilot observations (allowed model inputs)
  │    ├─ G5 precomputed target definitions + masks + uncertainty tags
  │    ├─ G6 multi-horizon future state and counterfactual families (restricted targets)
  │    └─ G7 QA diagnostics, convergence, provenance, matched replicates
  ├─ analytical and 3GPP stochastic episodes (different oracle semantics)
  ├─ independent measured campaigns (field-specific observability and license)
  └─ immutable benchmark views (train/val/cal/test/hidden test, task-specific)
```

**Do not duplicate G0–G2 for every noise realization.** One physical episode may have many waveform/receiver child observations, identified by `(parent_episode_id, waveform_id, noise_seed, impairment_seed)`. All descendants inherit the parent's split, even if channel statistical features look different.

### 3.2 Nine constituent dataset families (one registry)

| Family ID | Generation / acquisition | Mandatory truth | Intended scientific use | Restrictions |
|---|---|---|---|---|
| **F00-ANALYTICAL** | Closed-form free space, 0–K paths, exact 2-path, controlled arrays and gauges | Exact injected delays/gains/angles, known transformations | Physics unit tests, identifiability, CRLB conditions, counterfactual method debugging | Not credible evidence of real urban RF generalization. |
| **F01-TDLCDL** | 3GPP TDL-A–E/CDL-A–E across delays/speeds/arrays and frequencies | Simulator tap/cluster params, normalized power profiles, profile identity | Standards compliance, statistical-domain pretraining, cross-profile robustness | **Profile LOS category is not geometric visibility.** Cluster count is not RT ray count. |
| **F02-3GPP-SCENARIO** | 3GPP UMi/UMa/RMa/InH/InF models when version-compatible | 3GPP scenario LOS state and large-/small-scale params as provided | Realistic stochastic prior; scenario shifts; compare to RT | 3GPP LOS state is model condition, not a verified geometric ray result. |
| **F03-RT-STATIC** | Realistic independent city/indoor/industrial geometry and material scenes | Geometric direct-ray oracle, all solved path metadata, absolute CIR/CFR | Main LOS, delay/power/angle state inference, held-out scenes | Solver-completeness and material realism must be audited. |
| **F04-RT-DYNAMIC** | Retraced moving TX/RX/obstacles and sequences | Time-stamped geometry, path births/deaths, true channels, blockage transitions | Future CSI/blockage/beam, Doppler, uncertainty | Do not treat Doppler-only phase evolution as geometry-change truth. |
| **F05-RT-INTERVENTIONS** | Matched changes to real blocker, TX/UE pose, material/antenna | Before/after independent physical solves, intervention key | Counterfactual path reasoning and causality claims | Removing a direct coefficient is a **mathematical** intervention, not a real blockage. Keep separate. |
| **F06-NR-OBSERVATIONS** | PUSCH/DMRS, optional PRS, raw IQ/OFDM grid, synchronization/channel estimators on F00–F05 | Known transmitted symbols and injected impairments; parent physical truth | Receiver realism, noisy estimation, BLER/beam selection | Label and model-input access separate; varying SNR views share split. |
| **F07-STRESS/OOD** | Deliberate shifts: unseen materials, bandwidth, antennas, clock, interference, diffuse settings, outages | Parent/oracle truth with shift category | Robustness, calibration, failure analysis, worst-group performance | Stress configurations should be prespecified, not tuned against final test. |
| **F08-MEASURED/EXTERNAL** | CAEZ-5G and future legally obtained hardware datasets | Only measured fields/quality and documented reference labels | External plausibility, CSI forecasting, calibration, limited real-world transfer | Not every measured sample has phase, geometric LOS, raw paths or reliable geometry. License may prohibit redistribution. |

**Optional `F09-ISAC` extension:** radar/RCS target families using *Sionna RT's sensing/RCS workflow* require a different waveform, monostatic/bistatic geometry, object-scatterer truth, and sensing-specific metrics. Register only once a separate native pipeline is validated; it must not contaminate communication-only labels. The nine families above constitute UMCD-1 initial scope.

### 3.3 Independence hierarchy — mandatory split groups

`asset_family → independent site/city/building → material calibration/realization → scene version → physical trajectory/session → link geometry → parent episode → time frame → RF/noise view`.

The independent statistical unit for cross-scene claims is at least **scene/site**, not channel frame. For temporal performance, use independent sequences with block/bootstrap uncertainty. Counterfactual pairs, frequency-band variants and alternate receiver observations of the same physical realization must never straddle train vs test.

### 3.4 Train-time exposure policies

- **Observable-only:** received IQ/grid, estimated CSI, estimated noise, available pilots/timing, known hardware and system settings.
- **Simulation supervision:** target-specific oracle values (LOS/path/geometry) made available **only** as training loss labels for specified supervised experiments; never fed as inference features.
- **Teacher-oracle pretraining:** private full-path information permitted to teacher; student always evaluated receiver-only. Freeze protocol and label source.
- **Oracle upper-bound:** perfect CSI or full paths allowed **only** in results explicitly named oracle and not compared as ordinary fair baselines.
- **Measured deployment:** no scene assets or future path labels, unless application explicitly assumes a preloaded radio map and tracks its availability.

---

## 4. Canonical storage schema v1.0: entities, fields, axes and types

### 4.1 Design constraints

- Immutable data objects with SHA-256 and manifest DAG; software versions and generator configuration stored independently of filenames.
- Explicit **SI units** and angle conventions: m, s, Hz, rad, W, dBm; if decibels used, name reference quantity. No arbitrary 'normalized' field without normalization recipe.
- Coordinate frames: world XYZ convention, altitude datum, antenna-local basis, polarization frame, AoA/AoD azimuth and elevation/zenith convention. Store transformation matrices.
- **Complex information:** store real/imag components with exact dtype/endian or complex dtype supported safely by chosen container. `float32` real/imag physical channels default; selected analytic precision studies use `float64` and track numeric error.
- Ragged path cardinalities must not be silently padded without masks; use `path_offset`, `path_length` and columnar path tables or Arrow/Parquet lists.
- Axes are named by schema (`time`, `rx`, `rx_element`, `rx_pol`, `tx`, `tx_element`, `tx_pol`, `fft_bin`, `symbol`, `path`); no axis inferred from array rank alone.
- Frequencies: store **center/carrier** `f_c_hz`, **baseband subcarrier offsets** `f_baseband_hz`, true pilot mask and effective FFT-bin indices, DC/null/guard mappings.
- Signal time: UTC/reference capture time where meaningful, monotonic frame offsets, OFDM symbol time, PLL/clock reference, group-delay/propagation time conventions.
- Provenance confidence: `analytic_exact`, `3gpp_model_defined`, `rt_solver_truth`, `geometry_verified`, `calibrated_rt`, `measured`, `estimated`, `unavailable` — never imply model truth = independent experimental truth.
- Physically non-applicable fields should be null/masked with a **reason code**, not zero. A numerical zero is a value, not 'unknown'.
- Store **absolute-power** canonical channels and optional separately tagged normalized copies. Never destroy original path-loss information.

### 4.2 Primary tables and key columns

| Table | Key | Fields (minimum) | Private? |
|---|---|---|---|
| `dataset_release` | release_id | semver, build timestamp, license set, source hashes, code lock, generators, QA status | public metadata |
| `scene` | scene_id + scene_hash | source/drawing/meshes/material IDs, coordinate frame, size/bounds, realism audit, physics-assumption flags | some assets restricted |
| `scene_realization` | realization_id | geometry perturbation seed, object roster, calibrated material draw, weather, TX configuration, hash | private truth |
| `episode` | episode_id | parent scene ID, **network topology and directed link graph**, time range, band/carrier plan, split group, source family | split metadata restricted |
| `network_node` | node_id | BS/gNB/UE/relay/interferer, RF chain count, antenna array, port map, polarization, calibration, clocks, trajectory, power envelope | node physics/provenance |
| `directed_link` | link_id | episode ID, TX node/port set, RX node/port set, active state, serving/interfering/sounding role, uplink/downlink, band, TDD/FDD timing | restricted link truth |
| `link_association` | link_id,time | serving/neighbor BS, scheduling/user-stream ID, precoder, combiner, beam codebook, handover event; can have multiple serving nodes | signal schedule |
| `array_definition` | array_id | element coordinates, polarization basis, element pattern, orientation, spacing, coupling/calibration approximation, aperture, RF-chain mapping | array engineering |
| `pose_timeline` | episode_id,time | TX/RX/object pose, velocity, acceleration, clock, orientation, occlusion events and timestamp | private oracle |
| `rt_solver_run` | solve_id | solver seed/deterministic flag, path budget/depth, interaction flags, backend, versions, runtime fingerprint, convergence stats | provenance |
| `path_table` | episode_id, frame, tx, rx, path_index | valid flag; absolute delay; complex path amplitude; Doppler; AoA/AoD; interactions; per-path type; vertices/object/material IDs when exposed; solver sampling weight | private oracle |
| `channel_tensor` | channel_view_id | array URI, axis names, frequencies/time, complex CIR/CFR, element-wise validity mask, absolute scaling, reference gauge, aggregation flags | oracle and receiver-target copies separated |
| `waveform` | waveform_id | 38.211/214 assumptions, carrier/numerology, BWP/PRBs, DMRS/PRS configuration, Tx symbols, channel application domain, CP/FFT details | source metadata / pilots; transmitted payload private as needed |
| `receiver_view` | observation_id | parent episode, received IQ/OFDM grid, estimated CSI, error variance, pilot mask, synchronization/beam state, observation-quality mask, link-budget params | model-visible fields only |
| `impairment` | observation_id | thermal/interference noise, CFO, timing offset, IQ/PA/ADC, phase noise, EVM, AGC, seed, noise covariance, receiver calibration | permitted config or truth according to task |
| `targets` | target_set_id | semantic name/version/value/unit/reference, definition SHA, validity mask, unavailable reason, oracle algorithm and confidence | private until training/evaluation API |
| `counterfactual_pair` | intervention_id | base scene/route ID, changed objects/materials/pose, parent/child episode IDs, intervention category and causal scope | private linkage |
| `quality_report` | report_id | invariant results, convergence, near-boundary oracle audit, invalid reasons, approximation diagnostics | public summaries |
| `split_assignment` | split_id + group_id | locked assignment, hierarchical groups, leakage/proximity buffers, holdout rationale and signature | dataset API enforces |
| `measured_source` | campaign_id | site/capture device, original licensing, fields, timestamps/clock, published citations, withheld or unavailable labels | access controlled |

### 4.3 Minimal versioned episode manifest (illustrative JSON; not executed)

```json
{
  "schema": "umcd.episode.v1",
  "episode_id": "sha256:<content-addressed-key>",
  "family": "F04-RT-DYNAMIC",
  "scene_realization_id": "sceneXYZ-realization-005",
  "timeline_s": [0.0, 0.02, 0.04],
  "carrier_hz": 3500000000.0,
  "axes": {
    "canonical_cfr": ["frame", "rx", "rx_element", "tx", "tx_element", "freq"]
  },
  "frequency_axis_uri": "arrays/freq/<hash>",
  "parent_oracle_uri": "restricted/path_tables/<hash>",
  "channel_uri": "restricted/physical_cfr/<hash>",
  "receiver_observations_uri": "receiver_views/<hash>",
  "target_registry_uri": "restricted/targets/<hash>",
  "validity_table_uri": "restricted/validity/<hash>",
  "normalization": {"delay": false, "power": false},
  "clock_reference": "TX-referenced propagation model, no unknown hardware clock offset",
  "solver_run_id": "<hash>",
  "split_group_id": "siteXYZ-session005",
  "labels_not_in_model_features": true
}
```

Hashes and paths above are placeholders **only for schema illustration**, not actual generated data.

### 4.4 Storage decision

- `Parquet/Arrow` for immutable catalogs, targets, sparse path tables, categorical provenance, joins; `Zarr` (v2 or v3 chosen after actual interoperability test and frozen) for chunked multi-dimensional CSI/IQ/time arrays. Prefer one pinned safe implementation; do not arbitrarily swap backends during data generation.
- Keep raw scene geometries (`XML/OBJ/MTL/texture`) and calibration sidecars content-addressed. A scene hash alone is insufficient if runtime material mutation or code changes are not included.
- Partition object stores by release / family / site / realization / episode; chunk along time and receiver/sample axes for streaming. Avoid millions of tiny `.npz` files for production; existing `.npz` adapters remain useful for test fixtures.
- Use manifests with two-phase/atomic publication (temporary then checksum verified then committed), unique processing states, retries and idempotent jobs.
- Support compression **without lossy channel quantization** for canonical oracle; if data-reduced view uses compression/quantization, it is a separate named observation with distortion calibration.
- Keep explicit license/retention/sensitive-location access controls for captured real RF data. Do not redistribute CAEZ subsets if license forbids it.

### 4.5 Canonical complex channel convention — non-negotiable

For path delays `tau_p` and passband coefficient `a_p` at carrier `f_c`, a conceptual baseband channel is

`H(f_offset,t) = Σ_p a_baseband,p(t) * exp(-j*2π*f_offset*tau_p)`

where the precise carrier phase relation between `a_p` and `a_baseband,p` must follow the **pinned Sionna RT** `Paths.cir()` / `.cfr()` implementation; do not multiply carrier phase twice. In Sionna 2.2 documentation `a_baseband,p(0) = a_p exp(-j2π f_c tau_p)`. Frequencies passed to `cfr` are **baseband offsets in Hz**, not an arbitrary absolute RF-frequency axis.

**Mandatory invariant:** reconstruct a random checked subset manually from the **same** `Paths` object and match `Paths.cfr(...,normalize_delays=False,normalize=False)` within a precision-dependent tolerance measured against reference. For path-coefficient subsets, keep **the identical delay origin and coefficient convention** across all components. Separate full/LOS solves can change sampled indirect paths and cannot automatically be assumed to be exact additive decompositions.

---
## 5. Full target/label registry (what must be kept and what is actually identifiable)

All targets are defined **per link/antenna/polarization/time/horizon where relevant**, not simply per episode. Every label has: `name`, `unit`, `definition_version`, `source`, `gauge`, `time_reference`, `valid_mask`, `uncertainty/quality`, `visibility/access`, `supported_family`, `aggregation_scope`, `evaluator`, `null_reason`.

**Important:** the table below is the *required registry* for possible research tasks, **not** a claim that every value can or should be estimated from every receiver snapshot. The usable target subset depends on bandwidth, aperture, temporal baseline, clock synchronization, link budget and ground-truth availability.

### 5.1 Propagation category and detection targets

| ID | Target | Definition / reference | Validity / caution |
|---|---|---|---|
| L01 | `geometric_los` | Independently verified **unobstructed straight TX↔RX segment** including antenna location offsets; boolean. | RT/known geometry only; distinguish material transmission from unobstructed space. Near-mesh-edge ambiguous class flagged. |
| L02 | `direct_path_solver_present` | Native `Paths` contains valid depth-zero direct path. | Solver output diagnostic, not substitute for L01. Disagreements require retrace. |
| L03 | `direct_path_received_power_w` | Power from direct component under known antenna/polarization/link budget and common channel reference. | Zero for independently verified blocked; null if path extraction incomplete. |
| L04 | `direct_to_indirect_ratio_db` | Ratio of defined direct component power to remaining physical multipath power **in the chosen coherent or path-power convention**. | Preserve exact convention and receiver aperture; never use independently shifted delay origins. |
| L05 | `detectable_direct_path` | Direct contribution detectable under a **frozen specified detector, bandwidth, SNR and false-alarm criterion**. | Detector-specific; can be false despite geometric LOS. |
| L06 | `dominant_component_type` | Classification of highest-power **resolvable** component (direct/specular/diffuse/diffracted/transmitted/compound/unknown). | Observation resolution & composite paths may make ambiguous. |
| L07 | `channel_coverage` | Received power or SINR exceeds prespecified service threshold. | Receiver/link-budget dependent; not same as geometric NLOS. |
| L08 | `outage_type` | `physical_no_coverage`, `insufficient_snr`, `decoder_failure`, `solver_incomplete`, `missing_capture`, `valid_service`. | Carefully separate missing simulation from physically bad channel. |
| L09 | `transition_state` | LOS→NLOS, NLOS→LOS, blockage onset/offset, reappearance. | Requires *retraced ordered snapshots*, per event tolerance and reference. |
| L10 | `blocker_identity` | Physical object category/instance causing direct-ray obstruction. | Oracle-only, opt-in privacy; not necessarily inferable from CSI. |

### 5.2 Power, channel quality and delay statistics

| ID | Target (units) | Definition | Caveat |
|---|---|---|---|
| P01 | absolute received power (`W`, `dBm`) | Specified TX power, pattern/polarization, loss chain, bandwidth and reference plane. | Cannot recover after per-link unit-power normalization. |
| P02 | path gain/path loss (`linear`, `dB`) | `P_rx/P_tx` or negative dB; array-combining convention explicit. | Must not mix beamformed and element-level gain. |
| P03 | shadow-fading residual (`dB`) | Residual relative to **declared fitted/reference path-loss model**. | Not an intrinsic ray-tracer property. Fit model on train-only. |
| P04 | SNR, SINR, RSSI, RSRP/RSRQ (`dB`,`dBm`) | Receiver-, signal- and bandwidth-specific definitions. | NR reference-signal vs PUSCH measurement not interchangeable. |
| P05 | link capacity/spectral efficiency (`bit/s/Hz`) | Perfect/estimated CSI, precoder/power, bandwidth and interference assumptions recorded. | Oracle vs realizable receiver outcomes must be separated. |
| D01 | earliest *physical* path delay (`s`) | `min(tau_p)` over validated significant geometric ray paths, absolute TX timing origin. | RT numerical path threshold and solver completeness matter. |
| D02 | earliest *detectable* arrival (`s`) | First path/tap above defined detector or threshold in band-limited received signal. | Often later than D01; detector and sampling dependent. |
| D03 | relative first-path delay (`s`) | Delay to a declared synchronization/trigger/reference path. | Identifiable without absolute reference only up to gauge. |
| D04 | mean excess delay (`s`) | Power-weighted mean of delays relative to declared path/window origin. | Whether coherent overlap is considered must be specified. |
| D05 | RMS delay spread (`s`) | `sqrt(Σp w_p(τ_p−μ)^2 / Σp w_p)` on oracle power PDP or discretized observable PDP. | Keep `oracle_ray` and `bandlimited_pdp` as *different versions*. |
| D06 | delay percentile / tail quantile (`s`) | Delay containing x% cumulative power, with standardized window/noise rules. | Sensitive to weak tails / Monte Carlo solver floor. |
| D07 | coherence bandwidth (`Hz`) | Defined from complex frequency correlation vs `Δf` at a declared threshold (e.g. 0.5); not simply `1/(5σ)` with hidden constant. | Requires wide enough frequency grid and estimator definition. |
| D08 | true path-delay distribution | Ragged oracle `{tau_p,|a_p|²}` with consistent polarization combination. | Individual sampled diffuse rays are simulator-dependent. |
| D09 | observable PDP/CIR tensor | Band-limited finite-grid delay response with sampling, pulse shaping, FFT window. | Recoverable support limited by bandwidth; IFFT circular aliasing controlled. |
| D10 | excess delay relative to geometric direct path (`s`) | `tau_p−distance/c` for scene-grounded endpoints. | Negative tiny deviations signal precision or path errors; material group delay needs care. |
| D11 | CP exceedance/margin (`s` or Boolean) | Declared dominant/tail channel delay vs CP length including receiver timing window. | PUSCH/OFDM demod quality depends on more than RMS spread. |

### 5.3 Multipath structure and angular/spatial labels

| ID | Target | Definition | Prerequisite / caution |
|---|---|---|---|
| M01 | native solver path count | Valid traced/sampled RT rays after solver pruning. | **Diagnostic only**: depends on tracing budget and diffuse sampling. |
| M02 | dominant resolvable path count | Number under an explicit finite-bandwidth, finite-aperture, noise-limited estimator/oracle resolution rule. | This is not M01 and may not equal geometric reflection count. |
| M03 | cluster count / centroids | Clustering in delay-angle-Doppler with frozen metric, threshold and bandwidth. | Clustering algorithm/version must be stored; nonunique. |
| M04 | complex component gains | Ragged `a_p` with common phase reference and polarization frame. | Global phase gauge and solver sample weighting. |
| M05 | interaction sequence/type | Sequence of direct, specular, diffuse, transmission, diffraction, object/material IDs where exposed. | Composite interaction paths require list, not one simple type. |
| M06 | specular vs diffuse power ratio | Defined decomposition under controlled path categorization. | Diffuse sampled energy must converge under ray-budget changes. |
| M07 | Ricean-like K-factor (`dB`) | Explicit deterministic/specular component to residual stochastic/diffuse component definition. | Standard Rician K is not always physically meaningful for arbitrary RT scene. Use `k_undefined` mask when no appropriate decomposition. |
| A01 | AoA azimuth/zenith (`rad`) | Per-path incoming ray direction at RX in global and antenna frames. | Need actual array geometry for inferred AoA; coordinate convention frozen. |
| A02 | AoD azimuth/zenith (`rad`) | Per-path departure direction in global/antenna frames. | Requires TX configuration and array, and available ground truth. |
| A03 | RMS angular spread (`rad`) | Circular mean and dispersion for azimuth, explicit 3D convention for zenith. | Never use naïve linear mean across `±π` wrap. |
| A04 | dominant angular mode | Direction of strongest resolvable beam/path at a declared band/array. | Arrays/grating lobes can make modes aliased. |
| A05 | MIMO channel rank/singular spectrum | SVD of properly scaled channel at selected `f,t`, per-stream/polarization mapping. | Numerology and channel normalization specified. |
| A06 | spatial covariance and coherence | Full covariance or estimated eigenstructure over valid time/frequency support. | Sample support and temporal stationarity restrictions. |
| A07 | polarization metrics | Cross-pol coupling, XPR or 2×2 Jones response if accurately available. | Depends on dual-polarized antennas/material model; not on simple scalar SISO. |
| A08 | beam-gain vector | Power/SINR for each declared Tx/Rx beam codeword. | Label includes codebook, array, RF chains and interference assumptions. |
| A09 | best beam index/top-K | Top beam(s) under a specified metric and exhaustive oracle search. | Codebook-based, not intrinsic geometric direction. |

### 5.4 Mobility, dynamics and forecasting labels

| ID | Target | Required temporal truth and caution |
|---|---|---|
| T01 | path Doppler (`Hz`) | Native solver/direction/velocity; receiver temporal baseline makes it inferable. |
| T02 | Doppler power spectrum and spread | Declare window, taper, stationarity, FFT bins, sampling, anti-aliasing; moving blockers create nonstationarity. |
| T03 | coherence time (`s`) | Defined from time correlation threshold, *not* universal `0.423/f_D` for any channel. |
| T04 | true next-time complex CSI | Future retraced geometry and complete receiver/antenna/frequency grid, aligned clocks and phase reference. |
| T05 | future **observable** CSI | Future noisy estimated CSI under pre-registered receiver recipe; not future perfect CSI as inference input. |
| T06 | multi-horizon delay statistics | Delay spread, early path delay and path power at future `Δt`. |
| T07 | future LOS/detectable LOS | Separate geometry and detector event labels. |
| T08 | time-to-blockage / event probability | Censoring-aware event labels and right-censor indicator; dynamic obstacle trajectories. |
| T09 | path birth/death and tracking association | Per-component association under frozen algorithm; avoid implicit path-index stability across snapshots. |
| T10 | future beam-quality vector / best beam | Define future codebook, no future CSI leakage into chosen past beam. |
| T11 | future outage probability and duration | Link budget, service threshold, sampling and right-censor treatment. |
| T12 | channel prediction interval / distribution | Calibrated forecast uncertainty for future targets; avoid nominal 95% coverage unsupported by calibration. |
| T13 | geometric position, velocity, heading | Optional supervised motion reference; include sensor noise and synchronization if observable. |
| T14 | occluding object trajectory | Oracle-only, mechanism analyses and controlled intervention, not ordinarily receiver-accessible. |

### 5.5 Waveform, receiver and downstream targets

| ID | Target | How measured / caveat |
|---|---|---|
| R01 | perfect complex physical CSI | Simulated channel reference, not receiver output. |
| R02 | estimated complex CSI | LS/LMMSE/other estimator output with pilot mask, interpolation and uncertainty. |
| R03 | raw received IQ / OFDM grid | Includes noise, interference, CFO, timing, front-end effects and realistic pilots; large storage. |
| R04 | channel reconstruction error | Complex EVM/NMSE on held-out pilot/data positions; protect against using perfect CSI as input. |
| R05 | timing/frequency synchronization error | Relative to known simulator clock; real ground truth only with independent synchronization reference. |
| R06 | DMRS estimation uncertainty | Estimated covariance/noise variance vs injected truth; estimator-specific. |
| R07 | PUSCH BLER/BER/CRC/EVM | Real decoder/mapper/channel pipeline, MCS/TBS/coding/rate mapping matched. |
| R08 | channel-equalization gain | Matched detector, beam/precoder, throughput under identical conditions. |
| R09 | positioning/ranging error | Reference location and calibrated clock/array geometry; report 2D/3D RMSE and P50/P90/P95. |
| R10 | beam recovery/failure events | Defined service condition, beam search time/overhead and misalignment thresholds. |
| R11 | SNR/noise floor/calibration uncertainty | Use injected physical model or measured receiver noise, never posthoc target-dependent normalization without declaring it. |
| R12 | out-of-distribution and abstain label | Shift identifier and confidence/coverage policy; OOD label is benchmark metadata, not model-observable scenario name. |

### 5.6 Derived properties and material/environment targets (oracle diagnostics)

- Building class, façade/material family, thickness, roughness, permittivity/conductivity/scattering coefficient/frequency interpolation, surface orientation/normal, interaction object chain, indoor/outdoor/tunnel state, occluder size/speed/shape, route density, foliage state and precipitation assumptions.
- TX height, RX height, inter-site distance, elevation change, total path lengths, source distance, link direction and antenna pointing errors.
- Boundary-distance to nearest occlusion, mesh edge, material-change edge and direct-ray clearance margin; store these as **label-quality diagnostics** and stratification variables.
- Simulation quality: path depth distributions, direct path agreement, ray-budget dependence, received-power stability, energy/polarization checks, sample-to-sample numerical repeatability.
- **No ungrounded inverse task:** material/obstacle semantics are only evaluated as model outputs when multiple environments/materials and independently verified measurements render the inverse task meaningful. Otherwise oracle metadata is for stratification only.

### 5.7 Label semantic variants that must never be conflated

1. `los/geometric_unoccluded`, `los/solver_direct_path`, `los/detectable_given_receiver`, `profile/family_los_category`, `los/measured_annotated` are **five different labels**.
2. `delay/rms_from_solver_rays`, `delay/rms_from_bandlimited_pdp`, `delay/rms_receiver_estimated` are different targets.
3. `path_count/solver_rays`, `path_count/physical_clusters`, `path_count/resolvable`, `path_count/fitted_order` are different targets.
4. `rx_power/unscaled_physical`, `rx_power/normalized_for_shape`, `rx_power/post_agc` are different observables.
5. `toa/absolute_tx_reference`, `toa/relative_to_sync`, `toa/first_detected` are different targets.
6. `channel/complex_with_known_clock`, `channel/complex_with_random_phase`, `channel/magnitude_only` are different modalities.
7. `channel_future/physical_retraced`, `channel_future/doppler_extrapolated_short_horizon`, `channel_future/measured` are different truth sources.
8. `blockage/physical_object_intervention` and `blockage/remove_direct_ray_mathematically` are different interventions.
9. `path_power/sum_incoherent_ray_energies` and `received_power/coherent_array_superposition` are different operations; never silently substitute.
10. `coverage/service_outage` and `ray_solver/empty_paths` are different statuses.

### 5.8 Missingness, ambiguity and target masks

Use structured `validity` (per target element):

- `VALID`: estimand defined, reference available and QA passed.
- `NOT_APPLICABLE`: e.g. AoA on truly scalar link without array reference, K-factor decomposition undefined.
- `NOT_OBSERVABLE`: available only to oracle but not identifiable from given measurement under specified gauge; label can still train a *predictive prior* model if expressly framed, not a unique physical estimator.
- `REFERENCE_MISSING`: measured source lacks geometry/phase/time/synchronization labels.
- `LOW_CONFIDENCE_GT`: solver nonconvergence, marginal LOS boundary, manual annotation uncertain.
- `SOLVER_FAILURE`: not a physical outage.
- `CENSORED`: future event not observed before trajectory end.
- `OUTSIDE_CALIBRATED_DOMAIN`: target exists, but claimed material/physics fidelity unsupported.
- `SENSOR_DROPOUT`, `SATURATED`, `TIMESTAMP_GAP`, `LICENSE_RESTRICTED`.

Every training loss and metric must use declared masks and *report support counts*. No zero-filling for unavailable targets without a parallel mask and baseline showing no artefactual signal.

---

## 6. Geometry and scenario coverage: what scenes actually belong in UMCD-1

### 6.1 Master scene families — base and held-out

| Scene group | Representative geometry / signal phenomena | Recommended role | Specific audit |
|---|---|---|---|
| `S00-analytic` | Free space, reflector plane, two-path, open/blocked wall | Unit oracle only | Exact path timing/phase; numerical energy/diffraction checks. |
| `S01-urban-canyon` | Street canyons, intersections, variable building heights, vehicle blockage | Core | Direct plus multi-bounce, corner NLOS, blocked LOS, roofs/cars. |
| `S02-dense-urban` | Heterogeneous blocks, plazas, narrow alleys, glass/metal/concrete | Core | Independent *realistic* asset sources; high material uncertainty. |
| `S03-suburban/rural` | Low density, vegetation, long ranges, open LOS, occasional blockers | Expansion | Foliage model qualification, dynamic range and terrain. |
| `S04-indoor-office` | Partition walls, corridors, furniture, glass, doors | Core | Transmission through thin slabs, wall thickness & material calibration. |
| `S05-indoor-factory/warehouse` | Machinery, racks, AGVs, ceiling clutter, NLOS around metal | Core/expansion | Material/roughness and moving obstacles, coherent metal multipath. |
| `S06-station/airport/mall` | Large halls, moving crowds, metal/concrete/glass surfaces | Expansion | Human blockers, many antennas, dynamic blockage. |
| `S07-outdoor-to-indoor` | Façade/window penetration, floor-level motion | Expansion | Transmission approximation and power validation. |
| `S08-tunnel/garage` | Confined waveguide-like propagation, long multipath tails | High-risk OOD | Applicability of RT assumptions, material completeness. |
| `S09-mmwave street` | Higher carrier, blockage sensitivity, narrow beams, diffuse loss | Separate calibrated band | Frequency-dependent material/antenna uncertainty, 3GPP range. |
| `S10-public-unseen-site` | Independent outside-city/campus/building asset unseen in training | Frozen zero-shot test | No related mesh clones or asset reuse. |
| `S11-measured-campaign` | Independent testbeds (e.g., CAEZ with its actual available fields) | External transfer | Capture protocol, permitted labels, licenses, timestamps. |

**Avoid assigning one building to both train and test merely by moving the receivers to another floor.** Independent assets, construction styles and material recipes matter more than millions of nearby receiver positions.

### 6.2 Geometry design variables to cover

- Building dimensions, street widths, roof/wall/ground topology, intersection type, alley depth, urban density, floor plans, ceiling height, door states, stairwells, corridors, shelves, beams, vehicles, street furniture, movable screens, trees/foliage (only qualified approximation).
- TX site height/orientation, sector azimuth/downtilt, UE height (handheld/vehicle/fixed), mounting/body orientation, inter-antenna separation and RF cable/wiring reference.
- Calibrated material classes across frequency: dry/wet concrete (only if validated), brick, glass, metal, asphalt, drywall, wood, human tissue/vehicle body approximations. Do not assume identical scattering across bands.
- Different antenna patterns: isotropic reference, 3GPP directional pattern, 1×1/2×2/4×4/8×8 (subject to realism and resources), dual polarization, beam codebooks; actual element geometry and orientation.
- Obstruction states: no blocker, near-line blocker, complete direct-ray blocker, translucent/transmissive slab, mobile human/vehicle, crowd, diffraction around edges.
- Path classes: single direct, direct+specular, direct+diffuse, direct+transmission/compound, NLOS dominant reflection, diffraction-only NLOS, deep NLOS with outage, negative/no-service, near-grazing visibility.
- Timing and distances: short/medium/long links, different excess delays, CP margin near failure, coexisting direct and delayed paths within one resolution cell.

### 6.3 Independent site vs scene variant

Create **different sites** with independently sourced geometry and generation grammars; treat a simple seed-changing of one procedural generator as a **variant within the same family**, not as proof of domain generalization. Hold out generator grammar, textures/meshes, material sources, site footprint and deployment pattern.

### 6.4 Geometry quality and exclusion gates

For each scene version store:

1. XML asset closure + meshes/MTL/textures + unit scaling + axis conventions + mesh topology validation.
2. Collision/penetration checks: antennas not buried in walls, intended thin surfaces correct, duplicate/degenerate mesh faces flagged, objects do not teleport across timestamps.
3. Scene preview/render audit of random and failure-class links and all LOS↔NLOS boundaries.
4. Human-independent geometry oracle where feasible: direct-segment intersection ray test not relying solely on solver-generated depth-zero ray availability.
5. Local clearance/margin around segment; borderline occlusion geometry labeled low confidence.
6. Calibration status `unqualified`, `plausibility_checked`, `field_calibrated`; nothing becomes `field_calibrated` without measurement and documented optimization/holdout.
7. Material value and frequency range validation, including undefined property behavior.
8. Source license/privacy/geolocation review for real-world imported building maps.

---

## 7. Controlled generation parameter matrix (design of experiments)

**Important:** values here are **illustrative candidate levels**, not 3GPP-mandated limits or validated optimum values. Finalize after propagation and receiver sanity checks, hardware budget, and chosen carrier band. Continuous and logarithmic random draws between levels are encouraged, with frozen seeds and registered priors.

| Dimension | Pilot levels / ranges | Final selection principle | Guard |
|---|---|---|---|
| Carrier (`f_c`) | 0.7, 2.6, **3.5**, 6, 28 GHz | Core n78 3.5 GHz first; separate higher-frequency calibrated domain | Material model validated per band; array spacing/wavelength. |
| NR SCS | 15, 30, 60 kHz (higher only when supported) | Primary 30 kHz n78; others controlled shift | Numerology legal for configured band/FR; check CP and FFT indices. |
| Occupied bandwidth | e.g. ~5, 20, 40, 80/100 MHz subject to resource grid | 20/40/100 MHz allow identifiable delay sweeps | Active subcarriers/guard/DC; bandwidth changes timing resolution. |
| RX/TX per-link antenna ports | 1×1, 1×2, 2×1, 2×2, 4×4, 8×8, 16×4, 32×8, 64×8, up to 128×16 conditionally | SISO debug; SISO+SIMO+MISO+2×2/4×4 core; massive MIMO staged | Always declare convention `N_tx_port × N_rx_port`, RF chains/layers separately; synthetic vs explicit array approximation and compute limits. |
| Network node/active-link count | 1×1, 1×N_rx_node, N_tx_node×1, N_tx_node×N_rx_node | 2 BS × 4 UE mandatory pilot; 3 BS × 8 UE extended | Native RT can process multiple nodes but `scene.tx_array`/`scene.rx_array` are shared types; heterogeneity requires audited groups. |
| Duplex / multiuser | UL, DL, TDD, FDD, one-user, MU-MIMO, co-channel multiuser | SU baseline, UL/DL + MU interference pilot | TDD reciprocity needs calibration; FDD distinct carrier-dependent channels. |
| TX/RX polarization | V, H, ±45°, dual/cross | Start documented TR38901 patterns | Physical Jones basis, cross-pol coupling. |
| Link length / TX height | Stratified near/intermediate/far; terrestrial heights by scene | Realistic per UMi/UMa/indoor application | No antenna inside mesh; far-field assumption checked. |
| LOS state | Natural prevalence + carefully targeted rare categories | Keep natural sampling separately from balanced stress | Do not force 50/50 as deployment base rate. |
| Direct/indirect power ratio | Continuous, focus weak direct and near boundary | Condition primarily via geometry/materials | Do not invent direct power by relabeling unphysical profile. |
| Delay spreads | 30/100/300/1000 ns **for TDL/CDL sweeps**; RT natural | Compare shape and tail regimes | RT should not be forcibly remapped to a desired RMS spread. |
| Speeds | static, pedestrian, bicycle/vehicle-like plausible distributions | Grounded per site; sub-frame Doppler check | Max Doppler vs pilot time sampling Nyquist. |
| Horizons | 1, 2, 5, 10, 20, 50, 100+ frames **as physically supported** | Map to 1 ms → seconds; pre-register horizon units | Long horizons require retracing and sampled blocker motion. |
| Ray depth | 0,1,2,3,5 + convergence reference | Scene-dependent increase | More depth not automatically more accurate if sampling poor. |
| `samples_per_src` | powers-of-ten convergence ladder (scene-dependent) | Select stability of relevant *observables*, not ray count | Static default 200 is pilot, not production proof. |
| Diffuse scattering | off/control and on with audited material coefficients | Include material/solver-budget sweeps | Zero coefficient may yield no diffuse rays despite flag. |
| Diffraction/refraction | baseline separate toggles and full validated combinations | Physics sensitivity | Sionna RT first-order diffraction; thin-wall transmission approximation. |
| Thermal noise | received thermal floor with NF/B; SNR-controlled diagnostic branch | Physical link budget primary | Never normalize away fading in absolute-power branch. |
| Interference | none, co-channel, adjacent (validated waveform) | Downstream realistic SINR | Avoid interference class leaking through artificial dataset metadata. |
| CFO/clock | 0, realistic ranges/offset trajectories | Derived from hardware or declared stress | Physical phase/gauge and reference must be defined. |
| RF distortion | phase noise, IQ imbalance, AGC, quantization, clipping, antenna calibration | Controlled ablations first | Model proper temporal correlations; store exact injection seed. |
| Pilot design | PUSCH DMRS maps; PRS if independently implemented/validated | Link application and timing observability | Pilot masks, symbol positions, hopping patterns exact. |
| Mobility | receiver-only, transmitter-only, blockers-only, combined | Change one factor at a time for causal attribution | Moving geometry needs retracing. |

### 7.1 Sampling method

1. Freeze **site-level prior** and site holdouts.
2. Build a stratification table over rare physical categories: short-range LOS+rich multipath, weak-direct LOS, diffraction-only NLOS, NLOS strong single reflection, no-path, near-grazing, LOS→NLOS transitions, severe clock errors.
3. Within each independent site, sample **whole episodes/trajectories**, not iid frames, with documented oversampling of rare categories in a separate research benchmark view.
4. Use a space-filling/stratified design across meaningful dimension subsets, not a Cartesian product of all rows of the matrix. The full Cartesian product would be computationally impossible and produce unrepresentative marginal distributions.
5. For each physical parent, sample a *limited, pre-registered* set of paired receiver conditions (SNR, impairments, bandwidth), sharing geometry/propagation truth.
6. Maintain **natural-prior**, **balanced-stress**, and **challenge-tail** query views from the same stored events; never overwrite the underlying prevalence.
7. Record `sampling_probability`, `sampling_policy`, and `parent_group_size`; only calculate deployment-weighted metrics when the weighting estimand/prior is justified.
8. Validate that train/val/test distributions have the intended invariances/shifts and that no unique ID encoded as a feature can recover a target.

### 7.2 Coverage ledger per release

Required multidimensional summary (include *groups*, not merely frames): `family × independent_site × LOS category × direct/indirect power band × receiver_SNR × bandwidth × array type × blocker motion × outage category × target-validity × split`.

For each cell record: candidate episodes, generated episodes, accepted episodes, rejected episodes, **independent groups**, total frames, effective temporal sample size, QA failure rate, label confidence, mean/quantiles, solver budget and uncertainty. Do not claim complete matrix coverage where count is zero.

### 7.3 Sampling distributions versus label shortcuts

- Ensure both geometric LOS and NLOS occur in **each** scene type where possible; avoid one-class scenes becoming proxies.
- Vary SNR, bandwidth, antenna pattern and path-loss characteristics **within each class** to prevent trivial shortcut detection.
- Keep lossless base physical path samples rather than resynthesizing labels from an observed model output.
- Probe how well a classifier predicts the label from **metadata alone**, from received total power alone, and from randomly permuted phase/antenna axes; excessive easy accuracy is a shortcut warning.

---
## 8. Physical path oracle: precise Sionna RT generation contract

### 8.1 Separate the oracle from the ray solver result

For each `(site, realization, TX, RX, frame)` run:

**Step 1 — geometric visibility:** Use a declared independent geometric segment-intersection oracle with antenna reference points, scene transforms, occlusion margins, tolerances, material penetration policy, and intersecting object IDs. Compute ray clearance before using `PathSolver`. Label `geometric_los` and `geometry_ambiguous`.

**Step 2 — native solver:** With one frozen Sionna RT release and runtime, compute `paths_full = PathSolver(deterministic=True)(scene, los=True, specular_reflection=True, diffuse_reflection=..., refraction=..., diffraction=..., max_depth=..., samples_per_src=..., synthetic_array=..., seed=...)`. Do not default away from higher-order phenomena without recording the omission.

**Step 3 — path extraction:** **Before** reducing to CFR, extract all available `Paths` arrays and their native shapes/field conventions (`valid`, `tau`, `a`, interaction type, path vertices, AoA/AoD, Doppler, object/mat IDs where genuinely present). Record field availability and map Dr.Jit to NumPy/Torch in a tested version-specific adapter. Preserve raggedness; do not assume `path_index` is stable across solves.

**Step 4 — absolute physical channel:** Generate `paths_full.cir(..., normalize_delays=False)` and `paths_full.cfr(frequency_offsets, ..., normalize_delays=False, normalize=False)`. Store frequency references, carriers, time samples, channel scale, antenna axes, polarization. **Unit-test the passband→baseband convention and axis mapping.** Do not substitute unit-energy normalization.

**Step 5 — LOS-path isolation:** Prefer extracting and reconstructing the **direct subset from the same full path set** via a tested interactions/type mask so that coefficients and reference are identical. If a separate `max_depth=0` direct solve is used to cross-check presence, don't blindly subtract the two CFR outputs: different solver path sets and delay references can invalidate the residual. Compare the direct component to the full-set counterpart by physical delay/geometry, not arbitrary path index.

**Step 6 — indirect/mixed analysis:** Compute `H_direct` and `H_indirect` from the same path basis, verify `H_total ≈ H_direct + H_indirect` per frequency/antenna/time. Report direct power, coherent total power, indirect-component power and whether cross terms produce constructive/destructive interference. `P_total` need **not** equal `P_direct+P_indirect` when computed from coherent CFR power. Log this distinction.

**Step 7 — QA outcomes:** `ACCEPTED`, `SUSPECT_ORACLE`, `RETRACE_REQUIRED`, `SOLVER_INCOMPLETE`, `INVALID_MESH`, `PHYSICAL_OUTAGE`, `RECEIVER_OUTAGE`, with reason and alternative-solver/budget evidence. Do not silently resample away rejected/outage cases.

**Step 8 — parent publication:** Commit content-hashed oracle and channel only after numerical QA; then create deterministic children (receiver settings/noise views) without rerunning RT.

### 8.2 Physics-related approximations to record explicitly

- Thin-slab transmission/refraction; no angular deflection for transmitted rays in the documented RT solver approximation; thick buildings may need different physics.
- First-order-only RT diffraction support; high-order/waveguide effects may be insufficient in tunnels and highly blocked environments.
- Diffuse sampled ray behavior and material scattering-power calibration. A higher ray count may merely reflect more Monte Carlo samples.
- Synthetic antenna array approximations (phase reweighting at one traced center) versus explicit-element traces; test near/far-field validity.
- Antenna radiation pattern and polarization from geometry/arrays are part of the physical channel, not posthoc cosmetic scaling.
- Numerical precision and Dr.Jit/Mitsuba backend differences; freeze variant, versions, adaptive loop option and fast-math settings when material.
- Ray truncation depth, max-path clipping and minimum effective received path power. A low-power ray's absence from `Paths` is not automatically absence from electromagnetics.
- Material properties and scattering spectra at the carrier frequency; source geometry alone does not verify physical plausibility.

### 8.3 Solver convergence and conditional failure checks

For a stratified set of scenes and difficult links, sweep `max_depth`, `samples_per_src`, `max_num_paths_per_src`, diffraction/scattering switches and deterministic seeds. Evaluate not just path count but:

1. LOS depth-zero agreement with geometric oracle and status of pathological near-edge links.
2. Fraction of accepted LOS links with **multiple** indirect physical paths and realistic direct/indirect ratios.
3. Distribution of true empty-path observations under increased ray budget, conditional on geometry class.
4. Stability of total channel power **and** band-limited CSI tensor (complex phase to be compared only under common reference).
5. Stability of RMS delay spread/PDP tail, strongest path timing, AoA modes, Doppler, beam ranking.
6. Convergence across independent seed replicates for stochastic diffuse scattering; deterministic setting only addresses rerun reproducibility, not adequacy of one sample.
7. Hard cases: grazing edge, weak direct under reflection, multiple nearly coincident paths, transmit-through-wall geometry, diffraction-only NLOS, long-delay weak tails.
8. Beam/array sensitivity and physical no-path/receiver-outage distinction.
9. CPU-LLVM vs CUDA-OptiX *numerical distribution and aggregate* comparison when both valid; do not require unjustified bitwise equality.
10. Per-class accept/reject/retrace proportions so exclusion does not artificially inflate classifier performance.

**Acceptance rule:** convergence tolerances must be selected from physical accuracy requirements and independently computed reference error curves in a preregistered QA configuration. Avoid a universal arbitrary 'within 1%' rule for all statistics, especially near nulls or low power. Publish failure percentages even if small.

### 8.4 Paths data abstraction and native API adapters

Define `CanonicalRTPathSet` with field capability flags. For each field, adapter maps `(native_path_handle, pinned_version)` → `(array, axes, units, gauge, validity, provenance)`. Use a native integration test that asserts expected shapes for: SISO, MIMO synthetic array, MIMO explicit array, no path, one path, many paths, LOS+reflection, diffraction, diffuse and mobility. Unsupported fields are recorded as unavailable with actual reason and never fabricated.

The Sionna RT developer guide explicitly discusses path object structure and the distinction between synthetic and explicit arrays; the 2.2 mobility tutorial explicitly advises geometry retracing for longer movement. Those are mandatory implementation references, not optional background reading.

### 8.5 Path identity / matching over time

- Avoid equating path index across retraces; ordering may change after small motion or stochastic diffuse sampling.
- For path tracking, use controlled association based on delay, angular proximity, Doppler, interaction-object chain and continuous geometry; allow births, deaths, splits and merges.
- Store association confidence and ambiguous matches; use local order-free path-set metrics as primary where association is uncertain.
- For diffuse power, aggregate in delay-angle cells rather than insisting each Monte Carlo sample is an enduring physical scatterer.
- Treat track IDs as derived labels from a specific association algorithm version, not direct solver truth.

---

## 9. Standardized stochastic generation: TDL/CDL and full 3GPP scenario models

### 9.1 TDL/CDL baseline corpus

- Generate every scalable A–E model with **each** declared delay spread, carrier, SCS, bandwidth, mobility and antenna state that the API supports.
- Keep Sionna PHY **native** `a,tau` and parameter-table filename/hash, `spec_version`, common generator seed, `normalize_delays` option, nominal powers, empirical power and RMS spread.
- Validate empirical means, total normalized tap power, delay scaling, K-power partition and Doppler distribution against pinned 3GPP table expectations; define confidence intervals over independent draws.
- TDL-A/B/C and CDL-A/B/C are model-defined NLOS families and D/E LOS families; this is a **profile grouping label only**. Dataset class `profile_model_los` is not `geometric_los`.
- TDL/CIR taps are not geometrically localized rays and may lack physical AoA/object interactions. Do not force angle/scene fields to fictitious values.
- CDL provides clustered angular modeling but its angular rays/cluster meaning differs from RT ray interaction chains; store `cluster` fields under their own namespace.
- Specify whether input sample index represents independent channel draw, per-channel temporal evolution, or different noise view of same draw. Prevent reusing independent seeds across splits except intentional matched tests kept within one split.
- Have train, held-out profile, held-out delay spread, carrier, bandwidth and antenna domain subsets with auditable group lock. A leave-one-profile-out fold may contain a **single** binary proxy class; AUROC for that fold is undefined. Report compatible metrics.
- Use TDL/CDL for **algorithm initialization/benchmarks and 3GPP conformance**, not as a shortcut to physical LOS inference.

### 9.2 3GPP stochastic geometry-based scenario models

- Where supported and locally tested, generate full channel families `UMi`, `UMa`, `RMa`, `InH`, `InF` with geometry-conditioned distances, velocities, antennas, shadowing and scenario LOS state.
- Explicitly record whether the API's scenario `los` flag is sampled or externally forced; avoid assuming it equals a ray-traced direct geometric line.
- Ensure plausible source locations and correlated large-/small-scale parameters; site priors should not be chosen based on test labels.
- Compare power-delay/angular and correlation statistics with the targeted 3GPP spec; guard applicability ranges, cross-frequency scaling and standardized delay normalization.
- Store scenario-specific labels (`scenario_los`) separately from RT ground truth and measured annotated LOS.

### 9.3 Source-family roles and transfer protocols

- `TDL/CDL → RT` is a **domain-shift experiment**, not evidence of the same label oracle. Require clearly stated target mapping and which tasks are shared (e.g. receiver-observable delay spread / PDP), plus exclusions where no canonical equivalence exists.
- `3GPP full scenario → RT` can test statistical-vs-geometric transfer with more comparable scenario conditions; evaluation still acknowledges label ontology differences.
- `RT → measured` is strongest external evidence but measured labels may be restricted to CSI magnitude, timing, position or beam quality.
- A unified MLP/complex model should use **task-specific output masks and domain embeddings if permitted**, without turning source-ID into an exploitable task shortcut.

---

## 10. NR PHY and receiver-observation dataset: preserve what deployment really sees

### 10.1 PHY processing chain and independent observables

```text
Parent physical paths / canonical H(f,t,ant)
    ├─ Perfect-CSI oracle measurement (research upper bound, never standard input)
    └─ Waveform generator [legal NR carrier/BWP/numerology, payload, DMRS/PRS]
         → OFDM modulation + CP / dynamic time-domain channel
         → path loss + interference + thermal noise + RF/synchronization impairment
         → ADC/sample-clock/AGC if enabled
         → raw complex IQ (largest storage tier)
         → sync/FFT/CP removal → received grid
         → receiver channel estimator (LS/LMMSE) + pilot mask + error variance
         → receiver-only CSI / signal features
         → downstream decoder, beamformer, BLER/positioning evaluation
```

### 10.2 Primary PUSCH/DMRS view

- Keep current working `PUSCHTransmitter`, `PUSCHReceiver`, `PUSCHLSChannelEstimator` and LMMSE paths from pinned Sionna PHY; validate against official native tests, spec tables and independent synthetic channel fixtures.
- Store exact: carrier band, center frequency, SCS, BWP size/start, PRBs, FFT size, CP length per symbol, slot format, port/layer/polarization, actual DMRS and data RE locations, pilot symbols and power, MCS, stream/layer mapping, precoding, estimator config.
- Preserve three observation tiers: `RX_RAW_IQ` (with sample timing and calibration), `RX_RESOURCE_GRID` (with null/DC/guard positions), `RX_ESTIMATED_CSI` (with correct effective-subcarrier mapping). Do not assume these are interchangeable.
- Keep each injected physical episode's perfect H separately and never prefill missing CSI at deployment features from perfect H unless running oracle control.
- Frequency-domain diagonal OFDM multiplication is valid only when assumptions (CP margin, intra-symbol variation/ICI and synchronization) justify it. Compare to time-domain convolution at adverse delay spread and mobility.
- `E_b/N_0`, SNR per receive antenna, SNR per active RE and thermal NF/B are different; store the exact noise-generation reference. Prefer **received physical noise floor and link budget** as the main dataset, plus explicitly normalized SNR-controlled stress views.
- Receiver observations should include path loss, AGC, EVM, residual CFO, noise covariance, estimated channel errors and optional interference as realistically modeled; don't let nuisance seeds encode labels.

### 10.3 PRS/SRS/SSB / other NR signals (not automatic)

- PRS can improve positioning/time-delay observability, but not all Sionna PHY convenience APIs necessarily implement a validated complete 3GPP PRS chain. **Treat PRS/SRS/SSB generation as a separate development/standards-conformance item**; do not label a generic comb pilot as 3GPP PRS.
- Include downlink/uplink differences in TX/RX array geometry, reference clock assumptions and duplexing. For SRS, record antenna/beam sounding behavior; for SSB, define RSRP/beam measurements.
- Pilot resource-element mapping must be independently checked against **the applicable release/version of 3GPP TS 38.211**, with other scheduling parameters against 38.214/38.331 as required.

### 10.4 Impairment model registry

| Impairment | Primary stored parameter | Physical / statistical relationship that can go wrong |
|---|---|---|
| Thermal noise | `noise_psd_w_per_hz`, bandwidth, NF, temperature | Complex noise variance per RE/antenna; no implicit samplewise auto-scaling. |
| Co-channel interference | interferer geometry, waveform, power, clocks, correlation | Interference non-Gaussian/colored and not merely AWGN. |
| CFO | `cfo_hz`, estimator residual | Phase evolves with **actual elapsed time**, not arbitrary frame index. |
| Sampling-frequency offset | ppm and independent clock | Resampling and accumulated timing drift. |
| Timing offset | samples + fractional seconds; CP/FFT anchor | Incorrect CP alignment and absolute ToA gauge. |
| Phase noise | PSD/model params / oscillator process seed | Common phase error and ICI; state correlation across frames. |
| IQ imbalance | gain/phase skew, image coefficient | Mirror-frequency impairment and antenna dependence. |
| AGC/clipping | gain trajectory, saturation count | Absolute power lost or nonlinearity; keep pre/post AGC separately. |
| ADC quantization | resolution, full-scale, dithering | Nonlinearity, saturation and bias. |
| Antenna calibration | complex per-chain gains/phases, drift | Fake angular modes without correction. |
| Mutual coupling | array coupling model (if enabled) | Can violate synthetic-array simplifications. |
| Packet drop/timestamp jitter | missingness mask and actual timestamps | Impacts forecasting and interpolation; don't fill with future data. |

### 10.5 Receiver variants as matched ablations

For each fixed channel episode sample `one calibrated physical view`, then optional paired variants changing exactly one factor: LS→LMMSE, no CFO→CFO, perfect timing→estimated timing, 1×1→2×2, 20→100 MHz, interference off→on, PUSCH frequency→time domain. Pairing is **within** split and logged as `paired_condition_id`.

### 10.6 Measured CSI modality distinctions

CAEZ official preprocessed absolute/magnitude features in the inspected project have shape and noise/timestamps/positions but **lack usable complex phase** in that downloaded representation; preserve the magnitude-only modality. A different authorized complex-channel extraction from the original data would require its own license check, source hash, synchronization/phase-quality audit and storage. Never infer absent phase by assigning zero imaginary component and claiming complex CSI.

---

## 11. Temporal and counterfactual data: not just more independent channel snapshots

### 11.1 Time axes and sample windows

- **Fast time**: IQ sample period inside a waveform.
- **OFDM symbol time**: cyclic-prefix and pilot/FFT schedule.
- **Channel time**: coherence-interval scale governed by Doppler and mobile scatterers.
- **Scene time**: meaningful object displacement, visibility and blockage change.
- **Experiment time**: measured acquisition, drift, hardware changes, site and session.

Every timeline needs explicit timebase and synchronization relationships. Do not equate 0.1 s steps to continuous 14-symbol slots or pretend all 0.1 s CFO phases can be averaged unchanged.

### 11.2 Required trajectory subfamilies

1. Static TX, mobile UE (straight, turning, stop/start, speed variation).
2. Mobile UE with independent vehicle/human blockers causing true LOS transitions.
3. Mobile transmitter (e.g. V2X) with fixed/mobile receiver.
4. Fixed endpoints, moving occluders, allowing blockage prediction without UE movement.
5. Slow-moving materials/objects causing changing indirect routes despite static LOS.
6. Routes with extended deep NLOS, rays reappearing, and periods below receiver detection threshold.
7. Multiple independent trajectories that revisit similar coordinates at **different times** but stay wholly inside the same split to avoid route/time leakage.
8. Scene-change interventions (door open/closed, vehicle placed/removed) with matched experimental controls.

### 11.3 Generate synchronized future targets

For every window with observed times `t_0...t_n`, pre-register future horizons `h` in **seconds**, fetch targets only at `t_n+h` subject to timestamp tolerance and no gap, and attach `horizon_valid` / right-censor flag. Use the **same** actual future propagation truth for all competing models. Do not report future CSI in a mismatched clock gauge; align only by operations that would be available online or evaluate phase-invariant metrics separately.

### 11.4 Retra​cing vs Doppler approximation

- Doppler evolution over short motion: use a path list from one geometric snapshot to evolve phase over a short horizon; mark `DOPPLER_APPROX` and test deviation vs actual retrace.
- Geometric evolution: update scene-object poses and material/antenna state, retrace, recalculate newly born/vanished paths. Mark `RETRACED_TRUTH` and prioritize for blockage and long-horizon forecasting.
- Validate the approximation-versus-retrace deviation as a function of displacement measured in wavelengths, scene topology, path mechanism and horizon; no universal accuracy window.
- Time-varying object geometry must have deterministic track IDs but stochastic path indices. Track the true blocker and independent route identity.

### 11.5 Counterfactual paired experiments

Create two disjoint intervention classes:

**Mathematical** `REMOVE_DIRECT_COMPONENT`: reconstruct `H−H_direct` from the *same* path set with shared phase/delay. This isolates importance of direct component for model representation. It **does not** simulate all changes from physically adding a blocker.

**Physical** `ADD_OR_MOVE_BLOCKER`, `CHANGE_MATERIAL`, `CHANGE_ANTENNA`, `OPEN_DOOR`: alter scene geometry/material, rerun all rays, include changed indirect paths and power. A causal effect is estimable only for a well-defined intervention with matched baselines and no post-treatment input leakage.

- Keep paired records and all their children in a single split.
- Log whether intervention affected direct visibility, path gain, diffraction, reflection, shadowing, or interference.
- Evaluate counterfactual prediction with receiver-visible context and clearly state whether the proposed future intervention is provided to the model as information.
- Do not train on intervention metadata and call it receiver-only detection unless online source of that information exists.

---

## 12. Dataset build DAG, reprocessing, caching and pipeline responsibilities

### 12.1 Immutable DAG

```text
RELEASE_LOCK
  ├─ pinned Sionna/RT/PHY/Mitsuba/DrJit + device + spec tables
  ├─ frozen scene assets + material calibration + scene versions
  ├─ independent episode manifest + split lock + scientific priors
  ↓
GEOMETRY_ORACLE  →  NATIVE_PATH_SOLVE  →  PATH_QA
  ↓                              ↓
GEOMETRY_TRUTH            RAW_PATHS + PHYSICAL_CIR/CFR
                                  ↓
                      SIMULATOR-TRUTH TARGETS
                                  ↓
                        CHANNEL/WAVEFORM VIEWS
                                  ↓
                      NR_RX + DEVICE IMPAIRMENTS
                                  ↓
                     RECEIVER-ONLY OBSERVATIONS
                                  ↓
                  METADATA/MASK/INTEGRITY JOIN
                                  ↓
                    DATASET RELEASE + SPLITS
                                  ↓
                    MODEL TRAINING / RESULTS
```

Each stage has a deterministic content-addressed key from **parents + source code digest + generator config + dependency versions + all seeds**, plus a successful QA gate. Reprocessing receiver impairments should not trigger RT generation. Editing radio-material state *must* invalidate dependent RT cache even when XML file bytes are unchanged.

### 12.2 Artifact identity & lineage

- `site_id`, `scene_asset_hash`, `runtime_scene_hash` (includes modified material/object state), `realization_id`, `episode_id`, `solve_id`, `oracle_id`, `physical_channel_id`, `receiver_view_id`, `target_definition_hash`, `split_lock_hash`, `release_id`.
- Log all independent RNG streams (`scene`, `position`, `material`, `RT_solver`, `diffuse`, `trajectory`, `waveform_payload`, `noise`, `impairments`, `train`) rather than one mutable global seed.
- Record host/pinned library/CPU/GPU version, native backend, c++/CUDA/OptiX availability, FFT/array implementation and precision.
- Manifest completeness flag false forbids content-addressed reuse, even if output samples exist.
- Repeated generation with same key must either reproduce within declared numerical reproducibility contract or have an explicitly new run identity/replicate key.
- `files missing` / `partial write` / `checksum mismatch` are fatal to publish, not automatic silent retry successes.

### 12.3 Proposed repository additions (targeted extension, not rewrite)

```text
datasets/
  umcd_schema.py            # pydantic/dataclass/Arrow contracts; v1 validators
  umcd_catalog.py           # site/episode/parent lineage registry
  umcd_targets.py           # every estimand/units/gauge/validity rule
  umcd_split_lock.py        # hierarchical inheritance & leakage gates
  umcd_storage.py           # chunked Parquet/Zarr + immutability/checksums
  umcd_views.py             # memory-efficient task access; model-input firewall
propagation/
  geometric_oracle.py       # independent LOS, clearance and ambiguity labels
  path_decomposition.py     # coherent decomposition on one path set
  path_association.py       # optional temporal identities, confidence
  path_statistics.py        # physical vs band-limited delay/angle/power labels
integrations/sionna/
  rt_paths_export.py        # native Paths capability map, ragged extraction
  rt_conformance.py         # pinned native version and array/gauge smoke tests
  rt_dynamic_scene.py       # moving geometry re-trace and intervention contracts
  rt_physical_channel.py    # unnormalized source-of-truth channel and axes
  standardized_scenarios.py# TDL/CDL + UMi/UMa/RMa/InH/InF conformant adapters
  nr_observation_views.py   # PUSCH/DMRS canonical views and future PRS adapters
receivers/
  impairments.py            # parameterized hardware/clock/PA/ADC experiments
  receiver_calibration.py   # noise/clock/array calibration with provenance
scenes/
  catalog/                  # independent site families and calibration levels
experiments/
  dataset_doe.py            # pre-registered parameter draws and coverage targets
  dataset_generator.py      # distributed resumable episode generation
  dataset_release.py        # QA gate and immutable release assembler
evaluation/
  umcd_leakage.py           # scene/asset/link/trajectory/paired-view leakage
  umcd_physqa.py            # conditional solver, channel, timing checks
  umcd_benchmark.py         # frozen tasks, validity masks and trusted baselines
config/datasets/umcd/
  pilot.yaml
  core_physical.yaml
  dynamic.yaml
  stress.yaml
  measured.yaml
  release_v1.yaml
scripts/
  audit_umcd_schema.py
  audit_umcd_physics.py
  audit_umcd_coverage.py
  audit_umcd_leakage.py
  audit_umcd_storage.py
  build_umcd_release.py
```

Use the existing `artifacts/`, provenance, cache, backend selection and validation modules; do not duplicate functionality without a documented migration plan. The file tree is **proposed**, not present in the uploaded source.

### 12.4 Config must be explicit (example illustrative YAML)

```yaml
schema_version: umcd.1
release: umcd-r1-pilot
sionna:
  rt_version: "2.2.0"
  phy_version: "2.2.0"
  mitsuba_variant: auto   # probe, record resolved; freeze for production release
  deterministic_solver: true
source:
  family: F03-RT-STATIC
  scene_catalog: config/scenes/independent_sites.yaml
  seed_scene: 31001
  seed_rt: 31002
channel:
  carrier_hz: 3.5e9
  normalize_delays: false
  normalize_power: false
  store_raw_paths: true
  store_absolute_cfr: true
  store_optional_normalized_view: true
rt:
  max_depth: 3
  samples_per_src: 100000  # pilot candidate ONLY; convergence decides actual
  diffuse_reflection: true
  refraction: true
  diffraction: true
  synthetic_array: true
observation:
  waveform: PUSCH_DMRS
  subcarrier_spacing_hz: 30000
  frequency_grid_policy: resource_grid_effective_fft_mapping
  noise_model: absolute_physical_receiver_noise
  store_raw_iq: tiered
split:
  lock_before_generation: true
  group_by: [asset_family, site, scene_version, trajectory, episode_parent]
qa:
  require_native_sionna: true
  require_solver_budget_convergence: true
  require_model_input_target_firewall: true
```

**Values above are candidate engineering starting points, not validated final configuration or assured runtime feasibility.** A dedicated pilot resolves the actual ray budget, timing, memory, scene size and solver support.

---
## 13. Hard scientific correctness gates (stop generation on failure)

The following checks are mandatory **test definitions**. Test IDs allow PRs, CI runs and release manifests to cite exact evidence. Do not count a mocked API test as a native physics test.

### Gate G0 — environment, source and provenance

- [ ] **G0.01** pinned `sionna`, `sionna-rt`, `mitsuba`, `drjit`, PyTorch, Python, NumPy and expected GPU/CPU backend versions print and match lockfile.
- [ ] **G0.02** selected polarized Mitsuba variant and native `PathSolver` can compute a known free-space path on selected host; CUDA OptiX failure triggers explicit documented LLVM fallback **before** importing/activating RT, not silent mid-run migration.
- [ ] **G0.03** scene asset closure complete: XML/mesh/MTL/texture/associated metadata content hashed; unresolved assets invalidate production cache.
- [ ] **G0.04** dynamically altered radio material properties, scene-object pose/velocity, antenna pattern and frequency enter runtime scene hash.
- [ ] **G0.05** RNG streams independent and deterministic where promised; repeat for same hardware/backend identity.
- [ ] **G0.06** versioned TDL/CDL profile-table hash and external 3GPP spec revision recorded separately.
- [ ] **G0.07** no future-dated or unverifiable benchmark report promoted on filename alone; every result links to raw manifests, commit, generated-by command, timestamps and QA state.
- [ ] **G0.08** licensing and permitted field redistribution validated; measured datasets may require nonpublic local storage.

### Gate G1 — geometry, LOS semantics and physical scenes

- [ ] **G1.01** free-space straight path has the expected geometric visibility and propagation delay `d/c` within numerical tolerance.
- [ ] **G1.02** independent segment-intersection oracle vs depth-zero `PathSolver` agrees in well-conditioned configurations.
- [ ] **G1.03** near-edge/grazing/mesh-contact links flagged ambiguous, not forced into confident labels.
- [ ] **G1.04** antennas outside forbidden solids; units/coordinate frames and object transforms consistent.
- [ ] **G1.05** geometry manipulations (add opaque blocker, open door) produce expected visibility **when physical model supports it**.
- [ ] **G1.06** transmissive thin wall handled separately from geometrically unobstructed LOS; physical transmission is not 'direct free-space LOS'.
- [ ] **G1.07** blocked scene with diffraction/reflection retains indirect paths when actual geometry supports them.
- [ ] **G1.08** LOS-rich scene contains valid direct **and** multiple indirect paths after configured solver convergence.
- [ ] **G1.09** scene motion interpolation never causes discontinuous teleportation; timestamps/poses monotonic.
- [ ] **G1.10** scene families are independent; no hashed assets/mesh variants duplicate into frozen held-out site unintentionally.

### Gate G2 — native path extraction, phase, timing, power

- [ ] **G2.01** `Paths` native API version compatibility: required properties exist with asserted shapes and dtypes.
- [ ] **G2.02** `valid` masks actually gate padded paths; zero/uninitialized entries excluded from computations.
- [ ] **G2.03** explicit `normalize_delays=False` on canonical `cir`, `cfr`, and `taps` where applicable.
- [ ] **G2.04** explicit `normalize=False` for canonical absolute-power CFR.
- [ ] **G2.05** `cfr()` manual reconstruction from the same path set equals API output after correctly applying carrier-to-baseband phase convention.
- [ ] **G2.06** `H_full = H_direct+H_indirect` for a **same-path-set** partition at identical reference, array and frequency axes.
- [ ] **G2.07** frequency bin ordering (center/DC/guard/negative offsets) matches declared NR resource grid, including odd/even FFT-size fixtures.
- [ ] **G2.08** first physical arrival delay invariant to optional regridding/receiver-view generation; normalized view cannot overwrite original.
- [ ] **G2.09** lossless canonical complex serialization roundtrips within dtype error without sign conjugation or axis permutation.
- [ ] **G2.10** path loss/free-space coefficient follows a validated link-budget convention for test geometry and antenna gains.
- [ ] **G2.11** received coherent power not erroneously computed as sum of powers of individual paths where interference cross terms matter.
- [ ] **G2.12** manually reconstructed single-, two- and multiple-path CFRs match analytical test cases including close delays, nulls and nonzero reference offsets.
- [ ] **G2.13** MIMO explicit vs synthetic array differences quantified on representative near-/far-field cases.
- [ ] **G2.14** dual polarization/basis rotation controls yield expected channel transformation.
- [ ] **G2.15** unused solver path metadata/padded values never enter target computations; last array dimension not assumed to mean path without assertion.

### Gate G3 — ray budget and physics fidelity

- [ ] **G3.01** `max_depth` sweep stabilizes relevant delay/power/angular results for representative LOS and NLOS links.
- [ ] **G3.02** `samples_per_src` sweep stabilizes quantiles/distributions, separately for specular/diffuse components.
- [ ] **G3.03** diffuse model enabled **and** nonzero material scattering coefficient validated; flag-only enablement insufficient.
- [ ] **G3.04** diffuse ray count changes with sample budget without being mistaken for changing physical path count.
- [ ] **G3.05** `max_num_paths_per_src` clipping is audited; no silent path truncation in stress cases.
- [ ] **G3.06** refraction penetration/slab assumptions recorded; thick-object results not called high-fidelity without calibration.
- [ ] **G3.07** first-order edge diffraction limitations acknowledged; deep diffraction NLOS stress marked model-limited.
- [ ] **G3.08** deterministic solver replay passes on pinned exact environment; numerical backend changes recorded and quantified.
- [ ] **G3.09** weak-power tails do not spuriously vanish solely from aggressive fixed threshold; threshold sensitivity reported.
- [ ] **G3.10** empty/full/direct path outcomes audited by class with retries and reference budgets.
- [ ] **G3.11** geometric-LOS/solver-direct disagreement excluded from headline until resolved or labeled low-confidence.
- [ ] **G3.12** realized natural vs deliberately stratified class priors both logged.

#### G3 acceptance contract for MVP dataset admission

These are project engineering margins for the declared MVP task; they are not 3GPP limits, evidence of simulator truth, or a rule to maximize ray count/depth. Freeze the candidate configuration, challenger configuration, observable definitions, link budget, waveform, and strata before each qualification run. Compare paired outputs from both configurations; neither output is designated ground truth. An experiment that fails a criterion narrows the supported data/task scope or keeps the release gate closed.

**Evidence design and tail power.** Qualify at least three independent scene/site families; City41, City42, and City43 count as one family because they share a procedural generator. The former minimum of 20 independent scene-link episode clusters per LOS/NLOS stratum/family is a pilot floor only. The outer statistical unit is an independently realized site/scene episode; links, TX/RX pairs, solver seeds, paths, subcarriers, symbols, and frames nested within one realization are not independent qualification samples. Preserve this hierarchy and resample independent realization clusters, stratified by family and LOS/NLOS. For complex CSI, in addition to median and p95 error, define one failure indicator per independent cluster (`cluster CSI failure` if any prespecified episode-level `e_H > 0.05`) and require the one-sided 95% exact-binomial upper confidence bound on the stratum-specific failure probability to be `≤ 0.05`. With zero observed failures this requires at least 59 independent clusters in that family × LOS/NLOS stratum; with failures, use the exact bound and increase sample size as needed. Twenty clusters cannot establish this tail claim. If the bound is not met, keep the corresponding G3 claim/gate unresolved or narrow the admitted scope; do not substitute link count or a bootstrap over nested observations. Three families support only limited transfer evidence across families, so report each family separately and do not imply a well-estimated population of all environments. Every selected run must be provenance complete and free of path-cap saturation; report empty, rejected, and cap-filled outcomes.

**Complex CSI.** Compare unnormalized complex CFR on the declared NR active subcarriers, antenna ports, symbols, absolute-delay reference, and carrier-phase convention. Per episode use the symmetric metric `e_H = ||H_candidate-H_challenger||_F / max((||H_candidate||_F + ||H_challenger||_F)/2, H_floor)`, where `H_floor` is a predeclared equivalent channel-gain noise norm on that exact evaluated RE × port × symbol grid, derived from the frozen receiver noise/link budget. Within every scene-family × LOS/NLOS stratum, require median `e_H ≤ 0.01`, 95th percentile `e_H ≤ 0.05`, and a cluster-bootstrap 95% upper confidence bound for the 95th percentile no greater than 0.05. Also require the independent-cluster failure-rate bound in Evidence design. Do not align away a common phase or independently normalize either channel.

**Delay and angle statistics.** For resolvable first-arrival delay, require paired absolute error `≤ 0.1/B_eff`; for power-weighted RMS delay spread require `|Δτ_rms| ≤ max(1 ns, 0.05 × max(τ_rms,candidate, τ_rms,challenger))`. For 3D spherical arrival/departure spread require `|Δσ_angle| ≤ max(1 degree, 0.05 × max(σ_candidate, σ_challenger))`. Apply the same median/95th-percentile/cluster-bootstrap rule as CSI to each metric; evaluate LOS/NLOS and specular/diffuse components separately where the component exists. A missing or unresolvable statistic is masked with a reason, not scored as zero.

**Path-level truth.** Preserve every solver-valid record, including its raw multiplicity and provenance; raw record count or a fingerprint alone is not physical-path identity. For comparison, form explicitly defined physical/effective path-equivalence classes using ordered interaction types, object/surface identity, interaction geometry, delay, and endpoint directions, with resolutions frozen for the declared bandwidth and array. Primitive IDs and geometry hashes may help matching but are not universal permanent IDs. Since neither solver configuration is truth, use one-to-one class matching and report bidirectional coverage: the fraction of relevant candidate classes matched by challenger classes and the fraction of relevant challenger classes matched by candidate classes; require both `≥ 0.95`. Compare class-level aggregate complex contributions, delay, and directions using the limits below and list every unmatched receiver-relevant class. Before the exact G5 PUSCH receiver is qualified, a class is receiver-relevant if either (a) isolated class omission changes the declared active-grid complex CSI by at least 1% under the symmetric `e_H` metric, or (b) its post-link-budget class power is at least 6 dB above the equivalent receiver-noise power on the same resources. Do not use BLER-based relevance until the applicable G5 receiver checks pass. Do not require identity of stochastic diffuse rays; compare their delay/angle/power quantiles under the delay/angle and CSI limits above, preserving budget provenance. A topology/geometry hash is a matching aid, never a universal path ID.

**NR receiver impact.** On the declared, G5-qualified PUSCH/DMRS waveform, receiver, decoder, and frozen SNR grid, select one common SNR comparison point before examining candidate/challenger outcomes (using a separate pilot or prespecified reference). At this same SNR require the paired candidate-vs-challenger absolute BLER difference to be `≤ 1 percentage point`, with its cluster/block-bootstrap 95% confidence interval wholly inside `[-1,+1]` percentage point. Separately estimate each configuration's `SNR@10% BLER` and compare the shift; require absolute shift `≤ 0.5 dB` with the 95% confidence interval inside that margin. The 10% crossings must be bracketed by the frozen SNR grid; otherwise mark the shift unresolved and do not extrapolate freely. Choose enough independent transport blocks for the BLER interval half-width to be at most 1 percentage point; preserve failures and decoder outcomes. If a receiver/task is outside this waveform scope, G3 does not qualify it by analogy.

These criteria make a finite-budget robustness decision for the named MVP scope; they do not prove physical truth. Any change to carrier, bandwidth, antenna aperture, material family, scene family, receiver, or target definition requires a separately stated qualification stratum. Until the applicable criteria pass, RT generation remains limited to bounded, labeled diagnostic/pilot work and the production D1 RT corpus stays gated.

**G3 qualification preregistration readiness (2026-10-23; no qualification run authorized by this entry):** The preceding contract now defines task-specific pass/fail margins for complex CSI, delay/angular statistics, path-equivalence coverage, and PUSCH receiver impact, but no experiment is frozen. The candidate configuration remains a proposal for static SISO at 3.5 GHz with 30-kHz SCS and 24-RB PUSCH/DMRS: deterministic Sionna RT at `max_depth=3`, compared with a same-everything challenger at `samples_per_src=1,000,000` against a `100,000` candidate. A depth challenger would be evaluated separately at fixed sampling budget; neither configuration is truth. The sample budget, path cap, solver switches, noise/link budget, equivalent-channel floor, path-equivalence resolutions, and BLER SNR grid still require a run-specific frozen, hash-locked manifest. The [scene-family audit](G3_SCENE_FAMILY_LICENSE_AUDIT_2026-10-23.md) and [machine-readable asset/material report](results/g3-scene-asset-material-audit-20261023.json) now establish complete asset closure for eight candidates and exact installed-to-checkout asset-set matches for five Sionna built-ins. City41/42/43 are one procedural family. Etoile, Florence, and Munich are distinct OSM-derived geographic sites and candidate site strata. The user selected use subject to ODbL compliance; release-specific attribution and derivative/redistribution handling still need to be recorded for the intended artifacts. Box and simple-canyon scenes remain controlled fixtures. At 3.5 GHz, all audited materials have scattering coefficient zero. The S=0.5 interventions are synthetic and uncalibrated, so G3.03 remains unsatisfied. Existing 20-seed sweeps are diagnostic repeats within selected geometry and cannot substitute for the minimum 59 independent clusters per family × LOS/NLOS stratum needed for the stated zero-failure bound. **Readiness decision: do not start a confirmatory G3 campaign or production D1 generation yet.** First obtain defensible nonzero material/scattering evidence, freeze all run-specific configuration and path-matching/noise/receiver parameters, record release-specific ODbL handling, and estimate RT and receiver-outcome compute/storage cost. Any smaller pilot remains exploratory and must be labeled as such; substantial qualification generation requires a separate user decision after the concrete run manifest and cost estimate are reviewable.

### Gate G4 — analytical and standardized model conformance

- [x] **G4.01** TDL A–E nominal tap delays and powers match pinned resource tables (CPU-native Sionna 2.2.0/TR 38.901 19.2 check verifies SHA-256 digests for all five versioned resources, independently converts tabulated dB powers to normalized linear PDPs, combines the colocated specular/diffuse first tap for D/E, and compares expected powers and scaled delays with the model; `tests/test_integrations/test_tdl_profile_tables_native.py::test_tdl_19_2_nominal_parameters_match_hash_pinned_resource_table`).
- [x] **G4.02** scalable TDL A–E tap delays change correctly with configured RMS delay spread (CPU-native Sionna 2.2.0 test checks the generated weighted RMS and every tap delay scale linearly at 30/100/300/1000 ns for A–E; the normalized resource-table RMS ratio is retained, including the small D-profile deviation from rounded tabulated values; `tests/test_integrations/test_tdl_profile_tables_native.py::test_scalable_tdl_delays_have_configured_power_weighted_rms`). Fixed-delay TS 38.101-4 profiles A10/A30/B100/C60/C300/D10/D30 are outside these scalable A–E checks.
- [x] **G4.03** CDL A–E angular/delay nominal params match pinned tables and defined array orientations (the existing native 20-case A–E × 30/100/300/1000 ns matrix checks scaled delays, normalized powers, profile angles/XPR, and LOS K partition against Sionna 2.2.0/TR 38.901 19.2 resources; a CPU adapter-to-native comparison covers nonzero UT/BS rotations, both link directions, CDL-A/D, and a rotated-array control in `tests/test_integrations/test_cdl_native.py::test_cdl_adapter_applies_defined_array_orientations_on_cpu`; this is a bounded adapter/profile contract, not exhaustive antenna-pattern or angular-distribution conformance).
- [x] **G4.04** model-D/E specular/scattered power partition and configured K settings produce expected empirical behavior (a seeded CPU Monte Carlo regression checks table-derived K factors, mean power for every tap, and the LOS first-tap fourth moment under the finite 20-sinusoid Rician model for both profiles D/E; `tests/test_integrations/test_tdl_profile_tables_native.py::test_tdl_los_specular_and_diffuse_power_partition_matches_rician_moments`; this is a deterministic implementation-consistency check, not a confidence-interval qualification of the full channel distribution, which remains under G4.07).
- [x] **G4.05** 3GPP scenario state and profile category are not written into a `geometric_los` target field (TDL/CDL adapter metadata declares geometric LOS unavailable; standardized channel records reject non-null geometric-LOS labels; learning views emit only the separately named `profile_los_proxy`; standardized PUSCH views keep that proxy separate from channel-CFR truth and provenance retains a null geometric label. Exact target-key assertions cover both A/B/C and D/E profile classes and the TDL/CDL-to-PUSCH path in `tests/test_datasets/test_standardized.py::test_profile_los_is_not_geometric_los_and_family_category_is_checked` and `tests/test_integrations/test_standardized_pusch_native.py::test_standardized_tdl_cdl_to_pusch_dmrs_is_seeded_and_target_is_separate`). This establishes the standardized-profile path contract; it does not validate geometric labels from RT or measured-data sources.
- [x] **G4.06** independent batch draws and temporal evolution samples are explicitly distinguished (TDL/CDL adapter metadata identifies the batch axis as independent stochastic draws and the time axis as samples within one channel draw; records include sample interval and whether temporal evolution is observed. CPU-native moving/static tests check seeded replay, changed-seed and cross-batch distinction, within-draw time behavior, and standardized-store round-trip retention in `tests/test_integrations/test_channel_draw_semantics_native.py::test_batch_draws_and_time_samples_have_distinct_semantics`. This labels generator semantics; it does not by itself establish statistical independence confidence, which remains part of G4.07).
- [x] **G4.07** empirical delay, Doppler, spatial and covariance distributions have confidence intervals under independent draws (CPU-native `tests/test_integrations/test_g4_statistical_confidence_native.py`: per-tap PDP and ensemble RMS-delay intervals cover TDL/CDL A–E over 4,096 independent batch draws per profile; paired-time Doppler-correlation intervals cover TDL A–E over 8,192 independent draws, with Jakes and Rician references; spatial covariance checks cover a configured complex 2×2 TDL-A receive correlation, one complex 4×4 TDL-A receive-correlation matrix with simultaneous 99% Bonferroni-adjusted normal intervals over its real/imaginary entries (8,192 independent draws), and a two-element CDL-A array. Tap, time, and antenna coordinates remain paired within each draw. The 3.5-SE and asymptotic simultaneous intervals are simulation-model conformance checks for these specified profiles/configurations, not evidence for arbitrary arrays, cross-polarization, or RT/material realism.)
- [x] **G4.08** TDL/CDL restrictions (valid carrier, BW, array/model limitations) are checked at standardized-record admission (`datasets/standardized.py::validate_standardized_channel_admission`, called by `StandardizedChannelRecord`). The declared sampled bandwidth is `num_subcarriers × subcarrier_spacing_hz`. Native Sionna 2.2 adapter tests in `tests/test_integrations/test_g4_admission_native.py` verify multi-antenna TDL/CDL records and that out-of-range diagnostic generation is rejected at admission. Dataset contract tests in `tests/test_datasets/test_standardized.py` accept carrier endpoints 0.5/100 GHz and 2 GHz sampled bandwidth, reject outside carrier/bandwidth limits and inconsistent sampled-bandwidth provenance, verify supported table versions, TDL correlation metadata and CDL panel counts against antenna axes, and preserve the single-TX/single-RX link restriction. The 0.5–100 GHz / ≤2 GHz applicability limits and single-link model limitation are stated by the [Sionna 2.2 TDL documentation](https://nvlabs.github.io/sionna/v2.2.0/phy/api/channel/wireless/sionna.phy.channel.tr38901.TDL.html) and [CDL documentation](https://nvlabs.github.io/sionna/v2.2.0/phy/api/channel/wireless/sionna.phy.channel.tr38901.CDL.html); those classes do not enforce the frequency/bandwidth limits themselves.
- [x] **G4.09** profile-only classification is explicitly scoped to the A/B/C-versus-D/E `PROFILE_PROXY` target, not geometric LOS. `experiments/profile_holdout.py::run_profile_holdout` emits a metadata-only control using only `metadata.profile`; `tests/test_experiments/test_profile_holdout.py::test_profile_holdout_keeps_outer_profiles_and_validation_groups_disjoint` verifies its exact label recovery, confusion matrix, label semantics, and explicit `physical_geometric_los_inference=false` interpretation. The perfect control score is expected because profile metadata defines this proxy target; it provides no evidence of geometric LOS detection or receiver-observable performance.

### Gate G5 — NR waveform, channel and receiver correctness

- [ ] **G5.01** generated pilot/resource grid conforms to specified applicable TS 38.211 version and Sionna native API; no generic comb pattern called NR PRS (native CPU Sionna grid/FFT/DMRS-axis checks cover 15/30/60/120-kHz and 2/6/12/24-RB configurations in `tests/test_integrations/test_nr_pusch_native.py::test_native_pusch_grid_axes_and_fft_mapping_across_numerologies`; `test_pusch_type1_dmrs_positions_match_ts_38211_full_slot_reference` verifies Type-1 full-slot mapping-A DMRS positions and RE comb against TS 38.211 V19.1.0 tables; `test_pusch_dmrs_values_match_independent_ts_38211_gold_sequence` checks actual Sionna pilot values against an independent Gold-sequence implementation for inherited cell identity and explicit `N_ID`/`n_SCID` cases. Cell and DMRS scrambling identities now travel in `NRPUSCHConfig` and receiver metadata; transform precoding, frequency hopping, multiport, other DMRS types/mappings, and broader conformance remain open).
- [x] **G5.02** effective subcarrier indices map exactly to transmitted FFT bins and receiver channel estimate without DC/guard shifts (native CPU, noiseless alternating adjacent-bin channel checked at DMRS tones for 2/6/12/24-RB PUSCH grids at 15/30/60/120 kHz; `tests/test_integrations/test_nr_pusch_native.py::test_pusch_dmrs_estimator_recovers_frequency_selective_noiseless_channel_on_cpu`; fractional allocations and partial-band/BWP offsets remain open).
- [x] **G5.03** PUSCH perfect-channel/noiseless case behaves as expected under valid CP and waveform settings (native CPU 6-RB waveform round trip, perfect unity CSI, exact payload recovery, all TB CRCs pass; `tests/test_integrations/test_nr_waveform_impairments_native.py::test_pusch_cpu_waveform_roundtrip_decodes_noiseless_payload`; not multipath or impairment qualification).
- [x] **G5.04** CP-exceedance stress reveals physically plausible ISI/ICI where time-domain model predicts it; static frequency multiplication baseline is not used outside its assumptions (the analytic inside/outside-CP fixture in `tests/test_integrations/test_nr_waveform_impairments_native.py::test_pusch_diagonal_ofdm_channel_is_valid_inside_cp_and_fails_beyond_cp_on_cpu` is complemented by standardized TDL path-delay application in `tests/test_integrations/test_standardized_pusch_native.py::test_short_cp_time_channel_changes_pusch_observation_vs_ideal_cfr_control`; a longer-CP 24-RB TDL/CDL fixture checks CFR estimation when the discretized tap span fits the CP. Scope remains static snapshots and tested grids, not mobility or multipath BLER/CRC qualification).
- [x] **G5.05** OFDM symbol clock and RT Doppler sampling use correct real duration and frequency units (normal-CP timing tests, 30-kHz explicit-time Doppler phase contract, and native CPU RT trajectory symbol-time provenance; `tests/test_integrations/test_nr_timing.py`, `tests/test_integrations/test_rt_to_nr_contract.py::test_rt_doppler_uses_cp_aware_nr_symbol_start_times_in_seconds`, `tests/test_integrations/test_rt_trajectory_native.py::test_cpu_rt_path_retraces_and_stores_receiver_only_trajectory`; does not close RT solver qualification).
- [x] **G5.06** physical thermal-noise PSD to complex sample covariance matches injected and measured covariance statistically (proper-complex helper statistical test plus CPU absolute-power no-path RT observation checks `kTB·F`, component variance, properness, and adjacent frequency/symbol covariance; `tests/test_integrations/test_receiver_noise.py::test_thermal_noise_injection_has_declared_complex_sample_covariance`, `tests/test_integrations/test_rt_trajectory_native.py::test_absolute_power_policy_generates_all_no_path_noise_only_episode`; current scope is ideal independent complex-RE receiver noise, not analog filtering or correlated antenna noise).
- [x] **G5.07** precise SNR definition (pre-/post-beam, per RE/antenna, plus NF) travels with each view (ideal standardized PUSCH AWGN mode records aggregate post-channel/pre-beam grid SNR and explicitly records no NF; physical thermal-noise qualification remains G5.06).
- [x] **G5.08** absolute path loss retained in physical receiver view; normalized diagnostic explicitly separate (native CPU analytic unit/half-gain RT-DMRS views with fixed transmit power and thermal noise preserve the physical 0.5 ratio while normalized per-link observations match; `tests/test_integrations/test_rt_dmrs_observation_native.py::test_absolute_power_dmrs_mode_preserves_link_gain_with_fixed_thermal_noise`; scene-level propagation/path-loss accuracy remains a separate RT qualification question).
- [ ] **G5.09** CFO/phase-noise/timing/IQ impairments injected in physically consistent order with known reference (partial: ordered deterministic composition, seeded Wiener phase-noise statistics, shared continuous-time receiver oscillator across antenna elements, static standardized-CIR waveform application with CP spillover/ISI, CP-supported and repeated CP-exceeding TDL decode/CRC, paired high-SNR impairment-induced TDL PUSCH LS-NMSE impact, zero-impairment equivalence, and impairment-view seed/parameter provenance are verified; mobility and broader impairment coverage remain open). The exploratory [`standardized_pusch_cp_exceedance.py`](../../experiments/standardized_pusch_cp_exceedance.py) probe records 16 independent TDL-A channel draws at 3.5 GHz, 30-kHz SCS, 6 RB, 1-µs delay spread and 60-dB declared receive-grid SNR. Maximum excess delay is 9.66 µs versus the recorded 3.24-µs CP. CRC/bit failures were 11/16 (Wilson 95% interval 0.444–0.858) for the sampled time-domain channel with DMRS LS CSI; 7/16 (0.231–0.668) with its diagonal-CFR oracle; 9/16 (0.332–0.769) for the paired ideal-CFR diagnostic with DMRS LS; and 0/16 (0–0.194) for ideal-CFR plus diagonal-CFR oracle. The probe now also reports paired time-domain-minus-ideal-CFR BLER intervals by resampling the independent channel-parent groups while retaining each parent's paired receiver outcomes; repeated child rows do not increase the group count. These small-sample outcomes show receiver sensitivity and separate estimation error from diagonal-model mismatch for this frozen case; the ideal-CFR and oracle conditions are diagnostics, neither is physical truth, and the probe does not establish a general BLER curve or close G5.09. The original machine-readable run is [`g5-standardized-pusch-cp-exceedance-20261023.json`](results/g5-standardized-pusch-cp-exceedance-20261023.json); the cluster-bootstrap analysis replay is [`g5-standardized-pusch-cp-exceedance-cluster-bootstrap-20261023.json`](results/g5-standardized-pusch-cp-exceedance-cluster-bootstrap-20261023.json).
- [x] **G5.10** LS estimator bias/variance and LMMSE noise behavior verified against analytical fixtures (native CPU LS pilot and correlated-noise 3×2 LMMSE cases; interpolation-wide behavior remains open).
- [x] **G5.11** estimated CSI stored on correct antenna/layer/symbol/frequency axes, with pilot mask and observation missingness (typed mask/schema and NR-store round trip; `tests/test_datasets/test_nr_observation_mask.py`; physical missingness-generation mechanisms remain separate).
- [x] **G5.12** same parent channel with different noise seeds has independent receiver views but identical hidden propagation truth (standardized TDL PUSCH CPU fixture; `tests/test_integrations/test_standardized_pusch_native.py::test_independent_noise_views_keep_waveform_parent_and_channel_truth`; full G5 remains open).
- [x] **G5.13** BLER/BER/CRC/beam outcomes derived from real matched receiver pipeline, not posthoc channel score alone (native CPU Sionna PUSCH transmitter/`PUSCHReceiver` AWGN trend in `tests/test_integrations/test_nr_waveform_impairments_native.py::test_pusch_frequency_grid_decoder_has_correct_high_vs_low_snr_trend`, plus exact payload/CRC recovery for a standardized CP-supported TDL path using its DMRS LS estimate in `tests/test_integrations/test_standardized_pusch_native.py::test_standardized_time_domain_tdl_pusch_decodes_with_dmrs_estimate_on_cpu`; CP-exceeding BLER/CRC and broader curves remain open).
- [ ] **G5.14** PRS/SRS/SSB implementations only admitted after independently validating mapping and observable timing claims.
- [x] **G5.15** no future/fixed oracle CSI is fed into ordinary model input adapter (adversarial NR record with extreme oracle target and metadata exposes only typed received-grid/receiver-estimate fields; `tests/test_datasets/test_nr_model_input_firewall.py::test_oracle_csi_targets_and_metadata_never_enter_nr_model_inputs`; corpus-wide and downstream-loader isolation remains G7 scope).

### Gate G6 — trajectories, forecasting and interventions

- [ ] **G6.01** time series increasing and synchronized; expected channel coherence fits chosen time grid.
- [ ] **G6.02** `t+h` targets align by timestamps (not arbitrary frame offset when gaps exist).
- [ ] **G6.03** estimated Doppler agrees with time evolution under short stationary-path conditions.
- [ ] **G6.04** short-horizon Doppler-only evolution compared with physical re-tracing across displacement in wavelengths.
- [ ] **G6.05** physical blockage intervention changes full path set and creates/destroys indirect components where geometry predicts.
- [ ] **G6.06** no path index tracking assumption across retraces; use association uncertainty and birth/death logic.
- [ ] **G6.07** future phase comparison uses a consistent clock reference or phase-invariant metric.
- [ ] **G6.08** future event right-censoring handled (no label leakage from 'not observed' to 'did not occur').
- [ ] **G6.09** no frames from same route, parent physical realization, or intervention pair straddle split.
- [ ] **G6.10** causal prediction only uses interventions known at forecast time, not post-event geometry if claiming deployment realism.

### Gate G7 — storage, label validity, privacy, split integrity

- [ ] **G7.01** all required fields present and typed; missing target fields carry reason codes rather than invented zero.
- [ ] **G7.02** each numeric target's units/gauge/aggregation source is registered and versioned.
- [ ] **G7.03** nulls/infs/NaNs, negative powers, invalid durations, missing array axes handled explicitly.
- [ ] **G7.04** target-mask shapes broadcast correctly and selection counts reported.
- [ ] **G7.05** empty physical channels are valid records; loss functions handle them without divide-by-zero.
- [ ] **G7.06** immutable writes verified by re-read and SHA checks including compressed chunk integrity.
- [ ] **G7.07** input adapter cannot access oracle labels, file paths, source profile category or scene IDs unless experiment explicitly allows them.
- [ ] **G7.08** train/val/cal/test split lock signed before model tuning; neighboring spatial buffers and overlapping time windows detected.
- [ ] **G7.09** counterfactual twins, noise variants, resampled bandwidth copies and repeated routes inherit parent split.
- [ ] **G7.10** near-duplicate detection using geometry/site hash and channel correlation flags suspicious cross-split items.
- [ ] **G7.11** measured license and non-redistribution enforced by export routine, not only written documentation.
- [ ] **G7.12** compressed data does not materially change physical target values; dtype and quantization error audited.
- [ ] **G7.13** dataset can be loaded in shards with consistent deterministic order and no process-wide 100 GB materialization.
- [ ] **G7.14** target-level valid-count coverage report generated for every split and family.
- [ ] **G7.15** released dataset 'main' tests do not silently combine source-family-specific labels into fake universal classes.

### Gate G8 — independent statistical and fairness checks

- [ ] **G8.01** held-out scenes have both label classes for AUROC or explicitly mark AUROC undefined.
- [ ] **G8.02** effective group sample counts used for CIs; not millions of correlated frames treated as millions of trials.
- [ ] **G8.03** natural deployment priors and balanced stress priors evaluated separately; calibration corrected only with documented priors.
- [ ] **G8.04** missing/low-confidence oracle items tracked and selection-bias audited.
- [ ] **G8.05** dataset release includes at least one unseen site/material/antenna domain and a frozen external measured evaluation appropriate to measured fields.
- [ ] **G8.06** calibrations, normalization transforms and synthetic material fits trained on train-only data (or explicitly on calibration partition), never on test.
- [ ] **G8.07** independent classic estimator baselines (PDP, peak delay, MUSIC/ESPRIT where applicable, OMP/SAGE/matched-filter etc.) used to check whether difficult ground truth is actually observable.
- [ ] **G8.08** null/falsification controls: time shuffled, phase randomized, source-ID-only, power-only, metadata-only, label-permuted and oracle-leak checks.
- [ ] **G8.09** preregistered threshold selection and calibration do not query sealed test outcomes.
- [ ] **G8.10** externally measured artifacts are independently source-verified, correctly timestamp-aligned and have explicit permitted target labels.

**Gate workflow:** `G0–G2` before any physical data batch; `G3` scene/frequency/solver convergence; `G4–G5` standardized and receiver admission; `G6` dynamic dataset admission; `G7` each shard/release; `G8` before benchmarking and paper figures.

---

## 14. Dataset split program: protect the generalization claims

### 14.1 Four mandatory independent split axes

1. **Geographic/site:** all geometry assets, derived variants and trajectories of unseen cities/buildings excluded from training.
2. **Temporal/session:** complete motion sessions and close-by time windows grouped, with gap/embargo for repeated physical capture sessions.
3. **Physical model / material:** hold out distinct material property families and solver-scattering settings; separate calibration of known vs unknown materials.
4. **Equipment/channel configuration:** antenna designs/array geometries, carrier, bandwidth, receiver hardware and/or noise regimes unseen in training.

Combinations of these four axes provide a harsher OOD benchmark. Never merely randomize frames and describe the result as unseen environment.

### 14.2 Canonical released benchmark partitions

| Benchmark view | Train | Validation and calibration | Locked test | Claim tested |
|---|---|---|---|---|
| `B00-analytic` | Analytical synthetic sets | Independent parameter draws | Exact close/separated path fixtures | Physical constraints and inverse identifiability. |
| `B01-profile` | TDL/CDL profiles | Held-out draw seeds/speeds | Held-out profile or delay spread | Model generalization in standardized channels, not geometric LOS. |
| `B02-rt-in-domain` | Multiple physical scene episodes | New episodes, buffer-separated | Independent links **within same sites** | In-domain RT inference. |
| `B03-rt-site-ood` | Multiple site families | Different sites not used in train | Entire independent sites/buildings | Spatial/geometry transfer. |
| `B04-material-ood` | Known material recipes | Unseen material-family calibration | New frequency/material combination | Material robustness. |
| `B05-array-ood` | Trained array types | Held-out array calibration | New array/polarization hardware | Spatial representation transfer. |
| `B06-time-forecast` | Independent trajectories | Validation routes and time blocks | Sealed trajectories with moving blockers | Causal future channel/blocked-state forecasting. |
| `B07-stress-tail` | Baseline priors | Predefined thresholds | Rare low-SNR, weak LOS, outage, aliasing & clock stress | Worst-group/uncertainty and abstention. |
| `B08-physical-counterfactual` | Some matched interventions | Different interventions/sites | Unseen scene/object intervention families | Physical sensitivity and causal generalization. |
| `B09-measured-external` | Simulated + authorized measured train if stated | Measured calibration sites separate | Sealed independent measured campaigns | Simulator-to-real transfer **only for available measured labels**. |
| `B10-link-utility` | Train inference model on allowed views | Select downstream policy | Held-out link trajectories/service cases | Beam/BLER/positioning benefit with realistic overhead. |

### 14.3 Split allocation rules

- Freeze site and group assignments before sample generation; specify intended site and group counts after pilot dataset power/sensitivity analysis.
- TDL/CDL standardized profile holdouts also keep channel draw seed and all receive-noise variants with same split.
- Scene variants with common base mesh/geometry must share their parent `asset_family_id` across splits unless the experiment is explicitly **within-family perturbation**, not unseen-site transfer.
- Use a spatial boundary buffer and test near-neighbor channel correlation/position overlap; a meter-scale buffer is a *diagnostic*, not universal independence guarantee.
- External measurement train/cal/test divisions must obey each campaign's UE, O-RU, protocol and timeseries structure; if site and UE are confounded, say so instead of claiming factor-separated generalization.
- No trained feature scaling, calibration, learned K-threshold, label-distance weighting or synthetic scene material tuning uses sealed test statistics.
- Report group-specific held-out class counts and effective independent sample sizes. Classes entirely absent in a fold make certain metrics undefined.

---

## 15. What makes the dataset scientifically 'best' — release acceptance criteria

**No single dataset can be shown best in all respects.** Adopt **multi-objective quality**, not a score that rewards sheer size.

| Quality dimension | Required measurable evidence | Failed state |
|---|---|---|
| Physical fidelity | Analytic free-space tests, full-path conservation, calibrated geometry/materials, solver convergence | Arbitrarily many paths/frames with wrong phase or unqualified scene. |
| Observational realism | Correct PUSCH pilots, physical power/noise, impairment and receiver variants, CP/motion audit | Unrealistically perfect CSI disguised as raw reception. |
| Breadth | Distinct geometry families, bands, arrays, blockers, delay regimes, indirect-dominant LOS and NLOS | A few procedural clones with many samples. |
| Rare-event support | Explicit count and uncertainty for weak LOS/early blockage/outages | Inference fails exactly where deployed. |
| Structured labels | Validated path/CSI/angles/clock references/target semantics | Physically inconsistent multitask targets. |
| Independence | Hash/buffer/time/counterfactual split QA, enough independent sites/trajectories | Inflated accuracy through spatial or temporal leakage. |
| Identifiability | Known limits vs bandwidth/aperture/SNR/gauge and calibrated abstention | Algorithm penalized for fundamentally impossible regression. |
| Reproducibility | Exact dependency/scene/code/seed/backend locks, checksums, hashes, deterministic DAG | Non-reconstructable plots and outputs. |
| Transferability | OOD sites/materials/arrays and appropriate measured data | Only train/val random split performance. |
| Scientific utility | Multiple tasks and genuine downstream benefit | Label encyclopedia with no research question. |
| Ethics/licenses | Permission and secure storage of measured material | Illegal redistribution or unknown source. |
| Maintainability | Reader APIs, schema versions, migration and validation | Single fragile notebook / giant `.npz`. |

### 15.1 Predeclare quantitative criteria, don't invent success thresholds

For **engineering invariants** choose tolerances from dtype, numerical reference and calibrated accuracy specification. For **physical-statistical convergence**, use sequential/independent reference runs and define target-dependent confidence widths. For **research metrics**, declare minimum practically meaningful improvement *before* sealed test evaluation and report confidence intervals even when the method loses. For **dataset coverage**, set independent site/group targets after pilot variance estimates and available compute budget.

### 15.2 Independent qualification of geometry/material fidelity

- Minimum acceptable first release: diverse **plausibility-audited** urban and indoor scenes, with explicit `not_field_calibrated` tag.
- Better release: material parameters calibrated using independent measured path-loss/PDP/angle data, with calibration and evaluation sites separated.
- Strongest release: multiple independent acquisition campaigns with geometry/antenna/clock metadata and nontrivial measured validation tasks; where ground truth impossible, publish only observable metrics.

### 15.3 'Not bigger than necessary' stopping criterion

Stop adding frames when marginal contribution is negligible to: (a) new independent group coverage, (b) rare-event support, (c) uncertainty precision, (d) acquisition diversity or (e) physical model QA. Invest next resources in an independent **new site**, measured calibration or complex receiver fidelity rather than adjacent repeated samples.

---

## 16. Storage and compute planning without invented throughput estimates

### 16.1 Physical size accounting formulas

For dense complex CFR with dimensions `[N_frames, N_rx, N_rx_ant, N_tx, N_tx_ant, N_freq]`, `complex64` storage before compression ≈ `8 × product(dimensions)` bytes. For realistic channel IQ with `[N_frames, N_rx, N_ant, N_samples_per_frame]`, also 8 bytes per complex sample. Path tables add `O(N_valid_paths × fields)` storage and metadata. Full IQ can dominate by orders of magnitude over channel statistics.

**Worked accounting example — hypothetical, not generated:** 100,000 frames × 4 Rx elements × 1024 frequencies × 8 B ≈ 3.28 GB for just one TX element and one complex CFR time slice, **before** time expansion, multiple Tx antennas, redundant receivers, masks, labels or raw waveform. A million such frames are ≈ 32.8 GB; adding 14 OFDM symbols multiplies this component by ~14. These arithmetic examples must not be mistaken for performance or planned count guarantees.

### 16.2 Three-tier storage policy

- **Tier A — all accepted physical episodes:** retain scene identity, core path oracles, compact true CIR/CFR, receiver estimated CSI, targets/masks/metadata. Do not keep all raw IQ for all noise/receiver variants.
- **Tier B — stratified waveform subset:** store all raw IQ and detailed synchronization/decoding intermediates on enough independent conditions to validate realistic receiver performance.
- **Tier C — deep-physics diagnostics and measured captures:** exact high-resolution ray/path metadata, float64 references, giant dense multi-antenna time tensors, material/scene calibration and controlled rare-event capture. Store with strict access and retention policy.

Parent physical event is reused by derived observations; additional SNR seeds have low incremental cost when waveform is generated from saved H. Beware that physically changing antenna/frequency may require **re-solving** RT instead of naively rescaling saved H.

### 16.3 Estimating required compute

Measure on each actual native backend:

`cost_per_episode = scene_load + geometry_oracle + Σframes(trace_runtime) + path_extraction + channel_view + NR_receiver_views + QA + storage`.

Record p50/p90/p99 by scene, LOS state, ray budget, array and path density, not one optimistic mean. Project total runtime as sample counts weighted by actual observed class/scene distribution; add calibrated memory bound/queue time/retry overhead. Compute quota / GPU RAM / storage may force reducing dense ray precision or keeping only selected time windows. Evaluate LLVM CPU option where OptiX unavailable, with exact backend identity and convergence.

### 16.4 Staged scale proposals (not promised sample counts)

| Stage | Source/group target | Approximate workload design | Promotion criterion |
|---|---|---|---|
| `DEV-0` | Analytic and native 1-link fixtures | Dozens of cases | All G0/G1/G2 core invariants pass. |
| `PILOT-1` | 3 independent site families; balanced rare cases for QA | Hundreds of independent episodes, limited sequence length | Physics ray-depth/budget convergence, output shape and costs measured. |
| `PILOT-2` | 6+ genuinely different sites, source families and receiver variants | Thousands of grouped episodes; natural and balanced views | Group split audit; uncertainty estimates stable enough to plan scale. |
| `CORE-1` | Independent site/scene asset inventory beyond pilot; realistic mobility | Scale based on effective sample-size calculations and runtime budget | All G0–G8; locked full-target validity/coverage, no dataset leakage. |
| `MEAS-1` | Multiple external measured campaigns if legally/technically feasible | Each campaign separate with own labels | Modality/license consistency; independent transfer evidence. |
| `CHALLENGE-1` | Sealed new sites/hardware/obstruction configurations | Rare-case and high-resolution test only | Unseen conditions remain untouched by model/scene calibration. |

### 16.5 Required resource profiling outputs

`ray_runtime_vs_depth.csv`, `ray_runtime_vs_samples.csv`, `memory_peak_by_backend.csv`, `path_count_vs_budget.csv`, `rt_power_convergence.json`, `receiver_runtime_by_waveform.json`, `storage_ratio_by_tier.json`, `accepted_per_hour_by_scene.csv`, `outage_vs_solver_failure.json`, `shard_throughput.json`, `cost_sensitivity.md`. Each includes hardware/backend/versions, repeated-run variance, timestamp and raw source hashes.

---

## 17. Task-to-data eligibility matrix — one registry, task-specific correct views

Legend: `Y` scientifically grounded label possible with mandated QC, `C` conditional on parameterization/observable, `P` proxy only, `–` unavailable without invention.

| Research output | F00 analytical | F01 TDL/CDL | F02 scenarios | F03 RT static | F04 RT dynamic | F05 intervention | F06 NR child | F08 measured |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Geometric LOS | C | – | – | Y | Y | Y | parent | C (independent annotations) |
| Profile-category/3GPP LOS | – | P | P | – | – | – | parent | – |
| Direct-path detectability | Y | C | C | Y | Y | Y | Y | C (reference) |
| Outage/received power | C | C* | C | Y | Y | Y | Y | Y if measured |
| RMS/mean delay spread | Y | Y | Y | Y | Y | Y | Y receiver-estimated | C (wideband complex/time) |
| First-path relative delay | Y | C | C | Y | Y | Y | Y | C (timing reference) |
| Absolute ToA | C (known clock) | – | C | Y oracle | Y oracle | Y oracle | C (synchronized receiver) | C (timing/clock truth) |
| PDP/channel reconstruction | Y | Y | Y | Y | Y | Y | Y | C (magnitude/CSI depending source) |
| Path/cluster order | Y | P | P | C (resolution-defined) | C | C | C | – absent independent ref |
| Angular state | Y (arrays) | C (CDL array) | C | Y | Y | Y | C (arrays) | C (calibrated array/phase) |
| Doppler spread | Y | C | C | C (evolution) | Y | Y | C (temporal) | C (temporal) |
| Future CSI | C | C (channel evolution) | C | – (static alone) | Y | Y | Y (future observations) | C (time-aligned measured) |
| Future LOS/blockage | – (special toy) | – | C model state | – | Y | Y | Y parent+future | C only with annotations |
| Beam selection/outage | C | C | C | C (array) | Y | Y | Y | C (beam metadata) |
| BLER/BER | – until waveform | – until waveform | – until waveform | – until waveform | – until waveform | – until waveform | Y | C if available |
| Identifiability/calibration | Y | C | C | Y | Y | Y | Y | C (reference needed) |

`*` TDL/CDL normalize PDP power; absolute received-power labels require **additional explicitly documented** path-loss and link-budget process, not an assumed intrinsic link loss.

### 17.1 Task selection for a publishable first paper

**Primary (must work):** `geometric_los` and `detectable_direct_path`; RMS delay spread, early/relative path delay, dominant received path power; bounded two-/variable-path reconstruction and uncertainty; reconstructed sparse complex receiver CSI.

**Second layer (when validated):** AoA/angular spread for MIMO; Doppler & time-varying path state; path resolvable order; K-like ratios with valid definitions.

**Third layer (requires retraced motion):** multi-horizon future CSI + blockage/beam and service outage; event predictions and uncertainty.

**System-level proof:** beam failure or link BLER reduction and realistic positioning gain, with oracle/detector-only/MLP baselines and measured external validation where available.

**Avoid one headline aggregate 'everything score'.** Separate physical quantities, meaningful unit metrics and a prespecified task-macro summary only if scaling/weighting is principled.

---

## 18. Mandatory extension: every antenna and multi-node topology (not just fixed 2×2 MIMO)

This section overrides any earlier shorthand that implied all episodes contain one transmitter and one receiver. UMCD-1 is a **network-aware channel dataset** with arbitrary numbers of network devices and a variable number of physical antenna ports **per device**. The generator must never silently reshape, broadcast, repeat or truncate nodes, antennas, polarization ports, channels or frames merely to fit a particular model.

### 18.1 Exact convention and topology coverage matrix

**Convention for all tables in this section:** `Ntx × Nrx` is the *number of transmitted and received antenna ports for a single directed link*, **not** the number of network nodes or spatial layers. Example `4×2 MIMO` = one source link with four TX ports and one destination link with two RX ports. A separate `2 BS × 6 UE` describes six nodes receiving from two distinct network nodes. Port count, radiating elements, panels, RF chains, layers and device count are different fields.

| Category | Canonical link | Minimum release configurations | Additional variants | Study / caveat |
|---|---|---|---|---|
| SISO | 1×1 | 1×1, single- and dual-band scenes | all carrier/bandwidth and motion variants | Full absolute complex scalar channel, direct+indirect paths, outage. |
| SIMO | 1×N | 1×2, 1×4 | 1×8, 1×16, dual-pol RX | AoA, diversity, spatial combining; frequency and calibration defined. |
| MISO | N×1 | 2×1, 4×1 | 8×1, 16×1, dual-pol TX | AoD, transmit beamforming; no invented RX angular aperture. |
| SU-MIMO | N×M | 2×2, 4×2, 4×4 | 8×4, 8×8, 16×4 | Complex full channel matrix, eigenvalues/rank/condition number, spatial correlation. |
| Massive MIMO | many BS ports × few UE ports | 16×2, 32×4 as staged pilots | 64×4, 64×8, 128×8, 128×16 only if validated | Strong synthetic-array approximation warning; aperture near-/far-field diagnostics. |
| Dual-polarized MIMO | two polarizations per relevant element | 2×2, 4×4 with port-to-element mapping | ±45°, V/H, cross-polarized arrays, XPR | Keep port dimension and physical polarization basis, not multiply antennas blindly. |
| Multiuser MIMO | one BS ↔ several UEs | 1 BS, 2 / 4 UEs | 8 / 16 UEs; multiple scheduled streams | Joint channels, shared BS, independent UE RF phases, precoder/combiner and scheduling. |
| Multi-BS / multi-UE | multiple BSs and UEs | 2 BS × 4 UE | 3 BS × 8 UE, 7-cell cluster (conditional) | Serving/nonserving links, intercell interference, associations/handover. |
| Distributed / CoMP | several cooperating APs ↔ UEs | 2 AP × 1 / 2 UE pilot | 4 / 8 AP × 4 UE | Network synchronization and joint processing explicit; without clock truth, cooperative phases not identifiable. |
| Cell-free MIMO | geographically distributed APs | research-only after native multi-BS pilot | network heterogeneous arrays, opportunistic subset | Do not infer phase-coherent coordination without cross-site clocks/calibration. |
| Full-/half-duplex & UL/DL | directed waveform per link | DL and UL separate experiments | TDD pairs, FDD separate frequencies; bidirectional connectivity | UL↔DL transposition only under appropriate propagation conditions, clocks and RF calibration. |
| Multi-band / carrier aggregation | independent frequency-specific link realizations | single n78 3.5 GHz first; same geometry at 2.6 / 28 GHz staged | paired FR1/FR2 and multi-carrier views | Frequency-dependent materials/arrays/path visibility; no unjustified absolute phase coherence. |
| Dense interference | several co-channel active TX nodes | 1 interferer + 1 wanted TX | multiple synchronized/asynchronous interferers, adjacent-channel waveforms | Path truth must include *all* contributors; SINR is not channel SNR. |
| Relay / multi-hop | two or more serial links | optional after all link stages pass | decode/forward, amplify/forward | Node state and relay processing necessary; two propagation hops are not one ray-traced path. |
| RIS/reconfigurable surfaces | specialized programmable scattering | **not core** | optional only with separately verified EM/RT model | Ordinary flat passive materials do **not** imply a validated RIS model. |
| XL-/near-field MIMO | very large aperture | conditional targeted subset | spherical wave/visibility regions, element-specific rays | May require explicit arrays and high-fidelity calibration; synthetically traced center-ray approximations can fail. |
| ISAC / sensing | communication plus target returns | optional F09, separate | bistatic/monostatic RCS | Sensing truth has different semantics and a separate path/waveform API. |

**Required base variety:** SISO, SIMO, MISO, square MIMO, rectangular MIMO, one example of dual polarization, one 2-BS/4-UE topology, downlink and uplink. Massive MIMO, advanced CoMP, and exotic geometry cannot become mandatory at an impossible memory/runtime cost; they remain **conditional release tiers** with an explicit readiness gate and declared missingness.

### 18.2 Sionna 2.2 capabilities: do not confuse the three channel generators

1. **TDL** (`sionna.phy.channel.tr38901.TDL`): according to 2.2 source, models **one transmitter and one receiver with multiple antennas at either end**; optional `num_tx_ant`, `num_rx_ant`, and spatial correlation matrices. It does not natively model a joint multi-BS network with physically correlated links. TDL A–E scalable profiles and additional conformance profiles `A10`, `A30`, `B100`, `C60`, `C300`, `D10`, `D30` are different families and require separate applicability labels.
2. **CDL** (`sionna.phy.channel.tr38901.CDL`): also **one TX–RX pair**, but it accepts array geometries, panel patterns, angle/cluster parameters, UL/DL orientation/direction and explicitly versioned tables. `CDL-A` to `E` includes cluster angular structure but **not** actual 3D building/intervention geometry. The source documents `a` shape `[batch,1,Nrx_ant,1,Ntx_ant,P,T]`, `tau` shape `[batch,1,1,P]`; preserve the singleton node axes, do not falsely imply multi-node independence.
3. **Full 3GPP scenario channels** (`UMi`, `UMa`, `RMa`, `InH`, `InF`): topology-aware models are appropriate for joint base-station/user configurations where supported by the pinned version. Validate link correlation, path loss, shadow fading, direction, blockage, LSP spatial consistency, antenna type, and each model's valid parameter range. `enable_spatial_consistency` and `enable_blockage` are not safely assumed enabled; some temporal behavior has limitations.
4. **Sionna RT `PathSolver`**: solves propagation between antennas of **all** scene transmitters and receivers. Native path tensors expose network node and antenna axes. The current Scene API exposes **one `scene.tx_array` shared across TX nodes and one `scene.rx_array` shared across RX nodes**. Arbitrary **heterogeneous device arrays within a single solver call are not promised** by this interface.
5. **Heterogeneous-array workaround**: partition nodes by validated TX-/RX-array-compatible groups, solve each group with the *same* scene realization/pose/material snapshot, reindex back into a global sparse link graph, and audit path availability, frame time, phase gauge, power scaling and solver variance across groups. A group split can destroy joint phase/visibility consistency; classify it as `grouped_solver_approximation` until independently validated. Prefer native support if a later version provides it.
6. **Explicit vs synthetic arrays**: `synthetic_array=True` traces from device centers and constructs element responses through phase shifts; `False` uses each element as an actual source/target. Include comparison subsets for different apertures, ranges and scattering complexity. A large array with `synthetic_array=True` is not automatically a realistic XL-MIMO simulation.
7. **Multiuser signal mixing is not automatic**: `PathSolver` computes propagation paths; downlink scheduling, transmit symbols, per-node clocks, beamforming, superposition, interference, RF chain behavior, receiver estimation and detection must be implemented and validated separately in PHY/receiver modules.
8. **No uniform `all channels` claim**: we cover all *specified* Sionna-supported TR 38.901 families, checked variants and selected nonstandard stress channels; we do **not** claim all conceivable electromagnetic phenomena or frequency regimes.

Authoritative source/API anchors:
- `TDL`: https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/tdl.html
- `CDL`: https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/cdl.html
- scenario channels: https://nvlabs.github.io/sionna/v2.2.0/phy/api/channel/wireless/3gpp.html
- `UMi` multi-topology: https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/umi.html
- RT `Scene`: https://nvlabs.github.io/sionna/rt/api/scene.html
- RT `PathSolver`: https://nvlabs.github.io/sionna/rt/api/paths_solvers.html
- RT `Paths`: https://nvlabs.github.io/sionna/rt/api/paths.html

### 18.3 Node and link data model (more general than a dense tensor)

Define distinct identifiers `network_id`, `site_id`, `scene_version_id`, `network_episode_id`, `node_id`, `array_id`, `port_id`, `carrier_id`, `directed_link_id`, `stream_id`, `beam_id`, `time_id`. None can be inferred from row order. Required tables:

- `node`: physical/logical role (`gNB`, `UE`, `AP`, `interferer`, `relay`), clock group, RF hardware identifier, max power, noise figure, duplex capability, array ID, valid interval.
- `array`: `array_id`, model, element positions in **meters** and wavelengths (reference frequency), element orientation/normal, Jones/polarization pattern, RF chain map, port mapping, mutual coupling/calibration status, array reference point.
- `trajectory`: time-stamped position/velocity/orientation per node **and** moving objects; coordinate frame and interpolation policy.
- `directed_link`: ordered `(source_node, target_node, band, carrier, time_interval)`; allowed TX/RX port sets, active/idle status, channel solver identity, antenna-group approximation flag.
- `association`: `UE ↔ serving BS`, primary/secondary/neighbor/interferer, scheduled cell, handover/beam transition, shared resources.
- `transmission`: payload, time/frequency/port/layer/beam/precoder, power control, PA/hardware model, waveform and reference-signal metadata.
- `network_channel`: materialized complex channel per **directed** link; sparse storage of link blocks; full network response reconstructed by link lookup, never a zero-filled implicit matrix of disconnected links.
- `interference_graph`: co-timed active TX nodes, resource overlaps, synchronization relation, received interfering channel and waveform, known cancellation status.
- `node_clock`: oscillator frequency, initial unknown phase, timing offset/drift, sampling-frequency offset, phase-noise evolution, sharing/calibration metadata.

Minimum node/port validation: every link references existing node and port IDs; transmitted port sets belong to the TX node, received ports to the RX node; unique physical antennas cannot accidentally be reused under contradictory coordinates; all nodes in a multiuser joint solve share a consistent world/time reference.

### 18.4 Canonical channel arrays and signal-equation contract

For one band, define the **uncombined physical propagation channel** for each directed link

\[
H_{r\leftarrow t}[k,n]\in\mathbb C^{N_{r,\mathrm{port}}\times N_{t,\mathrm{port}}}
\]

with reference grid subcarrier `k` and physical time `n`. Each matrix element is generated coherently from valid path coefficients with a **declared passband-to-baseband convention**, physical delay reference and unchanged link budget. In a network:

\[
\mathbf y_r[k,n]
= \sum_{t\in\mathcal A[k,n]}
\mathbf H_{r\leftarrow t}[k,n]\,
\mathbf W_t[k,n]\,\mathbf x_t[k,n]
+\boldsymbol\eta_r[k,n],
\]

where `A` is the set of actually active transmitters on the resource element, `W` is any declared precoder, `x` contains the **transmitted** symbols with exact physical power convention, and `eta` includes thermal and separately modeled receiver noise/distortion. Carrier synchronization, timing offsets, intercarrier interference, filtering and receiver front-end nonlinearities require an appropriate time-domain treatment; the equation above is a per-subcarrier abstraction only when its assumptions hold. Never add SNR in dB directly, or equate path power with received post-equalization SINR.

Recommended stored axes, keyed rather than assumed by tensor length:

```text
ORACLE_LINK_CFR:
  [episode, frame, directed_link, rx_port, tx_port, subcarrier]
ORACLE_LINK_PATHS (ragged):
  [episode, frame, directed_link, rx_port, tx_port, path_offset:path_offset+count]
NETWORK_LINK_SPARSE_INDEX:
  [episode, frame, carrier_id, directed_link] -> path/channel chunk URI
NR_RX_GRID:
  [episode, frame, rx_node, rx_port, ofdm_symbol, fft_bin]
NR_CHANNEL_ESTIMATE:
  [episode, frame, rx_node, rx_port, tx_node/stream, tx_port_or_layer, ofdm_symbol, effective_subcarrier]
BEAM_GAINS:
  [episode, frame, rx_node, tx_node, tx_beam, rx_beam, subcarrier_or_band]
```

A heterogeneous-array frame **cannot** generally be represented by a single dense `[Rx, A_rx, Tx, A_tx, F]` tensor without padding and masks. Store path/channel blocks per link or grouping of identical shapes, preserving port order and semantic axis names. Any dense batch view must return port masks, valid-link masks, active-transmitter masks, pilot observability masks and axis labels.

### 18.5 Nontrivial antenna/port/RF distinctions and analysis pitfalls

| Distinction | Must store | How it could invalidate a result |
|---|---|---|
| Radiating element vs antenna port | pattern, phase center, port weights, polarization | A two-pol element could be confused with two independent spatially separated antennas. |
| Port vs RF chain | analog/hybrid precoder, number of chains, hardware sharing | A 64-port array does not imply 64 observed independent CSI streams. |
| Layer vs port | stream-to-port/layer mapping, rank, scheduling | Artificially estimating nonexistent spatial streams inflates capacity. |
| Calibration vs geometry | direction-specific complex gain/phase and clock state | UL/DL channel reciprocity is not plug-and-play hardware reciprocity. |
| Antenna reference point | exact local/world coordinates, orientation, pattern | Incorrect azimuth/elevation and sub-wavelength positions alter relative phase. |
| Synthetic vs explicit elements | boolean and solver budget | Phase-shift approximation may not preserve element-specific blockage/reflection. |
| Field polarization | Jones basis, cross-pol patterns, handedness | Treating scalar attenuation as XPR destroys polarization physics. |
| Correlation matrix | Hermitian PSD, units, eigen spectrum and draw process | Non-PSD matrix or artificial independent draws gives impossible MIMO statistics. |
| Near-field | array aperture, distance, wavelength and approximation | Plane-wave model fails outside validity; cannot use it for spherical-wave claims. |
| Antenna coupling | included/not modeled/calibrated | Ideal uncoupled ports overstate diversity and spatial degrees of freedom. |
| Multi-node phase | independent oscillator, initial phase/drift, synchronization | Distributed coherent combination is fabricated absent reference/calibration. |
| Beam steering | direction, pattern, codebook, quantization, RF chain | Beam labels cannot be claimed without a valid array and schedule. |

### 18.6 Network-scale generation sequence

1. Sample the **site and scene**, material realization and weather under declared physics model; freeze the site split before sampling terminals.
2. Sample **network graph**: number of sites, gNBs/APs/UEs; node positions/heights, user clustering, serving and neighboring relationships; validate no overlap with illegal geometry or split buffers.
3. Sample compatible **array types** per node; check RF chains, polarization, antenna spacing and carrier-specific element patterns.
4. Sample **global physical time**; build synchronized pose and world-object motion. Derive active link graph, including all serving and interfering connections.
5. For each band, run Sionna RT using its supported **homogeneous-array scene solve** or well-marked grouped solves, with identical physics/time seed policy and scene hash. Preserve missing/no-path links.
6. Extract all path/channel blocks with ordered TX/RX node and port IDs; verify direct/indirect decomposition, gauge and coherent sums for each block.
7. Construct network received grids **jointly from all active sources** and cross-links rather than separately generating interference as AWGN. Preserve independent waveform/oscillator states and duty cycles.
8. Run ideal and realistic receiver estimation variants; identify which transmitter-specific channels are actually observable given pilot reuse/orthogonality and synchronization.
9. Derive multiuser labels (link SINR, RSRP/RSRQ with proper measurement definitions, beam and handover events, rate/BLER) **only** when required waveform and detector details exist.
10. Write per-site/route and per-network-episode split ancestry, QA, manifests and content hashes. Reprocessing must yield identical physical truth when only the receiver noise seed changes.

### 18.7 Mandatory multi-node tests before scale

- M01: 1×1 RT channel agrees with same single-link slice extracted from a multi-node RT scene (where physically/simulator-equivalent).
- M02: Reordering transmitter/receiver insertion never changes link IDs/physical identity after name-based reindexing.
- M03: Swapping Tx/Rx for a reciprocity test uses an explicitly valid physical scenario and correct port orientation; never assume hardware reciprocity.
- M04: 1×4 SIMO and 4×1 MISO have correct axis orientations, power definitions and non-identical beam domains.
- M05: For a known free-space array geometry, relative phase follows expected array manifold within error tolerance.
- M06: Single- and dual-pol port counts match physical port map; cross-polarization components are not silently dropped.
- M07: `synthetic_array=True` and `False` differ by documented tolerance on an appropriate far-field case and are **allowed** to diverge on targeted near-field cases.
- M08: Full multi-node received grid equals summation of individually generated **time/frequency-aligned** source contributions in linear, impairment-free reference mode.
- M09: With interferers disabled, desired-only channel and receiver output match no-interference reference.
- M10: Cochannel interference power and SINR respond monotonically in a controlled toy channel when interferer power increases, holding everything else fixed.
- M11: Wrong-cell pilot reuse is recognized as contamination; never return perfect per-user CSI unless pilots permit it.
- M12: Disabling one node affects only its intended source signal and future dynamically coupled processing, not geometric truth for other links if the environment/pose is unchanged.
- M13: FDD frequencies generate independently frequency-grounded channels using correct material and antenna parameters; UL/DL roles remain directed and tagged.
- M14: Varying gNB antenna count at fixed scene leaves underlying geometry unchanged; port-specific results and solver approximation change as expected.
- M15: Per-node clocks are tracked; receiver-observable relative phases are invariant under irrelevant global phase transforms when labels are invariant.
- M16: Two neighboring UEs never share receiver noise samples merely because the same scene or seed was used; deliberate common interference is recorded separately.
- M17: Full heterogeneous-antenna grouped solve receives an `approximation_status`; no file incorrectly claims one native joint solve.
- M18: Link under no-path condition is represented as a physical null link with accompanying solver-completeness validation, not omitted or replaced by zeros and a positive visibility label.
- M19: SISO-to-MIMO reuse of same geometry is paired **within one split**; the paired view cannot leak to test through another antenna configuration.
- M20: Serving-BS association and handover target are computed from fully specified time-domain policy and available measurements, not from forbidden future ground truth.

---

## 19. Full propagation/channel family catalog and applicability ledger

The word *full* refers to preserving all supported and scientifically useful **fields** for a defined physical model, not assuming every parameter is simultaneously meaningful for every type of source.

### 19.1 Channel catalog

| ID | Channel source / variants | Mandatory native/derived fields | Network/antenna restrictions | Validation |
|---|---|---|---|---|
| C00 | No-channel, noiseless, identity, free-space LOS | exact injected `a, tau`, phase, noise = zero | any validated SISO/MIMO toy graph | analytically exact. |
| C01 | Controlled 2-, 3-, K-path specular channels | `a,tau,angle` where geometry exists; path birth/death | array manifold and clock assumptions explicit | reconstruct CFR, identify resolvability thresholds. |
| C02 | Rayleigh iid/flat block fading, correlated fading | channel matrix, correlation PSD, Doppler if simulated | arbitrary valid array covariance | statistical moments/eigenvalues. |
| C03 | Rician, two-wave, diffuse and Nakagami-like stress (if explicitly implemented) | generation parameters, K definition, complex amplitude samples | synthetic stress, not automatic 3GPP compliant | moment, noncentrality, normalization. |
| C04 | TDL-A, B, C, D, E | taps, delays, powers, time evolution, profile table/version | 1 TX + 1 RX with arrays and correlations | table/conformance checks. |
| C05 | TDL-A10, A30, B100, C60, C300, D10, D30 | fixed-delay TS 38.101-4 conformance tables | 1 TX + 1 RX | no incorrect delay-spread rescaling. |
| C06 | CDL-A, B, C, D, E | clusters/angles/array CIR, direction, mobility | 1 TX + 1 RX with arrays | profile tables, angular spread, polarization. |
| C07 | UMi | scenario LOS, large/small scale, pathloss, shadow, time and topology | validate native multiuser config | spec statistics and correlation. |
| C08 | UMa | as above | spatially suitable macro network | spec statistics and correlation. |
| C09 | RMa | as above | realistic rural topologies | spec and height/distance limits. |
| C10 | InH | as above | indoor hotspot topology | spec and deployment limits. |
| C11 | InF | as above | factory variants/scenes | spec and placement limits. |
| C12 | RT specular only | full path oracle; direct + reflection, refraction as declared | multi-node graph | solver convergence and geometry. |
| C13 | RT specular + diffuse | path diversity, scattering coefficients; logged random solver uncertainty | multi-node graph | convergence and physics plausibility. |
| C14 | RT plus diffraction | interactions, first-order edge/wedge model | multi-node graph | convergence with edge examples. |
| C15 | RT refraction/transmission | thickness/material validity, transmitted path details | multi-node graph | slab/thickness analytic cases. |
| C16 | RT full audited interactions | direct, specular, diffuse, refraction, first-order diffraction | multi-node graph | against simplified toggles and path budget. |
| C17 | RT moving TX/RX/objects | time-indexed re-trace path sets, visibility changes | synchronized multi-node trajectories | delay/Doppler/transition checks. |
| C18 | RT deliberate interventions | paired scene changes | preserve parent and counterfactual ancestry | repeated seeds, causal semantics. |
| C19 | Frequency selective, multi-band, optional CA | independent carrier-grounded complex links | per-band materials and array pattern | frequency and group-delay convention. |
| C20 | Interfered multiuser NR | all contributing channels, waveforms, scheduling | ≥2 active transmitters | explicit received-signal sum and pilot contamination. |
| C21 | Measured complex CSI/IQ | measurement metadata, partial reference truths | based on external sensor/campaign | licensing, clock/calibration and reference QA. |
| C22 | Measured magnitude/RSRP only | power observations, measured availability/coverage | cannot claim complex phase or path oracle | respect observability masks. |
| C23 | Experimental sensing/RCS | target position, velocity, RCS and echo | F09 only after separate validation | separate radar waveform validation. |

An entry is a **candidate data family**, not proof it is presently implemented in the uploaded repository. `C03` non-3GPP synthetic fading must not be labeled compliant simply because it has plausible statistics. Fading and geometry families require different provenance and label ontologies.

### 19.2 Exact table and feature coverage by configuration

- For **TDL-A–E**, declare delay-spread parameter and correlation matrix for each `(Ntx_ant,Nrx_ant)`; for fixed TDL conformance profiles, respect the *fixed* tap delays documented in the installed implementation.
- For **CDL-A–E**, preserve angular cluster information, strong specular component semantics, array orientations and polarization; do not give RT geometry-specific object IDs to CDL.
- For **UMi/UMa/RMa/InH/InF**, record topology and LOS scenario state, path loss, shadowing, LSPs, spatial consistency settings, optional blockage and original stochastic seed. Geometry in scenario models is not equivalent to a polygon visibility oracle.
- For **RT**, preserve native solved paths **including invalid masks** and path interaction/type/depth (where the solver supplies them), absolute propagation delays, complex coefficients, angles, Doppler, geometry poses, materials and complete result settings.
- For **NR**, preserve both ideal channel and independent pilot-derived channel estimate; label `receiver_truth` vs `receiver_estimate`, and *do not train models using true source/channel information unavailable at deployment* unless it is a clearly identified privileged-oracle baseline.
- For **external measurements**, map only source-supported fields and never synthesize missing ground truth in the evaluation set.

### 19.3 Common and shifted distributions—do not create a giant accidental Cartesian product

Define an intentional core design with 3 layers:

1. `core_balanced_factorial`: a balanced/stratified scientific comparison over **a few pre-registered primary factors** (channel family, SISO/SIMO/MISO/MIMO group, low/medium/high bandwidth, distance, LOS visibility, pilot SNR). Match training examples across methods and scenes.
2. `natural_prevalence`: sample from a declared deployment scenario prior (user density, radio sites, materials, blocker trajectories, SNR, serving link load). Do **not** artificially balance LOS class and claim field prevalence or calibrated probabilities.
3. `shift_challenge`: hold out entire sites/materials, antenna shapes, carriers, user counts, channel families, RF calibration sets, hardware campaigns. Pre-register each and cap the number of targets used for test-only diagnostic peeking.

Record for every released slice: empirical factor frequencies, `sampling_probability`, designed enrichment/selection rule, intended use, and optional valid importance weight. *A uniform mixture of physically unrelated scenes is not a naturally distributed deployment dataset.*

---

## 20. Generator and model-facing contracts: trainable views without information leakage

### 20.1 Independent layers per physical episode

```text
UMCD-1 release metadata
└── site/scene/material source assets (restricted geometry oracle)
    └── network propagation episode (all synchronized devices and objects)
        ├── nodes, antenna arrays, RF-chain/polarization maps, clock groups
        ├── all directed link blocks (serving AND interfering)
        │   ├── native RT/path oracle (ragged, absolute delay, interaction metadata)
        │   ├── ideal physical CIR/CFR (complex, absolute power)
        │   ├── derived PDP, spatial covariance, delay/angular/Doppler statistics
        │   └── task target definitions + per-field applicability and validity
        ├── time-linked future channels + causal event/route chain
        └── waveform/receiver child observations (one parent → many children)
            ├── OFDM/PUSCH DMRS and optional SRS/PRS/SSB observations
            ├── actual joint interference grid and RF impairments
            ├── synchronization/estimator/equalizer/decoder diagnostics
            ├── model-visible non-privileged features
            └── withheld labels/restricted oracle for controlled evaluation
```

### 20.2 Formal API functions (proposed; not existing implementation claims)

```python
# Public abstract signatures, not runnable current repo code.
scene = scene_factory.sample(scene_id, seed, prior="deployment")
network = topology_factory.sample(scene, seed, num_bs=2, num_ue=4,
                                  arrays_by_node=..., time_grid=...)
truth = propagation_engine.solve(network, band_id="n78", mode="rt",
                                 solver_config=..., preserve_paths=True)
quality = physics_qa.validate(truth, reference_cases=...)
parents = episode_store.commit_if_valid(network, truth, quality)
views = observation_engine.derive(parents, signal_schedule=...,
                                  front_end=..., seed=...)
targets = target_engine.compute(parents, task_registry_version="v1")
dataset = dataset_reader.open(release="umcd-1.0.0",
                              task="relative_first_path_delay",
                              split="train", modality="nr_dmrs_csi",
                              node_configuration="any_mimo", allow_oracle=False)
```

Contracts:

- `propagation_engine` preserves link axes and distinguishes `[num_rx_nodes, num_rx_ports, num_tx_nodes, num_tx_ports, path]` RT-native patterns from ragged per-link canonical form.
- `episode_store` writes only after validator passes, atomically records failure reasons and forbids overwriting the immutable physical artifact under the same content hash.
- `target_engine` cannot derive absolute ToA from delay-normalized data, angles from scalar SISO, Doppler from a static independent draw, or real geometric LOS from profile identity. Such targets must carry typed invalidity reasons.
- `dataset_reader` must not expose unrestricted metadata/path identities into model features. Explicit `oracle=True` is available to offline evaluation only with provenance logging.
- `split_registry` handles **site/trajectory/material/network episode/pair ancestries**; all reciprocal links and alternative-array views from one scene remain in the same split, unless a controlled held-out-array test uses an independent-site design.

### 20.3 Exact target definition manifest example

```yaml
schema: umcd.target.v1
name: rms_delay_spread
units: s
reference: absolute_physical_channel_path_oracle
valid_when:
  - solved_path_table_passes_convergence
  - total_incoherent_path_power_above_defined_floor
  - at_least_one_valid_path
  - declared_link_and_band_match
estimator: rms_spread_of_valid_per_path_powers_in_declared_gauge
ambiguities:
  - diffractive_and_diffuse_solver_discretization
  - bandwidth_limited_resolvability_differs_from_oracle_path_spread
unavailable_reason_enum:
  - NO_PATH
  - SOLVER_UNCONVERGED
  - INSUFFICIENT_SOURCE_FIELDS
  - INVALID_PHYSICAL_REFERENCE
  - UNSUPPORTED_MODEL
  - CLOCK_UNCALIBRATED
privacy: oracle_only_until_target_release
```

**Do not equate** a full unlimited-bandwidth ray-oracle delay spread with an operationally estimated finite-bandwidth PDP spread. Store both with separate `target_semantic_id`, conditioning and units.

### 20.4 Timing, gauges, phase and power controls

- Store hardware-free propagation delay in seconds from Sionna **before delay normalization**; store a separately tagged reference-relative shift if a normalized algorithm view is needed. For absolute ToA at the receiver, simulate and record synchronization offsets (or abstain).
- Store physical complex coefficients and direct+indirect path decomposition under **one identical timing and phase convention**; compute full CFR by one sum or audited path filtering of the same `Paths` tensor.
- Distinguish the incoherent sum `Σ_p |a_p|²` used for a path-power PDP from the coherent squared magnitude `|Σ_p a_p e^{-j2πfτ_p}|²` at a particular subcarrier; these are not generally equal.
- Record every normalization: path-to-unit-energy, SNR normalization, pilot energy per RE, antenna-element gain, precoder power, OFDM IFFT scaling, receiver AGC, target standardization and model feature normalization; each has a version and train-split-fitted parameters.
- Guard multi-node phase: an independent TX's unknown RF phase cannot be silently set to zero and then exploited as a stable geometric fingerprint; perform phase-offset sweeps, per-node oscillator modeling and invariance ablations.

### 20.5 Multi-resolution storage views and projected data sizes

Storage tiers:

- **Tier A, essential:** dataset/scene/network manifests; per-link raw path truth; geometry/material hashes; full float32/complex64 **oracle** CIR/CFR on canonical measurement grids; target/mask provenance; QA/split tables. Per-link paths are ragged and versioned.
- **Tier B, study-specific:** time windows of actual Rx IQ/NR resource grids, channel estimates/uncertainty, beam measurements, repeated impairments and trajectories. Derive from Tier A where scientifically valid and reconstruction is reproducible.
- **Tier C, expensive conditional:** dense full-band full-antenna time-varying channels, high-sample-rate waveform captures, large-array explicit RT benchmark, full calibration grids, multi-frequency and raw scene render artifacts. Store by slice, not as a requirement for every episode.

Resource formula for a dense complex64 full network block per epoch:

`bytes = frames × subcarriers × Σ_valid_links(Nrx_port(link) × Ntx_port(link)) × 8`; add symbol-time expansion, receiver IQ, raw path tables, format overhead, compression effects and data copies. Complex128 doubles the complex-array component. **The true byte count depends on sparse link graph and heterogeneous arrays, not simply `Nrx_node × Ntx_node × one antenna count`.**

### 20.6 Version and upgrade handling

- Freeze release-wide `source_archive_sha256`, `git_commit_or_unknown`, `sionna_rt_git_sha`, `sionna_phy_git_sha`, `mitsuba_version`, `drjit_version`, Python/PyTorch/GPU versions, numerical precision/algorithm mode, RT `deterministic=True` when used, scene edit hash, array and material model hash, RF impairment implementation SHA.
- Use strict semver: new optional target/view `minor`, changed target meaning/axis/gauge/unit or sample content `major`, corrected metadata without changed samples `patch` with visible audit note.
- Generate `migration_report.json` for renamed fields; retain original hashes and old target IDs. Never silently update raw data on an old release.
- Keep a machine-readable `known_limitations.yaml` and a source-capability manifest. Scientifically unsupported channels/tasks are marked `not_supported`, not filled with synthetic guesses.

---

## 21. Output-target-by-topology eligibility and evaluation contract

The question “what can we predict” must be answered for the actual array, number of nodes, sampling grid and receiver measurements. More output heads do **not** automatically make a better paper.

Legend: `Y` = scientifically eligible with the specified oracle/measurement; `C` = conditional; `–` = not physically identifiable from this observation alone.

| Output | SISO static CSI | SIMO/MISO | MIMO | Multi-BS/UE | Dynamic traces | Requirements / caution |
|---|:--:|:--:|:--:|:--:|:--:|---|
| Binary geometric LOS (oracle label) | Y | Y | Y | per link Y | Y | Physical geometry is label; learn from permitted receiver observations only. |
| Direct-path detectability | Y | Y | Y | per link Y | Y | Receiver SNR/bandwidth and detection threshold defined. |
| Path outage / total rx power | Y | Y | Y | per link/network | Y | Link budget; distinguish solver miss. |
| RMS delay spread / mean excess delay | Y | Y | Y | per link | Y | Full path truth or finite-band receiver estimator definition. |
| Strongest/first path relative delay/power | C | Y | Y | per link | Y | Finite bandwidth/clock resolution; weak path may be unobservable. |
| Absolute ToA | C | C | C | C | C | Known timing reference/clock; otherwise **not identifiable**. |
| AoA | – | C if RX array | Y if RX array | per receiving array | C | Calibrated RX array and adequate spatial aperture. |
| AoD | – | C if TX array | Y if TX array | per transmitting array | C | Calibrated TX array/beam probing. |
| Angular spread, spatial covariance | – | C | Y | per link/node | C | Array, phase calibration and sufficient observations. |
| Polarization/XPR | – | C | C | conditional | C | Dual-pol supported and calibrated; scalar channels cannot recover XPR. |
| MIMO rank/condition number | trivial scalar | C | Y | per-link or joint | C | Defined streams, physical channels and correct gauge. |
| Path count/order | C | C | C | per link C | C | Count **resolvable** paths vs RT solver rays; variable cardinality. |
| Doppler spectrum/spread | – | – | – | – | C | Time-resolved samples and controlled clock drift. |
| Future CSI / path transitions | – | – | – | – | Y | Continuous trajectories/retrace, known forecast horizon. |
| Future blockage / handover | – | – | – | C | Y | Moving world + policy, no future-input leakage. |
| Beam quality/top beam | – | C | Y | C | C | Valid codebook, beam sweep and schedule. |
| Signal/interference separation | – | – | C | Y | C | Multiuser waveform and synchronized active sources. |
| SINR / CQI / throughput / BLER | C | C | C | Y | Y | Receiver/coding/scheduler defined; CQI definitions standardized where claimed. |
| Position / geometry inverse | C | C | C | C (better geometry diversity) | C | Clock geometry and ambiguity; independent metric-scale ground truth. |
| Unknown/OOD risk and confidence | Y | Y | Y | Y | Y | Calibration on independent groups; proper scoring and reject options. |

A feature/label may be present in **oracle storage** while impossible to infer uniquely from a given receiver view. This is scientifically useful for identifying failure conditions and deriving impossibility/uncertainty claims; do not delete difficult cases or treat the generator label as automatically learnable.

### 21.1 Primary paper endpoint (strictly prioritize)

A coherent first top-tier candidate should pursue one main mechanistic claim:

> A symmetry- and observability-aware complex multipath representation improves out-of-domain **joint direct-path detectability, delay-related state estimation, and calibrated uncertainty**, while reconstructing only what can be inferred from receiver observations.

The **core evaluation** uses 1×1, 1×4 SIMO, 4×1 MISO, 2×2, 4×4, one 2-BS/4-UE interfered topology across TDL/CDL statistical pretraining, independent Sionna RT scenes, one actual NR pilot receiver, and a carefully scoped external source. Larger spatial/network tasks are focused extension results and separate later-paper candidates.

### 21.2 Second-paper and optional results

- **MIMO multipath localization/beam paper:** angular estimation, codebook beam choice, path diversity and spatial uncertainty; must have calibrated arrays and meaningful ground truth.
- **Mobility / forecast paper:** predicted future complex CSI, Doppler/multipath events, beam/blockage survival and policy outcomes; requires retraced moving geometry and time-aware splits.
- **Distributed/multi-node paper:** multiuser interference, pilot contamination, serving link association, cooperative beamforming with realistic clocks; needs spatially consistent multi-link physics.
- **Foundational representation paper:** mechanism extends beyond RT/scalar CSI into at least one other complex inverse problem, with rigorous symmetry/identifiability analysis and capacity/compute-controlled baselines.

The umbrella dataset is *one product*, but do not compress four distinct original method claims into one unfocused manuscript.

---

## 22. Frozen experiment registry, required artifacts and result definitions

**No numerical performance value below is a completed result.** Every row is a required experiment to run **after** the native simulator and data gates pass. Use `E` IDs so later figures cannot quietly omit negative experiments.

| Experiment | Required comparison | Dataset slices | Primary metric | Mandatory artifacts |
|---|---|---|---|---|
| E00 environment/API | Native Sionna RT/PHY installed vs pinned expected interfaces | single native smoke scene/TDL/CDL | pass/fail, exact versions | `E00_env.json`, `E00_native_interface.md` |
| E01 analytical channel | closed-form path manifold vs code | C00–C01 × SISO/MIMO | complex error, phase residual | `E01_physics_errors.parquet` |
| E02 3GPP profiles | all scalable & fixed TDL, CDL A–E against spec table | C04–C06 arrays | profile moment/tap/angle stats | `E02_profile_conformance.csv` |
| E03 geometric RT | path depth/sampling toggles, scene realism | C12–C16, LOS/NLOS/weak-direct | convergence of power/delay/angle | `E03_rt_convergence.parquet` |
| E04 multi-antenna | SISO/SIMO/MISO/MIMO, explicit vs synthetic arrays | SISO through 8×8 | manifold/coherent error, runtime | `E04_array_ablation.csv` |
| E05 multi-node | 1 BS×1 UE, 2 BS×4 UE, mixed serving/interferer | C16/C20 | full link-grid conservation | `E05_network_joint_check.json` |
| E06 receiver | ideal complex CFR vs true raw IQ / pilot LS/LMMSE | multiple SNR/BW | channel NMSE, BER/BLER if decoded | `E06_receiver_fidelity.parquet` |
| E07 LOS | classic delay/K tests; MLP; CNN; CV-CNN; proposed | independent sites | AUROC, AUPRC, ECE, sensitivity | `E07_los_by_domain.parquet` |
| E08 delay spread | classical PDP estimator; MLP; model-based; proposed | all eligible channel families | MAE/RMSE/interval coverage in ns | `E08_delay_spread.parquet` |
| E09 first path | ToA lower bound and relative path delay | timing-grade eligible | error CDF, exceedance probability | `E09_first_path.parquet` |
| E10 multipath reconstruction | sparse classical / MLP / constrained variable-path | 2-path → rich RT | complex NMSE, matching, calibration | `E10_path_recon.parquet` |
| E11 angles | MUSIC/ESPRIT or declared suitable estimators + neural | calibrated RX/TX arrays | wrapped angular MAE and angular spread | `E11_angles.parquet` |
| E12 Doppler | classical temporal estimator + neural | continuous time sequences | Doppler error, temporal coherence | `E12_doppler.parquet` |
| E13 polarization | single/cross/dual-pol | supported calibrated array | XPR error / covariance | `E13_polarization.parquet` |
| E14 spatial rank | baseline direct matrix statistics, neural estimators | SISO/MIMO valid | rank error/condition error | `E14_mimo_spatial.parquet` |
| E15 uncertainty | uncalibrated, calibrated, interval/conformal baselines | all valid | NLL, Brier, ECE, risk-coverage | `E15_uncertainty.parquet` |
| E16 domain transfer | TDL/CDL→RT→measured, held-out site/array/band | frozen external groups | worst-group metric + 95% CI | `E16_transfer.csv` |
| E17 robustness | clock, CFO, IQ imbalance, SNR, outage, materials, solver | paired challenge views | degradation & selective risk | `E17_robustness.parquet` |
| E18 multi-node interference | desired-only vs all active transmitters, pilot reuse | multi-BS/UE | SINR, CSI contamination, BLER | `E18_interference.parquet` |
| E19 beam selection | oracle vs classical, baseline DL, proposed | array and codebook subset | beam gain loss, beam failure | `E19_beam.csv` |
| E20 forecast | persistence/Kalman/AR/MLP/sequence baseline/proposed | RT dynamic + measured valid | horizon CSI NMSE, EVM, calibration | `E20_future_csi.parquet` |
| E21 blockage | current visibility vs future real obstacle changes | RT re-traced sequences | event AUPRC, detection lead time | `E21_blockage.csv` |
| E22 system outcomes | no AI / classic / matched MLP / proposed in NR | decoder/link budget valid | BLER, goodput, spectral efficiency | `E22_system.parquet` |
| E23 intervention | mathematical path removal vs physical blockage | paired physical solves | counterfactual consistency gap | `E23_causal_pairs.parquet` |
| E24 mechanistic ablations | original MLP, matched CNN, CV-CNN, hybrid, component drops | full core tasks | paired Δ + compute | `E24_model_ablations.parquet` |
| E25 data completeness | audit each family/antenna/topology split | complete manifest | validity/coverage/dup rate | `E25_data_quality.json` |
| E26 training scaling | data-size and model-size with identical split | 1%, 10%, 30%, 100% | learning curve + GPU/energy | `E26_scaling.csv` |
| E27 calibration/real | held-out hardware/site if obtainable and licensed | F08 measured | task-specific accuracy / failure | `E27_real_world.csv` |
| E28 leakage controls | metadata shortcuts, duplicate scenes, profile identity | purposely adversarial splits | leakage-detection and drop | `E28_leakage_report.json` |
| E29 negative results | CNN underperformance, irreducible ambiguity, model failures | eligible cohorts | significance + interpretability | `E29_negative_results.md` |
| E30 computational cost | RT vs other generators, per view/storage | pilot/full | accepted episodes/hour, cost | `E30_cost_model.csv` |

### 22.1 Required tables and figures in eventual paper(s)

1. `Table A`: exact dataset taxonomy, sites/scenes, frequencies/bands, channel family, valid targets, measured vs simulator, license and availability.
2. `Table B`: antenna/network topology per split (`#TxNodes,#RxNodes,Tx/Rx ports,RF chains,pol,carrier,UL/DL`), number of independent **sites** and **trajectories** (not only frames).
3. `Table C`: solver and physics configuration: scene assets/materials, depth, rays/samples, enabled mechanisms, synthetic/explicit arrays, backend, numerical convergence.
4. `Table D`: receiver: actual NR numerology, DMRS/PRS/SRS pilot resources, estimator, clock/noise/interference/decoder config.
5. `Table E`: target audit: semantic version, oracle/proxy/measured label, unit, validity, missingness/condition and definition reference.
6. `Table F`: leakage and representativeness: split ancestral independence, per-class/sample selection, dedup overlap, scene distance exclusion, natural LOS base rates.
7. `Table G`: all baselines with capacity/parameters/MACs/latency and input modalities held constant.
8. `Table H`: primary task results and 95% site-clustered uncertainty intervals, not unclustered per-frame significance.
9. `Table I`: cross-domain held-out scenes, materials, arrays, carriers and external measurements.
10. `Table J`: ablations of MLP and convolution, phase handling, counterfactual loss, reconstruction and uncertainty.
11. `Figure 1`: end-to-end schematic with strict privileged-oracle separation.
12. `Figure 2`: complete raw-path/NR-observed response from identical physical episode (LOS+many NLOS reflections, NLOS and outage).
13. `Figure 3`: 3D independent scene diversity and object/material truth; honest synthetic/field provenance.
14. `Figure 4`: joint path-delay/angle/Doppler spectra with resolvability boundaries and bandwidth effect.
15. `Figure 5`: empirical power/delay/angle statistics, standardized-vs-RT distributions and held-out site transfer.
16. `Figure 6`: ROC/PR and confidence/reliability for LOS+detectability, stratified by link configuration.
17. `Figure 7`: first-path delay and RMS-spread error CDF across bandwidth/noise/dynamic scenes.
18. `Figure 8`: antenna configuration comparisons: 1×1 → SIMO/MISO → 4×4 → validated massive-MIMO subset.
19. `Figure 9`: multi-node channel/interference graph and SINR/BLER under changing number of active sources.
20. `Figure 10`: forecast horizon error and real blockage/beam survival under complete re-tracing.
21. `Figure 11`: risk-coverage and uncertainty/identifiability curves; ambiguity and abstention shown.
22. `Figure 12`: negative and solver-limited cases (missed rays, unknown clocks, profile recognition, synthetic array errors).
23. `Figure 13`: compute/storage/cost/acceptance-vs-ray-budget frontier.
24. Appendix: all task-specific validity matrices, reproducibility seeds, numerical tolerances and QA criteria.

**Results must be generated by scripts from immutable `E**` artifacts**, never manually typed into plots. Every plotted cohort has hash, sample count, metric semantics, CI method, scene split ID and compute code SHA. Do not use an aggregate “accuracy across all outputs” metric that mixes dimensionless classes with dimensional regression.

### 22.2 Correct uncertainty and significance procedure

- Independent unit for CI is the site or independent trajectory/network episode (whichever defines actual generalization). Bootstrap by site/route cluster, not every highly correlated PRB or adjacent frame.
- Declare five or more training seeds for core pilot model selection, **ten or more** for final neural-model uncertainty where feasible and computationally justified. Report actual n, no retroactive invented count.
- For multi-model comparisons use paired per-site/episode differences, site/scene bootstrapping, transparent multiple-testing control where applicable, and confidence intervals on effect size.
- Evaluate calibration on a dedicated calibration split; hidden external site(s) cannot tune hyperparameters, thresholds, noise distributions or model selection.
- Use proper metrics: AUROC/AUPRC for LOS; NLL/Brier/ECE with declared bins and priors; time/distance error in ns/meters; complex CSI NMSE with stable denominator/zero-channel handling; path-set matching with permutation-invariant assignment and explicit path-count penalties; BLER confidence intervals after enough blocks.
- Report missing-label percentages, hard-case failure, abort/solver-unconverged count, site imbalance, and compute budget in primary results rather than concealing them.

---

## 23. Implementation plan: stages, file-level responsibilities and release gates

Use **verification-first, additive modules**. Preserve the uploaded project's typed NR records, artifact provenance, native tests, cache identity, target isolation and paired-regression infrastructure. Avoid copying trading-specific domain logic from Aalto RL; reuse design patterns such as immutable experiment configs, replayable seeds, split freezing, independent baselines and robust logging.

### Stage S0: source freeze and API matrix

- Record current uploaded ZIP SHA, list code dependencies and test suites; compare GitHub commit only if actually accessible.
- Record native RT/PHY/Mitsuba/Dr.Jit and build versions, GPU backend and reproducible wheel hashes; pin rather than relying on mutable `main`.
- Native smoke tests: 1×1 and 2×2 Sionna `PathSolver`, `paths.a/tau/angles/doppler`, `.cir(normalize_delays=False)`, `.cfr(normalize_delays=False,normalize=False)`, zero path, specular/diffuse/refraction/diffraction and PUSCH/DMRS.
- Stop if model API signatures differ; implement version-specific adapters with typed capability registry.
- **Deliver:** `sionna_compatibility_matrix.json`, source/dependency lock, script + logs, contract tests all passing.

### Stage S1: full multi-node canonical schema

- Extend `datasets/nr.py` types by additive `datasets/network.py`, `datasets/path_oracle.py`, `datasets/targets.py`, `datasets/episode_store.py` and `datasets/schema/` versioned manifests; preserve compatibility with older single-link records.
- Introduce `Node`, `AntennaArraySpec`, `DirectedLink`, `NetworkEpisode`, `PathOracleSet`, `PhysicalChannelView`, `ReceiverObservation`, `TargetValue` with units/reference and `ValidityReason` enum.
- Implement sparse per-link ragged path store, named axis metadata, validity masks, complex dtype safety, checksums and atomic publish. Schema tests should reject illegal antenna counts, invalid port maps, missing link IDs and unmarked normalized delays.
- **Deliver:** JSON-schema/YAML contracts, round-trip tests, backward-compatibility guide, cached fixture data and storage profiling.

### Stage S2: path extraction and physical oracle

- Extend `integrations/sionna/adapter.py`, `rt_to_nr.py`, `rt_identity.py` and `rt_trajectory.py` via a new `integrations/sionna/path_oracle.py`, `rt_network.py`, `array_layouts.py` adapter; do not remove verified single-link paths.
- Extract path type/depth/interaction/object/primitive IDs, coherent complex amplitude, propagation delay, angular and Doppler state and versions. Expose synthetic/explicit array flags and link/node mapping explicitly.
- Implement **geometry-based** direct segment visibility oracle independent of whether stochastic ray shooting happened to find a specific ray. Compare and classify near-grazing ambiguous geometry separately.
- Compute direct/indirect channels by coherent filtering from one common path set. Validate all `G2`, `M01..M20` tests.
- **Deliver:** tiny fully explained physical fixtures, power/gauge conformance report, path budget convergence plots.

**Implementation snapshot (2026-10-22):** The first native adapter is present at
`integrations/sionna/path_oracle.py`. It converts one explicitly selected
Sionna TX/RX device pair per call and groups ragged paths by link-local RX/TX
array ports. Native LLVM tests compare coefficient, absolute delay, interaction
order/type, scene-local object/primitive IDs, scene-XYZ vertices, Doppler and
angle fields against pinned Sionna output for synthetic and explicit arrays,
V/VH polarization combinations, and empty solves. The immutable bundle stores
the ordered ragged interaction trace and its coordinate-frame declaration.
`integrations/sionna/rt_network.py` now assembles one complete UMCD episode path
table from explicit Sionna TX/RX/frame bindings. It requires exact link×frame
coverage, applies ordered link-local port maps, retains empty port groups, and
records solver-failure/unconverged states without fabricating path rows. A
native 2-TX×2-RX scene test verifies all four directed device pairs are mapped
into one episode. `integrations/sionna/array_layouts.py` now converts static
Scene devices and normalized antenna element positions into UMCD node/array
records, preserves pattern-major port order and the scene device-axis indices,
and can bind a shared `Paths` result to every extracted link. Native LLVM
verification covers a 2-TX×2-RX scene with two elements and dual polarization.
RF-chain counts and physical polarization labels remain explicit caller inputs;
moving devices are rejected pending trajectory support. The finite-segment
visibility oracle is now independent of PathSolver and separately cross-checked
during legacy RT generation. Pose-timeline extraction, calibrated array/port
semantics, solver convergence, and path-budget convergence plots are not
complete; Stage S2 remains in progress.

**Geometry visualization note:** [`rt-urban43-spatial-holdout-20261022.png`](figures/rt-urban43-spatial-holdout-20261022.png) shows the synthetic City43 4×4 axis-aligned building layout in 3D and top-down, with three TX locations and 120 sampled RX links (20 geometric LOS and 20 geometric NLOS per disjoint TX/RX region). This makes the spatial sampling visible; it is not surveyed urban geometry or measured channel ground truth. The separate [solver convergence plot](figures/rt-box-los-nlos-convergence-20261022.png) uses a controlled thin-screen fixture and therefore has no building geometry. Paired-link convergence results show material max-depth sensitivity through depth 9: depth 8→9 substantially changes path counts, CFR, and delay spread for both LOS and geometric NLOS. G3.01 remains open.

**City43 label-map correction:** The earlier spatial-holdout figure reversed the display names/colors for integer labels 0 and 1. The generated targets were correct: 0 is blocked geometric NLOS and 1 is visible geometric LOS. A pinned LLVM segment-visibility audit of all 120 saved endpoints confirms 60/60 LOS clear and 60/60 NLOS blocked, with no ambiguous cases or disagreements. The corrected figure and provenance sidecar are in `figures/rt-urban43-spatial-holdout-20261022.{png,svg,json}`; the reusable audit command and result are recorded in the physics verification status and `results/rt-urban43-spatial-label-audit-20261022.json`.

**City43 path-cap follow-up:** The same 1k/10k/1M-cap experiment has now been run on the 16-building synthetic scene, using audited center LOS/NLOS endpoints, depth 3, 20k ray samples/source, seeds 11/23/47, and uncalibrated S=0.5 concrete/glass/metal overrides. The 1k cap changes the LOS median path count from 1,866 at 10k/1M to 981 and its RMS delay spread from 34.05 ns to 22.26 ns. The blocked-link counts and reported aggregate observables match across caps in this fixture. This is useful second-scene-family evidence, but not a cross-cap CFR-vector comparison or a general cap qualification; G3.05 remains open. Reproduction reports and the matched summary are `results/rt-city43-los-nlos-depth3-s05-cap{1000,10000,1000000}-20261022.json` and `figures/rt-city43-los-nlos-path-cap-sweep-s05-20261022.json`. The [3D companion](figures/rt-city43-los-nlos-cap-sweep-scene-20261022.png) displays the actual 16 OBJ building meshes and both audited links; `scripts/plot_rt_city43_cap_sweep_scene.py` regenerates it.

**Direct cap-to-channel check:** Optional v4 convergence reports preserve complex CFR vectors on the declared 256-subcarrier absolute-delay grid. Pairing the three City43 seeds by link, the median LOS relative CFR L2 error against the 1M cap is 0.131 at 1k and 4.55e-7 at 10k; NLOS is at most 2.62e-7 at 1k. The CFR-aware audit and plot are in `figures/rt-city43-los-nlos-path-cap-sweep-s05-cfr-20261022.{json,png,svg}`, with source vectors preserved in the `results/*-cfr-20261022.json` reports. This establishes a genuine cap effect for one synthetic LOS link, not global cap sufficiency or material realism; G3.05 is still open.

**Cross-geometry path-cap and replay diagnostics:** The controlled thin-screen scene gives a different cap response: at depth 6/100k samples and screen S=0.5, LOS CFR median relative L2 error is 0.417 at cap 1k and 0.300 at 10k versus 1M; blocked NLOS is 0.565 at 1k and below 7.15e-7 at 10k. This confirms the cap cannot be selected from the City43 case alone; G3.05 remains open. Repeating the exact cap-1k box run preserves all 42 path counts and bounds CFR relative L2 replay error by 4.26e-7 under the recorded 1e-6 audit tolerance, while only 13/42 CFR hashes are identical. G3.08 remains open beyond this exact backend/configuration. Reports, audit, figure, and the reproducible `scripts/audit_rt_solver_replay.py` are recorded in the physics verification status.


**Matched diffuse-scattering intervention:** A fresh 30-case-per-condition Sionna 2.2 LLVM experiment holds the `box_one_screen` geometry, links, seeds, depth range, ray budgets, and solver flags fixed while changing only `screen-mat` scattering from 0 to synthetic S=0.5. The audit confirms no diffuse paths in the zero control; the intervention yields diffuse paths in 21/30 matched cases and none at depth zero. At depth 3/20k samples, median counts are 1,151 LOS and 44 blocked NLOS. CFR L2 differences show that this synthetic intervention changes the resulting channel. This is solver-mechanics evidence, not calibrated material evidence or a general scattering model; the zero/nonzero CFR reports, audit, source scripts, and budget figure are reproducible and their limitations are explicit in the physics status.

**City43 building-scene scattering replication:** The same fixed-coefficient intervention was run on the actual 16-building City43 OBJ scene at its audited center LOS/NLOS endpoints. Across three seeds, two budgets, and four depths, zero scattering yields no diffuse paths; changing only concrete to synthetic S=0.5 yields diffuse paths in 24/30 cases, with the expected zero at depth zero. At max depth/20k rays the median diffuse-path counts are 1,256 LOS and 256 NLOS, while median CFR relative L2 differences from baseline are 0.072 and 0.482. Companion 3D geometry and sampling plots are linked in the physics status. This establishes the solver mechanism in a second, actual building-mesh scene; material calibration, multiple material families, and measured validation remain open.

**City43 solver replay:** The concrete-scattering City43 report was independently repeated with identical solver and link settings. Across 24 link/seed/depth comparisons, paths match exactly; complex CFR maximum relative L2 error is 3.35e-7, while only 6 CFR hashes are bitwise identical. This verifies same-backend numerical replay for this one building-scene configuration and preserves the per-case audit, but broader configuration/backend reproducibility remains open under G3.08.

Material defaults in that thin-screen fixture are zero scattering in pinned Sionna 2.2, despite the enabled diffuse-solver flag. Do not treat the older v2 depth sweep as diffuse-scattering evidence. The new v3 runner records actual material coefficients, per-path diffuse/non-diffuse contribution metrics, and path-cap configuration. A synthetic S=0.5 screen control verifies that diffuse paths appear and increase with `samples_per_src`; its component-separated [budget figure](figures/rt-box-los-nlos-diffuse-budget-s05-20261022.png) shows materially different LOS/NLOS variability. The coefficient is not calibrated, and sample convergence, depth convergence with nonzero scattering, and cap stress remain outstanding in Stage S2/G3.

### Stage S3: standardized and control families

- Add factories for all **supported** TDL/CDL profiles and fixed conformance tables, analytic controls, flat/correlated fading and full 3GPP scenario-specific classes.
- Record exact `spec_version`, model/parameter table files, array/correlation configuration and unsupported link topology; do **not** pretend one TDL/CDL draw is correlated to another network link.
- Construct controlled cross-domain comparisons and train-only statistics. Hold out whole channel families/profiles if label balance permits.
- **Deliver:** conformance tables, empirical distribution QA, dataset-to-profile mapping, source capability table.

### Stage S4: realistic scene diversity and topology generator

- Reuse `scenes/generated/research_scenes` and scene asset hash infrastructure; expand independent urban, suburban/rural, indoor office, factory, mixed indoor/outdoor, dense street canyon, open area and parked/moving vehicle families.
- Material priors and surface thickness valid per band; unique assets across site holdouts and documented scene-level correlation; no unexplained OSM map/mesh license or privacy leakage.
- Generate 1 BS×1 UE, 1 BS×N UE, 2 BS×4 UE and shared-interference site layouts, homogeneous antenna groups first, followed by gated heterogeneous cases.
- Add moving objects/blockers, retrace physically meaningful discrete time points and path identity matching confidence; no near-field unsupported claim.
- **Deliver:** independent site registry, geometry/occlusion QA and diversity report, on-disk scenes/versioned material manifests, solver runtime estimates.

### Stage S5: NR PHY and realistic observations

- Extend `receivers/nr/pusch.py`, `integrations/sionna/rt_dmrs.py`, `nr_sync.py` and `ofdm_impairments.py`; tie carrier/numerology/grid and channel coherently and preserve absolute signal scaling.
- Implement one validated link/reference waveform first: PUSCH DMRS. Add SRS, PRS, SSB and other reference signals **only with independent mapping and receiver conformance tests**, not generic pilot renaming.
- Multi-node per-resource-element waveform superposition; link path loss and beamforming/power; independent receiver noise and clock; contamination and interference with ground truth.
- Expose realistic model inputs (`raw IQ`, `pilot RE`, `LS/LMMSE channel estimate`, error covariance) using restricted adapters; no raw oracle in inference.
- **Deliver:** receiver estimation fidelity plots, EVM and BLER sanity curves, impairment composition tests, matched ideal-vs-real observation views.

### Stage S6: temporal, multiband and counterfactual production

- Implement a synchronized world timeline for each moving TX/RX/object; remove the assumption that moving UE alone explains future blocker changes.
- Generate future channels at physically defined `Δt`, including scene re-tracing; produce path appearance/disappearance, beam quality and receiver outcomes on the future schedule.
- Support paired frequency views with independent material-specific propagation solves; frequency response sampled correctly at NR subcarriers.
- Generate direct-coefficient mathematical ablation and **separate** physical occluder interventions, keeping site/episode/pair family in one split.
- **Deliver:** coherence and event audit, future-label eligibility and timing reference tests, physical-vs-mathematical intervention report.

### Stage S7: prospective sampling, calibration, external data

- Generate a **pilot** covering all mandatory antenna categories, channel families and low/high difficulty regimes; quantify failure and observed distributions before freezing production ranges.
- Define planned core factor grid and realistic deployment prior; explicitly record enrichment of weak-direct, high-delay-spread, rare transitions and no-path outcomes.
- Qualify external measured datasets by licensing, precision, clock, array, and label ontologies; reserve selected sites completely held out. Keep measured magnitude-only data explicitly out of complex phase/ToA evaluations.
- **Deliver:** pilot acceptance and scale budget, coverage gaps, frozen sampling priors, external provenance and license register.

### Stage S8: generation and validation campaign

- Roll out immutable data shards with complete lineage, bounded RT budgets validated by convergence, job retries keyed on hashes, corruption detection and reason-specific failure counters.
- Run mandatory Gate G0–G8, M01–M20 and task eligibility assertions; no training on `unverified_rt` as if it were calibrated physical data.
- Materialize train/validation/calibration/site-held-out/site+array-held-out/real-external benchmark views without duplicating parent physical episodes.
- **Deliver:** signed release manifest, deterministic QA code, generation failures/cost metrics, published size/statistical coverage and honest known limitations.

### Stage S9: baselines, proposed representation and publication

- Start with physically meaningful formulas and estimators, your legacy MLP, parameter-matched MLP/CNN/CV-CNN and a non-neural sparse multipath method. Use identical receiver inputs and exact splits.
- Then trial the proposed phase-symmetry and physics-reconstruction mechanism, with a separate uncertainty head and uncertainty/observability checks; keep negative results.
- Complete E07–E29, multiple train seeds, clustered CI, site-level transfer and optional measured confirmation.
- Only after satisfying evidence gates write the main paper and release reproducibility package; *journal/conference tier is an output of research quality, not a guarantee*. A proof of performance from debug/analytic scenes alone is not enough.

### 23.1 Critical-path order and explicit stop conditions

`native simulator contracts → schema/ontology → oracle/path/gauge tests → scene realism/convergence → raw paths/channels → NR receiver verification → splits/coverage → pilot data → scale → baselines → new method → independent results → paper`.

**Abort production** if any of: mismatch in absolute delay gauge; abnormal power scaling; missing interfering-link channels; documented Sionna API mismatch; unresolved array axes; inconsistent direct/indirect reconstruction; unvalidated scene materials; cross-site/paired data leakage; nonreproducible source assets; false measured labels; uncalibrated absolute ToA; invalid numeric/solver convergence. Treat invalid output as a recorded failure, not training data.

---

## 24. Dataset release gates, definitions of done, and one complete inventory

### 24.1 Dataset governance and scientific truth labels

| Status | Definition | Permitted claims |
|---|---|---|
| `proposed` | Contract written; generator not validated | no empirical data claims |
| `synthetic_debug` | Analytical/mock data only, deterministic program tests | engineering behavior only |
| `native_verified` | Pinned Sionna run and numeric/shape/gauge checks passed | solver-grounded, within declared simplifications |
| `scenario_qualified` | Model table/moment/fading compliance checked | standards-model-grounded, not necessarily geometric truth |
| `geometry_qualified` | meshes/material assumptions checked, direct ray and convergence independently audited | geometry-derived labels under RT approximations |
| `receiver_qualified` | verified NR waveform/receiver contract with realistic observables | link simulation findings under declared hardware model |
| `externally_calibrated` | calibrated with disjoint measured physical references and documented procedure | physically calibrated within validated sites/conditions |
| `external_tested` | locked independent measured evaluation with complete appropriate labels | genuine external-domain evidence for **eligible** tasks |

An experiment may cite one or more statuses; no status retroactively changes the physical meaning of a previously measured target.

### 24.2 Absolute required acceptance checks

- [ ] `release_id` and immutable full source/build identity.
- [ ] Exact Sionna RT and PHY pinned source and API conformance results.
- [ ] Scene geometry/material licenses, mesh and radio material runtime hashes.
- [ ] Analytical scalar/array/path checks, phase and unit conventions verified.
- [ ] Native TDL/CDL A–E and fixed conformance model distribution checks; supported scenario channels audited.
- [ ] `num_tx_node`, `num_rx_node`, array/port count/geometry, direction and link IDs present for every channel.
- [ ] SISO, SIMO, MISO and MIMO native tests pass, including full multi-node 2 BS×4 UE smoke case.
- [ ] Runtime supports sparse variable-sized per-link path/channel storage and preserves zero-path/zero-coverage cases.
- [ ] Strict direct/indirect phase/gauge decomposition correctness and consistent power normalization.
- [ ] Geometry-based LOS distinct from detected LOS and profile-category labels.
- [ ] All target names, units, references, masks, invalidity/observability/solver-failure reasons are machine readable.
- [ ] Target coverage by family, scene, band, topology, antenna and split is reported and not cherry-picked.
- [ ] No oracle metadata, site/class label or generator ID leaks into deployed model features.
- [ ] Train/val/cal/test site groups and counterfactual/array/frequency/trajectory descendants cannot overlap.
- [ ] NR DMRS waveform, effective FFT bin mapping and channel application checked against a pinned Sionna PHY reference.
- [ ] Realistic receiver link budget and noise floor verified in linear units; SNR-normalized diagnostic variants clearly separate.
- [ ] Multi-node interference channel+waveform sum, contamination and measured SINR definitions pass tests.
- [ ] Doppler and future links arise from valid time geometry and observed clock modeling; no long-horizon extrapolation masquerades as retrace.
- [ ] Scene/solver convergence and spatial realism stress cases passed or documented as failed/unsupported.
- [ ] Every released shard has checksum, provenance, licensing, content-addressed immutable pointers and atomic commit.
- [ ] Native/derived/estimated/oracle/measured confidence is labeled; external measurements only evaluate eligible targets.
- [ ] Every metric/figure auto-generated from frozen data and code hashes; negative and failed cases remain visible.
- [ ] Resource acceptance pilot measured before scaling; costs and known limitations included.

### 24.3 Intended repository deliverable tree (proposal, not existing source)

```text
umcd/
  contracts/
    manifest.v1.schema.json
    network_episode.v1.schema.json
    path_oracle.v1.schema.json
    directed_link.v1.schema.json
    target_registry.v1.yaml
    capability_matrix.v1.yaml
    invalidity_reasons.v1.yaml
  configs/
    channels/{analytic,tdl,cdl,3gpp_scenario,rt}.yaml
    antennas/{siso,simo,miso,mimo,dual_pol,massive}.yaml
    networks/{single_link,multiuser,multicell,coop}.yaml
    rf/{n78_pusch_dmrs,prs,srs,ssb}.yaml
    physics/{material_prior,rt_convergence,phase_gauge}.yaml
    benchmarks/{core,natural,heldout_scene,heldout_array,measured}.yaml
  generators/
    analytic.py
    tr38901_tdl_cdl.py
    tr38901_network.py
    rt_scene.py
    rt_network.py
    path_oracle.py
    trajectories.py
    interventions.py
    nr_signal.py
    impairments.py
  labels/
    propagation.py
    delay_angle_doppler.py
    mimo_spatial.py
    time_future.py
    multiuser_system.py
    uncertainty.py
    validity.py
  storage/
    manifests.py
    path_ragged.py
    channel_chunks.py
    target_views.py
    hashes.py
    version_migrate.py
    export.py
  qa/
    analytic.py
    standardized_tables.py
    rt_paths.py
    arrays_and_links.py
    nr_receiver.py
    network_superposition.py
    trajectories.py
    splits_and_leakage.py
    full_release.py
  splits/
    site_lock.py
    trajectory_groups.py
    family_groups.py
    test_manifest.py
  benchmark/
    tasks.yaml
    baselines.yaml
    metric_definitions.yaml
    result_registry.yaml
  tests/
    fixtures/{free_space,two_path,blocked_link,dual_pol,multi_node}.json
    test_*_contracts.py
  docs/
    simulator_support.md
    label_ontology.md
    scientific_limitations.md
    physical_units_phase_gauge.md
    dataset_card.md
    provenance.md
  results/
    E00_E30/...  # generated after execution, never prefilled
```

### 24.4 One canonical release file/folder layout

```text
releases/umcd-1.0.0/
  README.md
  DATASET_CARD.md
  CITATION.cff
  VERSION.txt
  LICENSES/    # license and source-specific redistribution rules
  manifest.json
  lockfiles/
  capability_matrix.json
  target_registry.yaml
  network_registry.parquet
  antenna_registry.parquet
  scene_registry.parquet
  site_registry.parquet
  episodes.parquet
  nodes.parquet
  directed_links.parquet
  associations.parquet
  frames.parquet
  propagation_paths/   # ragged path tables, restricted oracle
  oracle_channel/      # immutable complex CIR/CFR, restricted oracle
  receiver_observations/  # model-facing records; IQ may have access limits
  derived_targets/     # target masks/units/gauges/ground truth provenance
  trajectories/
  counterfactual_pairs/
  splits/{train,val,cal,test_site,test_array,test_real}/
  qa/{native_api,geometry,physics,statistical,receiver,split}/
  provenance/{source_sha,scene_assets,material_hashes,rt_settings}/
  benchmark/{task_views,metrics,baseline_configs,schemas}/
  generation_ledger.parquet
  failure_ledger.parquet
  known_limitations.yaml
```

For measured data whose license prevents redistribution, store manifest-only pointers, licensing and access instructions, and publish derived aggregate evaluation only if legally permitted. Do not bundle private/location-sensitive logs.

### 24.5 Final dataset card must answer

1. Who generated which data, using what hardware/runtime commit and materials, and with what ownership/licenses?
2. Which scenes/sites and time intervals are *truly independent*; which vary only receiver noise, array or carrier on the same physical scene?
3. How many **independent** sites/scenes/episodes/networks/trajectories, links, paths, NR grids, and receiver views are present by family and split?
4. What are the supported antenna/network topologies, their actual port/radio configurations and incomplete/optional variants?
5. Which labels are exact analytic, standards-model-defined, geometry-derived, numerical-RT solved, receiver-estimated or externally measured?
6. Which materials, propagation mechanisms and practical hardware effects are approximated/omitted?
7. What are the zero-path/outage/unconverged distributions and the missing/invalid rates for every label?
8. How were realistic deployment priors and stress enrichment sampled; what are valid population weights?
9. How can a third party independently reproduce raw-path records, receiver views and every main figure?
10. What findings are unsupported because of a finite solver budget, unavailable arrays, non-synchronized measured clocks, limited band/frequency, no independent calibration or other known limitations?

---

## 25. Practical prioritized build and experiment matrix

### 25.1 Smallest adequate **mandatory core** before production

| Milestone | Exact corpus | Target proof | Gate |
|---|---|---|---|
| Core-0 | 1×1 analytic free-space + 2-path + no-path | exact channel, delay/power convention, zero-path semantics | G0–G2 pass |
| Core-1 | 1×1, 1×2 SIMO, 2×1 MISO, 2×2/4×4 MIMO TDL/CDL A–E | 3GPP table/moment/correlation, port axes | G4 + M04–M06 pass |
| Core-2 | 1×1/2×2 RT geometry: direct-only, reflection-only, direct+multiple reflections, blocked, near-edge, no-ray | physical path truth, solver convergence, direct/indirect decomposition | G1–G3 pass |
| Core-3 | 1 BS×1 UE + 2 BS×4 UE with same shared arrays | multi-link truth, correct network indexing and interference | M01–M20 pilot pass |
| Core-4 | primary NR PUSCH/DMRS + ideal CSI / realistic CSI at several SNRs/BWs | receiver fidelity and validity of deployment-visible features | G5 pass |
| Core-5 | two independent urban sites plus independent indoor scenes, nontrivial trajectories | split independence, dynamic retracing, forecast horizon | G6–G8 pass |
| Core-6 | complete frozen core benchmark with MLP/analytic baselines | first coherent end-to-end result set | E07/E08/E09/E15/E16 passes review |

**This table is a set of milestones, not evidence they have already succeeded.** It deliberately does not force 128-element explicit-array RT across thousands of moving UEs before research can start.

### 25.2 Rich extended release tiers

- **UMCD-1 Core:** mandatory topologies, all scalable TDL/CDL profiles, selected scenario channels, fully qualified single-/multi-link RT, n78 DMRS, LOS/delay/reconstruction tasks, independent site splits.
- **UMCD-1 Spatial:** 4×4, 8×8, dual-pol, coherent MIMO rank, calibrated AoA/AoD/XPR, codebook beam data, selected explicit vs synthetic arrays.
- **UMCD-1 Network:** ≥2 BS and multiple UEs, cross-link channels, correct interference, pilot reuse, SINR/BLER, UL/DL; optional coordinated AP case with clock uncertainty.
- **UMCD-1 Temporal:** retraced mobility/blockage scenes, predictive CSI, future event labels, dynamic path presence, beam failure and real trajectories.
- **UMCD-1 External:** measured legal source(s), calibrated real labels where available, external-domain test and honest modality-limited evaluations.
- **UMCD-1 Frontier (conditional):** massive/XL MIMO, FR2 high-detail materials, multiband carrier aggregation, advanced full-duplex, RIS, sensing. Release only with source/native EM/hardware evidence sufficient for the claimed task.

### 25.3 Final decision rules

1. **Physical validity beats sheer sample count.** Many samples from one simplistic scene are not a benchmark for unseen-environment generalization.
2. **Breadth without observability is harmful.** Do not mark AoA, Doppler, absolute ToA or coherent multi-AP phases as supervised ground truths when the receiver input cannot physically support them. Retain oracle values and masks to study non-identifiability honestly.
3. **Consistency across multiple links is essential.** It is not enough to sample independent TDL links and call them a cellular network; interference/correlation and serving context need proper network semantics.
4. **Keep old MLP as strong baseline.** The full dataset permits real-vs-complex CNN, matched MLP, physics-aware models and classical estimators to compete on genuinely matched receiver inputs.
5. **Preserve hard cases.** Weak direct signal, rich multipath LOS, NLOS strong reflection, full blockage, interference and solver failures are precisely the regimes in which the method must demonstrate its reliability or abstention.
6. **Separate native capability from planned implementation.** Passing mocking/CI tests is not proof native Sionna works. Every dataset record has a traceable capability/QA tier.
7. **Cross-domain evaluation is mandatory.** At least one external scene family, one antenna/receiver shift and task-eligible measured validation are required for the most ambitious generalization claims.
8. **Do not guarantee 'absolute results'.** The objective is an auditable dataset with exact definitions, useful controls and independent evidence, not a claim of error-free simulation of the entire physical world.

### 25.4 Completion declaration (only after execution)

UMCD-1 can be called **research-ready** only if its release report contains: native source/version fingerprints; exact family/topology/task coverage; acceptance-gate evidence; fixed split assignments and independent sample counts; model-input/target separation; waveform/receiver correctness; path/phase/power and multi-node consistency; documented realistic vs enriched priors; immutable artifact checksums; reproducibility scripts; failure statistics; honest known limitations and compatible measured validation where provided.

**The result intended from this document is a scientifically defensible *dataset creation system and benchmark*, not predetermined performance metrics or a guaranteed S-tier paper.**

---

## References: verified authoritative technical starting points

- NVIDIA Sionna 2.2 RT `PathSolver`: https://nvlabs.github.io/sionna/rt/api/paths_solvers.html
- NVIDIA Sionna 2.2 RT `Paths` (CIR/CFR/angles/interactions/normalization): https://nvlabs.github.io/sionna/rt/api/paths.html
- NVIDIA Sionna 2.2 RT `Scene` (multiple devices, shared TX/RX arrays): https://nvlabs.github.io/sionna/rt/api/scene.html
- NVIDIA Sionna RT developer guide (synthetic vs explicit arrays): https://nvlabs.github.io/sionna/rt/developer/dev_understanding_paths.html
- NVIDIA Sionna 2.2 3GPP TR 38.901 model catalog: https://nvlabs.github.io/sionna/v2.2.0/phy/api/channel/wireless/3gpp.html
- NVIDIA Sionna 2.2 TDL implementation: https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/tdl.html
- NVIDIA Sionna 2.2 CDL implementation: https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/cdl.html
- NVIDIA Sionna 2.2 UMi implementation and multi-cell topology description: https://nvlabs.github.io/sionna/_modules/sionna/phy/channel/tr38901/umi.html
- NVIDIA Sionna RT repository: https://github.com/NVlabs/sionna-rt
- NVIDIA Sionna PHY repository: https://github.com/NVlabs/sionna
- 3GPP TR 38.901, precise table/version per simulator's `spec_version` and pinned spec edition; 3GPP TS 38.211/38.212/38.213/38.214 and TS 38.101-4 for waveform and fixed-delay conformance contexts. Verify exact version rather than treating every Release 19 increment as identical.

**Path-cap audit update (2026-10-22):** A matched depth-3, 20k-sample/source, three-seed S=0.5 box_one_screen comparison stores full CFR vectors at caps 1k/10k/1M. The LOS complex channel changes substantially at 1k (median relative L2 0.4648 versus 1M) and closely matches at 10k (6.66e-7); this does not generalize to all links or make 1M ground truth. The blocked link changes negligibly in this fixture. The initial reports lacked a pre-validation candidate counter; the follow-up instrumentation below records candidate-buffer saturation directly. G3.05 remains open until this diagnostic is integrated into canonical generation and broader stress coverage is complete.


**Path-cap audit update (2026-10-22):** The convergence runner records the candidate-buffer count before invalid-path filtering and whether the configured per-source budget filled. In a matched S=0.5, depth-3, 20k-sample/source box-scene sweep, all three LOS runs fill cap 1k; the larger caps store 1,324–1,345 candidates and show a median relative CFR error of 0.4648 at 1k versus cap 1M. The blocked link remains far below cap. The canonical static and retraced-trajectory generators now use this counter and fail closed on cap-filled outputs; cap values are part of immutable run identity. G3.05 remains open until the broader scene/material/depth/budget stress matrix and future multi-source accounting are verified. The implementation and tests are recorded in physics-status edition 3.179.
**Edition 3.174 per-source cap audit and diagnostics:** Candidate accounting now histograms Sionna's buffer source indices, so the per-transmitter cap is measured independently rather than inferred from a global path total. Static and trajectory records include the per-source pre-validation counts, and the dataset ray-budget and path-QA replay tools apply the recorded cap and stop on saturation. A native two-transmitter LLVM test verifies `(1, 1)` source counts at cap 1. G3.05 still requires broader scene/material/depth/sample matrices and cross-cap complex-channel comparisons; no universal safe cap is claimed.
**Edition 3.175 broader direct path-cap evidence:** Candidate-aware, CFR-vector sweeps now cover the S=0.5 box LOS/blocked pair at depth 6 with 100k samples/source and the real City43 16-building mesh scene at depth 3 with 20k samples/source; each uses seeds 11/23/47 and caps 1k/10k/1M. At box depth 6, cap 10k saturates LOS and changes its channel (median relative L2 0.300 vs cap 1M); at City43 depth 3, cap 1k saturates LOS only and changes CFR by median 0.131, while cap 10k does not saturate either selected link. Thus clipping depends on scenario and link even when returned valid-path totals are below the cap. The reports include pre-filter counts, cap flags, complex CFR and source/configuration hashes; the [3D scene view](figures/rt-city43-los-nlos-capstress-depth3-s05-20k-scene-20261022.png) makes all 16 building meshes visible. These synthetic material interventions and finite reference caps do not qualify propagation realism. G3.05 remains open pending broader configurations and prespecified release tolerances.

**Edition 3.182 City43 replay precision:** Comparing the overlapping depth-10, 20k-ray/source rows between the 20k-only and 20k/100k reports gives exact valid-path and per-source candidate counts for 18/18 cases, but exact CFR hashes for 0/18. Complex CFR relative L2 errors are 2.9705e-7–1.1422e-6 (median 4.3944e-7). Scene, solver-runner, package-source, and runtime fingerprints match while source revisions differ. This qualifies close numerical agreement for those shared rows only; G3.08 remains open, and 20k-versus-100k ray-budget sensitivity continues to block convergence claims under G3.01/G3.02. Per-case results and the reproduction command are recorded in `docs/research/PHYSICS_VERIFICATION_STATUS.md` and `docs/research/results/rt-city43-spatial-los-nlos-replay-audit-20261022.json`.

**Edition 3.183 City43 1M-ray budget extension:** Added a matched 1M-sample/source report for the same six links, three seeds and depths 0–10 (198 solver runs), with no reference run filling the recorded per-source path cap. At depth 10, the 20k-to-100k, 100k-to-1M, and 20k-to-1M median complex-CFR relative L2 errors are respectively 0.763/0.710, 0.843/0.817, and 0.970/0.638 for LOS/NLOS. Each budget pair changes valid-path counts in 18/18 depth-10 cases. For 100k-to-1M, median low/reference CFR mean-power ratios are 0.0247 LOS and 0.8689 NLOS; non-diffuse path-power ratios are 0.1586 and 0.9914, while diffuse path-power ratios are 0.9518 and 0.9642. Thus the LOS discrepancy is concentrated in the non-diffuse component in this fixture, while both conditions retain large complex-CFR errors. At depth zero the direct-path cases match exactly; at every tested positive depth, the 100k path counts differ from 1M for all paired link/seed cases. These high and nonmonotonic finite-budget differences leave G3.01/G3.02 open; 1M is only a larger comparison budget, not ground truth. The compressed raw report and per-case audits are `docs/research/results/rt-city43-spatial-los-nlos-depth10-s05-1m-20261022.json.gz`, `docs/research/results/rt-city43-spatial-los-nlos-20k-vs-100k-20261022.json`, `docs/research/results/rt-city43-spatial-los-nlos-100k-vs-1m-20261022.json`, and `docs/research/results/rt-city43-spatial-los-nlos-20k-vs-1m-20261022.json`.

**Edition 3.184 zero-scattering City43 control:** Repeated the 20k/100k/1M-source depth-0–10 matrices on the same 16-building scene, six links and three seeds, explicitly setting all three scene-material scattering coefficients to zero. With diffuse paths absent, the depth-10 100k-to-1M median CFR L2 error is 0.780 LOS and 0.141 NLOS; every LOS and NLOS valid-path count still changes (9/9 each). Median LOS coherent CFR mean-power ratio is 0.0483, while non-diffuse path-power ratio is 0.224; the NLOS ratios are 0.992 and 0.992. Thus the large LOS budget sensitivity persists without diffuse scattering, while the synthetic S=0.5 intervention increases the NLOS CFR discrepancy substantially in this selected geometry. At depth 3, the zero-scattering control is much closer (median CFR L2 9.3e-7 LOS, 2.8e-5 NLOS), showing the depth interaction. These are paired synthetic solver outputs, not a calibrated material model or proof that either ray budget is converged. G3.01/G3.02 remain open. Reports and audits: `docs/research/results/rt-city43-spatial-los-nlos-depth10-s0-20k-20261022.json`, `...-s0-100k-20261022.json`, `...-s0-1m-20261022.json`, `...-s0-20k-vs-100k-20261022.json`, `...-s0-100k-vs-1m-20261022.json`, and `...-s0-20k-vs-1m-20261022.json`.

**Edition 3.185 thin-screen depth-3 1M-ray diffuse-control extension:** Re-ran the exact named LOS and blocked-NLOS links with the current pinned LLVM runner, three matched seeds, max-depth 0–3, and both 100k and 1M samples/source, once with screen scattering S=0.5 and once with S=0. This makes each budget comparison share the same runner/package/runtime identity. At depth 3, zero scattering gives median complex-CFR relative L2 errors of 1.01e-5 LOS and 8.65e-6 NLOS from 100k to 1M, with unchanged valid-path counts and no cap-filled reference runs. With S=0.5, corresponding median errors are 0.266 LOS (range 0.213–0.422) and 0.0163 NLOS (0.0083–0.0428); counts change for all six matched link/seed cases. The 1M S=0.5 references contain about 55k valid paths on the LOS link and 1.0k on the blocked NLOS link; candidate caps are not filled. This contrast implicates diffuse sampling in the selected fixture’s residual sensitivity, while not proving that a million samples is converged or that S=0.5 represents a real material. The box is a single synthetic thin-screen scene, so City43 high-depth LOS sensitivity and independent-scene/material qualification remain unresolved. G3.02/G3.04 remain open. Hash-bound reports, per-case comparisons, and the [budget/depth figure](docs/research/figures/rt-box-depth3-diffuse-budget-100k-1m-20261022.png) are preserved under `docs/research/results/rt-box-los-nlos-depth0-3-{s05,s0}-100k-1m-20261022.json` and `docs/research/results/rt-box-los-nlos-{s05,s0}-depth3-100k-vs-1m-20261022.json`.

**Edition 3.186 20-seed component-distribution audit:** Added paired-seed bootstrap summaries for 100k-versus-1M samples/source at depth 3. On the S=0.5 thin-screen report, 20 seeds across the same single LOS and blocked-NLOS links yield 40 matched link/seed pairs. NLOS diffuse-only RMS delay spread is 2.962 ns at 100k versus 2.156 ns at 1M, with paired median delta +0.804 ns and 95% bootstrap interval [+0.623,+0.932] ns. The aggregate RMS delay spread changes by only −0.000485 ns (interval [−0.001783,+0.000505]); diffuse path-power delta is near zero and its interval spans zero. Thus aggregate stability masks a component-level shift in this fixture. A seed-matched S=0 control has zero diffuse paths and near-identical total observables. The intervals resample solver seeds only; they do not cover independent-scene/material uncertainty, and no admission tolerance has yet been declared. G3.02/G3.04 remain open. The 200-run raw reports, hash-bound summaries, and six-test auditor are `docs/research/results/rt-box-los-nlos-depth0-3-{s05,s0}-100k-1m-20seeds-20261022.json.gz`, `docs/research/results/rt-box-los-nlos-{s05,s0}-depth3-distribution-audit-20seeds-20261022.json`, and `scripts/audit_rt_solver_observable_distributions.py`.

**Edition 3.187 bounded City43 LOS lineage and association audit (2026-10-23):** The frozen MVP G3 contract above was retained: finite candidate/challenger configurations, symmetric unnormalized complex-CSI error with a predeclared noise floor, independent scene/site clusters for tail claims, equivalence-class path comparison, channel-domain receiver relevance before G5, and a common-SNR paired BLER comparison. The bounded diagnostic used only City43 `center-los`, seed 23, depth 10, S=0, and 100k/1M samples/source. It did not extend the budget/depth/scene/seed sweep.

The first extra zero-interaction slots appear immediately after Sionna 2.2 `SBCandidateGenerator._shoot_and_bounce()`; the explicit `_los()` stage contains exactly one populated LOS record. At 100k, 8 additional slots appear within the shoot-and-bounce candidate-buffer prefix; at 1M, 74 do. Those slots have default interaction/shape/primitive/vertex/direction values and are invalid before image processing. A temporary masked hook at the candidate write site found no recorded sample/depth/hash write for those slots (zero-valued hook fields are uninitialized sentinels, not sample 0 or a valid depth-0 candidate). They are therefore not observed nonzero-interaction candidates whose metadata was later cleared. `ImageMethod` processes invalid candidates and changes these empty slots into valid direct-like paths; field calculation and invalid-path compaction preserve their multiplicity, and final `Paths` contains 9/75 zero-interaction records. No subsequent stage creates additional zero-order rows. The first responsible stage is thus shoot-and-bounce candidate-buffer reservation/write-offset handling, with image-method processing being the stage that promotes the empty slots into returned paths. The bounded evidence does not identify a particular ray thread or a specific hash winner for these slots, so do not label this a confirmed Sionna defect or claim a fully localized source line. The source-side hook was temporary and has been removed; the pinned Sionna checkout is clean.

The write-site-instrumented repeat reproduced 194/597 valid paths and 9/75 zero-order outputs, with complex-CFR relative differences from the uninstrumented lineage run of `1.36e-7` and `1.90e-7`; the hook did not materially perturb this case. This is a numerical replay check, not a convergence or ground-truth claim. The compressed lineage JSON is `results/rt-city43-center-los-seed23-depth10-s0-path-lineage-writehook3-100k-vs-1m-20261023.json.gz`; it records the temporary instrumented package fingerprint, so it must not be confused with a clean pinned-package run. The clean-package baseline is `results/rt-city43-center-los-seed23-depth10-s0-path-lineage-100k-vs-1m-20261023.json.gz`.

The existing-artifact association audit avoids one-to-one identity claims for zero-order records. Their aggregate contribution explains 99.9949% of the total CFR delta; all 9 low-budget and 75 high-budget zero-order rows collapse to one coefficient and one delay at the declared reporting precision, with 8/9 and 74/75 carrying zero TX-angle metadata. For nonzero-order records, 1 mm matches 142/185 low and 142/522 high records; at 5 mm, 148/185 and 148/522 match, with counts unchanged at 10/20/50 mm. Matched coefficient p95 error remains at most `2.62e-5`, delay p95 `5.68e-5 ns`, direction p95 `3.84e-6 rad`, and vertex-displacement p95 `0.288 mm`. The association threshold does not change the conclusion that matched paths barely move and budget-only contributions dominate their own nonzero-order residual. Full machine-readable results are `results/rt-city43-center-los-seed23-depth10-s0-path-match-sensitivity-20261023.json`.

**Gate / roadmap consequence:** This bounded lineage question is closed at stage level; the more specific thread/hash source remains unresolved and is recorded as such. Do not spend the next project phase on more City43 budgets. G3.01/G3.02 and other applicable G3 acceptance checks remain open, so production D1 RT corpus generation is still gated. Resume the UMCD-1 implementation on work that does not depend on RT admission (standardized/analytical conformance, receiver qualification scaffolding, data-contract/split/integrity tooling), and return to RT qualification only under a separately frozen criterion-driven experiment. The path sensitivity artifact is descriptive and cannot convert either finite budget into truth.

**Edition 3.188 paired receiver uncertainty by parent:** The standardized CP-exceedance probe now includes a paired time-domain-minus-ideal-CFR BLER interval that resamples complete standardized channel-parent groups and retains paired receiver outcomes within each parent. A regression duplicates every outcome tenfold within the same four parents and verifies that the interval is unchanged, preventing repeated views/rows from being counted as independent channels. The analysis implementation is source-hash-bound in the v2 report. This is a statistical-inference scaffold for the existing 16-draw exploratory diagnostic, not a new qualification sweep or a G5.09 pass. The interval remains conditional on the sampled TDL-A draw population; broader SNR, channel, mobility, impairment, and release-family evidence remains open.

**Edition 3.189 CRC-only matched SNR diagnostic:** Added a three-point (60/80/100 dB), 16-parent TDL-A receiver diagnostic with exact parent/channel-truth, transmit-seed, and noise-seed lineage checks across SNR points, plus paired parent-cluster uncertainty. A payload cross-check that regenerated transmitted bits from stored seeds disagreed with CRC outcomes in part of the 16-parent run. The receiver-record contract does not store the actual transmitted payload, so the regenerated-bit comparison is not accepted as BLER evidence; its cause is not claimed localized. The diagnostic now uses Sionna's transport-block CRC as its sole BLER outcome and the result has a new CRC-only schema at `docs/research/results/g5-standardized-pusch-cp-exceedance-snr-curve-v3-20261023.json`. The earlier v1/v2 reports remain exploratory historical artifacts; do not interpret their payload-mismatch fields. Sixteen TDL-A parents give wide intervals and cannot establish a deployment BLER curve, a G5 pass, or G3 receiver impact. This bounded PHY diagnostic does not reopen City43 or authorize RT production D1 generation.
