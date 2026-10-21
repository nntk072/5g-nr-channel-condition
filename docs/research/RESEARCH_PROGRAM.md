# Research upgrade for Nguyen (2025): physical LOS, standardized profiles and domain shift

## Status and integrity statement

This is a **tested baseline/framework**, not a finished manuscript or a top-tier result.
Sionna 2.2 RT/PHY integration is version-aware but requires run-time testing on an
appropriately provisioned host. Debug synthetic datasets must **not** be reported
as publication results. Profile labels A/B/C=0 and D/E=1 are a **surrogate** for
LOS status, not independently measured geometric visibility. No claimed accuracy
or S-tier performance is included.

## Paper we propose

**Beyond Channel-Profile Recognition: Physically Grounded LOS/NLOS Detection Under
3GPP-to-Ray-Tracing Domain Shift in 5G NR**

**Hypotheses (falsifiable):**
1. Randomly split TDL/CDL classification may learn standardized profile identity
   rather than transferable geometric visibility.
2. Source-trained MLP and classical classifiers lose AUROC on unseen ray-traced
   scenes even when signal SNR and numerology are matched.
3. Domain alignment (CORAL-MLP) can improve *unseen-scene* performance over an
   identical MLP without CORAL. If not supported, publish the negative finding.
4. LOS links with both direct and indirect paths are a harder and more realistic
   setting than artificially single-ray LOS links.

## What this implementation changes

- New `loslab` Python package, leaving legacy notebooks intact as historical artifacts.
- Lazy imports for Sionna 2.2 + separate PHY and RT modules.
- Standardized TDL/CDL-A-E generated from the Sionna 2.2 PyTorch channel models.
- Actual RT channel includes LOS, specular, diffuse reflection and refraction.
  A separate depth-zero oracle detects direct geometric visibility; *it never
  feeds model input*. Only full-channel CFR (with independent observation noise)
  is provided to the classifier. A strict default rejects LOS links lacking
  a distinct indirect path.
- MLP remains primary baseline. RF, logistic regression, and equal-capacity
  CORAL-MLP (penalty-on vs penalty-off) evaluate whether domain-invariant
  representations, **not CNN depth alone**, contribute.
- Dataset file has independently stored `h`, `y`, `group`, `domain`, `source`,
  and metadata. Feature extraction consumes **only `h`**.
- Train/validation/test group-disjoint; holdout supports `rt_munich`/`rt_etoile`.
  The operating threshold is selected on validation, never held-out test.
- Five prespecified seeds, calibration metrics, block bootstrap intervals and
  per-seed prediction files. Results have exact dataset SHA256 and provenance.
- Aalto_MS_Thesis inheritance: adapted run provenance, canonical configuration
  hashes and atomic JSON artifact patterns from the author's Aalto repository.
  The unrelated trading/RL stack is deliberately not copied or used.

## Why this alone is not yet a flagship contribution

CORAL and an MLP are existing ideas; novelty cannot consist solely of
implementing them. To compete for IEEE TWC/JSAC / exceptional ICC/Globecom
submissions, the work needs a new physical/algorithmic insight validated by
controlled experiments, preferably independent real measured channels. The
next algorithm direction to test is **path-presence uncertainty with group-DRO
and physical consistency**, but it is only a hypothesis, *not implemented* here.

## Critical missing publication-grade components

1. **Realistic 5G NR receiver:** Replace direct CFR observations with actual
   PUSCH/PRS reference-signal estimates and nonideal channel estimation; current generic comb-pilot LS stress test is **not** standardized NR;
   include AWGN, CFO, phase noise, quantization, timing error, interference.
   Current `add_estimation_noise()` is a diagnostic approximation ONLY.
2. **Scenario realism:** obtain multiple independent 3D scenes (with materials),
   calibrate to measured PDP/delay spreads, vary carrier, array, mobility,
   material, spatial blockage and scattering, and verify realistic direct/
   indirect path energy. The built-in Munich/Étoile scenes alone are insufficient.
3. **Geometric visibility oracle quality:** validate ground truth via direct-ray
   and obstruction checks across scenes, materials/transmission settings, and
   near-boundary cases; include LOS with NLOS multipath, NLOS with refracted paths,
   and no-reception as separate scenario definitions.
4. **Measurement data:** measurement campaign or publicly available multi-scene
   CSI with reliable geometric LOS status; no unsupported inference of LOS labels.
5. **Strong competitor baselines:** K-factor, wideband delay spread, GBM/XGBoost,
   CNN and complex-valued net from thesis, calibrated logistic, group-DRO,
   alternative adaptation, and *computation/latency/parameter* comparisons.
6. **Evaluation control:** scene-level holdouts (not merely coordinate blocks),
   boundary-buffered spatial splits, multiple independent trajectories, no reuse
   of camera/receiver path across splits, balanced LOS/NLOS per scene and device,
   CIs computed across independent scenes and repeats rather than arbitrary rays.
7. **Generalization robustness:** train TDL+CDL and test new RT scenes (zero-shot),
   train RT seen scenes and test held-out city; test unseen channel profiles,
   non-matched SNR, delay spread, bandwidth, mobility and carrier, and real data.
8. **Risk of self-plagiarism:** cite user's 2025 Tampere thesis explicitly,
   say which code/results were reused, and report the scientifically novel part.

## Suggested experiments for one strong paper

- E0: Reproduce 2025 baseline *with code/data*, not reported table numbers.
- E1: TDL/CDL A-E grouped test vs held-out Sionna RT scenes; profile-only
  oracle upper bound used strictly for diagnosis, not as a fair classifier.
- E2: Domain invariance ablation with RF, MLP, same-architecture CORAL0/CORAL.
- E3: Counterfactual: retain scene and reflected rays, block ONLY the direct path;
  use paired transmitter/receiver channels and oracle label changes.
- E4: Compare pure LOS direct-only vs LOS with multipath; vary relative direct
  power/K-factor and test near-occlusion uncertainty.
- E5: Five-seed paired deltas + scene-level confidence intervals;
  calibration/OOD detection; rate of coverage for abstention thresholds.
- E6: Nonideal NR pilot estimation with parameterized receiver impairments.
- E7: Independent real-data evaluation; demonstrate downstream positioning gains.

**Pre-registration:** choose target domains and seeds before fitting anything;
never tune on held-out ray-traced or real test labels. Metrics primary: AUROC
on a held-out scene and 95% scene/link-block bootstrap CI. Secondary: AUPRC,
balanced accuracy, Brier score, ECE and run cost. Track source/checkpoint SHA256 and observed-comb SNR.

## Source docs (accessed October 21, 2026)

- Sionna 2.2 installation: https://nvlabs.github.io/sionna/v2.2.0/installation.html
- Sionna RT PathSolver: https://nvlabs.github.io/sionna/rt/api/paths_solvers.html
- Sionna RT Paths: https://nvlabs.github.io/sionna/rt/api/paths.html
- Sionna PHY TDL: https://nvlabs.github.io/sionna/v2.1.0/phy/api/channel/wireless/sionna.phy.channel.tr38901.TDL.html
- Sionna PHY CDL: https://nvlabs.github.io/sionna/v2.1.0/phy/api/channel/wireless/sionna.phy.channel.tr38901.CDL.html
- User's old thesis: `NguyenLong.pdf`, Tampere University, July 2025.
