# G5 TDL-C signed-lag DFT estimator confirmation protocol

**Status:** preregistered before the confirmatory run. This is a bounded CPU
comparison on one standardized TDL-C configuration. It does not close G5.09,
validate measured propagation, qualify a deployment receiver, or authorize
production RT generation.

## Question

For sampled TDL-C at the registered 100-ns delay spread, does increasing
PUSCH DM-RS additional position from 0 to 3 improve mobile receiver BLER and
decoded goodput when both configurations use the same signed-lag finite-delay
DFT estimator, while position 0 meets the static-reference adequacy limits?

## Estimator fixed before confirmation

- Extract raw Sionna DM-RS pilot observations and divide by the known pilot
  symbols before fitting.
- Fit integer channel delays from -6 through +7 samples inclusive using the
  complex least-squares frequency basis on the actual centered FFT bins.
- Reconstruct the full CFR and apply the estimator's registered linear
  interpolation between pilot-bearing OFDM symbols, holding the nearest
  estimate outside the pilot-symbol range.
- Propagate pilot noise variance through the least-squares fit and temporal
  interpolation as implemented in
  `experiments/g5_cp_bounded_dft_estimator.py`.
- Do not change the delay interval, interpolation, variance model, estimator,
  thresholds, seeds, or endpoints after the run starts. Any implementation
  correction requires a new protocol revision and unused seed roster.

The signed support `[-6, 7]` is the pinned Sionna sampled-channel support for
the 12-RB, 30-kHz, 100-ns configuration. It includes all 14 taps used by the
sample-domain TDL link and is distinct from the earlier synthetic two-path
confirmation's 0–8-sample model.

## Frozen design

- Profile: Sionna TDL-C, specification version 19.2; 100-ns delay spread,
  3.5-GHz carrier, and mobile speed 120 km/h.
- NR configuration: 12-RB PUSCH, 30-kHz SCS, one layer, CPU, configured
  receive-grid SNR 18 dB, and cyclic-prefix overrun disabled. All three
  conditions must report complete CP coverage.
- Conditions: static position 0; mobile position 0; and mobile position 3.
  Use the signed-lag finite-delay DFT estimator in every condition.
- Independent channel seeds: 948001 through 948080 inclusive. Each channel
  seed contributes one transport block per condition; the channel seed is the
  independent resampling unit. This range was audited against every numeric
  value in the saved JSON and compressed JSON result artifacts before
  preregistration and had no collisions.
- Pairing: static and mobile position 0 must share transmitted bits and
  waveform-noise seeds. Mobile positions 0 and 3 must share channel-truth and
  waveform-noise seeds for each parent. Record all channel-truth, payload,
  noise, and estimator bindings.
- Family-wise alpha is 0.01 over three claims; use Bonferroni alpha 0.01/3 per
  claim. Use 20,000 paired channel-seed bootstrap replicates with BLER seed
  948099 and goodput seed 948100. Use a one-sided exact Clopper–Pearson bound
  for static channel-level failure prevalence.

## Acceptance claims

All three claims must pass for a bounded result on this named TDL-C setup:

1. **Static adequacy:** position-0 static BLER is at most 0.05 and its
   family-adjusted one-sided exact upper bound on channel-level failure
   prevalence is at most 0.10.
2. **Mobile BLER benefit:** the equal-channel mean absolute BLER reduction
   (position 0 minus position 3) is at least 0.10, and the lower endpoint of
   its family-adjusted paired cluster-bootstrap interval is above zero.
3. **Mobile goodput benefit:** the lower endpoint of the family-adjusted paired
   cluster-bootstrap interval for position-3 minus position-0 successfully
   decoded information bits per slot is above zero.

Retain every channel-level outcome and every interval whether or not the
claims pass. A failed static reference makes the result `NOT_MET`; mobile
benefit cannot rescue it. No outcome may be dropped or pooled with prior TDL
or two-path runs.

## Reproduction

Run `uv run --no-sync --locked python experiments/qualify_g5_tdl_dft_density.py`.
The output is
`docs/research/results/g5-tdl-c-signed-lag-dft-confirm-20261024.json`.
Verify it with
`uv run --no-sync --locked python experiments/audit_g5_tdl_dft_density.py`.
The report binds this protocol, source hashes, the exact estimator interval,
seed roster, retained per-channel outcomes, and recomputed acceptance claims.
