# G5 TDL-A–E signed-lag DFT family confirmation protocol

**Status:** frozen before the confirmatory run. This is a bounded CPU receiver
comparison across five standardized statistical channel profiles. It does not
close G5.09, validate measured propagation, qualify a deployment receiver, or
authorize production RT generation.

## Question

Across sampled Sionna TDL-A through TDL-E at the registered 100-ns delay
spread, does increasing PUSCH DM-RS additional position from 0 to 3 improve
mobile receiver BLER and decoded goodput with the same signed-lag finite-delay
DFT estimator, while position 0 satisfies the static-reference adequacy
limits for every profile?

## Frozen estimator

Use the raw Sionna DM-RS observations divided by their known pilot symbols;
fit integer delays -6 through +7 samples inclusive using complex least squares
on the actual centered FFT bins; reconstruct the CFR; linearly interpolate in
time between pilot-bearing symbols and hold the nearest estimate outside the
pilot range; and propagate pilot-noise variance through the fit and temporal
interpolation as implemented in `experiments/g5_cp_bounded_dft_estimator.py`.
The identical estimator configuration applies to every profile and condition.
No estimator, interval, interpolation, noise model, or criterion may change
after the run starts. Any correction requires a new protocol and disjoint
seeds.

## Frozen design

- Profiles: Sionna TDL-A, B, C, D, and E, specification version 19.2.
- Shared channel and NR settings: 100-ns delay spread, 3.5-GHz carrier,
  120-km/h mobile speed, 12-RB PUSCH, 30-kHz SCS, one layer, CPU, configured
  receive-grid SNR 18 dB, and cyclic-prefix overrun disabled. All conditions
  must show complete cyclic-prefix coverage.
- Per-profile conditions: static position 0; mobile position 0; mobile
  position 3. All use the frozen estimator above.
- Independent channel seeds: 949001–949080 inclusive for each profile. Each
  seed contributes one transport block per condition; channel seed is the
  independent resampling unit. Before preregistration, these values were
  recursively checked against numeric values in all 222 saved JSON and
  compressed-JSON research result artifacts; there were no collisions with
  retained prior outcomes.
- Pairing: static and mobile position 0 share transmitted bits and waveform
  noise seeds. Mobile positions 0 and 3 share channel truth and waveform-noise
  seeds within each profile and parent. Retain all channel, payload, noise,
  estimator, and source bindings.
- Family-wise alpha: 0.01 across 15 claims (three claims × five profiles),
  Bonferroni alpha 0.01/15 per claim. Use 20,000 paired channel-seed bootstrap
  replicates per profile, with BLER/goodput seeds 949101/949102 for TDL-A,
  then consecutive pairs through 949109/949110 for TDL-E. Use the adjusted
  one-sided exact Clopper–Pearson upper bound for static channel-level failure
  prevalence.

## Acceptance criteria

Every one of the 15 claims must pass for `PASS_BOUNDED_TDL_AE_FIXTURE`:

1. **Static adequacy, each profile:** position-0 static BLER ≤0.05 and its
   family-adjusted one-sided exact upper bound on channel-level failure
   prevalence ≤0.10.
2. **Mobile BLER benefit, each profile:** equal-channel mean absolute BLER
   reduction (position 0 minus position 3) ≥0.10, and the family-adjusted
   paired cluster-bootstrap interval lower endpoint is above zero.
3. **Mobile goodput benefit, each profile:** the family-adjusted paired
   cluster-bootstrap interval lower endpoint for position-3 minus position-0
   successfully decoded information bits per slot is above zero.

Retain all channel outcomes and intervals, including failures. Any failed
static reference makes the family `NOT_MET`; mobile effects cannot rescue it.
Do not pool this study with the TDL-C-only signed-lag result, the earlier
linear-estimator families, or the two-path study. A pass supports only these
five named synthetic profile/configuration combinations; it is not measured
channel validation or general G5.09 closure.

## Reproduction and audit

Run `uv run --no-sync --locked python
experiments/qualify_g5_tdl_dft_profile_matrix.py`; audit with
`uv run --no-sync --locked python
experiments/audit_g5_tdl_dft_profile_matrix.py`. The family report retains
each profile report and binds the protocol, source inventory, estimator,
seed roster, complete per-seed outcomes, and all acceptance calculations.
