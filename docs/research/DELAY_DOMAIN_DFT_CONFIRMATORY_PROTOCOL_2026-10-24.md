# G5 finite delay-domain DFT receiver confirmation protocol

**Status:** frozen before the confirmatory run. This protocol tests one
synthetic two-path PUSCH configuration with a new finite delay-domain DM-RS
estimator. It does not revise the failed linear-estimator confirmation, qualify
measured channels, or authorize production RT generation.

## Question

With the same two-path channel law and receiver criteria as the frozen G5.09
DM-RS-density study, does additional-position-3 DM-RS improve channel-cluster
BLER and useful transport-block goodput relative to additional-position 0 when
both receivers use the finite delay-domain DFT estimator defined below?

## Estimator fixed before confirmation

- Use Sionna's raw received DM-RS observations before adjacent-frequency CDM
  averaging, and divide each observation by its known pilot symbol.
- For each DM-RS symbol, fit a complex least-squares impulse response on integer
  delays 0 through 8 samples inclusive. The design matrix uses the actual
  centered FFT-bin indices and the complex exponential basis
  `exp(-j 2 pi k l / N_FFT)`.
- Reconstruct the frequency response over all FFT bins. Interpolate the fitted
  tap coefficients linearly between DM-RS symbols and hold the nearest estimate
  outside the first/last DM-RS symbol.
- Propagate the raw pilot-noise variance through the least-squares pseudoinverse
  and the frequency-response basis; for temporal interpolation, combine the
  endpoint variances with squared interpolation weights.
- This implementation supports the frozen single-TX-stream configuration.
  The fitted delay span includes one sample beyond the 7-sample cyclic prefix;
  any resulting ISI remains part of the waveform experiment and is not
  corrected or discarded by the estimator.

No estimator span, interpolation rule, threshold, seed, or endpoint may change
after this protocol is frozen. A failed run may be debugged only for an
implementation defect identified independently of outcomes; a protocol change
requires a new version and a new unused seed range.

## Frozen design

- Confirmatory independent channel seeds: integers 88,001 through 88,064
  inclusive. These are disjoint from the earlier 73,001, 84,001, 85,001, and
  86,001 development/confirmation ranges. An exact numeric-value audit found
  no collisions for the 88,001–88,064 roster in existing result JSON files.
- Independent unit: one seeded channel draw. Eight PUSCH transport blocks are
  nested within each channel and do not increase the independent sample count.
- Channel: direct coefficient `0.85+0j`; one echo with integer delay sampled
  uniformly from 2–8 samples, amplitude uniformly from 0.20–0.50, phase
  uniformly from `[-pi, pi]`, and Doppler 0 Hz (static) or 750 Hz (continuous
  sample time). Channel draws use `random.Random(channel_seed)` in the frozen
  draw order: delay, amplitude, then phase.
- NR: Sionna 2.2 CPU; 6-RB BWP, 30-kHz SCS, one layer, default MCS, 8 blocks per
  channel, complex waveform AWGN variance 0.001, and DM-RS additional positions
  0 and 3. Both receivers use the finite delay-domain DFT estimator above.
- Pairing: for each channel seed, both DM-RS settings use transmitter seed
  `channel_seed + 1,000,000` and waveform-noise seed `channel_seed + 2,000,000`.
  The same channel coefficients and generated noise waveform are used in both
  static and mobile conditions. Per-configuration noise digests and all
  channel-level CRC outcomes are retained. Transport-block sizes may differ
  because of DM-RS overhead; goodput is decoded information bits per slot.

## Acceptance claims

One family covers these three claims at family-wise alpha 0.01. Bonferroni
allocation is alpha/3 per claim; paired bootstrap intervals use the independent
channel as the resampling unit with 20,000 replicates. The BLER bootstrap seed
is 88,099 and the goodput bootstrap seed is 88,100.

1. **Static reference adequacy:** additional-position-0 static BLER point
   estimate is at most 0.05, and the one-sided exact Clopper–Pearson upper bound
   on channel-level any-failure prevalence is at most 0.10.
2. **Mobile BLER benefit:** at 750 Hz, the equal-channel mean absolute BLER
   reduction (position 0 minus position 3) is at least 0.10 and the lower end
   of its Bonferroni-adjusted paired cluster-bootstrap interval is above zero.
3. **Mobile goodput benefit:** at 750 Hz, the equal-channel mean difference in
   successfully decoded transport-block bits per slot (position 3 minus
   position 0) has a Bonferroni-adjusted paired cluster-bootstrap interval whose
   lower end is above zero.

Report the static contrasts, every channel-level outcome, and every interval
whether or not the claims pass. All three claims must pass for this named
estimator/configuration result to meet the frozen family. A pass is not a
general G5.09, field-channel, standards-conformance, or production-RT
qualification.

## Reproduction

Run `uv run --no-sync --locked python experiments/qualify_g5_delay_domain_dft.py`.
The command writes
`docs/research/results/g5-delay-domain-dft-confirm-20261024.json` and binds the
protocol digest, source hashes, seed roster, estimator configuration, retained
outcomes, statistical method, and result digest. The report must be independently
audited before interpreting the three claims.
