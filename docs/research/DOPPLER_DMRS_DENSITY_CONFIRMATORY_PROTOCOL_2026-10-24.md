# G5 Doppler-aware DM-RS density confirmation protocol

**Status:** frozen before the confirmatory run. This protocol tests one
synthetic 6-RB, 30-kHz PUSCH configuration. It is not a general G5.09 pass,
field-channel claim, or authorization for production RT generation.

## Question

On a preregistered two-path channel distribution with a 750-Hz echo Doppler,
does PUSCH DM-RS additional-position 3 improve channel-cluster BLER and useful
transport-block goodput per slot relative to additional-position 0, under the
same channel, waveform-noise, and transmitter seeds? The static condition
measures the resource-overhead cost when the channel does not age.

## Frozen design

- Confirmatory channel seeds: integers 85,001 through 85,064 inclusive. They
  are disjoint from the 84,001–84,064 development scan and all earlier 73,001–
  73,064 Doppler stress seeds.
- Independent unit: one seeded channel draw. Eight PUSCH transport blocks are
  nested within each channel and do not increase the channel sample count.
- Channel: direct coefficient `0.85+0j`; one echo with delay sampled uniformly
  from integer samples 2–8, amplitude uniformly 0.20–0.50, phase uniformly over
  `[-pi, pi]`, and Doppler 0 Hz (static) or 750 Hz (continuous sample time).
- NR: Sionna 2.2 CPU, 6 RB, 30-kHz SCS, one layer, same default MCS, 8 blocks per
  channel, waveform AWGN complex variance 0.001, and DM-RS additional positions
  0 and 3. The linear-interpolation PUSCH receiver is used for both.
- Pairing: each channel uses the same transmitter seed, waveform-noise seed,
  channel coefficients, and channel realization for both DM-RS configurations
  and both channel conditions. The noise waveform digest is retained for each
  configuration. Transport-block sizes differ because the DM-RS overhead
  differs; goodput is therefore reported as successfully decoded information
  bits per slot, not BLER alone.
- No parameter, seed, configuration, or endpoint may be changed after examining
  confirmatory outcomes. A failed run may be debugged only for implementation
  faults documented independently of the result; any protocol change requires
  a new disjoint seed range and a new protocol file.

## Acceptance claims

One family covers these three claims at family-wise alpha 0.01. Bonferroni
allocation is alpha/3 per claim; bootstrap intervals use the independent
channel as the resampling unit and 20,000 replicates with a fixed seed.

1. **Static reference adequacy:** additional-position-0 static BLER point
   estimate is at most 0.05, and the one-sided exact Clopper–Pearson upper
   bound on channel-level any-failure prevalence is at most 0.10.
2. **Mobile BLER benefit:** at 750 Hz, the equal-channel mean absolute BLER
   reduction (position 0 minus position 3) is at least 0.10 and the lower end
   of its Bonferroni-adjusted paired cluster-bootstrap interval is above zero.
3. **Mobile goodput benefit:** at 750 Hz, the equal-channel mean difference in
   successfully decoded transport-block bits per slot (position 3 minus
   position 0) has a Bonferroni-adjusted paired cluster-bootstrap interval
   whose lower end is above zero.

The static-condition goodput and BLER contrasts, the mobile BLERs, all raw
channel-level outcomes, and all confidence intervals will be reported whether
or not the three claims pass. No acceptance threshold is inferred from the
confirmatory results. A pass supports only this fixed synthetic model and NR
configuration; broader Doppler, SNR, coding, mobility, measured-channel, and
G5.09 qualification remain separate requirements.

## Reproduction

Run `uv run --no-sync --locked python experiments/qualify_g5_doppler_dmrs_density.py`.
The output is
`docs/research/results/g5-doppler-dmrs-density-confirm-20261024.json` and binds
the exact protocol SHA-256, source hashes, configuration, seed lists, retained
outcomes, statistical method, and result digest.
