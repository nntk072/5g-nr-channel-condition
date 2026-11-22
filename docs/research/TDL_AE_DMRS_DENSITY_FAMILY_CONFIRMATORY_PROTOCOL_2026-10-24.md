# G5 TDL-A–E DM-RS density family confirmation protocol

**Status:** frozen before the confirmatory run. This protocol extends the
bounded TDL-C receiver question across all five standardized TDL profiles. A
pass is still conditional standardized-channel evidence, not G5.09 closure,
field-channel validity, or authorization for production RT generation.

## Question

For 3GPP TR 38.901 TDL-A, TDL-B, TDL-C, TDL-D, and TDL-E, each with 100-ns
delay spread at 3.5 GHz, 30-kHz SCS, and 120-km/h UE speed, does
`dmrs-AdditionalPosition=3` improve matched PUSCH BLER and successfully decoded
information bits per slot over position 0, while position-0 static reception
meets the predeclared reference-adequacy rule for every profile?

## Frozen design

- Independent units: 64 independently seeded channel draws per profile, seeds
  934001 through 934064. The range is disjoint from the TDL-C development and
  confirmation ranges 932001–932064 and 933001–933064 and from the previously
  retained 73,001–85,064 two-path studies.
- Channel: native Sionna 2.2 sample-domain TDL-A–E; 100-ns RMS delay spread;
  3.5-GHz carrier; zero speed for the reference and 120 km/h for the mobile
  conditions. Channel memory must fit the generated cyclic prefix;
  `allow_cp_overrun` is false. Any CP-coverage failure invalidates that profile
  and makes the family verdict `NOT_MET`.
- NR: CPU, one receive antenna, 12-RB PUSCH, 30-kHz SCS, 18-dB configured
  receive-grid SNR, one transport block per independently seeded channel, and
  DM-RS additional positions 0 and 3. The receiver uses the configured
  estimator and Sionna matched PUSCH decoder.
- Conditions for every profile and seed: static position 0, mobile position
  0, and mobile position 3.
- Pairing: within each profile, use the same channel seed, transmitter RNG
  seed, and waveform-noise seeds across conditions. Verify byte-identical
  payloads between static and mobile position 0. Position-0 and position-3
  transport-block sizes may differ; retain payload hashes but do not require
  them to match. Within mobile conditions, require byte-identical channel-truth
  bindings. Do not assume cross-profile channel draws are paired or independent
  beyond their separately generated profile-specific realizations.
- Information goodput is decoded transport-block bits per slot, counting a
  failed CRC as zero useful bits. Each channel seed contributes one paired
  observation; blocks are not multiplied into the independent sample count.
- Family-wise alpha is 0.01 across 15 claims: three claims for each of five
  profiles. Bonferroni allocation is alpha/15 per claim. Use 20,000 paired
  channel-seed bootstrap replicates with fixed profile-specific recorded seeds
  for BLER-reduction and goodput intervals. Use a one-sided exact
  Clopper–Pearson upper bound at alpha/15 for static reference channel
  failures.
- Do not change parameters, seeds, configurations, criteria, or endpoints after
  inspecting outcomes. Any implementation repair needs an independently
  documented defect, a revised protocol, and a fresh disjoint seed range.

## Acceptance claims

Each profile must satisfy all three claims under the common 15-claim family;
the overall family verdict is `PASS_BOUNDED_TDL_AE_FIXTURE` only if all 15
claims pass.

1. **Static reference adequacy:** position-0 static BLER is at most 0.05, and
   the one-sided family-adjusted exact upper bound on the fraction of channel
   draws with any CRC failure is at most 0.10.
2. **Mobile BLER benefit:** the equal-channel mean absolute BLER reduction
   (mobile position 0 minus mobile position 3) is at least 0.10, and the lower
   end of its family-adjusted paired bootstrap interval is above zero.
3. **Mobile goodput benefit:** the family-adjusted paired bootstrap interval
   for mobile position-3 minus position-0 decoded information bits per slot
   has a lower end above zero.

Retain all per-seed outcomes, source/configuration/protocol digests, channel
truth/noise/payload bindings, CP coverage, point estimates, confidence bounds,
and each profile's verdict regardless of the family result. A failure for any
profile makes the overall family `NOT_MET`; do not pool profiles to conceal a
failed profile or widen the target after seeing results.

This design generalizes only across the five named TDL profiles under one
carrier, delay spread, SCS, speed, SNR, PUSCH allocation, receiver, and seed
count. It does not establish continuous channel-law validity, measured-channel
performance, broader SNR/mobility coverage, G5.09 closure, G3 physical truth,
or production RT authorization.
