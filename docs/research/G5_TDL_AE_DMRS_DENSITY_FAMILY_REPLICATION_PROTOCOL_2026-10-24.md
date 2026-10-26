# G5 TDL-A–E DM-RS density family replication protocol

**Status:** preregistered before the replication run. This is a new,
independent confirmation with disjoint channel seeds; it does not pool or
replace the retained 64-seed study, whose verdict remains `NOT_MET`.

## Question

Under the same bounded TDL-A–E receiver question as the first family protocol,
does DM-RS additional-position 3 improve paired mobile PUSCH BLER and decoded
goodput while each profile's position-0 static reference meets the frozen
adequacy rule?

## Frozen design

- Independent units: 80 separately seeded channel draws per profile, seeds
  935001 through 935080. This range is disjoint from all prior retained TDL-C
  and TDL-A–E channel ranges, including 934001–934064. The earlier 64-seed
  study is not pooled with this replication.
- Profiles and conditions: TDL-A, B, C, D, and E at 100-ns delay spread,
  3.5-GHz carrier, 120-km/h mobile speed, 30-kHz SCS, 12-RB PUSCH, 18-dB
  configured receive-grid SNR, and static-position-0, mobile-position-0, and
  mobile-position-3 conditions. CPU execution only; CP overrun is disabled and
  every run must report complete CP coverage.
- Pairing: within each profile and seed, retain the same payload and noise
  bindings for static/mobile position 0 and the same channel-truth/noise
  bindings for mobile positions 0 and 3. One channel seed is one independent
  unit; the transport block is nested within it.
- Family-wise alpha remains 0.01 across 15 claims (three per profile), with
  Bonferroni alpha/15. Use 20,000 paired channel-seed bootstrap replicates and
  fixed profile-specific recorded seeds. The static failure-prevalence bound
  is one-sided exact Clopper–Pearson at alpha/15.
- The 80-unit count is frozen before this replication. With zero failures, the
  exact bound at alpha/15 first reaches the unchanged 0.10 static-reference
  limit at 70 units; 80 provides a prespecified margin. This sample-count
  design changes no acceptance criterion and uses no outcomes from the first
  study to calculate a pooled estimate.
- No parameter, seed, configuration, or endpoint may be changed after
  inspecting this replication. Any implementation repair requires an
  independently documented defect, a revised protocol, and another disjoint
  seed range.

## Acceptance claims

For each profile, all three claims must pass under the same 15-claim family:

1. **Static reference adequacy:** position-0 static BLER is at most 0.05, and
   the one-sided family-adjusted exact upper bound on channel draws with any
   CRC failure is at most 0.10.
2. **Mobile BLER benefit:** equal-channel mean absolute reduction (mobile
   position 0 minus position 3) is at least 0.10 and the family-adjusted
   paired-bootstrap interval lower end is above zero.
3. **Mobile goodput benefit:** the family-adjusted paired-bootstrap interval
   for position-3 minus position-0 decoded bits per slot has a lower end above
   zero.

Any failed claim makes this replication `NOT_MET`; retain all outcomes and
profile verdicts. A pass supports only the five named standardized profiles
under this fixed link, delay spread, speed, SNR, waveform, and receiver. It is
not measured-channel validity, broad G5.09 closure, or production RT
authorization.
