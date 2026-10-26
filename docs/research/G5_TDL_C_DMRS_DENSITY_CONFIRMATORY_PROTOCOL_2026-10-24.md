# G5 TDL-C DM-RS density confirmation protocol

**Status:** frozen before the confirmatory run. This protocol evaluates one
standardized statistical channel and PUSCH configuration. A pass is bounded
receiver evidence, not a general G5.09 qualification, field-channel claim, or
authorization for production RT generation.

## Question

For a 3GPP TR 38.901 TDL-C channel with 100-ns delay spread at 3.5 GHz, 30-kHz
SCS, and 120-km/h UE speed, does `dmrs-AdditionalPosition=3` improve PUSCH
BLER and successfully decoded information bits per slot over position 0? A
matched zero-speed position-0 condition checks reference receiver adequacy.

## Frozen design

- Independent units: 64 separately seeded TDL channel draws, each carrying one
  transport block per receiver condition. Seeds are 933001 through 933064;
  they are disjoint from the 932001–932064 TDL-C development scan, the
  84,001–84,064 two-path DM-RS scan, the 85,001–85,064 confirmation, and the
  73,001–73,064 Doppler stress scan.
- Channel: Sionna 2.2 sample-domain TDL-C; 100-ns RMS delay spread; 3.5-GHz
  carrier; zero speed for the reference and 120 km/h for mobile conditions.
  Channel memory must fit the generated cyclic prefix (`allow_cp_overrun` is
  false); a run that reports otherwise is invalid.
- NR: CPU, one receive antenna, 12-RB PUSCH, 30-kHz SCS, 18-dB configured
  receive-grid SNR, one frame per independently seeded channel, and DM-RS
  additional positions 0 and 3. The receiver uses its configured DM-RS
  estimator and Sionna's matched PUSCH decoder.
- Conditions: static position 0, mobile position 0, and mobile position 3.
- Pairing: use the same channel seed, transmitter RNG seed, and waveform-noise
  seeds across all conditions. Verify byte-identical payloads between static
  and mobile position 0. The position-0 and position-3 transport-block sizes
  can differ, so their payload byte hashes are retained but are not required
  to match. Within each mobile seed, require byte-identical channel-truth
  bindings between DM-RS positions. Static and mobile truth are paired by seed
  index but are not asserted to have identical time-varying channel tensors.
- Information goodput is decoded transport-block bits per slot, counting a
  failed CRC as zero useful bits. Contrasts are paired by channel seed; each
  seed contributes one observation.
- Family-wise alpha is 0.01 across the three claims below. Bonferroni
  allocation is alpha/3 per claim. Use 20,000 paired channel-seed bootstrap
  replicates with a fixed recorded RNG seed for each interval; use a one-sided
  exact Clopper–Pearson upper bound at alpha/3 for reference channel failures.
- No parameter, seed, configuration, or endpoint may be changed after
  inspecting confirmatory outcomes. An implementation repair requires a
  documented defect independent of the observed outcomes and a new protocol
  plus disjoint seeds.

## Acceptance claims

1. **Static reference adequacy:** the position-0 static BLER is at most 0.05,
   and the one-sided family-adjusted exact upper bound on the proportion of
   channels with a CRC failure is at most 0.10.
2. **Mobile BLER benefit:** the equal-channel mean absolute BLER reduction
   (mobile position 0 minus mobile position 3) is at least 0.10, and the lower
   end of its family-adjusted paired bootstrap interval is above zero.
3. **Mobile goodput benefit:** the family-adjusted paired bootstrap interval
   for mobile position-3 minus position-0 decoded information bits per slot
   has a lower end above zero.

Report all three conditions, every per-seed outcome, the point estimates,
confidence bounds, source/configuration/protocol digests, and CP-coverage
checks regardless of the verdict. If any claim fails, the overall result is
`NOT_MET`; no post-hoc estimator or channel changes may be used to reinterpret
this confirmation. A pass supports only this TDL-C, speed, SNR, PUSCH, and
receiver configuration. It does not establish measured-channel validity,
general receiver performance, G5.09 closure, or physical RT qualification.

## Reproduction

Run `uv run --no-sync --locked python experiments/qualify_g5_tdl_c_dmrs_density.py`.
The runner writes
`docs/research/results/g5-tdl-c-dmrs-density-confirm-20261024.json` and refuses
to overwrite an existing result.
