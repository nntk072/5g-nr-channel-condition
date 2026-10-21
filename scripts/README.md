# Command wrappers

Reusable research logic belongs in importable packages. This directory is for
thin data, training, evaluation, scene, and research command wrappers when they
are needed for batch or scheduler entry points. The public local CLI is
currently `python -m loslab` (or the installed `loslab` command).
# Reproducible research runs

Use `run_rt_cross_scene_pilot.sh` to generate and evaluate the small, exploratory
three-scene RT pilot. Run it from a checkout with the pinned Sionna RT 2.2.0
environment active and Mitsuba's LLVM variant available:

```bash
PATH="/home/nguyenl37/.venvs/sionna220-gpu/bin:$PATH" \
  bash scripts/run_rt_cross_scene_pilot.sh outputs/rt-cross-scene-pilot-rerun
```

The script refuses to overwrite an existing output directory. Each NPZ row
contains its scene, OFDM-grid, and solver configuration. CPU RT can differ by
small floating-point amounts between runs, so reproduce labels, groups, and
metrics within numerical tolerance rather than expecting identical file hashes.
The three built-in scenes and 40 links per scene are not a confirmatory or
standards-conformance dataset.

Use `run_tdl_pusch_bler_curve.py` for a seeded Sionna TDL-to-PUSCH BLER curve
on the CUDA PHY environment:

```bash
PATH="/home/nguyenl37/.venvs/sionna220-gpu/bin:$PATH" \
  python scripts/run_tdl_pusch_bler_curve.py \
  outputs/tdl-pusch-bler-run.json \
  --frames 1024 --batch-size 128 --snr-db 0 6 12 18 24 --seed 31000
```

The script refuses to overwrite a result, records codeword counts and Wilson
intervals, and checks that transmitted bits and TDL channel truth match across
SNR points. Its current `OFDMChannel` path models TDL fading with perfect
synchronization and no sample-domain ISI/ICI; it is not a time-domain waveform
receiver-capability or standards-conformance claim.

Use `run_tdl_pusch_time_bler_curve.py` to evaluate sample-domain TDL fading and
the PUSCH waveform receiver, including the sampled channel filter memory:

```bash
PATH="/home/nguyenl37/.venvs/sionna220-gpu/bin:$PATH" \
  python scripts/run_tdl_pusch_time_bler_curve.py \
  outputs/tdl-pusch-time-bler-run.json \
  --frames 1024 --batch-size 64 --snr-db 0 6 12 18 24 --seed 41000 \
  --n-size-bwp 12 --subcarrier-spacing-khz 30
```

The default 12-PRB grid provides a 13-sample shortest cyclic prefix for the
configured 13-sample discrete channel memory. The runner refuses a configuration
whose discrete channel memory exceeds its shortest CP, reuses the same waveform,
TDL realization, and standardized noise seed at each SNR, and records measured
transmit waveform power, per-point Wilson intervals, and hashes. The curve is a
simulation result with ideal synchronization; it does not validate blind timing,
fractional timing/CFO recovery, external propagation measurements, or standards
conformance.

For a controlled mobility comparison, generate control and treatment curves
with the same configuration, seed, and codeword count, changing only
`--speed-mps`. The curve output retains the per-codeword CRC mask. Compare them
with the paired bootstrap and exact McNemar analysis:

```bash
python scripts/compare_tdl_pusch_time_curves.py \
  outputs/tdl-pusch-time-control.json \
  outputs/tdl-pusch-time-treatment.json \
  outputs/tdl-pusch-time-paired-comparison.json
```

The comparator verifies matching bits, batch seeds, noise seeds, and all
configuration fields other than mobility before calculating paired outcomes.
Its uncertainty is conditional on the chosen simulation; it does not replace
independent-scene inference.

To compare additional PUSCH DMRS positions at fixed mobility, use identical
SNRs, frame count, batch size, and seed, then vary only
`--dmrs-additional-position`. The comparator checks that channel-truth hashes,
batch seeds, and noise seeds match; the transmitted transport-block sizes may
differ because extra DMRS symbols consume resources. It therefore reports both
paired BLER and successfully decoded transport-block bits per codeword, so the
pilot-density tradeoff includes payload goodput:

```bash
python scripts/run_tdl_pusch_time_bler_curve.py outputs/dmrs0.json \
  --frames 512 --batch-size 64 --snr-db 12 18 24 --seed 42100 \
  --speed-mps 30 --dmrs-additional-position 0
python scripts/run_tdl_pusch_time_bler_curve.py outputs/dmrs3.json \
  --frames 512 --batch-size 64 --snr-db 12 18 24 --seed 42100 \
  --speed-mps 30 --dmrs-additional-position 3
python scripts/compare_tdl_pusch_time_curves.py outputs/dmrs0.json \
  outputs/dmrs3.json outputs/dmrs-paired.json \
  --varying-config-field dmrs_additional_position
```

This remains a single-configuration TDL-A simulation with ideal synchronization
and codeword-conditional uncertainty; it does not establish scene-level or
standards-wide receiver performance.

To isolate multipath delay relative to the cyclic prefix while holding the NR
PUSCH allocation and payload fixed, run the deterministic two-path diagnostic:

```bash
python scripts/run_pusch_cp_margin_experiment.py \
  docs/research/results/pusch-static-cp-margin-sweep-12prb-30khz-512.json \
  --frames 512 --batch-size 64 --seed 82417 --snr-db 30 \
  --n-size-bwp 12 --subcarrier-spacing-khz 30 \
  --in-cp-guard-samples 2 --beyond-cp-samples 3 \
  --additional-echo-delays-samples 1 4 8 13 20
```

The shortest CP is 13 samples. The default reference places the echo at 11
samples; the paired beyond-CP condition places it at 16. Additional requested
delays are evaluated on the same transmitted waveforms and standardized noise,
with paired CRC comparisons against the 11-sample reference. The artifact
records transmit/receive waveform power and per-batch measured receive SNR for
each delay. The path gains are deterministic and the burst starts with zero
channel prehistory, so this is a controlled ISI diagnostic, not a TDL/CDL
distribution or standards-wide receiver result.
