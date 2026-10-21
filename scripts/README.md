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
