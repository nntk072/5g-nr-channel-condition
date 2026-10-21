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
