# Historical baseline audit and version migration

## Reviewed materials

- `NguyenLong.pdf` (July 2025 Tampere thesis), including Chapter 6 results,
  the Conclusions, and its original experiment description.
- `nr_pusch/configs_new.py`, `nr_pusch/model_new.py`, `mlp/mlp_model.py`,
  `evaluation.py`, `utils.py`, `ml_algorithms.py`.
- 2025 experiment notebooks (`tdl_cdl.ipynb`, `sionna_siso.ipynb`,
  `sionna_mimo.ipynb`, `mlp_los_nlos.ipynb`, `los_nlos_final.ipynb`).
- `Aalto_MS_Thesis-main` artifact provenance, atomic writes, hashing and
  matched-seed statistical design. No unrelated trading code migrated.

## What the published thesis actually reported

This is an important correction to the user's later recollection:

| LOS/NLOS overall in Table 6.3 | SISO | MIMO |
|---|---:|---:|
| MLP | 75.67% | 77.99% |
| ResNet | 95.15% | 96.93% |
| VGG | 92.66% | 94.44% |

For *10-class channel-profile classification*, Table 6.5 reports other
results; the tasks must not be mixed when arguing whether CNN layers matter.

The new research should determine with an equal-budget **controlled ablation**
whether any signal-processing or learned contribution disappears after removing
a module. This cannot be concluded simply from reported historical accuracy.

## Structural issues that motivate rewrite

1. Many experiment definitions are notebook-only and reuse mutable global state.
2. Older versions of Sionna used TensorFlow; Sionna 2.2 PHY is PyTorch and
   RT uses `PathSolver` (Dr.Jit/Mitsuba). Compatibility must be explicit.
3. Existing `mlp/mlp_model.py` sets GPU env variables *after* importing TF,
   writes a checkpoint every epoch and directly supplies test data as
   training validation, which risks test-selection bias if used for tuning.
4. Profile A/B/C vs D/E labels are profile metadata, not an independent
   geometric obstruction oracle. A random split can reward profile recognition.
5. CNN/MLP/RF trained on differently processed observables cannot establish
   architectural causality. Compare with equal input and train/val/test splits.
6. Existing metadata stores `model_type` and `scenario` alongside `y`; this is
   okay for provenance, but evaluation code must never use metadata as feature.
7. Ray tracing must add direct + indirect paths in LOS, rather than assigning
   separate artificial channel types as the target label.

## Concrete 2.2 migration (not executed natively in current environment)

| Old notebook family | 2026 implementation |
|---|---|
| old `scene.compute_paths(...)` | `PathSolver(...)(scene, max_depth=...)` |
| old `Paths.cir()` logic | `Paths.cfr(out_type='numpy')`, complex ndarray |
| old TF-backed TDL/CDL | PyTorch `sionna.phy.channel.tr38901.TDL/CDL` |
| fixed profile=label | two-tier semantic: profile proxy vs RT geometric oracle |
| one fixed simple RT scene | saved procedural city XML/OBJ + material/seed manifests |
| raw per-sample random split | domain holdout and spatial-block groups |
| one run/accuracy only | matched seeds, AUROC, AUPRC, Brier, ECE and block CIs |
| arbitrary serialized notebook cells | pure modules, tests, artifact hashes |

Sionna native integration must still be tested with **exact installed versions**;
no claim that generating RT channels has succeeded here.
