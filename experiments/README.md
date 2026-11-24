# experiments

**Purpose.** Workflows that turn configured inputs into retained result
evidence: the experiment runner, pilot sampling, link-level BLER curves,
acceptance validation for receiver curves, trajectory forecasts, and the pilot
command-line interface.

**Public API.**

- `experiments.runner` — configured experiment entry point
  (`experiment_runner`), feature construction (`features`), validation controls
  and label routing (`validation`), and contract verification (`contracts`).
- `experiments.acceptance` — torch-free validation and scoring of bounded
  acceptance contracts for SNR curves.
- `experiments.link_curves` — seeded Sionna TDL-PUSCH BLER curves (native PHY).
- `experiments.pilot_sampling`, `experiments.pilot_cli` — deterministic pilot
  sampling, outcome coverage reports, and the `loslab-pilot` command.
- `experiments.trajectory_targets`, `experiments.rt_trajectory_forecast` —
  timestamp-aligned forecast targets and receiver-only forecasts for stored RT
  trajectories.
- `experiments.cp_margin`, `experiments.cp_bounded_dft_estimator` — cyclic-prefix
  margin diagnostic and a finite delay-domain DM-RS estimator.

**Invariants.**

- Runs are write-once and refuse to overwrite existing outputs. Completion is
  recorded only by the `artifacts` completion marker.
- Geometric line-of-sight labels and path-oracle truth are targets or
  evidence, never model inputs. The runner's contracts enforce this.
- Results carry their configuration, seeds, and source hashes. A result is
  evidence for the stated configuration only; it is not a qualification of a
  scene, a receiver, or a standard.
- Native PHY workflows (`link_curves`, `cp_margin`) require the optional Sionna
  extras and skip cleanly without them.
