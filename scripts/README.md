# scripts

**Purpose.** Thin command-line entry points for the NR data-generation,
registry, audit and reporting workflows. Reusable logic belongs in the importable
packages; a script only parses arguments, calls a package function, and prints a
summary.

**Public API.** Each script is run as `python -m scripts.<name> --help` (or
`python scripts/<name>.py --help`). The installed console commands are listed in
[`docs/reference/cli.md`](../docs/reference/cli.md), which is the command
reference.

Scripts are grouped by workflow:

- Dataset generation: `generate_rt_dataset`, `generate_rt_nr_trajectory`,
  `build_standardized_pusch_dataset`, `run_profile_proxy_dataset`.
- Receiver observations: `build_rt_dmrs_observation_dataset`,
  `generate_rt_dmrs_observations`.
- Split registries and bindings: `build_channel_split_registry`,
  `build_standardized_split_registry`, `bind_split_registry_children`,
  `pair_stored_frequency_views`, `materialize_benchmark_partitions`,
  `materialize_direct_path_ablation`.
- Release and reporting: `export_public_release`, `report_runtime_inventory`,
  `report_standardized_capabilities`, `report_rt_backend_capabilities`,
  `check_rt_backend`, `sample_scene_families`.
- Experiments: `run_channel_experiment`.

**Invariants.**

- Every script refuses to overwrite an existing output. Outputs are written
  atomically and carry their hashes and configuration.
- Scripts that need Sionna RT or Mitsuba fail with an explicit message on a host
  without them. They do not fall back to another solver.
- Research campaign drivers (run sequences, host paths, and result
  interpretations) are not part of this package. They are kept outside the
  tracked tree.
