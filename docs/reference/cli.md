# Command reference

This page lists every command that the project installs or ships. Each entry
was verified to import, so every listed name runs.

## Installed console commands

From `[project.scripts]` in `pyproject.toml`:

| Command | Module | Purpose |
| --- | --- | --- |
| `loslab-pilot` | `experiments.pilot_cli` | Deterministic pilot sampling and result audits. |
| `loslab-runtime-inventory` | `integrations.runtime_inventory` | Exact, read-only inventory of software and model tables used by runs. |
| `loslab-report-standardized-capabilities` | `scripts.report_standardized_capabilities` | Schema-validated, deterministic standardized generator scope reports. |
| `loslab-build-standardized-split-registry` | `scripts.build_standardized_split_registry` | Candidate split lock from immutable standardized-channel runs. |
| `loslab-bind-split-registry-children` | `scripts.bind_split_registry_children` | Bind frozen child rosters to a candidate split registry without changing parents. |
| `loslab-pair-stored-frequency-views` | `scripts.pair_stored_frequency_views` | Pair independently solved channel bundles under a frozen split lock. |
| `loslab-materialize-direct-path-ablation` | `scripts.materialize_direct_path_ablation` | Mathematical direct-path ablation from an existing UMCD bundle. |
| `loslab-materialize-benchmark-partitions` | `scripts.materialize_benchmark_partitions` | Signed, non-copying benchmark index from ChannelDatasets. |
| `loslab-joint-evidence-artifact` | `evaluation.joint_evidence_artifact` | File-bound execution and replay of the joint numerical evidence bundle. |
| `loslab-export-public-release` | `scripts.export_public_release` | Hash-checked public release after external signature verification. |

## Scripts

Scripts run with `python -m scripts.<name> --help`. They are grouped by workflow in
[`scripts/README.md`](../../scripts/README.md).

| Script | Purpose |
| --- | --- |
| `bind_split_registry_children` | Console command above. |
| `build_channel_split_registry` | Immutable candidate split registry from saved ChannelDatasets. |
| `build_rt_dmrs_observation_dataset` | Convert saved RT CFR datasets to PUSCH/DMRS receiver observations. |
| `build_standardized_pusch_dataset` | Receiver-visible PUSCH/DMRS records from Sionna TDL or CDL draws. |
| `build_standardized_split_registry` | Console command above. |
| `check_rt_backend` | Probe installed Mitsuba variants with a minimal Sionna RT trace. |
| `export_public_release` | Console command above. |
| `generate_rt_dataset` | Geometric line-of-sight and NLOS RT datasets from full multipath ray tracing. |
| `generate_rt_dmrs_observations` | Apply a Sionna PUSCH/DMRS receiver to saved RT channel-frequency-response datasets. |
| `generate_rt_nr_trajectory` | One timestamped, hash-addressed Sionna RT-to-NR trajectory. |
| `materialize_benchmark_partitions` | Console command above. |
| `materialize_direct_path_ablation` | Console command above. |
| `pair_stored_frequency_views` | Console command above. |
| `report_rt_backend_capabilities` | Installed and probe-verified Sionna RT backend capabilities. |
| `report_runtime_inventory` | Local software, accelerator, and Sionna model-table inventory. |
| `report_standardized_capabilities` | Console command above. |
| `run_channel_experiment` | Channel-condition baselines on saved datasets from a versioned configuration. |
| `run_profile_proxy_dataset` | Provenance-recorded TDL/CDL profile-proxy dataset. |
| `sample_scene_families` | Immutable, deterministic scene-candidate sample report. |

Native commands (`check_rt_backend`, `generate_rt_*`, `build_rt_dmrs_*`,
`generate_rt_nr_trajectory`) need the Sionna RT extras and a host with Mitsuba.
They fail with an explicit message otherwise.
