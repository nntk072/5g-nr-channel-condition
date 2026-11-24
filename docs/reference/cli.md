# Command reference

Installed console commands, from `[project.scripts]` in `pyproject.toml`.

| Command | Module | Purpose |
| --- | --- | --- |
| `loslab-audit-channel-population` | `scripts.audit_channel_population` |  |
| `loslab-audit-odbl-scene-source` | `scripts.audit_odbl_scene_source_snapshot` |  |
| `loslab-bind-split-registry-children` | `scripts.bind_split_registry_children` | Bind frozen child rosters to a candidate split registry without changing parents. |
| `loslab-build-result-promotion-manifest` | `scripts.build_result_promotion_manifest` |  |
| `loslab-build-standardized-split-registry` | `scripts.build_standardized_split_registry` | Build a candidate release split lock from immutable standardized-channel runs. |
| `loslab-export-public-release` | `scripts.export_public_release` | Export a hash-checked public release after external signature verification. |
| `loslab-g3-joint-evidence` | `evaluation.joint_evidence_artifact` | File-bound execution and replay for the five-branch G3 numeric bundle. |
| `loslab-materialize-benchmark-partitions` | `scripts.materialize_benchmark_partitions` | Materialize a signed, non-copying six-cohort benchmark index from ChannelDatasets. |
| `loslab-materialize-direct-path-ablation` | `scripts.materialize_direct_path_ablation` | Materialize a mathematical direct-path ablation from an existing UMCD bundle. |
| `loslab-pair-stored-frequency-views` | `scripts.pair_stored_frequency_views` | Pair existing independently solved channel bundles under a frozen split lock. |
| `loslab-pilot` | `experiments.pilot_cli` | Command-line interface for deterministic pilot sampling and result audits. |
| `loslab-report-standardized-capabilities` | `scripts.report_standardized_capabilities` | Publish schema-validated, deterministic standardized generator scope reports. |
| `loslab-runtime-inventory` | `integrations.runtime_inventory` | Exact, read-only inventory of the software and model tables used by runs. |
| `loslab-verify-g3-run-report-signature` | `scripts.verify_run_report_signature` |  |

## Script commands

Scripts in `scripts/` run as `python scripts/<name>.py --help`.

| Script | Purpose |
| --- | --- |
| `scripts/bind_split_registry_children.py` | Bind frozen child rosters to a candidate split registry without changing parents. |
| `scripts/build_channel_split_registry.py` | Build an immutable candidate split registry from saved ChannelDatasets. |
| `scripts/build_rt_dmrs_observation_dataset.py` | Convert saved RT CFR datasets to PUSCH/DMRS receiver observations. |
| `scripts/build_standardized_pusch_dataset.py` | Build receiver-visible PUSCH/DMRS records from Sionna TDL or CDL draws. |
| `scripts/build_standardized_split_registry.py` | Build a candidate release split lock from immutable standardized-channel runs. |
| `scripts/check_rt_backend.py` | Probe installed Mitsuba variants with a minimal native Sionna RT trace. |
| `scripts/export_public_release.py` | Export a hash-checked public release after external signature verification. |
| `scripts/generate_rt_dataset.py` | Generate geometric LOS/NLOS RT datasets from full multi-path ray tracing. |
| `scripts/generate_rt_dmrs_observations.py` | Apply a Sionna PUSCH/DMRS receiver to saved RT channel-frequency-response datasets. |
| `scripts/generate_rt_nr_trajectory.py` | Generate one timestamped, hash-addressed Sionna RT-to-NR trajectory. |
| `scripts/materialize_benchmark_partitions.py` | Materialize a signed, non-copying six-cohort benchmark index from ChannelDatasets. |
| `scripts/materialize_direct_path_ablation.py` | Materialize a mathematical direct-path ablation from an existing UMCD bundle. |
| `scripts/pair_stored_frequency_views.py` | Pair existing independently solved channel bundles under a frozen split lock. |
| `scripts/report_rt_backend_capabilities.py` | Report installed and probe-verified Sionna RT backend capabilities. |
| `scripts/report_runtime_inventory.py` | Write a local software, accelerator, and Sionna model-table inventory. |
| `scripts/report_standardized_capabilities.py` | Publish schema-validated, deterministic standardized generator scope reports. |
| `scripts/run_channel_experiment.py` | Evaluate channel-condition baselines on saved datasets from a versioned configuration. |
| `scripts/run_profile_proxy_dataset.py` | Generate a provenance-recorded TDL/CDL profile-proxy dataset. |
| `scripts/sample_scene_families.py` | Create an immutable, deterministic S00–S11 scene-candidate sample report. |
