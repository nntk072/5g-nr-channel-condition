# integrations

**Purpose.** Adapters between the project's NR contracts and native Sionna
PHY and Sionna RT. Everything native is isolated here; the rest of the
repository imports nothing from Sionna.

**Public API.**

- `integrations.sionna` — PHY and RT adapters. The main groups are:
  - PUSCH, PBCH, SSB, PRS and SRS waveform and receiver adapters
    (`nr_pusch`, `pbch_*`, `ssb_*`, `nr_prs`, `srs/`, `srs_measurements`);
  - standardized TDL/CDL channel adapters (`tdl`, `cdl`, `tdl_pusch`,
    `standardized_pusch`, `scenario_gbsm*`);
  - Sionna RT scene, solve and path conversion (`scene_assets`, `backend`,
    `rt_identity`, `rt_network`, `rt_to_nr`, `path_oracle`, `rt_cache`,
    `rt_trajectory`, `los_oracle`);
  - multi-node network PHY (`network_phy/`, `network_receiver`,
    `network_sinr`).
- `integrations.runtime_compatibility` — opt-in checks of the Sionna PHY and RT
  APIs used by the project.
- `integrations.runtime_inventory` — exact, read-only inventory of software,
  accelerator, and model tables used by runs (`loslab-runtime-inventory`).
- `integrations.sionna_runtime` — runtime identity for Sionna's split PHY and RT
  distributions.

**Invariants.**

- Native PHY behaviour needs evidence on the host where it runs. Mock and
  analytical fixtures show software behaviour only.
- RT backends (Mitsuba, Dr.Jit) are selected explicitly. A missing backend is an
  error, never a silent fallback to a different solver.
- Adapters preserve Sionna's conventions (delay reference, carrier alignment,
  antenna ports, polarization) and name every conversion in the output contract.
- Imports of native packages are lazy, so the package imports without them.
