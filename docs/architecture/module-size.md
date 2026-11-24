# Module size

Product modules stay below about 1,000 lines. Eleven modules are above that
line. Each one is listed here with the reason it has not yet been split. The
split is tracked as follow-up work in the reading plan.

| Module | Lines | Reason it stays whole for now |
| --- | ---: | --- |
| `evaluation/acquisition/verify.py` | 1397 | One fail-closed verification sequence; splitting would separate checks whose order is part of the contract. |
| `integrations/sionna/srs/srs_configuration.py` | 1391 | SRS configuration tables and their validators share constants; a split needs the table owners agreed. |
| `integrations/sionna/rt_trajectory.py` | 1327 | Single trajectory pipeline from solve output to observations. |
| `integrations/sionna/rt_network.py` | 1288 | Single mapping from whole RT solve outputs to episode frames. |
| `integrations/sionna/srs_measurements.py` | 1208 | Measurement primitives and their strict report contracts are versioned together. |
| `experiments/runner/experiment_runner.py` | 1171 | Run entry point; its steps are already separated into `experiments/runner/`. |
| `integrations/sionna/network_phy/operations.py` | 1169 | Link budget and waveform composition operate on the same record types. |
| `integrations/sionna/srs/srs_lookups.py` | 1161 | Lookup tables derived from the SRS constants. |
| `integrations/sionna/nr_prs.py` | 1117 | PRS sequence, mapping and timing for one reference signal. |
| `datasets/standardized.py` | 1072 | Standardized record types and their admission checks share one set of invariants. |
| `evaluation/acceptance/path_equivalence.py` | 1052 | One equivalence contract and its roster checks. |
| `datasets/episode_store/storage.py` | 1016 | Immutable storage primitives used by every episode reader. |
