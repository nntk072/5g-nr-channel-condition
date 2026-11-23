# Research plan topic: data generation

This note is built from the archived master research plan at git tag
`docs/master-research-plan-archive`, file `docs/research/MASTER_RESEARCH_PLAN.md`. Each section below is copied
verbatim; the source line range is given before it. The archived plan is
not restored to the tree. Rebuild this note with
`python scripts/build_research_topics.py`.

<!-- archive docs/research/MASTER_RESEARCH_PLAN.md lines 1272-1419 -->
## 5. Data program — physically meaningful, stratified and auditable

### 5.1 Data tiers (do not treat all as the same benchmark)

| Tier | Data | Purpose | Required audits | Never claim |
|---|---|---|---|---|
| D0 | analytically solvable SISO/MIMO free space, two rays, known Doppler | mathematical unit tests, oracle validation | exact Fourier sum, path delay, phase, aperture, power | realistic field performance |
| D1 | Sionna PHY TDL/CDL A–E (`spec_version="19.2"`) | standardized controls, profile recognition and source pretraining | balanced profile+delay/SNR, antenna, speed, profile masks; `datasets.standardized` stores immutable oracle, noisy-CFR observation, and profile-proxy target layers separately | geometric visibility truth independent of model design |
| D2 | scenario models UMi/UMa/RMa/InH/InF with actual LOS-state logic and spatial consistency | bridge from fixed-profile families to scenario distributions | scenario labels, geometry/statistical consistency, correlation lengths, blockers | equivalent to true detailed RT |
| D3 | controlled RT canonical scenes | demonstrate causal mechanisms and sensor resolution | geometry/oracle cross-check, path solver sweeps, direct+multiple indirect paths | complete physical city fidelity |
| D4 | independent high-fidelity urban/indoor/factory/campus RT scenes | primary held-out-scene generalization | material uncertainty, mesh QA, representative scene diversity | validated against the world without measurements |
| D5 | independently recorded, permission-cleared NR CSI or I/Q with surveyed ground truth | external transfer and simulator-bias audit | hardware calibration, clock/LOS labels, ethics/permissions | absolute LOS labels from subjective RSS alone |

The 3D scenes currently checked into the repository (`city41`, `city42`, `city43`) are useful *deterministic controlled fixtures*. They are **not** enough for strong external-validity claims: same procedural rule family and simple box meshes make held-out seeds much easier than an entirely independent city/asset pipeline.

### 5.2 Scene families and environments to build

**Controlled foundation:** free-space link; flat reflective ground; single wall, corner/wedge, corridor, hall, office partitions, two-wall transmission, moving metal screen, narrow street canyon, foliage proxy, rich indoor room, factory machinery line, parking vehicles, open plaza.

**External test families:** two or more **independent asset/geometry pipelines** (e.g., a measured/surveyed or authored building, independent procedural generator, Sionna built-ins such as Munich/Étoile). Across each, vary:

- geometry topology: streets/corners, buildings, material thicknesses, multilevel stories, vehicle density, indoor rooms, factory racks, irregular surfaces;
- materials: concrete, brick, glass, metal, wood, plasterboard, composite; explicit thickness/permittivity/conductivity/scattering parameters; frequency-valid ranges;
- EM mechanisms: direct, one-/multi-bounce specular, diffuse, refraction, diffraction; interaction mixtures and near-occlusion;
- UE and BS height, carrier, polarization, array geometry, element patterns, orientation, FoV, placement and steering;
- propagation distance, antenna gains, transmit power, thermal noise, interference and receiver sensitivity;
- receiver mobility and independent blocker/vehicle trajectories; include trajectories with clear LOS → partial shadow → blocked LOS → reappearance;
- a **no-detected-path / outage** category rather than dropping links.

**Important control:** Sionna’s built-in ITU material models can have `scattering_coefficient=0` by default. Scene realism depends on **material parameters**, not just turning the diffuse flag on. Log the enabled mechanism and actual fraction of paths/power it contributes.

### 5.3 Geometry and physical-consistency validation

For every scene version, generate an immutable `scene_manifest.json` with source/geometry license, XML/OBJ checksum, coordinate frame and units, vertices/faces/bounding boxes, transmitters/arrays, receivers, material assignments/thickness, frequency validity, scattering coefficients, and test outputs. Validate:

1. **Watertight/nonwatertight mesh policy** documented; normal orientation and face winding; no self-intersection that breaks physics unexpectedly.
2. TX/RX physically outside solid obstacles; adjustable clearance, no placements underneath ground/floors or inside vehicles.
3. TX/RX height and orientation in physically plausible ranges; min/max link distance; no overlapping antennas.
4. Direct-segment collision/ray oracle agreement on hand-reviewed LOS/NLOS examples; include wall grazing/epsilon tests.
5. Cross-process deterministic replay of canonical scene samples using identical package lock/precision/backends.
6. Path counts, path delay ordering, delay vs total traveled length, path power vs range, per-interaction oracle labels, diffraction/corner cases.
7. Coherent and incoherent power conventions: `|Σ a_p e^{-j2πfτ_p}|²` can differ from `Σ|a_p|²`; use correct convention for the target.
8. Material changes produce expected **trends** rather than assumed exact values (e.g., metal reflectivity, thickness and transmission); compare ray-level behavior with analytical Fresnel/free-space limiting cases.
9. Convergence with `samples_per_src`, `max_depth`, deterministic mode and raster/mesh precision. Report unresolved links and solver failures separately.
10. External visual QA: save rendered images from multiple angles of every final evaluation scene, ray overlays for a stratified set and screenshots of blockage events.

### 5.4 Link and trajectory generation protocol

**Static data:** sample geographic/geometry positions *before* looking at the label distribution; record full acceptance statistics. Do not make the training LOS ratio a hidden artifact of rejection sampling. For balanced diagnostic subsets, keep separate raw prevalence, balanced indices and inverse-probability weights.

**Dynamic data:** generate physically meaningful trajectories `p_UE(t)` and blocker motion `p_b(t)` at a declared cadence, velocity/acceleration/turn constraints, and time-stamped geometry hash. For transitions, re-trace at interval boundaries or every snapshot according to validated temporal resolution. Preserve entire sequences in one split; no overlap, shifted copies or mirrored scenes that leak across splits.

**Spatial exclusion:** assign super-groups by scene/asset family, sub-groups by street/block, link/trajectory, material seed, and BS site; hold out complete scenes, use a buffer in meters or a conservatively chosen estimated correlation distance across train/test boundaries, and perform a **nearest-neighbor geographic leakage audit**. One random `group_split()` is not a substitute.

**Obstruction labels:** A moving occluder can modify reflections as well as the direct ray. For a real counterfactual, re-run the **entire scene** with blocker inserted/removed and retain identifiers linking the two interventions; do not simply subtract the direct coefficient and call the resulting channel physically obstructed.

### 5.5 Minimum benchmark matrix: design examples, NOT final fixed sample counts

Construct the initial factorial design using levels below, then refine with compute/power pilot *without consulting the locked test result*. Numbers listed are proposal levels, not empirically optimal ranges:

| Axis | Example levels | Critical control |
|---|---|---|
| Carrier | 3.5 GHz (legacy anchor); 6 GHz; 28 GHz if assets/materials qualified | material/antenna availability; don't pretend one material is valid at every frequency |
| NR subcarrier spacing | 15 / 30 / 60 kHz only where selected numerology valid | same occupied bandwidth across models when comparing frequency effects |
| Occupied bandwidth | 2.88 MHz legacy active allocation; wider practical cases selected with explicit PRBs/FFT | pilot count and resolvability changes confound gains |
| Receiver noise | relative-SNR diagnostics and separate thermal-noise/link-budget simulation | record exact definition: SNR, Eb/N0, SINR are different |
| Mobility | static; pedestrian; vehicle; dynamic blocker | dynamic trajectories/cadence actually recomputed |
| Arrays | SISO; 2×4 or comparable calibrated MIMO; larger array *if justified* | preserve aperture, array/polarization geometry and source directions |
| Scenarios | LOS-direct-only; LOS-mixed; weak-direct LOS; pure NLOS-reflection; NLOS-transmission; NLOS-diffraction; outage | never map scenario mix to a single binary class without a label schema |
| Pilot density | validated NR DMRS patterns; controlled missing-RE stress subset | equal measurement budget in model comparison |
| Physical corruption | AWGN, CFO, timing offset, phase noise, quantization, channel aging, interference | single-factor and interaction tests; independent seeds |
| Scene types | urban, indoor, industrial, controlled analytical | separate topology family in at least one final held-out set |
| Train data | D1 only; D2 only; D3/D4 only; mixed D1–D4 | report exact domain mixing and labels |

**Design/cost rule:** the *full Cartesian product is computationally infeasible*. Use staged screening (orthogonal or Latin-hypercube sampling) followed by a smaller preregistered stress factorial. Publish the sampling policy and condition counts. Preserve at least several independent scenes per external comparison; no confidence interval based on a single scene can justify cross-scene generalization.

### 5.6 Dataset storage schema (must be testable)

Proposed canonical source sample dimensions:

```yaml
sample:
  id: immutable_unique_id
  source_type: {analytical, tdl, cdl, tr38901_scenario, rt, measurement}
  scene_id: immutable_scene_or_session_id
  asset_family_id: independent_scene_family
  trajectory_id: global_trajectory_or_link_id
  timestamp_s: float64_or_null
  tx_rx_geometry: oracle_private_metadata
  receiver_observation:
    complex_pilot_IQ: [time, rx_ant, tx_stream, ofdm_symbol, pilot_re]  # if available
    H_hat_complex: [time, rx_ant, tx_stream, ofdm_symbol, used_subcarrier]
    observed_re_mask: same_reference_grid
    H_hat_error_variance: same_reference_grid_or_broadcastable
    receiver_known_config: {carrier_hz, scs_hz, fft_size, cp, prbs, dmrs, array_geometry, calibrated_noise}
  simulation_privileged:
    H_true: ideal_complex_reference
    paths: {valid_mask, delays_s, amplitudes, angles_rad, doppler_hz, interaction_types}
    geometry: {los, blocker_objects, positions_m, materials}
  target_values:
    geometric_los: value_or_unknown
    observable_direct: value_or_indeterminate
    delay_spread_s: float_or_null
    path_set: list_or_empty
    future_H: array_or_null
    beam_labels: list_or_null
    uncertainty_flags: explicit_mask_per_target
  provenance:
    seed: integer
    code_commit: sha
    artifact_hashes: hashes
    api_lock: lock_id
    label_definition_version: semantic_version
    licensing: dataset_and_scene_terms
```

This is a **schema design**, not currently implemented serialization. Use Zarr/HDF5/Parquet + NumPy complex buffers as appropriate; confirm inter-process safety/immutability and float precision. No implicit flattening, no untyped `metadata_json` as the only semantics registry, no conversion of seconds to ns without explicit metadata. Raw/processed dataset manifests should have independent hashes; split indices saved as immutable artifacts.

### 5.7 Truth labels, soft labels and ambiguity policy

For each target create:

- `oracle_definition.md`: exact formula and whether it comes from geometry, path solver, receiver estimator or lab instrument.
- `label_scope`: per-path / per-antenna / per-link / per-trajectory; array-wide definitions cannot silently replace element-wise ones.
- `validity_mask`: estimated is possible / oracle unknown / censored / no coverage / insufficient information.
- `tolerance`: numerical/physical ambiguity thresholds determined from independent analytical checks, not optimized for model wins.
- `quality_flag`: solver convergence and real measurement calibration status.
- `label_version`: update triggers a new dataset and a new paper experiment ID.

For a path that is physically present but substantially below the noise floor, `geometric_los=True` and `observable_direct=indeterminate/false` may **both be correct**. This case is central to the proposed identifiability-aware mechanism; do not discard it as a contradiction.

### 5.8 Real measurements: highest-value gate

Prepare a separate `REAL_MEASUREMENT_PROTOCOL.md` with:

1. **Data permission:** own collection or an appropriately licensed public dataset, respecting any university/company/Nokia restrictions; user ownership of source code does not automatically grant rights to third-party measured data, scene assets, or proprietary device firmware.
2. **Hardware:** transmitter/receiver models, antenna arrays, RF front-end, clock/time synchronization, GPS/survey quality, position/orientation, frequency reference, trigger times, IQ dynamic range, AGC and calibration logs.
3. **Capture:** NR pilots or known probe waveform plus true/estimated channel, transmit power, noise floor, interference background, SNR/SINR estimate, RF impairments, polarization.
4. **Truth:** surveyed visibility and blocker video/map/manual inspection independent of CSI; timestamp alignment; LOS boundary/partial obstruction/penetration annotations; blind second audit on ambiguous cases.
5. **Sites/sessions:** distinct buildings/streets/days/hardware setups when possible; at least one held-out site untouched by tuning. Never mix recordings from one walk into both train and test.
6. **Privacy/security:** avoid releasing traceable personally identifying trajectories/metadata without approval; document anonymization and geographic redaction that do not corrupt labels.
7. **Transfer:** source-only zero-shot test first; if allowed, separate **unsupervised target calibration** and **few-shot labeled adaptation** with clear labels and separate evaluation splits.
8. **External reproducibility:** if raw data cannot be shared, publish de-identified aggregate evaluations, provenance hashes, data statement, access process and appropriately sanitized scripts.

**No real dataset = permissible simulation-heavy work, but change claim wording.** Never call Sionna-only validation “real-world robustness.”

---
