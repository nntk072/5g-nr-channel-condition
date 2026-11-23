# G4.07 CDL-D/E Rotated LOS Dual-Polarized Array Covariance Protocol

## Frozen scope

- Profiles: LOS CDL-D and CDL-E, TR 38.901 v19.2.
- Topology: one vertical single-polarized TX port and a two-element, half-wavelength UT ULA with vertical/horizontal ports at each element.
- UT orientations: Euler angles `(0.37, 0.28, -0.41)`, `(0, 0, 0.65)`, and `(0.28, -0.36, 0)` rad; BS orientation is zero in every case. The roster includes mixed-axis, pure-yaw, and roll/pitch rotations. Omni element patterns isolate orientation-dependent polarization-basis and array-axis rotation.
- Observable: the zero-frequency, cluster-summed 4-by-4 receive-channel second-moment matrix, with ports ordered V-element0, V-element1, H-element0, H-element1.
- Independent inputs: ETSI-derived component powers, angles, angular spreads, and XPR from `G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`; LOS K factor from `G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json`; Table 7.5-3 ray offsets. Both reference fixture digests and the pinned ETSI source digest are checked. The Sionna CDL table hash is implementation provenance only.
- Seed roster: orientation 0 uses `1720000`/`1720001` for D/E, orientation 1 uses `1720002`/`1720003`, and orientation 2 uses `1720004`/`1720005`. Run the structured G4 seed-collision audit before execution.
- Sampling unit: one independently generated stochastic channel draw; 16,384 draws/profile.
- Inference: a two-sided Bonferroni 99% family over 16 unique real Hermitian components × two profiles × three orientations (96 claims). Every interval must contain its independent reference and have half-width at most 0.05 normalized mean power.

## Reference construction

The diffuse component uses normalized ETSI cluster powers and the Cartesian product of each cluster's 20 azimuth and 20 zenith offsets. The receive polarization displacement angle is calculated from the declared UT Euler orientation and each arrival direction using TR 38.901 Eq. 7.1-15. Vertical/horizontal receive field powers use the corresponding co-/cross-polar terms with the table XPR. The local-y array axis is independently rotated from the same Euler angles before applying the half-wavelength spatial phase. After the receive basis is rotated, the same independent co-polar and cross-polar phasors contribute to both V and H components. The independent oracle therefore retains the resulting V/H covariance term `cos(psi) sin(psi) (1/XPR - 1)` and its corresponding inter-element spatial phases.

The LOS component uses the independently transcribed LOS angles and K factor. For the declared single-V transmitter, the rotated receive basis gives V amplitude `cos(psi)` and H amplitude `-sin(psi)` before the common rotated-array steering phase. A deterministic regression compares this rank-one vector against Sionna Step 11 for both profiles and all three orientations. The reference combines the LOS term at `K/(K+1)` with the diffuse covariance at `1/(K+1)`.

## Limits

These are three discrete orientations and one array configuration. They do not establish a continuous orientation distribution, directional-pattern behavior, mutual coupling, rotated transmit arrays, temporal evolution, geometric RT truth, measured-channel accuracy, or overall G4 qualification. The bounded family was executed on 2026-11-09 after correcting the independent oracle conventions; the frozen statistical criteria and sample counts were unchanged. A pass cannot authorize production RT generation.

## Reproduction commands

After the implementation pass and Halti environment setup, run the seed audit first:

```bash
uv run --locked --no-sync python scripts/audit_seed_roster_collisions.py \
  --root docs/research/results --root outputs --root artifacts \
  --seed-range 1720000 1720006
```

Proceed only if the report is `CLEAR_WITHIN_SCANNED_JSON_CORPUS`; that status is scoped to the scanned local files and is not a global seed reservation. Then run this criterion-defined native regression:

```bash
uv run --locked --no-sync python -m pytest -q -o addopts= \
  tests/test_integrations/test_g4_cdl_de_rotated_los_dual_pol_array_covariance_native.py
```


## 2026-11-09 retained validation

The fresh structured seed audit over `docs/research/results`, `outputs`, and `artifacts` scanned 2,277 eligible JSON/JSON.GZ files and returned `CLEAR_WITHIN_SCANNED_JSON_CORPUS` for seeds `1720000`–`1720005`, with zero collisions, malformed seed fields, or incomplete files. The audit does not establish global seed exclusivity.

The native file passes 12/12 tests: six deterministic Sionna Step-11 LOS-vector checks plus the six frozen 16,384-draw profile/orientation covariance cases. The zero-orientation CDL-D/E covariance control separately passes 2/2. No confidence threshold, family size, seed roster, or draw count was relaxed. This closes only the declared three-orientation CDL-D/E covariance family; G4 remains open.
