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

The diffuse component uses normalized ETSI cluster powers and the Cartesian product of each cluster's 20 azimuth and 20 zenith offsets. The receive polarization displacement angle is calculated from the declared UT Euler orientation and each arrival direction using TR 38.901 Eq. 7.1-15. Vertical/horizontal receive field powers use the corresponding co-/cross-polar terms with the table XPR. The local-y array axis is independently rotated from the same Euler angles before applying the half-wavelength spatial phase. Independent diffuse cross-polar phases yield zero V/H ensemble covariance.

The LOS component uses the independently transcribed LOS angles and K factor. Its deterministic receive vector contains the orientation-derived V/H field components and the rotated-array steering phase. The reference combines this rank-one term at `K/(K+1)` with the diffuse covariance at `1/(K+1)`.

## Limits

These are three discrete orientations and one array configuration. They do not establish a continuous orientation distribution, directional-pattern behavior, mutual coupling, rotated transmit arrays, temporal evolution, geometric RT truth, measured-channel accuracy, or overall G4 qualification. The expanded regression definition is authored but has not been executed. A pass cannot authorize production RT generation.

## Deferred validation commands

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
