# G4.07 CDL Single-Angle Atom Roster Check

**Status:** Deterministic native implementation regression; not a statistical
acceptance test. The earlier Fourier-confidence design incorrectly treated
repeated CDL ray layouts as independent Monte Carlo draws. Sionna randomizes
within-cluster angle ordering, but each row retains the same table-defined
marginal atoms and powers. A zero-width interval from those rows is not valid
sampling uncertainty, so the stochastic acceptance interpretation is retired.

## Check

- Profiles: CDL-A, CDL-B, CDL-C, CDL-D, and CDL-E under Sionna PHY 2.2.0.
- Source: package tables are SHA-256 pinned by the native test and compared to
  the independent v19.2.0 ETSI transcription in
  `G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`.
- The deterministic reference uses float32 operations matching the pinned
  implementation, excludes only the specular LOS component, and sorts diffuse
  clusters by their declared delays.
- For eight seeded rows/profile, the test sorts each AoA/AoD/ZoA/ZoD ray-angle
  marginal within each cluster and requires exact equality with the 20
  table-derived atoms. It also compares each native cluster power with the
  table-derived normalized power (absolute tolerance `1e-7`). The seed controls
  repeatable random permutations; it is not an independent-sample claim.
- Joint angle-coupling checks remain in their separate Fourier-family tests.

The native cases are marked `native_sionna`. The seed values are covered by
`G4_SEED_ROSTER_AUDIT_2026-10-26_ALL_FROZEN.json`; the deterministic check does
not consume a stochastic acceptance roster. A pass verifies only these
table-defined marginal ray atoms and powers for the pinned implementation. It
does not establish joint angular coupling, temporal behavior, arbitrary array
orientation/pattern behavior, geometric RT validity, measured-channel
accuracy, or G4 qualification.
