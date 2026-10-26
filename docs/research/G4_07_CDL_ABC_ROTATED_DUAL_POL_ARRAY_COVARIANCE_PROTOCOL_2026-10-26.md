# G4.07 CDL-A/B/C rotated dual-polarized array covariance protocol

This bounded regression closes one remaining code-definition gap between the
zero-orientation NLOS covariance family and the rotated LOS CDL-D/E family. It
compares the native Sionna CDL-A/B/C channel output against an independently
constructed expectation from the byte-pinned ETSI TR 38.901 V19.2.0 component
fixture. It is a standardized-model consistency check, not physical-channel
qualification.

- Profiles: NLOS CDL-A, CDL-B, and CDL-C; 3.5 GHz carrier; 30 kHz spacing;
  eight subcarriers; 100 ns delay spread.
- Array: two half-wavelength UT elements on local y, dual-VH omni elements;
  one single-V omni BS port. The UT Euler orientations are
  `(0.37, 0.28, -0.41)`, `(0, 0, 0.65)`, and `(0.28, -0.36, 0)` radians.
- Draws: 8,192 independent stochastic channel draws per profile and
  orientation. Seeds are `1720100` through `1720108`, assigned by orientation
  then profile. Before execution, run
  `python scripts/audit_seed_roster_collisions.py --root <saved-results-root>
  --seed-range 1720100 1720109` against the declared result inventory and
  retain its structured output with the run evidence. Report v2 marks skipped
  oversized files as `INCOMPLETE_SCAN` and the CLI exits 2; such a report is
  not seed-collision clearance. The current repository-wide research scan is
  incomplete: 263 files were checked with zero malformed fields and zero
  collisions for this roster, but one oversized compressed artifact remains
  unscanned.
- Reference: diffuse powers, AoA/ZoA, cluster spreads, and XPR come from
  `G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`, whose SHA-256 is
  `24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090`; the
  source PDF digest is `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.
  The Sionna packaged-table digests identify the system-under-test only.
- Observable: the 4-by-4 covariance of the cluster-summed narrowband vector,
  in V-element0, V-element1, H-element0, H-element1 order. The independent
  expectation applies the TR 38.901 receive polarization-basis rotation and
  rotates the local-y array axis from the declared UT Euler angles. Independent
  cross-polar phases imply zero V/H ensemble covariance for these NLOS cases.
- Inference: two-sided Bonferroni 99% intervals cover 16 real Hermitian
  components × three profiles × three orientations (144 claims). Every
  interval must contain the independent reference and have half-width no
  greater than 0.05 normalized mean power.

The test is `tests/test_integrations/test_g4_cdl_abc_rotated_dual_pol_array_covariance_native.py`.
The frozen orientation set does not establish continuous-orientation behavior,
mutual coupling, arbitrary array geometries, directional element patterns,
temporal evolution, geometric RT truth, or overall G4 qualification. Do not
interpret 8,192 draws as ground truth or use this finite family to authorize
production RT generation. The regression and seed-collision audit are authored
but have not been executed in this implementation pass.
