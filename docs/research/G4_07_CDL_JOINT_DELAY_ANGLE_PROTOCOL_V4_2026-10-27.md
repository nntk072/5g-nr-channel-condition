# G4.07 CDL Joint Delay and Four-Angle Characteristic Protocol v4

**Status:** Preregistered redesign after v3 showed that single-coordinate
characteristics have no sampling variation under Sionna's random-coupling
operation. V1–v3 reports remain immutable, unpooled records and their seeds
will not be reused. The physical reference remains the hash-pinned ETSI TR
38.901 v19.2 profile table.

## Question

For each diffuse CDL-A–E profile, does Sionna 2.2.0 preserve the table-defined
delay and four marginal angle ray rosters, and does its per-draw random
four-angle coupling reproduce selected joint delay-angle characteristics?
CDL-D/E's specular LOS atom remains excluded.

## Frozen design

- Profiles: CDL-A through CDL-E, 100 ns delay spread, 3.5 GHz carrier,
  specification version 19.2.
- Independent CPU coupling draws: 8,192 per profile.
- Seeds: 273065–273069, respectively; disjoint from v1 (270065–270069), v2
  (271065–271069), and v3 (272065–272069). Run and preserve the seed-collision
  audit before any native execution.
- Seed-collision audit: `results/g4-cdl-joint-delay-angle-v4-seed-collision-audit-20261027.json`, SHA-256
  `323038e2b04b11d6d28eaddbb7d7d543447659a30e85c837d01a7e2aebfd0cf7`;
  status `CLEAR_WITHIN_SCANNED_JSON_CORPUS` across 2,268 saved JSON/JSON.GZ
  files, with no collisions, malformed seed fields, or incomplete files.
- Delay harmonics: 1 and 2 of normalized cluster delay.
- Stochastic joint vectors: `(1,1,0,0)`, `(1,-1,1,0)`, and `(1,1,1,1)` over
  `(AoA,AoD,ZoA,ZoD)` in that order. These include at least two independently
  shuffled coordinates, so their per-draw joint moments vary under random
  coupling.
- Stochastic family: 60 real/imaginary claims (5 profiles × 2 harmonics × 3
  vectors × 2 components); Bonferroni two-sided family-wise alpha 0.01, using
  the empirical standard error over independent coupling draws.
- Acceptance for every stochastic claim: its simultaneous interval contains
  the float64 table-derived reference and its half-width is at most 0.05.

## Deterministic structural checks

Single-coordinate vectors `(1,0,0,0)` and `(0,1,0,0)` are invariant to
permutations of the fixed ray offsets. They are not treated as independent
statistical draws. For each profile, compare the native float32 delay roster,
diffuse powers, and per-coordinate ray-angle multisets against the
table-derived roster computed with the native float32 conversion and
TR 38.901 ray-offset equations. Require exact equality after the prescribed
cluster/ray ordering. This checks the deterministic single-coordinate
characteristics without manufacturing a confidence interval from identical
values. The independent float64 table reference remains the stochastic joint
moment target.

## Execution and scope

Run only the test
`test_cdl_joint_delay_angle_characteristics_match_table_reference` after the
seed audit. Preserve all profile-level results and failures. Do not pool or
reuse outcomes from v1–v3. A pass establishes only the listed ray-roster and
selected joint-characteristic checks for these profiles; it does not establish
the full delay-angle law, arbitrary arrays/orientations/patterns, geometric RT
validity, or measured-channel accuracy. Production RT remains gated by G3.
