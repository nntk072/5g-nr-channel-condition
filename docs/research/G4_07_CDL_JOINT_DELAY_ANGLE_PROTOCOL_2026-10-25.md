# G4.07 CDL Joint Delay and Four-Angle Characteristic Protocol

**Status:** Test definition and acceptance family are frozen before execution.
The seed-collision scan is a mandatory pre-run gate. Do not run or change the
seed roster before preserving the scan JSON and SHA-256. Run:

```text
python scripts/audit_seed_roster_collisions.py --root docs/research/results --root outputs --root artifacts --seed 270065 --seed 270066 --seed 270067 --seed 270068 --seed 270069
```

Include every other saved result directory used by the project. A clear result
only describes the scanned JSON/JSON.GZ files; it does not prove global seed
exclusivity.

## Question

Do native Sionna 2.2.0 CDL-A–E diffuse rays reproduce selected joint
characteristics that couple normalized cluster delay with all four AoA, AoD,
ZoA, and ZoD coordinates, relative to a reference built from the hash-pinned
TR 38.901 v19.2 profile tables?

## Frozen design

- Profiles: CDL-A through CDL-E.
- Independent CPU channel draws: 8,192 per profile, seed
  `270000 + ord(profile_letter)`.
- Delay harmonics: 1 and 2 of normalized cluster delay
  `tau / delay_spread`, with a 100 ns delay spread.
- Four-angle frequency vectors, in order: `(1,0,0,0)`, `(0,1,0,0)`,
  `(1,1,0,0)`, `(1,-1,1,0)`, and `(1,1,1,1)`.
- Claims: the 10 delay/angle combinations per profile, each assessed on real
  and imaginary parts, for 100 total claims.
- Family-wise error: two-sided Bonferroni-adjusted normal intervals at
  family-wise alpha 0.01, using the empirical standard error across
  independent channel draws.
- Precision: every simultaneous interval half-width is at most 0.05.
- Acceptance: every interval contains the table-derived reference and meets
  the precision cap. Seeds, harmonics, vector roster, and criteria are not
  changed after observing results.

## Reference and scope

For each diffuse cluster, the reference multiplies the normalized-delay
Fourier term by the product of the four profile-table ray-angle characteristic
terms, averages the 20 angular atoms within each coordinate, then weights by
normalized diffuse cluster power. Native draws use the generated cluster
delays, cluster powers, and jointly coupled ray-angle draws. CDL-D/E's
specular LOS atom is excluded; only their diffuse clusters are compared.

The implementation is
`test_cdl_joint_delay_angle_characteristics_match_table_reference` in
`tests/test_integrations/test_g4_cdl_joint_angular_characteristic_native.py`.
This finite selection is a joint delay/angular consistency check for the
pinned profiles. It does not establish the full delay-angle law, arbitrary
arrays/orientations, antenna-pattern behavior, geometric RT validity, or
measured-channel accuracy.
