# G4.07 CDL Bounded Joint Four-Angle Fourier Protocol

**Status:** The finite family, reference construction, fresh seeds, and
acceptance limits are frozen in test code before execution. The structured
seed-collision scan remains a mandatory pre-run gate. Do not run this family or
change its seed roster before recording that scan. Run
`python scripts/audit_seed_roster_collisions.py --root docs/research/results --root outputs --root artifacts --seed 260065 --seed 260066 --seed 260067 --seed 260068 --seed 260069`
and add every other saved result directory used by the project. Preserve the
complete scan JSON and its SHA-256 with the eventual run record. A clear result applies only
to the scanned JSON/JSON.GZ files and does not establish global seed
exclusivity. String/float encodings, inclusive seed ranges, or seed-bearing
result keys that match a reserved seed are reported as possible collisions;
other non-integer values or malformed ranges under seed-named fields make the
scan unresolved and block the run.

## Question

Do native Sionna 2.2.0 CDL-A–E diffuse ray-angle draws reproduce the
table-derived joint characteristic values over the four-angle integer
frequency domain `[-2, 2]^4` with at least two nonzero components?

## Frozen design

- Profiles: CDL-A, CDL-B, CDL-C, CDL-D, and CDL-E.
- Independent CPU draws: 8,192 per profile, with exact seed
  `260000 + ord(profile_letter)`.
- Frequency roster: one representative from each conjugate pair in
  `{-2,-1,0,1,2}^4` with at least two nonzero entries (304 vectors/profile).
  The omitted negative vector has the conjugate characteristic value and does
  not add an independent claim.
- Claims: real and imaginary parts for every roster vector and profile
  (3,040 total).
- Family-wise error: two-sided Bonferroni-adjusted normal intervals at
  family-wise alpha 0.01, using the empirical standard error across independent
  channel draws.
- Precision: every simultaneous interval half-width is at most 0.05.
- Acceptance: every component interval contains the table-derived reference
  and meets the precision limit. The roster, seeds, threshold, or multiplicity
  correction must not change after outcomes are observed.

## Reference and scope

For each diffuse cluster, the reference averages its 20 profile-table ray atoms
separately along AoA, AoD, ZoA, and ZoD, multiplies those four characteristic
values for the requested frequency vector, then weights clusters by normalized
profile power. The native statistic uses all four generated angles jointly,
averages over rays, and normalizes diffuse cluster power per draw. CDL-D/E
specular LOS atoms are excluded from this diffuse-law comparison.

This is a finite low-frequency characterization, not proof of the complete
continuous four-angle distribution, arbitrary array/pattern behavior,
geometric RT validity, or measured-channel accuracy. The test is
`test_cdl_four_angle_bounded_joint_fourier_family_matches_table_reference` in
`tests/test_integrations/test_g4_cdl_joint_angular_characteristic_native.py`.
