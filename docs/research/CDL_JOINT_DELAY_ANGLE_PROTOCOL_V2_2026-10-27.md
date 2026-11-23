# G4.07 CDL Joint Delay and Four-Angle Characteristic Protocol v2

**Status:** Preregistered repair after the v1 attempt was invalidated by a
cluster-order assertion defect. The v1 attempt is preserved in
`results/g4-cdl-joint-delay-angle-attempt-20261027.json` and is not pooled or
counted as scientific evidence. This revision changes neither the estimand,
family-wise error rate, precision cap, sample count, nor acceptance criteria.

## Question and frozen acceptance family

Use the question, reference construction, delay harmonics, four-angle vectors,
100 real/imaginary claims, Bonferroni family-wise alpha 0.01, and maximum
simultaneous interval half-width 0.05 from
`G4_07_CDL_JOINT_DELAY_ANGLE_PROTOCOL_2026-10-25.md`. All claims must contain
their table-derived reference and meet the same precision cap.

## Preregistered replication

- Profiles: CDL-A through CDL-E, Sionna 2.2.0, specification version 19.2.
- Independent CPU channel draws: 8,192 per profile.
- Seeds: 271065 through 271069, respectively. These are disjoint from the
  invalidated v1 seeds 270065 through 270069 and must pass the seed-collision
  audit before execution.
- Do not pool this replication with v1 or any earlier result.
- Seed-collision audit: `results/g4-cdl-joint-delay-angle-v2-seed-collision-audit-20261027.json`, SHA-256
  `2740d11cb263b5ddab67b4585095aeca88dbd04a99f81f855fc14582affc7f8c`;
  status `CLEAR_WITHIN_SCANNED_JSON_CORPUS` across 2,264 saved JSON/JSON.GZ
  files, with no collisions, malformed seed fields, or incomplete files.

## Implementation correction

Sionna returns each profile's delay clusters in source-table order. The
independent reference builder sorts the ETSI rows by delay. The v1 test compared
these two representations element by element and stopped before computing any
statistical claim. In v2, sort each draw's native cluster delays and apply the
same per-draw cluster permutation to native powers and four-angle arrays. Then
compare sorted native delays with sorted table delays and compute the existing
joint moments. This matches cluster identities without changing any statistic
or acceptance threshold.

## Scope

A pass supports only these selected normalized-delay/four-angle characteristics
for the five named CDL profiles and this pinned implementation. It does not
establish the full delay-angle law, arbitrary arrays/orientations/patterns,
geometric RT validity, or measured-channel accuracy. Production RT remains
gated by the broader G3 criteria.
