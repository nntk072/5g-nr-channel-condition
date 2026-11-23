# G4.07 CDL Joint Delay and Four-Angle Characteristic Protocol v3

**Status:** Preregistered repair after the v2 attempt stopped at a structural
float-precision check. The v1 and v2 attempts are preserved separately and are
not pooled or counted as scientific evidence. This revision changes neither
the estimand, family-wise error rate, precision cap, sample count, nor
acceptance criteria.

## Question and frozen acceptance family

Use the question, reference construction, delay harmonics, four-angle vectors,
100 real/imaginary claims, Bonferroni family-wise alpha 0.01, and maximum
simultaneous interval half-width 0.05 from
`G4_07_CDL_JOINT_DELAY_ANGLE_PROTOCOL_2026-10-25.md`. All claims must contain
their table-derived reference and meet the same precision cap.

## Preregistered replication

- Profiles: CDL-A through CDL-E, Sionna 2.2.0, specification version 19.2.
- Independent CPU channel draws: 8,192 per profile.
- Seeds: 272065 through 272069, respectively. They are disjoint from v1
  (270065–270069) and v2 (271065–271069), and must pass a fresh seed-collision
  audit before execution.
- Do not pool this replication with either invalidated attempt or any earlier
  result.
- Seed-collision audit: `results/g4-cdl-joint-delay-angle-v3-seed-collision-audit-20261027.json`, SHA-256
  `663cccaa361d5786476cd693c4c8b0af0956b88f62cfd19135324f0096c224f3`;
  status `CLEAR_WITHIN_SCANNED_JSON_CORPUS` across 2,266 saved JSON/JSON.GZ
  files, with no collisions, malformed seed fields, or incomplete files.

## Implementation correction

Sionna returns source-order clusters and stores generated delays as float32;
the independent ETSI table values are float64. V1 compared source order with a
sorted table. V2 aligned cluster order but required zero relative error between
the different floating-point representations. V3 sorts each draw's native
delays and applies the same per-draw permutation to native powers and
four-angle arrays. The structural delay check uses absolute tolerance `2e-7`
and relative tolerance `2 * torch.finfo(native_delay_dtype).eps`, accounting
for the two float32 rounding operations in delay scaling and normalization.
The statistical estimand and all acceptance thresholds remain unchanged.

## Scope

A pass supports only these selected normalized-delay/four-angle characteristics
for the five named CDL profiles and this pinned implementation. It does not
establish the full delay-angle law, arbitrary arrays/orientations/patterns,
geometric RT validity, or measured-channel accuracy. Production RT remains
gated by the broader G3 criteria.
