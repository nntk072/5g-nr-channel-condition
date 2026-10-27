# G4.07 CDL Joint Delay and Four-Angle Characteristic Protocol v5

**Status:** Preregistered reproducibility run of the v4 estimand and acceptance
criteria. V1–v4 seeds and outcomes are not reused or pooled. The sole v5
implementation change is machine-readable serialization of each existing
claim's estimate, reference, interval, and pass state.

## Frozen design and acceptance

Follow `G4_07_CDL_JOINT_DELAY_ANGLE_PROTOCOL_V4_2026-10-27.md` without changing
its deterministic float32 roster checks, stochastic joint vectors, 60-claim
family, alpha 0.01, confidence interval method, 8,192 draws/profile, or 0.05
half-width cap. Profiles remain CDL-A–E using Sionna 2.2.0 and TR 38.901
version 19.2.

- Fresh CPU seeds: 274065–274069, respectively. These are disjoint from v1
  270065–270069, v2 271065–271069, v3 272065–272069, and v4 273065–273069.
- Seed-collision audit: `results/g4-cdl-joint-delay-angle-v5-seed-collision-audit-20261027.json`, SHA-256
  `bc9017496548c33a986166777a994264bb22cf0ab51dcc02c17d8c4a297cde45`;
  status `CLEAR_WITHIN_SCANNED_JSON_CORPUS` across 2,270 saved JSON/JSON.GZ
  files, with no collisions, malformed seed fields, or incomplete files.
- Execute only
  `test_cdl_joint_delay_angle_characteristics_match_table_reference`, with
  output capture disabled so the five JSON metric records are retained.
- The `G4_CDL_JOINT_DELAY_ANGLE_METRICS ` records must serialize every
  profile's 12 claims, including estimates, references, simultaneous interval
  endpoints, half-widths, profile seed, sample count, and pass state. Preserve
  both the complete test log and a validated JSON report built from these
  records.

## Scope

This is an artifact/reproducibility repeat of the v4 bounded criterion, not a
broader channel qualification. The v4 limitations continue to apply; overall
G4/G3 and production RT gates remain open.
