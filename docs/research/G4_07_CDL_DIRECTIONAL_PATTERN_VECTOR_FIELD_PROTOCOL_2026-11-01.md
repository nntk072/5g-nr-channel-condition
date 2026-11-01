# G4.07 CDL Directional-Pattern Vector-Field Protocol

**Status:** This document freezes the revised independent reference and fresh
seeds. It supersedes the scalar-gain reference in the 2026-10-25 dual-endpoint
protocol. The previous cohort was inspected only to diagnose the reference
defect and is not evidence for this revised analysis. Use the retained-result
runner below for native execution; a passing regression is bounded evidence,
not a consolidated qualification report.

## Retained-result workflow

Run `./.venv/bin/python -m scripts.run_g4_cdl_directional_pattern_power
--output docs/research/results/g4-cdl-directional-pattern-power-result-20261101.json`.
The runner uses the frozen seeds and draw count, captures the per-cluster
reference, estimate, interval, and case identity from each test, checks the
exact 70-case/1,358-claim roster and interval arithmetic, validates the strict
result schema, and publishes without overwriting an existing file. A
nonpassing or incomplete run is retained as such and exits nonzero. The result
binds the test/protocol/schema/reference digests and the seed-audit digest. The
summary audit recomputes interval membership, cluster counts, power totals,
case identity, and the frozen family roster; it does not reconstruct sample
variance without raw draws.

## Scope and frozen cases

- Runtime: Sionna PHY 2.2.0 on CPU, CDL-A–E, TR 38.901 parameter tables 19.2.
- Patterns: `38.901` and `38.901-handheld` at 3.5 GHz.
- Arrays: one vertically polarized element at each end.
- One-end cohort: each of the two UT patterns with omni BS, over zero UT
  orientation and UT orientations `(0.2, -0.1, 0.3)` and
  `(-0.7, 0.35, 1.1)` radians: 30 cases.
- Dual-end cohort: all four ordered UT/BS pattern pairs, with both endpoints
  zero-oriented and with distinct UT `(0.2, -0.1, 0.3)` and BS
  `(-0.7, 0.35, 1.1)` orientations: 40 cases.
- Each case uses 16,384 independent stochastic channel draws, 100 ns delay
  spread, 30 kHz subcarrier spacing, and eight subcarriers. Total across both
  cohorts: 1,146,880 draw rows.
- The 99% Bonferroni families retain their prior scopes: 194 cluster claims
  for the zero-orientation one-end cohort, 388 claims for the two nonzero
  one-end orientations, and 776 claims across the dual-end cases. Every
  reference must lie inside its simultaneous interval, whose half-width is
  capped at 5% of that case's total expected pattern-weighted power.
- Fresh seeds are assigned in `test_g4_cdl_antenna_pattern_power_native.py`:
  one-end seeds are `2,700,000`–`2,700,029`; dual-end seeds are
  `2,701,000`–`2,701,039`. The pre-run scan is
  `G4_CDL_ANTENNA_PATTERN_VECTOR_FIELD_SEED_AUDIT_2026-11-01.json`, SHA-256
  `190c3c0865cabceae01ec18468939cee5f94e5090bd4c24c73efd97a30c5b39c`. It
  scanned 2,276 JSON/JSON.GZ files under research results, outputs, and
  artifacts and found no collisions, malformed seed fields, or incomplete
  files. This only establishes clearance within that saved corpus snapshot.

## Independent vector-field expectation

The table-derived ray angles, cluster powers, spreads, LOS K factor, and XPR
come from the byte-pinned ETSI transcription, not adapter metadata. For each
endpoint, the expected local element power follows TR 38.901 Tables 7.3-1/2.
The global direction is independently inverse-rotated using the composite
Euler matrix; the local vertical field is then rotated to global polarization
components using the TR 38.901 displacement angle `psi` (7.1-11/15):

`f = sqrt(G(theta_local, phi_local)) * [cos(psi), sin(psi)]`.

For diffuse rays, the four initial polarization phases are independent and
uniform. With linear XPR `X`, and global receive/transmit field components
`f_rx` and `f_tx`, the independent phase expectation is:

`E|h_field|^2 = rx0*tx0 + rx1*tx1 + (rx0*tx1 + rx1*tx0)/X`,

where `rxj = E[f_rx,j^2]` and `txj = E[f_tx,j^2]`. The endpoint moments average
over all 20-by-20 azimuth/zenith offset pairings, matching the independent CDL
angle permutations. The diffuse expectation is multiplied by normalized
cluster power and divided by `K+1`. For CDL-D/E's zero-delay merged cluster,
the deterministic LOS field uses Sionna 2.2's pinned `diag(1,-1)` phase matrix;
its power is added with weight `K/(K+1)`. The NLOS/LOS cross term has zero
expectation because NLOS initial phases are independent and uniform. The
expected reference contains no native generated outputs.

The regression implementation is
`tests/test_integrations/test_g4_cdl_antenna_pattern_power_native.py`. All
cohorts must be run with these frozen seeds and limits. Do not replace a seed,
increase the draw count, change a margin, or waive a failed cluster after
observing results.

## Interpretation

Passing supports only these five pinned profile tables, two element patterns,
single-element vertical ports, specified orientations, and the stated
power-statistic families. It does not qualify arbitrary polarization/array
layouts, geometric RT, measured scenes, NR receiver impact, or overall G4.
