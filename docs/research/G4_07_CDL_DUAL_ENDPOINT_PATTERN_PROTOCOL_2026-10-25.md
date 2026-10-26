# G4.07 CDL Dual-Endpoint Directional-Pattern Protocol

**Status:** Frozen in code and protocol before execution; no native run has been
performed for this extension.

## Objective and scope

Extend the existing hash-pinned TR 38.901 v19.2 CDL directional-element
cluster-power comparison from a directional UT with an omnidirectional BS to
directional element patterns at both link ends. The check covers CDL-A through
CDL-E, both supported 38.901 element patterns at each endpoint, zero
orientation, and one pair of distinct nonzero 3D endpoint orientations.

This is standardized stochastic-model consistency. It does not qualify
arbitrary arrays, arbitrary element patterns, geometric RT, measured scenes,
or a receiver. The code path is in
`tests/test_integrations/test_g4_cdl_antenna_pattern_power_native.py` as
`test_cdl_directional_patterns_at_both_link_ends_match_independent_reference`.

## Frozen cases and source

- Runtime: locked Python/Sionna 2.2.0, native CPU, TR 38.901 profile tables
  version 19.2.
- Profiles: CDL-A, CDL-B, CDL-C, CDL-D, and CDL-E.
- Patterns at each endpoint: `38.901` and `38.901-handheld`, for four UT/BS
  pattern pairs per profile.
- Orientations: both endpoints at zero orientation, and one case with distinct
  UT `(0.2, -0.1, 0.3)` and BS `(-0.7, 0.35, 1.1)` rad orientations.
- Independent draws: 16,384 per profile/pattern-pair/orientation case; 40
  cases, exactly 655,360 channel draws in total.
- Channel setup: 3.5 GHz, 100 ns delay spread, 30 kHz subcarrier spacing,
  eight subcarriers, one single-polarized element per endpoint.
- Seeds: contiguous case roster `2,500,000` through `2,500,039`, ordered by
  profile, UT/BS pattern pair, then orientation. Before execution, run the
  G4.07 seed-collision audit against the saved research-result roots and record
  its report digest. Do not run if any seed collides or the scan is unresolved.
- Profile table hashes are fixed in the regression source and checked before
  the comparison.

## Independent reference and acceptance

For each diffuse cluster, the reference evaluates the TR 38.901 endpoint power
pattern over the full 20-by-20 independent azimuth/zenith offset pairing at
the UT and BS, then multiplies the two mean endpoint gains by the normalized
table cluster power. For CDL-D/E, the direct LOS contribution is evaluated at
the table LOS directions at both endpoints and combined with the first diffuse
cluster's power. The expectation is derived from the pinned profile table and
the standard power-pattern equation; it does not call Sionna's pattern code.

All 776 per-cluster power claims share one two-sided 99% Bonferroni interval
family. Every independent-reference value must lie within its interval, and
each interval half-width must be no greater than 5% of the total expected
pattern-weighted diffuse power for that case. A case failure is retained as a
failure; no seed replacement, sample-size increase, or post-run tolerance
change is allowed.

## Disposition

A pass supports only these five standardized profile tables, two element
patterns, four pattern-pair combinations, and two endpoint-orientation cases.
It does not close G4 or authorize production RT generation. The run and its
independent result summary must be retained and audited before any claim is
made.
