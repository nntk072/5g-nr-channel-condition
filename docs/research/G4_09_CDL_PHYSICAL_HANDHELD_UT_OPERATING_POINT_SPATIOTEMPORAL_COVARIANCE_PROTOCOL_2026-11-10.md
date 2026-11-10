# G4.09 physical HandheldUTArray operating-point CDL space-time covariance protocol - 2026-11-10

## Purpose

This protocol freezes a bounded Sionna PHY 2.2.0 / TR 38.901 V19.2.0 implementation-consistency family for the cross-time covariance E[h_i(0)^* h_j(tau)] of Sionna's physical HandheldUTArray across multiple handset orientations, mobility directions, carriers, and temporal lags.

It extends G4.08 without changing the physical-handheld oracle. The independent reference in evaluation/g4_cdl_handheld_ut_spatiotemporal_covariance.py reconstructs the TR 38.901 handheld candidate geometry, per-port local bases, Table 7.3-2 element field response, physical array phase, receive Doppler, diffuse Jones/XPR statistics, and CDL-D/E specular contribution. Sionna is used only as the system under test for the retained native family.

## Frozen design

- Profiles: CDL-A, B, C, D, E.
- Specification/runtime: TR 38.901 V19.2.0 through Sionna PHY 2.2.0 CPU.
- Delay spread: 100 ns.
- OFDM adapter request: 30 kHz SCS and 8 subcarriers.
- BS: one single-V omni PanelArray port at orientation (0,0,0) rad.
- UT: physical single-V HandheldUTArray with the Table 7.3-2 handheld element pattern.
- UT rosters:
  - tr38901-4: candidate locations (1,7,3,5), four ports.
  - tr38901: all candidate locations (1,2,3,4,5,6,7,8), eight ports.
- Handset dimensions: 0.15 m depth x 0.07 m width.
- Operating point zero_x_lowband:
  - UT orientation (0,0,0) rad.
  - UT velocity (30,0,0) m/s.
  - carrier 2.6 GHz.
  - lag 0.25 ms.
- Operating point tilted_y_midband:
  - UT orientation (-0.35,0.22,-0.48) rad.
  - UT velocity (0,-30,0) m/s.
  - carrier 3.5 GHz.
  - lag 1.0 ms.
- Operating point oblique_highband:
  - UT orientation (0.45,-0.25,0.8) rad.
  - UT velocity (-12,24,13) m/s.
  - carrier 4.9 GHz.
  - lag 0.4 ms.
- Independent stochastic channel draws: 16,384 per profile/operating-point/roster case.
- Case count: 5 profiles x 3 operating points x 2 rosters = 30.
- Observable:
  - all 4x4 complex entries for tr38901-4;
  - all 8x8 complex entries for tr38901.
- Simultaneous coordinates: 5 x 3 x (4x4 + 8x8) x 2 = 2,400 real/imaginary claims.
- Family-wise alpha: 0.01.
- Confidence rule: one two-sided Bonferroni normal family across all 2,400 coordinates.
- Precision rule: every simultaneous interval half-width must be at most 0.05 in the channel's native normalized-power units.
- Formal seeds: 2,820,000 through 2,820,029, mapped profile-major, then operating-point order above, then roster order tr38901-4, tr38901.
- No formal seed, draw count, profile, operating point, roster, handset dimension, observable, confidence rule, or precision criterion may be changed after observing the formal native result.

## Independent-reference semantics

For each physical handheld port, the oracle constructs the standardized candidate position and local port basis from the handset rectangle. For every CDL ray it evaluates the Table 7.3-2 element field in the port-local frame, rotates the field through the declared handset orientation into the global spherical basis, and applies the physical port-position phase at the declared carrier.

For diffuse CDL components, the oracle uses the exact-digest-pinned ETSI-derived component fixture, independently expands standardized ray offsets, applies receive-side spatial and Doppler phases for the declared velocity and lag, and analytically averages the four diffuse Jones entries using unit co-polar variance and 1/XPR cross-polar variance. The single-V omni BS field is included explicitly.

For CDL-D/E specular LOS, the oracle applies deterministic diag(1,-1) Jones coupling, the same physical handheld port fields and positions, and receive-side Doppler.

## Pinned provenance

- ETSI-derived CDL tap/angle fixture: docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json.
- Fixture SHA-256: 24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090.
- ETSI source bytes SHA-256: b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4.
- Sionna antenna.py: 09e20b89f4c4d990bed1a3ce3d6314f9fe7e45c090737a6f5790166d4a63f926.
- Sionna channel_coefficients.py: 47059c7bf0998d2a230909e70743832ee928a738a97b9b5a25a185bb4947b5fc.
- Sionna cdl.py: 91e90eb7498a3725f628f2f4d4dca6c8d7438ac549bd5d68a3f45563f282361d.

## Pre-freeze oracle and precision diagnostics

The G4.08 portable and native deterministic oracle checks already establish candidate positions, right-handed port bases, Table 7.3-2 boresight gain, physical HandheldUTArray geometry, per-port field rotation, and adapter/native agreement for both supported rosters.

Before freezing this operating-point protocol, a non-admissible 4,096-draw diagnostic used separate seeds 9,820,000 onward over the entire proposed 30-case, 2,400-coordinate family. It produced 23 passing cases and 7 cases with at least one interval/precision miss, 36 failed claims, maximum absolute error 0.03153823556399016, and maximum simultaneous half-width 0.0870422435498099. The result showed that 4,096 draws were insufficient for the unchanged 0.05 precision criterion.

The formal draw count was therefore selected as 16,384 before any formal seed was used. A separate non-admissible 16,384-draw diagnostic with seeds 9,821,000 onward checked the highest-risk oblique_highband slice across all five profiles and both rosters: 10/10 cases and 800/800 claims passed, with maximum absolute error 0.015220850333881097 and maximum half-width 0.042287631445639705. A second 16,384-draw diagnostic selection covering the earlier tilted concern also passed 10/10 selected cases and 800/800 claims, with maximum absolute error 0.013797393848295078 and maximum half-width 0.026318133726655764.

These diagnostic seeds and estimates are non-admissible and cannot replace the formal execution. No formal seed in 2,820,000-2,820,029 was observed during protocol design.

## Seed audit

docs/research/G4_CDL_PHYSICAL_HANDHELD_UT_OPERATING_POINT_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json must clear all formal seeds 2,820,000-2,820,029 against the saved research-result/output/artifact JSON corpora before formal native execution. The audit is corpus-scoped and does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned CDL profiles, two physical single-V handheld rosters, default 0.15 m x 0.07 m handset dimensions, and the three finite operating points above. It does not establish continuous orientation, velocity, carrier, or lag coverage; dual-polarized HandheldUTArray behavior; arbitrary candidate subsets or handset dimensions; arbitrary arrays/topologies/patterns; mutual coupling; measured-channel truth; RT/material truth; receiver qualification; general 3GPP conformance; overall G4 qualification; production D1 authorization; or advanced-training authorization.

This family remains distinct from the earlier normal PanelArray family whose 38.901-handheld label tested the Table 7.3-2 element radiation law without Sionna's physical handheld-device geometry.
