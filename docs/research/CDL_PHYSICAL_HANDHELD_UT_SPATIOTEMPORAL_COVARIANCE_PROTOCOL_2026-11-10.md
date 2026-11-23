# G4.08 physical HandheldUTArray CDL space-time covariance protocol - 2026-11-10

## Purpose

This protocol freezes one bounded Sionna PHY 2.2.0 / TR 38.901 V19.2.0 implementation-consistency family for the cross-time covariance E[h_i(0)^* h_j(tau)] of Sionna's physical HandheldUTArray geometry.

It closes the specific physical-handheld gap left by the earlier patterned PanelArray family. The independent oracle is evaluation/g4_cdl_handheld_ut_spatiotemporal_covariance.py. It reconstructs the TR 38.901 handheld candidate positions, per-port local bases, Table 7.3-2 element field response, array phase, Doppler, diffuse Jones statistics, and CDL-D/E specular contribution without calling Sionna for expected covariance values.

## Frozen design

- Profiles: CDL-A, B, C, D, E.
- Specification: TR 38.901 V19.2.0 through Sionna PHY 2.2.0 CPU.
- Carrier: 3.5 GHz.
- Delay spread: 100 ns.
- OFDM adapter request: 30 kHz SCS and 8 subcarriers.
- BS: one single-V omni PanelArray port at zero orientation.
- UT: physical single-V HandheldUTArray with Table 7.3-2 handheld element pattern.
- UT rosters:
  - tr38901-4: candidate locations (1,7,3,5), four ports.
  - tr38901: all candidate locations (1,2,3,4,5,6,7,8), eight ports.
- Handset dimensions: 0.15 m depth x 0.07 m width.
- UT orientation: (0.2,-0.1,0.3) rad.
- BS orientation: (0,0,0) rad.
- UT velocity: (10,20,20) m/s.
- Lag: 0.7 ms.
- Independent stochastic channel draws: 16,384 per profile/roster case.
- Case count: 5 profiles x 2 rosters = 10.
- Observable:
  - all 4x4 complex entries for tr38901-4;
  - all 8x8 complex entries for tr38901.
- Simultaneous coordinates: 5 x (4x4 + 8x8) x 2 = 800 real/imaginary claims.
- Family-wise alpha: 0.01.
- Confidence rule: one two-sided Bonferroni normal family across all 800 coordinates.
- Precision rule: every simultaneous interval half-width must be at most 0.05 in the channel's native normalized-power units.
- Formal seeds: 2,819,000 through 2,819,009, mapped by profile-major then roster order (tr38901-4, tr38901).
- No seed, draw count, roster, orientation, carrier, lag, velocity, observable, confidence rule, or precision criterion may be changed after observing the frozen native result.

## Independent-reference semantics

For each selected physical handheld port, the oracle independently constructs the standardized candidate position and its local port basis from the handset rectangle. For each CDL ray it evaluates the Table 7.3-2 element field in the port-local frame, rotates that field through the handset orientation into the global spherical basis, and applies the physical port-position phase.

For diffuse CDL components, the oracle uses the exact-digest-pinned ETSI-derived component fixture, independently expands the standardized ray offsets, applies the receive-side spatial and Doppler phases, and analytically averages the four diffuse Jones entries using unit co-polar variance and 1/XPR cross-polar variance. The single-V omni BS field is included explicitly.

For CDL-D/E specular LOS, the oracle applies deterministic diag(1,-1) Jones coupling, the same physical handheld port fields and positions, and receive-side Doppler.

## Pinned provenance

- ETSI-derived CDL tap/angle fixture: docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json.
- Fixture SHA-256: 24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090.
- ETSI source bytes SHA-256: b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4.
- Sionna antenna.py: 09e20b89f4c4d990bed1a3ce3d6314f9fe7e45c090737a6f5790166d4a63f926.
- Sionna channel_coefficients.py: 47059c7bf0998d2a230909e70743832ee928a738a97b9b5a25a185bb4947b5fc.
- Sionna cdl.py: 91e90eb7498a3725f628f2f4d4dca6c8d7438ac549bd5d68a3f45563f282361d.

## Pre-freeze oracle validation

Before freezing this protocol:

- Portable oracle checks passed for the four-port candidate positions, right-handed orthonormal per-port bases, boresight Table 7.3-2 gain, covariance dimensions, finite values, and invalid roster rejection.
- Native deterministic checks matched the independently reconstructed candidate positions and port bases against Sionna HandheldUTArray for both supported rosters.
- Native Step-11 field checks matched the independent per-port field calculation at two unrelated angle tuples for both rosters to numerical precision.
- The CDL adapter matched direct Sionna physical HandheldUTArray generation for both rosters and recorded explicit array-kind metadata.
- A non-admissible 4,096-draw diagnostic over all five profiles and both rosters inspected the same 800 coordinate family at the frozen geometry. It had zero simultaneous interval misses. The largest raw standardized discrepancy was 3.66836 sigma, the largest absolute discrepancy was 0.01719473, and the largest simultaneous half-width was 0.03707447.
- Inverse-square-root scaling of that diagnostic projects a largest half-width of about 0.01854 at 16,384 draws, comfortably below the pre-existing 0.05 precision ceiling. Diagnostic seeds and estimates are non-admissible and cannot replace the frozen execution.

## Seed audit

docs/research/G4_CDL_PHYSICAL_HANDHELD_UT_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json must clear seeds 2,819,000-2,819,009 against the saved docs/research/results, outputs, and artifacts JSON/JSON.GZ corpora before native execution. The audit is corpus-scoped and does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned CDL profiles, the two frozen physical single-V handheld rosters, default 0.15 m x 0.07 m handset dimensions, one UT orientation, one velocity, one single-V omni BS port, one carrier, and one lag. It does not establish dual-polarized HandheldUTArray behavior, arbitrary candidate subsets or handset dimensions, arbitrary arrays/topologies/patterns, continuous orientation/velocity coverage, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, general 3GPP conformance, overall G4 qualification, production D1 authorization, or advanced-training authorization.
