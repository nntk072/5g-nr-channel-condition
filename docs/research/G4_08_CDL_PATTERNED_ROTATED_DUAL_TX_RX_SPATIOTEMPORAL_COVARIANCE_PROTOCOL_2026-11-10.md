# G4.08 patterned rotated dual-TX/dual-RX CDL space-time covariance protocol — 2026-11-10

## Purpose

This protocol freezes one bounded Sionna PHY 2.2.0 / TR 38.901 V19.2.0 implementation-consistency family for the full cross-time covariance `E[h_p(0)^* h_q(tau)]` when both endpoints simultaneously have two physical dual-VH elements, nonzero 3D orientation, directional element patterns, spatial phase, polarization-basis rotation, Step-8 random angular coupling, and receive-side Doppler.

The independent oracle is `evaluation/g4_cdl_patterned_rotated_dual_tx_rx_spatiotemporal_covariance.py`. It extends the already checked rotated dual-endpoint equations with the TR 38.901 Tables 7.3-1/7.3-2 element amplitude law evaluated after inverse rotation into each endpoint local frame. It does not call Sionna for expected values.

## Frozen design

- Profiles: CDL-A, B, C, D, E.
- Specification: TR 38.901 V19.2.0 through Sionna PHY 2.2.0 CPU.
- Carrier: 3.5 GHz.
- Delay spread: 100 ns.
- OFDM adapter request: 30 kHz SCS and 8 subcarriers.
- TX: two physical elements, half-wavelength local-y spacing, dual-VH ports.
- RX: two physical elements, half-wavelength local-y spacing, dual-VH ports.
- Native endpoint port order: `V0,V1,H0,H1`.
- Flattened channel order: RX-port-major / TX-port-minor, 16 channel ports.
- TX/BS orientation: `(-0.7, 0.35, 1.1)` rad.
- RX/UT orientation: `(0.2, -0.1, 0.3)` rad.
- Element-pattern pairs: Cartesian product of `38.901` and `38.901-handheld` at TX and RX, four pairs total.
- The `38.901-handheld` label here is the Table 7.3-2 element radiation law on a `PanelArray`; this protocol does not cover Sionna's separate physical `HandheldUTArray` device geometry.
- Receive velocity: `(10,20,20)` m/s.
- Lag: 0.7 ms.
- Independent stochastic channel draws: 32,768 per profile/pattern-pair case.
- Case count: 5 profiles × 4 pattern pairs = 20.
- Observable: all 16×16 complex entries of `E[h_p(0)^* h_q(0.7 ms)]`.
- Simultaneous coordinates: 20 × 16 × 16 × 2 = 10,240 real/imaginary claims.
- Family-wise alpha: 0.01.
- Confidence rule: one two-sided Bonferroni normal family across all 10,240 coordinates.
- Precision rule: every simultaneous interval half-width must be at most 0.05 in the channel's native normalized-power units.
- Formal seeds: 2,817,000 through 2,817,019, mapped by profile-major then pattern-pair order.
- No seed, draw count, pattern/orientation roster, carrier, lag, velocity, observable, confidence rule, or precision criterion may be changed after observing the frozen native result.

## Independent-reference semantics

For each diffuse CDL component, the oracle:

1. reads only the exact-digest-pinned ETSI-derived CDL component fixture;
2. expands the 20 standardized ray offsets independently in azimuth and zenith at each endpoint, matching Sionna Step-8 ensemble angular-coupling semantics;
3. inverse-rotates each global ray into the endpoint local frame and evaluates the selected TR 38.901 element power pattern;
4. multiplies the square-root element gain into the independently derived rotated V/H field basis;
5. includes TX and RX local-y spatial phase and receive-side Doppler;
6. combines the four independent diffuse Jones entries with unit co-polar variance and `1/XPR` cross-polar variance.

For CDL-D/E specular LOS, the oracle applies the same patterned endpoint fields, deterministic `diag(1,-1)` Jones coupling, TX/RX spatial offsets, and receive Doppler.

## Pinned provenance

- ETSI-derived CDL tap/angle fixture: `docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`.
- Fixture SHA-256: `24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090`.
- ETSI source bytes SHA-256: `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.
- Sionna `antenna.py`: `09e20b89f4c4d990bed1a3ce3d6314f9fe7e45c090737a6f5790166d4a63f926`.
- Sionna `channel_coefficients.py`: `47059c7bf0998d2a230909e70743832ee928a738a97b9b5a25a185bb4947b5fc`.
- Sionna `cdl.py`: `91e90eb7498a3725f628f2f4d4dca6c8d7438ac549bd5d68a3f45563f282361d`.

## Pre-freeze oracle validation

Before freezing this protocol:

- 12 dependency-light oracle tests passed with coverage disabled, including exact reduction to the already frozen omni oracle for CDL-A-E.
- Eight deterministic native Sionna Step-11 comparisons spanning both supported directional element patterns at both endpoints and two independent angle tuples matched the independent patterned field transform to numerical precision.
- A non-admissible 4,096-draw diagnostic over CDL-A/D/E and all four pattern pairs at a separate retained rotated orientation found no systematic discrepancy; its largest raw standardized coordinate discrepancy was 4.13 sigma across thousands of inspected coordinates.
- A second non-admissible 4,096-draw diagnostic over all five profiles and all four pattern pairs at the frozen orientation found maximum raw standardized discrepancy 4.2528 sigma across 10,240 inspected coordinates. Under the subsequently frozen simultaneous family, its largest 4,096-draw half-width was 0.09697; inverse-square-root scaling motivated the 32,768-draw budget while the pre-existing 0.05 precision ceiling remained unchanged. Diagnostic seeds and estimates are non-admissible and cannot replace the frozen execution.

## Seed audit

`docs/research/G4_CDL_PATTERNED_ROTATED_DUAL_TX_RX_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json` clears seeds 2,817,000–2,817,019 against the saved `docs/research/results`, `outputs`, and `artifacts` JSON/JSON.GZ corpora before native execution. The audit scanned 2,298 eligible artifacts with no collisions, malformed seed fields, or incomplete files. Its SHA-256 is `9a80f72bc544894e5b4a7ab2fb53c079ee72720e0f67b4006b5cc6fb4d6ba355`. The audit is corpus-scoped and does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned CDL profiles, four frozen PanelArray element-pattern pairs, one simultaneous nonzero TX/RX orientation pair, one 2×2 physical dual-VH topology, one velocity, one carrier, and one lag. It does not establish arbitrary or physical handheld-device arrays, continuous orientation/velocity coverage, arbitrary patterns, beam codebooks, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, general 3GPP conformance, overall G4 qualification, production D1 authorization, or advanced-training authorization.
