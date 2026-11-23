# G4.07 CDL Joint Space-Time Covariance Protocol

**Status:** frozen bounded native protocol. Passing is Sionna/TR 38.901 model-consistency evidence for the named finite cases only and is not overall G4 qualification.

## Scientific gap

Existing retained families check CDL spatial covariance and temporal covariance separately. This protocol checks their joint dependence by estimating the full two-element cross-time matrix `E[h_i(0)^* h_j(tau)]`, so antenna separation and Doppler phase occur in the same claim rather than in separate marginal regressions.

## Frozen design

- Runtime: Sionna PHY 2.2.0, CPU.
- Profiles: CDL-A, B, C, D, E from the pinned TR 38.901 v19.2.0 fixture.
- Carrier: 3.5 GHz; delay spread: 100 ns; SCS: 30 kHz; eight subcarriers.
- UT array: two single-V omnidirectional elements on the local-y axis at 0.5 wavelength spacing, zero orientation. BS: one single-V omnidirectional element.
- Lag: 0.7 ms, represented by two native time samples.
- Velocity cases: `x=(30,0,0)` m/s and `oblique_122=(10,20,20)` m/s; both have speed 30 m/s.
- Independent channel draws: 16,384 per profile/velocity case.
- Seeds: 2,810,000 through 2,810,009, ordered by profile A–E and then velocity `x`, `oblique_122`.
- Claims: every real and imaginary component of the 2x2 cross-time matrix, eight claims/case and 80 simultaneous claims total.
- Family-wise alpha: 0.01 with one two-sided Bonferroni normal family over all 80 claims.
- Precision ceiling: every interval half-width must be at most 0.05 normalized mean power.
- Failed or incomplete frozen executions are retained; seeds, draw counts, alpha, family size, lag, geometry, and precision ceiling must not be changed after observing results.

## Independent reference

`evaluation/g4_cdl_spatiotemporal_covariance.py` reads the exact SHA-256-pinned ETSI-derived CDL component fixture. For a local-y element separation `d` wavelengths and arrival direction `u`, the reference for `h_i(0)^* h_j(tau)` uses the combined phase

`2*pi*d*(j-i)*u_y + 2*pi*f_c*tau*(v dot u)/c`.

Diffuse cluster expectations average the Cartesian product of the standard 20 azimuth and 20 zenith ray-offset sets, matching their independent coupling. CDL-D/E retain their separately listed specular component as a deterministic direction and add the diffuse components with their independently transcribed normalized powers. The reference does not inspect generated channel values or adapter metadata to construct its expected answer.

The native estimate normalizes each cross-time matrix by the same draw-level mean receive power. Its standard errors use the paired ratio-estimator influence value `cross - estimate*power_draw`; this preserves the within-draw dependence among time and antenna coordinates.

## Seed audit

The pre-run collision audit is `G4_CDL_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-09.json`. It scans the saved research-results, outputs, and artifacts corpus for the frozen ten-seed roster. Clearance is corpus-scoped only and never establishes global seed exclusivity.

## Interpretation limits

A pass supports only the five pinned CDL profiles, this two-element zero-oriented omni receive array, the two declared velocity vectors, one carrier, one speed, and one lag. It does not establish continuous orientation or velocity coverage, arbitrary arrays or element patterns, mutual coupling, multiport transmit covariance, measured-channel truth, RT/material truth, receiver qualification, overall G4 qualification, production D1 authorization, or advanced-training authorization.
