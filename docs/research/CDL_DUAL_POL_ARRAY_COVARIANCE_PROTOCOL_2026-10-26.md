# G4.07 CDL Dual-Polarized Array Covariance Protocol

## Frozen scope

- Profiles: NLOS CDL-A, CDL-B, and CDL-C, TR 38.901 v19.2.
- Topology: one vertical single-polarized TX port and a two-element, half-wavelength UT ULA with vertical/horizontal ports at each element.
- Observable: the zero-frequency, cluster-summed 4-by-4 receive-channel second-moment matrix, with ports in V-element0, V-element1, H-element0, H-element1 order.
- Independent reference: ETSI-derived cluster powers, mean AoA/ZoA, angular spreads, and XPR from `G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`; Table 7.5-3 ray offsets and independent azimuth/zenith coupling. Sionna's packaged parameter-table bytes are checked only as system-under-test provenance.
- Seed roster: `1700000`, `1700001`, and `1700002`, assigned in A/B/C order. Run the structured G4 seed-collision audit before executing this family.
- Sampling unit: one independently generated stochastic channel draw. Use 8,192 draws/profile.
- Inference: one two-sided Bonferroni 99% family across all 16 unique real Hermitian components and all three profiles (48 claims). Each interval must contain its independently derived reference and have half-width no greater than 0.05 normalized mean power.

For NLOS, cross-polarization phases are independent, so the expected cross-polar receive covariance blocks are zero. Each same-polarization block averages its field-power-weighted spatial phase over the Cartesian product of the independently coupled 20-ray azimuth and zenith sets. V/H field powers use the TR 38.901 polarization-basis rotation at the frozen zero array orientation, with the H response using the complementary co/cross-polar terms and the profile's inverse linear XPR.

## Limits and interpretation

This family addresses one finite multi-element and dual-polarization covariance gap for the three named NLOS profiles. It does not cover LOS CDL-D/E, arbitrary arrays or orientations, element-pattern variation, array coupling beyond the specified feed model, time evolution, geometric RT, measured channels, or statistical qualification of the complete G4 family. A passing regression is bounded standardized-model consistency evidence only.
