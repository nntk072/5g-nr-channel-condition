# G4.07 CDL-D/E LOS Dual-Polarized Array Covariance Protocol

## Frozen scope

- Profiles: LOS CDL-D and CDL-E, TR 38.901 v19.2.
- Topology: one vertical single-polarized TX port and a two-element, half-wavelength UT ULA with vertical/horizontal ports at each element.
- Endpoint orientations: zero at both ends; omni element patterns. This deliberately fixes the LOS polarization basis and does not claim rotated-LOS coverage.
- Observable: the zero-frequency, cluster-summed 4-by-4 receive-channel second-moment matrix, with ports ordered V-element0, V-element1, H-element0, H-element1.
- Independent inputs: ETSI-derived component powers/angles/spreads in `G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`, the independently transcribed LOS K factor in `G4_CDL_PROFILE_SCALAR_REFERENCE_2026-10-26.json`, and Table 7.5-3 ray offsets. Their SHA-256 digests are frozen in the regression. Sionna's packaged table hash is checked only as system-under-test provenance.
- Seed roster: `1710000` for D and `1710001` for E. Run the structured G4 seed-collision audit before executing this family.
- Sampling unit: one independently generated stochastic channel draw; 16,384 draws/profile.
- Inference: one two-sided Bonferroni 99% family across all 16 unique real Hermitian components and both profiles (32 claims). Every interval must contain its independent reference and have half-width at most 0.05 normalized mean power.

## Independent expectation

The pinned CDL table transcription provides the specular-to-diffuse power ratio. The reference uses `K/(K+1)` for the deterministic LOS term and `1/(K+1)` for the diffuse term. Diffuse cluster powers are normalized over diffuse clusters; each cluster's same-polarization spatial covariance is the Cartesian average over the 20 azimuth and 20 zenith rays. At zero endpoint orientation, the single-V transmitter and LOS basis yield a V-only deterministic receive vector. Its outer product is combined with the diffuse covariance. Independent cross-polar phases make the diffuse V/H covariance blocks zero; the H/H diffuse component uses the inverse linear XPR.

## Limits

This is a bounded standardized-model regression definition, not statistical qualification of G4. It does not cover rotated LOS polarization, other array sizes/geometries, directional element patterns, mutual coupling, multiport TX, temporal evolution, RT, or measured channels. The test and protocol are authored but have not been executed; no result is claimed. Passing this finite family cannot authorize production RT corpus generation.
