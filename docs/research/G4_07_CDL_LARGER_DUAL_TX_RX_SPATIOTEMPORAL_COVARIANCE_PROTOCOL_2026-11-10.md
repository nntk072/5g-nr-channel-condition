# G4.07 CDL Larger Dual-TX/Dual-RX Joint Space-Time Covariance Protocol

**Status:** frozen bounded native protocol. Passing is finite Sionna/TR 38.901 implementation-consistency evidence for this family only and is not overall G4 qualification.

## Scientific gap

The retained dual-TX/dual-RX joint space-time covariance family covers two physical dual-VH elements at each endpoint. The implementation feature map explicitly leaves larger dual-polarized topologies open. This protocol extends the same independent zero-orientation oracle to larger endpoint rosters without changing the previously validated equations.

## Frozen design

- Profiles: CDL-A through CDL-E, TR 38.901 v19.2.0.
- Runtime: Sionna PHY 2.2.0, CPU.
- Carrier: 3.5 GHz; delay spread: 100 ns; SCS: 30 kHz; eight subcarriers.
- Endpoint topologies, expressed as physical (TX,RX) elements: (4,2), (2,4), (3,3).
- Every physical element exposes dual-VH omnidirectional ports in one zero-oriented panel, with half-wavelength local-y spacing.
- Native one-panel dual-VH port order is pattern-major: all V physical elements followed by all H physical elements.
- Native channel axes are batch, rx, rx_ant, tx, tx_ant, cluster, time; flattened channel ports are RX-port-major / TX-port-minor.
- Flattened channel-port counts: 32 for (4,2), 32 for (2,4), and 36 for (3,3).
- Velocity: (10,20,20) m/s, magnitude 30 m/s.
- Lag: 0.7 ms represented by two native time samples.
- Draws: 16,384 independent stochastic channel draws/case.
- Seeds: 2,815,000 through 2,815,014, ordered by profile A-E and then topology roster.
- Observable: raw E[h_p(0)^* h_q(tau)]; no per-draw or post-hoc normalization is permitted.
- Claims: every real and imaginary component of every frozen cross-time matrix. The family contains 33,440 simultaneous claims.
- Family-wise alpha: 0.01, one two-sided Bonferroni normal family over all 33,440 claims.
- Precision ceiling: every simultaneous interval half-width must be at most 0.05 in the standardized co-polar channel-power units.
- Frozen failures are retained. Seeds, draw count, topology roster, geometry, polarization, lag, alpha, claim family, observable, and precision ceiling must not be changed after observing native results.

## Independent reference

evaluation/g4_cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance.py is reused unchanged. It is already parameterized by physical TX/RX element counts and uses only the digest-pinned ETSI-derived CDL component fixture plus analytic TR 38.901 polarization, spatial, and Doppler laws.

At zero endpoint orientation, diffuse Jones entries (VV,VH,HV,HH) have independent Step-10 phases. Same-entry covariance therefore carries power 1 for co-polar entries and 1/XPR for cross-polar entries, while covariance across distinct diffuse Jones entries is zero. Receive spatial phase and Doppler are evaluated jointly per ray; transmit spatial phase is evaluated per departure ray. CDL-D/E specular LOS uses the frozen signed diag(1,-1) polarization contribution.

The pinned ETSI-derived tap/angle fixture is docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json, SHA-256 24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090, sourced from ETSI bytes SHA-256 b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4.

## Seed audit

G4_CDL_LARGER_DUAL_TX_RX_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json must clear the complete frozen seed roster against the saved docs/research/results, outputs, and artifacts JSON corpora before native execution. The audit is corpus-scoped and does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned CDL profiles, the three frozen zero-oriented dual-VH omni topology pairs, one velocity, one carrier, and one lag. It does not establish arbitrary or larger arrays beyond this roster, rotated or patterned dual-endpoint arrays, continuous orientation or mobility coverage, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, general 3GPP conformance, overall G4 qualification, production D1 authorization, or advanced-training authorization.
