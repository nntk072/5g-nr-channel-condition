# G4.07 CDL Joint TX/RX Dual-Polarization Space-Time Covariance Protocol

**Status:** frozen bounded native protocol. Passing is finite Sionna/TR 38.901 implementation-consistency evidence and is not overall G4 qualification.

## Scientific gap

Retained G4 evidence separately covers static joint TX/RX spatial covariance, dual-VH receive-array covariance, and single-polarization joint TX/RX space-time covariance. This protocol checks their interaction in one observable: a two-element transmit array, a dual-VH two-element receive array, and Doppler evolution in the full cross-time channel-port covariance matrix.

## Frozen design

- Profiles: CDL-A through CDL-E, TR 38.901 v19.2.0.
- Runtime: Sionna PHY 2.2.0, CPU.
- Carrier: 3.5 GHz; delay spread: 100 ns; SCS: 30 kHz; eight subcarriers.
- TX array: zero-oriented local-y half-wavelength ULA with two single-V omnidirectional elements.
- RX array: zero-oriented local-y half-wavelength ULA with two element positions and dual-VH omnidirectional ports, yielding four receive ports in the frozen Sionna ordering \`V0,V1,H0,H1\`.
- Velocity: \`(10,20,20)\` m/s (30 m/s magnitude).
- Lag: 0.7 ms, represented by two native time samples.
- Draws: 16,384 independent channel draws/profile.
- Seeds: 2,812,000 through 2,812,004 in profile order A-E.
- Channel-port ordering: receive-port-major / transmit-element-minor. The eight ports are \`(V0,TX0)\`, \`(V0,TX1)\`, \`(V1,TX0)\`, \`(V1,TX1)\`, \`(H0,TX0)\`, \`(H0,TX1)\`, \`(H1,TX0)\`, \`(H1,TX1)\`.
- Observable: raw \`E[h_p(0)^* h_q(tau)]\`; there is no per-draw or post-hoc renormalization. This preserves the XPR-dependent horizontal-port power.
- Claims: all real and imaginary components of the 8x8 cross-time matrix, 128 claims/profile and 640 simultaneous claims total.
- Family-wise alpha: 0.01, one two-sided Bonferroni normal family over all 640 claims.
- Precision ceiling: every interval half-width must be at most 0.05 in the standardized channel's normalized vertical-port power units.
- Frozen failures are retained. Seeds, draw counts, geometry, polarization, lag, alpha, claim family, observable, and precision ceiling must not be changed after observing native results.

## Independent reference

\`evaluation/g4_cdl_joint_tx_rx_dual_pol_spatiotemporal_covariance.py\` is independent of generated channel coefficients and uses the exact-digest-pinned ETSI-derived CDL component fixture.

The vertical/vertical block deliberately reuses the already-established independent single-V joint TX/RX space-time oracle. For a diffuse cluster, the horizontal/horizontal receive block uses the same transmit spatial phase and receive spatial-plus-Doppler phase but scales the component power by \`1/XPR\`, as required for the cross-polar term from a single-V transmitter at the frozen zero endpoint orientations. The V/H and H/V ensemble blocks are zero because the pinned TR 38.901 ray construction assigns independent polarization phases. For CDL-D/E, the deterministic LOS component is V-only at the frozen zero orientations; the horizontal block therefore contains diffuse power only.

For flattened physical element pair \`(rx,tx)\`, the spatial/Doppler phase is:

\`2*pi*d_rx*Delta_rx*u_rx,y + 2*pi*d_tx*Delta_tx*u_tx,y + 2*pi*f_c*tau*(v dot u_rx)/c\`.

Diffuse departure and arrival angle couplings are independently averaged over the Cartesian products of the standard 20 azimuth and 20 zenith ray offsets. Profile powers and XPR come only from the pinned ETSI-derived fixture.

## Seed audit

\`G4_CDL_JOINT_TX_RX_DUAL_POL_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-09.json\` must clear the five frozen seeds within the saved \`docs/research/results\`, \`outputs\`, and \`artifacts\` corpus snapshot before native execution. This audit is corpus-scoped and does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned CDL profiles, this zero-oriented two-element single-V TX / dual-VH two-element RX half-wavelength omni configuration, one velocity, one speed, one carrier, and one lag. It does not establish dual-polarized transmission, arbitrary array geometry or element patterns, continuous orientation/velocity coverage, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, overall G4 qualification, production D1 authorization, or advanced-training authorization.