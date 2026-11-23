# G4.07 CDL Joint TX/RX Space-Time Covariance Protocol

**Status:** frozen bounded native protocol. Passing is finite Sionna/TR 38.901 implementation-consistency evidence and is not overall G4 qualification.

## Scientific gap

The retained G4 evidence separately covers joint TX/RX spatial covariance and single-TX two-element receive space-time covariance. This protocol couples both link-end array phases with Doppler in the same observable by checking the full four-port cross-time matrix of a 2x2 link.

## Frozen design

- Profiles: CDL-A through CDL-E, TR 38.901 v19.2.0.
- Runtime: Sionna PHY 2.2.0, CPU.
- Carrier: 3.5 GHz; delay spread: 100 ns; SCS: 30 kHz; eight subcarriers.
- Arrays: zero-oriented local-y half-wavelength ULAs with two single-V omnidirectional elements at both UT and BS.
- Velocity: `(10,20,20)` m/s (30 m/s magnitude).
- Lag: 0.7 ms, represented by two native time samples.
- Draws: 16,384 independent channel draws/profile.
- Seeds: 2,811,000 through 2,811,004 in profile order A–E.
- Port ordering: RX-major/TX-minor, yielding four complex ports.
- Claims: all real and imaginary components of `E[h_p(0)^* h_q(tau)]`, 32 claims/profile and 160 simultaneous claims total.
- Family-wise alpha: 0.01, one two-sided Bonferroni normal family over all 160 claims.
- Precision ceiling: every interval half-width must be at most 0.05 normalized mean power.
- Frozen failures are retained. Seeds, draw counts, geometry, lag, alpha, claim family, and precision ceiling must not be changed after observing native results.

## Independent reference

`evaluation/g4_cdl_joint_tx_rx_spatiotemporal_covariance.py` uses only the exact-digest-pinned ETSI-derived CDL component fixture. For flattened port `(rx,tx)`, the `h_p(0)^* h_q(tau)` phase combines receive spatial separation, transmit spatial separation, and receive Doppler:

`2*pi*d_rx*Delta_rx*u_rx,y + 2*pi*d_tx*Delta_tx*u_tx,y + 2*pi*f_c*tau*(v dot u_rx)/c`.

For diffuse clusters, departure and arrival angle couplings are independent under the pinned CDL ray construction, so the cluster expectation is the product of the departure spatial mean and the joint arrival-space/Doppler mean. Each endpoint mean averages the Cartesian product of the standard 20 azimuth and 20 zenith offsets. CDL-D/E specular components remain separate deterministic directions with their independently transcribed powers.

The native estimate uses the same draw-level mean four-port receive power as the denominator for every matrix component. Ratio-estimator influence values preserve the within-draw dependence across ports and time.

## Seed audit

`G4_CDL_JOINT_TX_RX_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-09.json` clears the five frozen seeds only within the saved research-results/outputs/artifacts corpus snapshot. It does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned profiles, this zero-oriented 2x2 single-V omni half-wavelength link, one velocity, one speed, one carrier and one lag. It does not establish larger arrays, continuous orientation/velocity coverage, directional patterns, dual-polarization behavior, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, overall G4 qualification, production D1 authorization, or advanced-training authorization.
