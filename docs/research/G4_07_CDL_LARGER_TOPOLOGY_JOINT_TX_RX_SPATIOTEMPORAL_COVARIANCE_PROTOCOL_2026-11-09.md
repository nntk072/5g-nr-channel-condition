# G4.07 CDL Larger-Topology Joint TX/RX Space-Time Covariance Protocol

**Status:** frozen bounded native protocol. Passing is finite Sionna/TR 38.901 implementation-consistency evidence and is not overall G4 qualification.

## Scientific gap

The retained joint TX/RX space-time covariance family exercises a 2x2 single-polarization link, while the retained joint spatial covariance family separately exercises 4x4, 1x4, and 4x1 topologies. This protocol extends the same independent space-time oracle to those larger and asymmetric single-V topologies so transmit phase, receive phase, and Doppler are checked in one observable beyond the 2x2 case.

This is a separate bounded family. It does not overwrite, reinterpret, or repair the preserved 2x2 frozen failure, and it does not alter the later dual-VH receive-family result.

## Frozen design

- Profiles: CDL-A through CDL-E, TR 38.901 v19.2.0.
- Runtime: Sionna PHY 2.2.0, CPU.
- Carrier: 3.5 GHz; delay spread: 100 ns; SCS: 30 kHz; eight subcarriers.
- Arrays: zero-oriented local-y half-wavelength ULAs with single-V omnidirectional elements at both link ends.
- Topologies, expressed as `(TX elements, RX elements)`: `(4,4)`, `(1,4)`, and `(4,1)`.
- Velocity: `(10,20,20)` m/s (30 m/s magnitude).
- Lag: 0.7 ms, represented by two native time samples.
- Draws: 16,384 independent channel draws/case.
- Cases: 15 total, ordered profile-major A-E and topology-minor `(4,4)`, `(1,4)`, `(4,1)`.
- Seeds: 2,813,000 through 2,813,014 in that frozen case order.
- Port ordering: RX-major/TX-minor.
- Claims: every real and imaginary component of `E[h_p(0)^* h_q(tau)]` for every ordered port pair.
  - 4x4: 16 ports, 512 claims/case.
  - 1x4: 4 ports, 32 claims/case.
  - 4x1: 4 ports, 32 claims/case.
  - Total family: 2,880 simultaneous claims.
- Family-wise alpha: 0.01, one two-sided Bonferroni normal family over all 2,880 claims.
- Precision ceiling: every interval half-width must be at most 0.05 normalized mean power.
- Frozen failures are retained. Seeds, draw counts, topology roster, geometry, lag, alpha, claim family, and precision ceiling must not be changed after observing native results.

## Independent reference

`evaluation/g4_cdl_joint_tx_rx_spatiotemporal_covariance.py` is already parameterized by transmit and receive element count and uses only the exact-digest-pinned ETSI-derived CDL component fixture. For flattened port `(rx,tx)`, the `h_p(0)^* h_q(tau)` phase combines receive spatial separation, transmit spatial separation, and receive Doppler:

`2*pi*d_rx*Delta_rx*u_rx,y + 2*pi*d_tx*Delta_tx*u_tx,y + 2*pi*f_c*tau*(v dot u_rx)/c`.

For diffuse clusters, departure and arrival ray couplings are independent under the pinned CDL ray construction, so the cluster expectation is the product of the departure spatial mean and the joint arrival-space/Doppler mean. Each endpoint mean averages the Cartesian product of the standard 20 azimuth and 20 zenith offsets. CDL-D/E specular components remain separate deterministic directions with their independently transcribed powers.

The native estimator uses the draw-level mean channel power across all flattened ports as the common denominator. Ratio-estimator influence values preserve within-draw dependence across ports and time. The same estimator definition is used for all three topologies.

## Seed audit

`G4_CDL_LARGER_TOPOLOGY_JOINT_TX_RX_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-09.json` clears the 15 frozen seeds only within the saved `docs/research/results`, `outputs`, and `artifacts` JSON corpus snapshot. The audit scanned 2,294 eligible saved JSON artifacts and found no collisions, malformed seed fields, or incomplete files. It does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned profiles, these three zero-oriented single-V omni half-wavelength topologies, one velocity, one speed, one carrier, and one lag. It does not establish arbitrary or larger arrays, continuous orientation or velocity coverage, directional patterns, dual-polarized transmission or reception, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, overall G4 qualification, production D1 authorization, or advanced-training authorization.
