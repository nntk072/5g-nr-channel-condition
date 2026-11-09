# G4.08 CDL Rotated Dual-TX/Dual-RX Joint Space-Time Covariance Protocol

**Status:** frozen bounded native protocol. Passing is finite Sionna/TR 38.901 implementation-consistency evidence for this family only and is not overall G4 qualification.

## Scientific gap

Retained G4 evidence already covers zero-oriented dual-polarized TX/RX joint space-time covariance and discrete receive-side rotations. The remaining bounded gap addressed here is simultaneous nonzero TX and RX orientation with dual-VH arrays at both endpoints, including transmit spatial phase, receive spatial phase, endpoint polarization-basis rotation, Step-8 angular coupling, and receive-side Doppler in the same cross-time covariance observable.

## Frozen design

- Profiles: CDL-A through CDL-E, TR 38.901 v19.2.0.
- Runtime: Sionna PHY 2.2.0, CPU.
- Carrier: 3.5 GHz; delay spread: 100 ns; SCS: 30 kHz; eight subcarriers.
- Physical topology: two TX elements and two RX elements.
- Every physical element exposes dual-VH omnidirectional ports in one panel with half-wavelength local-y spacing.
- Native one-panel dual-VH port order is pattern-major: V0, V1, H0, H1 at each endpoint.
- Native channel axes are batch, rx, rx_ant, tx, tx_ant, cluster, time; flattened channel ports are RX-port-major / TX-port-minor.
- Frozen endpoint-orientation roster, listed as `(TX alpha,beta,gamma) -> (RX alpha,beta,gamma)` in radians:
  1. `(-0.22, 0.31, 0.47) -> (0.37, 0.28, -0.41)`
  2. `(0.41, -0.27, 0.0) -> (0.0, 0.0, 0.65)`
  3. `(0.0, -0.33, -0.52) -> (0.28, -0.36, 0.0)`
- Velocity: (10,20,20) m/s, magnitude 30 m/s, at the receive endpoint.
- Lag: 0.7 ms represented by two native time samples.
- Draws: 16,384 independent stochastic channel draws/case.
- Seeds: 2,816,000 through 2,816,014, ordered by profile A-E and then orientation roster.
- Observable: raw `E[h_p(0)^* h_q(tau)]`; no per-draw or post-hoc normalization is permitted.
- Channel-port count: 16; each case contains 512 real/imaginary claims.
- Claims: every real and imaginary component of every frozen cross-time matrix. The family contains 7,680 simultaneous claims.
- Family-wise alpha: 0.01, one two-sided Bonferroni normal family over all 7,680 claims.
- Precision ceiling: every simultaneous interval half-width must be at most 0.05 in the standardized co-polar channel-power units.
- Frozen failures are retained. Seeds, draw count, orientation roster, topology, geometry, polarization, lag, alpha, claim family, observable, and precision ceiling must not be changed after observing native results.

## Independent reference

`evaluation/g4_cdl_rotated_dual_tx_rx_spatiotemporal_covariance.py` is the preregistered reference. It does not read adapter metadata and is based on the digest-pinned ETSI-derived CDL tap/angle fixture plus analytic TR 38.901 field, array, polarization, and Doppler equations.

For omni dual-VH ports, each endpoint maps the local V/H basis into the GCS through the TR 38.901 displacement angle `psi`. Diffuse channel coefficients are represented as `r^T J t`, where `r` and `t` are the rotated receive and transmit field vectors and the four entries of `J` have independent Step-10 phases. Their variances are 1 for co-polar entries and `1/XPR` for cross-polar entries.

Sionna's CDL Step-8 implementation independently shuffles AoA, AoD, ZoA, and ZoD within each cluster for each stochastic draw. The ensemble reference therefore averages receive azimuth/zenith and transmit azimuth/zenith combinations independently. This permits exact endpoint moment factorization without assuming a fixed ray pairing. The Sionna `cdl.py` implementation hash is pinned by the native test and retained result because this coupling behavior is part of the bounded comparison contract.

Rotated physical panel positions use the forward TR 38.901 orientation matrix. With a one-row panel, element differences lie along the rotated local-y direction. Receive phase combines this spatial term with Doppler; transmit phase contains the rotated departure spatial term. CDL-D/E LOS uses the deterministic Step-11 `diag(1,-1)` Jones matrix together with both endpoint field transforms, both spatial offsets, and receive-side Doppler.

The pinned ETSI-derived tap/angle fixture is `docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`, SHA-256 `24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090`, sourced from ETSI bytes SHA-256 `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.

## Pre-freeze oracle validation

Before this protocol was frozen:

- 29 targeted portable/deterministic tests passed with coverage disabled after an initial narrow-test invocation was rejected only by the repository-wide coverage threshold.
- Six deterministic Sionna Step-11 checks across three TX/RX orientation pairs and two angle tuples matched the independent field-transform and rotated-array-offset equations to numerical precision.
- A non-admissible 4,096-draw diagnostic at the first orientation pair produced maximum standardized discrepancies of 2.84 sigma for CDL-A and 2.43 sigma for CDL-D. Those diagnostic seeds and estimates are not part of this retained protocol and cannot replace the frozen execution.

## Seed audit

`G4_CDL_ROTATED_DUAL_TX_RX_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json` clears the complete frozen seed roster against the saved `docs/research/results`, `outputs`, and `artifacts` JSON corpora before native execution. The audit is corpus-scoped and does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned CDL profiles, three frozen simultaneous TX/RX orientations, one 2x2 physical dual-VH omni topology, one velocity, one carrier, and one lag. It does not establish continuous orientation or velocity coverage, arbitrary endpoint topologies, patterned arrays, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, general 3GPP conformance, overall G4 qualification, production D1 authorization, or advanced-training authorization.
