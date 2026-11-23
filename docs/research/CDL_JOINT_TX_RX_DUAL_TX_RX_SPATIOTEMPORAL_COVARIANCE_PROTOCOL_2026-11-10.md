# G4.07 CDL Dual-TX/Dual-RX Joint Space-Time Covariance Protocol

**Status:** frozen bounded native protocol. Passing is finite Sionna/TR 38.901 implementation-consistency evidence for this family only and is not overall G4 qualification.

## Scientific gap

Retained G4 evidence covers single-polarization joint TX/RX space-time covariance and a single-V transmit / dual-VH receive space-time family. The remaining bounded interaction gap is dual-polarized transmission and reception in the same full cross-time channel-port covariance observable.

Before freezing this protocol, the independent oracle was exercised by portable reduction, Hermitian, XPR-scaling, LOS-sign, finiteness, and invalid-input checks. A native structural regression also established the one-panel two-element Sionna dual-VH port order as `V0,V1,H0,H1` through `PanelArray.ant_ind_pol1`, `PanelArray.ant_ind_pol2`, and co-located antenna positions.

## Frozen design

- Profiles: CDL-A through CDL-E, TR 38.901 v19.2.0.
- Runtime: Sionna PHY 2.2.0, CPU.
- Carrier: 3.5 GHz; delay spread: 100 ns; SCS: 30 kHz; eight subcarriers.
- TX array: one zero-oriented panel with two local-y half-wavelength physical elements, dual-VH omnidirectional ports. Native TX port order is `V0,V1,H0,H1`.
- RX array: one zero-oriented panel with two local-y half-wavelength physical elements, dual-VH omnidirectional ports. Native RX port order is `V0,V1,H0,H1`.
- Native channel axes: `[batch, rx, rx_ant, tx, tx_ant, cluster, time]`; channel ports are flattened RX-port-major / TX-port-minor after selecting the single RX and TX links.
- Flattened channel-port count: 16.
- Velocity: `(10,20,20)` m/s, with magnitude 30 m/s.
- Lag: 0.7 ms represented by two native time samples.
- Draws: 16,384 independent stochastic channel draws/profile.
- Seeds: 2,814,000 through 2,814,004 in profile order A-E.
- Observable: raw `E[h_p(0)^* h_q(tau)]`; no per-draw or post-hoc normalization is permitted.
- Claims: all real and imaginary components of the 16x16 cross-time matrix, 512 claims/profile and 2,560 simultaneous claims total.
- Family-wise alpha: 0.01, one two-sided Bonferroni normal family over all 2,560 claims.
- Precision ceiling: every simultaneous interval half-width must be at most 0.05 in the standardized co-polar channel-power units.
- Frozen failures are retained. Seeds, draw count, geometry, polarization, lag, alpha, claim family, observable, and precision ceiling must not be changed after observing native results.

## Independent reference

`evaluation/g4_cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance.py` uses only the digest-pinned ETSI-derived CDL component fixture and analytic TR 38.901/Sionna polarization laws; generated channel coefficients are never used to construct the reference.

At zero endpoint orientation with dual-VH omni elements, each diffuse ray selects one of the four Jones entries `(VV,VH,HV,HH)`. Sionna PHY 2.2.0 implements four independent Step-10 random phases. Therefore ensemble covariance between different Jones entries is zero. Same-entry covariance uses coefficient power `1` for `VV` and `HH`, and `1/XPR` for `VH` and `HV`. For the same Jones entry, the already-established spatial/Doppler expectation is reused: the receive spatial-plus-Doppler mean times the transmit spatial mean for each CDL component.

For CDL-D/E specular LOS, the pinned Sionna PHY 2.2.0 LOS Jones matrix is `diag(1,-1)` at the frozen zero endpoint orientations. Consequently the deterministic specular contribution survives in `VV<->VV`, `HH<->HH`, and signed `VV<->HH` / `HH<->VV` covariance, while `VH` and `HV` have zero specular contribution. Diffuse and specular powers are normalized jointly using the pinned ETSI-derived component powers, which is equivalent to the frozen CDL Rician K-factor scaling.

The installed source files that define the port and polarization contract are pinned by SHA-256 in the native test:

- `sionna/phy/channel/tr38901/antenna.py`: `09e20b89f4c4d990bed1a3ce3d6314f9fe7e45c090737a6f5790166d4a63f926`
- `sionna/phy/channel/tr38901/channel_coefficients.py`: `47059c7bf0998d2a230909e70743832ee928a738a97b9b5a25a185bb4947b5fc`

The pinned ETSI-derived tap/angle fixture remains `docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`, SHA-256 `24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090`, sourced from ETSI bytes SHA-256 `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.

## Seed audit

`G4_CDL_JOINT_TX_RX_DUAL_TX_RX_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json` scanned the saved `docs/research/results`, `outputs`, and `artifacts` JSON corpora before protocol freeze. It scanned 2,295 eligible artifacts and found zero collisions, zero malformed seed fields, and zero incomplete files for seeds 2,814,000-2,814,004. The audit is corpus-scoped and does not establish global seed exclusivity.

## Interpretation limits

A pass applies only to the five pinned CDL profiles, this zero-oriented two-physical-element dual-VH TX / dual-VH RX half-wavelength omni configuration, one velocity, one carrier, and one lag. It does not establish arbitrary arrays or patterns, continuous orientation or mobility coverage, mutual coupling, measured-channel truth, RT/material truth, receiver qualification, general 3GPP conformance, overall G4 qualification, production D1 authorization, or advanced-training authorization.
