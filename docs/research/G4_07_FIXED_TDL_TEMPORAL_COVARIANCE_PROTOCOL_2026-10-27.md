# G4.07 Fixed TDL temporal covariance protocol

## Frozen scope

- Profiles: fixed TDL-A10, A30, B100, C60, C300, D10, and D30 from TS 38.101-4 V19.2.2 Annex B.2.1.
- Independent component reference: `G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json`, SHA-256 `fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3`, bound to TS 38.101-4 source SHA-256 `b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81`.
- Native implementation: Sionna 2.2.0 CPU, one TX and RX port, 8 subcarriers at 30 kHz spacing, 3.5 GHz carrier, 30 m/s fixed speed, 20 sinusoids, five time samples spaced by `1/(8 f_D,max)`, and 32,768 independent channel draws/profile.
- Sampling unit: independent channel draw. Taps and lag coordinates from one draw are not independent replicates.
- Included components: every delay cluster whose ETSI rows are all labeled Rayleigh. This gives 82 taps across the seven profiles. The D10/D30 zero-delay tap is excluded because the table assigns both a LOS component and a Rayleigh component to the merged tap.
- Lags: `1/(8 f_D,max)`, `1/(4 f_D,max)`, and `1/(2 f_D,max)`, corresponding to sample indices 1, 2, and 4. The reference for each lag is isotropic-scattering `J0(2π f_D,max lag)`.
- Family: real and imaginary coordinates of normalized complex covariance residual for each included tap and lag, totaling 492 simultaneous coordinates. For draw `i`, tap `k`, and lag `l`, residual is `conj(h[i,k,0]) * h[i,k,l] - J0(2π f_D,max l) * |h[i,k,0]|²`; sample mean and standard error are normalized by mean initial tap power.
- Acceptance: Bonferroni two-sided normal intervals at family-wise alpha 0.01 must each lie entirely within `[-0.05, +0.05]` normalized by mean tap power. This is a bounded regression criterion, not a standards tolerance or ground truth.
- Seeds: `1_750_000 + profile_index` for profiles ordered A10, A30, B100, C60, C300, D10, D30. The pre-run collision report is `G4_FIXED_TDL_TEMPORAL_SEED_AUDIT_2026-10-27.json`, SHA-256 `ffffcd0b526c14bd62f38a735b678c5b91311fb6918e0ac641e83fc524b32d43`. It scanned 2,275 JSON/JSON.GZ files under `docs/research/results`, `outputs`, and `artifacts`; there were no collisions, malformed seed fields, or incomplete files. This is clearance only within that corpus snapshot, not global exclusivity.

The regression derives the Rayleigh tap roster from the independent component table and requires all components assigned to a cluster to be Rayleigh. It does not infer fading class from Sionna metadata.

## Interpretation and reproduction

This checks three temporal covariance points of the named diffuse components. It does not establish the full Doppler spectrum, cross-tap dependence, Rician LOS temporal behavior, spatial behavior, measurement validity, receiver impact, or G4 qualification. Native generation was not run while defining the protocol.

After code implementation and consolidated validation, run:

```bash
uv run --locked python -m pytest -q -m native_sionna \
  tests/test_integrations/test_g4_fixed_tdl_temporal_reference_native.py
```
