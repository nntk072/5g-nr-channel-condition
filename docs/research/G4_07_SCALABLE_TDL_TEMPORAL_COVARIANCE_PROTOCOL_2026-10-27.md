# G4.07 Scalable TDL temporal covariance protocol

## Frozen scope

- Profiles: scalable TDL-A through TDL-E, TR 38.901 V19.2.0.
- Independent component reference: `G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json`, SHA-256 `38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd`, bound to TR 38.901 source SHA-256 `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.
- Native implementation: Sionna 2.2.0 CPU, one TX and RX port, 8 subcarriers at 30 kHz spacing, 100 ns delay spread, 3.5 GHz carrier, 30 m/s fixed speed, 20 sinusoids, five time samples spaced by `1/(8 f_D,max)`, and 32,768 independent channel draws/profile.
- Sampling unit: independent channel draw. Multiple taps and time-lag coordinates from one draw are not treated as independent replicates.
- Included components: all 95 ETSI-labeled Rayleigh taps across A–E. For TDL-D/E, omit only tap zero, where Sionna merges the table's deterministic LOS and zero-delay diffuse terms.
- Lags: `1/(8 f_D,max)`, `1/(4 f_D,max)`, and `1/(2 f_D,max)`, corresponding to sample indices 1, 2, and 4. The independent reference for each lag is isotropic-scattering `J0(2π f_D,max lag)`.
- Family: real and imaginary parts of normalized complex covariance residual for each tap and each lag, for 570 simultaneous coordinates. For draw `i`, tap `k`, and lag `l`, residual is `conj(h[i,k,0]) * h[i,k,l] - J0(2π f_D,max l) * |h[i,k,0]|²`. Its sample mean and standard error are normalized by the sample mean initial tap power.
- Acceptance: Bonferroni two-sided normal intervals at family-wise alpha 0.01 must each lie entirely inside `[-0.05, +0.05]` normalized by mean tap power. This is a bounded regression criterion, not ground truth.
- Seeds: `1_741_000 + profile_index` for A–E. The pre-run collision report is `G4_SCALABLE_TDL_TEMPORAL_SEED_AUDIT_2026-10-27.json`, SHA-256 `7c94438bc6380d99f0d6ffe66e83cb5eb1303df0ad32d7a9b705c42f75af9a1c`. It scanned 2,273 JSON/JSON.GZ files under `docs/research/results`, `outputs`, and `artifacts`; there were no collisions, malformed seed fields, or incomplete files. This is clearance only within that corpus snapshot, not global exclusivity.

For TDL-D/E, the test does not apply a zero-mean Rayleigh covariance criterion to the merged LOS tap. The frozen reference roster maps only later Rayleigh components to native tap indices.

## Interpretation and reproduction

The test checks three points of the standardized diffuse temporal covariance for each scalable profile and each independently table-labeled Rayleigh tap. It does not establish the complete Doppler spectrum, inter-tap dependence, array/spatial behavior, measured propagation validity, receiver impact, or G4 qualification. Native statistical generation has not been run as part of protocol implementation.

After code implementation and consolidated validation, run:

```bash
uv run --locked python -m pytest -q -m native_sionna \
  tests/test_integrations/test_g4_scalable_tdl_temporal_reference_native.py
```
