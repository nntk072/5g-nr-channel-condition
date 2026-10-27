# G4.07 Scalable TDL Rayleigh Component Properness Protocol

## Frozen scope

- Profiles: scalable TDL-A through TDL-E, TR 38.901 V19.2.0.
- Independent component reference: `G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json`, SHA-256 `38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd`, bound to TR 38.901 source SHA-256 `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.
- Native implementation: Sionna 2.2.0 CPU TDL adapter, one RX port, one TX port, 24 subcarriers at 30 kHz spacing, 100 ns delay spread, 3.5 GHz carrier, one time sample, and 32,768 independent batch draws/profile.
- Sampling unit: one independent channel draw. Taps within one draw are not treated as independent replicates.
- Frozen seed rule: `1_740_000 + profile_index` for profiles ordered `A, B, C, D, E`. The seed roster audit ran before this protocol was frozen: `G4_SCALABLE_TDL_PROPERNESS_SEED_AUDIT_2026-10-27.json`, SHA-256 `e2cb2de0141253d9e295cea201dd7502488931659f27ad44084922338db9142f`. It scanned 2,273 JSON/JSON.GZ files under `docs/research/results`, `outputs`, and `artifacts`, with no collisions, malformed seed fields, or incomplete files. This is clearance only within that saved corpus snapshot, not global exclusivity.
- Family: both real coordinates of normalized complex mean and both real coordinates of normalized complex pseudocovariance for every reference-labeled Rayleigh tap. The roster contains 95 taps and 380 simultaneous two-sided claims. Bonferroni family-wise alpha is 0.01; every interval must contain zero and every coordinate half-width must be at most 0.05 normalized by tap RMS amplitude.
- For TDL-D/E, the zero-delay diffuse Rayleigh component is merged with a deterministic LOS component in adapter tap zero; that Ricean aggregate is excluded from the zero-mean/proper-complex family. All later ETSI-labeled Rayleigh taps remain included.

## Interpretation

The ETSI table identifies the included components as Rayleigh. Zero mean and zero pseudocovariance test the project's complex-baseband realization of those components. These are bounded software/model checks for the named profiles and configuration. They do not establish the full Doppler spectrum, inter-tap dependence, spatial behavior, physical propagation, receiver impact, or G4 qualification, and no finite draw count is ground truth.

## Reproduction

After the seed audit above, run the frozen native regression with:

```bash
uv run --locked python -m pytest -q -m native_sionna \
  tests/test_integrations/test_g4_scalable_tdl_complex_circularity_native.py
```

This protocol was authored without executing the statistical generation.
