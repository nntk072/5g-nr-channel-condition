# G4.07 Scalable TDL Doppler-spectrum protocol v1

## Frozen scope

- Profiles: scalable TDL-A through TDL-E using 3GPP TR 38.901 V19.2.0.
- Independent profile input: `G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json`, SHA-256 `38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd`, bound to TR 38.901 source SHA-256 `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.
- Native implementation under test: Sionna PHY 2.2.0 CPU; 3.5 GHz carrier, 30 m/s fixed speed, 20 sinusoids, 8 subcarriers at 30 kHz, 100 ns delay spread, 2,048 coefficient-time samples, and sample rate `4 f_D,max`.
- Draws: 256 independent channel draws per profile, seeded by `1_743_000 + profile_index` for profile order A–E. The seed roster audit is `G4_SCALABLE_TDL_DOPPLER_SPECTRUM_SEED_AUDIT_2026-10-27.json`, SHA-256 `3a4c6cad865a0e5fa3ee07c166c8ebc70c8bc4bb39a1ea017d7b8400def76216`. It scanned 2,274 regular JSON/JSON.GZ files under research results, outputs, and artifacts: no collisions, malformed seed fields, or incomplete files. This is clearance only within that snapshot.
- Sampling unit: one independent channel draw. All ETSI-labeled Rayleigh taps within a draw are normalized individually and averaged into one spectrum; taps and frequency cutoffs do not increase the independent sample count. For TDL-D/E, omit the zero-delay native tap that merges deterministic LOS and diffuse components.
- Spectrum estimator: subtract each tap's finite-sample mean, apply a symmetric Hann window, compute the full two-sided complex periodogram, normalize each tap to unit spectral mass over the sampled Nyquist band, then average taps within each channel draw. FFT work is chunked in groups of 16 draws.
- Reference: isotropic-scattering Jakes spectrum, whose normalized-frequency CDF is `0` for `f/f_D <= -1`, `1` for `f/f_D >= 1`, and `0.5 + asin(f/f_D)/pi` in between. Evaluate at normalized cutoffs `[-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]`.
- Family: 35 profile × cutoff mean-CDF comparisons. Compute two-sided normal intervals over independent draw clusters with Bonferroni familywise alpha 0.01. Each interval must lie wholly within ±0.05 absolute CDF error from its Jakes reference value.

The sampling rate places the Jakes support inside the Nyquist band with a factor-of-two margin. The confidence statement is a project regression criterion for the bounded Sionna configuration, not a 3GPP tolerance or population truth.

## Interpretation and reproduction

The portable tests validate the analytic CDF, bounded estimator, seed/source binding, and cluster-level interval calculations. The native test exercises a limited empirical spectrum summary for the named TDL configurations. It does not establish the exact continuous Doppler spectrum, inter-tap dependence, spatial behavior, measured propagation validity, receiver impact, or G4 qualification. The native cases have not been run; no RT generation or qualification experiment is authorized by this protocol.

After code implementation and consolidated software validation, run:

```bash
uv run --locked python -m pytest -q -m native_sionna \
  tests/test_integrations/test_g4_scalable_tdl_doppler_spectrum_native.py
```
