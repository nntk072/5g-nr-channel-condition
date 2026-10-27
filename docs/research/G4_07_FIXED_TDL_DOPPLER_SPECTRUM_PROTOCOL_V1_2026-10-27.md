# G4.07 Fixed TDL Doppler-spectrum protocol v1

## Frozen scope

- Profiles: fixed TDL-A10, A30, B100, C60, C300, D10, and D30 using the TS 38.101-4 V19.2.2 Annex B.2.1 component roster.
- Independent profile input: `G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json`, SHA-256 `fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3`, bound to TS 38.101-4 source SHA-256 `b3c6a73c7a40219e7270c08c7a57c84cb46f6757b303a64ec768ba04a6079e81`.
- Native implementation under test: Sionna PHY 2.2.0 CPU; 3.5 GHz carrier, 30 m/s fixed speed, 20 sinusoids, 8 subcarriers at 30 kHz, 2,048 coefficient-time samples, and sample rate `4 f_D,max`.
- Draws: 256 independent channel draws per profile, seeded by `1_760_000 + profile_index` in A10/A30/B100/C60/C300/D10/D30 order. The seed roster audit is `G4_FIXED_TDL_DOPPLER_SPECTRUM_SEED_AUDIT_2026-10-27.json`, SHA-256 `b95c11501bfc0d701f4143a2194a3728d486d77ead5e49e2d1d858b2f2c5459f`. It scanned 2,275 regular JSON/JSON.GZ files under research results, outputs, and artifacts with no collisions, malformed seed fields, or incomplete files. Clearance is limited to that saved-corpus snapshot.
- Sampling unit: one independent channel draw. Each ETSI-labeled Rayleigh-only tap is normalized individually; tap spectra are averaged into one CDF per draw. The mixed LOS-plus-diffuse first tap in D10/D30 is omitted.
- Spectrum estimator: subtract each tap's finite-sample mean, apply a symmetric Hann window, compute the full two-sided complex periodogram, normalize each tap to unit spectral mass over the sampled Nyquist band, and average taps within each draw. FFT work is chunked in groups of 16 draws.
- Reference: isotropic-scattering Jakes spectrum, whose normalized-frequency CDF is `0` for `f/f_D <= -1`, `1` for `f/f_D >= 1`, and `0.5 + asin(f/f_D)/pi` in between. Evaluate at normalized cutoffs `[-0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75]`.
- Family: 49 profile × cutoff mean-CDF comparisons. Compute two-sided normal intervals over independent draw clusters with Bonferroni family-wise alpha 0.01. Each interval must lie wholly within ±0.05 absolute CDF error from its Jakes reference value.

The criterion is a bounded project regression for the named native configuration, not an ETSI tolerance or population guarantee. The sample rate places the Jakes support inside the Nyquist band with a factor-of-two margin.

## Interpretation and reproduction

The portable estimator is shared with the scalable-TDL Doppler protocol. The native cases have not been run; protocol implementation does not establish a continuous Doppler law, inter-tap dependence, spatial behavior, measured propagation validity, receiver impact, or G4 qualification. No RT generation or qualification experiment is authorized by this protocol.

After code implementation and consolidated software validation, run:

```bash
uv run --locked python -m pytest -q -m native_sionna \
  tests/test_integrations/test_g4_fixed_tdl_doppler_spectrum_native.py
```
