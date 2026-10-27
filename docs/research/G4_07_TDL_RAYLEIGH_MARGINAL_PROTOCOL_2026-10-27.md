# G4.07 TDL Rayleigh Marginal Protocol v1

## Question and scope

For each ETSI-labeled Rayleigh-only tap in scalable TDL-A–E and fixed TDL
A10/A30/B100/C60/C300/D10/D30, does the pinned Sionna PHY 2.2.0 coefficient
sample have the table-scaled exponential power marginal implied by a zero-mean
complex Gaussian (Rayleigh-envelope) coefficient? This adds a marginal-shape
check beyond zero-mean and proper-complex second moments. It does not establish
joint Gaussianity, inter-tap independence, temporal/spatial dependence,
specular LOS behavior, or G4 qualification.

## Frozen design

- Independent draw count: 32,768 channel draws/profile; one complex coefficient
  per tap/draw (`num_time_steps=1`, one TX/RX antenna, CPU, `num_sinusoids=20`).
- Profiles: scalable A/B/C/D/E at 3.5 GHz, 100 ns delay spread, 30-kHz SCS,
  24 subcarriers, nominal bandwidth 100 MHz; fixed A10/A30/B100/C60/C300/D10/D30
  at the frequencies and nominal bandwidth constraints encoded by the pinned
  TS 38.101-4 profile reference.
- Reference bytes: scalable TR 38.901 v19.2.0 profile component table
  `G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json`, SHA-256
  `38b897029bd4cc6de49eece5096e69c75291eaf6dac45bebe04767ed24a4f0bd`; fixed
  TS 38.101-4 v19.2.2 component table
  `G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json`, SHA-256
  `fe22560a71f4e5683ea6780f73e84466155b26f5f97cc4b63e3250c102fafda3`.
- Only taps whose complete component roster is labeled Rayleigh are included.
  TDL-D/E and D10/D30 mixed LOS-plus-diffuse first taps are excluded. The
  expected mean power for each included native tap is the ETSI component's
  linear power divided by the full profile's linear total power; it is not fit
  from the stochastic draws.
- Sampling unit: one independent channel draw, as declared by the native
  adapter's batch-axis metadata. The fixed/scalable families contain 82 and 95
  Rayleigh-only taps, respectively, for 177 simultaneous tap-CDF claims.
- Statistic: two-sided Kolmogorov distance between sorted
  `|h_tap|^2 / expected_mean_power` samples and the unit-mean exponential CDF.
  Use a Dvoretzky–Kiefer–Wolfowitz radius
  `sqrt(log(2 * 177 / 0.01) / (2 * 32768)) = 0.012642297418096`, providing a
  99% simultaneous distribution-free confidence bound over all thresholds and
  taps by a union bound.
- Pass criterion: for every included tap, empirical KS distance plus the
  simultaneous DKW radius is at most 0.05. The 0.05 margin is a project
  regression tolerance, not an ETSI numerical limit.
- Seeds: `1,780,000` through `1,780,011` in the profile order listed above.
  Seed audit `G4_TDL_RAYLEIGH_MARGINAL_SEED_AUDIT_2026-10-27.json`, SHA-256
  `89988383b244c1f388034277f3ed5cf23d1b4b8aa55584066e3be7b595aaf351`, scanned
  2,275 saved JSON/JSON.GZ files under research results, outputs, and artifacts;
  status is `CLEAR_WITHIN_SCANNED_JSON_CORPUS` with no collisions, malformed
  seed fields, or incomplete files. This is not global seed exclusivity.

## Execution

After code implementation and consolidated software validation, run only:

```bash
uv run --locked python -m pytest -q -m native_sionna \
  tests/test_integrations/test_g4_tdl_rayleigh_marginal_native.py
```

Do not pool this family with properness, covariance, Doppler, earlier TDL
reports, or any reused seeds. A pass is bounded marginal consistency for these
profiles/configurations only. Preserve failures and do not change the frozen
margin or reuse these seeds for a revised protocol. Production RT generation
remains gated by criterion-driven G3 acceptance.
