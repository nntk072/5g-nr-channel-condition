# G4 CDL Temporal Covariance Under 3D UT Velocity

## Frozen scope

- Profiles: CDL-A, CDL-B, CDL-C, CDL-D, and CDL-E from TR 38.901 V19.2.0.
- Observable: the normalized complex temporal covariance of the zero-frequency, cluster-summed channel between two samples separated by 0.7 ms:
  \[
  \widehat{R}(\tau)=
  \frac{\frac{1}{N}\sum_{q=1}^N H_q(0)^*H_q(\tau)}
       {\frac{1}{N}\sum_{q=1}^N |H_q(0)|^2}.
  \]
- Conditions: carrier 3.5 GHz; speed 30 m/s; delay spread 100 ns; two time samples; CPU execution; independent stochastic channel draws.
- Velocity directions: positive x, y, and z axes; the fixed oblique vector (10, 20, 20) m/s; and the 12 normalized icosahedral vertices at 30 m/s. The test roster contains 16 directions, not only axis cuts.
- Sampling unit: one independently generated stochastic channel draw. Use 8,192 draws for each profile × velocity configuration.
- Inference family: 5 profiles × 16 velocities × 2 real components = 160 two-sided claims. Use Bonferroni familywise alpha 0.01. Every simultaneous interval must contain its independent complex reference component and have half-width no greater than 0.05.
- Seeds: `evaluation.g4_seed_roster.cdl_temporal_velocity_seed_roster` assigns `932400 + profile_index * 16 + velocity_roster_index`, with profiles ordered A–E and velocity cases following the insertion order in `tests/test_integrations/test_g4_cdl_temporal_vector_reference_native.py`. The dependency-free regression at `tests/test_evaluation/test_g4_seed_roster.py` checks the complete roster and invalid inputs before native execution.

## Independent reference

The expected covariance is computed from the pinned ETSI-derived per-cluster power, angle, and spread fixture, not from Sionna's expected-value metadata. The exact fixture is `G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json` with SHA-256 `24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090`. Its declared source is ETSI TR 138 901 V19.2.0, with source-PDF SHA-256 `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.

Normalize the transcribed linear cluster powers over the profile. For a ray direction \(u\), velocity \(v\), carrier \(f_c\), lag \(\tau\), and speed of light \(c\), the temporal phase is \(2\pi f_c\tau(v\cdot u)/c\). Sum its complex exponential using the profile's independent angular data and the frozen 20 ray-offset values from TR 38.901 Table 7.5-3. Apply the deterministic LOS contribution separately for CDL-D/E and sum the diffuse cluster expectations with their normalized powers. The test's Cartesian azimuth/zenith expectation follows its frozen randomized ray-coupling construction; the exact implementation is in `_independent_temporal_covariance`.

The installed Sionna 2.2 profile-resource SHA-256 values in `_PROFILE_SHA256` identify the system-under-test tables only. They are not used to construct the independent expected covariance.

## Decision calculation

For each configuration, estimate the complex covariance from the retained draw-level channel coefficients. Estimate real and imaginary uncertainty with the normalized ratio-estimator influence values \(H_q(0)^*H_q(\tau)-\widehat{R}(\tau)|H_q(0)|^2\), divided by mean reference power and \(\sqrt{N}\). Use the two-sided normal critical value for alpha `0.01 / 160`. Require both component errors to be within their simultaneous half-widths, and require the maximum half-width to be at most 0.05.

## Limits and status

This is a finite, native regression definition for one temporal covariance observable. It does not prove population-level G4 qualification, cover other carrier/speed/array/pattern configurations, establish spatial consistency, or authorize production RT generation. Test execution and statistical results remain pending. The report must not describe these 8,192 draws as ground truth.
