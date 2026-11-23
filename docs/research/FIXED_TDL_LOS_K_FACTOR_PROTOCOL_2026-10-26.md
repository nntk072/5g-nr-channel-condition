# Fixed TDL LOS K-factor reference check

## Scope

This is a deterministic fixed-profile parameter check for TS 38.101-4 V19.2.2 TDL-D10 and TDL-D30. It fills the bounded K-factor comparison gap alongside the pinned per-component delay/power reference. It is not a population-level fading or general G4 qualification.

## Independent expectation

The expectation derives from the source-hash-bound `G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json` component rows. Convert each listed component power from dB to linear power. Sum the `LOS`/`LOS path` components for the specular power, sum the `Rayleigh` components for diffuse power, then compute `K = P_specular / P_diffuse`. The co-located deterministic and diffuse components at delay zero remain distinct in this calculation.

The per-component powers are rounded to 0.1 dB in the standard table. Since numerator and denominator are each formed from rounded powers, the frozen comparison tolerance is ±0.1 dB. Expected values are D10 `K=7.9843184844` (`9.0223785208 dB`) and D30 `K=7.9457979438` (`9.0013751705 dB`). The independent reference also binds the ETSI PDF SHA-256 and exact component-fixture SHA-256.

## Regression

`tests/test_integrations/test_g4_tdl_los_k_factor_reference_native.py` contains a dependency-light source/hash/schema and formula check plus two native Sionna CPU cases using the fixed TDL adapters. The native check compares `metadata.k_factor_linear` in dB against the independently derived expected value and confirms LOS profile identity. It uses one fixed seed because the criterion concerns configured profile metadata, not a Monte Carlo distribution.

The native cases require the G4 seed-collision audit and a pinned Sionna 2.2 environment before execution. Passing the deterministic check would verify only the adapter's D10/D30 K-factor parameter against rounded Annex B component powers. It would not test the Rician fading law, LOS phase, temporal spectrum, spatial covariance, receiver impact, measured propagation, or production RT suitability.
