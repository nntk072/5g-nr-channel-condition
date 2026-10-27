# G4.07 Fixed TDL-D First-Tap Rician Marginal Protocol

## Frozen scope

- Profiles: fixed TDL-D10 and TDL-D30 from TS 38.101-4 V19.2.2 Annex B.
- Observable: the squared magnitude of native first-tap coefficients, normalized by the independent first-tap total power. The first resolved tap contains both the standardized LOS component and a Rayleigh diffuse component.
- Independent reference: LOS and Rayleigh powers are recomputed separately from the byte-pinned `G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json` component rows. Sionna's packaged table is recorded only as system-under-test identity.
- Native execution: Sionna PHY 2.2.0 CPU, 3.5 GHz carrier, 30 kHz spacing, 24 subcarriers, one time sample, and 32,768 independent channel draws/profile. The D10 bandwidth is 200,000,001 Hz to satisfy its strict Annex-B applicability threshold; D30 uses 100 MHz.
- Sampling unit: one independent channel draw. No time samples, taps, or repeated antenna ports are pooled as independent units.
- Fresh seeds: D10 `1,880,010`; D30 `1,880,011`. The saved-corpus audit scanned 2,276 JSON/JSON.GZ files and found no collision, malformed seed field, or incomplete scan. Audit SHA-256: `3640230f34c566c72854201cf133a8ab0be70b9ab3fcdc95eab2b3c26995890e`; receipt: `docs/research/G4_FIXED_TDL_RICIAN_MARGINAL_SEED_AUDIT_2026-10-27.json`. The receipt is outside scanned roots so reruns do not report their own reserved roster as a collision. This is scoped to the declared saved corpus, not global seed exclusivity.
- Acceptance: a two-sided DKW bound simultaneous across both profile CDFs at family-wise alpha 0.01. Each empirical normalized-power CDF distance plus its Bonferroni-adjusted DKW radius must be at most 0.05.

For diffuse power `P_d`, specular power `P_s`, and normalized power `x = |h|^2/(P_d+P_s)`, the expected CDF is `F(x) = ncx2.cdf(2*x*(P_d+P_s)/P_d, df=2, nc=2*P_s/P_d)`. `P_d` and `P_s` are not estimated from the sampled channel.

The scorer and native regression are `evaluation/g4_tdl_rician_marginal.py` and `tests/test_integrations/test_g4_fixed_tdl_rician_marginal_native.py`. Portable scorer/reference tests pass. The two bounded native cases were exercised once during implementation and passed before the final Sionna parameter-table digest assertion was added; no durable result artifact was emitted, so the native family still needs a bound rerun during consolidated validation. A passing regression would check this finite fixed-profile software marginal only. It would not establish LOS phase evolution, broader Rician parameter applicability, spatial/temporal behavior, measured propagation validity, receiver impact, or G4 qualification.

## Reproduction

```bash
uv run --locked python scripts/audit_seed_roster_collisions.py \
  --root docs/research/results --root outputs --root artifacts \
  --seed-range 1880010 1880012
uv run --locked python -m pytest -q -m native_sionna \
  tests/test_integrations/test_g4_fixed_tdl_rician_marginal_native.py
```

Normative source identity: ETSI TS 38.101-4 V19.2.2, Annex B.2.1.1. The pinned component fixture preserves printed powers and the D10/D30 LOS-plus-diffuse first-tap composition.
