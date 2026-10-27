# G4.07 Fixed TDL Rayleigh Component Properness Protocol

## Frozen scope

- Profiles: fixed TDL-A10, A30, B100, C60, C300, D10, and D30 from TS 38.101-4 V19.2.2 Annex B.2.1.
- Independent reference: the exact-digest-pinned per-component table in `G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json`. A delay cluster is included only when every component assigned to that cluster is labeled Rayleigh. Thus all taps are included for A10/A30/B100/C60/C300; the merged LOS-plus-Rayleigh first cluster is excluded for D10/D30, while their remaining Rayleigh taps are included.
- Native implementation: Sionna 2.2.0 CPU TDL adapter, one RX port, one TX port, 24 subcarriers at 30 kHz spacing, one time sample, one 32,768-draw independent batch/profile. Carrier and nominal bandwidth are fixed per profile by the regression definition; those values do not change the phase-statistic null being checked.
- Sampling unit: one independent channel draw. Taps within one draw are not treated as independent replicates.
- Frozen seed rule: `1_730_000 + profile_index` for profiles ordered `A10, A30, B100, C60, C300, D10, D30`. The native regression verifies the exact seed roster against `G4_FIXED_TDL_PROPERNESS_SEED_AUDIT_2026-10-27.json` (SHA-256 `c9887cb5a39b858baacacbdd783b478638722fb657c512cad34fc83c285e3e56`). The audit scanned 2,273 JSON/JSON.GZ files under `docs/research/results`, `outputs`, and `artifacts`; it found no collisions, malformed seed fields, or incomplete files. This is scoped to the saved corpus and does not establish global seed exclusivity.
- Family: all two real coordinates of the normalized complex mean and all two real coordinates of normalized complex pseudocovariance for every ETSI-labeled Rayleigh tap. The roster contains 82 taps and 328 simultaneous two-sided claims. Bonferroni family-wise alpha is 0.01; every interval must contain zero and each coordinate half-width must be at most 0.05 normalized by tap RMS amplitude.

## Interpretation

The independent ETSI table declares the included tap components as Rayleigh. This protocol tests the project's complex-baseband implementation of that component using zero-mean and zero-pseudocovariance criteria; the proper-complex second-moment criterion is a project-level statistical check, not a separately tabulated ETSI limit. The first D10/D30 cluster is not used because the adapter exposes the coincident deterministic LOS and diffuse components as one Ricean tap; testing that aggregate as zero-mean Rayleigh would be invalid. This finite family does not qualify Rician K-factor, temporal spectrum, inter-tap dependence, spatial correlation, receiver impact, or physical propagation. A passing result is bounded software/model evidence and does not qualify G4 or authorize production RT generation.

## Reproduction

From the repository root, after completing the collision audit and setting up the locked native environment:

```bash
uv run --locked python -m scripts.audit_seed_roster_collisions \
  --root docs/research/results --root outputs --root artifacts \
  --seed-range 1730000 1730007
uv run --locked python -m pytest -q -m native_sionna \
  tests/test_integrations/test_g4_fixed_tdl_complex_circularity_native.py
```

Normative source identity: [ETSI TS 38.101-4 V19.2.2](https://www.etsi.org/deliver/etsi_ts/138100_138199/13810104/19.02.02_60/ts_13810104v190202p.pdf). Its Annex B fixed TDL rows identify Rayleigh diffuse components and the separate LOS component; this protocol tests only the named Rayleigh component family.
