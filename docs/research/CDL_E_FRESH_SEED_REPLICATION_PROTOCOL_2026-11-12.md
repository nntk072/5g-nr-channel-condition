# CDL-E four-element covariance fresh-seed replication protocol

**Status:** preregistered design; not executed. The 2026-11-09 frozen result
remains a failure. A replication can characterize repeatability only; it cannot
replace or retroactively pass that result, qualify G4, or authorize production
RT generation or advanced training.

## Question

Does an independently seeded 8,192-draw CDL-E run under the same finite
four-element ULA configuration contain the independent ETSI-derived covariance
reference in every predeclared simultaneous interval while meeting the
unchanged precision ceiling? A pass or failure is evidence for this exact
repeat, not a causal diagnosis of the original discrepancy.

## Frozen design

- Profile: CDL-E, TR 38.901 v19.2.0.
- Runtime: Sionna PHY 2.2.0 on CPU; no GPU or RT backend is involved.
- Carrier: 3.5 GHz; SCS: 30 kHz; eight subcarriers; 100 ns delay spread.
- TX/RX: one stationary downlink; four vertical omnidirectional UT elements
  on a local-y half-wavelength ULA; zero UT orientation.
- Draws: 8,192 independent stochastic channel draws from one fixed seed.
- Reserved seed: `2823000`.
- Observable: sum retained path coefficients across clusters for each draw,
  then the raw 4×4 empirical spatial covariance. Do not normalize individual
  draws or the empirical covariance.
- Reference: independent ray expectation from
  `docs/research/CDL_TAP_ANGLE_REFERENCE_2026-10-26.json`; fixture SHA-256
  `24ee2f8f73f26e27522a296a4a0860bdaa7f284e9138855666e09636bc97b090`;
  ETSI source SHA-256
  `b71460fb4978a59fbffc864652016cf161c9e17cce19f0d0e9b46bea630b1cc4`.
- Profile table SHA-256:
  `04328f33d8d79b24aa3d0303fecb58b465ed7e13874930b05e76be0d97743480`.
- Claim family: four diagonal powers plus real and imaginary parts of six
  unique off-diagonal entries, for 16 simultaneous two-sided intervals.
- Family-wise confidence: 99%, Bonferroni normal critical value across all 16
  claims. Every interval must contain its reference; every half-width must be
  at most 0.05 normalized mean power.
- A failed or incomplete execution is retained as such. Do not change the seed,
  draw count, reference, claim roster, confidence family, or threshold after
  observing outcomes; do not add a replacement seed after a failure.

## Seed audit

`CDL_E_FRESH_SEED_AUDIT_2026-11-12.json` reports
`CLEAR_WITHIN_SCANNED_JSON_CORPUS` for seed `2823000`: 2,302 regular JSON and
JSON.GZ files were scanned under `docs/research/results`, `outputs`, and
`artifacts`, with no collisions, malformed seed fields, or incomplete files.
Audit SHA-256:
`4eb74446ac3e1f50b1fc9644c43f8974bc899ab30b1a3897e570a8dacf38ae86`.
This is clearance only within the saved corpus snapshot; it is not global
seed exclusivity.

## Provenance and retention required before execution

The existing frozen test is
`tests/test_integrations/test_cdl_spatial_covariance_4element_native.py`
(SHA-256 `a37b5133f839a7ee2b8b332caf27bd26e6842d1b8015fc2cd95becce94fca5a2`)
and hard-codes seed `6714`. The replication uses a separate, versioned runner
and report path; do not edit that frozen case or run it with an overridden
seed. The separate runner is
`scripts/run_cdl_e_fresh_seed_replication.py`; its report contract is
`datasets/schema/cdl-e-covariance-replication.v1.schema.json`. Before
execution, the runner binds its own source hash, evaluator and frozen-test
hashes, this protocol hash, the seed-audit hash, the profile-table hash, the
reference-fixture hash, Sionna/Torch/Python versions, platform, command, and
exact seed/draw count. It retains each of the 16 estimates, reference values,
standard errors, interval bounds, half-widths, and pass decisions, including a
failed result, and independently recomputes report arithmetic.

The reproducible native command is:

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-sync --locked \
  python -m scripts.run_cdl_e_fresh_seed_replication \
  --output docs/research/results/g4-cdl-e-fresh-seed-replication-20261112.json
```

The runner and schema are covered by synthetic claim fixtures only. This
preregistration and software check do not execute the native replication or
constitute a G4 qualification result. Keep the native command deferred until
the separately gated validation stage.
