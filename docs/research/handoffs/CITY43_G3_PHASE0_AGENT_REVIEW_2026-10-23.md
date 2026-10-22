# Agent review brief: City43 high-depth LOS instability and G3 next step

**Prepared:** 2026-10-23 (Europe/Helsinki)
**Repository:** `git@github.com:nntk072/5g-nr-channel-condition.git`
**Current reviewed commit:** `8fe2fd1` (`Document City43 zero-scattering path attribution`)
**Review mode requested:** analysis and planning only. Do not implement changes, launch more RT experiments, or generate the production D1 RT corpus. Return an evidence-based recommendation that can guide the next continuation decision.

## Copy/paste request for the reviewing agent

Please fetch/pull the repository and independently review the City43 high-depth, zero-scattering LOS path-budget instability. Read this brief and the linked source/report files, inspect the implementation and relevant Sionna 2.2.0 source if available, and tell us what the existing evidence does and does not establish. Your job is to orient the next research step, not to implement it.

Answer these questions directly:

1. Does the evidence support path discovery/returned-record multiplicity, interaction-order growth, matched-path coefficient/geometry instability, coherent accumulation, or a combination? Rank these explanations and cite the concrete fields and calculations that support each conclusion.
2. Are the 75 high-budget zero-interaction records actually duplicate representations of the same direct LOS contribution, or could the saved fields/matching procedure make distinct paths look identical? Inspect the raw path diagnostics, the `Paths`→CFR bridge, native Sionna 2.2.0 solver flow, and the audit implementation. Separate demonstrated facts from likely interpretations.
3. Can the current artifacts distinguish explicit direct-LOS insertion from sampled-ray discovery, candidate deduplication, path-buffer handling, invalid-path filtering, or another internal source? If not, specify precisely which missing internal observability prevents attribution. Do not infer a Sionna defect from output multiplicity alone.
4. Does the path matching and CFR-delta decomposition stand up to an independent code review? Check the ordered interaction/object/primitive signature, the 2 cm assignment threshold, angular and delay-equivalent displacement costs, duplicate handling, closure error, and numerical assumptions. Identify plausible failure modes or sensitivity checks that matter to the conclusion.
5. Assess the frozen G3 acceptance contract against the actual UMCD MVP tasks. Are its CSI, delay/angle, receiver-relevant path, and NR receiver criteria operationally measurable and statistically defensible? Call out undefined terms, hidden choices, multiplicity issues, or mismatches. Recommend only necessary refinements, with exact replacement wording/thresholds and rationale; preserve the rule that neither finite-budget solver output is truth.
6. Recommend the single best next action. If more evidence is needed, propose one bounded, discriminating Phase‑0 follow-up (or explain why source inspection is enough), state the hypothesis, exact observable, inputs/settings, stop condition, what each outcome would mean, and how it would change the G3 decision. Do not recommend ray-count/depth convergence as an end in itself, broad sweeps, or production generation before the applicable criteria pass.
7. End with a short decision memo: **what we know**, **what remains unresolved**, **what should happen next**, and **what must stay gated**. Explicitly distinguish this one-case diagnostic from multi-family G3 qualification.

Please do not change files or run qualification experiments. A read-only audit, source inspection, and analysis of the committed artifacts are sufficient for this review. If you cannot access the external Sionna checkout, say so and limit claims about internals to the project artifacts/source that you can inspect.

## State of the work at the handoff

The user asked to resume the Unified Multipath Superdataset Plan while keeping production RT generation gated, complete one bounded Phase‑0 investigation of the City43 high-depth, zero-scattering LOS instability using existing path-level diagnostics, and define task-specific G3 acceptance criteria before further qualification experiments. That bounded investigation and criteria write-up are complete and committed. This is **not** a pass of G3 or completion of the UMCD plan. No production D1 RT corpus was generated.

The unified plan contains the frozen **“G3 acceptance contract for MVP dataset admission”** at `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md`, under Gate G3. At a glance it requires paired candidate/challenger comparisons across at least three independent scene/site families, at least 20 independent scene-link episode clusters per LOS/NLOS stratum/family, cluster-based uncertainty, and no path-cap saturation. It sets task-specific margins for unnormalized complex CSI, first-arrival/RMS-delay/3D angular statistics, receiver-relevant deterministic/specular path precision/recall and parameter error, and paired PUSCH/DMRS BLER and SNR impact. The full definitions and caveats are in that section; review the actual contract rather than relying on this synopsis.

The criteria are project engineering margins, not 3GPP limits or evidence of simulator truth. The contract expressly treats finite-budget comparisons as robustness decisions, not convergence-to-truth tests. Until applicable criteria pass, RT production D1 remains gated.

## Exact Phase-0 case and provenance

- Scene: `scenes/generated/research_scenes/city43/urban_00043.xml` (one procedural City43 scene family).
- Link: `center-los`, seed `23`; TX `(0, -55, 18)` m; RX `(-8.590140794952246, -15.375005963836855, 1.5)` m.
- Solver comparison: 100,000 vs 1,000,000 samples/source; maximum interaction depth 10; native material scattering set to zero; no production corpus generation.
- Channel/report setup: Sionna RT 2.2.0, LLVM path, 3.5 GHz, 256 CFR subcarriers at 30 kHz spacing; native per-path CFR contributions and diagnostics at depth 10 only.
- Neither run fills the configured candidate-path cap. Candidate counts before validation are 6,831 and 29,110; valid returned path counts are 194 and 597.
- The raw report records source revision `0bc55f1b6813d827d6235eb08c3324b29dcf9aa6`; the audit was produced by the committed auditor at `d369ff8` and records report/script/package hashes and runtime versions.
- Paired complex CFR relative L2 difference: `0.8708894955` (87.09%).

## Main numerical findings to verify

- Conservative path matching gives 157 matched records, 440 high-budget-only records, and 37 low-budget-only records. Matching groups by ordered nonzero interaction type/object/primitive sequence, then uses Hungarian assignment with a maximum 0.02 m cost based on interaction-vertex displacement, delay-equivalent distance, and endpoint-direction displacement. This is a diagnostic association rule, not a universal path identity.
- Matched paths are numerically stable: symmetric complex coefficient relative-error p95 `2.48e-5` (max `7.88e-5`); absolute delay-error p95 `5.68e-5 ns`; interaction-vertex displacement p95 `0.286 mm`; maximum endpoint arrival/departure direction-error p95 `3.77e-6 rad`.
- Zero-interaction record count grows from 9 to 75. At each budget those records have one 1-ps-resolution delay group, one coefficient group at `1e-12` resolution, and one geometry fingerprint. The common delay is about `146.015 ns`, consistent with the direct TX–RX distance. TX departure-angle fields are `(0,0)` for 8/9 low-budget and 74/75 high-budget records. The reviewer should check the raw records and fingerprint construction before calling these duplicates.
- Relative to the total CFR-delta norm, high-only path contributions account for `0.99996544`, low-only contributions `0.00071487`, matched-path coefficient/geometry changes `0.00000340`, and additive closure error `6.75e-7`. Per-path CFR reconstructs each aggregate within the runner’s recorded tolerance.
- At high budget, the order-0 coherent component is `1.3639e-4` versus total CFR mean power `1.3927e-4` (about 97.93%). The high-budget coherent-to-incoherent power ratio is 75.85, versus 9.85 at low budget. Incoherent summed path power rises from `2.36e-7` to `1.84e-6`, far less than the 59.96× aggregate CFR mean-power increase.
- Orders 8–10 grow from 59 to 330 records, but their summed incoherent path power is negligible in this S=0 case. Thus raw interaction-order count growth alone does not explain the observed channel change.
- Working interpretation: **the returned path set gains many repeated direct-like, zero-interaction records and their contributions coherently accumulate; matched-path coefficient/geometry drift is not the driver in this pair.** This is an output-level attribution. The report does not carry internal candidate lineage, so it cannot show which solver stage produced those records.

## Artifacts to read

1. `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md` — decision summary, complete G3 acceptance contract, G3 checklist, and downstream dataset gates.
2. `docs/research/PHYSICS_VERIFICATION_STATUS.md` — Phase‑0 finding, neighboring City41/42 and City43 diagnostics, and open-gate context.
3. `docs/research/results/rt-city43-center-los-seed23-depth10-s0-path-attribution-100k-vs-1m-20261022.json.gz` — full hash-bound paired solver report, including CFR vectors and per-path records. Decompress/read it; do not discard provenance.
4. `docs/research/results/rt-city43-center-los-seed23-depth10-s0-path-attribution-audit-20261022.json` — machine-readable attribution metrics and provenance.
5. `scripts/audit_rt_path_budget_attribution.py` and `tests/test_scripts/test_audit_rt_path_budget_attribution.py` — matching, decomposition, duplicate accounting, and synthetic unit tests.
6. `scripts/run_rt_solver_convergence.py` — path diagnostics and `--path-diagnostics-max-depth-only` behavior; native path-CFR reconstruction.
7. Prior paired-depth/budget evidence referenced by Edition 3.190 in the plan and status file. City41–43 share a generator and count as one scene family under the G3 contract; do not call them three independent sites.

The original Sionna reference checkout was cloned outside the project at `/home/nguyenl37/sionna`, pinned to v2.2.0 (`2f0a3820269b53fd022780a6f2f5f720f7ceb9b4`) with RT submodule v2.2.0 (`15b5ee036917a4c2a9b6420e05570b77717e2fcc`). If available, inspect the candidate generator, LOS insertion, shoot-and-bounce path, deduplication/hash-table behavior, path-buffer validity filtering, and path-to-CFR implementation. The previous project-side review observed that the solver has a direct LOS insertion path and also a sampled-ray path, but the saved experiment contains no internal lineage tying any returned record to either source. Independently verify those source details and their relevance to this exact run.

## Reproduction (only if needed for artifact verification)

The report already exists; do not rerun solver qualification merely to reproduce it. If needed to verify that the pinned runner can load the source case, use a **new output filename** because the runner refuses overwrite:

```bash
uv run --no-sync --locked python scripts/run_rt_solver_convergence.py \
  /tmp/city43-center-los-seed23-depth10-s0-path-attribution-review.json \
  --scene scenes/generated/research_scenes/city43/urban_00043.xml \
  --max-depth 10 --samples-per-src 100000 1000000 --seeds 23 \
  --link-case center-los 0 -55 18 -8.590140794952246 -15.375005963836855 1.5 \
  --include-cfr-vectors --include-path-diagnostics --path-diagnostics-max-depth-only
```

This command is included for provenance/understanding only. The requested task is an independent read-only review, not another run.

## Required shape of the response

Return a concise but technically grounded recommendation with:

- A verdict on the four proposed mechanisms (discovery/multiplicity, interaction-order growth, coefficient/geometry instability, coherent accumulation), including confidence and evidence.
- Any methodological flaw that could materially reverse that verdict.
- Any G3 acceptance-contract edits needed before the next qualification experiment; distinguish critical edits from optional strengthening.
- One recommended next action and a bounded decision tree for its possible outcomes.
- A clear statement that production D1 RT generation remains gated unless the applicable G3 criteria pass.

Do not report a ray budget or maximum depth as “ground truth,” and do not recommend simply increasing either until outputs stop changing.
