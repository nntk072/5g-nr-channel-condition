# Active implementation queue

Checkpoint: 2026-11-11, based on `751aad4` plus the current PKG08 audit work.
This queue is the executor's next-action list; the research and dataset
plans remain authoritative for scope and acceptance. Update this file only when
a task closes, its dependency changes, or a concrete blocker appears; retain
chronological detail in evidence artifacts rather than duplicating it across plans. Gate IDs in the ledger use
`DS.*` and `RS.*` namespaces so equal historical numbers cannot be confused.

The first ledger snapshot is [REQUIREMENT_LEDGER.json](REQUIREMENT_LEDGER.json).
It indexes 182 backlog rows, all 104 numbered unified-plan acceptance criteria
(22 source-checked boxes and 82 open boxes), and 30 dataset/research gate or
phase records. Checked-box marks are preserved but are not implementation or
scientific-qualification claims. Most entries remain
`unknown` / `not_run`: that is deliberate until a source-level audit links an
implementation, verifier and exact evidence. The feature-map parser does not
determine ledger status.

## Current task

### E00 — finish common-runner replay provenance

- **Requirements:** `PKG01-006` (partial common-runner implementation); replay
  contract and runnable producer integration.
- **Status:** focused implementation and verification complete (37 passed).
- **Changed files:** `artifacts/run_replay.py`, its JSON schema, manifest and
  experiment runner integration, CLI replay arguments, and focused tests.
- **Delivered:** a digest-bound command/source/config/input/backend/RNG/attempt
  contract; complete and failed terminal transitions; a common API wrapper
  that writes failure receipts when a started manifest exists; CLI retry
  identity arguments.
- **Known boundary:** failures before the initial manifest exists have no
  replay receipt. Process interruption leaves a durable `running` receipt for
  recovery; no recovery/reconciliation workflow is implemented yet. Other
  experiment producers were found in
  `scripts/run_observable_multipath_experiment.py` and
  `experiments/capacity_sweep.py` and have not been audited for replay coverage.
- **Verification command:**

  ```bash
  ./.venv/bin/python -m pytest --no-cov -o addopts='' -q \
    tests/test_artifacts/test_run_replay.py \
    tests/test_artifacts/test_manifest.py \
    tests/test_experiments/test_runner.py
  ```

- **Closure:** command passes; changed-file Ruff and schema checks pass; check
  the feature-map freshness and `git diff --check`; ledger records common-runner
  coverage as partial, with non-run producers and interrupted-run recovery
  explicitly open. This task does not qualify a scientific run.

## Ready queue

### E01 — reconcile the full requirement ledger

- **Dependencies:** E00 focused verification result.
- **Scope:** inspect every backlog row and every named requirement under both
  governing plans. Replace generic/unreviewed ledger entries with stable source
  references, mandatory/conditional/later scope plus justification, source
  integration evidence, exact dependencies, verification status, separate
  scientific qualification status, closure rule and next task. Add any
  governing requirement missing from the backlog; do not delete or soften a
  requirement. Reconcile split gate naming and distinguish later gated work
  from prerequisite implementation.
- **Deliverable:** complete, source-audited `REQUIREMENT_LEDGER.json`, a concise
  blocker register and an ordered ready queue with finite task cards.
- **Verification:** ledger validator/schema check and a cross-reference audit
  that every backlog row and plan requirement has exactly one ledger record or
  a documented alias. Run
  `./.venv/bin/python -m scripts.validate_requirement_ledger` for structural
  and source-file coverage before the manual source audit.
- **Exit rule:** no `unknown` status remains without a specific inspection task
  and owner/dependency; external blockers are separate from code status.

This audit is divided into fixed batches so it cannot silently become a
general-purpose “review everything” task. `E01.PKG00` is closed: all four
package-0 rows were traced to the feature-map builder/schema/tests and the new
ledger validator. The inventory is structurally executable, but its
classifications remain heuristics and the duplicate-history/helper-wiring
requirements remain open. The validator reports exact row/gate coverage; it
does not claim source audit completion.

`E01.PKG01` is now closed after source review and a focused suite passed 107 tests. The cache fixture was updated to bind the antenna configuration already included in the runtime cache identity. The exact command is listed on PKG01 rows in the ledger. PKG01 remains software-implemented within the declared slice; full runtime matrix/attestation, arbitrary override coverage, and ODbL release records remain explicit limitations.

`E01.PKG02` is closed as a source audit. The bounded dataset/visibility/trajectory/network/duplicate-audit suite passed 306 tests with 1 dependency skip and 1 non-failing Torch warning. PKG02 rows now distinguish partial contracts from absent family-registry support. No native RT qualification was run.

`E01.PKG03` is closed as a source audit. The path/analytic/storage contract suite passed 211 tests with 1 dependency skip; the frequency-order/RT-to-NR contract suite passed 13 tests. These are software checks only; native path/array qualification remains open. The six PKG03 records distinguish partial path/storage work from complete semantic separation of geometric LOS and standardized proxies.

`E01.PKG04` is closed as a source audit. The G3 contract/evaluator/acquisition/signature suite passed 483 tests in 231.06 seconds. This validates software contracts, not external material/CSI/path evidence or G5 receiver qualification; all G3 receipts remain `NOT_QUALIFIED`. PKG04-032 preserves the bounded City43 Phase-0 diagnosis from existing path-level evidence and does not reopen convergence experiments.

`E01.PKG05` is closed as a source audit with the remaining aggregate gaps
explicitly partial. `PKG05-001` retains its bounded G4.10 result (10/10 cases,
3,200/3,200 claims); `PKG05-002/003` cover canonical profile/capability
contracts; `PKG05-004/005` cover scenario GBSM storage, EpisodeBundles and its
bounded CLI; `PKG05-007/010` cover TDL/CDL profile and table identity;
`PKG05-011/014` cover reference-audit contracts, fixed-TDL scalar/K-factor
references and adapter input validation; `PKG05-016/017` cover capabilities
and CDL scalar reference contracts; `PKG05-018/023` index finite retained
PDP/RMS, polarization, angular, pattern and circularity evidence;
`PKG05-025` indexes the bounded D/E zero-orientation covariance pass.
`PKG05-006/015/026` remain partial aggregate G4 inventory/coverage. The
`PKG05-024` frozen CDL-E miss remains failed and unresolved in cause. The
focused software slice passed 136 tests; the additional source-reference and
CPU-oracle checks passed 39, 22 and 5 tests respectively. These checks did not
rerun native G4 families. Exact local ETSI PDF-byte audits remain unavailable
without user-sourced PDFs. The new report publisher is schema-validated and
deterministic; the original tuple/array mismatch is fixed. G4.10 qualifies
only within scope and overall G4 remains open.

`E01.PKG06` is now active. `PKG06-001` is source-audited: the type-1 SRS
frequency-tree recurrence requires explicit `n_srs` when hopping applies and
is connected to per-symbol scheduled counters and single-/multiport mapping.
Its selected 13-test software slice passes. ASN.1/PER configuration, CRC/polar
decoding, full scheduling completeness, independent conformance and receiver
qualification remain open. `PKG06-002` also source-audits cleanly: shared
proper-complex AWGN, the PUSCH estimator and seeded TDL noise reject invalid
variance types/ranges and float32 overflow/underflow before use. Its focused
slice passes 61 tests with one expected Torch sparse-invariant warning. These
input-contract tests do not qualify a receiver. `PKG06-005` is complete in the
bounded software scope: network PUSCH handoff rejects incomplete and trailing
slot samples before demodulation; its focused 3-test slice passes. This does not
establish acquisition or receiver qualification. `PKG06-003/004/006` remain
partial: static BWP PUSCH decode and bounded network superposition/impairments
exist, while hop-specific DMRS, transform precoding, calibrated propagation,
complete acquisition/estimation and broad multi-cell receiver integration are
not implemented or qualified. Keep those backend and evidence gaps visible;
do not treat the single V100 smoke case as a G5 result.

`PKG06-011/012/013/014` are now source-audited as complete within their
bounded software contracts. The SRS schedule API enforces selected UL-BWP
subcarrier bounds; the same-grid occupancy report rejects incomparable grid
contexts and caps expansion; the multiport estimator records rank and
availability per subcarrier; and its optional per-RX/per-symbol variance
profile drives weighted LS and is retained read-only. Seven selected
regressions pass. These checks do not establish multi-cell RF behavior,
synchronization, or receiver qualification. Continue with `PKG06-021` onward in
dependency order.

`PKG06-015` through `PKG06-020` are also source-audited within their bounded
software scope. The retained-result path recomputes one-sided channel failure
bounds and CRC/block counts from Boolean outcomes, binds ordered receiver and
transport-block rosters, caps requested work before receiver initialization,
and validates PUSCH noise before float32 conversion. Forty-five selected
software/input-contract checks pass. The synthetic report remains
`not_qualified`; these controls do not satisfy G5 acceptance.

`PKG06-010` is source-audited as partial. Forty selected DCI/SRS measurement
regressions pass for decoded-state DCI widths and caller-supplied positioning
measurement/report arithmetic. Raw ASN.1/PER and CRC/polar, complete physical
configuration/scheduled-cell binding, first-path and clock calibration, and
synchronized receiver qualification remain open.

`PKG06-021/022/023/024` are complete within decoded-state scheduling scope:
the Hyper-SFN positioning gate, resource-set composition, UL-BWP feasibility,
and frozen-roster multi-cell scheduling pass eight selected regressions.
Raw RRC/PER interoperability and receiver qualification remain open.

`PKG06-025` remains partial after an 18-test PBCH acquisition preflight slice:
bounded blind acquisition/PBCH configuration exists, while general ASN.1/PER
extension decoding and broad case/band/beam/RF receiver coverage remain open.
`PKG06-026` closes its bounded PSS/SSS input/work boundary after nine selected
checks; this does not qualify detection performance. `PKG06-027` remains a
partial frequency-domain static-channel PRS receiver baseline. `PKG06-028/029/030`
close their bounded PRS sequence/mapping, DL-RSTD arithmetic, and decoded
occasion-resolution software contracts; the 55-test PRS module passes. None
of these results qualifies a raw-sample positioning receiver.

`PKG06-031/032/033/034/036` source-audit as complete within the declared
synthetic software/inference contracts; the 65 G5 acceptance, audit, and curve
checks plus two paired cluster-comparator checks pass. `PKG06-035` remains
partial: broader receiver configurations, independent signer roles, trusted
timestamp custody, source-matched replay, and qualification are open. These
curves remain diagnostic; no G5 acceptance verdict has passed. All
`PKG06-001` through `PKG06-036` now have source-level bounded dispositions, so
`E01.PKG07` is active. `PKG07-001` remains partial after strict time-axis,
stored alignment, and split-bound trajectory checks. Named multi-source clock
alignment is now recorded and replayed with each episode grid, with no
interpolation. The RT trajectory adapter binds its generated complex channel
to the digest of the declared solver/TX-pose/RX-pose grid. This records
software linkage but does not prove independent clock calibration; scene-object
source declarations and physically justified tolerances remain open. The
combined time-axis, episode-store, and non-generating RT contract slice passes
50 checks. `PKG07-005` remains
partial after mathematical direct-path ablation and immutable bundle replay
checks; scene-based physical interventions and scientific effect admission
remain open. Fifteen time-axis/counterfactual checks and nineteen persisted
ablation/trajectory checks pass. `PKG07-002` now has a bounded retained-table
path evolution diagnostic in `datasets/path_evolution.py`: five tests cover
descriptor-based association independent of row order, births/deaths,
near-tie ambiguity, solver/missing-group censoring, caller-declared phase
gauges, source-table/report digests, schema validation, and candidate-pair
limits. Six focused regressions and Ruff pass. It does not retrace scenes or
validate physical path completeness. `PKG07-003` now also preserves the
optional `phase_gauge_id` in backward-readable channel layers and provides a
fail-closed coherent-comparison disposition over gauge, link/time/port axes,
frequency grid, delay reference, and power reference. Four focused
phase-gauge/storage checks and 59 target/model-input/evaluation checks pass.
`TargetValue` now requires explicit per-element bounds when a reason is
`CENSORED`, persists the intervals, and reads pre-interval layers
backward-compatibly; source-specific censor generation and observability
bindings remain open. Calibration evidence and pipeline-wide causal eligibility
also remain open. Older censored targets without intervals retain their
unavailable mask and migrate to `LEGACY_REASON_UNRECORDED`. `PKG07-004` now has
a `PairedFrequencyViews` contract and schema binding two distinct-carrier
complex views to the same scene geometry/assets and source/split parent, with
per-view content hashes, separate material/solver/run provenance digests and no
cross-band phase claim. `pair_stored_frequency_bundles` resolves and hash-checks
both episode stores, requires a common split group, and records both bundle
hashes. Immutable pair-manifest publication and read-time source replay are
implemented, including altered-bundle and tampered-manifest rejection. Nine
pair-contract checks pass. `audit_paired_frequency_split_binding` checks the
two episode-child bindings, common network-scene parent, scene realization, and
assigned partition against a supplied registry. This input-scoped audit does
not verify signer custody or release roster completeness. This does not
generate paired solves;
`build_paired_frequency_rt_requests` now prepares shared-scene, same-route
requests with independent band/material configs; a 39-test non-generating RT
contract plus pair-contract slice passes. It does not execute solves. Pair
solve execution, signed release-registry/roster integration, and scientific
qualification remain open.

The frozen CDL-E four-element covariance miss is now linked to
`M1.G4_CDL_E_FAILURE_DISPOSITION`. Its exact `Re(C[0,1])` discrepancy is
0.0135743014 against a 0.0129694116 half-width. Five CPU checks of the
ETSI-derived oracle pass its Hermitian, unit-diagonal and PSD invariants; this
does not resolve whether the one native miss reflects finite-sample variation
or an oracle/implementation discrepancy. Keep the original result failed. The
separate post-hoc 65,536-draw diagnostic cannot replace it.

`E01.PKG07` is source-audited. `E01.PKG08` is active: rows `PKG08-001` through
`PKG08-016` now have source/test evidence and separate software/qualification
statuses. `PKG08-015` is partial with signer custody and authoritative inventory
recorded as external blockers; `PKG08-016` has a complete software audit, with
reviewer custody and evidence truth still external. The 219-test target/NR/
storage slice, signature/tamper checks, and 40-test corpus-auditor module pass.
`PKG08-017` remains partial after its scalar-boundary test slice (7 passed);
its full closure depends on auditing `PKG08-018` through `PKG08-022`. `PKG08-018` is source-audited complete for its named data-object routes (10
focused adapter/profile-holdout tests); `PKG08-019` now has 8 passing focused
value/shape regressions and is complete within the registered-adapter scope.
Continue with `PKG08-020` through `PKG08-022`, then return to close or disposition `PKG08-017`, followed by
`PKG08-023` through `PKG08-039`,
then packages 09–10, `E01.DS.GATES`, `E01.RS.PHASES`, `E01.RS.APPENDIX_E`, and
`E01.CROSSWALK`.
Each package batch is the fixed set of `PKGnn-*` IDs in the ledger and closes
only after every row has inspected source/test integration, implementation
state, exact verification, qualification state, dependencies and closure
rule. The three gate batches collect the exact mandatory/conditional scope and
acceptance definitions from their governing sections. `E01.CROSSWALK` checks
complete one-to-one requirement coverage and resolves dependencies/blockers.

For every batch, retain the exact source files inspected in that ledger row;
do not copy the whole chronological status history. If a row is already
implemented, reconcile it and identify its exact verifier rather than
reimplementing it.

### Implementation cards discovered by E01.PKG02

These cards are ready to refine after the E01 requirement and split crosswalk;
they do not authorize native qualification.

#### M1.G4_CDL_E_FAILURE_DISPOSITION — inspect the frozen four-element miss

- **Requirements:** `PKG05-024`, G4 profile/covariance criteria.
- **Dependencies:** finish `E01.PKG05` and the G4 gate crosswalk; retain the
  frozen protocol/result and independent ETSI fixture identities.
- **Current evidence:** result artifact
  `docs/research/results/g4-cdl-four-element-covariance-frozen-execution-20261109.json`;
  frozen seed 6714, 8,192 draws/profile, CDL-E `Re(C[0,1])` absolute error
  0.0135743014 and half-width 0.0129694116. The deterministic ray oracle passes
  Hermitian/unit-diagonal/PSD checks. A larger post-hoc diagnostic is
  inadmissible to replace this failure.
- **Deliverable:** a source-level independent derivation review of the ray
  angle coupling, phase/sign convention, power normalization, LOS/diffuse
  composition and claim extraction, with a documented defect/no-defect/unclear
  disposition. Do not change the frozen result or thresholds. If the cause is
  not identifiable from retained evidence, leave the failure unresolved and
  define a separately preregistered fresh-seed protocol as a later task.
- **Verification:** dependency-light oracle checks plus a reproducible
  calculation from the retained claim; no Sionna/native rerun in this card.
- **Exit:** either a specific source defect and a separately versioned fix, or
  a documented unresolved statistical miss with evidence for the next protocol.

#### M1.SCENE_FAMILY_REGISTRY — connect planned scene families to split candidates

- **Requirements:** `PKG02-004`, `DS.G1`, `DS.G7`, and scene roles in unified
  plan §§6.1–6.4.
- **Dependencies:** E01 package 3 (path/target contracts), package 8 (split and
  release), and the gate crosswalk; inspect `datasets/split_policy.py`, the
  existing pilot candidate sampler, and `evaluation/near_duplicate_audit.py`.
- **Deliverable:** versioned S00–S11 family catalog joined to concrete
  scene/site candidate identities, source-kind, geometry-asset digest,
  licensing/release state, and independence-group identity. Integrate family
  stratification with the existing candidate and signed split flow. Keep
  analytic fixtures, procedural variants, ODbL sites, surveyed sites and
  measured campaigns distinct. Empty or unqualified families must report
  unsupported/shortfall; registry membership is not evidence of scene
  availability or qualification.
- **Verification:** schema/hash roundtrip; deterministic sampling under a
  frozen seed; reject the same site or exact geometry asset crossing a frozen
  holdout boundary; tests for ODbL use eligibility versus public-release
  eligibility. Do not add a second split allocator.
- **Exit:** named family candidates flow into existing signed split audit, and
  unsupported families remain visible in reports.

#### M1.SCENE_VISIBILITY_EDGE_CASES — close the geometry oracle contract

- **Requirements:** `PKG02-001`, unified plan G1.01–G1.06.
- **Dependencies:** existing `propagation/visibility.py` and
  `propagation/scene_visibility.py` contracts.
- **Deliverable:** source-bound analytical cases for grazing, endpoint/mesh
  contact, zero-length and near-degenerate segments, unit/affine transforms,
  and a thin wall whose geometric occlusion remains distinct from RF
  transmission. Borderline geometric cases must be ambiguous; do not silently
  map them to clear or blocked.
- **Verification:** dependency-light analytic tests plus a separately named
  native comparison only when the environment and frozen geometry fixture are
  ready. Native comparison is not required to close the software contract.
- **Exit:** deterministic classifications and tolerances are documented and
  tested; material transmission remains an independent model/evidence branch.

#### M1.SCENE_STATE_TIMELINE — represent supported material and pose changes

- **Requirements:** `PKG02-002`, unified plan G0.04/G1.04/G1.09/G6.01.
- **Dependencies:** E01 gate crosswalk and the declared time/pose contract in
  `datasets/network.py`; review `integrations/sionna/rt_trajectory.py`.
- **Deliverable:** exact supported interpolation modes for node pose and
  moving-scene objects; per-time geometry/material-state identity; explicit
  errors for unsupported rotation, deformation, clock alignment or material
  updates. Freeze interpolation mathematics before implementation rather than
  inferring it from a sample grid.
- **Verification:** time-gap, endpoint, continuity and digest-binding tests;
  analytic motion fixtures before any retraced RT evidence.
- **Exit:** generated pose/material state reproduces exactly from the stored
  contract, with unsupported motion rejected or marked missing.

#### M1.MULTINODE_RT_SUPPORT_MATRIX — connect topology data to generation

- **Requirements:** `PKG02-005`, unified plan §18 and G5 topology consumers.
- **Dependencies:** E01 gate crosswalk; inspect `datasets/network.py`,
  `integrations/sionna/rt_network.py`, `integrations/sionna/network_phy.py`,
  and current backend compatibility tests.
- **Deliverable:** finite matrix for supported TX/RX nodes, ports, links,
  serving/interfering relationships and unsupported configurations; then wire
  one dependency-ready supported topology from path generation through channel
  storage and the network PHY consumer. Preserve one-to-one node/device/port
  bindings and exact failure states.
- **Verification:** analytical multi-link composition and shape/identity
  contract tests; native backend qualification stays a later, separately
  frozen task.
- **Exit:** every requested topology is either generated and consumed through
  the same episode identity or explicitly rejected with a reason.

### E02 — close the split-to-evaluation integration slice

- **Dependencies:** E01 audit of split, benchmark, lineage and evaluator
  requirements; retain existing signed partition materializer and result
  promotion gates.
- **Scope:** trace frozen parent/child assignments through actual experiment
  consumers, including benchmark cohort availability, descendant membership,
  row/parent/component bindings and inference-time leakage. Implement only gaps
  demonstrated in this end-to-end route; do not add another parallel split
  helper if existing contracts can be wired in.
- **Deliverable:** one runnable consumer path, retained audit artifact and
  finite unsupported-cohort list with precise external dependencies.
- **Verification:** targeted tests for every changed integration boundary and
  a small deterministic fixture covering assignment through evaluation.
- **Exit rule:** consumers use the signed partition identity and validate exact
  parent/child lineage before scoring; the artifact reports unsupported or
  missing cohorts instead of silently treating them as qualified.

### E03 — canonical path, target and storage contract audit

- **Dependencies:** E01 source map for `DS.G1`, `DS.G2`, `DS.G7`, targets and
  serialization; use analytical and already-retained evidence first.
- **Scope:** identify concrete gaps in same-path reconstruction, units/gauge,
  dtype/masks, missingness, arrays, retry/failure state, and model-input
  consumers. Each change must name an original requirement and downstream
  consumer.
- **Deliverable:** closed implementation cards for verified gaps plus exact
  remaining qualification dependencies; no broad rewrite.
- **Verification:** focused round-trip, analytic-oracle and failure-path checks
  for changed contracts.

### E04 — bounded receiver integration and G3 readiness

- **Dependencies:** E01 task-level scope audit; existing G5 receiver work and
  task-specific G3 acceptance definitions.
- **Scope:** map exact required waveform/receiver configurations, implement
  missing interface/criteria plumbing, and distinguish code gaps from backend
  limitations. Freeze run-specific G3 contracts and sampling design before
  any qualifying experiment.
- **Deliverable:** bounded support matrix, receiver-to-evaluator path, explicit
  external blockers and any concrete user decision required for unsupported
  mandatory modes.
- **Entry rule:** do not launch scientific qualification until the applicable
  G3 contract and sampling are frozen. Production D1 RT and advanced training
  remain gated until their existing criteria pass.

## Scientific and release guardrails

- Keep the 2x2 single-V CDL-B covariance miss failed and visible; a diagnostic
  or different cohort cannot replace it.
- Do not reopen City43 Phase-0 with convergence sweeps or call a finite ray
  budget ground truth.
- Use Etoile, Florence and Munich under the accepted ODbL policy. Public
  redistribution still needs release-specific source/extract, conversion,
  attribution/source-offer, third-party-rights and review evidence.
- Do not launch broad CI or expensive/native qualification during prerequisite
  implementation. Run the consolidated software verification only after its
  implementation entry conditions are met.
- Do not mark a phase, requirement or mission complete from plan text, parser
  counts, synthetic fixtures or a passed software contract alone.
