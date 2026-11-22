# Active implementation queue

Checkpoint: 2026-11-12, source audits recorded through `639a508`; the phase and
crosswalk source audits are complete, the V100 bounded runtime probe is
complete, and target-family implementation is active.
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

## Recently closed prerequisite

### E00 — finish common-runner replay provenance

- **Requirements:** `PKG01-006` (partial common-runner implementation); replay
  contract and runnable producer integration.
- **Status:** common-runner replay, capacity-sweep parent replay, and the
  standalone observable-multipath completed-run receipt are implemented and
  focused-verification complete. The common-runner suite previously passed 37
  tests; capacity sweep/auditor passes 9, and the observable-multipath runner /
  summarizer integration passes focused recovery and producer tests.
- **Changed files:** common replay contract/schema, manifest and experiment
  runner integration, CLI replay arguments, plus
  `experiments/capacity_sweep.py` and its v2 schema/LOSO auditor; the
  observable-multipath runner/summarizer and their focused producer/consumer
  tests.
- **Delivered:** a digest-bound command/source/config/input/backend/RNG/attempt
  contract; complete and failed terminal transitions; a common API wrapper
  that writes failure receipts when a started manifest exists; CLI retry
  identity arguments.
- **Known boundary:** failures before the initial manifest exists have no
  replay receipt. Process interruption leaves a durable `running` receipt.
`scripts/audit_interrupted_runs.py` now inventories those receipts and hidden
staging directories; an explicit digest-pinned operator action can mark a
schema-valid common-runner or capacity-sweep v2 receipt failed. The
  capacity-sweep parent now binds config, source/runtime/backend, input hashes,
  and child/control RNG streams; its LOSO consumer audits the v2 receipt and
  continues to read historical v1 sweeps. The standalone analytical pilot now
  publishes manifest v3 with the generated synthetic-dataset hash, run config,
  source/runtime identity and RNG streams; the summarizer validates the receipt
  while preserving v1/v2 reads. Its staging directory is only published after
  all outputs and hashes are complete, so failures before publication leave no
  user-visible partial result.
- **Verification command:**

  ```bash
  ./.venv/bin/python -m pytest --no-cov -o addopts='' -q \
    tests/test_artifacts/test_run_replay.py \
    tests/test_artifacts/test_manifest.py \
    tests/test_experiments/test_runner.py
  ```

- **Closure:** common-runner, capacity-sweep parent/auditor, observable-
  multipath producer/summary focused tests pass; Ruff/format, schema, feature-map
  freshness, ledger, compilation, and `git diff --check` pass. Interrupted-run
  inventory and explicit failed-state recovery are implemented for common
  `loslab.run.v2`, capacity-sweep v2, and observable-multipath staging v1/v2
  receipts. The v2 observable runner writes a preparation receipt before
  synthetic dataset generation without claiming a dataset hash, then upgrades
  it to a replay-bound receipt before model fitting; the final completed
  manifest atomically replaces it before output publication. The tool does not prove a
  process is inactive; recovery requires a current manifest digest and operator
  attestation. It inventories but never resumes or deletes staging data.
  Ordinary exceptions clean staging as before; process interruption leaves a
  recoverable receipt and preserved partial artifacts. The configuration and
  provenance snapshot is assembled before staging allocation, and the receipt
  is written immediately after directory creation; only interruption in the
  directory-create-to-atomic-receipt interval can leave a staging directory
  without a manifest. Other standalone formats remain open. This does not
  qualify a scientific run or model.

- **Capacity-sweep integration:** parent manifest v2 binds the common config
  digest, exact input hashes, source/runtime/backend identity, every child seed,
  and phase/observation/permutation-control RNG streams. Input hashes are
  rechecked before report publication. Complete and failed transitions update
  the replay receipt, and `summarize_capacity_loso` validates v2 receipts while
  retaining v1 read compatibility. `tests/test_experiments/test_capacity_sweep.py`
  and `tests/test_scripts/test_summarize_capacity_loso.py` pass 9 tests. This
  covers one canonical orchestration consumer; no model qualification is
  claimed.
- **Capacity-sweep verification command:**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' ./.venv/bin/python -m pytest \
    -o addopts='' -q tests/test_experiments/test_capacity_sweep.py \
    tests/test_scripts/test_summarize_capacity_loso.py
  ```

- **Interrupted-run recovery:** `scripts/audit_interrupted_runs.py` performs a
  bounded read-only inventory of `manifest.json` / `run_manifest.json` files
  and hidden staging directories. To terminalize a common-runner manifest,
  pass its relative path, the digest from the inventory, and an explicit
  operator statement that the process is no longer active. Existing output
  files and staging directories are preserved.

  ```bash
  ./.venv/bin/python -m scripts.audit_interrupted_runs OUTPUT_ROOT
  ./.venv/bin/python -m scripts.audit_interrupted_runs OUTPUT_ROOT \
    --mark-failed RUN_ID/manifest.json \
    --expected-sha256 MANIFEST_SHA256 \
    --operator-attestation 'I verified this process is no longer active.'
  ```

  Focused coverage: `tests/test_scripts/test_audit_interrupted_runs.py`
  (11 cases; 19 pass including the shared manifest/replay contract tests).
  The observable runner module passes 3 tests, including checks that the
  preparation receipt exists before dataset generation and the upgraded
  receipt is recoverable before decoder training.

- **Observable-multipath integration command:**

  ```bash
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' ./.venv/bin/python -m pytest \
    -o addopts='' -q tests/test_scripts/test_two_path_runner_cli.py
  ```

## Most recently delivered prerequisite

The shared multi-node RT producer integration below is implemented as a
dependency-ready software slice, but it is no longer the active code task.
Its native backend and scientific qualification remain gated. The current
implementation task is E03 in the ready queue; E04 follows its applicable
software dependencies, and neither queue entry authorizes production RT.

### M1.MULTINODE_RT_SUPPORT_MATRIX — shared-solve producer integration

- **Requirements:** `PKG02-005`, Unified Plan §18, and G5 topology consumers.
- **Current implementation:** scene extraction now carries an exact serving/
  interference role per TX→RX association. Shared `Paths` bindings convert to
  absolute CFR views and provenance-bound bundles; the D1-gated publisher and
  network-PHY consumer are wired, and the common static single-link publisher
  uses the shared converter. `network_episode_identity()` covers the full
  topology content, and `build_rt_network_generation_request()` binds that
  identity, complete scene assets, materials, backend selection, grid, solver
  settings, seed, and destinations into an exact D1 request. The new
  `run_authorized_static_network_episode()` runner checks D1 authorization and
  destination before backend activation, cross-checks run identity against the
  request, and revalidates the live scene assets, material manifest, geometry,
  carrier, node/array/pose state, and network identity before constructing the
  solver. It records per-source candidate counts and turns cap-filled and
  solver-exception outcomes into masked path/channel layers plus an immutable,
  digest-bound receipt. Successful output is converted into the topology-bound
  UMCD bundle and published immutably.
- **Verification:** 39 focused topology, Sionna adapter, and RT network tests
  pass; two native tests were deselected. Tests cover request authorization,
  incomplete-matrix rejection, live-scene identity checks, and deterministic
  fake-solver success, cap, and exception paths including receipt schema and
  bundle reload. Ruff, format, and compilation pass. No native multi-link solve
  ran; this is software contract verification, not solver/backend qualification.
- **Remaining:** native Sionna execution and qualification for the documented
  topology roster, scene/material/runtime evidence, and scientific acceptance
  remain open. The pilot runner is dependency-ready software; production RT
  remains gated until the required G3 criteria and scientific gates pass.
- **Next:** complete remaining software work in the ordered queue, then run the
  separately frozen native qualification only after its environment and
  acceptance criteria are ready.

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

The `PKG00-002` authority pass now identifies the ledger as the one current
requirement disposition and this queue as the one ordered action list. The
bounded six-scene MVP checklist is labeled historical; no dated results were
removed or rewritten. The full edition-to-requirement crosswalk remains open.
`scripts/build_historical_status_crosswalk.py` now inventories dated entries
from `MASTER_RESEARCH_PLAN.md` and `MVP_STATUS.md`, binding source/ledger and
manual-map digests, exact headings and note blocks. Reviewed mappings are kept
in `HISTORICAL_STATUS_MANUAL_MAPPINGS.json`; they add only canonical ledger IDs
with a rationale and cannot duplicate explicit matches. Numbered
dataset criteria such as `G4.07` map only when the corresponding exact
`DS.G4.07` ledger row exists; bare `G4` remains unresolved because the plans
reuse gate numbers. Current state: 603 exact-ID/numbered-criterion matches, 80
reviewed manual mappings, and 706 notes still unresolved. The expanded G4
batch covers finite covariance, polarization, temporal, PDP/RMS, angular,
and marginal-family evidence; these mappings identify owning requirements
without upgrading any frozen result or scientific qualification state. The
path extractor also indexes existing bare repository paths cited in prose,
not only Markdown links and backticks. Fifty-four additional cited notes now have pinned mappings: 21 duplicate G4 result references, nine source-audited G4 notes, and 24 direct evidence-to-owner mappings across replay/runtime, synchronized time, receiver software, corpus release, Phase-0, priors/baselines, and plan tooling. Each keeps its own source hashes; their outcomes and qualification states remain unchanged. Manual records pin
the exact heading and full note-block hashes, requirement IDs, and rationale.
They do not alter historical notes or current requirement status. The helper
inventory remains separately open under `PKG00-003`.

Verification: `tests/test_scripts/test_build_historical_status_crosswalk.py`
passes (3 tests); `scripts/build_historical_status_crosswalk.py --check`
confirms the saved inventory matches current source and manual-map bytes.
`PKG00-003` has one additional canonical edge: the exported
`standardized_draw_subject()` adapter was source-found to have no production
consumer. `make_standardized_run_split_registry()` now builds draw-level locks,
and `loslab-build-standardized-split-registry` consumes immutable stored-run
manifests and publishes a strict report plus candidate lock. Receiver view
children inherit the draw split through the shared child-binding API. Reused
seed/index pairs fail closed without explicit per-draw opaque identities. Five
focused API/builder tests pass; the global helper inventory remains open.

`loslab-audit-helper-wiring` now emits a source-hash-bound inventory of 254
exported top-level functions across `datasets`, `evaluation`, `experiments`,
`integrations.sionna`, `propagation`, and `receivers`, scanning internal
callers in `loslab`, scripts and the source packages, plus test references in a
separate inventory field. After correcting same-module and imported-submodule
alias resolution, the matrix records source-audited routes for all 185
functions with detected direct production callsites. The other 69 have no
production callsite but all have direct test references; 14 are exact-name
references in governing plans/backlog. All 69 now have explicit no-internal-
consumer dispositions in the source matrix; this closes export classification
only. Test coverage is not production workflow wiring. The report schema and
focused test pass. This remains a syntactic audit: dynamic/indirect calls and
semantic completeness remain open. The report also enumerates 16 non-exported
plan-named definitions and their direct source/test references. Manual method
tracing found three in-repository routes missed by AST name matching. Four
definitions have no production consumer: pose alignment, the test-only fixed-
TDL reference, partition-support replay, and multi-horizon target indexing.
Their evidence boundaries and dispositions are in the matrix; enumeration does
not establish that an unconnected helper fulfills its broader plan row. A canonical
software path now composes already channelized links, adds seeded receiver AWGN
and invokes the bounded network PUSCH decoder under a strict hash-bound
workflow report; its outcome remains `OBSERVED_NOT_QUALIFIED`. The next TX
workflow now applies node-specific IQ imbalance before caller-declared causal
MIMO FIR propagation, then enters this bounded PUSCH path. The persisted NR
inference reader now has a row-aware public API; its focused regression verifies
the identity stays outside model inputs and oracle corruption does not block
inference. The common ML runner consumes `ChannelDataset` channel views, while
this API serves stored NR PUSCH receiver tensors. There is no current MVP model
training target for that receiver payload, so do not invent a second model task
or connect labels by row order. Keep this reader as an external inference API
until an NR learned-task target and evaluator are explicitly required. Next:
resolve the four no-production helper cases only where an owning requirement
requires an internal consumer, then audit dynamic/indirect routes. E04 still
requires exact G5 receiver scope, sourced criteria, powered independent-channel
design and reviewer authority before qualification.
`run_network_grid_sinr` provides a bounded, hash-bound workflow for aligned
grid composition and per-resource SINR, and also calls the aggregate SINR
helper for fixed serving maps; resource-varying maps retain per-RE metrics
without an invalid fixed-map aggregate. It can optionally apply declared
frequency-dependent link-budget
profiles to explicitly unit-mean fading inputs, binding those inputs and the
resulting channel in the receipt. This closes only the declared-profile grid
route; calibration and frequency-to-causal-FIR mapping remain open. The TX
workflow accepts declared FIRs but does not derive them from RT. PKG00-003
remains open.

The stored-frequency helper edge is now executable through
`loslab-pair-stored-frequency-views`. It verifies exact immutable bundle
references and both child bindings against the frozen source-scene parent,
then publishes content-addressed pair and split-audit manifests. It consumes
existing independent solves only, marks results `NOT_QUALIFIED`, and never
invokes RT. This wires storage and split lineage; producing frequency-specific
RT solves and satisfying the paired-frequency G3 criteria remain open.
Focused verification commands:
`python -m pytest -o addopts='' --confcutdir=tests/test_scripts tests/test_scripts/test_pair_stored_frequency_views.py -q`
and `python -m pytest -o addopts='' --confcutdir=tests/test_datasets tests/test_datasets/test_frequency_pair.py -q`.

The mathematical direct-path helper is also wired through
`loslab-materialize-direct-path-ablation`. It reads an exact path-table bundle,
checks frozen split parent/children, and publishes a replay-validated derived
bundle into a separate store while leaving the source store untouched. This
implements same-path component removal only; it does not model or retrace a
physical blocker. A content-addressed receipt binds the source and derived
bundles, path-table digest, split-lock digest, and child IDs. Verification:
`python -m pytest -o addopts='' --confcutdir=tests/test_scripts tests/test_scripts/test_materialize_direct_path_ablation.py -q`.

Source tracing then found the exported `bind_split_children()` API had no
production caller. The new `loslab-bind-split-registry-children` command
consumes strict child-roster files, uses the existing binding helper, preserves
the parent lock, and publishes a candidate bound lock/report. Three focused
tests verify split inheritance, no parent mutation, orphan rejection, and
duplicate-child rejection. This
does not establish child-roster completeness or release approval.

The manual audit
must map dated notes to ledger IDs and retained evidence or explicitly classify
them as context with a reason, without rewriting their result claims.

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
and the source-level derivation review found no arithmetic defect; the implied
error is 3.5801 standard errors against a 3.4205 simultaneous critical value.
The cause (sampling variation versus simulator/reference mismatch) remains
unresolved. The review is recorded in
`CDL_E_FROZEN_MISS_REVIEW_2026-11-11.md`. Keep the original result failed;
the separate post-hoc 65,536-draw diagnostic cannot replace it. A fresh-seed
preregistration is queued and must not run until its seed roster and environment
are frozen.

The fresh-seed claim auditor now also validates the declared alpha and precision
criterion domains, exact claim field set, finite real claim values, nonnegative
uncertainties/errors, Boolean decisions, and finite interval bounds before
recomputing outcomes. The focused evaluator suite passes 15 tests, including
malformed-claim and invalid-criterion cases; Ruff passes. This is software
input-integrity evidence only. The frozen seed-6714 failure remains authoritative,
and no fresh-seed native replication or overall G4 qualification is claimed.

`E01.PKG07` is source-audited. `E01.PKG08` is active. Rows `PKG08-001` through
`PKG08-024` have source/test evidence and separate software/qualification
statuses; `PKG08-039` is independently source-audited as complete release-gate
software, with current-scene legal/provenance eligibility still blocked.
`PKG08-015` remains partial on trusted signer custody and authoritative
inventory; `PKG08-016` is software-complete while reviewer custody/evidence
truth remain external. `PKG08-017` now closes the repository-local producer/consumer audit after
fit/save/predict/load and archive-import source enumeration; external consumers
remain outside the claim. Broader model-input scientific and measured-data
qualification remains open. `PKG08-023` is source-audited complete for the declared NR/UMCD NPZ
stores: 43 streaming/memory/UMCD tests and six focused NR trajectory-store tests
pass. `PKG08-024` adds retry injection across all UMCD episode layers and the
standardized channel/learning stores (five standardized cases pass). Bounded
synthetic RSS diagnostics exercise public NR and UMCD write/read paths. The NR
16/128/512-record sweep and UMCD 1/8/32-bundle, 1/24/4096-subcarrier sweep are
retained with live-RSS baselines. NumPy-only NR validation avoids loading Torch,
reducing measured incremental RSS in the 16-record fixture from hundreds of MiB
to about 2 MiB. `PKG08-024` remains partial pending representative corpus sizing,
production-runtime repeatability, and a final scope audit. `PKG08-025` and
`PKG08-029` also remain partial for authoritative inventory/signer custody and
full semantic descendant coverage. The other rows `PKG08-026`–`028` and
`PKG08-030`–`039` have source-level software dispositions; the declared seed
scan and package-specific focused checks are recorded in the ledger. None of
these software statuses qualifies a corpus or clears a public release.

`E01.PKG09` is now active. `PKG09-001` and `PKG09-002` are software-complete
for profile identity and explicit independent-group counting. `PKG09-003`
remains partial for broader evaluator/model-route coverage; the canonical
runner support replay and UMCD report integration checks pass (77 runner,
adapter, and support tests; one focused UMCD integration test). Physical
independence, scientific support adequacy, site/array eligibility, and all
later model/evaluation routes remain open. `PKG09-009` now has a grouped
conformal/bootstrap interface that names physical episode/site/acquisition
units when declared and always reports physical independence as unverified;
33 focused tests pass. `PKG09-004`–`006` have train-only prior, descriptive
coverage/population, and validation-threshold software contracts, each still
partial at its external/scientific boundaries. `PKG09-007` remains partial for
physical classic estimators; `PKG09-008` is software-complete as a
nondeployment oracle-leak control. Their focused slices pass 20/54 and 9/34
selected tests respectively.

`E01.PKG10` is now active. `PKG10-001` remains partial after adding immutable
evaluation-report DAG materialization; generation, storage, receiver, and
paper-output nodes remain open/gated. The shared node API and existing
dataset-concatenation and candidate-registry integrations (`PKG10-002`–`004`)
are software-complete. `PKG10-005` lineage promotion and `PKG10-007` measured
resource budgeting remain partial. `PKG10-006` pilot design and `PKG10-008`
signed cohort materialization are software-complete but explicitly not
qualified. `PKG10-009` software is complete for the planned baseline routes:
CFR-derived estimators and sparse OMP/BIC, raw MLP, and parameter-matched
MLP/CNN/CV-CNN on shared splits with replayable input contracts. Scientific
evaluation and qualification remain gated on qualified inputs and frozen
acceptance evidence. `PKG10-010`–`011` remain gated on qualified inputs,
targets, and scientific criteria. `E01.DS.GATES` remains active: the G3
top-level contract and its twelve criterion rows now have source-level
dispositions. All twelve criteria remain scientifically `not_qualified`;
software-complete rows cover only bounded cap fail-closed handling, diffuse
accounting, documented model limits, LOS semantic separation, and prior
logging. The selected non-native contract suite passes 83 tests with 2 native
tests skipped; the separate G3 evaluator preflight passed 108 tests. Neither
suite qualifies G3. `E01.DS.GATES.G0` is also source-audited: selected runtime,
asset-closure, provenance, promotion, and ODbL release-contract checks pass,
but Halti-specific native runtime evidence and scene-specific public-release
receipts remain open. `E01.DS.GATES.G1` now has source dispositions for its
ten geometry/LOS criteria; the analytical ambiguity and split-identity
contracts pass their selected suite, while native solver agreement, scene
motion interpolation, and independent-site registry evidence remain open.
`E01.DS.GATES.G2` now has fifteen criterion-level dispositions. The selected
path, analytical-channel, RT-to-NR contract, adapter, and storage suite exits
0; native path/CFR equality, absolute link-budget scale, and representative
MIMO/polarization comparisons remain open. `E01.DS.GATES.G4` now records
bounded standardized profile/reference and statistical coverage. G4.07 remains
partial, and its native suites have been deferred until the implementation
pass is complete. `E01.DS.GATES.G5` now has source dispositions for its
fifteen receiver criteria: waveform, DMRS, timing, noise, impairment, storage,
and oracle-firewall software routes are distinguished from calibrated
receiver and G5 acceptance evidence. Its selected native/receiver suites were
not run. `E01.DS.GATES.G6` now has criterion dispositions for timestamp-based
targets, path evolution, gauge/censoring, intervention, and split leakage.
Physical motion/intervention evidence and deployment-causal input admission
remain partial. `E01.DS.GATES.G7` now maps target validity, storage,
input-firewall, split, duplicate-screening, coverage, and export controls.
Full family inventory, training-route coverage, production memory scale, and
per-scene release evidence remain open. `E01.DS.GATES.G8` is now source-audited
for effective groups, prior-separated cohorts, null controls, split-bound
tuning, and measured-data gaps. No model training or sealed-test evaluation
ran. Dataset gate source audits G0–G8 are complete; next reconcile
`E01.RS.PHASES`, `E01.RS.APPENDIX_E`, and `E01.CROSSWALK`, then implement the
remaining dependency-ready code gaps before broad validation or advanced
training.
Each package batch is the fixed set of `PKGnn-*` IDs in the ledger and closes
only after every row has inspected source/test integration, implementation
state, exact verification, qualification state, dependencies and closure
rule. The three gate batches collect the exact mandatory/conditional scope and
acceptance definitions from their governing sections. `E01.CROSSWALK` checks
complete one-to-one requirement coverage and resolves dependencies/blockers.

#### E01.DS.GATES.G3 — criterion-level source reconciliation (closed 2026-11-11)

- **Requirements:** `DS.G3`, `DS.G3.01`–`DS.G3.12`.
- **Delivered:** each criterion now names its bounded software evidence,
  exact scope, external blocker, software verification state, qualification
  rule, and next scientific task. Existing retained diagnostics are indexed;
  no convergence or production RT run was launched.
- **Verification:** `uv run --no-sync --locked pytest -q --no-cov -m 'not native_sionna' tests/test_scripts/test_rt_solver_convergence.py tests/test_integrations/test_rt_path_power_threshold.py tests/test_scripts/test_audit_rt_solver_replay.py tests/test_experiments/test_pilot_sampling.py tests/test_integrations/test_sionna_adapter.py tests/test_integrations/test_rt_trajectory_contract.py` (83 passed, 2 skipped); `./.venv/bin/python -m scripts.validate_requirement_ledger`.
- **Disposition:** software statuses are bounded; every G3 criterion and the
  top-level gate remain `not_qualified`. The current implementation has no
  admitted calibrated material package, independent-unit/power justification,
  or G5-qualified receiver result.
- **Next:** audit the other dataset gate records in dependency order; do not
  run G3 qualification until those evidence prerequisites and the frozen
  run-specific design are admitted.

#### E01.DS.GATES.G0 — runtime, source, and provenance criteria (source audit complete 2026-11-11)

- **Requirements:** `DS.G0`, `DS.G0.01`–`DS.G0.08`.
- **Delivered:** source dispositions now distinguish portable inventory and
  fail-closed asset/table/promotion contracts from Halti-host runtime checks,
  legacy result gaps, dynamic-input identity coverage, and per-scene ODbL
  redistribution records. No host qualification or release was performed.
- **New runnable export route:** `loslab-export-public-release` invokes the
  strict v3 exporter with an exact detached manifest signature and externally
  supplied OpenSSH signer policy. Its CLI and exporter suites pass 49 tests.
  This command packages only fully declared/reviewed manifests; current OSM
  extract/conversion provenance and third-party license closure remain absent,
  so no Etoile/Florence/Munich archive was generated or cleared.
- **Verification:** `uv run --no-sync --locked pytest -q --no-cov tests/test_integrations/test_runtime_inventory.py tests/test_integrations/test_sionna_runtime.py tests/test_integrations/test_scene_assets.py tests/test_scripts/test_audit_research_result_provenance.py tests/test_scripts/test_build_result_promotion_manifest.py tests/test_datasets/test_public_release.py` (exit 0); `uv run --no-sync --locked python -m scripts.validate_requirement_ledger`.
- **Disposition:** DS.G0 remains `not_qualified`; every G0 implementation
  status is limited to its named software contract. The Halti runtime/backend
  receipt and public-release eligibility for Etoile, Florence, and Munich
  remain open.
- **Next:** continue with G1/G2 and G4–G8 source audits; retain accepted ODbL
  use policy and require scene-specific evidence before redistribution.

#### E01.DS.GATES.G1 — geometry, LOS semantics, and scene identity (source audit complete 2026-11-11)

- **Requirements:** `DS.G1`, `DS.G1.01`–`DS.G1.10`.
- **Delivered:** the ledger distinguishes independent geometry/ambiguity and
  semantic-separation contracts from native PathSolver agreement, scene-wide
  frame/antenna review, dynamic-scene interpolation, solver convergence, and
  independent-site evidence. Existing assets and diagnostics are indexed;
  no RT generation was launched.
- **Verification:** `uv run --no-sync --locked pytest -q --no-cov tests/test_propagation/test_geometry.py tests/test_propagation/test_visibility.py tests/test_propagation/test_scene_visibility.py tests/test_integrations/test_rt_trajectory_contract.py tests/test_datasets/test_rt_split_identity.py tests/test_datasets/test_split_lineage_audit.py` (exit 0); `uv run --no-sync --locked python -m scripts.validate_requirement_ledger`.
- **Disposition:** G1 remains `not_qualified`. Native free-space/visibility
  checks require the Halti runtime gate; production scene claims also require
  G3 convergence/material acceptance and the scene-family registry.
- **Next:** audit G2 and the remaining dataset gate records; schedule bounded
  native oracle checks only after G0.02 and scene prerequisites pass.

#### E01.DS.GATES.G2 — native path extraction and channel semantics (source audit complete 2026-11-11)

- **Requirements:** `DS.G2`, `DS.G2.01`–`DS.G2.15`.
- **Delivered:** each criterion now traces to the path adapter, analytical
  reconstruction, RT-to-NR view, complex storage contract, or an explicit
  native evidence gap. Valid-mask use, phase/delay/power semantics, same-path
  decomposition, and storage are distinguished from solver truth and physical
  scale. No native RT qualification was run.
- **Verification:** `uv run --no-sync --locked pytest -q --no-cov tests/test_datasets/test_sionna_paths_adapter.py tests/test_datasets/test_analytical.py tests/test_integrations/test_rt_to_nr_contract.py tests/test_integrations/test_sionna_adapter.py tests/test_datasets/test_umcd_store_integrity.py` (exit 0); `uv run --no-sync --locked python -m scripts.validate_requirement_ledger`.
- **Disposition:** G2 remains `not_qualified`. Halti native API/CFR and link
  budget checks plus representative MIMO/polarization comparisons remain
  pending G0 runtime admission and separate physical evidence.
- **Next:** audit G4–G8; perform only the bounded native checks after G0.02
  passes and within the frozen supported envelope.

#### E01.DS.GATES.G4 — standardized model conformance (source audit complete 2026-11-11)

- **Requirements:** `DS.G4`, `DS.G4.01`–`DS.G4.09`.
- **Delivered:** criterion records now link to TDL/CDL resource adapters,
  independent reference tests, statistical families, standardized-record
  admission, and profile-proxy separation. Bounded finite profile coverage is
  stated separately from broad conformance and geometric-RT truth.
- **Verification:** source/reference paths validated with
  `uv run --no-sync --locked python -m scripts.validate_requirement_ledger`.
  The G4 native suites were intentionally deferred; this is not a qualification
  result.
- **Disposition:** G4 remains `not_qualified`; G4.07 is partial because its
  statistical checks cover named finite profiles/configurations, not the full
  supported continuous law, arbitrary arrays, or physical RT.
- **Next:** audit G5–G8 and finish dependency-ready implementation. After that
  stage, run the exact G4 criterion suites on the pinned Halti environment and
  reconcile any misses without broadening their scope.

#### E01.DS.GATES.G5 — NR waveform, channel, and receiver correctness (source audit complete 2026-11-11)

- **Requirements:** `DS.G5`, `DS.G5.01`–`DS.G5.15`.
- **Delivered:** criterion rows now link to supported NR waveform/DMRS and
  timing adapters, noise and impairment models, channel-estimation/storage
  contracts, and the oracle-input firewall. Bounded implementation status is
  separated from calibrated RF, broad receiver modes, and G5 scientific
  acceptance.
- **Verification:** cited source/test paths and ledger structure validated by
  `uv run --no-sync --locked python -m scripts.validate_requirement_ledger`.
  No native receiver suite or qualification cohort was run in this pass.
- **Disposition:** G5 remains `not_qualified`; physical noise/RF calibration,
  complete mode coverage, independently justified channel units, and a frozen
  accepted receiver-impact report remain blockers.
- **Next:** audit G6–G8 and complete code-level receiver integration/resource
  planning before running the bounded G5 qualification.

#### E01.DS.GATES.G6 — trajectories, forecasting, and interventions (source audit complete 2026-11-11)

- **Requirements:** `DS.G6`, `DS.G6.01`–`DS.G6.10`.
- **Delivered:** criterion records map to named-clock alignment, timestamp and
  gap-aware targets, path evolution, phase-gauge comparison, censoring
  intervals, counterfactuals, and split-lineage audits. Mathematical direct
  ablation is kept distinct from physical scene interventions.
- **Verification:** source/test paths and ledger structure validated with
  `uv run --no-sync --locked python -m scripts.validate_requirement_ledger`.
  No trajectory qualification or scene retrace was run.
- **Disposition:** G6 remains `not_qualified`; calibrated synchronized motion,
  native scene interventions, physical route independence, and a complete
  deployment-causal input firewall remain open.
- **Next:** audit G7–G8 and research phase gates; implement any demonstrated
  input-eligibility gaps before a frozen temporal qualification design.

#### E01.DS.GATES.G7 — storage, labels, privacy, and split integrity (source audit complete 2026-11-11)

- **Requirements:** `DS.G7`, `DS.G7.01`–`DS.G7.15`.
- **Delivered:** child records trace to typed targets/missingness, immutable
  storage, signed split and lineage controls, duplicate screening, target
  coverage, model-input firewalls, and ODbL export enforcement. Software
  contracts remain distinct from authoritative corpus completeness and legal
  release review.
- **Verification:** evidence paths and ledger structure validated by
  `uv run --no-sync --locked python -m scripts.validate_requirement_ledger`.
  No consolidated G7 suite or release build was run.
- **Disposition:** G7 remains `not_qualified`; the authoritative family/site
  inventory, full model-route matrix, production-scale memory evidence, and
  Etoile/Florence/Munich release-specific ODbL records remain open.
- **Next:** audit G8 and research phases; integrate remaining source-family,
  model-route, support, and release checks against the frozen candidate roster.

#### E01.DS.GATES.G8 — independent statistics and fairness (source audit complete 2026-11-11)

- **Requirements:** `DS.G8`, `DS.G8.01`–`DS.G8.10`.
- **Delivered:** criteria now map to class/effective-group support, pilot
  priors, low-confidence and population accounting, split-bound
  threshold/calibration, baseline/null-control routes, and external
  measured-data contracts.
- **Verification:** source paths and ledger structure validated by
  `uv run --no-sync --locked python -m scripts.validate_requirement_ledger`.
  No training, sealed-test query, or measured-data evaluation ran.
- **Disposition:** G8 remains `not_qualified`; physical independent-unit
  authority, selection-bias closure, complete classic baseline roster, and
  source-verified measured held-out data remain open.
- **Next:** the source reconciliation now covers `E01.RS.PHASES`,
  `E01.RS.APPENDIX_E`, and all 316 ledger crosswalk records: zero unreviewed
  scope entries, unknown implementation states, or missing next tasks. This is
a planning/source audit, not implementation completion or gate
qualification. The V100 runtime recipe is now locked to CUDA 12.6, Torch's
small CUDA device smoke passes, and the CPU LLVM Sionna RT API imports. The
first target-family slice (`rms_delay_spread` units, reference, aggregation,
and nonnegative range) passes 180 focused tests; explicit multi-value axes and
the other primary targets remain open. Continue the finite PKG00
history-reconciliation and canonical-helper wiring cards, then close the
split-to-evaluation implementation route before broad validation or advanced
training.

The pinned Sionna extras now resolve to PyTorch CUDA 12.6 on Linux, which is
compatible with the installed CUDA 12.9 driver. A bounded device smoke passed
on the Tesla V100 (compute capability 7.0). Mitsuba 3.9.1 reports that the
V100 is below the 7.5 minimum for its selected CUDA variant; its LLVM CPU
variant imported successfully when CUDA was hidden, selecting
`llvm_ad_mono_polarized`. The V100 GPU RT variant is not supported by the
reported minimum; native RT API/backend qualification remains open. No scene
was loaded and no path solver or production RT work was run. See the
environment note at the top of `MVP_STATUS.md`.

For every batch, retain the exact source files inspected in that ledger row;
do not copy the whole chronological status history. If a row is already
implemented, reconcile it and identify its exact verifier rather than
reimplementing it.

### Implementation card discovered by E01.PKG08

#### M1.PKG08_MODEL_INPUT_PRODUCER_RECONCILIATION — repository route audit closed 2026-11-12

- **Requirements:** `PKG08-017`; interfaces covered by `PKG08-018`–`PKG08-022`.
- **Dependencies:** finish `E01.PKG08`–`E01.CROSSWALK`; use the current model-input route inventory and preserve the CAEZ data-use license.
- **Current evidence:** canonical runner v2+ contracts, profile-holdout v1 contract, registered adapter audit, `TwoPathModelInputs`, CAEZ Ridge input/target/temporal-pair contracts, analytical two-path checkpoint sidecars, and the source inventory in `docs/research/MODEL_INPUT_ROUTE_INVENTORY.md`. The 2026-11-12 fit/save/predict/transform/load/import enumeration resolved every active match to a listed route. Real-CNN, complex-CNN, and CORAL raw checkpoints retain `external_input_contract_required`; canonical runner publications bind and replay sidecars. The 69-file historical archive is README-classified, with no active imports.
- **Deliverable:** exact producer/consumer matrix; replayable input/target/split contract for every retained model artifact; explicit nondeployment scope for synthetic diagnostics; explicit historical disposition for archived scripts.
- **Verification:** focused tests prove truth/metadata exclusion, exact matrix/partition binding, and two-path checkpoint loader replay against input/target/split and checkpoint digests. Use only bounded synthetic fixtures; no measured CAEZ archive, production RT, or advanced training run is needed for the software contract.
- **Verification:** the three standalone CNN/CORAL loader status tests, one two-path checkpoint contract test, and one canonical runner publication test pass. Exact searches and limitations are recorded in the route inventory. No licensed measured archive or external consumer is claimed as audited.
- **Closure:** repository-local active routes are enumerated and have replayable artifact contracts or explicit nondeployment/external-contract status. Broader scientific qualification, measured CAEZ evaluation, and unknown external consumers remain outside this source audit.

#### M1.TARGET_FAMILY_SEMANTICS — complete the primary target contracts

- **Requirements:** `PKG08-008`, dataset-plan G7.01–G7.05 and G7.14, and the
  target manifest requirements in Unified Plan §§17 and 20.3–20.4.
- **Implemented slice:** `umcd.rms_delay_spread.v1` pins seconds, a path-oracle
  producer, accepted reference/aggregation vocabulary, and a finite
  nonnegative real value domain. `umcd.rms_delay_spread.v2` declares its `frame`
  axis in the definition digest; `EpisodeBundle` requires link identity and an
  exact frame-roster shape. NR metadata schema v5 serializes value axes while
  v1-v4 readers remain supported. The focused target/UMCD/NR/storage/
  standardized slice passes 311 tests under the CPU LLVM backend. The reviewed
  dynamic geometric-LOS producer now publishes v4 with an explicit `frame` axis;
  its focused contract and storage checks pass (7 + 3 tests). The producer emits
  one value per frame, while its separate NR record remains scalar. NR RT
  trajectory CFR producers now publish v2 definitions with explicit
  effective-subcarrier and OFDM-symbol axes and observation-derived shape
  validation. TDL-PUSCH and standardized CFR producers also publish v2 with
  their reviewed seven-axis tensor layout. The complete target-contract module
  passes (39 tests), and NR v5 round-trips the actual standardized and TDL CFR
  v2 definitions with their seven axes (3 focused tests, including the generic
  axis contract). Existing v1-v3 records remain readable without inferred axes.
- **Next tasks in dependency order:**
  1. Continue auditing current targets for producer-backed range,
     gauge/reference, mask, or missingness invariants; enforce only rules
     specified by the producer contract.
  2. Review producer semantics before axis-tagging additional multi-value
     target families. Preserve current records until their axes are reviewed;
     do not reinterpret stored vectors.
  3. Define `detectable_direct_path` only after freezing detector, bandwidth,
     SNR/noise, and false-alarm criteria; keep it distinct from geometric LOS.
  4. Define `relative_first_path_delay` only after timing reference, acquisition,
     resolvability and censoring rules are fixed. Absolute ToA stays unavailable
     without synchronized clock provenance.
  5. Define dominant received-path power only after path-power and receiver
     link-budget references are fixed; keep normalized TDL/CDL power distinct
     from physical received power.
  6. Add a separate finite-bandwidth receiver-estimated delay-spread definition;
     do not pool it with path-oracle RMS delay spread.
- **Deliverable:** versioned schemas/producers for each accepted family with
  explicit axes, units, reference/gauge, aggregation, validity conditions,
  unavailable reasons, ranges, and legacy read/publication policy.
- **Verification:** focused contract/storage round trips and negative domain,
  axis, missingness, legacy-migration and schema tests. Use synthetic fixtures
  only; no target family is scientifically qualified by these checks.
- **External dependencies:** frozen detector configuration, calibrated clock
  evidence, and link-budget evidence block the corresponding families; do not
  invent defaults to clear those dependencies.
- **Closure:** all plan-mandated target families have a canonical producer and
  consumer contract, unknown axes/references cannot publish as canonical, and
  legacy data remain readable without silently upgrading scientific meaning.

### Implementation cards discovered by E01.PKG02

These cards are ready to refine after the E01 requirement and split crosswalk;
they do not authorize native qualification.

#### M1.G4_CDL_E_FAILURE_DISPOSITION — inspect the frozen four-element miss

- **Requirements:** `PKG05-024`, G4 profile/covariance criteria.
- **Dependencies:** finish `E01.PKG05` and the G4 gate crosswalk; retain the
  frozen protocol/result and independent ETSI fixture identities.
- **Current evidence:** result artifact
  `docs/research/results/cdl-four-element-covariance-frozen-execution-20261109.json`;
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

**Disposition recorded 2026-11-11:** source review found no arithmetic defect.
The implied frozen error is 3.580059 standard errors against a 3.420527
simultaneous critical value; the miss remains a failure, with stochastic
variation versus simulator/reference mismatch unresolved. See
`docs/research/CDL_E_FROZEN_MISS_REVIEW_2026-11-11.md`. The five dependency-
light oracle checks pass. No native rerun or result/threshold edit was made.

**Preregistration increment (2026-11-12):** seed `2823000` clears the retained
JSON/JSON.GZ result corpora with no collision, malformed field, or incomplete
file (2,302 files; corpus-scoped only). The separate CDL-E replication
protocol, digest-bound seed receipt, per-coordinate evaluator, immutable
report schema, and one-shot CPU runner are now present. Five deterministic
synthetic evaluator tests pass alongside the 23-test seed-audit suite (28
tests total). The original frozen seed-6714 failure remains unchanged. The native
replication was not run. This card's preregistration/software deliverable is
implemented; native execution remains a later gated action.

#### M1.G4_CDL_E_FRESH_SEED_REPLICATION_PROTOCOL — preregister without running

- **Requirements:** `PKG05-024` and the applicable G4 covariance criteria.
- **Dependencies:** complete the G4 gate crosswalk; audit all frozen G4 seed
  rosters and reserve independent seeds; pin the exact test, ETSI fixture,
  Sionna version and execution environment.
- **Deliverable:** a separately versioned, pre-outcome protocol for the named
  CDL-E setup. Freeze the seed roster, draw count, 16 covariance coordinates,
  simultaneous 99% interval and 0.05 half-width ceiling; require complete
  draw/coordinate artifacts and independent recomputation. Preserve the
  original failure as a separate immutable result.
- **Verification:** protocol/schema/seed-collision checks only. Native execution
  is a later, separately authorized qualification action and must not occur
  until the protocol and environment are frozen.
- **Closure:** a valid preregistration artifact exists; it does not change G4
  status and does not imply the replication passed.

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
- **Progress (2026-11-12):** `config/scenes/scene-family-registry.v1.json`
  now binds the full taxonomy and seven source-audited candidates, with source
  evidence digests, asset-tree digests, source kind, independence group,
  license state, and separate internal/public-release dispositions.
  `datasets.scene_families` provides canonical validation, permutation-stable
  SHA-256 family sampling, explicit unsupported/shortfall rows, and a relation
  adapter for the existing signed split union-find. The CLI and versioned
  sample report schema are in `scripts/sample_scene_families.py` and
  `datasets/schema/scene-family-sample.v1.schema.json`. The built-in
  street-canyon candidate is excluded from sampling because per-scene data
  licensing is not independently recorded. ODbL candidates remain eligible
  for internal research under the accepted policy and uncleared for public
  release.
- **Verification:** schema/hash roundtrip; deterministic sampling under a
  frozen seed; reject the same site or exact geometry asset crossing a frozen
  holdout boundary; tests for ODbL use eligibility versus public-release
  eligibility. Do not add a second split allocator.
- **Exit:** named family candidates flow into existing signed split audit, and
  unsupported families remain visible in reports.
- **Current status:** partial. Six focused tests pass. The direct RT dataset
  adapter accepts an optional candidate ID, verifies the actual asset digest,
  and records registry/family/leakage bindings in split-parent metadata; the
  existing split-lineage builder consumes those relations. The cached RT
  adapter now accepts the candidate ID, validates it against loaded asset
  bytes, binds candidate/family/registry/geometry identity into the cache key,
  and passes the identity to per-sample UMCD run identities. UMCD episodes now
  preserve optional split leakage relations through immutable storage, and the
  existing split-policy episode adapter consumes those relations. Static UMCD
  publication and the RT trajectory producer carry candidate relations,
  provenance, and run identity into their outputs. Focused cache (13), UMCD
  store-integrity (28), array-layout (7), and episode-schema compatibility
  (103) tests pass. The repository's Sionna `PathSolver` generation entry
  points are now covered by direct/cached adapter or trajectory integrations;
  future producer routes must use the same candidate binding. The new
  `umcd-episode-metadata.v1` schema documents the embedded episode object and
  its optional leakage relations; `_episode_payload` validates every newly
  written episode metadata object against it. The combined UMCD schema/store
  test modules pass (131). Two of twelve families have eligible candidates at
  the one-group design point; S00 plus S03–S11 report shortfalls. No
  qualification is implied.

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
- **Software closure (2026-11-12):** the independent mesh oracle now has
  focused regressions for clear/interior intersection, edge/endpoint/coplanar
  ambiguity, nearly grazing crossings, degenerate faces, coincident and
  tolerance-short segments, metric scaling and affine transforms, blocker
  insertion/removal, and a zero-thickness opaque surface whose geometric
  result carries no material-transmission claim. The loaded-scene adapter is
  also checked against an affine scale/shear/translation before unit
  conversion. The two dependency-light modules pass 19 tests with Ruff,
  formatting, compilation, and diff checks clean.
- **Boundary:** this closes the bounded software predicate contract only.
  Depth-zero PathSolver agreement for well-conditioned scene cases (G1.02),
  antenna placement outside forbidden solids, and material-aware RF
  transmission remain separately unverified; no native RT or production data
  generation was run.
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
- **Progress (2026-11-12):** froze
  `loslab.rt_trajectory_motion_contract.v1` and its JSON Schema: receiver and
  object positions use `p_i=p_0+v*t_i` on a trajectory-relative clock, with
  quasi-static retracing at each sample and no between-sample interpolation.
  Look-at orientation is recomputed per snapshot. Rotation, deformation,
  external clock alignment, and per-frame material updates are explicitly
  unsupported. The contract is bound into D1 requests, RT run identity, record
  provenance, and the generation sidecar. Object translations now use one
  finite-input checked equation helper; trajectory generation fails if the
  loaded material manifest changes during the run. The 39-test dependency-light
  trajectory contract suite passes; no path solver ran.
- **Current status:** software contract partial. The route/configuration and
  material invariant are explicit, but replay against native scene transforms
  and complete per-time state attestation remain unverified. G0/G1/G3
  qualification is not implied.
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
- **Progress (2026-11-12):** added a finite downlink support matrix in
  `docs/research/NETWORK_TOPOLOGY_SUPPORT.md` and
  `episode_channels_to_network_phy_input()`. It requires a complete TX×RX
  link matrix, exactly one serving transmitter per UE, explicit interfering
  links, homogeneous TX/RX port counts, one fully valid absolute channel view
  per link, and exact parent episode/grid alignment. The adapter preserves
  episode/node/frame identity and expands stored per-link CFRs over the caller's
  requested OFDM-symbol axis without claiming temporal channel evolution. A
  dependency-light 2×2 test writes and reloads the UMCD bundle, composes its
  views with the existing network PHY, and checks desired/interference powers;
  incomplete topology is rejected. Focused software tests pass (5).
- **Additional progress (2026-11-12):** added
  `sionna_bindings_to_channel_views()` to convert the exact link/frame binding
  set from `Paths.cfr()` into one absolute-delay, absolute-power UMCD view per
  directed link. It uses explicit device indices and link-local port maps,
  preserves episode frame and frequency axes, rejects malformed/nonfinite CFR
  data, and masks failed/unconverged solves so they cannot be consumed as
  measured zeros. Two dependency-light regressions cover 2×2 port reordering
  and failed-solve masking; `build_sionna_rt_episode_bundle_from_paths()` now
  composes those views with the canonical path/provenance bundle builder, and
  `write_sionna_rt_episode_bundle_from_paths()` applies the existing D1
  destination gate before immutable publication. The existing static RT UMCD
  producer now uses the same path-to-view converter before its provenance-bound
  bundle publication. A publisher contract test checks the gate and storage
  handoff. The RT network and Sionna adapter slices pass 25 tests with 1 native
  test deselected; Ruff, format and compilation checks pass. Pytest required
  plugin autoload and numerical
  thread fanout to be disabled after the default invocation stalled at full
  CPU without output.
- **Boundary (superseded by the static shared-solve runner progress below):**
  at this checkpoint, the software route covered caller-supplied `Paths`
  results through per-link CFR views, canonical immutable bundle publication,
  and the network PHY input adapter; only the common static single-link
  producer used the converter. The runner and retained failure states described
  in the later progress entry now close that bounded software gap. Native
  multi-link solver/backend qualification remains open. A narrow native 2×2
  path-binding smoke previously passed but is not qualification.
- **Topology extraction progress (2026-11-12):**
  `sionna_scene_to_network_episode()` now accepts a complete
  `link_roles_by_association` map. This lets a scene-derived complete 2×2
  downlink mark one serving GNB and one interfering transmitter per UE rather
  than assigning one role to every pair. Missing/extra role entries are
  rejected. Scene extraction plus network-composition slices pass 17 tests
  with 2 native tests deselected; Ruff, format, and compilation pass.
- **Static shared-solve runner progress (2026-11-12):** added
  `run_authorized_static_network_episode()` and a versioned generation-receipt
  schema. The runner binds the live scene and run identity to the D1 request,
  audits source candidate counts, publishes successful topology-bound channel
  and path layers, and retains explicit masked `UNCONVERGED` or
  `SOLVER_FAILURE` outcomes. Fake-solver tests cover those three terminal
  states and reload the immutable bundle/receipt. This completes the bounded
  software producer route; native multi-link backend qualification and the
  full planned topology roster remain separate open items.
- **Latest verification (2026-11-12):** 39 focused tests pass and two native
  tests are deselected; Ruff, formatting, and compilation pass. The full suite
  and native solver qualification have not been run.
- **Exit:** every requested topology is either generated and consumed through
  the same episode identity or explicitly rejected with a reason.

### E02 — split-to-evaluation integration (software slice closed 2026-11-12)

- **Dependencies:** E01 audit of split, benchmark, lineage and evaluator
  requirements; retain existing signed partition materializer and result
  promotion gates.
- **Scope:** trace frozen parent/child assignments through actual experiment
  consumers, including benchmark cohort availability, descendant membership,
  row/parent/component bindings and inference-time leakage. Implement only gaps
  demonstrated in this end-to-end route; do not add another parallel split
  helper if existing contracts can be wired in.
- **Closure:** `run_experiment` verifies the separately
  signed cohort declaration and the canonical signed split registry, snapshots
  exact named dataset bytes, replays the six-cohort index, checks that the
  train/validation/calibration rows match the split lock, and uses only the
  selected declared test cohort for scoring. The run manifest and result bind
  the chosen cohort, replay digest, and unsupported empty test-cohort list.
  Pre-manifest failures now create an exclusive
  `benchmark-cohort-preflight.json` receipt, preserving any existing receipt.
  The model-input verifier checks that model-sidecar cohort provenance
  matches the replay result and that its selected-test digest matches the exact
  inference indices. Unsupported cohorts include reason codes and external
  dependencies. Focused signed scoring, sidecar-tamper, failure-preservation
  and unsupported-cohort tests pass (3); manifest-replay tests pass (2).
  Ruff, formatting, compilation, ledger validation and `git diff --check`
  pass. These are software checks; no cohort is scientifically qualified.
- **Deliverable:** one runnable consumer path, retained audit artifact and
  finite unsupported-cohort list with precise external dependencies.
- **Verification:** targeted tests for every changed integration boundary and
  a small deterministic fixture covering assignment through evaluation.
- **Exit rule:** consumers use the signed partition identity and validate exact
  parent/child lineage before scoring; the artifact reports unsupported or
  missing cohorts instead of silently treating them as qualified.
- **Boundary:** this closes the six-cohort scoring route only. PKG08-025 still
  tracks corpus-wide child artifact semantics, authoritative inventory and
  signer custody; those broader lineage gates are not implied complete here.
- **Candidate-registry software verification (2026-11-12):** the separate
  `build_channel_split_registry.py` workflow's focused suite passes all 10
  cases, including complete legacy RT migration, bounded archive expansion,
  exact source digests, exclusive publication, and exact cache reuse. The
  registry stays `CANDIDATE_NOT_EXTERNALLY_REVIEWED` / `NOT_QUALIFIED`; this
  does not establish authoritative inventory or release eligibility.

### E03 — canonical path, target and storage contract audit

- **Dependencies:** E01 source map for `DS.G1`, `DS.G2`, `DS.G7`, targets and
  serialization; use analytical and already-retained evidence first.
- **Current implementation:** producer/source inspection found that the current
  standardized-PUSCH writer emits `umcd.standardized.profile-los-proxy.v1` as
  one binary value with shape `(1,)`, but the canonical validator did not bind
  this version to the fixed singleton shape or prohibit value axes. It now
  shares the scalar guard used by the current profile-proxy and geometric-LOS
  versions. Seven focused scalar/producer cases and the complete 42-test target
  contract module pass. Canonical path-oracle RMS and complex CFR target
  admission also rejects NaN and infinity. Historical target ranks remain
  readable; other target families are still open. Same-path direct-path
  ablation now rejects `SOLVER_FAILURE` and `UNCONVERGED` source groups rather
  than treating their empty storage as a zero channel; an explicit `NO_PATH`
  group remains represented as a finite-budget zero without a physical absence
  claim. An analytical three-path case checks closely spaced delays, coherent
  direct-path cancellation at the reference bin, signed phase across nonzero
  frequency offsets, and exact same-path indirect residual. The combined
  counterfactual/storage slice passes 35 tests; Ruff, format, and compilation
  pass. The recognized canonical target registry has matching provenance pins
  for all 19 versions. `direct_indirect_diagnostics()` now also rejects
  nonfinite full or direct CFR input instead of returning NaN/Inf power and
  ratio summaries; its focused adapter selection passes 8 tests with Ruff,
  format, and compilation clean. `PathOracleTable.validate_for()` now requires
  one explicit solve outcome for every episode link/frame/RX-port/TX-port pair;
  omitted rows cannot vanish from immutable episode storage. Path coefficient
  metadata is pinned to `sionna_rt_path_coefficient_v1` in the table constructor,
  storage reader and layer schema. The UMCD schema/store/counterfactual slice
  passes 140 tests; the Sionna path-adapter/RT-network slice passes 19 tests
  with one native test deselected. No native channel reconstruction ran. The
  UMCD writer now rejects seven legacy target versions whose axes were not
  digest-bound: standardized and TDL CFR v1, RT trajectory absolute and symbol
  CFR v1, RT geometric-LOS v2, and dynamic visibility v2/v3. Their historical
  records remain readable; producers already publish axis-bound successors.
  All seven refusal cases plus UMCD schema, store-integrity and target-contract
  suites pass (182 tests). Canonical target admission now also fails closed if
  a recognized version has no producer/reference/aggregation pin; the complete
  target/storage focused slice passes (183 tests). Ruff, formatting and
  compilation pass for the changed contract files. No native channel
  reconstruction ran. E03 now exposes `reconstruct_path_oracle_cfr()` as the
  shared path-to-CFR implementation: it preserves absolute path delays,
  carrier-relative frequency offsets and complex128 accumulation, returns
  explicit no-path groups as zero, and rejects failed/unconverged solves. The
  direct-path ablation consumer uses the same transform for its full and
  indirect-only views. The transform includes the carrier-phase term applied
  by Sionna `Paths.cir()` to passband coefficients, plus the subcarrier-offset
  phase. New counterfactual storage is layer v2 and binds `carrier_hz`; v1
  records remain readable but are write-protected because their carrier phase
  cannot be reconstructed from the saved layer alone. The counterfactual,
  UMCD schema, and store-integrity selection passes 149 tests; native
  `Paths.cfr()` parity and calibrated received-power truth remain open. The
  analytical CFR utilities now reject nonfinite path gains/delays and invalid
  FFT dimensions/spacings. Odd/even analytical grids verify the full centered
  frequency axis, unnormalized complex path gains, and coherent zero-bin sum;
  the existing RT-to-NR checks cover odd/even FFT ordering and DC placement.
  The focused analytical/path/NR frequency-order selection passes 44 tests.
  A direct integration regression now feeds a unit-mean analytical multipath
  CFR through the flat network link budget and verifies complex phase
  preservation, 6 dB voltage/power scaling, and transmit-power conversion. It
  passes with the existing flat-budget oracle check (2 selected tests); this
  verifies coefficient arithmetic only, not calibrated propagation.
  Producer/consumer tracing also found that RT trajectory NR records carry one
  geometric-LOS boolean per record, while UMCD stores the full trajectory on
  an explicit frame axis. NR admission now enforces the scalar per-record
  shape for `umcd.rt-trajectory.geometric-los.v2`; the UMCD route continues to
  use the axis-bound dynamic v4 definition. The focused target-contract and NR
  suite passes 119 tests. The same audit found the standardized PUSCH producer's
  profile-LOS proxy is fixed at `(1,)`, while NR admission accepted any
  one-element shape; admission now enforces `(1,)` and rejects `(1, 1)`. The
  expanded target-contract and NR suite passes 120 tests. Remaining target
  versions and schemas still need the complete matrix audit.
  The NR v5 writer now also refuses four axis-ambiguous canonical CFR v1
  versions whose producer-backed v2 successors bind tensor axes. Trajectory
  publication preflights every child record before writing any, so a rejected
  legacy target cannot leave a partially published route. The NR reader remains
  unchanged for historical records; the expanded target-contract and NR suites
  pass 125 tests.
  The current RMS-delay v2 contract now has a bounded path-oracle producer in
  `datasets.path_oracle.rms_delay_spread_target()`: one RX/TX port pair per
  link, exact frame coverage, power-weighted delay moments, no implicit power
  floor, explicit `NO_PATH`/zero-power missingness, and fail-closed solver
  errors. Its analytic producer regressions cover a known two-path spread,
  unavailable outcomes, multiport refusal, and solver-failure refusal. This
  wires a software producer only; multiport aggregation and physical truth
  qualification remain open. Exact focused command: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' ./.venv/bin/python -m pytest -o addopts='' --confcutdir=tests/test_datasets -q tests/test_datasets/test_umcd_schema.py -k 'rms_delay_spread_producer'` (2 passed).
  The helper is now wired into `build_sionna_rt_episode_bundle()` as an
  explicit opt-in derived target, and the timestamped F04 RT trajectory
  producer enables that option. It derives the RMS target from the same
  `PathOracleTable` assembled for the path layer, so path extraction is not
  repeated; multiport callers are rejected by the helper. An integration-level
  fake-table test confirms one frame-axis target per single-port link and
  reasoned missingness for no-path frames. This does not run a path solver.
  `EpisodeBundle` now requires the current v2 target to have its declared path
  oracle layer and replays the canonical producer against that exact table;
  altered values, validity masks, reasons, censoring, or definition reject
  before storage. Six focused target/storage cases pass, and the complete
  RT-network module passes 17 tests. This validates source-to-target consistency
  only; it does not validate path-table physical truth or permit multiport
  aggregation.
  The current registry contains 19 canonical versions. Producer/storage review
  classifies four CFR v2 versions, dynamic visibility v4, and RMS v2 as
  axis-bound UMCD targets; standardized profile v1, standardized profile v2,
  and static visibility v2 are fixed singletons, while trajectory geometric
  visibility v2 is a fixed scalar in each NR record and remains UMCD-write
  protected. UMCD publication is now read-only for superseded or axis-unbound
  CFR v1s, trajectory geometric-visibility v1/v2, static visibility v1,
  dynamic visibility v2/v3, and RMS v1; the reader preserves legacy bytes/ranks.
  Trajectory geometric-visibility v1 retains its per-frame binary semantics for
  reads, but the frame axis remains digest-unbound and publication is refused.
  Static visibility v1’s old ranks remain constructible for reads. The focused
  guard slice passes nine refusal cases, plus a legacy-rank compatibility case.
  Exact commands:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' ./.venv/bin/python -m pytest -o addopts='' --confcutdir=tests/test_datasets -q tests/test_datasets/test_umcd_schema.py -k 'superseded_target_versions_are_read_only'`
  (9 passed), and
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' ./.venv/bin/python -m pytest -o addopts='' --confcutdir=tests/test_datasets -q tests/test_datasets/test_target_contracts.py -k 'historical_static_visibility_v1'`
  (1 passed).
  A handcrafted historical target object also passes the full manifest,
  content-digest, and bundle-reader path without adding axes or changing the
  frame-value array; exact command `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  CUDA_VISIBLE_DEVICES='' ./.venv/bin/python -m pytest -o addopts=''
  --confcutdir=tests/test_datasets -q tests/test_datasets/test_umcd_schema.py
  -k 'historical_frame_visibility_v1_layer_reads_without_axis_upgrade'` passes
  (1 test).
  This closes only the audited write/read policy rows; target families lacking
  producers, delay-estimator targets, receiver-detectable direct path, and
  physical received-path power remain open.
- **Producer/consumer matrix increment (2026-11-12):** added a versioned
  source inventory for all 19 recognized target definitions, with producer,
  shape/axis and publication policy, stored contract, consumer path, and source
  evidence. A strict JSON Schema now defines the matrix artifact. A focused
  regression validates that schema and enforces complete one-to-one version
  coverage and checks every cited source/schema file. The matrix deliberately
  leaves detector-, timing-, receiver-, and calibrated-power-dependent target
  families without invented producers. Verification: the selected inventory
  regression passes (1 test); JSON parsing and `git diff --check` pass. This
  closes the source-map portion only; E03 remains active because target
  producer/read round trips, malformed-shape cases, and remaining analytical
  consumer checks are not yet complete.
- **NR trajectory-array publication guard (2026-11-12):** the matrix review
  found that the NR writer rejected legacy CFR axes but could still admit
  historical frame-vector geometric-visibility targets as if they were one
  timestamp record. NR publication now rejects the axis-unbound trajectory
  geometric-visibility v1 and F04 dynamic-visibility v2/v3 definitions while
  leaving existing NR reads intact. The regression confirms v1 fails before
  publication and parameterized cases verify refusal plus historical
  content-addressed readback for all three versions. The focused writer/read
  guard passes (3 tests); this is storage compatibility only, not channel
  qualification.
- **Static F03 visibility store contract (2026-11-12):** added a canonical
  `EpisodeBundle` write/read regression for visibility v2 plus a malformed
  `(1, 1)` singleton-shape rejection case. The focused constructor and
  content-addressed round-trip selection passes (2 tests). This closes that
  producer/storage boundary only; loaded-scene and geometry truth remain
  unqualified.
- **NR scalar target preservation (2026-11-12):** a canonical trajectory
  geometric-visibility v2 round trip exposed that the NR storage snapshot
  promoted zero-dimensional values to `(1,)`, violating the per-record scalar
  contract. `_as_numpy()` now takes an owned copy of 0-D arrays without changing
  rank; non-scalars remain contiguous snapshots. The v5 scalar LOS round-trip
  regression passes; the broader NR record/store suite passes (105 tests).
  The focused UMCD visibility/store selection passes (2 tests), and the
  inventory/legacy-target checks pass (13 selected tests). Ruff, format,
  compilation, JSON parsing, the ledger validator and `git diff --check` pass.
  This corrects storage shape only; LOS geometry is not qualified.
- **Standardized profile proxy v1 consumer correction (2026-11-12):** source
  inspection found `umcd.standardized.profile-los-proxy.v1` is actively emitted
  by `build_standardized_pusch_dmrs_records()` and admitted as a fixed `(1,)`
  NR target; the earlier inventory entry incorrectly called it producerless
  and read-only. The inventory now records the exact producer and current NR
  route. A schema-v5 content-addressed write/read regression passes alongside
  the geometric-visibility scalar regression (2 tests). This remains a
  profile-category proxy, not geometric visibility.
- **RT CFR v2 NR persistence (2026-11-12):** added schema-v5 write/read
  round trips for absolute-delay CFR v2 and symbol/Doppler CFR v2, asserting
  exact complex arrays, target names, and digest-bound axis definitions. Both
  focused cases pass. These tests establish the NR storage contract only;
  native path/CFR equivalence remains an external G2/G3 qualification item.
- **Legacy dynamic visibility reader compatibility (2026-11-12):** added
  content-addressed `EpisodeBundle` readback for superseded dynamic visibility
  v2 and v3, verifying historical frame arrays retain their original values
  and axisless definitions after canonical writes were disabled. Both focused
  cases pass; no axes are inferred.
- **Read-only target legacy rank matrix (2026-11-12):** added handcrafted,
  manifest- and object-digest-valid UMCD bundles for all nine versions blocked
  from canonical publication plus RMS v1. Ten parameterized cases verify the
  historical values, ranks and axisless definitions survive bundle readback;
  they do not infer axes or assert physical truth.
- **UMCD CFR frequency-grid admission (2026-11-12):** `EpisodeBundle` now
  checks every declared `effective_subcarrier` target axis against the parent
  episode's exact subcarrier-grid length. Four parameterized cases cover the
  standardized, TDL-PUSCH, RT absolute-CFR, and RT symbol-CFR v2 contracts;
  each rejects a mismatched grid, accepts a matching one, and round-trips
  values plus declared axes through the content-addressed UMCD store. This
  closes a storage-shape invariant only and does not establish numerical CFR
  truth. The combined NR/store/target/UMCD schema slice passes 289 tests.
- **Positive absolute-frequency admission (2026-11-12):** network episode,
  physical-channel view, frequency-domain observation, path-oracle CFR,
  standardized TDL/CDL record, and Scenario GBSM record now reject a carrier
  plus baseband offset that is nonpositive or nonfinite. This closes the shared
  RF-grid input invariant before path phase evaluation; it does not verify a
  simulator or calibrated frequency reference. The UMCD-schema and path-
  counterfactual modules pass 146 tests, the standardized software module
  passes 58 tests with 2 native cases deselected, and the Scenario GBSM store
  module passes 19 tests. Changed modules pass Ruff, format, and compilation.
- **G6 horizon-indexer consumer (2026-11-12):** the receiver-only RT trajectory
  evaluator now has an opt-in `horizons_s` route through the canonical
  `build_trajectory_horizon_targets()` indexer. It uses exact elapsed-time
  matches, retains frozen split and explicit/frozen-gap controls, and preserves
  the prior adjacent-sample behavior when horizons are omitted. Nine focused
  tests pass. This closes one helper-consumer edge only; no forecast criterion,
  training route, or G6 scientific qualification is established.
- **G6 runnable multi-horizon spec (2026-11-12):** per-trajectory specs can
  freeze elapsed-time horizons, timestamp tolerance, and a maximum inter-sample
  gap. The CLI records the applied design alongside each exact input-spec
  digest, runner source digest, UTC generation time, and CLI command; it
  validates the v2 report schema. Its focused evaluator/CLI slice passes 16
  tests. This is a
  fixed-baseline workflow only; it makes no qualification claim.
- **Scope:** identify concrete gaps in same-path reconstruction, units/gauge,
  dtype/masks, missingness, arrays, retry/failure state, and model-input
  consumers. Each change must name an original requirement and downstream
  consumer.
- **Deliverable:** closed implementation cards for verified gaps plus exact
  remaining qualification dependencies; no broad rewrite.
- **Verification:** focused round-trip, analytic-oracle and failure-path checks
  for changed contracts.
- **Disposition:** the current-target producer/consumer software audit is
  recorded in `TARGET_FAMILY_PRODUCER_CONSUMER_MATRIX.md`, including enforced
  value domains, shapes, axes, missingness/source replay, and the limits where
  physical meaning cannot be admitted. The current RMS-delay v2 source replay
  check is covered by 136 UMCD schema tests and 17 RT-network tests. This does
  not close PKG08-008: future target families and physical reference evidence
  remain open.
- **Next:** E04's G5 receipt binding cannot be implemented as an admission
  contract until the exact receiver scope, sourced acceptance criteria, powered
  independent-channel design, and independent reviewer authority are frozen.
  While that task is externally defined, continue the remaining manual
  package-0 history crosswalk (PKG00-002) and semantic/dynamic helper-route
  audit (PKG00-003); the multi-horizon target-indexer production edge is now
  wired through the fixed-baseline evaluator. Receiver-estimated
  delay, detectable direct path, and physical received-path power remain open
  until detector/timing/calibration criteria are frozen. Native path
  coefficient/reference comparison remains a separate G2/G3 qualification
  item.

### E04 — bounded receiver integration and G3 readiness

- **Dependencies:** E01 task-level scope audit; existing G5 receiver work and
  task-specific G3 acceptance definitions.
- **Current evidence:** the 2026-11-12 [G5 receiver support matrix](RECEIVER_SUPPORT_MATRIX_2026-11-12.md)
  maps PUSCH, sample-domain impairment, multi-link, SSB/PBCH, PRS, and SRS
  receiver paths to their implemented boundaries and named blockers. It records
  the pinned Sionna 2.2.0 frequency-hopping/transform-precoding limitation and
  explicitly separates software support from qualification. The existing G3
  joint contract freezes complex CSI, delay/angular, path-equivalence, and
  receiver-impact margins; each run still needs its own config/strata and
  powered sample-size contract. Caller-resolved intra-slot PUSCH grids now have
  a bounded dehop/replay software interface with digest-bound metadata. The new
  adapter verifies the caller-resolved layout against the receiver config and
  forwards the rehomed allocation through the ordinary LS/DMRS decoder; a CPU
  loopback passes and its strict report remains `NOT_QUALIFIED`. This does not
  resolve scheduling, independently generate hop-specific DMRS, implement
  transform precoding, or qualify receiver performance. No receiver or G3
  qualification run occurred.
  A dependency-ready sample-domain workflow now connects channelized per-link
  captures to composition, seeded receiver AWGN and `decode_noisy_network_pusch`
  through `run_network_pusch_receiver()`. Its strict receipt binds the exact
  waveform/configuration and implementation hashes and remains
  `OBSERVED_NOT_QUALIFIED`; the 32-test network receiver module passes on CPU
  with one unavailable-CUDA case skipped. Calibrated link generation, budget-
  to-FIR construction, SINR truth, synchronization/acquisition, powered G5
  acceptance and G3 receipt admission remain open. No qualification run
  occurred.
  The G3 contract-v4 path already binds the exact candidate/challenger configs,
  waveform, link budget/noise and outcome-row execution digest; four current
  preflight regressions pass for those bindings. No duplicate binding interface
  was added. Existing G5.09 signed contracts cover a synthetic two-path scope;
  they do not provide a general G5-qualified receiver receipt for G3 admission.
  A v3 contract now binds the exact receiver, decoder and waveform context in
  addition to timestamped acceptance criteria. It requires independent-reviewer
  policy admission and a detached signature, and emits CPU v17/CUDA v25 curve
  schemas. The 68-test acceptance and audit-contract suites and the 30-test
  retained-result auditor suite pass; Ruff, format, compilation, requirement
  ledger validation, and all three new schemas validate. Pure context assembly
  and auditor replay are covered; the receiver module import stalled in the
  local Sionna stack before native runner tests ran. This does not
  qualify a receiver mode or define the powered G3 sampling design.
  Source audit confirms that G3 contract v4 binds its criterion family,
  configurations, waveform, link budget/noise, powered sample design and raw
  path criteria, but its run manifest and receiver-impact evaluator do not
  validate an external G5-qualified receiver receipt. The manifest remains
  `NOT_QUALIFIED`; synthetic G5 v3 acceptance cannot satisfy that prerequisite.
  The complex-CSI scorer now uses `phase_comparison_disposition()` for records
  that carry paired physical-view contracts validated by strict schema
  `evaluation/schema/complex-csi-channel-view-contract.v1.schema.json`,
  records the declared-gauge audit, and rejects mismatched or masked views; 50
  focused CSI/joint-evaluator tests pass. Legacy raw-array diagnostics remain
  accepted with `NOT_PROVIDED`, and the contract does not establish physical
  phase calibration.
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
- **Next:** freeze the task-specific G5 receiver scope, scientifically sourced
  acceptance criteria, powered independent-channel design, and independent
  review authority for the exact receiver/waveform needed by G3. Then add a
  verifiable G5 receipt binding to a new G3 contract version before enabling
  signed G3 receiver-impact admission. Existing synthetic G5 curves cannot be
  promoted to that receipt. Do not label a new receipt qualified until its
  result auditor, source/runtime identity, reviewer authority, and exact
  receiver scope are verified.
  For PUSCH hopping, replayable TS 38.211 V19.2.0 Table 6.4.1.1.3-6 timing,
  port-0 DM-RS sequence/reference-grid generation, and a one-layer local
  allocation mapper now cover bounded A/B rows through seven symbols per hop.
  The mapper consumes caller-resolved per-hop modulation symbols in the
  frequency-subcarrier-then-symbol order and feeds the existing carrier-grid
  composer. It now also returns the exact boolean data-RE roster and binds its
  packed-bit digest in the replayed allocation metadata. Next implement a
  per-call estimator/demapper route that consumes the standards-derived pilot
  and data masks without mutating shared Sionna objects; its interface must
  preserve the mapper's active allocation span and reject unsupported CDM,
  PT-RS, or puncturing cases. Then bind grant/UL-SCH coding. The pinned Sionna
  receiver remains non-hopping and transform precoding remains unsupported.
  Do not admit a G5 run from the support matrix or software loopbacks alone.

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
