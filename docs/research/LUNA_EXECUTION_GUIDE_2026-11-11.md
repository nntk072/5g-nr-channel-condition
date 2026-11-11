# Proposed execution guide for Luna

Date: 2026-11-11. Reviewed checkout: `42e9d08` on `main`, with active,
uncommitted run-replay provenance work. This guide proposes execution order;
it does not resume the cleared thread goal, authorize experiments, change
scientific thresholds, or remove requirements from either governing plan.

## Why the execution plan needs revision

The governing plans contain useful scientific definitions and retained
evidence, but their current form leaves the executor to choose scope repeatedly.
At this checkpoint the backlog, research plan, MVP status, and unified plan
total 8,077 lines. Their chronological entries are useful history, but are too
large and repetitive to serve as the active task queue.

The feature map reports 182 items: 130 `code_complete`, 23 `partial`, 29
`absent`, zero `externally_blocked`, and 144 `unmapped`. These are parser
classifications, not an implementation audit. The builder maps checked boxes
to `code_complete` and otherwise-open descriptions without resolved links or a
partial marker to `absent`. For example, the map labels run-replay provenance
absent although its implementation is currently in the worktree. It also
does not identify external blockers unless the description uses its explicit
marker. These counts must not drive scientific or implementation completion.

Gate names also collide: unified-plan G4 means standardized channel conformance,
whereas research-plan Appendix E G4 means strong baselines. The backlog's
instruction to complete packages 0–10 before consolidated validation includes
later model/publication work that itself depends on qualification. This needs
an explicit milestone boundary rather than repeated discretionary exceptions.

Recent G4 evidence covers increasingly large finite covariance families. Such
results are useful within their protocols, but more configurations do not by
themselves resolve G3 measurements, G5 receiver acceptance, release lineage,
or the preserved failed covariance claim. An unlimited "remaining arrays and
orientations" objective has no achievable stopping condition.

## Documents and identifiers

Keep both governing plans and all historical protocols/results. Use this
hierarchy for the proposed execution process:

1. User decisions and the exact acceptance definitions in the governing plans.
2. An approved scope/requirement ledger, proposed below.
3. A short active queue containing task cards whose dependencies are satisfied.
4. An evidence index containing exact artifact/commit identities and outcomes.
5. Historical status entries, consulted when a task references them.

The active, finite task queue is [EXECUTION_QUEUE.md](EXECUTION_QUEUE.md); its
initial requirement index is [REQUIREMENT_LEDGER.json](REQUIREMENT_LEDGER.json)
with schema [requirement-ledger.v1.schema.json](schema/requirement-ledger.v1.schema.json).
The queue records E00 as focused-verified and E01 as the current audit task.

Use `DS.G0`–`DS.G8` for unified dataset gates and `RS.Phase0`–`RS.Phase10`
plus `RS.AppendixE.G0`–`RS.AppendixE.G9` for research phases/gates in new
planning records. Preserve existing protocol IDs and filenames. Do not infer
that equal gate numbers mean equal requirements.

## Milestones and their dependencies

| Milestone | Required outcome | Completion evidence |
| --- | --- | --- |
| M0: establish the executable contract | Finish/review the current replay change; reconcile every backlog item and every named governing-plan requirement into finite deliverables | Preserved source changes, focused verification, explicit requirement/task mapping and blockers |
| M1: complete prerequisite implementation | Required data contracts, generation/storage/receiver interfaces, splits, evaluation and runnable workflow code exist within the declared support envelope | Each required task has source/schema/interface references and meaningful focused verification; downstream gated work is listed explicitly |
| M2: consolidated software verification | Run the repository's current CI-equivalent checks once against the completed prerequisite implementation; repair failures | Exact commit, environment, commands and complete retained results; no use of historical green checkpoints as current proof |
| M3: scoped scientific qualification | Complete applicable frozen G1/G2/G3/G4/G5 evidence with representative inputs, justified sampling and required review | Task/configuration-specific acceptance and provenance; failures/unresolved branches remain visible |
| M4: production and research delivery | Only after applicable gates pass: production corpus, qualified benchmark, advanced model work, research evaluation and reproduction outputs | Release/claim-specific evidence required by both governing plans |

M1 is a prerequisite implementation milestone, not completion of the whole
mission. M3 may remain externally blocked while independent M1 work continues.
M4 requirements stay in the ledger and are not silently removed or marked
complete. Do not require M4 completion before allowing M2.

Focused tests needed to verify an implementation change belong with its task.
Broad CI, native qualification families, and expensive RT generation are
separate named tasks with explicit entry conditions. Tests using signatures
from fixture keys establish software behavior; they do not create independent
reviewer custody or scientific authority.

## Immediate task: finish the existing worktree unit

Task `E00`, linked initially to backlog `PKG01-006`:

- Inspect the current diff and preserve all existing edits. The reviewed dirty
  files are `artifacts/__init__.py`, `artifacts/manifest.py`,
  `experiments/runner.py`, `loslab/__main__.py`, their existing manifest/runner
  tests, and the new `artifacts/run_replay.py`, replay schema, and replay tests.
  Re-inspect before acting; this inventory can change.
- Review the integrated path from CLI/API arguments through started,
  completed, and failed manifests, including retry-parent binding and the
  actual RNG streams used by the runner. Check pre-run failures and interrupted
  runs explicitly rather than assuming every failure has a completed receipt.
- Verify the schema and focused replay/manifest/runner modules. Use the
  installed repository environment; establish the intended Halti host before
  native execution. Do not infer host identity from the pathname alone.
- Record any remaining producer routes outside this change. Finishing the
  common runner does not complete replay provenance for every producer.
- Update the requirement status, feature map and one current checkpoint;
  commit the coherent verified unit and push if already authorized and the
  transport works. Preserve unrelated edits and all local history.

Suggested focused verification after reviewing the code:

```bash
./.venv/bin/python -m pytest --no-cov -o addopts='' -q \
  tests/test_artifacts/test_run_replay.py \
  tests/test_artifacts/test_manifest.py \
  tests/test_experiments/test_runner.py
```

Run Ruff/format checks only on intentionally changed Python files. Validate
the replay JSON schema, regenerate/check the feature map after backlog edits,
and run `git diff --check`. Resolve new findings before claiming task closure.

## Build the finite queue before choosing another experiment

Task `E01` follows E00. Produce one explicit requirement ledger by auditing
the 182 backlog items and joining all named governing-plan requirements,
including items that the backlog does not yet represent. Preserve stable
requirement identities; ordinal checklist IDs alone change when rows are added.

Each requirement needs these fields:

| Field | Required content |
| --- | --- |
| Identity and authority | Stable ID, governing document/section/gate, existing backlog aliases |
| Scope | Dataset or research deliverable; exact supported configurations; mandatory, conditional, or later-stage status with governing justification |
| Implementation | `unknown`, `absent`, `partial`, or `complete`, justified by inspected source and integration; never inferred from a missing link |
| Verification | `not_run`, `pass`, `fail`, or `incomplete`, with exact command/commit/artifact |
| Qualification | `not_applicable`, `not_qualified`, `failed`, `unresolved`, or `qualified_within_scope`, with applicable scientific criteria |
| Dependencies | Exact prerequisite requirement/task IDs and any external inputs/decisions |
| Closure | Deliverable, interface/schema, expected evidence, pass/fail rule and next task |

Record external dependency status separately from implementation state. A
complete verifier can still need calibrated measurements or an external trust
policy. No evidence means `unknown` until inspection, rather than `absent`.
The feature map may summarize this ledger; it must not supply audit conclusions
from prose/checkbox heuristics.

The first dependency-ready tasks after E01 should come from these workstreams,
ordered by their contribution to the runnable prerequisite pipeline:

1. **Split and benchmark integration:** join the existing signed partition
   materializer to its actual experiment consumers; establish whether all
   required site/array/external cohorts and descendants are consumed and
   audited. Close exact row/parent/component and inference-leakage gaps found
   in that path before adding another parallel helper.
2. **Canonical dataset/path/target interfaces:** audit and close the remaining
   G1/G2/G7 contract, same-path reconstruction, dtype/mask, serialization,
   failure/retry, units/gauge and model-input requirements. Use analytical and
   existing retained fixtures before requesting additional RT execution.
3. **Bounded G5 end-to-end receiver workflow:** define the exact required
   waveform/receiver/configuration envelope from the governing plan, map
   existing tests/results, and implement the missing interfaces and criteria
   plumbing. Distinguish backend-unsupported modes from implementation gaps;
   do not silently drop a mandated mode. Escalate an actual scope/backend
   decision once the options and consequences are concrete.
4. **Remaining pipeline and temporal interfaces:** close required DAG,
   provenance/status propagation, causal time/target alignment, resource
   accounting and evaluator consumers. Keep required dynamic/counterfactual
   deliverables visible even when native evidence depends on later gates.
5. **Scientific readiness:** prepare G3 acquisition/criterion packages and a
   finite G4 applicability matrix. Identify external measurements, authoritative
   inventories and independent reviewer inputs by deliverable and owner.

These are audit workstreams, not five unlimited "complete everything" task
cards. E01 must turn each into finite tasks with exact paths, dependencies and
exit tests before an executor selects work. Existing implementation may close
a requirement by verified reconciliation; do not reimplement it.

## G4 stopping and failure rules

Before any new G4 family, name the original requirement or exact configuration
used by a required downstream workflow that the new family will qualify.
Record whether existing evidence already covers it. Freeze the required
finite applicability matrix for M1/M3. Arbitrary arrays, every continuous
orientation, and unlimited profile/operating-point combinations are not
executable completion criteria. If an unbounded requirement cannot be turned
into justified scope, present the scope decision to the user; do not invent
an exclusion or another automatic extension.

The retained 2x2 single-V joint TX/RX covariance family has a frozen failure:
159/160 claims passed and CDL-B missed one interval. Keep that result failed.
The larger-draw same-seed diagnostic and passes from different configurations
do not replace it. Create an explicit disposition task: verify the oracle and
implementation, distinguish a discovered defect from unresolved stochastic
failure, and state what any new protocol would establish. Use fresh audited
seeds if the protocol is revised. Never widen a threshold, pool a replacement,
or call a rerun the first execution.

Retain claim-level evidence before starting a native family. Existing exact
evidence should be indexed/formalized without regenerating the same cohort
unless a specific missing observable makes that necessary. Code/protocol
changes and confirmation evidence must have explicit chronology and identities.

## Hard gates and external blockers

- Production D1 RT and advanced training require the applicable scientific
  gates; a passing synthetic software fixture does not open either gate.
- G3 complex CSI, delay/angular statistics, path truth and receiver impact use
  the existing task-specific acceptance definitions. Freeze the complete
  run-specific contract and sampling design before experiments. G3 receiver
  impact requires the applicable G5-qualified receiver; the two-path fixture
  cannot qualify another receiver configuration by analogy.
- City43 Phase-0 is completed bounded diagnosis from existing path records.
  Do not reopen ray/depth sweeps or designate a finite budget ground truth.
- Etoile, Florence and Munich may be used under the accepted ODbL policy.
  Public redistribution still requires the release-specific source/extract,
  conversion, attribution/source-offer and third-party rights evidence.
  Missing historical records cannot be reconstructed by assertion.
- Calibrated measurements, authoritative physical independence and signer
  custody require actual external evidence. Keep them in an explicit blocker
  register while implementing independent software tasks.

## Task card and executor rules

Every active task card contains: stable ID; linked requirement IDs; current
evidence; dependencies; exact files/interfaces to inspect/change; concrete
deliverable; required verification command; scientific entry conditions;
resource bound where relevant; closure rule; blocker rule; and next task ID.

Keep one active task and a short ordered ready queue. Work on a coherent unit
through implementation, focused verification, evidence and commit. Do not end
an iteration after merely appending a status paragraph when the task still has
an available implementation step. Do not open a new experiment to avoid a
failed task or an unresolved dependency. Poll a verified live job rather than
restarting after an observation timeout.

Publish one concise current checkpoint per unit: requirement IDs closed or
advanced, files/commit, verification, remaining blockers and the next exact
task. Keep numerical result detail in the evidence artifact and link it from
the index. Preserve historical entries; replace their role as the active queue
rather than copying each new paragraph into four plans.

## Suggested continuation instruction after approving the ledger process

> Work as one agent in the verified Halti checkout. Read this execution guide
> and the current task card, inspect/preserve the worktree, and complete E00
> before new work. Then execute E01 to produce the explicit requirement and
> dependency ledger, cross-checking both governing plans without reducing their
> scope. Select subsequent work only from the resulting ready queue, prioritizing
> the required runnable dataset/split/receiver/evaluation path. A task closes
> only when its concrete deliverable and evidence satisfy its exit rule. Keep
> implementation, verification and scientific qualification separate; retain
> failures and external blockers. Freeze a finite G4 scope before adding more
> covariance families. Preserve the City43, ODbL, production-RT and advanced-
> training constraints. Commit coherent verified units and push when authorized
> and transport works. Report the next task ID and any actual user decision
> required. Do not claim the mission is complete until every original required
> downstream phase and evidence deliverable is satisfied.
