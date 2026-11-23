# Naming exceptions

The refactor removed project-phase tokens (`g3_`, `g4_`, `g5_`, `G3`, `G4`, `G5`)
from Python identifiers, module paths and file names. The names below keep the
old token on purpose. Each one is recorded with the reason it stays.

## Identifiers kept

| Identifier | Where it still appears | Reason |
| --- | --- | --- |
| `verify_g3_joint_evidence_run_manifest` | `evaluation/joint_evidence_run_manifest.py`, tests, `docs/research/JOINT_EVALUATOR_ARTIFACT_V1.md` | The frozen contract document `JOINT_EVALUATOR_ARTIFACT_V1.md` names this function. Its text cannot change, so the function keeps the name. |
| `plan_g3_independent_acquisition_units` | `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md` and the code that implements it | The plan document names the function. The plan text is preserved unchanged. |
| `compose_g3_joint_sample_size_design` | `evaluation/sample_size_planning.py`, scripts, tests, `docs/research/IMPLEMENTATION_BACKLOG.md` | The backlog document names the function. Also part of the digest-bound evaluator source set, so a rename needs a rebinding entry first. |
| `target_joint_g3_assurance` | `datasets/schema/joint-sample-size-design-composition.v1.schema.json`, `qualification-run-contract.v3.schema.json`, `qualification-run-contract.v4.schema.json` | Schema `const`/enum values are contract identities. Changing them changes the schema meaning. |

## String values kept

About 119 string literals still carry phase tokens. They are persisted values,
so renaming them would change recorded output, not just code. They are kept in
these groups:

- **Identity strings** such as `umcd.g5_…` and `loslab.g4_…`. These are contract identities (schema `$id` and identity fields) and never change.
- **Evidence keys and status values** such as `g5_09_closed`, which retained results use as keys.
- **Case labels** such as `g4_cdl_*_case` and `g3_*` record names that retained results and schema enums store.

## Version suffixes kept

A `_vN` suffix stays when it matches a frozen schema version (for example
`build_joint_acceptance_family_v3` and `…_v4`). Only the phase token is
removed; the version suffix identifies the contract the function produces.

## Rename rule

A name is renamed only when no tracked non-Python file (schema, retained
result, frozen document) refers to it and the new name does not collide with
an existing identifier. Renames were applied at the token level, so string
literals were not rewritten except identifier-like strings used with
`getattr`, `setattr` or monkeypatching.

## Final pass (2026-11-23)

- `migrate_legacy_target_definition_v1` was renamed to `migrate_legacy_target_definition` (commit `1ab90b8`). No frozen artifact or schema referred to it, and the helper inventory is regenerated.
- The remaining production functions with `_vN` suffixes are schema-version names (`build_qualification_run_contract_v3/v4`, `build_joint_acceptance_family_v2/v3/v4`, `_validate_observable_stage_receipt_v2`, `release_v2`). The goal lists only `g3`, `g4`, `g5` and `_v1` as phase words to remove, so these stay.
- Test function names that describe schema versions (for example `test_v2_joint_planner_*`) stay. They describe the contract under test, and renaming them would not change behaviour.
- Kept identifiers in the table above are unchanged in this pass.
