# Naming exceptions

The refactor removed project-phase tokens (`g3`, `g4`, `g5`) and the `_v1`
suffix from Python function and class names, module paths and file names. The
names below keep a phase token on purpose. Each row gives the reason the name
stays. A frozen document, a frozen schema, or a retained result refers to it, and
frozen bytes cannot change.

## Function and class names kept

| Name | Where it still appears | Reason it stays |
| --- | --- | --- |
| `verify_g3_joint_evidence_run_manifest` | `evaluation/joint_evidence_run_manifest.py`; tests; `docs/research/JOINT_EVALUATOR_ARTIFACT_V1.md` | The frozen contract document names this function. Its text cannot change, so the function keeps the name. |
| `plan_g3_independent_acquisition_units` | `evaluation/sample_size_planning.py`; `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md` | The frozen master plan names this function. Its text cannot change. |
| `compose_g3_joint_sample_size_design` | `evaluation/sample_size_planning.py`; scripts; tests; `docs/research/IMPLEMENTATION_BACKLOG.md` | The frozen backlog names this function. It is also in the digest-bound evaluator source set, so a rename would need a rebinding first. |

No class name carries a phase token.

## Identifier-like string constants kept

| Constant | Where it appears | Reason it stays |
| --- | --- | --- |
| `target_joint_g3_assurance` | `datasets/schema/joint-sample-size-design-composition.v1.schema.json`; `datasets/schema/qualification-run-contract.v3.schema.json`; `datasets/schema/qualification-run-contract.v4.schema.json` | A schema `const` and enum value. It is a contract identity, so changing it changes the schema's meaning. |

## Function and class names renamed in this pass

| Old name | New name | Note |
| --- | --- | --- |
| `test_episode_metadata_schema_covers_lineage_and_legacy_v1` | `test_episode_metadata_schema_covers_lineage_and_legacy_payloads` | Test name only; no frozen text names it. |
| `test_legacy_v1_target_definition_cannot_be_republished_as_target_v2` | `test_legacy_target_definition_cannot_be_republished_as_target_v2` | Test name only. |
| `test_nr_standardized_profile_proxy_v1_roundtrips_fixed_singleton_schema_v5` | `test_nr_standardized_profile_proxy_roundtrips_fixed_singleton_schema_v5` | Test name only. |
| `migrate_legacy_target_definition_v1` | `migrate_legacy_target_definition` | Production function, renamed in `1ab90b8`. No frozen artifact referred to it. |

## Version-suffixed names outside the rename scope

A `_vN` suffix with N ≠ 1 is a schema or contract version, not a project-phase
token, so these names are outside the goal's `g3`/`g4`/`g5`/`_v1` rule. They are
listed so the list is complete.

| Name | Reason it stays |
| --- | --- |
| `build_joint_acceptance_family_v2`, `_v3`, `_v4` | Each builds a frozen family version. |
| `build_qualification_run_contract_v3`, `_v4` | Each produces a frozen contract version. |
| `release_v2` | Names a released split-policy version. |
| `_validate_observable_stage_receipt_v2` | Validates a frozen receipt version. |

## String values kept

About 119 string literals still carry phase tokens. They are persisted values,
so renaming them would change recorded output, not just code. They are kept in
these groups:

- **Identity strings** such as `umcd.g5_…` and `loslab.g4_…`. These are contract identities (schema `$id` and identity fields) and never change.
- **Evidence keys and status values** such as `g5_09_closed`, which retained results use as keys.
- **Case labels** such as `g4_cdl_*_case` and `g3_*` record names that retained results and schema enums store.

## Rename rule

A name is renamed only when no tracked non-Python file (schema, retained
result, frozen document) refers to it and the new name does not collide with an
existing identifier.
