# scripts.audit_channel_split_lineage

Channel-split lineage audit: child artifacts, split rosters, and channel audits.

## Purpose

This package is one module split into concept modules. It keeps the public names of the original module, so callers import from `scripts.audit_channel_split_lineage` unchanged.

## Modules

| Module | Contents |
| --- | --- |
| `__main__.py` | Entry point for the channel-split lineage audit |
| `channel_audit.py` | Channel dataset audits and the command-line entry point |
| `child_artifacts.py` | Child artifact verification for the channel-split lineage audit |
| `split_artifacts.py` | Split roster loading for the channel-split lineage audit |

## Public API (36 names)

Re-exported from `__init__.py` through `__all__`:

`_CHILD_ARTIFACT_ROSTER_SIGNATURE_NAMESPACE`, `_GIT_SHA_RE`, `_INVENTORY_SCHEMA`, `_INVENTORY_SIGNATURE_NAMESPACE`, `_LEGACY_ASSET_RECONSTRUCTION_SCHEMA`, `_MAX_CHILD_ARRAY_HEADER_BYTES`, `_MAX_CHILD_ARTIFACT_TREE_DEPTH`, `_MAX_CHILD_ARTIFACT_TREE_FILES`, `_MAX_CHILD_METADATA_NPY_BYTES`, `_MAX_CHILD_ROSTER_BYTES`, `_REPOSITORY_ROOT`, `_SHA256_RE`, `_SPLIT_AUDIT_SCHEMA_PATH`, `_V3_SEMANTIC_CHILD_ARTIFACT_TYPES`, `_V4_SEMANTIC_CHILD_ARTIFACT_TYPES`, `_V6_MULTI_OWNER_ARTIFACT_TYPES`, `_V6_SEMANTIC_CHILD_ARTIFACT_TYPES`, `_attach_reconstructed_scene_assets`, `_enumerate_regular_files_beneath`, `_hash_inventory_relative_file`, `_load_frozen_inventory`, `_load_legacy_asset_reconstruction`, `_load_split_child_roster`, `_parse_dataset`, `_parse_split_child_roster`, `_read_channel_dataset_child_rows`, `_read_inventory_relative_file`, `_read_stable_regular_file`, `_require_exact_child_object_owners`, `_require_shared_content_owners_share_split`, `_stat_identity`, `_unique_json_object`, `_validate_split_audit_report`, `_load_split_child_artifact_roster`, `audit_channel_datasets`, `main`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.
- Files that retained evidence records by path are listed in `docs/reference/digest-rebindings.json`.

