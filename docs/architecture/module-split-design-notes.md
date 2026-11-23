# Module split design notes

Modules over 1,500 lines, what was split, and which splits are blocked. Digests
are SHA-256 of file bytes. "Base" is the evidence commit `016168e`; "now" is
the current tree (commit `1ab90b8`).

A split of a digest-bound module needs rebinding entries in
`docs/reference/digest-rebindings.json` before it is implemented. A rebinding
records that an old digest is superseded. It is not evidence that the new bytes
are equivalent.

## Digest-free splits (done)

No tracked retained result, config, or schema names these modules or their base digests.

| Module | Lines before | Now | Digest bound? |
| --- | ---: | --- | --- |
| `datasets/split_policy.py` | 1724 | `datasets/split_policy/` (`policy`, `registry`) | no |
| `datasets/episode_store.py` | 1632 | `datasets/episode_store/` (`episode`, `storage`) | no |
| `integrations/sionna/adapter.py` | 1647 | `integrations/sionna/adapter/` (`generate`, `raytrace`) | no |
| `integrations/sionna/network_phy.py` | 1587 | `integrations/sionna/network_phy/` (`records`, `operations`) | no |
| `scripts/audit_channel_split_lineage.py` | 2198 | `scripts/audit_channel_split_lineage/` | no |
| `integrations/sionna/srs/srs_constants.py` | 1957 | `integrations/sionna/srs/srs_constants/` (six modules) | no |

## Digest-bound splits

### `experiments/ordered_impairments_receiver.py` (split reapplied after rebinding)

- Base path: `experiments/g5_ordered_impairments_receiver.py`. Base digest `d6c5b9791e30f10e9efbc07a9e0409fe89ef618c7eabe06ffd49151e0640309a`.
- Pre-split digest (tree before this split): `786a45b4a1f9145645d239120b915e895d5a9a63f61d007e951ee96f19dd3101`.
- Post-split facade digest: `2e9f96621bc3be1cb1c97b2ce71873afa1adf001cbad56b1fb6530ff45190a34`.
- Split into `experiments/ordered_impairments/` (`constants`, `hashing`, `signal_models`, `statistics`, `paired_probe`, `independent_probe`, `independent_snr_curve`, `waveform_evidence`). The facade keeps the original path and its public names.
- Not in any evaluator roster. Retained results name the base digest, which is recorded in `source_digest_rebindings` (last entry wins) and now resolves to the facade digest.
- Rebinding entries were written before this change (commit history: rebinding commit, then split commit). A rebinding records supersession, not equivalence: the split was checked by its test suite (`tests/test_experiments/test_ordered_impairments_receiver.py`), not by a new scientific run.

### `evaluation/sample_size_planning.py` (deferred)

- Base path: `evaluation/g3_sample_size_planning.py`. Base digest `990a8be9cee63209c9da2aa4c1a62fcb1a47d9b4e169e293636b07988bbaf7a0`. Now `886f9211e51a10144606e27f8b3fb2fc53a991ee858b7e41e2873fb631bc70ad`.
- In the evaluator roster for V2 and V4.
- Blocker: the V4 roster is fixed at 41 entries by the frozen schema `datasets/schema/joint-evidence-run-manifest.v4.schema.json` (`minItems` = `maxItems` = 41). A split adds files outside that roster, so the signed bundle would stop covering the split code.
- Needed to unblock: a new roster version and schema, which is a protocol change, or your decision to accept the narrowed binding.
- Rebinding entries: none added, because nothing is implemented.

### `evaluation/acquisition_package.py` (deferred)

- Base path: `evaluation/g3_acquisition_package.py`. Base digest `07bb9da6c12b86503218f8c087dfd430e4f2363987c4a9c56d1c86fcca056db3`. Now `0a8db8921fe180f2bdaaec0adb20de957944f79ccddd5daf6440dacf2c7159ae`.
- In the V1, V2 and V4 rosters. This module is **not** digest-free, although the goal listed it as one.
- Same blocker as `sample_size_planning.py`.

### `evaluation/joint_evidence_run_manifest.py` (deferred)

- Base path: `evaluation/g3_joint_evidence_run_manifest.py`. Base digest `c44ab0d3ee9e5ad196ca4dfd715f5b120c8c563ec30133c2bf732d257b18f26b`. Now `92b065d5978cd4f783ac05de4576965c15540cc55165900f38bc8664617f9d63`.
- Lists itself in the V1, V2 and V4 rosters. The verifier reads it as part of its own bundle.
- Same blocker.

### `receivers/nr/pusch.py` (deferred)

- Base and now digest: `b6a632a92fbc92a294fe7baf718a69b665635e4e5d0e71f4528b5b4d1333f003`. Unchanged since base.
- Not in an evaluator roster. Seventy-one path references in results and configs, and about 18 research and audit tools hash this file as one unit.
- A split changes the bytes those tools verify. Their checks would have to be rewritten to accept a rebinding, and a rebinding of a split is not an equivalence proof, so this is deferred.

## Deferred for other reasons

- `experiments/audit_ordered_impairments_result/independent.py` (1,624 lines): two functions of about 600 and 970 lines. Splitting them changes control flow and needs a behaviour review.
- `scripts/audit_umcd_corpus.py` (1,460 lines): under the threshold; not split.

## Rebinding procedure for a digest-bound split

1. Record the base and current digest of each module, as in this note.
2. Decide whether the split narrows any signed roster. If it does, the roster change is a protocol change and needs approval first.
3. Add one `source_digest_rebindings` entry per superseded digest that retained records name. Never edit an existing entry.
4. Implement the split only after step 3 is recorded.

## Decision needed: remaining digest-bound splits (2026-11-23)

The four modules below are still over 1,500 lines. They are not split. The
blocker is evidence coverage, not missing rebinding entries.

| Module | Lines | Base digest | Current digest | Bound by |
| --- | ---: | --- | --- | --- |
| `evaluation/sample_size_planning.py` | 2,231 | `990a8be9cee63209c9da2aa4c1a62fcb1a47d9b4e169e293636b07988bbaf7a0` | `886f9211e51a10144606e27f8b3fb2fc53a991ee858b7e41e2873fb631bc70ad` | V2 and V4 evaluator rosters |
| `evaluation/joint_evidence_run_manifest.py` | 1,697 | `c44ab0d3ee9e5ad196ca4dfd715f5b120c8c563ec30133c2bf732d257b18f26b` | `92b065d5978cd4f783ac05de4576965c15540cc55165900f38bc8664617f9d63` | V1, V2 and V4 rosters; the verifier reads its own bundle |
| `evaluation/acquisition_package.py` | 1,666 | `07bb9da6c12b86503218f8c087dfd430e4f2363987c4a9c56d1c86fcca056db3` | `0a8db8921fe180f2bdaaec0adb20de957944f79ccddd5daf6440dacf2c7159ae` | V1, V2 and V4 rosters |
| `receivers/nr/pusch.py` | 1,542 | `b6a632a92fbc92a294fe7baf718a69b665635e4e5d0e71f4528b5b4d1333f003` | `b6a632a92fbc92a294fe7baf718a69b665635e4e5d0e71f4528b5b4d1333f003` (unchanged) | about 18 audit and receiver tools hash the file as one unit |

The goal lists `evaluation/acquisition_package.py` as digest-free. It is not: it
is in three rosters.

### Why the split is not implemented

A split keeps each roster path by leaving a facade at the original path. The
facade's digest changes, so a rebinding entry can be recorded for it. That does
not cover the code. The moved code lives in new files that are not in any signed
roster, and the roster schema fixes the count at 41 entries
(`datasets/schema/joint-evidence-run-manifest.v4.schema.json`,
`minItems` = `maxItems` = 41). Adding the new files to the roster changes a frozen
contract. Leaving them out means future signed evidence no longer covers the code
that produces the scores. That narrows the binding of the evaluator, which the
constitution does not allow a lower layer to do.

For `receivers/nr/pusch.py` the problem is the same. Its retained checks compare
one file's digest. After a split, the code those checks verify is spread across
several files, and only the facade is checked.

### Options for the owner

1. Approve a roster-protocol change that adds the new submodules to each roster, with a new schema version and a rebinding for every changed bundle digest. Then the split is a normal digest-bound change.
2. Approve the narrowed binding: split with a facade, record rebindings for the facade digests, and accept that future evidence does not hash the submodules.
3. Keep the four modules whole and record them as an accepted size exception.

Until one is chosen, no code in these modules changes.

## Implemented split: sample planning and run manifest (narrowed binding)

The owner chose the narrowed split (facade at the original path, submodules
outside the signed roster). Rebinding entries were recorded before the code
commit.

- `evaluation/sample_size_planning.py`: facade (99 lines), digest `fcff4edffa41ab4135a257f2321c7c844c782ef7ead53c2953b06477d26d2dc3`; previous `886f9211e51a10144606e27f8b3fb2fc53a991ee858b7e41e2873fb631bc70ad`; base `990a8be9cee63209c9da2aa4c1a62fcb1a47d9b4e169e293636b07988bbaf7a0`. Package `evaluation/sample_size/`: `constants`, `binomial`, `independent_units`, `joint_design`, `bootstrap`. Dependency order: constants, binomial, then the planners; no cycles.
- `evaluation/joint_evidence_run_manifest.py`: facade (132 lines), digest `fcbd0a189bc78a6d422f15ff344b232952302838c59c073c827c0812d92a3ce3`; previous `92b065d5978cd4f783ac05de4576965c15540cc55165900f38bc8664617f9d63`; base `c44ab0d3ee9e5ad196ca4dfd715f5b120c8c563ec30133c2bf732d257b18f26b`. Package `evaluation/run_manifest/`: `constants`, `common`, `sources`, `signatures`, `contracts`, `build`, `verify`, `cli`; the `__main__` guard stays in the facade.
- Roster coverage: the facade paths keep the V1, V2 and V4 rosters unchanged. The new submodules are not in any roster, so signed evidence no longer hashes them. This is the narrowed binding the owner chose.
- Verifier change: the source-file check in `evaluation/run_manifest/verify.py` accepts a recorded digest that is rebound to the current digest (`artifacts/rebinding.py: recorded_digest_resolves`). The recorded size belongs to the superseded bytes and is not compared for rebound entries.

### Implemented split: acquisition package (narrowed binding)

- `evaluation/acquisition_package.py`: facade, digest `05e11caeb31a8bb17d3333ab6152592b8ef4881336d8283c4c9e92e58d76e185`; previous `0a8db8921fe180f2bdaaec0adb20de957944f79ccddd5daf6440dacf2c7159ae`; base `07bb9da6c12b86503218f8c087dfd430e4f2363987c4a9c56d1c86fcca056db3` (`evaluation/g3_acquisition_package.py`). It is in the V1, V2 and V4 rosters; the facade path keeps the rosters unchanged.
- Package `evaluation/acquisition/`: `constants`, `noise`, `canonical`, `files`, `index`, `verify` (the verifier is about 1,400 lines, under the limit).
- The submodules are outside the signed roster, the same narrowed binding as above.

### Implemented split: NR PUSCH (narrowed binding)

- `receivers/nr/pusch.py`: facade (keeps the original path, which about 18 tools and retained results name), digest `f13ed883a2d546bb5e90d0ac7cb9d314c88c2b05e9f5719fd7c2f5c40af42895`; previous `b6a632a92fbc92a294fe7baf718a69b665635e4e5d0e71f4528b5b4d1333f003` (unchanged base).
- `NRPUSCHConfig` and `NRChannelEstimate` stay defined in the facade: their module path is part of the audited model-input contract identity (`datasets/model_input_audit.py`), so moving them would change that identity.
- Package `receivers/nr/pusch_core/`: `constants`, `types` (grid dataclasses), `dmrs`, `mapping`, `estimation`, `hopping`. Dependency order: constants and types, then dmrs, then mapping and hopping, then estimation; no cycles.
- The submodules are outside the digests that the audit tools name, the same narrowed binding as the other splits.

## Protocol v5: split submodules join the signed roster (owner-approved, 2026-11-23)

The owner chose a protocol change over the narrowed binding for the sample-planning,
run-manifest and acquisition modules. The change adds a new manifest version; it does
not edit any frozen file.

- New schema `datasets/schema/joint-evidence-run-manifest.v5.schema.json`, digest `7411d406e2c5ec9d340dc0c7b95afddb1e2e31a8593943a528cbe4e21caef16f`. Its `$id`, title and `schema` const are new; `evaluator_sources` is pinned to 63 entries. The V4 schema (`joint-evidence-run-manifest.v4.schema.json`, base bytes, digest `d6022a555f0d…`) is unchanged.
- Roster `V5_EVALUATOR_SOURCE_PATHS` (`evaluation/run_manifest/constants.py`, digest `55ea618f48273d61b2e67c973ce23945355e70fca3edc70a33853438357831d2` at the commit that adds it) equals the 41 V4 entries plus `NEW_V5_SOURCE_PATHS`: 22 files from `evaluation/sample_size/`, `evaluation/run_manifest/` and `evaluation/acquisition/`.
- The builder writes v5 manifests when a runtime inventory is included. The verifier accepts v1 to v5; v4 keeps its own schema and roster, so earlier v4 manifests still verify. Strict signed reviewer admission accepts v4 or v5.
- No retained signed manifest exists in the tree, so no bundle rebinding is needed for v5.
- Not in any roster: `receivers/nr/pusch_core/` (PUSCH is not an evaluator source; its narrowed binding stands).
