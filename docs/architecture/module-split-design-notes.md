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

### `experiments/ordered_impairments_receiver.py` (split reverted; deferred until approved)

- Base path: `experiments/g5_ordered_impairments_receiver.py`.
- Base digest: `d6c5b9791e30f10e9efbc07a9e0409fe89ef618c7eabe06ffd49151e0640309a`.
- Current digest (single file, restored to its pre-split form): `experiments/ordered_impairments_receiver.py`.
- Not in any evaluator roster. Eleven retained results name the path, and their recorded source digest (`95f130de…`) matches no tracked version of the file.
- History: the split was committed as `846af03` (into `experiments/ordered_impairments/`) with a facade at the original path, and `1a3b8ea` added `provenance.implementation_sha256`. Both were made before this note and before any rebinding entry, which breaks the ordering rule. The split was therefore reverted to the pre-split file and test (restored from `25ff349`), and it stays deferred.
- To proceed: approve the split, record a rebinding entry for the superseded digest, then reapply `846af03` and `1a3b8ea`. A rebinding records supersession, not equivalence.

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
