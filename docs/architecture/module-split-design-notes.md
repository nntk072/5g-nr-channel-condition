# Module split design notes

Modules over 1,500 lines that were **not** split in the production-structure
refactor, and the reason for each. Digests are SHA-256 of the file bytes.
"Base" is the pre-refactor commit `016168e`; "current" is the tree before the
split step. A split of any module below requires its rebinding entries to be
written first, as described in the reference at the end.

## Split completed in this step

Modules split into concept packages in the earlier steps (no retained digest binds them):

| Module | Lines before | Result | Digest change |
| --- | ---: | --- | --- |
| `datasets/split_policy.py` | 1724 | `datasets/split_policy/` (`policy`, `registry`) | file removed, no retained evidence binds it |
| `datasets/episode_store.py` | 1632 | `datasets/episode_store/` (`episode`, `storage`) | file removed, no retained evidence binds it |
| `integrations/sionna/adapter.py` | 1647 | `integrations/sionna/adapter/` (`generate`, `raytrace`) | file removed, no retained evidence binds it |
| `integrations/sionna/network_phy.py` | 1587 | `integrations/sionna/network_phy/` (`records`, `operations`) | file removed, no retained evidence binds it |
| `scripts/audit_channel_split_lineage.py` | 2198 | `scripts/audit_channel_split_lineage/` | file removed, no retained evidence binds it |
| `integrations/sionna/srs/srs_constants.py` | 1957 | `integrations/sionna/srs/srs_constants/` (six concept modules) | file removed, no retained evidence binds it |

Split in the final step:

| Module | Lines before | Result | Digest change |
| --- | ---: | --- | --- |
| `experiments/ordered_impairments_receiver.py` | 2473 | `experiments/ordered_impairments/` (eight concept modules); the original path is a facade with the CLI and the result writer | the facade's path is unchanged, so the recorded source path of the 2026-10-23 fixtures still resolves to it. Its digest changes. No retained evidence in the tree records the new digest. |
| `experiments/audit_ordered_impairments_result/waveform.py` | 829 | seven concept modules (`constants`, `hashing`, `preregistration`, `evidence_files`, `references`, `waveform`, `replay`) | no retained digest records these files |

Narrowed binding and its mitigation: `build_hash_bound_result` hashes the facade file
as the producer source. After the split, the implementation lives in
`experiments/ordered_impairments/`. New results therefore also record
`provenance.implementation_sha256`, a map from each module path to its SHA-256.
Results written before this change have no such field, and the verifier does not
read the modules, so it cannot check them. That is the reason the implementation
binding is described as narrowed, not preserved.


## Deferred: single-file source binding (attempted, then reverted)

`receivers/nr/pusch.py` (1542 lines) was split into a package, then reverted.
Retained results record it as one file (for example
`docs/research/results/delay-domain-dft-confirm-20261024.json`, key
`receivers/nr/pusch.py`), and about 18 experiment and audit tools hash that
single path when they run. A package has no single digest, so each of those
tools would fail, not report drift. The split needs a source-set redesign
and a rebinding for every tool, and it is deferred.

## Deferred: digest-bound by a frozen roster

These modules are listed in `EVALUATOR_SOURCE_PATHS` (V1, V2 or V4) in
`evaluation/joint_evidence_run_manifest.py`. The rosters are contract data that
the run manifest binds, so splitting a module changes the bundle the manifest
verifies. Splitting them needs a new roster version, not a refactor.

| Module | Lines | Current digest | Base digest | Blocked by |
| --- | ---: | --- | --- | --- |
| `evaluation/sample_size_planning.py` | 2231 | `886f9211e51a1014…` | `990a8be9cee63209…` | V2/V4 evaluator roster |
| `evaluation/acquisition_package.py` | 1666 | `0a8db8921fe180f2…` | `07bb9da6c12b8650…` | V1/V2/V4 evaluator roster |
| `evaluation/joint_evidence_run_manifest.py` | 1697 | `92b065d5978cd4f7…` | `c44ab0d3ee9e5ad1…` | lists itself in the roster |

## Deferred: retained result binds the module, or the module is a frozen roster

| Module | Lines | Reason |
| --- | ---: | --- |
| `receivers/nr/pusch.py` | 1542 | Above the 1,500-line target. Retained records and about 18 tools hash this single file; see the first deferred section. |
| `experiments/audit_ordered_impairments_result/independent.py` | 1624 | Two audit functions of about 600 and 970 lines hold the control flow. Splitting them means decomposing those bodies, which changes control flow, not only file layout. Deferred until a behaviour-equivalence review exists. The helpers the two functions call were split out of `waveform.py` in this step. |
| `scripts/audit_umcd_corpus.py` | 1460 | Under the threshold; not split. |

## CDL covariance family (not merged)

`evaluation/cdl_*.py` (nine modules) is kept as nine flat modules. Each module is
an independent covariance reference for a frozen protocol. Moving them into a
package, or sharing their helpers, would change the independence the protocols
rely on. In addition, `docs/research/REQUIREMENT_LEDGER.json` names three of them
by path, and changing those paths needs a recorded amendment to a protected
authority. A package move was tried and reverted; it is not in the branch
history.

## Rebinding procedure for a digest-bound split

1. Record the current and base digests of every module in the split (table above).
2. Split, then compute the new digests of every new file.
3. Add one `source_digest_rebindings` entry per old digest that retained evidence
   records, mapping it to the digest of the file that now carries those bytes.
   Never edit an existing entry in place.
4. Update the roster constant only with a new roster version, and keep the old
   roster unchanged so frozen manifests still verify.
