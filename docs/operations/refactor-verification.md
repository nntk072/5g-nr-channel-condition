# Production-structure refactor: verification record

This record is the final outcome of the production-structure refactor, from the
evidence base `016168e` to the head `f6e4cc3`. It states what was done, what was
deferred and why, every gate result exactly as logged, the native and torch
results, and the changed digests. Scientific status is not changed by anything
here: E04 remains engineering DONE and scientific NOT_QUALIFIED.

Path note: the goal named this file `docs/operations/REFACTOR_VERIFICATION.md`
and the naming note `docs/architecture/NAMING_EXCEPTIONS.md`. They are lowercase
here (`refactor-verification.md`, `naming-exceptions.md`) under the repository's
documentation naming rule.

## Gate results on `f6e4cc3`

Commands are the repository's gates (`./tools/check <gate>`). Exit codes and
summary lines are copied from the gate logs.

| Gate | Result |
| --- | --- |
| `lock` | exit 0 |
| `lint` | `All checks passed!`; `769 files already formatted`; exit 0 |
| `typecheck` | `Success: no issues found in 404 source files`; exit 0 |
| `architecture` | exit 0 |
| `test` | `3924 passed, 170 skipped, 1 warning in 332.61s (0:05:32)`; exit 0 |
| `neural` | `3924 passed, 170 skipped, 1 warning in 601.55s (0:10:01)`; coverage `75.75%` (threshold 75%); exit 0 |
| `build` | wheel `5g_nr_channel_condition-0.2.0` built; `import-smoke-ok`; exit 0 |
| `agent` | `180 passed in 12.15s`; exit 0 |
| `docs` | sphinx-build with warnings as errors; exit 0 |

The 170 skipped tests in `test` and `neural` are reported as skipped, not as
passes. They are the opt-in tests whose markers or optional dependencies are
absent from CI's environment.

## Torch tests, `envs/py312-gpu`, `CUDA_VISIBLE_DEVICES=-1`

Command: `python -m pytest --no-cov -o addopts= -q -rs -m "not native_sionna"`.

- Result on `f6e4cc3`: `3999 passed, 4 skipped, 1272 deselected, 1 warning in 343.44s (0:05:43)`; exit 0.
- Skipped, reported as skipped: 4 tests. Two are `test_classical.py` (xgboost not installed in this environment) and `test_observable_multipath.py` (CUDA unavailable by design).
- 1272 deselected are the native tests, which run in the native section below.

## Native parity, CUDA hidden, against the base extract

Command: `python -m pytest -m native_sionna --no-cov -o addopts= -q -rfE`, with
`CUDA_VISIBLE_DEVICES=-1` and `DRJIT_LIBLLVM_PATH` set to the LLVM 18 library.
The base extract ran in the same environment.

| Tree | Passed | Failed | Skipped | Deselected |
| --- | --- | --- | --- | --- |
| Current `f6e4cc3` | 1223 | 6 | 43 | 4003 |
| Base `016168e` (extract, earlier run) | 1215 | 14 | 43 | 3997 |

Failures on the current tree, all also present at base:

1. `test_admission_native.py` `[C60]`, `[B100]`, `[A30]`: the admission code and its
   expected message disagree. This is scientific admission behaviour; the expectation is not changed.
2. `test_cdl_joint_tx_rx_spatiotemporal_covariance_native.py` `[B]`: tolerance miss
   against the independent reference. Changing the tolerance would change a
   scientific acceptance criterion.
3. `test_cdl_spatial_covariance_4element_native.py` `[E]`: tolerance miss against
   the independent reference, same reason.
4. `test_rt_trajectory_native.py::test_cpu_rt_path_retraces_and_stores_receiver_only_trajectory`:
   the test asserts the geometric-LOS oracle is an "independent Mitsuba" oracle; the
   code describes a digest-bound NumPy inventory cross-checked by Mitsuba. That is a
   claim about oracle independence, so it is not changed here.

Base-only failures, no longer present on the current tree: the three dual-pol
cases, the CDL-D case, the two K-factor cases, and the two statistical-confidence
cases. Some of these changed in this branch, and the change is recorded as such:

- The statistical-confidence cases read the wrong axis (size-1 axis instead of the tap axis) and raised `IndexError` before any assertion. The index now reads the second tap as the test's comment intends (commit `f6e4cc3`). The tolerance and the expected value are unchanged.
- The dual-pol cases and K-factor cases were fixed by earlier test expectation and digest-pin corrections in this branch (`96a41ed`, `25ff349`, `4f585ba`).

Parity criterion. The refactor adds no native failure: every current failure
also fails at base, and the failing set shrinks from 14 to 6. Native parity
against the baseline is therefore met as "no regression"; it is not "all native
tests pass". The six open failures need an owner decision, because each one is a
scientific acceptance or claim question.

## Goal items

1. **Helper wiring.** `scripts/audit_helper_wiring.py` resolves package
   re-exports to their defining submodule. The helper count is `254`, and
   `tests/test_scripts/test_audit_helper_wiring.py` passes. The disposition mix
   changed (245 manual review, 1 no direct reference, 8 route mapping required)
   because the frozen research text, which names several helpers as routes, is
   back at base bytes. The count and the test are the stored contract.
2. **Naming.** The only production identifier with a `_v1` phase suffix,
   `migrate_legacy_target_definition_v1`, is renamed (`1ab90b8`). The three `g3`
   production names are kept because frozen artifacts or schemas reference them.
   The `_vN` schema-version names stay, since the goal lists only `g3`, `g4`,
   `g5` and `_v1`. Each kept name, with its reason, is in
   `docs/architecture/naming-exceptions.md`.
3. **Merges.**
   - Done: the nine CDL covariance modules are one package, `evaluation/cdl_covariance/`. Helper bodies that were byte-identical in every family that used them are in `shared.py`; families whose helpers differ keep their own. Importers are updated; the CDL unit tests pass (197 passed in the CDL evaluation subset).
   - Deferred: the `evaluation/g3_*` and `g4_*` modules are already concept-named, and the remaining evaluator modules are in the digest-bound roster (see item 4). The audit scripts (`experiments/audit_tdl_*`) each audit a different frozen protocol, so merging them would change audit provenance. Not merged.
4. **Splits.**
   - Done, digest-free: `datasets/split_policy`, `datasets/episode_store`, `integrations/sionna/adapter`, `integrations/sionna/network_phy`, the SRS constants package, and the lineage audit package. None of these is named in a retained result or config.
   - Reverted, deferred: `experiments/ordered_impairments_receiver.py`. The split (`846af03`) was made before its design note and rebinding entries, which the ordering rule forbids. The pre-split module and test are restored, and the split can be reapplied after approval and rebinding. See `docs/architecture/module-split-design-notes.md`.
   - Deferred, digest-bound, with design notes and old and new digests: `evaluation/sample_size_planning.py`, `evaluation/acquisition_package.py`, `evaluation/joint_evidence_run_manifest.py` (all in the frozen evaluator roster; splitting would add files outside a roster fixed at 41 entries by the V4 schema), and `receivers/nr/pusch.py` (named by 71 references and hashed by about 18 tools; a split cannot be proven equivalent by rebinding).
   - Note: the goal listed `evaluation/acquisition_package.py` and `receivers/nr/pusch.py` as digest-free. They are not; the design note records why.
5. **Documentation.** Non-research documents are under `guide/`, `architecture/`, `operations/` and `reference/`; the research folder keeps its names. `docs/README.md` is a generated index that reaches every document in `docs/`. `scripts/fix_doc_links.py` repairs broken links in editable documents (the master plan and the repository's own pages) by following the recorded moves, and turns unresolvable targets into plain text marked "not in this tree". Frozen research text is back at base bytes (see below). **Open:** 46 broken links remain inside frozen research text, which cannot change without approval; they are listed by file in the check output (`python scripts/fix_doc_links.py --check`). The frozen-link breakdown on `45e3e52`: `docs/research/MVP_STATUS.md` 21, `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md` 0 editable (its links are repaired or marked), `docs/architecture/output-tasks-and-channels.md` 6 and `docs/guide/quickstart.md` 5 and `docs/guide/luna-execution-guide-2026-11-11.md` 4 (frozen text moved from research), and about 10 more across `docs/research/`.
6. **Topic documents.** `docs/research/PLAN_ARCHITECTURE.md`, `PLAN_DATA_GENERATION.md`, `PLAN_RECEIVERS_AND_TARGETS.md`, `PLAN_QUALIFICATION_GATES.md` and `PLAN_OPEN_QUESTIONS.md` are generated by `scripts/build_research_topics.py` from the archived plan at tag `docs/master-research-plan-archive`. Each copied line occurs verbatim in the archive. The tag was not edited (blob `f30d8f567f97443aec722eabc459843aa4e5821f7574ce67080ab99747c5de93`, equal to the base plan), and the plan is not in the tree. The earlier paraphrased notes were replaced.
7. **Package list.** `config/packages.toml` defines the package list once. `tools/sync_packages.py` derives the setuptools, ruff, mypy and coverage settings. `tools/check` is the single CI source.
8. **Structure tests.** `tests/architecture/` fails on: `.gitkeep` under `data/`; phase or edition tokens in tracked paths; runtime import cycles between packages; and package directories without a `README.md`. The README test was checked by a real violation during this work (a new package without a README failed the test).
9. **This record.** Gate, torch and native results above.

## Frozen text

Frozen research documents keep their base bytes. Rename passes had changed the
text of sixteen research files and of six contract schemas. Those are restored
to base bytes at their current paths, and only their paths moved. The master
plan is the one authorised exception; it is edited for links and status and its
digest is recorded (below). The requirement ledger is restored to base bytes.

Frozen identity strings that code wrote from current file names are now written
as the frozen identities (for example the CDL-E protocol and seed-audit names,
and the directional-pattern reference names). Identity strings stay.

## Changed digests

Every changed path, with old and new SHA-256, is in
`docs/reference/refactor-digest-changes.tsv` (973 rows, from `016168e` to `f6e4cc3`).
Key changes:

- Master plan (`docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md`): base `3285eb1398ee83ea4d79b4b0116478ceed9d1374537fe2d3510a3c65a7585da2`; now `e1e6fb8435da71e5df073f9b6888d02f4fc7dea07aac3a5756737803a2075241`. The scene-family registry still pins the base value; the validator accepts the current bytes through appended `source_digest_rebindings` entries. The pin is not rewritten.
- Receiver facade (`experiments/g5_ordered_impairments_receiver.py` at base → `experiments/ordered_impairments_receiver.py`): base `d6c5b9791e30f10e9efbc07a9e0409fe89ef618c7eabe06ffd49151e0640309a`; now `2e9f96621bc3be1cb1c97b2ce71873afa1adf001cbad56b1fb6530ff45190a34`.
- `datasets/scene_families.py`: base `5309fd6110e6d68d248567a10f55553e5b5325c59e6b4ae02ed02aeb553324e3`; now `203d741a8147e4a24475c4c08eedf3e49052de589c31ac98cd3ca62c72309606`.
- `docs/research/HELPER_WIRING_INVENTORY.json`: base `32b22aa4719b9155ba76b59b93d6ac92cafbb02bfe4f70b6cb764899749c5b7e`; now `dde3664893eb1184b952a628bdccfc93beb5dcfd85697437cc27e5427bbeea32` (regenerated, 254 helpers).
- Digest-bound, not split: `evaluation/g3_sample_size_planning.py` base `990a8be9cee63209c9da2aa4c1a62fcb1a47d9b4e169e293636b07988bbaf7a0`, now `886f9211e51a10144606e27f8b3fb2fc53a991ee858b7e41e2873fb631bc70ad`; `evaluation/g3_acquisition_package.py` base `07bb9da6c12b86503218f8c087dfd430e4f2363987c4a9c56d1c86fcca056db3`, now `0a8db8921fe180f2bdaaec0adb20de957944f79ccddd5daf6440dacf2c7159ae`; `evaluation/g3_joint_evidence_run_manifest.py` base `c44ab0d3ee9e5ad196ca4dfd715f5b120c8c563ec30133c2bf732d257b18f26b`, now `92b065d5978cd4f783ac05de4576965c15540cc55165900f38bc8664617f9d63`; `receivers/nr/pusch.py` unchanged at `b6a632a92fbc92a294fe7baf718a69b665635e4e5d0e71f4528b5b4d1333f003`.
- Restored to base bytes (digest equals base): the sixteen frozen research files and six contract schemas listed in the first commit of the restoration series.

Rebinding entries are appended only. Counts in `docs/reference/digest-rebindings.json`: `file_moves` 651, `pin_rebindings` 116, `source_digest_rebindings` 241. No existing entry was edited.

## Commits since `016168e` (final round)

`git log --oneline 016168e..f6e4cc3`. The final-round commits are, oldest first:
`07d0c76`, `483b27f`, `5d06e9b`, `1a3b8ea`, `a41a635`, `d06895c`, `34f9cc6`, `1ab90b8`, `295ca28`, `578b17a`, `fc5589c`, `4767f11`, `1b0c460`, `63847a2`, `22b66c4`, `a161422`, `45e3e52`, `8f26703`, `aaf8ca5`, `f6e4cc3`.
