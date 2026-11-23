# Production-structure refactor: verification record

This record covers the production-structure refactor from the evidence base
`016168e` to the tree at `31ea11c`, the last code commit before this record.
It states what was done, what is deferred and why, every gate result exactly as
logged, the torch and native results, and the changed digests. Scientific status
is unchanged: E04 remains engineering DONE and scientific NOT_QUALIFIED.

Path note: the goal named this file `docs/operations/REFACTOR_VERIFICATION.md`
and the naming record `docs/architecture/NAMING_EXCEPTIONS.md`. Both now use
those exact names.

## Gate results on `31ea11c`

Commands are `./tools/check <gate>`. Results are copied from the gate logs.

| Gate | Result |
| --- | --- |
| `lock` | exit 0 |
| `lint` | `All checks passed!`; `804 files already formatted`; exit 0 |
| `typecheck` | `Success: no issues found in 414 source files`; exit 0 |
| `architecture` | exit 0 |
| `agent` | `180 passed in 8.45s`; exit 0 |
| `docs` | Sphinx build with warnings as errors; exit 0 |
| `build` | `Successfully built dist/5g_nr_channel_condition-0.2.0-py3-none-any.whl`; `import-smoke-ok`; exit 0 |
| `test` | `3924 passed, 172 skipped, 1 warning in 253.46s (0:04:13)`; exit 0 |
| `neural` | `Required test coverage of 75% reached. Total coverage: 75.79%`; `3924 passed, 172 skipped, 1 warning in 536.35s (0:08:56)`; exit 0 |

Skipped tests are reported as skipped, not as passes. The skip count is 172 in
`test` and `neural`. The earlier record showed 170 on `f6e4cc3`. The difference
is two: the single PUSCH native test file skipped once when Sionna is absent, and
its three split modules each skip once in that case. The skips are the optional
tests whose dependency or device is absent in CI (Sionna native tests, CUDA,
xgboost).

## Torch tests, `envs/py312-gpu`, `CUDA_VISIBLE_DEVICES=-1`

Command: `DRJIT_LIBLLVM_PATH=<LLVM 18 library> python -m pytest --no-cov -o addopts= -q -rs -m "not native_sionna"`
with `python` = `/home/l2nguyen/agent-final-2026-11-22/envs/py312-gpu/bin/python`.
The LLVM path is required: without it, the `conftest.py` import of Mitsuba fails
at configuration (an earlier attempt in this session exited 3 for that reason).

- Result on `31ea11c`: `3999 passed, 4 skipped, 1272 deselected, 1 warning in 267.40s (0:04:27)`; exit 0.
- Skipped: `test_network_receiver.py:81` (CUDA unavailable), `test_observable_multipath.py:186` (CUDA unavailable), and two in `test_classical.py` (xgboost not installed in this environment).
- The 1272 deselected tests are the native tests, which run in the native section.

An earlier torch run on this tree failed one test: `test_independent_snr_curve_retains_each_condition_input_across_snr`. The audit's SNR-curve function had moved to `experiments/audit_ordered_impairments_result/snr_curve.py`, and the test still patched the old module. It was fixed in `31ea11c`; the run above is on that fixed tree.

## Native parity, CUDA hidden, against the base

Command: `DRJIT_LIBLLVM_PATH=<LLVM 18 library> python -m pytest -m native_sionna --no-cov -o addopts= -q -rfEs`
in the same environment with `CUDA_VISIBLE_DEVICES=-1`.

| Tree | Passed | Failed | Skipped | Deselected | Duration |
| --- | --- | --- | --- | --- | --- |
| Current, run on `76977a8` | 1223 | 6 | 43 | 4003 | 1475.68s |
| Base `016168e` | 1215 | 14 | 43 | 3997 | from the earlier base run |

The base row is the count recorded by the earlier base run. The saved base log in
`/home/l2nguyen/agent-final-2026-11-22/refactor-baseline/native-parity-016168e-baseline.log`
has no summary line, so its counts could not be re-checked here. The base run
itself is not repeated.

The current run was on `76977a8`. The later commit `31ea11c` changed only
`tests/test_experiments/test_ordered_impairments_receiver.py`, a non-native test
line. The two native-marked tests in that file were run separately on the final
tree (see the native receiver check below).

Native failures on the current tree, with their base status:

1. `test_admission_native.py` `[A30]`, `[B100]`, `[C60]`: present at base. The admission code and its expected message disagree. This is a scientific admission behaviour question. The expectation is not changed.
2. `test_cdl_joint_tx_rx_spatiotemporal_covariance_native.py` `[B]`: present at base. A tolerance miss against the independent reference. Changing the tolerance would change a scientific acceptance criterion.
3. `test_cdl_spatial_covariance_4element_native.py` `[E]`: present at base. The same kind of tolerance miss.
4. `test_rt_trajectory_native.py::test_cpu_rt_path_retraces_and_stores_receiver_only_trajectory`: present at base. Re-run on a detached worktree of `016168e` in this session: `1 failed`. The test asserts that the geometric-LOS oracle is an "independent Mitsuba" oracle, while the code describes a digest-bound NumPy inventory cross-checked by Mitsuba. That is a claim about oracle independence, so it is not changed here.

Base failures that no longer occur on the current tree: the dual-polarization cases, the CDL-D case, the two K-factor cases, the network PHY cases, and the statistical-confidence cases. Some were fixed by test expectation and digest-pin corrections in this branch; the statistical-confidence case that read the wrong axis (raising `IndexError`) was fixed in `f6e4cc3`.

Parity criterion: no native failure was added. The failing set shrank from 14 to 6 against the earlier base run. Native parity is "no regression", not "all native tests pass". The six open failures need an owner decision, because each is a scientific acceptance or claim question.

Native receiver check on the final tree (`31ea11c`): `python -m pytest tests/test_experiments/test_ordered_impairments_receiver.py -m native_sionna` gives `2 passed, 41 deselected in 13.98s`; exit 0.

## Goal items

1. **Helper wiring.** `scripts/audit_helper_wiring.py` resolves package re-exports to their defining submodule. The helper count is `254`; `--check` exits 0; `tests/test_scripts/test_audit_helper_wiring.py` passes. The disposition mix is 245 manual review, 1 no direct reference, 8 route mapping required. The inventory was regenerated in `a2842d2`.
2. **Naming.** The `_v1` production name `migrate_legacy_target_definition_v1` was renamed (`1ab90b8`). The three `g3` production names are kept because frozen documents and schemas reference them: `verify_g3_joint_evidence_run_manifest`, `plan_g3_independent_acquisition_units`, `compose_g3_joint_sample_size_design`. `target_joint_g3_assurance` is a schema constant and is kept. The `_vN` schema-version names are kept. Each is listed with its reason in `docs/architecture/NAMING_EXCEPTIONS.md`.
3. **Merges.**
   - Done: the nine CDL covariance modules are one package, `evaluation/cdl_covariance/` (`4767f11`). Identical helper bodies are in `shared.py`.
   - Done: the SNR-curve audit is its own module in `experiments/audit_ordered_impairments_result/` (`adc5422`), and the ordered-impairment receiver is one concept package (`fccf1dc`).
   - Deferred: the `experiments/audit_tdl_*` audit modules (five files). Their retained result `docs/research/results/tdl-cpu-dmrs-density-development-20261024.json` records the digest of its own module (`experiments/audit_g5_tdl_dmrs_density_development.py`, `156d5617…`) as a source digest. Merging changes that module's bytes, so the retained provenance would depend on a rebinding instead of the original code. They are not merged. The frozen master plan also cites `experiments/audit_tdl_c_dmrs_density.py` as its reproduce command.
   - The `evaluation/g3_*` and `g4_*` modules already have concept names; the CDL family is merged.
4. **Splits.**
   - Done, digest-free: `datasets/split_policy/`, `datasets/episode_store/`, `integrations/sionna/adapter/`, `integrations/sionna/network_phy/`, the SRS constants package, and the channel-split lineage audit package.
   - Done, digest-bound, after rebinding: `experiments/ordered_impairments_receiver.py` (rebinding `1b3a3b2`, split `fccf1dc`). Old digest of the base file `d6c5b979…`; the split facade is `2e9f9662…`. Design note: `docs/architecture/module-split-design-notes.md`.
   - Done, tests: the SRS resource-mapping test file (five concept modules plus a support module), the UMCD schema test file, the channel-split lineage test file, and the joint-evidence, runner and native PUSCH test files. Every test case count matches the original file (for example 291, 137, 50, 48, 38 and 31).
   - Deferred, with reasons:
     - `evaluation/sample_size_planning.py` (2,231 lines), `evaluation/joint_evidence_run_manifest.py` (1,697 lines) and `evaluation/acquisition_package.py` (1,666 lines) are in the frozen V1/V2/V4 evaluator rosters. The V4 schema fixes the roster at 41 entries (`minItems` = `maxItems` = 41). A split would add code outside the signed roster, which weakens the binding of retained evidence. The frozen contract may not change, so these stay until the owner approves a roster-protocol change. Design notes with old and new digests are in `docs/architecture/module-split-design-notes.md`.
     - `receivers/nr/pusch.py` (1,542 lines) is named by 71 path references and hashed as one unit by about 18 tools and by the frozen-manifest test helper. A split changes the bytes those checks verify. The retained digest checks would then depend on a rebinding instead of the original file, so the split is not made. The goal listed this file as digest-free; it is not.
     - Files over 1,500 lines that remain: the four modules above. No test module is over 1,500 lines.
5. **Documentation.** Non-research documents are under `guide/`, `architecture/`, `operations/` and `reference/`. `docs/README.md` is a generated index. `scripts/fix_doc_links.py --check` reports 46 problems, all in frozen research text, and 0 in editable documents. The frozen documents cannot change text, so the 46 remain. They are listed by file in the check output. Frozen text also names moved test paths (for example `tests/test_integrations/test_nr_pusch_native.py`, `tests/test_experiments/test_runner.py`, `experiments/audit_tdl_c_dmrs_density.py`); those references are stale for the same reason.
6. **Topic documents.** `docs/research/PLAN_ARCHITECTURE.md`, `PLAN_DATA_GENERATION.md`, `PLAN_RECEIVERS_AND_TARGETS.md`, `PLAN_QUALIFICATION_GATES.md` and `PLAN_OPEN_QUESTIONS.md` are generated by `scripts/build_research_topics.py` from the archived plan at tag `docs/master-research-plan-archive`. Each copied line occurs verbatim in the archive. The tag was not edited. The plan is not in the tree.
7. **Package list.** `config/packages.toml` defines the package list once. `tools/sync_packages.py` derives the setuptools, ruff, mypy and coverage settings. `tools/check` is the single CI source.
8. **Structure tests.** `tests/architecture/` fails on `.gitkeep` under `data/`, phase or edition tokens in tracked paths, runtime import cycles between packages, and package directories without `README.md`. The `architecture` gate passes.
9. **This record.** The gate, torch and native results above.

Repository records renamed to the required names (`4fbb057`): `docs/operations/REFACTOR_VERIFICATION.md` and `docs/architecture/NAMING_EXCEPTIONS.md`. Rebinding entries for the two renames were appended to `docs/reference/digest-rebindings.json`.

## Frozen text

Frozen research documents keep their base bytes; only their paths moved. The
master plan is the one authorised exception: it is edited for links and status,
and its digest is recorded below. The requirement ledger and execution queue are
at base bytes, so their references to moved test paths are stale (see item 5).

Frozen identity strings written by code (`umcd.*`, `loslab.*`, schema `$id`
values, protocol names, the CDL-E seed-audit names) keep their values.

## Changed digests

Every changed path, with its old and new SHA-256 (base `016168e` to `31ea11c`), is in
`docs/reference/refactor-digest-changes.tsv`: 827 rows (543 renames, 138 additions,
82 deletions, 64 modifications). The table was regenerated for this record.

Key changes:

- Master plan `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md`: base `3285eb1398ee83ea4d79b4b0116478ceed9d1374537fe2d3510a3c65a7585da2`; now `e1e6fb8435da71e5df073f9b6888d02f4fc7dea07aac3a5756737803a2075241`. The scene-family registry still pins the base value; the validator accepts the current bytes through appended `source_digest_rebindings`. The pin is not rewritten.
- Receiver facade (`experiments/g5_ordered_impairments_receiver.py` at base → `experiments/ordered_impairments_receiver.py`): base `d6c5b9791e30f10e9efbc07a9e0409fe89ef618c7eabe06ffd49151e0640309a`; now `2e9f96621bc3be1cb1c97b2ce71873afa1adf001cbad56b1fb6530ff45190a34`.
- Audit SNR-curve module `experiments/audit_ordered_impairments_result/snr_curve.py`: new, `ff86e7721c7e1b6f34967c2db66ee773509b6fff6d388cd604415d56ba7f3df5`. The `independent.py` module changed to `c35c919f26e0dd39dc3956c057715ce264d4f60399951636ca774057d189d6ec`.
- `datasets/scene_families.py`: base `5309fd6110e6d68d248567a10f55553e5b5325c59e6b4ae02ed02aeb553324e3`; now `203d741a8147e4a24475c4c08eedf3e49052de589c31ac98cd3ca62c72309606`.
- `docs/research/HELPER_WIRING_INVENTORY.json`: base `32b22aa4719b9155ba76b59b93d6ac92cafbb02bfe4f70b6cb764899749c5b7e`; regenerated in `a2842d2`, 254 helpers. Its current digest is in the TSV.
- Digest-bound, not split: `evaluation/g3_sample_size_planning.py` base `990a8be9cee63209c9da2aa4c1a62fcb1a47d9b4e169e293636b07988bbaf7a0`, now `886f9211e51a10144606e27f8b3fb2fc53a991ee858b7e41e2873fb631bc70ad`; `evaluation/g3_acquisition_package.py` base `07bb9da6c12b86503218f8c087dfd430e4f2363987c4a9c56d1c86fcca056db3`, now `0a8db8921fe180f2bdaaec0adb20de957944f79ccddd5daf6440dacf2c7159ae`; `evaluation/g3_joint_evidence_run_manifest.py` base `c44ab0d3ee9e5ad196ca4dfd715f5b120c8c563ec30133c2bf732d257b18f26b`, now `92b065d5978cd4f783ac05de4576965c15540cc55165900f38bc8664617f9d63`; `receivers/nr/pusch.py` unchanged at `b6a632a92fbc92a294fe7baf718a69b665635e4e5d0e71f4528b5b4d1333f003`.
- Restored to base bytes (digest equals base): the sixteen frozen research files and six contract schemas listed in the restoration commits (`295ca28`, `a161422`).

Rebinding entries are appended only. Counts in `docs/reference/digest-rebindings.json`: `file_moves` 653, `pin_rebindings` 116, `source_digest_rebindings` 243. No existing entry was edited.

## Commits since `016168e`

`git log --oneline 016168e..31ea11c` lists 61 commits. The latest ones are `31ea11c`, `76977a8`, `a2842d2`, `98689d6`, `30a060b`, `134274a`, `07197a2`, `adc5422`, `fccf1dc`, `1b3a3b2` and `4fbb057`.
