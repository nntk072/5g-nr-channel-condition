# Production-structure refactor: verification record

This record covers the production-structure refactor from the evidence base
`016168e` to the tree at `7681f9a`. It states what was done, what is deferred and
why, every gate result as logged, the torch and native results, and the changed
digests. Scientific status is unchanged: E04 remains engineering DONE and
scientific NOT_QUALIFIED.

Path note: the goal named `docs/operations/REFACTOR_VERIFICATION.md` and
`docs/architecture/NAMING_EXCEPTIONS.md`. Both use those exact names.

## Gate results on `7681f9a`

Commands are `./tools/check <gate>`. Lines are copied from the gate logs.

| Gate | Result |
| --- | --- |
| `lock` | `uv lock --check`; `Resolved 157 packages`; exit 0 |
| `lint` | `All checks passed!`; `838 files already formatted`; exit 0 |
| `typecheck` | `Success: no issues found in 444 source files`; exit 0 |
| `architecture` | `tests/architecture`: 9 passed; exit 0 |
| `agent` | `180 passed in 8.31s`; exit 0 |
| `docs` | Sphinx build with warnings as errors; exit 0 |
| `build` | `Successfully built dist/5g_nr_channel_condition-0.2.0-py3-none-any.whl`; `import-smoke-ok`; exit 0 |
| `test` | `3921 passed, 172 skipped, 1 warning in 270.29s (0:04:30)`; exit 0 |
| `neural` | `Required test coverage of 75% reached. Total coverage: 75.88%`; `3921 passed, 172 skipped, 1 warning in 545.50s (0:09:05)`; exit 0 |

Skipped tests are reported as skipped, not as passes. The 172 skips in `test` and
`neural` are optional tests whose dependency or device is absent in CI (Sionna
native tests, CUDA, xgboost). The count changed from 170 (on `f6e4cc3`) because
the single PUSCH native test file was split into three modules, and each skips
once when Sionna is absent.

## Torch tests, `envs/py312-gpu`, `CUDA_VISIBLE_DEVICES=-1`

Command: `DRJIT_LIBLLVM_PATH=<LLVM 18 library> /home/l2nguyen/agent-final-2026-11-22/envs/py312-gpu/bin/python -m pytest --no-cov -o addopts= -q -rs -m "not native_sionna"`.
The LLVM path is required: without it the `conftest.py` import of Mitsuba fails at
configuration.

- Result on `7681f9a`: `3996 passed, 4 skipped, 1272 deselected, 1 warning in 306.24s (0:05:06)`; exit 0.
- Skipped: `test_network_receiver.py:81` (CUDA unavailable); `test_observable_multipath.py:186` (CUDA unavailable); two in `test_classical.py` (xgboost not installed).
- The 1272 deselected tests are the native tests, which run in the native section.

## Native parity, CUDA hidden, against the base

Command: `DRJIT_LIBLLVM_PATH=<LLVM 18 library> python -m pytest -m native_sionna --no-cov -o addopts= -q -rfEs` in the same environment, `CUDA_VISIBLE_DEVICES=-1`.

| Tree | Passed | Failed | Skipped | Deselected | Duration |
| --- | --- | --- | --- | --- | --- |
| Current, `7681f9a` | 1223 | 6 | 43 | 4000 | 1541.48s (0:25:41) |
| Base `016168e` | 1215 | 14 | 43 | 3997 | earlier base run |

The base row is the count from the earlier base run. The saved base log in
`/home/l2nguyen/agent-final-2026-11-22/refactor-baseline/native-parity-016168e-baseline.log`
has no summary line, so its counts were not re-checked here. The current row
differs from the earlier current row (4003 deselected) only because three
non-native test cases were removed with the crosswalk test.

The six failures on the current tree, with base status:

1. `test_admission_native.py` `[A30]`, `[B100]`, `[C60]`: present at base. The admission code and its expected message disagree. This is a scientific admission question; the expectation is not changed.
2. `test_cdl_joint_tx_rx_spatiotemporal_covariance_native.py` `[B]`: present at base. A tolerance miss against the independent reference. Changing the tolerance would change a scientific acceptance criterion.
3. `test_cdl_spatial_covariance_4element_native.py` `[E]`: present at base. The same kind of tolerance miss.
4. `test_rt_trajectory_native.py::test_cpu_rt_path_retraces_and_stores_receiver_only_trajectory`: present at base. Re-run on a detached worktree of `016168e`: `1 failed`. The test asserts an "independent Mitsuba" oracle; the code describes a digest-bound NumPy inventory cross-checked by Mitsuba. That is a claim about oracle independence, so it is not changed here.

Parity criterion: no native failure was added. The failing set matches the earlier run
exactly. Native parity is "no regression", not "all native tests pass". The six open
failures need an owner decision; each is a scientific acceptance or claim question.

### Diagnosis of the six native failures (no expectation changed)

Diagnosis only, as the owner asked. No test expectation, tolerance or claim was changed.

1. `test_admission_native.py` `[C60]`, `[B100]`, `[A30]`: the admission decision is correct in all three cases (each carrier is rejected). The failure is the message: the test expects `not applicable in FR1`, `not applicable in FR2` or `require an FR1 or FR2 carrier`, while `datasets/standardized.py:212` reports `carrier_frequency_hz must be within <ranges>`. The expected wording is not produced by the code. Decision needed: adopt the code's wording in the test, or add the FR-specific wording to the code.
2. `test_cdl_joint_tx_rx_spatiotemporal_covariance_native.py` `[B]`: the component check `abs(estimate - expected) <= half_width` (line 213) fails for one or more of the 16 components of the family-wise interval (alpha 0.01). The log does not print the half-width, so the size of the miss relative to the interval is not recorded here. Decision needed: whether the interval or the reference is wrong.
3. `test_cdl_spatial_covariance_4element_native.py` `[E]`: the same check (line 213). The logged absolute errors reach about 0.015 against an interval capped at `_MAX_INTERVAL_HALF_WIDTH = 0.05`. The per-component half-width is not printed in the log. Decision needed, as for case 2.
4. `test_rt_trajectory_native.py::test_cpu_rt_path_retraces_and_stores_receiver_only_trajectory` (line 82): the test requires the oracle label to start with `independent Mitsuba`. The code labels it `digest-bound NumPy loaded-scene triangle inventory, cross-checked by Mitsuba segment intersection and depth-zero Sionna PathSolver`. The code describes a NumPy oracle cross-checked by Mitsuba, not an independent Mitsuba oracle. Decision needed: whether the claim or the label is correct.

The two CDL cases and the RT case are scientific acceptance or claim questions. The admission cases are a wording decision.

## Goal items

1. **Helper wiring.** `scripts/audit_helper_wiring.py` resolves package re-exports and facade re-exports to the defining submodule. The helper count is `254`; `--check` exits 0; `tests/test_scripts/test_audit_helper_wiring.py` passes. Dispositions: 245 manual review, 1 no direct reference, 8 route mapping required. The mix is the same as before the splits; the only identity change is the moved `compose_pusch_intra_slot_hops`, now `receivers.nr.pusch_core.hopping.compose_pusch_intra_slot_hops`.
2. **Naming.** `NAMING_EXCEPTIONS.md` lists every kept name with its reason: the three `g3` functions, `target_joint_g3_assurance` (schema constant), and the `_vN` schema-version names that are outside the `g3`/`g4`/`g5`/`_v1` rule. Renamed: `migrate_legacy_target_definition_v1` (`1ab90b8`) and three `_v1` test names (`9e2fd90`).
3. **Merges.** Done: CDL covariance package `evaluation/cdl_covariance/` (`4767f11`); SNR-curve audit module (`adc5422`); TDL audit modules merged into `experiments/audit_tdl/` with a rebinding for the retained self-digest (`7c541cd`). The evaluation `g3`/`g4` modules were already concept-named.
4. **Splits.** No tracked module is over 1,500 lines (`git ls-files '*.py'` check). Each split is by dependency layer with a facade at the original path that re-exports every original name. Rebinding entries and design notes were committed before each code commit.
   - Digest-free: `datasets/split_policy/`, `datasets/episode_store/`, `integrations/sionna/adapter/`, `integrations/sionna/network_phy/`, the SRS constants package, the lineage audit package (earlier commits).
   - Digest-bound, narrowed binding (owner decision, option 2): `experiments/ordered_impairments_receiver.py` (`fccf1dc`); `evaluation/sample_size_planning.py` (`evaluation/sample_size/`); `evaluation/joint_evidence_run_manifest.py` (`evaluation/run_manifest/`); `evaluation/acquisition_package.py` (`evaluation/acquisition/`); `receivers/nr/pusch.py` (`receivers/nr/pusch_core/`). The facades keep the paths the rosters and tools name. The new submodules are outside the signed evaluator rosters and outside the digests the audit tools name. The verifier's source check accepts a rebound digest (`artifacts/rebinding.py: recorded_digest_resolves`).
   - `NRPUSCHConfig` and `NRChannelEstimate` stay defined in the PUSCH facade: their module path is part of the audited model-input identity.
   - Test modules over 1,500 lines were split by concept (SRS, UMCD, lineage, joint evidence, runner, native PUSCH); the total test-case counts match the original files.
   - Roster coverage is narrowed: signed roster evidence does not hash the submodules. A roster-protocol change would be needed for full coverage; that was not chosen.
5. **Documentation.** Non-research documents are under `guide/`, `architecture/`, `operations/` and `reference/`; `docs/README.md` is the generated index. `scripts/fix_doc_links.py --check` reports `0 link problem(s); 0 in editable documents`. The owner approved a frozen-text edit for link targets: 10 frozen documents changed in link text only, each with a rebinding entry from its base digest. Known stale references remain in frozen text: `docs/research/EXECUTION_QUEUE.md`, `docs/research/IMPLEMENTATION_BACKLOG.md` and `docs/research/REQUIREMENT_LEDGER.json` name the deleted crosswalk builder and test (item 9 below).
6. **Topic documents.** `docs/research/PLAN_ARCHITECTURE.md`, `PLAN_DATA_GENERATION.md`, `PLAN_RECEIVERS_AND_TARGETS.md`, `PLAN_QUALIFICATION_GATES.md` and `PLAN_OPEN_QUESTIONS.md` are generated by `scripts/build_research_topics.py` from the archived plan at tag `docs/master-research-plan-archive`, with each copied line verbatim in the archive. The tag was not edited; the plan is not in the tree.
7. **Package list.** `config/packages.toml` defines the package list once. `tools/sync_packages.py` derives the setuptools, ruff, mypy and coverage settings. `tools/check` is the single CI source.
8. **Structure tests.** `tests/architecture/` (9 tests, passing) fails on `.gitkeep` under `data/`, phase or edition tokens in tracked paths, runtime import cycles between packages, and package directories without `README.md`. The four new split packages have READMEs.
9. **This record.** The gate, torch and native results above.

### Deviation: deleted crosswalk builder and test

`scripts/build_historical_status_crosswalk.py` and `tests/test_scripts/test_build_historical_status_crosswalk.py` were deleted (`7681f9a`) at the owner's decision. The builder read `docs/research/MASTER_RESEARCH_PLAN.md`, which left the tree with the topic-document split, so `--check` could no longer run. The generated record `docs/research/HISTORICAL_STATUS_CROSSWALK.json` is frozen and stays. `docs/research/REQUIREMENT_LEDGER.json` cites the builder and its test as evidence for `PKG00-002`. The ledger is a protected authority and its text is frozen, so that citation now names removed files. This is recorded as a deviation; the requirement's cited evidence is no longer reproducible from this tree.

## Known issues not caused by this refactor

- Several retained results pin `receivers/nr/pusch.py` at digests that do not chain to the current file (`d756599070c350e65d96b823738c98ad7be59d9188eb7eba98683d748716594d` and `f00a94da21f925c7366a36072fb63dd2dd68669bd2414064e797a97d8fd8464d`). The rebinding entries do not connect them.
- The six native scientific failures above.

## Frozen text

Frozen research documents keep their base bytes except the ten documents whose link
targets were repaired (owner approval). The master plan is the one authorised
exception, edited for links and status, with its digest recorded. The requirement
ledger and execution queue are at base bytes except that the execution queue's
link text was repaired; the ledger is unchanged.

Frozen identity strings written by code (`umcd.*`, `loslab.*`, schema `$id` values,
protocol names, the CDL-E seed-audit names) keep their values.

## Changed digests

Every changed path, with its old and new SHA-256 (base `016168e` to `7681f9a`), is in
`docs/reference/refactor-digest-changes.tsv`: 874 rows (541 renames, 176 additions,
86 deletions, 71 modifications). Rebinding entries are appended only. Counts in
`docs/reference/digest-rebindings.json`: `file_moves` 658, `pin_rebindings` 116,
`source_digest_rebindings` 262.

Key changes (old → new):

- Master plan `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md`: `3285eb1398ee83ea4d79b4b0116478ceed9d1374537fe2d3510a3c65a7585da2` → `e1e6fb8435da71e5df073f9b6888d02f4fc7dea07aac3a5756737803a2075241`.
- Receiver facade `experiments/ordered_impairments_receiver.py` (base `experiments/g5_ordered_impairments_receiver.py`): `d6c5b9791e30f10e9efbc07a9e0409fe89ef618c7eabe06ffd49151e0640309a` → `2e9f96621bc3be1cb1c97b2ce71873afa1adf001cbad56b1fb6530ff45190a34`.
- Sample-planning facade `evaluation/sample_size_planning.py` (base `evaluation/g3_sample_size_planning.py`): `990a8be9cee63209c9da2aa4c1a62fcb1a47d9b4e169e293636b07988bbaf7a0` (base) and `886f9211e51a10144606e27f8b3fb2fc53a991ee858b7e41e2873fb631bc70ad` (pre-split) → `fcff4edffa41ab4135a257f2321c7c844c782ef7ead53c2953b06477d26d2dc3`.
- Run-manifest facade `evaluation/joint_evidence_run_manifest.py` (base `evaluation/g3_joint_evidence_run_manifest.py`): `c44ab0d3ee9e5ad196ca4dfd715f5b120c8c563ec30133c2bf732d257b18f26b` (base) and `92b065d5978cd4f783ac05de4576965c15540cc55165900f38bc8664617f9d63` (pre-split) → `fcbd0a189bc78a6d422f15ff344b232952302838c59c073c827c0812d92a3ce3`.
- Acquisition facade `evaluation/acquisition_package.py` (base `evaluation/g3_acquisition_package.py`): `07bb9da6c12b86503218f8c087dfd430e4f2363987c4a9c56d1c86fcca056db3` (base) and `0a8db8921fe180f2bdaaec0adb20de957944f79ccddd5daf6440dacf2c7159ae` (pre-split) → `05e11caeb31a8bb17d3333ab6152592b8ef4881336d8283c4c9e92e58d76e185`.
- PUSCH facade `receivers/nr/pusch.py`: `b6a632a92fbc92a294fe7baf718a69b665635e4e5d0e71f4528b5b4d1333f003` (base and pre-split) → `f13ed883a2d546bb5e90d0ac7cb9d314c88c2b05e9f5719fd7c2f5c40af42895`.
- TDL development audit (base `experiments/audit_g5_tdl_dmrs_density_development.py`): `156d5617393f208644bd9fc79d7e5a3076e1a90a568d05a1d9db0f7d4069b7b7` → `2a2b7454d0796db1ef9d2eef1cdbfd57dc05219181ee1ba5be9208449c43da90` (now `experiments/audit_tdl/dmrs_density_development.py`).
- Helper inventory `docs/research/HELPER_WIRING_INVENTORY.json`: base `32b22aa4719b9155ba76b59b93d6ac92cafbb02bfe4f70b6cb764899749c5b7e`; regenerated, 254 helpers; current digest in the table.
- Restored to base bytes (digest equals base): the sixteen frozen research files and six contract schemas (`295ca28`, `a161422`).

## Commits since `016168e`

`git log --oneline 016168e..7681f9a` lists the refactor commits, newest first, from `7681f9a` (crosswalk deletion) through `db791f8` (audit types and READMEs), `cea9a87` (PUSCH split), `034e1b9` (PUSCH rebinding), `761960b` (acquisition split), `e9d486a` (acquisition rebinding), `7d4369f` (sample and run-manifest split), `9e29b07` (their rebindings), `f5b0c94` (frozen link repair), `2cf656f` (digest-bound decision note), `7c541cd` (TDL merge), `9e2fd90` (naming pass).
