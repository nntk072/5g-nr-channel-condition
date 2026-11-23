# Production-structure refactor: verification record

This record covers the production-structure refactor from the evidence base
`016168e` to the tree at `9959d81`. It states what was done, what is deferred and
why, every gate result as logged, the torch and native results, and the changed
digests. Scientific status is unchanged: E04 remains engineering DONE and
scientific NOT_QUALIFIED.

Path note: the goal named `docs/operations/REFACTOR_VERIFICATION.md` and
`docs/architecture/NAMING_EXCEPTIONS.md`. Both use those exact names.

## Gate results on `9959d81`

Commands are `./tools/check <gate>`. Lines are copied from the gate logs.

| Gate | Result |
| --- | --- |
| `lock` | `uv lock --check`; `Resolved 157 packages`; exit 0 |
| `lint` | `All checks passed!`; `839 files already formatted`; exit 0 |
| `typecheck` | `Success: no issues found in 444 source files`; exit 0 |
| `architecture` | `tests/architecture`: 9 passed; exit 0 |
| `agent` | `180 passed in 8.06s`; exit 0 |
| `docs` | Sphinx build with warnings as errors; exit 0 |
| `build` | `Successfully built dist/5g_nr_channel_condition-0.2.0.tar.gz` and `...-py3-none-any.whl`; `import-smoke-ok`; exit 0 |
| `test` | `3923 passed, 172 skipped, 1 warning in 270.65s (0:04:30)`; exit 0 |
| `neural` | `Required test coverage of 75% reached. Total coverage: 75.89%`; `3923 passed, 172 skipped, 1 warning in 556.44s (0:09:16)`; exit 0 |

Skipped tests are reported as skipped, not as passes. The 172 skips in `test` and
`neural` are optional tests whose dependency or device is absent in CI (Sionna
native tests, CUDA, xgboost).

## Torch tests, `envs/py312-gpu`, `CUDA_VISIBLE_DEVICES=-1`

Command: `DRJIT_LIBLLVM_PATH=<LLVM 18 library> /home/l2nguyen/agent-final-2026-11-22/envs/py312-gpu/bin/python -m pytest --no-cov -o addopts= -q -rs -m "not native_sionna"`.
The LLVM path is required: without it the `conftest.py` import of Mitsuba fails at
configuration.

- Result on `9959d81`: `3998 passed, 4 skipped, 1272 deselected, 1 warning in 289.29s (0:04:49)`; exit 0.
- Skipped: `test_network_receiver.py:81` (CUDA unavailable); `test_observable_multipath.py:186` (CUDA unavailable); two in `test_classical.py` (xgboost not installed).
- The 1272 deselected tests are the native tests, run in the native section.

## Native parity, CUDA hidden, against the base

Command: `DRJIT_LIBLLVM_PATH=<LLVM 18 library> python -m pytest -m native_sionna --no-cov -o addopts= -q -rfEs` in the same environment, `CUDA_VISIBLE_DEVICES=-1`.

| Tree | Passed | Failed | Skipped | Deselected | Duration |
| --- | --- | --- | --- | --- | --- |
| Current, `9959d81` | 1226 | 3 | 43 | 4002 | 1484.96s (0:24:44) |
| Base `016168e` | 1215 | 14 | 43 | 3997 | earlier base run |

The base row is the count from the earlier base run. The saved base log in
`/home/l2nguyen/agent-final-2026-11-22/refactor-baseline/native-parity-016168e-baseline.log`
has no summary line, so its counts were not re-checked here.

The three failures on the current tree, all also failing at base:

1. `test_cdl_joint_tx_rx_spatiotemporal_covariance_native.py` `[B]`: a tolerance miss against the independent reference (the family-wise interval check on line 213).
2. `test_cdl_spatial_covariance_4element_native.py` `[E]`: the same kind of miss.
3. `test_rt_trajectory_native.py::test_cpu_rt_path_retraces_and_stores_receiver_only_trajectory`: the test requires the oracle label to start with `independent Mitsuba`; the code labels it `digest-bound NumPy loaded-scene triangle inventory, cross-checked by Mitsuba ...`. This is a claim about oracle independence.

Resolved since the last run: the three admission cases (`test_admission_native.py` `[C60]`, `[B100]`, `[A30]`). Owner decision: align the two message expectations with the code's message. The admission decision each case asserts is unchanged. The admission file now passes: 33 passed in the GPU environment.

Parity criterion: no native failure was added. Native parity is "no regression", not "all native tests pass". The three open failures need an owner decision; the CDL and RT cases are scientific acceptance or claim questions.

### Diagnosis of the native failures

1. Admission (resolved, wording only): the decision was correct; only the expected text differed from `datasets/standardized.py`.
2. CDL joint `[B]`: the component check (line 213) fails for one or more of the 16 family-wise interval components (alpha 0.01). The log does not print the half-width, so the size of the miss relative to the interval is not recorded here.
3. CDL 4-element `[E]`: the same check. The logged absolute errors reach about 0.015 against an interval capped at `_MAX_INTERVAL_HALF_WIDTH = 0.05`; the per-component half-width is not printed.
4. RT oracle label: the test requires `independent Mitsuba`; the code describes a NumPy oracle cross-checked by Mitsuba.

## Goal items

1. **Helper wiring.** `scripts/audit_helper_wiring.py` resolves package re-exports and facade re-exports to the defining submodule. The helper count is `254`; `--check` exits 0; `tests/test_scripts/test_audit_helper_wiring.py` passes. Dispositions: 245 manual review, 1 no direct reference, 8 route mapping required. The moved `compose_pusch_intra_slot_hops` is now `receivers.nr.pusch_core.hopping.compose_pusch_intra_slot_hops`.
2. **Naming.** `NAMING_EXCEPTIONS.md` lists every kept name with its reason: the three `g3` functions, `target_joint_g3_assurance` (schema constant), and the `_vN` schema-version names outside the `g3`/`g4`/`g5`/`_v1` rule. Renamed: `migrate_legacy_target_definition_v1` (`1ab90b8`) and three `_v1` test names (`9e2fd90`).
3. **Merges.** Done: the CDL covariance package `evaluation/cdl_covariance/` (`4767f11`); the SNR-curve audit module (`adc5422`); the TDL audit modules merged into `experiments/audit_tdl/` with a rebinding for the retained self-digest (`7c541cd`). The evaluation `g3`/`g4` modules were already concept-named.
4. **Splits.** No tracked module is over 1,500 lines. Each split is by dependency layer, with a facade at the original path that re-exports every original name. Rebinding entries and design notes were committed before each code commit.
   - Digest-free: `datasets/split_policy/`, `datasets/episode_store/`, `integrations/sionna/adapter/`, `integrations/sionna/network_phy/`, the SRS constants package, the lineage audit package (earlier commits).
   - Digest-bound, narrowed binding first (owner choice): `experiments/ordered_impairments_receiver.py` (`fccf1dc`), `evaluation/sample_size_planning.py` (`evaluation/sample_size/`), `evaluation/joint_evidence_run_manifest.py` (`evaluation/run_manifest/`), `evaluation/acquisition_package.py` (`evaluation/acquisition/`), `receivers/nr/pusch.py` (`receivers/nr/pusch_core/`).
   - Test modules over 1,500 lines were split by concept; the test-case counts match the originals.
   - Roster coverage, closed for new evidence by protocol v5 (owner-approved). `datasets/schema/joint-evidence-run-manifest.v5.schema.json` pins `evaluator_sources` at 63 entries: the 41 V4 entries plus 22 files from the split packages. The builder writes v5 manifests when a runtime inventory is included. The V4 schema and roster are unchanged, so earlier v4 manifests keep their 41-entry roster and still verify. Only v4 evidence keeps the narrowed binding. `receivers/nr/pusch_core/` is not an evaluator source and stays outside the rosters.
5. **Documentation.** Non-research documents are under `guide/`, `architecture/`, `operations/` and `reference/`; `docs/README.md` is the generated index. `scripts/fix_doc_links.py --check` reports `0 link problem(s); 0 in editable documents`. Owner-approved frozen-text edit: 10 frozen documents changed in link text only, each with a rebinding entry from its base digest.
6. **Topic documents.** `docs/research/PLAN_ARCHITECTURE.md`, `PLAN_DATA_GENERATION.md`, `PLAN_RECEIVERS_AND_TARGETS.md`, `PLAN_QUALIFICATION_GATES.md` and `PLAN_OPEN_QUESTIONS.md` are generated by `scripts/build_research_topics.py` from the archived plan at tag `docs/master-research-plan-archive`, with each copied line verbatim in the archive. The tag was not edited; the plan is not in the tree.
7. **Package list.** `config/packages.toml` defines the package list once. `tools/sync_packages.py` derives the setuptools, ruff, mypy and coverage settings. `tools/check` is the single CI source.
8. **Structure tests.** `tests/architecture/` (9 tests, passing) fails on `.gitkeep` under `data/`, phase or edition tokens in tracked paths, runtime import cycles between packages, and package directories without `README.md`. The four new split packages have READMEs.
9. **This record.** The gate, torch and native results above.

## Deviations and their status

- **Deleted crosswalk builder and test** (`7681f9a`, owner decision). `scripts/build_historical_status_crosswalk.py` and `tests/test_scripts/test_build_historical_status_crosswalk.py` were deleted because the builder read the master plan, which left the tree. The frozen record `docs/research/HISTORICAL_STATUS_CROSSWALK.json` stays.
- **Ledger citation for PKG00-002** (resolved, owner decision, `adcd5cb`). `docs/research/REQUIREMENT_LEDGER.json` no longer lists the deleted files as evidence. Its verification status is now `incomplete` with no command, because the cited test was deleted. The ledger validates against `docs/research/schema/requirement-ledger.v1.schema.json`. A source-digest rebinding records the edit: base de1c2b6d7b366cd3eb03de905915b7c2c5215cda1b54e6c618c65a7b40259b4e → 5508055d7d613ccd9b2c8d32249a944169bdd07849c612a4eb87d4c10320eb14.
- **Known remaining:** `scripts/validate_requirement_ledger.py` exits 1 because it reads `docs/research/MASTER_RESEARCH_PLAN.md`, which left the tree with the topic-document split. This is unchanged by today's edits.

## Retained PUSCH source digests (provenance recorded, not verified)

Two retained results record `receivers/nr/pusch.py` at `d756599070c350e65d96b823738c98ad7be59d9188eb7eba98683d748716594d` (commit `3034710`) and `f00a94da21f925c7366a36072fb63dd2dd68669bd2414064e797a97d8fd8464d` (commit `af7ce46`). `docs/reference/historical-source-revisions.json` records both commits; `tests/test_artifacts/test_historical_source_revisions.py` checks that each digest matches its commit's blob, and `artifacts.rebinding.historical_revision` returns the commit for a recorded digest.

A supersession rebinding was tried and reverted, because two drift-rejection tests then passed on the current file. So these results still do not verify against the current `pusch.py`. That is the correct state. Verifying them needs the historical code, which is a separate audit run and is not done here.

## Frozen text

Frozen research documents keep their base bytes, except the ten documents whose link targets were repaired (owner-approved), and the requirement ledger (owner-approved citation edit). The master plan is the one authorised exception, edited for links and status, with its digest recorded. Frozen identity strings written by code (`umcd.*`, `loslab.*`, schema `$id` values, protocol names, the CDL-E seed-audit names) keep their values.

## Changed digests

Every changed path, with its old and new SHA-256 (base `016168e` to `9959d81`), is in `docs/reference/refactor-digest-changes.tsv`: 878 rows (541 renames, 179 additions, 86 deletions, 72 modifications). Rebinding entries are appended only. Counts in `docs/reference/digest-rebindings.json`: `file_moves` 658, `pin_rebindings` 116, `source_digest_rebindings` 263.

Key changes (old → new):

- Protocol v5 (new file, not a change to an existing one): `datasets/schema/joint-evidence-run-manifest.v5.schema.json`, digest `7411d406e2c5ec9d340dc0c7b95afddb1e2e31a8593943a528cbe4e21caef16f`. Roster `V5_EVALUATOR_SOURCE_PATHS` in `evaluation/run_manifest/constants.py`, digest `55ea618f48273d61b2e67c973ce23945355e70fca3edc70a33853438357831d2`. Verifier `evaluation/run_manifest/verify.py`, digest `97e158792859536cb5ef24df321a0cfad2d87ae1c1c11f13e9989d4f4fd49de5`. Builder `evaluation/run_manifest/build.py`, digest `c3ae1ca5271a9705e860e36e9eaf48969873282f534b21d7a51a26b8c469d076`. The V4 schema is unchanged (base bytes).
- Master plan `docs/research/UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md`: `3285eb1398ee83ea4d79b4b0116478ceed9d1374537fe2d3510a3c65a7585da2` → `e1e6fb8435da71e5df073f9b6888d02f4fc7dea07aac3a5756737803a2075241`.
- Receiver facade `experiments/ordered_impairments_receiver.py` (base `experiments/g5_ordered_impairments_receiver.py`): `d6c5b9791e30f10e9efbc07a9e0409fe89ef618c7eabe06ffd49151e0640309a` → `2e9f96621bc3be1cb1c97b2ce71873afa1adf001cbad56b1fb6530ff45190a34`.
- Sample-planning facade `evaluation/sample_size_planning.py` (base `evaluation/g3_sample_size_planning.py`): `990a8be9cee63209c9da2aa4c1a62fcb1a47d9b4e169e293636b07988bbaf7a0` (base) and `886f9211e51a10144606e27f8b3fb2fc53a991ee858b7e41e2873fb631bc70ad` (pre-split) → `fcff4edffa41ab4135a257f2321c7c844c782ef7ead53c2953b06477d26d2dc3`.
- Run-manifest facade `evaluation/joint_evidence_run_manifest.py` (base `evaluation/g3_joint_evidence_run_manifest.py`): `c44ab0d3ee9e5ad196ca4dfd715f5b120c8c563ec30133c2bf732d257b18f26b` (base) and `92b065d5978cd4f783ac05de4576965c15540cc55165900f38bc8664617f9d63` (pre-split) → `fcbd0a189bc78a6d422f15ff344b232952302838c59c073c827c0812d92a3ce3`.
- Acquisition facade `evaluation/acquisition_package.py` (base `evaluation/g3_acquisition_package.py`): `07bb9da6c12b86503218f8c087dfd430e4f2363987c4a9c56d1c86fcca056db3` (base) and `0a8db8921fe180f2bdaaec0adb20de957944f79ccddd5daf6440dacf2c7159ae` (pre-split) → `05e11caeb31a8bb17d3333ab6152592b8ef4881336d8283c4c9e92e58d76e185`.
- PUSCH facade `receivers/nr/pusch.py`: `b6a632a92fbc92a294fe7baf718a69b665635e4e5d0e71f4528b5b4d1333f003` (base and pre-split) → `f13ed883a2d546bb5e90d0ac7cb9d314c88c2b05e9f5719fd7c2f5c40af42895`.
- TDL development audit (base `experiments/audit_g5_tdl_dmrs_density_development.py`): `156d5617393f208644bd9fc79d7e5a3076e1a90a568d05a1d9db0f7d4069b7b7` → `2a2b7454d0796db1ef9d2eef1cdbfd57dc05219181ee1ba5be9208449c43da90` (now `experiments/audit_tdl/dmrs_density_development.py`).
- Requirement ledger `docs/research/REQUIREMENT_LEDGER.json`: base de1c2b6d7b366cd3eb03de905915b7c2c5215cda1b54e6c618c65a7b40259b4e → `5508055d7d613ccd9b2c8d32249a944169bdd07849c612a4eb87d4c10320eb14`.
- Helper inventory `docs/research/HELPER_WIRING_INVENTORY.json`: base `32b22aa4719b9155ba76b59b93d6ac92cafbb02bfe4f70b6cb764899749c5b7e`; current `35e3758b624f78e3f9b08a6313d1bc62431875d3652da5e72c668df9e663c8e9`; 254 helpers.
- Restored to base bytes (digest equals base): the sixteen frozen research files and six contract schemas (`295ca28`, `a161422`).

## Commits since `016168e`

Newest first, from `9959d81` (retained-revision record) through `adcd5cb` (ledger citation and admission wording), `73c6404` (protocol-v5 record), `abe83c7` (V5 design note), `113f5cb` (protocol v5), `d91ad93` (native diagnosis), `7681f9a` (crosswalk deletion), and the earlier refactor commits. `git log --oneline 016168e..9959d81` lists them all.
