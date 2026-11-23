# Production-structure refactor: verification record

Branch `refactor/production-structure` (local only, not pushed). Evidence base
commit `016168e`. Every result below is the exact outcome of the command shown.
Logs are kept in the ignored state root
`/home/l2nguyen/agent-final-2026-11-22/refactor-baseline/`.

## Gate results (`./tools/check`, system python3 3.8 with the project `.venv`)

| Gate | Command | Exit | Result |
| --- | --- | --- | --- |
| lock | `./tools/check lock` | 0 | `uv lock --check`: Resolved 102 packages |
| lint | `./tools/check lint` | 0 | ruff check passed; `758 files already formatted` |
| typecheck | `./tools/check typecheck` | 0 | `Success: no issues found in 394 source files` |
| test | `./tools/check test` | 0 | `3697 passed, 223 skipped` (241.98 s), no torch installed |
| neural | `./tools/check neural` | 0 | After installing CPU torch 2.11.0 into `.venv` (the version CI pins): `3924 passed, 170 skipped`; coverage `76%`, above the 75% threshold (`uv pip install --python .venv/bin/python torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`). The earlier run without torch failed at 71.10%; that result is superseded. |
| build | `./tools/check build` | 0 | wheel `5g_nr_channel_condition-0.2.0` built; `import-smoke-ok` |
| agent | `./tools/check agent` | 0 | `180 passed` |

## Torch-dependent tests (`envs/py312-gpu`, `CUDA_VISIBLE_DEVICES=-1`)

Full suite with coverage, run as `pytest -p no:cacheprovider` in the torch
environment (torch 2.14.1+cu126):

- Exit 1. `18 failed, 5165 passed, 58 skipped, 11 warnings in 3105.79s`.
- Coverage `TOTAL 80%`, which meets the 75% threshold.

Classification of the 18 failures:

- **Fixed by this step (1):** `tests/test_experiments/test_ordered_impairments_receiver.py::test_independent_snr_curve_retains_each_condition_input_across_snr`. Cause: the test stubbed `_recompute_waveform_diagnostics` on the package, but the audit calls it from `independent.py` and `waveform.py`. The test now stubs both modules. The 18-test rerun reports it as passed.
- **Plot test (`test_plot_rt_urban43_spatial_holdout_3d.py`):** the retained audit records the scene path of the machine that produced it. `scripts/plot_rt_urban43_spatial_holdout_3d.py` now maps a recorded absolute path onto this checkout when that path does not exist. The retained JSON is unchanged. The test passes in the torch environment (`1 passed`) and skips in `.venv`, where matplotlib is absent.
- **Rerun on the current tree (17 tests):** `17 failed, 1 passed in 22.04s`. The 17 are the same tests that fail on the pre-refactor tree (see next section), plus the plot test.
- **Pre-refactor check:** the same 9 test files run on the `016168e` extract give `16 failed, 113 passed`. The failing set is identical to the current nine-file failures. `tests/test_scripts/test_plot_rt_urban43_spatial_holdout_3d.py` also fails on the `016168e` extract. It reads a hard-coded `/home/nguyenl37/...` path recorded in a retained audit, so it is an environment failure, not a refactor failure.

Conclusion: the refactor adds no torch-environment failures. All 17 remaining failures predate it.

## Native parity (`-m native_sionna`, `CUDA_VISIBLE_DEVICES=-1`, `DRJIT_LIBLLVM_PATH` set)

- Collected on both trees: `1235` native tests (`--collect-only` with addopts cleared).
- Current tree: exit 1. 5 failed, 52 skipped (sum of skipped entries in the log). Passed = 1235 − 5 − 52 = **1178** (derived, because the `-q` summary line was suppressed).
- `016168e` baseline: exit 1. The same 5 tests fail. Its log used `-rfE`, so skipped tests were not listed and its pass and skip counts are not recorded. The skip reasons are environmental: `mitsuba` and `sionna.rt` are not installed in this environment, and CUDA is hidden.
- Failing tests in both trees (identical): `test_cdl_dual_polarized_array_covariance_matches_pinned_reference[A|B|C]`, `test_native_rt_cochannel_pusch_dmrs_estimate_matches_two_cross_links`, `test_free_space_isotropic_link_is_reciprocal_when_tx_and_rx_are_swapped` (the last raises `ModuleNotFoundError: mitsuba`).

Skipped tests are reported as skipped and are not counted as passed.

## Digests changed

Full list with old and new SHA-256 for every path that differs between
`016168e` and the current tree: [refactor-digest-changes.tsv](../reference/refactor-digest-changes.tsv)
(see the file for the row count; rename-only moves keep their bytes).

Retained evidence keeps its recorded digests. Changes that retained evidence binds are recorded in
[digest-rebindings.json](../reference/digest-rebindings.json):

- **Rayleigh native test** (the only changed file whose base digest a retained result records): `e79903ac4bec37740728988e302d619e01dd6d4eea7b199da187dfbf70ad9c87` → `bc54333a64aae2ea388aa276d234c3c34a31e6a0ff14c45610a214837aba110e`, added as a new rebinding entry. The older entry is kept.
- **Frozen research documents (49) and retained replay fixtures (2):** restored to base bytes. The old and new digests for the 49 are in the restore list of commit `bf89488`; the two fixtures are `experiments/results/ordered_impairments_cpu_{fixture,replay}_2026-10-23.json`.
- **CDL-E protocol pin:** `78ddca6ec9592fe91c574c9d82dfeffed073d0695d52436cf78f54886cbb70ff` → `784750be5b1b7ddc8b95cdd3cd006f14be9b04bdabd772b0067f63f9f40ac73e` (the original), reverted in `scripts/run_cdl_e_fresh_seed_replication.py` and `datasets/schema/cdl-e-covariance-replication.v1.schema.json`.
- **Helper inventory:** `docs/research/HELPER_WIRING_INVENTORY.json` regenerated; its source-digest map now reflects current bytes.

## Step-by-step outcome

1. **Helper inventory (done).** The scanner follows package re-exports to the defining submodule. Helper count is **254**, the same qualified names as before. `tests/test_scripts/test_audit_helper_wiring.py` passes.
2. **Renames (done).** 126 identifiers renamed; 4 kept. Kept names and reasons: [NAMING_EXCEPTIONS.md](../architecture/NAMING_EXCEPTIONS.md).
3. **Merges (deferred).**
   - `evaluation/cdl_*covariance*.py` (9 modules): each is an independent reference for a separate frozen protocol. Merging would weaken the independence the protocols rely on and produce one module of about 2,600 lines. Also, the goal named `integrations/sionna`, but these modules are in `evaluation/`.
   - `experiments/audit_tdl_*` audits: each audits a different frozen protocol. Merging would change audit provenance.
   - `evaluation/*_joint_*`: already concept-named after the rename pass. The remaining candidates are in the digest-bound evaluator roster.
4. **Splits (done in part).** Six modules split into packages: `datasets/split_policy`, `datasets/episode_store`, `integrations/sionna/adapter`, `integrations/sionna/network_phy`, `scripts/audit_channel_split_lineage`, `integrations/sionna/srs/srs_constants`. `receivers/nr/pusch.py` was split and reverted, because retained results and about 18 research tools hash it as one file. Digest-bound modules are deferred with digests in [MODULE_SPLIT_DESIGN_NOTES.md](../architecture/MODULE_SPLIT_DESIGN_NOTES.md): `experiments/ordered_impairments_receiver.py`, `evaluation/sample_size_planning.py`, `evaluation/acquisition_package.py`, `evaluation/joint_evidence_run_manifest.py`.
5. **Docs (done in part).** 24 non-frozen documents moved from `docs/research/` to `docs/guide/`, `docs/architecture/`, `docs/operations/` and `docs/reference/`. Indexes are generated by `scripts/build_docs_index.py`; `docs/README.md` reaches every document. Link check (`scripts/check_doc_links.py`): 35 broken links remain, all inside frozen text that cannot change: `UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md` (33), `RECEIVER_SUPPORT_MATRIX_2026-11-12.md` (1), `CDL_DUAL_ENDPOINT_PATTERN_PROTOCOL_2026-10-25.md` (1). They point at pre-refactor `G*` file names. Protected authorities that name moved documents (`EXECUTION_QUEUE.md`, `REQUIREMENT_LEDGER.json`) are also unchanged.
6. **Topic notes (done).** Five notes in `docs/research/`: `PLAN_ARCHITECTURE.md`, `PLAN_DATA_GENERATION.md`, `PLAN_RECEIVERS_AND_TARGETS.md`, `PLAN_QUALIFICATION_GATES.md`, `PLAN_OPEN_QUESTIONS.md`. The archived plan is not restored to the tree.
7. **Package list (done).** One list in `config/packages.toml`. `tools/sync_packages.py` derives the setuptools, coverage, ruff and mypy lists in `pyproject.toml`. `tools/check` reads the same list. A test checks that `pyproject.toml` matches the source.
8. **Structure tests (done).** `tests/architecture/test_repository_structure.py` checks: no `.gitkeep` under `data/`; no G-number or edition token in tracked paths; no runtime import cycle between packages (module-level imports, `TYPE_CHECKING` excluded); a README in every package directory. Currently all pass.
9. **Verification (this document).** Gate results and torch and native results above.

## Remaining gaps

- The `neural` gate now passes in `.venv` (torch CPU 2.11.0). The `test` gate result above predates the torch install; the `neural` run covers the full suite and supersedes it.
- The native baseline's pass and skip counts are not recorded (see above).
- Stale links in frozen text (35) and protected authorities (3 documents) are unchanged, by rule.
- Items deferred in step 3 and step 4 need owner decisions before they can be done.
