# Phase 0 physics verification status

**Status:** in progress; the canonical dataset gate is not passed and large
dataset generation or model runs remain deferred.

## Governing references

- `MASTER_RESEARCH_PLAN_5G_MULTIPATH_SIONNA_2026-10-21.md`, named by the goal,
  is not present in this checkout. Its current maintained successor is
  [`MASTER_RESEARCH_PLAN.md`](MASTER_RESEARCH_PLAN.md), edition 3.111.
- [`UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md`](UNIFIED_MULTIPATH_SUPERDATASET_MASTER_PLAN_2026-10-22.md)
  is now tracked here. The ignored source copy under `outputs/` has SHA256
  `4554b6b6a478a5828881a42f41080f7951c59ee042c3883c9819f231fa17d626`; the
  tracked copy removes trailing whitespace and has SHA256
  `d8066029447e735dddbd95b60f487f0af257d7730476b92dbb2f5b3896f5bebe`.
- The local external Sionna reference is `/home/nguyenl37/sionna`, tag v2.2.0,
  commit `2f0a3820269b53fd022780a6f2f5f720f7ceb9b4`; RT submodule commit
  `15b5ee036917a4c2a9b6420e05570b77717e2fcc`.

## Gate ledger

| Requirement | State | Current evidence or remaining work |
|---|---|---|
| Audit existing repository and pinned Sionna implementation | In progress | The repository already contains analytical-channel fixtures, a native RT-to-NR bridge, CPU/LLVM RT tests, trajectory generation, NR receiver adapters, QA and provenance tooling. The pinned `Paths.cfr()` implementation was inspected directly. Continue the field/axis/units/material audit before declaring compatibility complete. |
| Preserve absolute path delay and phase in RT CFR | Adapter fixed; targeted native check passes | `integrations/sionna/adapter.py::rt_cfr` now passes `normalize_delays=False` and `normalize=False`. The native 10 m free-space test compares adapter output to the pinned `Paths.cfr()` result and verifies it differs from the delay-normalized response. Generation report schema v2 records the absolute-delay channel contract. |
| Test baseband phase and frequency-bin mapping | Partial | Native 10 m free-space checks reconstruct CFR manually from `Paths.a`, `Paths.tau`, the explicit `exp(-j 2π f_c τ)` passband-to-baseband factor, and baseband offsets. Even (24) and odd (25) NR FFT grids match the CIR/delay Fourier sum and bridge frequency-bin coordinates. A native two-path manual coefficient/delay reconstruction also matches within tolerance for pinned single-precision coefficients. Nonzero timing-reference shifts, stronger independent coefficient gold values, and time-varying bins remain. |
| Direct/indirect channel conservation | Partial | `direct_path_mask()` and coefficient-masked CFRs partition a single Sionna 2.2 `Paths` object. The generator uses this same-set direct component; native two-path LLVM tests verify complex CFR conservation, residual indirect power, and independent raw `Paths.a`/`Paths.tau` reconstruction. Both synthetic and explicit 2×3 array modes conserve all six link axes. Native empty-path fixtures exposed and fixed zero-length Dr.Jit masking and synthetic-mask broadcasting; both modes now return finite zero CFRs. Multi-depth interactions and non-synthetic multi-polarized partitions remain to verify. |
| Absolute power/link budget, path coefficients, and materials | Partial | The native isotropic 10 m path now checks `|a| = λ/(4πd)` and zero phase against Friis, in addition to `τ=d/c`; raw `Paths.a`/`Paths.tau` reconstruction is checked in free space and a two-path scene. Link-budget propagation through receiver normalization/noise, polarization and orientation conventions, and calibrated material/path-gain comparisons remain. Synthetic material sensitivity is not material calibration. |
| Geometric labels and invalid/no-path states | Partial | Six selected scenes have geometry-label and solver-outcome audits. The generator still exports only accepted balanced samples; empty solver outcomes are counted but are not canonical outage labels. Add boundary, penetration, orientation and uncertain/censored-state coverage. |
| Unified immutable data layers and target masks | Open | Existing `ChannelDataset` and NR observation/target contracts are useful starting points but do not implement the unified parent-episode, ragged path table, per-target observability masks, and linked-view schema in UMCD-1. |
| Regenerate canonical data after the delay-contract fix | Not started; deferred until remaining physics gates pass | Existing NPZ datasets, DMRS views, and fitted model reports are preserved as legacy normalized-delay runs. No old output was rewritten or relabeled. Regenerate with new source hashes only after the applicable D1 physics gates pass. |
| Scale/training and research claims | Deferred | Do not scale or interpret old model metrics as canonical absolute-channel evidence until the preceding gates are verified. |

## Latest focused verification

`uv run --no-sync --locked python -m pytest --no-cov -vv
tests/test_integrations/test_sionna_adapter.py
tests/test_integrations/test_rt_mock_integration.py
tests/test_integrations/test_rt_to_nr_native.py` passed **21 tests** in 20.80 s
on the pinned CPU/LLVM RT environment, including native 1×1 and explicit 2×3
array component-conservation cases. The complete repository suite passed **481
tests, 3 skipped, 88.12% coverage** in 337.93 s; that run began before the
final `_valid`-mask bookkeeping adjustment, and the focused native suite passed
after it. Ruff check and formatting, mypy (139 source files), `uv lock --check`,
`uv build`, and `git diff --check` pass for commit `2b03354`. Hosted CI run
[39710231241](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39710231241)
passes all four jobs for that commit. Follow-up local focused adapter/mock/native
integration checks pass after adding even/odd-grid frequency mapping, raw
coefficient/delay reconstruction for a two-path scene, a direct Friis
gain/phase check, empty-path handling, and synthetic/explicit MIMO partitions;
the complete current local suite also passes at 88.11% coverage. Ruff
lint/format and mypy pass. Hosted CI run
[39711191556](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39711191556)
passes all four jobs for commit `00928a3`; run
[39711520188](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39711520188)
passes all four jobs for commit `dac545c`; and run
[39712795569](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39712795569)
passes all four jobs for commit `70421a6`. Hosted CI does not install the native
Sionna RT stack, so native regressions are verified locally. The design-level
gates above remain open despite these checks.
