# Phase 0 physics verification status

**Status:** in progress; the canonical dataset gate is not passed and large
dataset generation or model runs remain deferred.

## Governing references

- `MASTER_RESEARCH_PLAN_5G_MULTIPATH_SIONNA_2026-10-21.md`, named by the goal,
  is not present in this checkout. Its current maintained successor is
  [`MASTER_RESEARCH_PLAN.md`](MASTER_RESEARCH_PLAN.md), edition 3.116.
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
| Direct/indirect channel conservation | Partial | `direct_path_mask()` and coefficient-masked CFRs partition a single Sionna 2.2 `Paths` object. The generator uses this same-set direct component; native two-path LLVM tests verify complex CFR conservation, residual indirect power, and independent raw `Paths.a`/`Paths.tau` reconstruction. Both synthetic and explicit 2×3 array modes conserve all antenna-pair axes for all four TX/RX V/VH polarization combinations. Native empty-path fixtures expose and cover zero-length Dr.Jit masking and synthetic-mask broadcasting. A new deterministic blocked-link test separates exactly one interaction from two-or-more interactions and verifies complex CFR reconstruction; it passes on the pinned LLVM backend. Gold values for interaction-specific, cross-polarization, and orientation-dependent coefficients remain open. |
| Absolute power/link budget, path coefficients, and materials | Partial | Native isotropic free-space tests check `|a| = λ/(4πd)`, phase, and `τ=d/c`. A pinned LLVM slab fixture compares end-to-end V/V and H/H reflected coefficients against independent complex TM/TE Fresnel-slab equations; both pass. The RT-to-DMRS adapter now has a distinct absolute-power mode with explicit per-resource-element transmit power and fixed `k_B T B F` noise; a native CUDA regression verifies that half channel gain remains half in the LS estimate, unlike the legacy normalized-SNR mode. Waveform calibration, arbitrary antenna orientations, materials fitted to measurements, and broader solver convergence remain open. Synthetic-material agreement is not real-material calibration. |
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
gain/phase check, empty-path handling, and synthetic/explicit MIMO partitions
across all four V/VH TX/RX polarization pairs;
the complete local suite at that snapshot passed at 88.11% coverage. Ruff
lint/format and mypy pass. Hosted CI run
[39711191556](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39711191556)
passes all four jobs for commit `00928a3`; run
[39711520188](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39711520188)
passes all four jobs for commit `dac545c`; and run
[39712795569](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39712795569)
passes all four jobs for commit `70421a6`; and run
[39713206686](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39713206686)
passes all four jobs for `1a1de6c`. Hosted CI does not install the native
Sionna RT stack, so native regressions are verified locally. The design-level
gates above remain open despite these checks.

**Edition 3.114 follow-up:** `tests/test_integrations/test_rt_blockage_diffuse_native.py::test_blocked_link_separates_single_and_multi_interaction_cfr_contributions`
passes on LLVM with a deterministic 2,000-ray/source solve. It verifies a
screen-blocked link has valid single- and multiple-interaction contributions,
no zero-interaction direct path, and full CFR equal to the masked order-wise
complex sum. The full local suite passes with 625 passed, 3 skipped, and 88.12%
coverage. Ruff check/format (224 maintained Python files), mypy (140 source
files), `uv lock --check`, `uv build`, and `git diff --check` also pass. Hosted
CI [run 39717796831](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39717796831)
passes all four jobs for code commit `9e10b32`; it does not install native RT.
This test improves one synthetic geometry gate but does not qualify real
material calibration, arbitrary antenna orientation, or solver convergence.

**Edition 3.115 follow-up:** `tests/test_integrations/test_rt_to_nr_native.py::test_rt_reflected_path_gain_matches_analytic_fresnel_slab`
passes for both parameterized native LLVM cases. It checks 45° incidence,
image-source path lengths, direct Friis gain, and the complex slab Fresnel
coefficient through the full Sionna RT path solver. The complete local suite
passes with 627 passed, 3 skipped, and 88.12% coverage; Ruff check/format and
mypy across 140 source files also pass. Hosted CI
[39719682064](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39719682064)
passes all four jobs for code commit `498082f`. The test remains local-native
evidence because hosted CI omits Sionna RT.

**Edition 3.116 follow-up:** The RT-to-DMRS adapter now supports a distinct
absolute-power mode: transmit symbols are scaled by explicit watts per active
resource element, thermal noise is fixed by `k_B T B F`, and the LS estimate is
returned in channel-gain units without per-link normalization. Analytic tests
check the noise equation; a native CUDA Sionna PHY regression demonstrates that
halving channel amplitude halves the resulting estimate, while the normalized
SNR diagnostic mode removes that difference. Focused adapter tests pass. This
does not yet include measured transmitter EIRP/antenna gain, receiver noise
calibration, synchronization error, or full waveform/link-budget validation.
The full local suite passes at 88.14% coverage; Ruff check/format across 224
maintained files, mypy across 140 source files, `uv lock --check`, and `uv build`
pass. Hosted CI for this change is pending; hosted CI does not install native RT
or exercise the CUDA-only PHY test.
