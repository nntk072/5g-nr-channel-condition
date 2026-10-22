# Phase 0 physics verification status

**Status:** in progress; the canonical dataset gate is not passed and large
dataset generation or model runs remain deferred.

## Governing references

- `MASTER_RESEARCH_PLAN_5G_MULTIPATH_SIONNA_2026-10-21.md`, named by the goal,
  is not present in this checkout. Its current maintained successor is
  [`MASTER_RESEARCH_PLAN.md`](MASTER_RESEARCH_PLAN.md), edition 3.134.
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
| Test baseband phase and frequency-bin mapping | Partial | Native 10 m free-space checks reconstruct CFR manually from `Paths.a`, `Paths.tau`, the explicit `exp(-j 2π f_c τ)` passband-to-baseband factor, and baseband offsets. Even (24) and odd (25) NR FFT grids match the CIR/delay Fourier sum and bridge frequency-bin coordinates. A native two-path manual coefficient/delay reconstruction also matches within tolerance for pinned single-precision coefficients. A native 0.37 ns common delay-origin shift now verifies the expected carrier-plus-subcarrier phase rotation and confirms `normalize_delays=True` removes that common reference. Multi-path clock/timing-offset identifiability, reference-estimation behavior, stronger independent coefficient gold values, and time-varying bins remain. |
| Direct/indirect channel conservation | Partial | `direct_path_mask()` and coefficient-masked CFRs partition a single Sionna 2.2 `Paths` object. The generator uses this same-set direct component; native two-path LLVM tests verify complex CFR conservation, residual indirect power, and independent raw `Paths.a`/`Paths.tau` reconstruction. Both synthetic and explicit 2×3 array modes conserve all antenna-pair axes for all four TX/RX V/VH polarization combinations. Native empty-path fixtures expose and cover zero-length Dr.Jit masking and synthetic-mask broadcasting. A new deterministic blocked-link test separates exactly one interaction from two-or-more interactions and verifies complex CFR reconstruction; it passes on the pinned LLVM backend. Gold values for interaction-specific, cross-polarization, and orientation-dependent coefficients remain open. |
| Absolute power/link budget, path coefficients, and materials | Partial | Native isotropic free-space tests check `|a| = λ/(4πd)`, phase, and `τ=d/c`. A pinned LLVM slab fixture compares end-to-end V/V and H/H reflected coefficients against independent complex TM/TE Fresnel-slab equations; both pass. The RT-to-DMRS adapter now has a distinct absolute-power mode with explicit per-resource-element transmit power and fixed `k_B T B F` noise; a native CUDA regression verifies that half channel gain remains half in the LS estimate, unlike the legacy normalized-SNR mode. Waveform calibration, arbitrary antenna orientations, materials fitted to measurements, and broader solver convergence remain open. Synthetic-material agreement is not real-material calibration. |
| Geometric labels and invalid/no-path states | Independent scene-geometry oracle implemented; coverage partial | New RT generation uses finite Mitsuba segment visibility as the primary geometric LOS label and cross-checks each determinate result with depth-zero Sionna PathSolver. Four parallel boundary probes and a grazing-incidence threshold mark uncertain links ambiguous; ambiguous links and geometry/solver disagreements are excluded and counted. Native LLVM tests cover clear, blocked, and screen-edge cases. Full penetration/orientation/coplanar coverage remains; empty full-solver outcomes are still counted but are not canonical outage labels, and balanced accepted samples remain a selected subset. Existing datasets are unchanged and must not be relabeled retroactively. |
| Unified immutable data layers and target masks | Partial | New additive UMCD v1 contracts model a multi-node parent episode, explicit antenna element/port weights, directed links, link/frame/port-keyed ragged path truth with separate solve outcomes, named-axis per-link CFR, and per-target element validity/reason codes. `datasets/episode_store.py` publishes separate SHA-256-addressed, pickle-free layers and a manifest-last immutable bundle. `integrations/sionna/path_oracle.py` converts pinned Sionna 2.2 `Paths` results into this ragged contract, including per-port coefficients, absolute delays, ordered interaction flags, scene-local object/primitive IDs, named-frame interaction vertices in meters, Doppler and wrapped azimuth/elevation. `integrations/sionna/rt_network.py` assembles complete link/frame path tables using explicit device bindings and link-local port maps. `integrations/sionna/array_layouts.py` now extracts static Sionna scene devices and array feed geometry with a native 2-TX×2-RX dual-polarized test. Remaining: pose timelines, RF-chain and polarization calibration, solver/material provenance, partial field validity, receiver-view linkage/access enforcement, split registry, schema migration, and production-scale Arrow/Zarr storage. |
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
pass. Hosted CI
[39721406035](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39721406035)
passes all four software jobs; it does not install native RT or exercise the
CUDA-only PHY regression.

**Edition 3.117 follow-up:** Added typed UMCD v1 episode, explicit antenna
element/port maps, directed links, ragged path-oracle, named-axis physical
channel, and reason-coded target contracts. The immutable episode store writes
separate content-addressed NPZ layers for metadata, path truth, channels, and
targets and publishes a versioned manifest last. Dataset tests, the complete
local suite (87.77% coverage), Ruff, mypy, lock validation, and package build
pass. Hosted CI
[39723486090](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39723486090)
passes all four jobs. This does not yet include receiver-view linkage, motion,
full solver/material provenance, a locked split registry, or production-scale
columnar/chunked storage.

**Edition 3.118 follow-up:** Added the native Sionna `Paths` → UMCD ragged
path-table adapter. It expands synthetic-array device-pair fields over the
coefficient port axes, retains each valid complex coefficient and absolute
delay, counts interaction depth, preserves interaction-type sequences, and
maps Sionna zenith/azimuth plus per-path Doppler to the schema angle convention.
Native LLVM tests compare these extracted arrays against Sionna's raw values
across synthetic and explicit 2×3 arrays, V/VH polarization combinations,
zero-path outcomes, and an explicitly selected pair in a multi-device scene.
All 11 focused cases pass. The complete local suite passes with 518 passed,
3 skipped, and 87.75% coverage; Ruff check/format pass across 231 maintained
files, mypy passes across 146 source files, and `uv lock --check`, `uv build`,
and `git diff --check` pass. Hosted CI
[run 39725668192](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39725668192)
passes all four jobs but does not install native RT. Stage S2 still needs
episode-level device/link enumeration, geometry-derived direct visibility,
solver convergence and power-budget convergence.

**Edition 3.119 follow-up:** Added CSR interaction traces to the UMCD path table:
per-path offsets index ordered Sionna interaction flags, scene-local object and
primitive IDs, and scene-coordinate vertices in meters. The episode store
records these as separate pickle-free arrays and preserves their coordinate
frame; validation rejects incomplete traces and count mismatches. The complete
suite passes with 519 passed, 3 skipped, and 87.77% coverage. Thirteen focused
schema/store/native LLVM cases pass. Ruff, format, mypy (146 files), lock check,
and package build pass. Hosted CI
[run 39727460346](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39727460346)
passes all four jobs but does not install native RT. Object IDs are scene-
specific; geometry-based visibility, full episode topology mapping, and solver
convergence remain unverified.

**Edition 3.120 independent geometric visibility oracle:** Added a finite-segment
Mitsuba query as the primary LOS label source, with four parallel boundary probes
and a grazing-incidence ambiguity rule. Dataset generation cross-checks every
unambiguous geometry result against Sionna's depth-zero LOS-only solver, records
per-link query evidence and separate geometry/solver accounting, and skips
ambiguous or disagreeing candidates. Native LLVM fixture tests pass for clear,
screen-blocked, and screen-edge segments; mock generator tests cover query
errors, solver errors, and geometry/solver disagreement. The updated 3D City43
figure also increases building contrast and shows the buildings alongside all
three transmitter positions and receiver samples. The full local suite passes
with 522 passed, 3 skipped, and 87.75% coverage. Ruff check and formatting pass
across 232 maintained Python files; mypy passes across 147 configured source
files; `uv lock --check`, `uv build`, and `git diff --check` pass. Hosted CI is
pending. Historical datasets and reports retain their recorded label provenance
and have not been rewritten.

**Edition 3.123 controlled-link convergence diagnostic:** A schema-v2 native LLVM Sionna 2.2 runner compares a same-scene visible LOS link and a segment-blocked geometric-NLOS link with absolute CFR delays, independent visibility records, multiple deterministic seeds, ray budgets, and interaction depths. Sample-budget changes at depth 7 are relatively small for several aggregate observables at this one thin-screen fixture, but the across-seed CFR and path-count spread is nonzero. Depth 3–7 materially changes path counts, CFR, delay spread, and angular spread, so G3.01 remains open. The largest finite budget is only a comparison reference. The [convergence diagnostic](figures/rt-box-los-nlos-convergence-20261022.png) has no building geometry; the separate [City43 spatial figure](figures/rt-urban43-spatial-holdout-20261022.png) displays the synthetic 4×4 building layout and three TX regions with 120 sampled links. Neither image represents measured buildings or field propagation. The local complete suite passes at 87.69% coverage; Ruff check and format pass across the maintained scope, mypy passes for 150 source files, and `uv lock --check`, `uv build`, and `git diff --check` pass. GitHub Actions [run 39739420589](https://github.com/nntk072/5g-nr-channel-condition/actions/runs/39739420589) passes all four jobs. Hosted CI has no Mitsuba, so the marked native RT matrix test is skipped there and executed locally.

**Edition 3.124 max-depth-9 follow-up (negative convergence result):** At depth 8→9, same-scene LOS valid-path counts increase from 802–804 to 1,223–1,226 and RMS delay spread from about 40.2 ns to 45.6 ns; geometric-NLOS counts increase from 482–483 to 852–855 and delay spread from about 35.0 ns to 42.3 ns. The depth-8 CFR relative L2 error against depth 9 is about 0.29 for LOS and 0.11–0.14 for NLOS. These results refute a convergence claim through depth 9 for this fixture; G3.01 is still open. Report: `results/rt-box-los-nlos-convergence-depth9-20261022.json`.

**Edition 3.125 scattering-coefficient audit:** The pinned `box_one_screen` runtime scene has zero scattering on both radio materials because its XML omits the coefficient and Sionna 2.2 ITU material defaults to S=0. Consequently, prior v2 depth/budget results with the diffuse solver flag set do not contain diffuse paths. The native v3 test now records material coefficients and per-path interaction bits and verifies that a synthetic screen override S=0.5 creates diffuse paths (55 LOS and 5 blocked-link diffuse paths at 1k samples, depth 3, seed 11; S=0 gives zero). A 200–100k sample sweep at S=0.5 confirms diffuse-path counts scale with the ray budget while the nondiffuse geometric component remains stable for this fixture. LOS diffuse metrics stabilize more than the blocked-link metrics; CFR differences versus the 100k reference remain too large for a convergence claim. This validates solver mechanics only, not the physical value of S=0.5. G3.02/G3.03 remain open pending justified material values and broader scenes; G3.04 is demonstrated only in this controlled fixture. Reports: `results/rt-box-los-nlos-material-component-s0-20261022.json`, `results/rt-box-los-nlos-material-component-s05-20261022.json`, and `results/rt-box-los-nlos-diffuse-budget-sweep-s05-components-20261022.json`.

**Edition 3.126 nonzero-scattering depth sweep (negative):** A complete 48-run LLVM sweep of the paired LOS/blocked links used screen S=0.5, depths 0–6, seeds 11/23/47, budgets 20k/100k, and a 1M path cap. At depth 6/100k, median LOS count is 10,846 (10,546 diffuse) and blocked-link count is 3,539 (3,358 diffuse). Compared with the selected depth-6 CFR reference, depth-5 median relative L2 errors remain 0.360 LOS and 0.898 blocked NLOS; delay and angular spreads continue changing through depth 6. Error is non-monotone, and the reference is not ground truth. The S=0.5 override is synthetic/un-calibrated, so this only establishes that depth sensitivity persists when diffuse paths are emitted in this fixture. It does not pass G3.01 or qualify material/geometry/path-cap coverage. Report: `results/rt-box-los-nlos-convergence-depth6-s05-20261022.json`; figure: `figures/rt-box-los-nlos-convergence-depth6-s05-20261022.png`.

**Edition 3.126 verification:** The complete local suite passes with 547 passed, 3 skipped, and 87.69% coverage. Maintained-scope Ruff check/format (241 files), mypy (151 source files), lock validation, package build, and whitespace checks pass. Native Sionna evidence is CPU/LLVM-specific; hosted CI results must be recorded separately after pushing the code revision.

**Edition 3.127 path-cap stress (partial):** A matched depth-6, 100k-sample, three-seed comparison used caps 1k/10k/1M and the synthetic screen S=0.5 fixture. At cap 1k, valid paths fall to 999 LOS and 537–539 blocked NLOS, with lower diffuse power and smaller delay/angular spreads. At cap 10k, blocked-link metrics match the 1M reference exactly for all three seeds, while LOS remains at 8,210–8,245 paths versus 10,748–10,850 at 1M and delay spread 22.8–23.0 ns versus 28.9 ns. This shows the cap can materially shape sampled diffuse outcomes. It does not prove 10k is sufficient elsewhere or establish CFR equivalence; gate G3.05 remains open. The cap reports and comparison figure are in `results/rt-box-los-nlos-convergence-depth6-s05-cap{1000,10000}-20261022.json` and `figures/rt-box-los-nlos-path-cap-sweep-s05-20261022.png`.

**Edition 3.128 spatial-label and figure correction:** The dataset generator's contract is `label = int(visibility.visible)`, so label 1 is clear geometric LOS and label 0 is blocked geometric NLOS. A native LLVM rerun of the independent segment-visibility oracle over all 120 saved City43 spatial-holdout links found 60/60 LOS clear, 60/60 NLOS blocked, zero mismatches, and zero ambiguous links. The saved data were correct; the earlier figure had the class names/colors reversed. The plot mapping and image are corrected, with label semantics, sample counts, input/scene hashes, and image hashes recorded in `figures/rt-urban43-spatial-holdout-20261022.json`. Reproduce the exhaustive check with `scripts/audit_rt_spatial_holdout_labels.py <run>/nr-dmrs-observations.npz <new-report.json> --mitsuba-variant llvm_ad_mono_polarized`; its 120 per-link visibility outcomes are recorded in `results/rt-urban43-spatial-label-audit-20261022.json`. This verifies geometric labels for this saved synthetic dataset, not measured propagation or channel-coefficient correctness.

**Edition 3.129 City43 path-cap follow-up:** Reused the independently audited City43 building scene and clear/blocked center links to compare caps 1k/10k/1M at depth 3, 20k samples, seeds 11/23/47, and uncalibrated S=0.5 concrete/glass/metal overrides. LOS has 981 paths (978 diffuse) at cap 1k and 1,852–1,885 (1,843–1,876 diffuse) at both larger caps; median RMS delay spread increases from 22.26 to 34.05 ns. The blocked link's 335–349 paths (323–338 diffuse) and aggregate reported metrics are unchanged across caps in this scene. This adds a building-scene case to the earlier thin-screen evidence but remains one synthetic, uncalibrated case, and the compact summaries do not contain cross-cap complex channel vectors. G3.05 remains open; no safe general cap or convergence claim is made. See `results/rt-city43-los-nlos-depth3-s05-cap{1000,10000,1000000}-20261022.json`, `figures/rt-city43-los-nlos-path-cap-sweep-s05-20261022.json`, and the [3D building-scene figure](figures/rt-city43-los-nlos-cap-sweep-scene-20261022.png). The reproducible figure generator is `scripts/plot_rt_city43_cap_sweep_scene.py`; Ruff, format, and mypy pass for that script.

**Edition 3.130 cross-cap CFR-vector verification:** Added an opt-in schema-v4 mode to `run_rt_solver_convergence.py` that records every complex CFR sample as aligned float64 real/imaginary vectors; the path-cap auditor now validates grid compatibility and reports per-link/seed relative complex-L2 error against the highest evaluated cap. A matched re-run of the City43 1k/10k/1M sweep finds median LOS errors of 0.1313 (range 0.0857–0.1480) at 1k and 4.55e-7 at 10k relative to 1M. NLOS errors remain below 2.62e-7 at all caps. This is strong direct numerical evidence for these paired links, yet only one geometry/material/depth/budget and three seeds; G3.05 stays open, and 1M is only a finite reference. The per-run reports and the [CFR-error comparison figure](figures/rt-city43-los-nlos-path-cap-sweep-s05-cfr-20261022.png) are the three `results/rt-city43-los-nlos-depth3-s05-cap*-cfr-20261022.json` files and the matching `figures/rt-city43-los-nlos-path-cap-sweep-s05-cfr-20261022.json` sidecar.

**Edition 3.131 second-geometry cap and replay evidence:** On `box_one_screen`, at depth 6/100k samples with three seeds and screen S=0.5, median LOS CFR error against cap 1M is 0.417 at cap 1k and 0.300 at cap 10k; blocked-NLOS is 0.565 at 1k and 6.76e-7 at 10k. This contrasts with City43, where 10k was essentially equal to 1M for both links, and confirms that the path cap is geometry/link dependent. G3.05 is still open. An independent repeat of the box 1k report compares all 42 saved CFR grids: path counts match in 42/42 cases, hashes match exactly in 13/42, and the maximum numerical CFR relative L2 error is 4.26e-7. The reusable replay auditor passes the explicitly recorded 1e-6 diagnostic tolerance; this is not proof of bitwise determinism or backend-independent replay, so G3.08 remains open. Evidence: `results/rt-box-los-nlos-convergence-depth6-s05-cap{1000,10000,1000000}-cfr-20261022.json`, `results/rt-box-los-nlos-convergence-depth6-s05-cap1000-cfr-repeat-20261022.json`, `results/rt-box-los-nlos-convergence-depth6-s05-cap1000-replay-20261022.json`, `figures/rt-box-los-nlos-path-cap-sweep-depth6-s05-cfr-20261022.json`, and the [CFR cap figure](figures/rt-box-los-nlos-path-cap-sweep-depth6-s05-cfr-20261022.png). Run the reusable checker with `scripts/audit_rt_solver_replay.py <reference-v4.json> <replay-v4.json> <audit.json>`.


**Edition 3.132 matched diffuse-scattering intervention:** Re-ran the same two `box_one_screen` links with identical scene, seed, depth, ray budgets (1k/20k), and native LLVM solver configuration, changing only `screen-mat` from S=0 to synthetic S=0.5. The three-seed, two-link, 30-case audit found no diffuse paths in all 30 zero-scattering cases; at S=0.5, 21 cases produced diffuse paths, with none at interaction depth zero. At depth 3/20k rays the median diffuse-path count was 1,151 (range 1,139–1,161) for LOS and 44 (41–47) for blocked NLOS. Relative complex-CFR L2 error against the zero-scattering control had seed medians 0.065 (0.047–0.141) and 0.044 (0.023–0.073), respectively. The [budget figure](figures/rt-box-scattering-control-s05-v4-20261022.png) and `results/rt-box-scattering-control-audit-v4-20261022.json` are reproduced from `results/rt-box-scattering-control-s{0,s05}-v4-20261022.json` with `scripts/audit_rt_scattering_control.py` and `scripts/plot_rt_scattering_budget.py`. This verifies a matched solver-mechanics intervention only: S=0.5 is uncalibrated, and neither this screen geometry nor these finite ray budgets validate real material scattering or convergence. G3.02/G3.03 and broader G3.04 remain open.


**Edition 3.133 City43 building-scene scattering intervention:** Repeated the matched S=0 versus synthetic `itu_concrete` S=0.5 intervention on the 16-building City43 mesh, using the audited center LOS and blocked-NLOS links, three seeds, depths 0-3, and 1k/20k rays/source. The 30-case audit finds no diffuse paths in any zero-scattering case; 24/30 intervention cases produce diffuse paths, with none at depth zero. At depth 3/20k rays, median diffuse counts are 1,256 (range 1,246-1,263) on LOS and 256 (246-259) on NLOS. Relative CFR L2 medians against the zero-scattering control are 0.072 (0.044-0.076) LOS and 0.482 (0.407-0.638) NLOS. The [combined City43 science figure](figures/rt-city43-scattering-buildings-combined-v4-20261022.png) places all 16 OBJ buildings and audited links above the scattering-budget metrics; separate [scene](figures/rt-city43-los-nlos-cap-sweep-scene-20261022.png) and [metrics](figures/rt-city43-scattering-control-concrete-s05-v4-20261022.png) figures remain available. Data, provenance, and paired audit are `results/rt-city43-scattering-control-*-v4-20261022.json`; reproduce the component figures with `scripts/run_rt_solver_convergence.py`, `scripts/audit_rt_scattering_control.py`, and `scripts/plot_rt_scattering_budget.py`, then combine them with `scripts/compose_city43_scattering_figure.py`. This is cross-geometry solver-mechanics evidence under an uncalibrated coefficient, not evidence that concrete has S=0.5 or that the model generalizes to measured buildings. G3.02/G3.03 and broader G3.04 remain open.


**Edition 3.134 City43 building-scene solver replay:** Repeated the identical S=0.5 concrete-only City43 solve (actual 16-building OBJ mesh, audited center LOS/NLOS links, 3 seeds, depths 0-3, 20k rays/source, LLVM CPU). Across 24 paired link/seed/depth cases, path counts match 24/24, CFR byte hashes match 6/24, median complex-CFR relative L2 error is 2.02e-7, and the maximum is 3.35e-7 under the pre-existing 1e-6 diagnostic tolerance. This extends replay evidence to a second scene family, but it is still a single repeated configuration on one LLVM backend; G3.08 remains open pending broader configurations and backend coverage. Reproduce with `scripts/audit_rt_solver_replay.py` on `results/rt-city43-scattering-replay-{reference,repeat}-v4-20261022.json`; the paired report and audit JSON are preserved beside them.

**Edition 3.135 common timing-origin shift:** Extended the pinned Sionna 2.2 native 10 m free-space/NR-grid check by adding a 0.37 ns shift to every RT path delay. With delay normalization disabled, the measured CFR ratio matches `exp(-j 2π (f_c + f_k) Δτ)` across all 24 subcarriers and the CIR retains the shifted absolute delay. With delay normalization enabled, the CFR is invariant to the common shift, demonstrating that this option removes the absolute timing reference. The focused native test passes on CPU/LLVM. This is an API/convention test, not proof that an unsynchronized receiver can identify absolute ToA; joint clock/carrier offset, multipath reference selection, and estimator observability remain open. Reproduce with `uv run --no-sync --locked python -m pytest --no-cov -q tests/test_integrations/test_rt_to_nr_native.py::test_rt_absolute_delay_cfr_maps_to_nr_baseband_bins`.
