# G3 candidate scene-family and license audit

**Status:** candidate audit only; not a G3 qualification or production-scene approval. Project policy is to use OSM-derived scenes subject to ODbL compliance; this does not itself certify any scene bundle or derivative release as compliant.

## Method and reproducibility

The audit loaded eight candidate scenes at 3.5 GHz with the repository's isolated `llvm_ad_mono_polarized` backend selection and activation path. The native backend probe succeeded with one direct path. No `PathSolver` was invoked for the scene inventory. Asset closure and active material state were recorded with the existing `scene_asset_manifest()` and `sionna_scene_material_manifest()` functions. Every scene resolved all referenced files. The five built-in scenes' installed asset-set hashes exactly matched the pinned Sionna RT source checkout.

The source checkout is Sionna commit `2f0a3820269b53fd022780a6f2f5f720f7ceb9b4`, with the Sionna RT v2.2.0 submodule at `15b5ee036917a4c2a9b6420e05570b77717e2fcc`. The audit runtime used Sionna RT 2.2.0, Mitsuba 3.9.1, Dr.Jit 1.5.0, and LLVM CPU. The machine-readable report is [`g3-scene-asset-material-audit-20261023.json`](results/g3-scene-asset-material-audit-20261023.json); its source thesis commit is recorded in that report.

## Scene inventory

| Scene | Candidate role and family status | Asset files | Asset-set SHA-256 | License/provenance status |
|---|---|---:|---|---|
| `box_one_screen` | Controlled box-and-screen fixture; useful for solver mechanics, not an independent urban site | 3 | `0096663dfcaf8a1a1079ba1912b76c226dc927b4c67e691796b2222f43cc98d4` | Bundled Sionna RT scene; package source is Apache-2.0. No separate per-scene data notice was found in the inspected checkout. |
| `simple_street_canyon` | Stylized rectangular-building fixture; useful for controlled geometry, not an independently sourced field site | 8 | `d2824c41043d248c8d3089fe8396b7cc225ad1c328a626ad70c897951797004b` | Bundled Sionna RT scene; package source is Apache-2.0. No separate per-scene data notice was found in the inspected checkout. |
| `etoile` | Paris / Arc de Triomphe OSM-derived site; candidate geographic site | 566 | `d0b6ee5c9406e8b75c5fc17388e16037e988b5057d37bc40fe55699fefba115d` | Project policy permits use subject to ODbL compliance; source attribution and release-specific derivative/redistribution handling must be recorded before release. |
| `florence` | Florence Cathedral OSM-derived site; candidate geographic site | 2,640 | `f5bb0decbe1806daecd10455d846db86a9590f29ff334398d07c18876f038651` | Project policy permits use subject to ODbL compliance; source attribution and release-specific derivative/redistribution handling must be recorded before release. |
| `munich` | Frauenkirche OSM-derived site; candidate geographic site | 2,368 | `df2ea477117f85ce9abffbc46ba30130d49789b3576cb28363350c2b1996670b` | Project policy permits use subject to ODbL compliance; source attribution and release-specific derivative/redistribution handling must be recorded before release. |
| `city41` | Procedurally generated 4×4 urban scene, seed 41 | 27 | `e16f963657b448a9ca6125661caf45c285165c377b497fdd0d92c7408e42d546` | Repository describes these as synthetic, without real-world geometric fidelity. No repository-level LICENSE file is tracked. |
| `city42` | Procedurally generated 4×4 urban scene, seed 42 | 27 | `e0eff74870e0aa24023d399daa0db328ba3b4f906550c75300c899abcd5018d2` | Same generator and family as City41/43; no repository-level LICENSE file is tracked. |
| `city43` | Procedurally generated 4×4 urban scene, seed 43 | 27 | `17e38ae4690fc34f6740d760284f09dc267fb0d2fda494ef9ed4132d97d1da2c` | Same generator and family as City41/42; no repository-level LICENSE file is tracked. |

Sionna RT's [v2.2.0 scene documentation](https://nvlabs.github.io/sionna/v2.2.0/rt/api/scene.html) identifies Etoile, Florence, and Munich as OpenStreetMap-derived and states that their data is under ODbL. The [OpenStreetMap copyright page](https://www.openstreetmap.org/copyright) identifies ODbL and requests contributor attribution; the [Open Data Commons ODbL page](https://opendatacommons.org/licenses/odbl/) describes the database license as attribution and share-alike. The source repository's Apache-2.0 software license therefore must not be used as a blanket license claim for OSM-derived scene data or a dataset containing it. Any distribution plan for those scenes or derived database products requires a deliberate license and attribution review.

The three City scenes are explicitly one procedural family in [`scenes/generated/research_scenes/README.md`](../../scenes/generated/research_scenes/README.md); their distinct seeds and asset hashes do not make them three independent families. Etoile, Florence, and Munich are distinct geographic source sites and are stronger candidates for independent site strata, but this audit does not decide whether they satisfy the frozen G3 term “independent scene/site families.” Their common OSM provenance and Blender conversion workflow must be accounted for in that decision. The box and simple canyon scenes are controlled fixtures, not substitutes for independent urban sites.

## Material findings

At the audited 3.5-GHz frequency, the active Sionna material manifest reports a scattering coefficient of exactly `0.0` for every material in all eight scenes. The pre-existing `S=0.5` interventions are synthetic and uncalibrated, so they do not establish a physically justified nonzero scattering configuration. Consequently, the asset audit establishes **G0.03 scene-file closure for these eight snapshots**, but **does not satisfy G3.03** or qualify any material/site combination for production RT.

## Qualification consequence

The three OSM city scenes are plausible geographic candidates for a later three-site campaign, subject to an explicit G3 family-independence rule and documented ODbL compliance for the intended outputs. The user approved use under that policy. No scene/material set is currently G3-ready: active scattering is zero and calibrated nonzero material evidence is absent. Keep any further scene tracing exploratory and do not produce production D1 RT data from this audit.
