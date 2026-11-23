# integrations.sionna.adapter

Public Sionna adapter: ray-traced channel generation entry points.

## Purpose

This package is one module split into concept modules. It keeps the public names of the original module, so callers import from `integrations.sionna.adapter` unchanged.

## Modules

| Module | Contents |
| --- | --- |
| `generate.py` | Ray-traced channel generation for the Sionna adapter |
| `raytrace.py` | Sionna ray-tracing scene, path metrics, and publication helpers |

## Public API (20 names)

Re-exported from `__init__.py` through `__all__`:

`GridConfig`, `RTPaths`, `RayScene`, `RaytraceGenerationError`, `_RT_EXECUTION_PROVENANCE_SCHEMA_PATH`, `_attach_rt_execution_provenance`, `_mask_path_coefficients`, `_phy_cir_to_cfr`, `_publish_static_rt_sample_umcd`, `_scene_family_run_binding`, `authorize_rt_data_generation`, `build_rt_data_generation_request`, `count_valid_paths`, `direct_indirect_diagnostics`, `direct_path_mask`, `generate_standardized`, `has_valid_path`, `rt_cfr`, `generate_raytraced`, `generate_raytraced_cached`

## Invariants

- Public names and behaviour are unchanged by the split; tests import the package, not the modules.
- Module-level imports point down the layers. Any cycle is a defect: `tests/architecture` checks this.
- Files that retained evidence records by path are listed in `docs/reference/digest-rebindings.json`.

