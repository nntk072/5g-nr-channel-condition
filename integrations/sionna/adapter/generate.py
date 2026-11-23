"""Ray-traced channel generation for the Sionna adapter."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from artifacts.provenance import package_source_sha256, versions
from datasets.channel import ChannelDataset
from datasets.rt_split_identity import rt_spatial_split_subject_id as _rt_spatial_split_subject_id
from propagation.physics import add_estimation_noise
from propagation.scene_visibility import SceneTriangleInventory, extract_scene_triangle_inventory

from integrations.sionna.candidate_audit import instrument_path_solver_candidate_counter
from integrations.sionna.d1_generation_gate import D1GenerationAuthorization
from integrations.sionna.los_oracle import (
    SegmentVisibilityState,
    independent_mesh_segment_visibility,
    segment_visibility,
)
from integrations.sionna.rt_identity import (
    rt_run_identity,
    rt_runtime_fingerprint,
    sionna_rt_siso_antenna_config,
    static_rt_solver_config,
)

from .raytrace import (
    GridConfig,
    RayScene,
    RaytraceGenerationError,
    _attach_rt_execution_provenance,
    _publish_static_rt_sample_umcd,
    _scene_family_run_binding,
    build_rt_data_generation_request,
    count_valid_paths,
    direct_indirect_diagnostics,
    direct_path_mask,
    has_valid_path,
    rt_cfr,
)

if TYPE_CHECKING:
    from integrations.sionna.backend import RTBackendSelection


def generate_raytraced(
    scene_config: RayScene = RayScene(),
    *,
    grid: GridConfig = GridConfig(),
    seed: int = 42,
    umcd_store_root: str | Path | None = None,
    _backend_selection: RTBackendSelection | None = None,
    _runtime_fingerprint: Mapping[str, str] | None = None,
    authorization: D1GenerationAuthorization | None = None,
    pilot: bool = False,
    approval_manifest: str | Path | None = None,
    expected_approval_sha256: str | None = None,
    _scene_assets_override: Mapping[str, Any] | None = None,
    _scene_override: Any | None = None,
    _scene_geometry_inventory_override: SceneTriangleInventory | None = None,
    _material_manifest_override: Mapping[str, Any] | None = None,
    output_destination: str | Path | None = None,
    failure_report_destination: str | Path | None = None,
    scene_family_candidate_id: str | None = None,
    _authorization_destinations: Sequence[str | Path] = (),
) -> ChannelDataset:
    """Balanced physical LOS/NLOS ray tracing. Fail closed when insufficient examples.

    Full PathSolver includes direct + reflection (and optional diffuse). A
    digest-bound NumPy query over loaded scene triangles determines geometric
    LOS; Mitsuba and depth-zero PathSolver cross-check it, and ambiguous or
    disagreeing candidates are excluded.
    Report how often LOS really contains >=2 valid paths; default rejects
    direct-only LOS. NLOS must contain at least one observable indirect path.
    """
    execution_started_at = (
        datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    command_argv = tuple(sys.argv)
    python_executable = sys.executable
    if authorization is None and not pilot and approval_manifest is None:
        raise PermissionError("direct RT dataset generation requires D1 authorization")
    try:
        from integrations.sionna.backend import activate_rt_backend, select_rt_backend

        selection = _backend_selection or select_rt_backend(scene_config.mitsuba_variant)
        if not selection.probe_passed:
            raise RuntimeError("an unverified Mitsuba backend cannot run RT generation")
        activate_rt_backend(selection)
        import sionna.rt as rt
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
    except ImportError as e:
        raise RuntimeError("Sionna RT 2.2 not installed; install sionna==2.2.0") from e
    cfg = replace(scene_config, mitsuba_variant=selection.variant)
    from integrations.sionna.scene_assets import (
        scene_asset_manifest,
        sionna_scene_material_manifest,
    )

    obstacles = []
    scene_path = Path(cfg.name)
    scene_manifest = scene_path.with_suffix(".manifest.json")
    if scene_path.is_file() and scene_manifest.is_file():
        obstacles = json.loads(scene_manifest.read_text(encoding="utf-8")).get(
            "buildings_and_vehicles", []
        )
    if not (
        cfg.max_depth >= 1
        and cfg.per_class >= 1
        and cfg.max_attempts >= 2 * cfg.per_class
        and cfg.block_size_m > 0
        and (cfg.transmitter_id is None or cfg.transmitter_id.strip())
    ):
        raise ValueError("Invalid ray tracing configuration")
    scene_ref = getattr(rt.scene, cfg.name, cfg.name)
    scene_xml_path = Path(str(scene_ref)).expanduser()
    if scene_xml_path.is_file():
        scene_assets = scene_asset_manifest(scene_xml_path)
    else:
        scene_reference = str(scene_ref)
        scene_assets = {
            "scene_xml_path": scene_reference,
            "asset_count": 0,
            "assets": [],
            "asset_set_sha256": hashlib.sha256(scene_reference.encode()).hexdigest(),
            "unresolved_references": [scene_reference],
            "complete": False,
        }
    scene_family_identity: Mapping[str, Any] | None = None
    if scene_family_candidate_id is not None:
        from datasets.scene_families import (
            load_scene_family_registry,
            scene_candidate_split_metadata,
        )

        geometry_asset_sha256 = scene_assets.get("asset_set_sha256")
        if not isinstance(geometry_asset_sha256, str):
            raise ValueError("scene asset set digest must be a string")
        scene_family_identity = scene_candidate_split_metadata(
            load_scene_family_registry(),
            candidate_id=scene_family_candidate_id,
            actual_geometry_asset_sha256=geometry_asset_sha256,
        )
    scene = _scene_override if _scene_override is not None else load_scene(scene_ref)
    scene.frequency = grid.carrier_hz
    scene.bandwidth = grid.n_subcarriers * grid.spacing_hz
    material_manifest = sionna_scene_material_manifest(scene)
    if (
        _material_manifest_override is not None
        and dict(_material_manifest_override) != material_manifest
    ):
        raise PermissionError("RT material state changed after request authorization")
    scene_geometry_inventory = (
        _scene_geometry_inventory_override
        if _scene_geometry_inventory_override is not None
        else extract_scene_triangle_inventory(scene)
    )
    from integrations.sionna.d1_generation_gate import (
        authorize_d1_generation,
        require_d1_authorization,
    )

    destinations = [*(_authorization_destinations or ())]
    destinations.extend(
        value
        for value in (output_destination, umcd_store_root, failure_report_destination)
        if value is not None
    )
    request = build_rt_data_generation_request(
        cfg,
        grid,
        seed,
        selection=selection,
        scene_ref=scene_ref,
        scene_assets=scene_assets,
        material_manifest=material_manifest,
        scene_geometry_inventory=scene_geometry_inventory,
        destinations=destinations,
    )
    if authorization is None:
        authorization = authorize_d1_generation(
            request,
            pilot=pilot,
            approval_manifest=approval_manifest,
            expected_approval_sha256=expected_approval_sha256,
        )
    require_d1_authorization(authorization, request, expected_operation="rt_data")  # type: ignore[arg-type]
    if getattr(authorization, "mode", None) == "pilot":
        from integrations.sionna.d1_generation_gate import require_pilot_destination

        if output_destination is not None:
            require_pilot_destination(authorization, output_destination, publisher="rt_dataset")  # type: ignore[arg-type]
        if umcd_store_root is not None:
            require_pilot_destination(authorization, umcd_store_root, publisher="umcd_bundle")  # type: ignore[arg-type]
        if failure_report_destination is not None:
            require_pilot_destination(
                authorization, failure_report_destination, publisher="failure_report"
            )  # type: ignore[arg-type]
        if _authorization_destinations:
            for destination in _authorization_destinations:
                require_pilot_destination(authorization, destination, publisher="rt_cache")  # type: ignore[arg-type]
    if scene_xml_path.is_file():
        loaded_scene_assets = scene_asset_manifest(scene_xml_path)
        if loaded_scene_assets.get("asset_set_sha256") != scene_assets.get("asset_set_sha256"):
            raise PermissionError(
                "RT scene asset closure changed between authorization and scene load"
            )
    if sionna_scene_material_manifest(scene) != material_manifest:
        raise PermissionError("RT material state changed after request authorization")
    runtime_fingerprint = (
        dict(_runtime_fingerprint)
        if _runtime_fingerprint is not None
        else rt_runtime_fingerprint(
            {key: value for key, value in versions().items() if key != "git_sha"}
        )
    )
    package_sha256 = package_source_sha256()
    solver_config = static_rt_solver_config(cfg.max_depth, cfg.max_num_paths_per_src)
    if umcd_store_root is not None and not scene_assets["complete"]:
        raise ValueError("UMCD RT publication requires a complete scene-asset closure")
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter(name="research_tx", position=list(cfg.tx_position)))
    receiver = Receiver(name="research_rx", position=list(cfg.rx_center))
    scene.add(receiver)
    full_solver = PathSolver(deterministic=True)
    full_candidate_counter = instrument_path_solver_candidate_counter(full_solver)
    los_oracle = PathSolver(deterministic=True)
    rng = np.random.default_rng(seed)
    hits = {0: 0, 1: 0}
    attempt_counts = {
        "attempts_started": 0,
        "obstacle_rejections": 0,
        "geometry_oracle_attempts": 0,
        "geometry_oracle_calls": 0,
        "geometry_oracle_errors": 0,
        "geometry_oracle_ambiguous": 0,
        "geometry_oracle_coincident": 0,
        "mitsuba_geometry_disagreements": 0,
        "depth_zero_oracle_attempts": 0,
        "depth_zero_oracle_calls": 0,
        "depth_zero_oracle_errors": 0,
        "depth_zero_geometry_disagreements": 0,
        "candidate_geometric_nlos": 0,
        "candidate_geometric_los": 0,
        "class_quota_skips": 0,
        "full_solver_attempts": 0,
        "full_solver_calls": 0,
        "full_solver_errors": 0,
        "full_solver_candidate_cap_filled": 0,
        "full_solver_candidate_cap_rejections": 0,
        "full_solver_empty_paths": 0,
        "los_direct_only_rejections": 0,
        "cfr_conversion_failures": 0,
    }
    full_solver_outcomes_by_geometric_class = {
        class_name: {
            "attempts": 0,
            "calls": 0,
            "errors": 0,
            "empty_paths": 0,
            "los_direct_only_rejected": 0,
            "cfr_conversion_failures": 0,
            "candidate_cap_filled_rejected": 0,
            "accepted": 0,
        }
        for class_name in ("geometric_nlos", "geometric_los")
    }
    samples = []
    ys = []
    groups = []
    domains = []
    sources = []
    meta = []
    umcd_bundle_refs: list[Mapping[str, object]] = []
    solver_failure = None
    for attempt in range(cfg.max_attempts):
        if min(hits.values()) >= cfg.per_class:
            break
        attempt_counts["attempts_started"] += 1
        angle = float(rng.uniform(-np.pi, np.pi))
        distance = float(cfg.radius_m * np.sqrt(rng.uniform(0, 1)))
        x = cfg.rx_center[0] + distance * np.cos(angle)
        y = cfg.rx_center[1] + distance * np.sin(angle)
        z = cfg.rx_center[2]
        if obstacles:
            from propagation.geometry import on_obstacle

            if on_obstacle(x, y, obstacles, margin=0.75):
                attempt_counts["obstacle_rejections"] += 1
                continue
        receiver.position = [float(x), float(y), float(z)]
        attempt_counts["geometry_oracle_attempts"] += 1
        try:
            numpy_visibility = independent_mesh_segment_visibility(
                scene,
                cfg.tx_position,
                (x, y, z),
                inventory=scene_geometry_inventory,
            )
            mitsuba_visibility = segment_visibility(scene, cfg.tx_position, (x, y, z))
        except Exception as error:
            attempt_counts["geometry_oracle_errors"] += 1
            solver_failure = {
                "stage": "geometry_oracle",
                "attempt_index": attempt,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            break
        attempt_counts["geometry_oracle_calls"] += 1
        numpy_state = numpy_visibility.geometry.outcome
        if (
            numpy_state == "ambiguous"
            or mitsuba_visibility.state is SegmentVisibilityState.AMBIGUOUS
        ):
            attempt_counts["geometry_oracle_ambiguous"] += 1
            continue
        if mitsuba_visibility.state is SegmentVisibilityState.COINCIDENT_ENDPOINTS:
            attempt_counts["geometry_oracle_coincident"] += 1
            continue
        numpy_visible = numpy_state == "visible"
        mitsuba_visible = mitsuba_visibility.visible
        if mitsuba_visible is None:
            attempt_counts["geometry_oracle_ambiguous"] += 1
            continue
        if numpy_visible != mitsuba_visible:
            attempt_counts["mitsuba_geometry_disagreements"] += 1
            continue
        label = int(numpy_visible)
        attempt_counts["depth_zero_oracle_attempts"] += 1
        try:
            oracle_paths = los_oracle(
                scene,
                max_depth=0,
                los=True,
                specular_reflection=False,
                diffuse_reflection=False,
                refraction=False,
                diffraction=False,
                seed=seed + attempt,
            )
        except Exception as error:
            attempt_counts["depth_zero_oracle_errors"] += 1
            solver_failure = {
                "stage": "depth_zero_oracle",
                "attempt_index": attempt,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            break
        attempt_counts["depth_zero_oracle_calls"] += 1
        solver_label = int(has_valid_path(oracle_paths))
        if solver_label != label:
            attempt_counts["depth_zero_geometry_disagreements"] += 1
            continue
        attempt_counts["candidate_geometric_los" if label else "candidate_geometric_nlos"] += 1
        if hits[label] >= cfg.per_class:
            attempt_counts["class_quota_skips"] += 1
            continue
        class_name = "geometric_los" if label else "geometric_nlos"
        class_outcomes = full_solver_outcomes_by_geometric_class[class_name]
        class_outcomes["attempts"] += 1
        attempt_counts["full_solver_attempts"] += 1
        try:
            full = full_solver(
                scene,
                max_depth=cfg.max_depth,
                los=True,
                specular_reflection=True,
                diffuse_reflection=True,
                refraction=True,
                diffraction=False,
                samples_per_src=cfg.samples_per_src,
                max_num_paths_per_src=cfg.max_num_paths_per_src,
                seed=seed + attempt,
            )
        except Exception as error:
            class_outcomes["errors"] += 1
            attempt_counts["full_solver_errors"] += 1
            solver_failure = {
                "stage": "full_solver",
                "attempt_index": attempt,
                "exception_type": type(error).__name__,
                "message": str(error),
            }
            break
        attempt_counts["full_solver_calls"] += 1
        class_outcomes["calls"] += 1
        candidate_path_count = full_candidate_counter.last_candidate_path_count
        if candidate_path_count is None:
            raise RuntimeError("RT solver did not expose its pre-validation candidate count")
        if candidate_path_count >= cfg.max_num_paths_per_src:
            attempt_counts["full_solver_candidate_cap_filled"] += 1
            attempt_counts["full_solver_candidate_cap_rejections"] += 1
            class_outcomes["candidate_cap_filled_rejected"] += 1
            continue
        n_paths = count_valid_paths(full)
        if n_paths == 0:
            class_outcomes["empty_paths"] += 1
            attempt_counts["full_solver_empty_paths"] += 1
            continue
        n_direct = count_valid_paths(oracle_paths)
        depth_zero_state = "visible" if n_direct else "blocked"
        mixed = bool(label and n_paths > n_direct)
        if label and cfg.require_los_multipath and not mixed:
            class_outcomes["los_direct_only_rejected"] += 1
            attempt_counts["los_direct_only_rejections"] += 1
            continue
        try:
            clean_channel = rt_cfr(full, grid)
            path_mask = direct_path_mask(full) if label else None
            direct_count = int(path_mask.sum()) if path_mask is not None else 0
            if label and direct_count != n_direct:
                raise ValueError(
                    "Full-solve direct-path mask disagrees with depth-zero oracle: "
                    f"full={direct_count}, oracle={n_direct}"
                )
            direct = (
                rt_cfr(full, grid, path_mask=path_mask)
                if path_mask is not None
                else np.zeros_like(clean_channel)
            )
            diagnostics = direct_indirect_diagnostics(clean_channel, direct)
            observed_channel = add_estimation_noise(clean_channel, grid.snr_db, rng)
        except (ValueError, RuntimeError):
            class_outcomes["cfr_conversion_failures"] += 1
            attempt_counts["cfr_conversion_failures"] += 1
            continue
        samples.append(observed_channel[0])
        ys.append(label)
        scene_id = scene_path.stem if scene_path.suffix else cfg.name
        group_id, split_identity_complete = _rt_spatial_split_subject_id(
            scene_asset_set_sha256=scene_assets["asset_set_sha256"],
            scene_asset_manifest_complete=scene_assets["complete"],
            scene_name=cfg.name,
            rx_position_m=(float(x), float(y), float(z)),
            block_size_m=cfg.block_size_m,
        )
        domain_id = f"rt_{scene_id}"
        if cfg.transmitter_id is not None:
            domain_id = f"{domain_id}_{cfg.transmitter_id}"
        groups.append(group_id)
        domains.append(domain_id)
        sources.append("sionna_rt")
        split_source_family = (
            "sionna_rt"
            if scene_family_identity is None
            else str(scene_family_identity["family_id"])
        )
        meta.append(
            {
                "scene": cfg.name,
                "split_subject_id": group_id,
                "split_parent_type": "network_episode",
                "split_source_family": split_source_family,
                "split_leakage_relations": (
                    []
                    if scene_family_identity is None
                    else scene_family_identity["split_leakage_relations"]
                ),
                "scene_family_candidate_id": (
                    None if scene_family_identity is None else scene_family_identity["candidate_id"]
                ),
                "scene_family_registry_sha256": (
                    None
                    if scene_family_identity is None
                    else scene_family_identity["registry_sha256"]
                ),
                "scene_family_qualification_status": (
                    "NOT_QUALIFIED"
                    if scene_family_identity is None
                    else scene_family_identity["qualification_status"]
                ),
                "split_identity_basis": (
                    "scene_asset_set_sha256+receiver_xy_grid"
                    if split_identity_complete
                    else "provisional_scene_name+receiver_xy_grid"
                ),
                "split_identity_complete": split_identity_complete,
                "split_block_size_m": float(cfg.block_size_m),
                "split_coordinate_frame_id": (
                    scene_assets["asset_set_sha256"] if split_identity_complete else None
                ),
                "split_positions_m": (
                    [float(x), float(y), float(z)] if split_identity_complete else None
                ),
                "scene_asset_set_sha256": scene_assets["asset_set_sha256"],
                "scene_asset_manifest_complete": scene_assets["complete"],
                "scene_asset_count": scene_assets["asset_count"],
                "geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE",
                "geometry_oracle": {
                    "primary_oracle": "numpy_loaded_scene_triangle_inventory",
                    "scene_geometry_sha256": numpy_visibility.scene_geometry_sha256,
                    "scene_object_count": len(numpy_visibility.object_names),
                    "scene_triangle_count": numpy_visibility.triangle_count,
                    "state": numpy_state,
                    "primary_visibility_state": numpy_state,
                    "segment_length_m": numpy_visibility.geometry.segment_length_m,
                    "intersected_triangle_count": len(
                        numpy_visibility.geometry.intersected_triangle_indices
                    ),
                    "ambiguous_triangle_count": len(
                        numpy_visibility.geometry.ambiguous_triangle_indices
                    ),
                },
                "geometry_oracle_cross_checks": {
                    "mitsuba_cross_check_state": mitsuba_visibility.state.value,
                    "mitsuba_segment_length_m": mitsuba_visibility.segment_length_m,
                    "mitsuba_first_intersection_distance_m": (
                        mitsuba_visibility.first_intersection_distance_m
                    ),
                    "mitsuba_incidence_cosine": mitsuba_visibility.incidence_cosine,
                    "mitsuba_boundary_probe_blocked": list(
                        mitsuba_visibility.boundary_probe_blocked
                    ),
                    "depth_zero_pathsolver_state": depth_zero_state,
                    "oracle_agreement": {
                        "numpy_mitsuba": True,
                        "numpy_depth_zero_pathsolver": True,
                    },
                },
                "tx": list(cfg.tx_position),
                "transmitter_id": cfg.transmitter_id,
                "rx": [x, y, z],
                "valid_paths_full": n_paths,
                "candidate_paths_before_validation": candidate_path_count,
                "candidate_paths_per_source_before_validation": list(
                    full_candidate_counter.last_candidate_path_counts_per_source or ()
                ),
                "max_num_paths_per_src": cfg.max_num_paths_per_src,
                "candidate_path_cap_filled": False,
                "valid_paths_direct_oracle": n_direct,
                "valid_paths_direct_component": direct_count,
                "los_plus_indirect_paths": mixed,
                **diagnostics,
                "seed": seed + attempt,
                "snr_db": grid.snr_db,
                "max_depth": cfg.max_depth,
                "mitsuba_variant": cfg.mitsuba_variant,
                "scene_config": asdict(cfg),
                "grid_config": asdict(grid),
                "solver_config": {
                    "deterministic": True,
                    "max_num_paths_per_src": cfg.max_num_paths_per_src,
                    "los": True,
                    "specular_reflection": True,
                    "diffuse_reflection": True,
                    "refraction": True,
                    "diffraction": False,
                },
            }
        )
        if umcd_store_root is not None:
            umcd_bundle_refs.append(
                _publish_static_rt_sample_umcd(
                    store_root=umcd_store_root,
                    scene=scene,
                    full_paths=full,
                    clean_cfr=clean_channel,
                    noisy_cfr=observed_channel,
                    label=label,
                    rx_position=(float(x), float(y), float(z)),
                    split_group_id=group_id,
                    scene_config=cfg,
                    grid=grid,
                    seed=seed + attempt,
                    selection=selection,
                    runtime_fingerprint=runtime_fingerprint,
                    package_sha256=package_sha256,
                    scene_assets=scene_assets,
                    material_manifest=material_manifest,
                    scene_geometry_inventory=scene_geometry_inventory,
                    scene_family_identity=scene_family_identity,
                    authorization=authorization,
                )
            )
        hits[label] += 1
        class_outcomes["accepted"] += 1
    full_solver_outcomes = {
        "empty_paths": attempt_counts["full_solver_empty_paths"],
        "candidate_cap_filled_rejected": attempt_counts["full_solver_candidate_cap_rejections"],
        "los_direct_only_rejected": attempt_counts["los_direct_only_rejections"],
        "cfr_conversion_failures": attempt_counts["cfr_conversion_failures"],
        "accepted": len(samples),
    }
    if (
        attempt_counts["attempts_started"]
        != attempt_counts["obstacle_rejections"] + attempt_counts["geometry_oracle_attempts"]
    ):
        raise RuntimeError("RT generation report does not account for every candidate attempt")
    if attempt_counts["geometry_oracle_attempts"] != (
        attempt_counts["geometry_oracle_calls"] + attempt_counts["geometry_oracle_errors"]
    ):
        raise RuntimeError("RT generation report does not account for every scene-geometry query")
    if attempt_counts["geometry_oracle_calls"] != (
        attempt_counts["geometry_oracle_ambiguous"]
        + attempt_counts["geometry_oracle_coincident"]
        + attempt_counts["mitsuba_geometry_disagreements"]
        + attempt_counts["depth_zero_oracle_attempts"]
    ):
        raise RuntimeError("RT generation report does not account for every geometry result")
    if attempt_counts["depth_zero_oracle_attempts"] != (
        attempt_counts["depth_zero_oracle_calls"] + attempt_counts["depth_zero_oracle_errors"]
    ):
        raise RuntimeError("RT generation report does not account for every LOS-oracle attempt")
    if attempt_counts["depth_zero_oracle_calls"] != (
        attempt_counts["candidate_geometric_los"]
        + attempt_counts["candidate_geometric_nlos"]
        + attempt_counts["depth_zero_geometry_disagreements"]
    ):
        raise RuntimeError("RT generation report does not account for every LOS-oracle result")
    if attempt_counts["full_solver_attempts"] != (
        attempt_counts["full_solver_calls"] + attempt_counts["full_solver_errors"]
    ):
        raise RuntimeError("RT generation report does not account for every full-solver attempt")
    if (
        attempt_counts["full_solver_candidate_cap_filled"]
        != (attempt_counts["full_solver_candidate_cap_rejections"])
    ):
        raise RuntimeError("RT generation report does not account for every cap-filled solve")
    if sum(full_solver_outcomes.values()) != attempt_counts["full_solver_calls"]:
        raise RuntimeError("RT generation report does not account for every full-solver outcome")
    if (
        sum(item["attempts"] for item in full_solver_outcomes_by_geometric_class.values())
        != (attempt_counts["full_solver_attempts"])
    ):
        raise RuntimeError("RT class report does not account for every full-solver attempt")
    if (
        sum(item["errors"] for item in full_solver_outcomes_by_geometric_class.values())
        != (attempt_counts["full_solver_errors"])
    ):
        raise RuntimeError("RT class report does not account for every full-solver error")
    if (
        sum(item["calls"] for item in full_solver_outcomes_by_geometric_class.values())
        != (attempt_counts["full_solver_calls"])
    ):
        raise RuntimeError("RT class report does not account for every full-solver call")
    for class_name, item in full_solver_outcomes_by_geometric_class.items():
        terminal_outcomes = (
            item["empty_paths"]
            + item["candidate_cap_filled_rejected"]
            + item["los_direct_only_rejected"]
            + item["cfr_conversion_failures"]
            + item["accepted"]
        )
        expected_attempts = item["calls"] + item["errors"]
        if item["attempts"] != expected_attempts or terminal_outcomes != item["calls"]:
            raise RuntimeError(f"RT class outcomes are incomplete for {class_name}")
    if sum(item["accepted"] for item in full_solver_outcomes_by_geometric_class.values()) != len(
        samples
    ):
        raise RuntimeError("RT class report does not match accepted sample count")
    generation_report = {
        "schema": "loslab.raytraced_generation_report.v4",
        "label_semantics": "geometric_direct_path_oracle",
        "label_oracle": {
            "primary": "numpy_loaded_scene_triangle_inventory",
            "cross_checks": [
                "mitsuba_finite_segment_visibility_query",
                "depth_zero_los_only_path_solver",
            ],
            "policy": "skip_ambiguous_or_disagreeing_candidates",
            "grazing_cosine_threshold": 1e-3,
            "boundary_probe_offset_m": 1e-3,
        },
        "scene_geometry": {
            "scene_geometry_sha256": scene_geometry_inventory.scene_geometry_sha256,
            "scene_object_count": len(scene_geometry_inventory.object_names),
            "scene_triangle_count": len(scene_geometry_inventory.triangles_m),
            "primary_oracle": "numpy_loaded_scene_triangle_inventory",
        },
        "channel_semantics": {
            "schema": "loslab.rt_absolute_channel.v1",
            "delay_reference": "absolute_sionna_path_delay_s",
            "normalize_delays": False,
            "normalize_channel_power": False,
            "frequency_reference": "baseband_subcarrier_offsets_hz",
            "phase_convention": "sionna_paths_cfr_baseband_equivalent",
            "component_partition": "same_paths_valid_zero_interaction_mask",
            "time_steps": 1,
        },
        "status": "solver_error" if solver_failure else "completed",
        "environment": versions(),
        "rt_backend_selection": selection.as_dict(),
        "package_source_sha256": package_source_sha256(),
        "scene_config": asdict(cfg),
        "scene_asset_manifest": scene_assets,
        "grid_config": asdict(grid),
        "solver_config": solver_config,
        "candidate_path_audit": {
            "counter_stage": "after_candidate_generation_before_invalid_path_filter",
            "counter_scope": "per_transmitter_source_from_candidate_buffer_indices",
            "saturated_samples_are_rejected": True,
        },
        "seed": seed,
        "requested_per_class": cfg.per_class,
        "attempt_counts": attempt_counts,
        "full_solver_outcomes": full_solver_outcomes,
        "full_solver_outcomes_by_geometric_class": full_solver_outcomes_by_geometric_class,
        "solver_failure": solver_failure,
        "accepted_counts": {
            "total": len(ys),
            "geometric_nlos": hits[0],
            "geometric_los": hits[1],
        },
        "selection_note": (
            "Balanced accepted records are conditional on the recorded scene, candidate sampler, "
            "solver, class quotas, and LOS-multipath filter. Candidate-cap-filled samples are "
            "rejected before data export. Empty full-solver path outcomes are counted but "
            "not exported as channel observations."
        ),
        "umcd_bundle_refs": [dict(item) for item in umcd_bundle_refs],
    }
    if scene_family_identity is not None:
        generation_report["scene_family_candidate"] = dict(scene_family_identity)
    generation_report["rt_run_identity"] = rt_run_identity(
        scene_asset_manifest=scene_assets,
        physical_config={
            "scene": asdict(cfg),
            "grid": asdict(grid),
            "antenna_config": sionna_rt_siso_antenna_config(orientation_policy="fixed_default"),
            "scene_geometry": {
                "scene_geometry_sha256": scene_geometry_inventory.scene_geometry_sha256,
                "scene_object_count": len(scene_geometry_inventory.object_names),
                "scene_triangle_count": len(scene_geometry_inventory.triangles_m),
            },
            **_scene_family_run_binding(scene_family_identity),
        },
        material_config=material_manifest,
        solver_config=solver_config,
        seed=seed,
        backend_selection=selection.as_dict(),
        runtime_versions=runtime_fingerprint,
        package_source_sha256=package_sha256,
    )
    if solver_failure is not None:
        _attach_rt_execution_provenance(
            generation_report,
            started_at_utc=execution_started_at,
            command_argv=command_argv,
            python_executable=python_executable,
            seed=seed,
        )
        raise RaytraceGenerationError(
            f"Sionna RT failed during {solver_failure['stage']} at candidate "
            f"{solver_failure['attempt_index']}: {solver_failure['message']}",
            generation_report,
        )
    if min(hits.values()) < cfg.per_class:
        generation_report["status"] = "insufficient_samples"
        _attach_rt_execution_provenance(
            generation_report,
            started_at_utc=execution_started_at,
            command_argv=command_argv,
            python_executable=python_executable,
            seed=seed,
        )
        raise RaytraceGenerationError(
            f"Insufficient visible paths after {cfg.max_attempts} attempts: "
            + f"LOS={hits[1]}, NLOS={hits[0]}. Change scene/positions/solver budgets.",
            generation_report,
        )
    _attach_rt_execution_provenance(
        generation_report,
        started_at_utc=execution_started_at,
        command_argv=command_argv,
        python_executable=python_executable,
        seed=seed,
    )
    return ChannelDataset(
        np.stack(samples),
        np.asarray(ys),
        np.asarray(groups),
        np.asarray(domains),
        np.asarray(sources),
        tuple(meta),
        generation_report,
        label_semantics="geometric_direct_path_oracle",
    )


def generate_raytraced_cached(
    scene_config: RayScene = RayScene(),
    *,
    cache_directory: str | Path,
    grid: GridConfig = GridConfig(),
    seed: int = 42,
    authorization: D1GenerationAuthorization | None = None,
    pilot: bool = False,
    approval_manifest: str | Path | None = None,
    expected_approval_sha256: str | None = None,
    output_destination: str | Path | None = None,
    failure_report_destination: str | Path | None = None,
    umcd_store_root: str | Path | None = None,
    scene_family_candidate_id: str | None = None,
) -> ChannelDataset:
    """Return a verified cached RT dataset or generate and publish it immutably."""
    if authorization is None and not pilot and approval_manifest is None:
        raise PermissionError("cached RT dataset access requires D1 authorization")
    from integrations.sionna.backend import activate_rt_backend, select_rt_backend
    from integrations.sionna.rt_cache import load_rt_dataset, store_rt_dataset
    from integrations.sionna.scene_assets import (
        scene_asset_manifest,
        sionna_scene_material_manifest,
    )

    selection = select_rt_backend(scene_config.mitsuba_variant)
    activate_rt_backend(selection)
    try:
        import sionna.rt as rt
    except ImportError as exc:
        raise RuntimeError("Sionna RT 2.2 not installed; install sionna==2.2.0") from exc
    cfg = replace(scene_config, mitsuba_variant=selection.variant)
    scene_ref = getattr(rt.scene, cfg.name, cfg.name)
    scene_xml_path = Path(str(scene_ref)).expanduser()
    if scene_xml_path.is_file():
        assets = scene_asset_manifest(scene_xml_path)
    else:
        reference = str(scene_ref)
        assets = {
            "scene_xml_path": reference,
            "asset_count": 0,
            "assets": [],
            "asset_set_sha256": hashlib.sha256(reference.encode()).hexdigest(),
            "unresolved_references": [reference],
            "complete": False,
        }
    scene_family_identity: Mapping[str, Any] | None = None
    if scene_family_candidate_id is not None:
        from datasets.scene_families import (
            load_scene_family_registry,
            scene_candidate_split_metadata,
        )

        geometry_asset_sha256 = assets.get("asset_set_sha256")
        if not isinstance(geometry_asset_sha256, str):
            raise ValueError("scene asset set digest must be a string")
        scene_family_identity = scene_candidate_split_metadata(
            load_scene_family_registry(),
            candidate_id=scene_family_candidate_id,
            actual_geometry_asset_sha256=geometry_asset_sha256,
        )
    scene = rt.load_scene(scene_ref)
    scene.frequency = grid.carrier_hz
    scene.bandwidth = grid.n_subcarriers * grid.spacing_hz
    material_manifest = sionna_scene_material_manifest(scene)
    scene_geometry_inventory = extract_scene_triangle_inventory(scene)
    from integrations.sionna.d1_generation_gate import (
        authorize_d1_generation,
        require_d1_authorization,
    )

    destinations = [cache_directory]
    destinations.extend(
        value
        for value in (output_destination, umcd_store_root, failure_report_destination)
        if value is not None
    )
    request = build_rt_data_generation_request(
        cfg,
        grid,
        seed,
        selection=selection,
        scene_ref=scene_ref,
        scene_assets=assets,
        material_manifest=material_manifest,
        scene_geometry_inventory=scene_geometry_inventory,
        destinations=destinations,
    )
    if authorization is None:
        authorization = authorize_d1_generation(
            request,
            pilot=pilot,
            approval_manifest=approval_manifest,
            expected_approval_sha256=expected_approval_sha256,
        )
    require_d1_authorization(authorization, request, expected_operation="rt_data")  # type: ignore[arg-type]
    if getattr(authorization, "mode", None) == "pilot":
        from integrations.sionna.d1_generation_gate import require_pilot_destination

        require_pilot_destination(authorization, cache_directory, publisher="rt_cache")  # type: ignore[arg-type]
        if output_destination is not None:
            require_pilot_destination(authorization, output_destination, publisher="rt_dataset")  # type: ignore[arg-type]
        if umcd_store_root is not None:
            require_pilot_destination(authorization, umcd_store_root, publisher="umcd_bundle")  # type: ignore[arg-type]
        if failure_report_destination is not None:
            require_pilot_destination(
                authorization, failure_report_destination, publisher="failure_report"
            )  # type: ignore[arg-type]
    package_hash = package_source_sha256()
    runtime_fingerprint = rt_runtime_fingerprint(
        {key: value for key, value in versions().items() if key != "git_sha"}
    )
    identity = rt_run_identity(
        scene_asset_manifest=assets,
        physical_config={
            "scene": asdict(cfg),
            "grid": asdict(grid),
            "antenna_config": sionna_rt_siso_antenna_config(orientation_policy="fixed_default"),
            "scene_geometry": {
                "scene_geometry_sha256": scene_geometry_inventory.scene_geometry_sha256,
                "scene_object_count": len(scene_geometry_inventory.object_names),
                "scene_triangle_count": len(scene_geometry_inventory.triangles_m),
            },
            **_scene_family_run_binding(scene_family_identity),
        },
        material_config=material_manifest,
        solver_config=static_rt_solver_config(cfg.max_depth, cfg.max_num_paths_per_src),
        seed=seed,
        backend_selection=selection.as_dict(),
        runtime_versions=runtime_fingerprint,
        package_source_sha256=package_hash,
    )
    if identity["complete"]:
        cached = load_rt_dataset(cache_directory, identity)
        if cached is not None:
            return cached

    dataset = generate_raytraced(
        cfg,
        grid=grid,
        seed=seed,
        _backend_selection=selection,
        _runtime_fingerprint=runtime_fingerprint,
        authorization=authorization,
        _scene_assets_override=assets,
        _scene_override=scene,
        _scene_geometry_inventory_override=scene_geometry_inventory,
        _material_manifest_override=material_manifest,
        output_destination=output_destination,
        failure_report_destination=failure_report_destination,
        umcd_store_root=umcd_store_root,
        scene_family_candidate_id=scene_family_candidate_id,
        _authorization_destinations=(cache_directory,),
    )
    if identity["complete"]:
        store_rt_dataset(cache_directory, dataset, identity)
    return dataset
