"""Sionna 2.2 adapters, imported lazily so CI runs without RT/CUDA.

Important: no fabricated LOS=one-path assumption. A physical LOS link may contain
one direct ray plus any number of reflected, diffuse and refracted rays.
LOS labels use an independent scene-segment query cross-checked against a
depth-zero geometric PathSolver invocation.
"""

from __future__ import annotations

import hashlib
import json
from copy import copy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, cast

import numpy as np
from artifacts.provenance import package_source_sha256, versions
from datasets.channel import ChannelDataset
from datasets.channel_view import (
    ChannelDelayReference,
    ChannelPowerScale,
    PhysicalChannelView,
)
from datasets.network import LinkDirection, LinkRole, NodeRole
from datasets.observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)
from propagation.physics import add_estimation_noise, frequency_response, subcarrier_frequencies

from integrations.sionna.candidate_audit import instrument_path_solver_candidate_counter
from integrations.sionna.los_oracle import SegmentVisibilityState, segment_visibility
from integrations.sionna.random import call_with_sionna_seed
from integrations.sionna.rt_identity import (
    rt_run_identity,
    rt_runtime_fingerprint,
    static_rt_solver_config,
)

if TYPE_CHECKING:
    from integrations.sionna.backend import RTBackendSelection


class RTPaths(Protocol):
    def cfr(self, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class GridConfig:
    n_subcarriers: int = 128
    spacing_hz: float = 30e3
    carrier_hz: float = 3.5e9
    snr_db: float = 15.0

    def __post_init__(self):
        if self.n_subcarriers < 8 or self.spacing_hz <= 0 or self.carrier_hz <= 0:
            raise ValueError("Invalid grid")


def _phy_cir_to_cfr(a: np.ndarray, tau: np.ndarray, f: np.ndarray) -> np.ndarray:
    """3GPP Sionna PHY: a=[B,rx,ra,tx,ta,path,time], tau=[B,rx,tx,path].

    Fix first rx/tx/time but retain all receive/tx antenna-pair channels. No
    mutation of path amplitudes, K factor or LOS flag. Returns [B,links,F].
    """
    a = np.asarray(a)
    tau = np.asarray(tau)
    if a.ndim != 7 or tau.ndim != 4 or a.shape[0] != tau.shape[0] or a.shape[-2] != tau.shape[-1]:
        raise ValueError(f"Unexpected Sionna PHY CIR shapes a={a.shape}, tau={tau.shape}")
    aa = a[:, 0, :, 0, :, :, 0]
    aa = aa.reshape(len(aa), -1, aa.shape[-1])
    tt = np.broadcast_to(tau[:, 0, 0, None, :], (len(aa), aa.shape[1], aa.shape[2]))
    return frequency_response(aa, tt, f).astype(np.complex64)


def generate_standardized(
    *,
    grid: GridConfig = GridConfig(),
    profiles: Sequence[str] = ("A", "B", "C", "D", "E"),
    families: Sequence[str] = ("tdl", "cdl"),
    per_profile: int = 40,
    seed: int = 42,
    delay_spreads_ns: Sequence[float] = (30, 100, 300, 1000),
    device: str = "cpu",
    spec_version: str = "19.2",
) -> ChannelDataset:
    """Generate standardized stochastic *profile-proxy* labels, not geometric LOS.

    A/B/C -> 0, D/E -> 1 is a profile-inherent label. These data alone cannot
    prove detection of a physically unoccluded direct path. Train on them,
    evaluate zero-shot on ray-traced geometry and, eventually, measured signals.
    """
    try:
        import torch
        from sionna.phy.channel.tr38901 import CDL, TDL, PanelArray
    except ImportError as e:
        raise RuntimeError("Install sionna==2.2.0 and its PHY dependencies for this adapter") from e
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(per_profile, bool) or not isinstance(per_profile, int) or per_profile < 1:
        raise ValueError("per_profile must be a positive integer")
    if not delay_spreads_ns or any(
        not np.isfinite(spread) or spread <= 0 for spread in delay_spreads_ns
    ):
        raise ValueError("delay_spreads_ns must contain finite positive values")
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    rng = np.random.default_rng(seed)
    frequency = subcarrier_frequencies(grid.n_subcarriers, grid.spacing_hz)
    h_list = []
    ys = []
    groups = []
    domains = []
    sources = []
    meta = []
    for family_index, family in enumerate(families):
        if family.lower() not in ("tdl", "cdl"):
            raise ValueError(f"Unknown family {family}")
        for profile_index, profile in enumerate(profiles):
            p = profile.upper()
            if p not in ("A", "B", "C", "D", "E"):
                raise ValueError(f"Invalid profile {profile}")
            # Ensure each profile sees identical distribution of delay spreads.
            indices = np.arange(per_profile)
            rng.shuffle(indices)
            spread_index = np.arange(per_profile) % len(delay_spreads_ns)
            for j, sp in enumerate(delay_spreads_ns):
                count = int(np.count_nonzero(spread_index == j))
                if not count:
                    continue
                common = dict(
                    model=p,
                    delay_spread=float(sp) * 1e-9,
                    carrier_frequency=grid.carrier_hz,
                    min_speed=0.0,
                    max_speed=3.0,
                    device=device,
                    spec_version=spec_version,
                )
                if family.lower() == "tdl":
                    cm = TDL(num_rx_ant=1, num_tx_ant=1, **common)
                else:
                    ut = PanelArray(
                        num_rows_per_panel=1,
                        num_cols_per_panel=1,
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="omni",
                        carrier_frequency=grid.carrier_hz,
                        device=device,
                    )
                    bs = PanelArray(
                        num_rows_per_panel=1,
                        num_cols_per_panel=1,
                        polarization="single",
                        polarization_type="V",
                        antenna_pattern="omni",
                        carrier_frequency=grid.carrier_hz,
                        device=device,
                    )
                    cm = CDL(ut_array=ut, bs_array=bs, direction="downlink", **common)
                channel_seed = int(
                    np.random.SeedSequence([seed, family_index, profile_index, j]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
                with torch.no_grad():
                    a, tau = call_with_sionna_seed(
                        device,
                        channel_seed,
                        lambda: cm(
                            batch_size=count,
                            num_time_steps=1,
                            sampling_frequency=grid.spacing_hz,
                        ),
                    )
                data = _phy_cir_to_cfr(
                    a.detach().cpu().numpy(), tau.detach().cpu().numpy(), frequency
                )
                data = add_estimation_noise(data, grid.snr_db, rng)
                for k, ix in enumerate(indices[spread_index == j]):
                    h_list.append(data[k])
                    ys.append(int(p in "DE"))
                    # Each sample is one link; follow-up correlated snapshots must reuse its group.
                    groups.append(f"{family}_{p}_s{seed}_link{int(ix)}")
                    domains.append(f"{family}")
                    sources.append(f"{family}")
                    meta.append(
                        {
                            "profile": p,
                            "family": family,
                            "spread_ns": float(sp),
                            "label_semantics": "PROFILE_PROXY",
                            "grid_carrier_hz": grid.carrier_hz,
                            "snr_db": grid.snr_db,
                            "spec_version": spec_version,
                            "generation_seed": seed,
                            "channel_seed": channel_seed,
                        }
                    )
    return ChannelDataset(
        np.stack(h_list),
        np.asarray(ys),
        np.asarray(groups),
        np.asarray(domains),
        np.asarray(sources),
        tuple(meta),
        label_semantics="profile_proxy",
    )


def has_valid_path(paths: object) -> bool:
    """Ground-truth geometric visibility from depth-0 solver, NOT CFR power."""
    valid = getattr(paths, "valid", None)
    if valid is None:
        raise ValueError("Sionna path object missing valid field")
    arr = np.asarray(valid.numpy() if hasattr(valid, "numpy") else valid)
    return bool(np.any(arr))


def count_valid_paths(paths: object) -> int:
    arr = getattr(paths, "valid", None)
    if arr is None:
        raise ValueError("Sionna path object missing valid field")
    a = np.asarray(arr.numpy() if hasattr(arr, "numpy") else arr)
    return int(np.count_nonzero(a))


def direct_path_mask(paths: object) -> np.ndarray:
    """Select valid zero-interaction paths from one Sionna 2.2 Paths object."""
    valid = getattr(paths, "valid", None)
    interactions = getattr(paths, "interactions", None)
    if valid is None or interactions is None:
        raise ValueError("Sionna Paths must expose valid and interactions fields")
    valid_array = np.asarray(valid.numpy() if hasattr(valid, "numpy") else valid, dtype=bool)
    interaction_array = np.asarray(
        interactions.numpy() if hasattr(interactions, "numpy") else interactions
    )
    if interaction_array.ndim != valid_array.ndim + 1:
        raise ValueError(
            "Sionna interactions must have one leading depth axis: "
            f"valid={valid_array.shape}, interactions={interaction_array.shape}"
        )
    if interaction_array.shape[1:] != valid_array.shape:
        raise ValueError(
            "Sionna valid/interactions path axes differ: "
            f"valid={valid_array.shape}, interactions={interaction_array.shape}"
        )
    return valid_array & np.all(interaction_array == 0, axis=0)


def _mask_path_coefficients(paths: RTPaths, path_mask: np.ndarray) -> RTPaths:
    """Copy pinned Sionna Paths and zero coefficients outside a validated mask."""
    import drjit as dr
    import mitsuba as mi

    valid = getattr(paths, "valid", None)
    coefficients = getattr(paths, "a", None)
    if valid is None or coefficients is None or len(coefficients) != 2:
        raise ValueError("Path subsetting requires Sionna 2.2 valid and complex coefficient fields")
    valid_array = np.asarray(valid.numpy() if hasattr(valid, "numpy") else valid, dtype=bool)
    mask = np.asarray(path_mask, dtype=bool)
    if mask.shape != valid_array.shape:
        raise ValueError(
            f"Path mask shape {mask.shape} does not match valid shape {valid_array.shape}"
        )
    if np.any(mask & ~valid_array):
        raise ValueError("Path mask selects invalid or padded Sionna paths")
    coefficient_shape = tuple(coefficients[0].shape)
    if mask.shape != coefficient_shape:
        if bool(getattr(paths, "synthetic_array", False)) and len(coefficient_shape) == 5:
            mask = mask[:, None, :, None, :]
            try:
                mask = np.broadcast_to(mask, coefficient_shape)
            except ValueError as error:
                raise ValueError(
                    "Path mask axes do not match coefficient axes: "
                    f"mask={mask.shape}, coefficients={coefficient_shape}"
                ) from error
        else:
            raise ValueError(
                "Path mask axes do not match coefficient axes: "
                f"mask={mask.shape}, coefficients={coefficient_shape}"
            )

    # Dr.Jit cannot broadcast a scalar over a zero-length tensor. An empty
    # Paths object already has no valid coefficients to remove, so copying it
    # is exactly the requested empty subset.
    if mask.size == 0:
        return copy(paths)

    masked: Any = copy(paths)
    selector = cast(Any, mi.TensorXb)(mask.tolist())
    if not hasattr(masked, "_a_real") or not hasattr(masked, "_a_imag"):
        raise ValueError("Unsupported Sionna Paths implementation: coefficient storage is missing")
    masked._a_real = dr.select(selector, masked._a_real, 0.0)
    masked._a_imag = dr.select(selector, masked._a_imag, 0.0)
    if hasattr(masked, "_valid"):
        masked._valid = selector
    return masked


def rt_cfr(paths: RTPaths, grid: GridConfig, *, path_mask: np.ndarray | None = None) -> np.ndarray:
    """Convert Sionna RT Paths to an absolute-delay [1, links, subcarriers] CFR.

    Sionna 2.2 Paths.cfr returns a complex ndarray for out_type='numpy'
    (some earlier interfaces returned a real/imag tuple).
    Preserve all antennas, absolute path delays, and carrier phase; use one
    frequency/time snapshot. Normalizing delays here would erase absolute ToA
    and make independently solved full/direct channels use different phase
    references, invalidating their coherent residual.
    """
    import drjit as dr
    import mitsuba as mi

    if path_mask is not None:
        paths = _mask_path_coefficients(paths, path_mask)

    f = (dr.arange(mi.Float, grid.n_subcarriers) - grid.n_subcarriers // 2) * grid.spacing_hz
    parts = paths.cfr(
        frequencies=f,
        num_time_steps=1,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    )
    if isinstance(parts, (tuple, list)) and len(parts) == 2:
        r, im = (np.asarray(x) for x in parts)
        if r.shape != im.shape:
            raise ValueError(f"RT CFR real/imag dimension mismatch {r.shape} / {im.shape}")
        arr = r + 1j * im
    else:
        arr = np.asarray(parts)
    if not np.iscomplexobj(arr) or arr.shape[-1] != grid.n_subcarriers:
        raise ValueError(f"Unexpected Sionna RT complex CFR shape/type {arr.shape} / {arr.dtype}")
    out = arr.reshape(1, -1, grid.n_subcarriers)
    if not np.all(np.isfinite(out)):
        raise ValueError("Nonfinite RT CFR")
    return out.astype(np.complex64)


def direct_indirect_diagnostics(full_cfr: np.ndarray, direct_cfr: np.ndarray) -> dict:
    """Diagnostic-only oracle power decomposition; never feed to classifier.

    Build the direct CFR from the full solve with
    ``rt_cfr(paths, grid, path_mask=direct_path_mask(paths))``; the residual
    then shares the exact path set, delay reference, and antenna coordinates.
    """
    full = np.asarray(full_cfr)
    direct = np.asarray(direct_cfr)
    if full.shape != direct.shape or not np.iscomplexobj(full) or not np.iscomplexobj(direct):
        raise ValueError("Expected aligned complex CFR tensors")
    p_direct = float(np.mean(np.abs(direct) ** 2))
    p_indirect = float(np.mean(np.abs(full - direct) ** 2))
    ratio = float(10 * np.log10(max(p_direct, 1e-24) / max(p_indirect, 1e-24)))
    return {
        "oracle_direct_power": p_direct,
        "residual_indirect_power": p_indirect,
        "direct_to_indirect_power_db": ratio,
    }


@dataclass(frozen=True)
class RayScene:
    name: str = "munich"
    tx_position: tuple[float, float, float] = (8.5, 21.0, 27.0)
    rx_center: tuple[float, float, float] = (44.0, 95.0, 1.5)
    radius_m: float = 90.0
    block_size_m: float = 30.0
    max_depth: int = 3
    samples_per_src: int = 120_000
    max_num_paths_per_src: int = 1_000_000
    mitsuba_variant: str = "auto"
    per_class: int = 16
    max_attempts: int = 2000
    require_los_multipath: bool = True
    transmitter_id: str | None = None
    # built-in 'munich', 'etoile', or a valid user-supplied scene path

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_num_paths_per_src, bool)
            or not isinstance(self.max_num_paths_per_src, int)
            or self.max_num_paths_per_src < 1
        ):
            raise ValueError("max_num_paths_per_src must be a positive integer")


class RaytraceGenerationError(RuntimeError):
    """RT generation failure that preserves the complete attempt report."""

    def __init__(self, message: str, generation_report: Mapping[str, object]) -> None:
        super().__init__(message)
        self.generation_report = dict(generation_report)


def _publish_static_rt_sample_umcd(
    *,
    store_root: str | Path,
    scene: object,
    full_paths: object,
    clean_cfr: np.ndarray,
    noisy_cfr: np.ndarray,
    label: int,
    rx_position: tuple[float, float, float],
    split_group_id: str,
    scene_config: RayScene,
    grid: GridConfig,
    seed: int,
    selection: RTBackendSelection,
    runtime_fingerprint: Mapping[str, str],
    package_sha256: str,
    scene_assets: Mapping[str, Any],
    material_manifest: Mapping[str, Any],
) -> Mapping[str, object]:
    """Publish one static RT path solve and its direct noisy-CFR observation."""
    from datasets.targets import TargetDefinition, TargetValidity, TargetValue, definition_digest

    from integrations.sionna.array_layouts import sionna_scene_to_network_episode
    from integrations.sionna.rt_network import SionnaPathBinding, write_sionna_rt_episode_bundle

    offsets = tuple(
        float(value)
        for value in (np.arange(grid.n_subcarriers, dtype=np.float64) - grid.n_subcarriers // 2)
        * grid.spacing_hz
    )
    identity = rt_run_identity(
        scene_asset_manifest=scene_assets,
        physical_config={
            "scene": asdict(scene_config),
            "grid": asdict(grid),
            "rx_position_m": list(rx_position),
            "material_state": dict(material_manifest),
        },
        solver_config=static_rt_solver_config(
            scene_config.max_depth, scene_config.max_num_paths_per_src
        ),
        seed=seed,
        backend_selection=selection.as_dict(),
        runtime_versions=runtime_fingerprint,
        package_source_sha256=package_sha256,
    )
    if not identity["complete"]:
        raise ValueError("UMCD static RT publication requires a complete run identity")
    sample_id = f"rt-{identity['cache_key_sha256']}"
    network = sionna_scene_to_network_episode(
        scene,
        episode_id=sample_id,
        split_group_id=split_group_id,
        family_id="F03-RT-STATIC",
        subcarrier_offsets_hz=offsets,
        node_roles={"tx:research_tx": NodeRole.GNB, "rx:research_rx": NodeRole.UE},
        associations=(("tx:research_tx", "rx:research_rx"),),
        direction=LinkDirection.DOWNLINK,
        link_role=LinkRole.SERVING,
    )
    episode = network.episode
    link = episode.links[0]
    path_binding = SionnaPathBinding(
        link_id=link.link_id,
        frame_index=0,
        rx_device_index=0,
        tx_device_index=0,
        paths=full_paths,
    )
    clean = np.asarray(clean_cfr).reshape(1, 1, 1, grid.n_subcarriers)
    noisy = np.asarray(noisy_cfr).reshape(1, 1, grid.n_subcarriers)
    noise_variance = float(
        max(float(np.mean(np.abs(clean_cfr) ** 2)), 1e-24) * 10.0 ** (-grid.snr_db / 10.0)
    )
    channel_view = PhysicalChannelView(
        view_id=f"{sample_id}:absolute-cfr",
        link_id=link.link_id,
        channel=clean,
        frame_times_s=(0.0,),
        rx_port_ids=tuple(
            episode.array(episode.node(link.rx_node_id).array_id).port_ids[index]
            for index in link.rx_port_indices
        ),
        tx_port_ids=tuple(
            episode.array(episode.node(link.tx_node_id).array_id).port_ids[index]
            for index in link.tx_port_indices
        ),
        carrier_hz=grid.carrier_hz,
        subcarrier_offsets_hz=np.asarray(offsets),
        delay_reference=ChannelDelayReference.ABSOLUTE_PROPAGATION,
        power_scale=ChannelPowerScale.ABSOLUTE,
    )
    receiver_observation = LinkedFrequencyDomainObservation(
        link.link_id,
        0,
        FrequencyDomainObservation(
            sample_id,
            FrequencyDomainModelInputs(
                observed_cfr=noisy,
                noise_variance=noise_variance,
                snr_db=grid.snr_db,
                carrier_hz=grid.carrier_hz,
                subcarrier_offsets_hz=np.asarray(offsets),
            ),
        ),
    )
    label_definition = TargetDefinition(
        semantic="geometric direct-path visibility",
        units="boolean",
        source="independent_segment_visibility_oracle",
        reference="finite scene-segment visibility cross-checked by depth-zero PathSolver",
        definition_sha256=definition_digest(
            "UMCD:F03-RT-STATIC:geometric_direct_path_visibility:v1"
        ),
    )
    target = TargetValue(
        entity_id=link.link_id,
        definition=label_definition,
        values=np.asarray([label], dtype=np.uint8),
        valid_mask=np.asarray([True]),
        reason_codes=(TargetValidity.VALID,),
        confidence="independent_geometry_with_solver_crosscheck",
    )
    stored = write_sionna_rt_episode_bundle(
        store_root,
        episode,
        (path_binding,),
        run_identity=identity,
        material_config=material_manifest,
        source_family="F03-RT-STATIC",
        producer="Sionna 2.2 deterministic PathSolver",
        channel_views=(channel_view,),
        targets=(target,),
        receiver_observations=(receiver_observation,),
    )
    return asdict(stored)


def generate_raytraced(
    scene_config: RayScene = RayScene(),
    *,
    grid: GridConfig = GridConfig(),
    seed: int = 42,
    umcd_store_root: str | Path | None = None,
    _backend_selection: RTBackendSelection | None = None,
    _runtime_fingerprint: Mapping[str, str] | None = None,
) -> ChannelDataset:
    """Balanced physical LOS/NLOS ray tracing. Fail closed when insufficient examples.

    Full PathSolver includes direct + reflection (and optional diffuse). A finite
    scene-segment query determines geometric LOS; depth-zero PathSolver results
    cross-check the label, and ambiguous/disagreeing candidates are excluded.
    Report how often LOS really contains >=2 valid paths; default rejects
    direct-only LOS. NLOS must contain at least one observable indirect path.
    """
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
    scene = load_scene(scene_ref)
    scene.frequency = grid.carrier_hz
    scene.bandwidth = grid.n_subcarriers * grid.spacing_hz
    material_manifest = sionna_scene_material_manifest(scene)
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
            visibility = segment_visibility(scene, cfg.tx_position, (x, y, z))
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
        if visibility.state is SegmentVisibilityState.AMBIGUOUS:
            attempt_counts["geometry_oracle_ambiguous"] += 1
            continue
        if visibility.state is SegmentVisibilityState.COINCIDENT_ENDPOINTS:
            attempt_counts["geometry_oracle_coincident"] += 1
            continue
        if visibility.visible is None:
            raise RuntimeError("geometry oracle returned a non-labelable visibility state")
        label = int(visibility.visible)
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
        block_x = int(np.floor(x / cfg.block_size_m))
        block_y = int(np.floor(y / cfg.block_size_m))
        group_id = f"{scene_id}_block_{block_x}_{block_y}"
        domain_id = f"rt_{scene_id}"
        if cfg.transmitter_id is not None:
            group_id = f"{scene_id}_tx_{cfg.transmitter_id}_block_{block_x}_{block_y}"
            domain_id = f"{domain_id}_{cfg.transmitter_id}"
        groups.append(group_id)
        domains.append(domain_id)
        sources.append("sionna_rt")
        meta.append(
            {
                "scene": cfg.name,
                "scene_asset_set_sha256": scene_assets["asset_set_sha256"],
                "scene_asset_manifest_complete": scene_assets["complete"],
                "scene_asset_count": scene_assets["asset_count"],
                "geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE",
                "geometry_oracle": {
                    "state": visibility.state.value,
                    "segment_length_m": visibility.segment_length_m,
                    "first_intersection_distance_m": visibility.first_intersection_distance_m,
                    "incidence_cosine": visibility.incidence_cosine,
                    "boundary_probe_blocked": list(visibility.boundary_probe_blocked),
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
                    split_group_id=str(scene_assets["asset_set_sha256"]),
                    scene_config=cfg,
                    grid=grid,
                    seed=seed + attempt,
                    selection=selection,
                    runtime_fingerprint=runtime_fingerprint,
                    package_sha256=package_sha256,
                    scene_assets=scene_assets,
                    material_manifest=material_manifest,
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
        "schema": "loslab.raytraced_generation_report.v3",
        "label_semantics": "geometric_direct_path_oracle",
        "label_oracle": {
            "primary": "independent_finite_scene_segment_visibility_query",
            "cross_check": "depth_zero_los_only_path_solver",
            "policy": "skip_ambiguous_or_disagreeing_candidates",
            "grazing_cosine_threshold": 1e-3,
            "boundary_probe_offset_m": 1e-3,
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
    generation_report["rt_run_identity"] = rt_run_identity(
        scene_asset_manifest=scene_assets,
        physical_config={
            "scene": asdict(cfg),
            "grid": asdict(grid),
            "material_state": material_manifest,
        },
        solver_config=solver_config,
        seed=seed,
        backend_selection=selection.as_dict(),
        runtime_versions=runtime_fingerprint,
        package_source_sha256=package_sha256,
    )
    if solver_failure is not None:
        raise RaytraceGenerationError(
            f"Sionna RT failed during {solver_failure['stage']} at candidate "
            f"{solver_failure['attempt_index']}: {solver_failure['message']}",
            generation_report,
        )
    if min(hits.values()) < cfg.per_class:
        generation_report["status"] = "insufficient_samples"
        raise RaytraceGenerationError(
            f"Insufficient visible paths after {cfg.max_attempts} attempts: "
            + f"LOS={hits[1]}, NLOS={hits[0]}. Change scene/positions/solver budgets.",
            generation_report,
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
) -> ChannelDataset:
    """Return a verified cached RT dataset or generate and publish it immutably."""
    from integrations.sionna.backend import activate_rt_backend, select_rt_backend
    from integrations.sionna.rt_cache import load_rt_dataset, store_rt_dataset
    from integrations.sionna.scene_assets import scene_asset_manifest

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
    package_hash = package_source_sha256()
    runtime_fingerprint = rt_runtime_fingerprint(
        {key: value for key, value in versions().items() if key != "git_sha"}
    )
    identity = rt_run_identity(
        scene_asset_manifest=assets,
        physical_config={"scene": asdict(cfg), "grid": asdict(grid)},
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
    )
    if identity["complete"]:
        store_rt_dataset(cache_directory, dataset, identity)
    return dataset
