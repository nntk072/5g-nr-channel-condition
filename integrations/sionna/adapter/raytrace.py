"""Sionna ray-tracing scene, path metrics, and publication helpers."""

from __future__ import annotations

import hashlib
import math
from copy import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence, cast

import numpy as np
from artifacts.provenance import canonical_hash, package_source_sha256, versions
from datasets.channel import ChannelDataset
from datasets.network import LinkDirection, LinkRole, NodeRole
from datasets.observations import (
    FrequencyDomainModelInputs,
    FrequencyDomainObservation,
    LinkedFrequencyDomainObservation,
)
from propagation.physics import add_estimation_noise, frequency_response, subcarrier_frequencies
from propagation.scene_visibility import SceneTriangleInventory, extract_scene_triangle_inventory

from integrations.sionna.d1_generation_gate import D1GenerationAuthorization
from integrations.sionna.random import call_with_sionna_seed
from integrations.sionna.rt_identity import (
    rt_run_identity,
    sionna_rt_siso_antenna_config,
    static_rt_solver_config,
)

_RT_EXECUTION_PROVENANCE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3] / "datasets/schema/rt-execution-provenance.v1.schema.json"
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
                    group_id = f"{family}_{p}_s{seed}_link{int(ix)}"
                    groups.append(group_id)
                    domains.append(f"{family}")
                    sources.append(f"{family}")
                    meta.append(
                        {
                            "profile": p,
                            "family": family,
                            "split_subject_id": group_id,
                            "split_parent_type": "standardized_channel_draw",
                            "split_source_family": family,
                            "split_leakage_relations": [],
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
    if not (
        np.isfinite(full.real).all()
        and np.isfinite(full.imag).all()
        and np.isfinite(direct.real).all()
        and np.isfinite(direct.imag).all()
    ):
        raise ValueError("Aligned complex CFR tensors must contain only finite values")
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
        if (
            isinstance(self.block_size_m, bool)
            or not isinstance(self.block_size_m, (int, float))
            or not math.isfinite(self.block_size_m)
            or self.block_size_m <= 0
        ):
            raise ValueError("block_size_m must be finite and positive")


def build_rt_data_generation_request(
    scene_config: RayScene,
    grid: GridConfig,
    seed: int,
    *,
    selection: Any,
    scene_ref: object,
    scene_assets: Mapping[str, Any],
    material_manifest: Mapping[str, Any],
    scene_geometry_inventory: SceneTriangleInventory,
    destinations: Sequence[str | Path] = (),
) -> dict[str, Any]:
    """Build the canonical request from effective runtime and resolved scene facts."""
    return {
        "operation": "rt_data",
        "scene": scene_config.name,
        "seed": seed,
        "config": {
            **asdict(scene_config),
            "total_work_units": scene_config.max_attempts * scene_config.samples_per_src,
            "antenna_config": sionna_rt_siso_antenna_config(orientation_policy="fixed_default"),
            "solver_config": static_rt_solver_config(
                scene_config.max_depth, scene_config.max_num_paths_per_src
            ),
        },
        "grid": asdict(grid),
        "package_source_sha256": package_source_sha256(),
        "backend": selection.as_dict(),
        "scene_asset_identity": {
            "scene_reference": str(scene_ref),
            "asset_set_sha256": scene_assets["asset_set_sha256"],
            "asset_count": scene_assets["asset_count"],
            "complete": scene_assets["complete"],
            "runtime_versions": versions(),
        },
        "material_state_sha256": canonical_hash(dict(material_manifest)),
        "scene_geometry": {
            "primary_oracle": "numpy_loaded_scene_triangle_inventory",
            "scene_geometry_sha256": scene_geometry_inventory.scene_geometry_sha256,
            "scene_object_count": len(scene_geometry_inventory.object_names),
            "scene_triangle_count": len(scene_geometry_inventory.triangles_m),
        },
        "destinations": sorted(str(Path(value).expanduser().resolve()) for value in destinations),
    }


def authorize_rt_data_generation(
    scene_config: RayScene,
    *,
    grid: GridConfig = GridConfig(),
    seed: int = 42,
    destinations: Sequence[str | Path] = (),
    pilot: bool = False,
    approval_manifest: str | Path | None = None,
    expected_approval_sha256: str | None = None,
) -> D1GenerationAuthorization:
    """Resolve the effective RT inputs and authorize their canonical request."""
    from integrations.sionna.backend import activate_rt_backend, select_rt_backend
    from integrations.sionna.d1_generation_gate import authorize_d1_generation
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
    effective = replace(scene_config, mitsuba_variant=selection.variant)
    scene_ref = getattr(rt.scene, effective.name, effective.name)
    scene_path = Path(str(scene_ref)).expanduser()
    if scene_path.is_file():
        assets = scene_asset_manifest(scene_path)
    else:
        ref = str(scene_ref)
        assets = {
            "scene_xml_path": ref,
            "asset_count": 0,
            "asset_set_sha256": hashlib.sha256(ref.encode()).hexdigest(),
            "complete": False,
        }
    scene = rt.load_scene(scene_ref)
    scene.frequency = grid.carrier_hz
    scene.bandwidth = grid.n_subcarriers * grid.spacing_hz
    material_manifest = sionna_scene_material_manifest(scene)
    scene_geometry_inventory = extract_scene_triangle_inventory(scene)
    request = build_rt_data_generation_request(
        effective,
        grid,
        seed,
        selection=selection,
        scene_ref=scene_ref,
        scene_assets=assets,
        material_manifest=material_manifest,
        scene_geometry_inventory=scene_geometry_inventory,
        destinations=destinations,
    )
    return authorize_d1_generation(
        request,
        pilot=pilot,
        approval_manifest=approval_manifest,
        expected_approval_sha256=expected_approval_sha256,
    )


class RaytraceGenerationError(RuntimeError):
    """RT generation failure that preserves the complete attempt report."""

    def __init__(self, message: str, generation_report: Mapping[str, object]) -> None:
        super().__init__(message)
        self.generation_report = dict(generation_report)


def _attach_rt_execution_provenance(
    report: dict[str, Any],
    *,
    started_at_utc: str,
    command_argv: Sequence[str],
    python_executable: str,
    seed: int,
) -> None:
    """Bind invocation and UTC execution bounds to the deterministic RT identity."""
    identity = report.get("rt_run_identity")
    identity_sha256 = (
        identity.get("cache_key_sha256")
        if isinstance(identity, Mapping) and identity.get("complete") is True
        else None
    )
    execution = {
        "schema": "loslab.rt_execution_provenance.v1",
        "schema_sha256": hashlib.sha256(
            _RT_EXECUTION_PROVENANCE_SCHEMA_PATH.read_bytes()
        ).hexdigest(),
        "started_at_utc": started_at_utc,
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "python_executable": python_executable,
        "command_argv": list(command_argv),
        "base_seed": seed,
        "attempt_seed_scheme": "base_seed_plus_zero_based_candidate_attempt_index",
        "candidate_attempt_count": report["attempt_counts"]["attempts_started"],
        "terminal_status": report["status"],
        "failure_stage": (
            None if report.get("solver_failure") is None else report["solver_failure"]["stage"]
        ),
        "rt_cache_key_sha256": identity_sha256,
    }
    report["execution_provenance"] = execution
    report["execution_provenance_sha256"] = canonical_hash(execution)


def _scene_family_run_binding(
    identity: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Return cache-key fields for a verified scene-family candidate, if any."""
    if identity is None:
        return {}
    return {
        "scene_family_candidate": {
            "candidate_id": identity["candidate_id"],
            "family_id": identity["family_id"],
            "registry_sha256": identity["registry_sha256"],
            "geometry_asset_sha256": identity["geometry_asset_sha256"],
        }
    }


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
    scene_geometry_inventory: SceneTriangleInventory,
    scene_family_identity: Mapping[str, Any] | None,
    authorization: D1GenerationAuthorization | None,
) -> Mapping[str, object]:
    """Publish one static RT path solve and its direct noisy-CFR observation."""
    from datasets.targets import (
        TargetDefinition,
        TargetValidity,
        TargetValue,
        target_definition_digest,
    )

    from integrations.sionna.array_layouts import sionna_scene_to_network_episode
    from integrations.sionna.rt_network import (
        SionnaPathBinding,
        sionna_bindings_to_channel_views,
        write_sionna_rt_episode_bundle,
    )

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
            "antenna_config": sionna_rt_siso_antenna_config(orientation_policy="fixed_default"),
            "rx_position_m": list(rx_position),
            "scene_geometry": {
                "scene_geometry_sha256": scene_geometry_inventory.scene_geometry_sha256,
                "scene_object_count": len(scene_geometry_inventory.object_names),
                "scene_triangle_count": len(scene_geometry_inventory.triangles_m),
            },
            **_scene_family_run_binding(scene_family_identity),
        },
        material_config=material_manifest,
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
        split_leakage_relations=(
            ()
            if scene_family_identity is None
            else tuple(
                (str(item["namespace"]), str(item["identity"]))
                for item in scene_family_identity["split_leakage_relations"]
            )
        ),
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
    noisy = np.asarray(noisy_cfr).reshape(1, 1, grid.n_subcarriers)
    noise_variance = float(
        max(float(np.mean(np.abs(clean_cfr) ** 2)), 1e-24) * 10.0 ** (-grid.snr_db / 10.0)
    )
    channel_views = sionna_bindings_to_channel_views(episode, (path_binding,))
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
        source="numpy_loaded_scene_triangle_inventory",
        reference=(
            "digest-bound loaded-scene triangle visibility cross-checked by Mitsuba "
            "segment intersection and depth-zero PathSolver"
        ),
        aggregation_rule="one direct-path visibility label per link; no temporal or port pooling",
        definition_version="umcd.f03_rt_static.direct_visibility.v2",
        definition_sha256=target_definition_digest(
            semantic="geometric direct-path visibility",
            units="boolean",
            source="numpy_loaded_scene_triangle_inventory",
            reference=(
                "digest-bound loaded-scene triangle visibility cross-checked by Mitsuba "
                "segment intersection and depth-zero PathSolver"
            ),
            aggregation_rule="one direct-path visibility label per link; no temporal or port pooling",
            definition_version="umcd.f03_rt_static.direct_visibility.v2",
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
        channel_views=channel_views,
        targets=(target,),
        receiver_observations=(receiver_observation,),
        authorization=authorization,
    )
    return asdict(stored)
