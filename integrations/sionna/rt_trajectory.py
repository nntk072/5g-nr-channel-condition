"""Build timestamped NR receiver observations from retraced Sionna RT paths."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from artifacts.provenance import canonical_hash, package_source_sha256, versions
from datasets.nr import NRChannelRecord, NRObservation, NRTrajectory, TargetDefinition
from datasets.nr_store import (
    NRStoredTrajectory,
    _publish_immutable,
    read_nr_trajectory,
    write_nr_trajectory,
)
from receivers.nr import NRChannelEstimate, NRPUSCHConfig

from integrations.sionna.rt_identity import rt_run_identity, rt_runtime_fingerprint


def _load_cached_trajectory(
    store_root: str | Path,
    trajectory_id: str,
    run_identity: dict[str, object],
) -> NRStoredTrajectory | None:
    """Load a stored trajectory only when its identity and every record verify."""
    if not run_identity.get("complete"):
        return None
    key = run_identity.get("cache_key_sha256")
    inputs = run_identity.get("inputs")
    if not isinstance(key, str) or not isinstance(inputs, dict) or canonical_hash(inputs) != key:
        raise ValueError("invalid complete RT trajectory run identity")
    trajectory_key = hashlib.sha256(trajectory_id.encode()).hexdigest()
    manifest_path = Path(store_root) / "manifests" / "trajectories" / f"{trajectory_key}.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest_values = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_values["sample_ids"] = tuple(manifest_values["sample_ids"])
        stored = NRStoredTrajectory(**manifest_values)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid stored RT trajectory manifest") from exc
    sidecar = manifest_path.with_name(f"{stored.content_sha256}.generation.json")
    if not sidecar.is_file():
        # An interrupted earlier publication can be regenerated and completed.
        return None
    try:
        generation = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid stored RT trajectory generation sidecar") from exc
    previous_identity = generation.get("rt_run_identity")
    if not isinstance(previous_identity, dict):
        raise ValueError("stored trajectory has no reusable RT run identity")
    if previous_identity.get("cache_key_sha256") != key:
        raise ValueError(
            "trajectory_id already exists for a different RT run identity; choose a new ID"
        )
    previous_inputs = previous_identity.get("inputs")
    if (
        previous_identity.get("schema") != "loslab.rt_run_identity.v1"
        or not previous_identity.get("complete")
        or not isinstance(previous_inputs, dict)
        or canonical_hash(previous_inputs) != key
    ):
        raise ValueError("stored trajectory generation identity content does not match its key")
    trajectory = read_nr_trajectory(store_root, stored)
    for record in trajectory.records:
        record_identity = record.provenance.get("rt_run_identity")
        record_inputs = record_identity.get("inputs") if isinstance(record_identity, dict) else None
        if (
            not isinstance(record_identity, dict)
            or record_identity.get("schema") != "loslab.rt_run_identity.v1"
            or not record_identity.get("complete")
            or record_identity.get("cache_key_sha256") != key
            or not isinstance(record_inputs, dict)
            or canonical_hash(record_inputs) != key
        ):
            raise ValueError("stored trajectory record is missing its matching RT run identity")
    return stored


@dataclass(frozen=True, slots=True)
class RTTrajectoryConfig:
    """Physical scene, route, and receiver settings for one RT trajectory."""

    trajectory_id: str
    scene: str
    transmitter_position_m: tuple[float, float, float]
    receiver_start_m: tuple[float, float, float]
    receiver_velocity_mps: tuple[float, float, float]
    num_steps: int = 4
    sample_interval_s: float = 0.1
    carrier_hz: float = 3.5e9
    subcarrier_spacing_khz: int = 30
    n_size_bwp: int = 6
    max_depth: int = 2
    samples_per_src: int = 200
    snr_db: float = 12.0
    seed: int = 20261021
    device: str = "cpu"
    mitsuba_variant: str = "auto"

    def __post_init__(self) -> None:
        if not self.trajectory_id.strip() or not self.scene.strip():
            raise ValueError("trajectory_id and scene must be non-empty")
        for name in ("transmitter_position_m", "receiver_start_m", "receiver_velocity_mps"):
            value = getattr(self, name)
            if len(value) != 3 or any(not math.isfinite(float(item)) for item in value):
                raise ValueError(f"{name} must contain three finite coordinates")
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int):
            raise TypeError("num_steps must be an integer")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if not math.isfinite(self.sample_interval_s) or self.sample_interval_s <= 0:
            raise ValueError("sample_interval_s must be finite and positive")
        if not math.isfinite(self.carrier_hz) or self.carrier_hz <= 0:
            raise ValueError("carrier_hz must be finite and positive")
        if not math.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite")
        for name in ("max_depth", "samples_per_src"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.max_depth < 0 or self.samples_per_src < 1:
            raise ValueError("max_depth must be nonnegative and samples_per_src positive")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if self.seed < 0 or self.seed + self.num_steps > 2**63 - 1:
            raise ValueError("seed and derived per-step seeds must fit in [0, 2**63 - 1]")

    def positions_and_times(
        self,
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, ...]]:
        """Return the constant-velocity receiver route and relative timestamps."""
        positions = tuple(
            (
                float(
                    self.receiver_start_m[0]
                    + self.receiver_velocity_mps[0] * index * self.sample_interval_s
                ),
                float(
                    self.receiver_start_m[1]
                    + self.receiver_velocity_mps[1] * index * self.sample_interval_s
                ),
                float(
                    self.receiver_start_m[2]
                    + self.receiver_velocity_mps[2] * index * self.sample_interval_s
                ),
            )
            for index in range(self.num_steps)
        )
        times = tuple(index * self.sample_interval_s for index in range(self.num_steps))
        return positions, times


def generate_rt_nr_trajectory(
    config: RTTrajectoryConfig,
    store_root: str | Path,
) -> NRStoredTrajectory:
    """Retrace each route snapshot and store the resulting DMRS trajectory.

    Ray traces are recomputed at every route position with deterministic solver
    seeds. Channel/path truth is stored only as targets and provenance; the model
    input consists solely of the noisy received grid and receiver estimate.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Sionna RT, Sionna PHY, Mitsuba, and PyTorch are required") from exc

    if config.device != "cpu" and not config.device.startswith("cuda"):
        raise ValueError("device must be cpu or a CUDA device such as cuda:0")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {config.device}")
    from integrations.sionna.backend import activate_rt_backend, select_rt_backend
    from integrations.sionna.scene_assets import scene_asset_manifest

    selection = select_rt_backend(config.mitsuba_variant)
    activate_rt_backend(selection)
    config = replace(config, mitsuba_variant=selection.variant)
    try:
        import sionna.rt as rt
    except ImportError as exc:
        raise RuntimeError("Sionna RT, Sionna PHY, Mitsuba, and PyTorch are required") from exc

    scene_ref = getattr(rt.scene, config.scene, config.scene)
    scene_path = Path(str(scene_ref)).expanduser()
    if scene_path.is_file():
        scene_asset_hashes = scene_asset_manifest(scene_path)
    else:
        scene_xml = scene_ref if isinstance(scene_ref, str) else repr(scene_ref)
        scene_asset_hashes = {
            "scene_xml_path": scene_xml,
            "asset_count": 0,
            "assets": [],
            "asset_set_sha256": hashlib.sha256(scene_xml.encode()).hexdigest(),
            "unresolved_references": [scene_xml],
            "complete": False,
        }
    positions, times = config.positions_and_times()
    runtime_versions = versions()
    source_sha256 = package_source_sha256()
    runtime_fingerprint = rt_runtime_fingerprint(
        {key: value for key, value in runtime_versions.items() if key != "git_sha"}
    )
    solver_config = {
        "deterministic": True,
        "max_depth": config.max_depth,
        "samples_per_src": config.samples_per_src,
        "los": True,
        "specular_reflection": True,
        "diffuse_reflection": True,
        "refraction": True,
        "diffraction": False,
    }
    run_identity = rt_run_identity(
        scene_asset_manifest=scene_asset_hashes,
        physical_config={
            "trajectory": asdict(config),
            "positions_m": [list(position) for position in positions],
        },
        solver_config=solver_config,
        seed=config.seed,
        backend_selection=selection.as_dict(),
        runtime_versions=runtime_fingerprint,
        package_source_sha256=source_sha256,
    )
    if run_identity["complete"]:
        cached = _load_cached_trajectory(store_root, config.trajectory_id, run_identity)
        if cached is not None:
            return cached
    try:
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
    except ImportError as exc:
        raise RuntimeError("Sionna RT, Sionna PHY, Mitsuba, and PyTorch are required") from exc
    scene = load_scene(str(scene_ref) if isinstance(scene_ref, Path) else scene_ref)
    receiver_config = NRPUSCHConfig(
        n_size_bwp=config.n_size_bwp,
        subcarrier_spacing_khz=config.subcarrier_spacing_khz,
    )
    from .adapter import count_valid_paths
    from .nr_pusch import SionnaPUSCHLSReceiver
    from .rt_to_nr import paths_to_nr_ofdm_channel

    receiver = SionnaPUSCHLSReceiver(receiver_config, device=config.device)
    grid = receiver.resource_grid
    scene.frequency = config.carrier_hz
    scene.bandwidth = grid.fft_size * config.subcarrier_spacing_khz * 1_000.0
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter("trajectory_tx", list(config.transmitter_position_m)))
    rx = Receiver("trajectory_rx", list(config.receiver_start_m))
    scene.add(rx)
    rx.look_at(scene.get("trajectory_tx"))
    tx = scene.get("trajectory_tx")
    tx.look_at(rx)

    full_solver = PathSolver(deterministic=True)
    direct_solver = PathSolver(deterministic=True)
    channels: list[np.ndarray] = []
    direct_path_counts: list[int] = []
    total_path_counts: list[int] = []
    geometric_los: list[bool] = []
    for index, position in enumerate(positions):
        rx.position = list(position)
        tx.look_at(rx)
        rx.look_at(tx)
        # Reuse common random numbers across the route so finite ray-sampling
        # variation does not masquerade as physical channel evolution.
        seed = config.seed
        direct = direct_solver(
            scene,
            max_depth=0,
            los=True,
            specular_reflection=False,
            diffuse_reflection=False,
            refraction=False,
            diffraction=False,
            seed=seed,
        )
        paths = full_solver(
            scene,
            max_depth=config.max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=False,
            samples_per_src=config.samples_per_src,
            seed=seed,
        )
        direct_count = count_valid_paths(direct)
        path_count = count_valid_paths(paths)
        if path_count == 0:
            raise RuntimeError(
                f"No valid RT paths at trajectory step {index}; outages need an explicit receiver policy"
            )
        mapped = paths_to_nr_ofdm_channel(
            paths,
            grid,
            normalize_delays=False,
            normalize=False,
            device=config.device,
        )
        channel = mapped.channel[0, 0, 0, 0, 0, 0].detach().cpu().numpy()
        if channel.shape != (grid.fft_size,) or not np.isfinite(channel).all():
            raise RuntimeError(f"Unexpected or nonfinite RT channel at trajectory step {index}")
        if float(np.mean(np.abs(channel) ** 2)) <= np.finfo(np.float32).tiny:
            raise RuntimeError(f"RT channel has no usable power at trajectory step {index}")
        channels.append(channel.astype(np.complex64))
        direct_path_counts.append(direct_count)
        total_path_counts.append(path_count)
        geometric_los.append(direct_count > 0)

    channel_tensor = torch.as_tensor(
        np.stack(channels), dtype=torch.complex64, device=config.device
    )
    tx_grid, _ = receiver.transmit(config.num_steps, seed=config.seed)
    channel_tensor = channel_tensor.reshape(config.num_steps, 1, 1, 1, grid.fft_size)
    noiseless = tx_grid * channel_tensor
    noise_variance = noiseless.abs().square().mean() * (10.0 ** (-config.snr_db / 10.0))
    noise_generator = torch.Generator(device=config.device).manual_seed(
        config.seed + config.num_steps
    )
    noise = torch.complex(
        torch.randn(
            noiseless.shape, dtype=torch.float32, device=config.device, generator=noise_generator
        ),
        torch.randn(
            noiseless.shape, dtype=torch.float32, device=config.device, generator=noise_generator
        ),
    ) * torch.sqrt(noise_variance / 2.0)
    estimate = receiver.estimate(noiseless + noise, noise_variance)
    indices = tuple(int(index) for index in grid.effective_subcarrier_ind)
    records: list[NRChannelRecord] = []
    for index, (position, timestamp) in enumerate(zip(positions, times, strict=True)):
        received = (noiseless + noise)[index : index + 1]
        channel_estimate = NRChannelEstimate(
            h_hat=estimate.h_hat[index : index + 1],
            error_variance=(
                estimate.error_variance
                if estimate.error_variance.shape[0] == 1
                else estimate.error_variance[index : index + 1]
            ),
            pilot_mask=estimate.pilot_mask,
            metadata={**receiver.metadata, "trajectory_step": index},
        )
        observation = NRObservation(
            sample_id=f"{config.trajectory_id}:{index:04d}",
            received_grid=received,
            channel_estimate=channel_estimate,
            fft_bin_indices=indices,
            receiver_config=receiver_config,
        )
        record_provenance = {
            "source": "sionna_rt_retraced_trajectory",
            "scene": config.scene,
            "trajectory_id": config.trajectory_id,
            "rt_run_identity": run_identity,
            "timestamp_s": timestamp,
            "receiver_position_m": list(position),
            "transmitter_position_m": list(config.transmitter_position_m),
            "geometric_los_oracle": geometric_los[index],
            "target_semantics": {
                "channel_cfr": "absolute-delay full-path complex CFR on effective PUSCH subcarriers",
                "geometric_los": "depth-zero LOS-only PathSolver visibility oracle",
            },
            "target_units": {
                "channel_cfr": "complex channel gain (dimensionless)",
                "geometric_los": "boolean",
            },
            "direct_path_count": direct_path_counts[index],
            "full_path_count": total_path_counts[index],
            "rt_seed": config.seed,
            "receiver_noise_seed": config.seed + config.num_steps,
            "receiver_snr_db": config.snr_db,
            "noise_policy": "common variance from trajectory-batch mean received-grid power",
            "noise_variance": float(noise_variance.detach().cpu()),
            "generation_config": asdict(config),
            "rt_backend_selection": selection.as_dict(),
            "solver": solver_config,
            "scene_asset_set_sha256": scene_asset_hashes["asset_set_sha256"],
            "scene_asset_manifest_complete": scene_asset_hashes["complete"],
            "scene_asset_count": scene_asset_hashes["asset_count"],
            "runtime_versions": runtime_versions,
            "package_source_sha256": source_sha256,
        }
        records.append(
            NRChannelRecord(
                observation=observation,
                targets={
                    "channel_cfr": channels[index],
                    "geometric_los": np.asarray(geometric_los[index], dtype=np.bool_),
                },
                target_masks={
                    "channel_cfr": np.ones(grid.fft_size, dtype=np.bool_),
                    "geometric_los": np.asarray(True, dtype=np.bool_),
                },
                target_definitions={
                    "channel_cfr": TargetDefinition(
                        semantic="absolute-delay full-path complex CFR on effective PUSCH subcarriers",
                        units="complex channel gain (dimensionless)",
                        source="sionna_rt_pathsolver",
                    ),
                    "geometric_los": TargetDefinition(
                        semantic="depth-zero LOS-only PathSolver visibility oracle",
                        units="boolean",
                        source="sionna_rt_depth_zero_oracle",
                    ),
                },
                provenance=record_provenance,
            )
        )

    trajectory = NRTrajectory(config.trajectory_id, times, tuple(records))
    stored = write_nr_trajectory(store_root, trajectory)
    # Emit a compact sidecar alongside the stored trajectory manifest for audit.
    manifest_path = Path(store_root) / stored.manifest_relative_path
    generation = {
        "config": asdict(config),
        "positions_m": [list(position) for position in positions],
        "sample_times_s": list(times),
        "geometric_los": geometric_los,
        "direct_path_counts": direct_path_counts,
        "full_path_counts": total_path_counts,
        "receiver_metadata": receiver.metadata,
        "scene_asset_manifest": scene_asset_hashes,
        "rt_backend_selection": selection.as_dict(),
        "runtime_versions": runtime_versions,
        "package_source_sha256": source_sha256,
        "rt_run_identity": run_identity,
        "limitations": [
            "Each timestamp is a separately retraced quasi-static RT snapshot.",
            "The synthetic scene materials and ray solver are not field calibrated.",
            "Outage snapshots are rejected pending a declared zero-power receiver policy.",
        ],
    }
    # The trajectory index is immutable; generation metadata belongs in a distinct sidecar.
    sidecar = manifest_path.with_name(f"{stored.content_sha256}.generation.json")
    _publish_immutable(
        sidecar,
        __import__("json").dumps(generation, indent=2, sort_keys=True, allow_nan=False).encode()
        + b"\n",
    )
    return stored
