"""Sionna 2.2 adapters, imported lazily so CI runs without RT/CUDA.

Important: no fabricated LOS=one-path assumption. A physical LOS link may contain
one direct ray plus any number of reflected, diffuse and refracted rays.
LOS oracle: an independent depth-zero geometric PathSolver invocation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from artifacts.provenance import package_source_sha256, versions
from datasets.channel import ChannelDataset
from propagation.physics import add_estimation_noise, frequency_response, subcarrier_frequencies

from integrations.sionna.random import call_with_sionna_seed


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


def rt_cfr(paths: RTPaths, grid: GridConfig) -> np.ndarray:
    """Convert Sionna RT Paths to [1, links, subcarriers] CFR without path metadata.

    Sionna 2.2 Paths.cfr returns a complex ndarray for out_type='numpy'
    (some earlier interfaces returned a real/imag tuple).
    Preserve all antennas but only one frequency/time snapshot.
    """
    import drjit as dr
    import mitsuba as mi

    f = (dr.arange(mi.Float, grid.n_subcarriers) - grid.n_subcarriers // 2) * grid.spacing_hz
    parts = paths.cfr(
        frequencies=f, num_time_steps=1, normalize_delays=True, normalize=False, out_type="numpy"
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

    For geometrically visible links, the direct-path-only CFR is obtained from
    a separate LOS-only solve; the residual represents indirect propagation.
    Both CFRs must share the same delay reference and antenna coordinates.
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
    mitsuba_variant: str = "llvm_ad_mono_polarized"
    per_class: int = 16
    max_attempts: int = 2000
    require_los_multipath: bool = True
    # built-in 'munich', 'etoile', or a valid user-supplied scene path


def generate_raytraced(
    scene_config: RayScene = RayScene(), *, grid: GridConfig = GridConfig(), seed: int = 42
) -> ChannelDataset:
    """Balanced physical LOS/NLOS ray tracing. Fail closed when insufficient examples.

    Full PathSolver includes direct + reflection (and optional diffuse). Independent
    depth-zero oracle determines geometric LOS without exploiting model profile.
    Report how often LOS really contains >=2 valid paths; default rejects
    direct-only LOS. NLOS must contain at least one observable indirect path.
    """
    try:
        import mitsuba as mi

        # Mitsuba selects its backend before Sionna RT types and scenes are
        # constructed. LLVM is the portable default; CUDA can be requested
        # explicitly on hosts with a working OptiX runtime.
        mi.set_variant(scene_config.mitsuba_variant)
        import sionna.rt as rt
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
    except ImportError as e:
        raise RuntimeError("Sionna RT 2.2 not installed; install sionna==2.2.0") from e
    cfg = scene_config
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
    ):
        raise ValueError("Invalid ray tracing configuration")
    scene_ref = getattr(rt.scene, cfg.name, cfg.name)
    scene = load_scene(scene_ref)
    scene.frequency = grid.carrier_hz
    scene.bandwidth = grid.n_subcarriers * grid.spacing_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter(name="research_tx", position=list(cfg.tx_position)))
    receiver = Receiver(name="research_rx", position=list(cfg.rx_center))
    scene.add(receiver)
    full_solver = PathSolver(deterministic=True)
    los_oracle = PathSolver(deterministic=True)
    rng = np.random.default_rng(seed)
    hits = {0: 0, 1: 0}
    attempt_counts = {
        "attempts_started": 0,
        "obstacle_rejections": 0,
        "depth_zero_oracle_calls": 0,
        "candidate_geometric_nlos": 0,
        "candidate_geometric_los": 0,
        "class_quota_skips": 0,
        "full_solver_calls": 0,
        "full_solver_empty_paths": 0,
        "los_direct_only_rejections": 0,
        "cfr_conversion_failures": 0,
    }
    samples = []
    ys = []
    groups = []
    domains = []
    sources = []
    meta = []
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
        attempt_counts["depth_zero_oracle_calls"] += 1
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
        label = int(has_valid_path(oracle_paths))
        attempt_counts["candidate_geometric_los" if label else "candidate_geometric_nlos"] += 1
        if hits[label] >= cfg.per_class:
            attempt_counts["class_quota_skips"] += 1
            continue
        attempt_counts["full_solver_calls"] += 1
        full = full_solver(
            scene,
            max_depth=cfg.max_depth,
            los=True,
            specular_reflection=True,
            diffuse_reflection=True,
            refraction=True,
            diffraction=False,
            samples_per_src=cfg.samples_per_src,
            seed=seed + attempt,
        )
        n_paths = count_valid_paths(full)
        if n_paths == 0:
            attempt_counts["full_solver_empty_paths"] += 1
            continue
        n_direct = count_valid_paths(oracle_paths)
        mixed = bool(label and n_paths > n_direct)
        if label and cfg.require_los_multipath and not mixed:
            attempt_counts["los_direct_only_rejections"] += 1
            continue
        try:
            channel = rt_cfr(full, grid)
            direct = rt_cfr(oracle_paths, grid) if label else np.zeros_like(channel)
            diagnostics = direct_indirect_diagnostics(channel, direct)
            channel = add_estimation_noise(channel, grid.snr_db, rng)
        except (ValueError, RuntimeError):
            attempt_counts["cfr_conversion_failures"] += 1
            continue
        samples.append(channel[0])
        ys.append(label)
        scene_id = scene_path.stem if scene_path.suffix else cfg.name
        groups.append(
            f"{scene_id}_block_{int(np.floor(x / cfg.block_size_m))}_{int(np.floor(y / cfg.block_size_m))}"
        )
        domains.append(f"rt_{scene_id}")
        sources.append("sionna_rt")
        meta.append(
            {
                "scene": cfg.name,
                "geometry_label": "GEOMETRIC_DIRECT_PATH_ORACLE",
                "tx": list(cfg.tx_position),
                "rx": [x, y, z],
                "valid_paths_full": n_paths,
                "valid_paths_direct_oracle": n_direct,
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
                    "los": True,
                    "specular_reflection": True,
                    "diffuse_reflection": True,
                    "refraction": True,
                    "diffraction": False,
                },
            }
        )
        hits[label] += 1
    if min(hits.values()) < cfg.per_class:
        raise RuntimeError(
            f"Insufficient visible paths after {cfg.max_attempts} attempts: "
            + f"LOS={hits[1]}, NLOS={hits[0]}. Change scene/positions/solver budgets."
        )
    full_solver_outcomes = {
        "empty_paths": attempt_counts["full_solver_empty_paths"],
        "los_direct_only_rejected": attempt_counts["los_direct_only_rejections"],
        "cfr_conversion_failures": attempt_counts["cfr_conversion_failures"],
        "accepted": len(samples),
    }
    if (
        attempt_counts["attempts_started"]
        != attempt_counts["obstacle_rejections"] + attempt_counts["depth_zero_oracle_calls"]
    ):
        raise RuntimeError("RT generation report does not account for every candidate attempt")
    if sum(full_solver_outcomes.values()) != attempt_counts["full_solver_calls"]:
        raise RuntimeError("RT generation report does not account for every full-solver outcome")
    generation_report = {
        "schema": "loslab.raytraced_generation_report.v1",
        "environment": versions(),
        "package_source_sha256": package_source_sha256(),
        "scene_config": asdict(cfg),
        "grid_config": asdict(grid),
        "solver_config": {
            "deterministic": True,
            "max_depth": cfg.max_depth,
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": False,
        },
        "seed": seed,
        "attempt_counts": attempt_counts,
        "full_solver_outcomes": full_solver_outcomes,
        "accepted_counts": {
            "total": len(ys),
            "geometric_nlos": hits[0],
            "geometric_los": hits[1],
        },
        "selection_note": (
            "Balanced accepted records are conditional on the recorded scene, candidate sampler, "
            "solver, class quotas, and LOS-multipath filter. Empty full-solver path outcomes are "
            "counted but are not exported as channel observations."
        ),
    }
    return ChannelDataset(
        np.stack(samples),
        np.asarray(ys),
        np.asarray(groups),
        np.asarray(domains),
        np.asarray(sources),
        tuple(meta),
        generation_report,
    )
