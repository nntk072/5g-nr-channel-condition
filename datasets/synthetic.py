"""Cheap physics-inspired multipath smoke tests, never publication evidence."""

from __future__ import annotations

import numpy as np
from propagation.physics import add_estimation_noise, frequency_response, subcarrier_frequencies

from .channel import ChannelDataset


def debug_multipath(
    n_scenes: int = 4,
    links_per_scene: int = 36,
    n_subcarriers: int = 64,
    seed: int = 19,
    snr_db: float = 12.0,
) -> ChannelDataset:
    """Both LOS and NLOS receive reflections; LOS also has a direct path.

    Deliberately not a 3GPP model; labels are known via construction.
    """
    if n_scenes < 2 or links_per_scene < 4 or links_per_scene % 2:
        raise ValueError("Need >=2 scenes, positive even links_per_scene >= 4")
    rng = np.random.default_rng(seed)
    f = subcarrier_frequencies(n_subcarriers, 30e3)
    records = []
    labels = []
    groups = []
    domains = []
    sources = []
    metadata = []
    for scene in range(n_scenes):
        # Scene-specific power profiles that are NOT explicit class labels.
        scale = float(rng.uniform(0.4, 2.0))
        for link in range(links_per_scene):
            y = int(link % 2 == 0)
            n_paths = int(rng.integers(5, 15))
            tau = np.sort(rng.uniform(0.08e-6, 2e-6, n_paths))
            # Reflection-only components exist in both classes.
            refl = (rng.standard_normal(n_paths) + 1j * rng.standard_normal(n_paths)) / np.sqrt(
                2 * n_paths
            )
            refl *= np.exp(-tau / (scale * 2e-6))
            a = refl.astype(np.complex128)
            if y:
                # Direct component may be weaker or stronger than reflected power.
                direct_power = 10 ** rng.uniform(-10, 8) / 10
                a = np.concatenate(
                    [[np.sqrt(direct_power) * np.exp(1j * rng.uniform(-np.pi, np.pi))], a]
                )
                tau = np.concatenate([[0.0], tau])
            h = frequency_response(a, tau, f).reshape(1, -1)
            h = add_estimation_noise(h, snr_db, rng)
            records.append(h)
            labels.append(y)
            groups.append(f"debug_scene_{scene}_link_{link}")
            domains.append(f"debug_scene_{scene}")
            sources.append("synthetic_debug")
            metadata.append(
                {
                    "scene": scene,
                    "direct_path": bool(y),
                    "reflected_paths": n_paths,
                    "note": "debug only; NOT Sionna or measured channel",
                }
            )
    return ChannelDataset(
        np.stack(records),
        np.asarray(labels),
        np.asarray(groups),
        np.asarray(domains),
        np.asarray(sources),
        tuple(metadata),
    )
