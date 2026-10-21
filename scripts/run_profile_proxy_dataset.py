"""Generate a provenance-recorded TDL/CDL profile-proxy dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import atomic_json, package_source_sha256, sha256_file, versions
from datasets.channel import ChannelDataset
from integrations.sionna.adapter import GridConfig, generate_standardized


def summarize_dataset(data: ChannelDataset) -> list[dict[str, Any]]:
    """Summarize row counts and received CFR powers by profile and delay spread."""
    groups: dict[tuple[str, str, float], list[float]] = {}
    for channel, metadata in zip(data.h, data.metadata, strict=True):
        key = (
            str(metadata["family"]).lower(),
            str(metadata["profile"]).upper(),
            float(metadata["spread_ns"]),
        )
        groups.setdefault(key, []).append(float(np.mean(np.abs(channel) ** 2)))
    return [
        {
            "family": family,
            "profile": profile,
            "delay_spread_ns": spread,
            "samples": len(powers),
            "profile_proxy_label": int(profile in "DE"),
            "mean_observed_cfr_power": float(np.mean(powers)),
            "std_observed_cfr_power": float(np.std(powers)),
        }
        for (family, profile, spread), powers in sorted(groups.items())
    ]


def run_dataset(
    output: Path,
    *,
    samples_per_profile: int,
    seed: int,
    device: str,
    spec_version: str,
    snr_db: float,
    n_subcarriers: int,
    delay_spreads_ns: Sequence[float],
) -> dict[str, Any]:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite dataset or manifest: {output}")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    grid = GridConfig(n_subcarriers=n_subcarriers, snr_db=snr_db)
    data = generate_standardized(
        grid=grid,
        profiles=("A", "B", "C", "D", "E"),
        families=("tdl", "cdl"),
        per_profile=samples_per_profile,
        seed=seed,
        delay_spreads_ns=delay_spreads_ns,
        device=device,
        spec_version=spec_version,
    )
    data.save(output)
    dependency_versions = versions()
    hardware: dict[str, Any] = {"device": device}
    try:
        import torch

        hardware["cuda_available"] = torch.cuda.is_available()
        if device.startswith("cuda") and torch.cuda.is_available():
            hardware["cuda_device_name"] = torch.cuda.get_device_name(torch.device(device))
    except ImportError:
        hardware["cuda_available"] = False
    manifest = {
        "schema": "loslab.profile_proxy_dataset.v1",
        "evidence_grade": "SIMULATED_3GPP_PROFILE_PROXY_NOT_GEOMETRIC_LOS",
        "dataset_path": str(output),
        "dataset_sha256": sha256_file(output),
        "dataset_bytes": output.stat().st_size,
        "source_revision": dependency_versions.get("git_sha"),
        "package_source_sha256": package_source_sha256(),
        "runner_sha256": sha256_file(__file__),
        "versions": dependency_versions,
        "hardware": hardware,
        "configuration": {
            "families": ["tdl", "cdl"],
            "profiles": ["A", "B", "C", "D", "E"],
            "samples_per_family_profile": samples_per_profile,
            "delay_spreads_ns": list(delay_spreads_ns),
            "seed": seed,
            "device": device,
            "carrier_hz": grid.carrier_hz,
            "subcarrier_spacing_hz": grid.spacing_hz,
            "n_subcarriers": n_subcarriers,
            "snr_db": snr_db,
            "spec_version": spec_version,
        },
        "sample_count": len(data.y),
        "class_counts": {str(label): int(np.sum(data.y == label)) for label in (0, 1)},
        "profile_summaries": summarize_dataset(data),
        "limitations": [
            "Labels indicate 3GPP profile proxies, not geometric LOS/NLOS or blockage truth.",
            "The generated channels are simulator outputs and are not measured-data validation.",
            "Profile proxy results do not replace scene-disjoint evaluation on qualified RT or measured data.",
        ],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-profile", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=78021)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--spec-version", default="19.2")
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--n-subcarriers", type=int, default=128)
    parser.add_argument("--delay-spreads-ns", type=float, nargs="+", default=[30, 100, 300, 1000])
    args = parser.parse_args()
    try:
        manifest = run_dataset(
            args.output,
            samples_per_profile=args.samples_per_profile,
            seed=args.seed,
            device=args.device,
            spec_version=args.spec_version,
            snr_db=args.snr_db,
            n_subcarriers=args.n_subcarriers,
            delay_spreads_ns=args.delay_spreads_ns,
        )
    except (RuntimeError, ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(f"Saved {manifest['sample_count']} profile-proxy rows to {args.output}")
    print(f"Manifest: {args.output.with_suffix(args.output.suffix + '.manifest.json')}")


if __name__ == "__main__":
    main()
