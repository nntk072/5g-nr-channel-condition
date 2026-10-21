"""Convert saved RT CFR datasets to PUSCH/DMRS receiver observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from artifacts.provenance import package_source_sha256, sha256_file, versions
from datasets.channel import ChannelDataset
from integrations.sionna.rt_dmrs import build_rt_dmrs_observation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--snr-db", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-size-bwp", type=int, default=6)
    parser.add_argument("--subcarrier-spacing-khz", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing dataset: {args.output}")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        parser.error(f"input dataset does not exist: {missing[0]}")
    try:
        datasets = [ChannelDataset.load(path) for path in args.inputs]
        observed = build_rt_dmrs_observation_dataset(
            datasets,
            snr_db=args.snr_db,
            seed=args.seed,
            device=args.device,
            n_size_bwp=args.n_size_bwp,
            subcarrier_spacing_khz=args.subcarrier_spacing_khz,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    report = {
        **observed.generation_report,
        "input_artifacts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.inputs
        ],
        "source_revision": versions().get("git_sha"),
        "runtime_versions": versions(),
        "package_source_sha256": package_source_sha256(),
    }
    output_dataset = ChannelDataset(
        h=observed.h,
        y=observed.y,
        group=observed.group,
        domain=observed.domain,
        source=observed.source,
        metadata=observed.metadata,
        generation_report=report,
    )
    output_dataset.save(args.output)
    print(f"Saved {len(output_dataset.y)} NR DMRS observations to {args.output}")


if __name__ == "__main__":
    main()
