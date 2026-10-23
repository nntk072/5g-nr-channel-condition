"""Convert saved RT CFR datasets to PUSCH/DMRS receiver observations."""

from __future__ import annotations

import argparse
from pathlib import Path

from artifacts.provenance import package_source_sha256, versions
from datasets.channel import ChannelDataset
from integrations.sionna.d1_generation_gate import authorize_d1_generation
from integrations.sionna.rt_dmrs import (
    build_rt_dmrs_generation_request,
    build_rt_dmrs_observation_dataset,
    read_stable_input_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--snr-db", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-size-bwp", type=int, default=6)
    parser.add_argument("--subcarrier-spacing-khz", type=int, default=30)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--pilot", action="store_true")
    policy.add_argument("--qualification-approval", type=Path)
    parser.add_argument("--qualification-approval-sha256")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing dataset: {args.output}")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        parser.error(f"input dataset does not exist: {missing[0]}")
    try:
        input_payloads = [read_stable_input_artifact(path) for path in args.inputs]
        datasets = [ChannelDataset.load_bytes(payload) for payload, _ in input_payloads]
        source_artifacts = [artifact for _, artifact in input_payloads]
        request = build_rt_dmrs_generation_request(
            datasets,
            source_artifacts=source_artifacts,
            snr_db=args.snr_db,
            seed=args.seed,
            device=args.device,
            n_size_bwp=args.n_size_bwp,
            subcarrier_spacing_khz=args.subcarrier_spacing_khz,
            destination=args.output,
        )
        authorization = authorize_d1_generation(
            request,
            pilot=args.pilot,
            approval_manifest=args.qualification_approval,
            expected_approval_sha256=args.qualification_approval_sha256,
        )
        observed = build_rt_dmrs_observation_dataset(
            datasets,
            authorization=authorization,
            source_artifacts=source_artifacts,
            snr_db=args.snr_db,
            seed=args.seed,
            device=args.device,
            n_size_bwp=args.n_size_bwp,
            subcarrier_spacing_khz=args.subcarrier_spacing_khz,
            destination=args.output,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))

    report = {
        **observed.generation_report,
        "input_artifacts": source_artifacts,
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
    output_dataset.save(args.output, exclusive=True)
    print(f"Saved {len(output_dataset.y)} NR DMRS observations to {args.output}")


if __name__ == "__main__":
    main()
