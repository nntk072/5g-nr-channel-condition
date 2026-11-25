"""Apply a Sionna PUSCH/DMRS receiver to saved RT channel-frequency-response datasets.

Outputs are simulation-only observations, not measured radio evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a Sionna PUSCH/DMRS receiver to saved RT CFR datasets"
    )
    parser.add_argument("--input", nargs="+", required=True, help="RT dataset NPZ files")
    parser.add_argument("--output", required=True, help="New observation dataset NPZ")
    parser.add_argument("--snr-db", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-size-bwp", type=int, default=6)
    parser.add_argument("--subcarrier-spacing-khz", type=int, default=30)
    nr_policy = parser.add_mutually_exclusive_group()
    nr_policy.add_argument("--pilot", action="store_true")
    nr_policy.add_argument(
        "--qualification-approval", type=Path, help="reserved; production authorization is disabled"
    )
    parser.add_argument(
        "--qualification-approval-sha256",
        help="reserved while production authorization is disabled",
    )
    args = parser.parse_args()
    from datasets.channel import ChannelDataset
    from integrations.sionna.d1_generation_gate import authorize_d1_generation
    from integrations.sionna.rt_dmrs import (
        build_rt_dmrs_generation_request,
        build_rt_dmrs_observation_dataset,
        read_stable_input_artifact,
    )

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing observation dataset: {output_path}")
    input_payloads = [read_stable_input_artifact(path) for path in args.input]
    sources = [ChannelDataset.load_bytes(payload) for payload, _ in input_payloads]
    source_artifacts = [artifact for _, artifact in input_payloads]
    request = build_rt_dmrs_generation_request(
        sources,
        source_artifacts=source_artifacts,
        snr_db=args.snr_db,
        seed=args.seed,
        device=args.device,
        n_size_bwp=args.n_size_bwp,
        subcarrier_spacing_khz=args.subcarrier_spacing_khz,
        destination=output_path,
    )
    try:
        authorization = authorize_d1_generation(
            request,
            pilot=args.pilot,
            approval_manifest=args.qualification_approval,
            expected_approval_sha256=args.qualification_approval_sha256,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    observed = build_rt_dmrs_observation_dataset(
        sources,
        authorization=authorization,
        source_artifacts=source_artifacts,
        snr_db=args.snr_db,
        seed=args.seed,
        device=args.device,
        n_size_bwp=args.n_size_bwp,
        subcarrier_spacing_khz=args.subcarrier_spacing_khz,
        destination=output_path,
    )
    observed.save(output_path, exclusive=True)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "samples": len(observed.y),
                "shape": list(observed.h.shape),
                "label_semantics": observed.label_semantics,
                "observation_semantics": observed.generation_report["observation_semantics"],
                "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
                "warning": "RT/DMRS simulation is not measured radio evidence.",
            },
            indent=2,
        )
    )
    return


if __name__ == "__main__":
    main()
