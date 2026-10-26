"""Run the preregistered TDL-C signed-lag DFT estimator confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.qualify_g5_tdl_c_dmrs_density import FAMILYWISE_ALPHA, run_confirmation

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("docs/research/G5_TDL_C_SIGNED_LAG_DFT_CONFIRMATORY_PROTOCOL_2026-10-24.md")
DEFAULT_OUTPUT = Path("docs/research/results/g5-tdl-c-signed-lag-dft-confirm-20261024.json")
CHANNEL_SEEDS = tuple(range(948_001, 948_081))
FAMILY_CLAIM_COUNT = 3
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / FAMILY_CLAIM_COUNT
BOOTSTRAP_SEED = 948_099


def run_dft_confirmation() -> dict[str, Any]:
    """Run the three preregistered TDL-C conditions using one DFT estimator."""
    return run_confirmation(
        profile="C",
        channel_seeds=CHANNEL_SEEDS,
        per_claim_alpha=PER_CLAIM_ALPHA,
        family_claim_count=FAMILY_CLAIM_COUNT,
        bootstrap_seed=BOOTSTRAP_SEED,
        protocol_path=PROTOCOL,
        extra_source_paths=(
            "experiments/qualify_g5_tdl_dft_density.py",
            "experiments/g5_cp_bounded_dft_estimator.py",
        ),
        channel_estimator_kind="finite_delay_dft",
        estimator_min_delay_samples=-6,
        estimator_max_delay_samples=7,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation: {output}")
    report = run_dft_confirmation()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "estimator": report["configuration"]["channel_estimator"],
                "acceptance": report["acceptance"],
                "result_sha256": report["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
