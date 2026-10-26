"""Run the preregistered disjoint 80-seed TDL-A–E family replication."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.qualify_g5_tdl_c_dmrs_density import FAMILYWISE_ALPHA, run_confirmation

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("docs/research/G5_TDL_AE_DMRS_DENSITY_FAMILY_REPLICATION_V2_PROTOCOL_2026-10-24.md")
DEFAULT_OUTPUT = Path(
    "docs/research/results/g5-tdl-ae-dmrs-density-family-reconfirm-v2-20261024.json"
)
PROFILES = ("A", "B", "C", "D", "E")
CHANNEL_SEEDS = tuple(range(936_001, 936_081))
FAMILY_CLAIM_COUNT = len(PROFILES) * 3
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / FAMILY_CLAIM_COUNT
BOOTSTRAP_SEED_BASE = 936_100


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(encoded.encode())


def run_replication() -> dict[str, Any]:
    """Run all five profiles without pooling with any earlier confirmation."""
    protocol_bytes = (ROOT / PROTOCOL).read_bytes()
    reports = [
        run_confirmation(
            profile=profile,
            channel_seeds=CHANNEL_SEEDS,
            per_claim_alpha=PER_CLAIM_ALPHA,
            family_claim_count=FAMILY_CLAIM_COUNT,
            bootstrap_seed=BOOTSTRAP_SEED_BASE + 2 * index,
            protocol_path=PROTOCOL,
            extra_source_paths=("experiments/qualify_g5_tdl_profile_matrix_replication.py",),
        )
        for index, profile in enumerate(PROFILES)
    ]
    if [item["configuration"]["profile"] for item in reports] != [
        f"TDL-{profile}" for profile in PROFILES
    ]:
        raise ValueError("profile runners returned an incomplete or reordered profile family")
    source_hashes = reports[0]["source_sha256"]
    if any(item["source_sha256"] != source_hashes for item in reports[1:]):
        raise ValueError("profile runs do not share one source/protocol inventory")
    protocol_digest = _sha256(protocol_bytes)
    if any(item["protocol_sha256"] != protocol_digest for item in reports):
        raise ValueError("profile runs do not share the frozen replication protocol")

    summaries = [
        {
            "profile": item["configuration"]["profile"],
            "status": item["status"],
            "profile_result_sha256": item["result_sha256"],
            "profile_report": item,
        }
        for item in reports
    ]
    claim_names = (
        "static_reference",
        "mobile_bler_reduction_position_0_minus_3",
        "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
    )
    passed = all(
        all(report["acceptance"].get(name, {}).get("pass") is True for name in claim_names)
        for report in reports
    )
    result: dict[str, Any] = {
        "schema": "umcd.g5_tdl_ae_dmrs_density_family_replication.v2",
        "status": "PASS_BOUNDED_TDL_AE_FIXTURE" if passed else "NOT_MET",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "g5_09_closed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": reports[0]["git_revision"],
        "runtime": reports[0]["runtime"],
        "protocol_path": PROTOCOL.as_posix(),
        "protocol_sha256": protocol_digest,
        "source_sha256": source_hashes,
        "source_inventory_sha256": _canonical_sha256(source_hashes),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "family_claim_count": FAMILY_CLAIM_COUNT,
        "per_claim_alpha_bonferroni": PER_CLAIM_ALPHA,
        "profiles": list(PROFILES),
        "independent_channel_seeds_per_profile": list(CHANNEL_SEEDS),
        "conditions_per_profile": 3,
        "transport_blocks_are_nested_within_channel_seed": True,
        "independent_of_prior_64_seed_family": True,
        "prior_family_result": "docs/research/results/g5-tdl-ae-dmrs-density-family-confirm-20261024.json",
        "invalid_runner_defect_run": "docs/research/results/g5-tdl-ae-dmrs-density-family-reconfirm-20261024.json",
        "profile_summaries": summaries,
        "interpretation": (
            "This disjoint replication supports only the named TDL-A-E fixture if all "
            "fifteen frozen claims pass. It is not measured-channel validity, broad "
            "G5.09 closure, G3 qualification, or production RT authorization. Its "
            "outcomes are not pooled with the earlier 64-seed family."
        ),
    }
    result["configuration_sha256"] = _canonical_sha256(
        {
            "profiles": list(PROFILES),
            "channel_seeds": list(CHANNEL_SEEDS),
            "family_claim_count": FAMILY_CLAIM_COUNT,
            "per_claim_alpha": PER_CLAIM_ALPHA,
            "profile_configurations": [item["configuration"] for item in reports],
        }
    )
    result["result_sha256"] = _canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing replication: {output}")
    report = run_replication()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "family_claim_count": report["family_claim_count"],
                "profile_statuses": {
                    item["profile"]: item["status"] for item in report["profile_summaries"]
                },
                "result_sha256": report["result_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
