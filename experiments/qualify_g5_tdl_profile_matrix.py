"""Run the frozen five-profile, family-wise TDL DM-RS density confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.qualify_g5_tdl_c_dmrs_density import (
    FAMILYWISE_ALPHA,
    run_confirmation,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("docs/research/G5_TDL_AE_DMRS_DENSITY_FAMILY_CONFIRMATORY_PROTOCOL_2026-10-24.md")
DEFAULT_OUTPUT = Path("docs/research/results/g5-tdl-ae-dmrs-density-family-confirm-20261024.json")
PROFILES = ("A", "B", "C", "D", "E")
CHANNEL_SEEDS = tuple(range(934_001, 934_065))
FAMILY_CLAIM_COUNT = len(PROFILES) * 3
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / FAMILY_CLAIM_COUNT
BOOTSTRAP_SEED_BASE = 934_100


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(encoded.encode())


def run_family_confirmation() -> dict[str, Any]:
    """Run every TDL profile under one Bonferroni family of 15 claims."""
    protocol_bytes = (ROOT / PROTOCOL).read_bytes()
    reports = []
    for profile_index, profile in enumerate(PROFILES):
        reports.append(
            run_confirmation(
                profile=profile,
                channel_seeds=CHANNEL_SEEDS,
                per_claim_alpha=PER_CLAIM_ALPHA,
                family_claim_count=FAMILY_CLAIM_COUNT,
                bootstrap_seed=BOOTSTRAP_SEED_BASE + 2 * profile_index,
                protocol_path=PROTOCOL,
                extra_source_paths=("experiments/qualify_g5_tdl_profile_matrix.py",),
            )
        )
    if [item["configuration"]["profile"] for item in reports] != [
        f"TDL-{profile}" for profile in PROFILES
    ]:
        raise ValueError("profile runners returned an incomplete or reordered profile family")
    source_hashes = reports[0]["source_sha256"]
    if any(item["source_sha256"] != source_hashes for item in reports[1:]):
        raise ValueError("profile runs do not share the same source/protocol inventory")
    if any(item["protocol_sha256"] != _sha256(protocol_bytes) for item in reports):
        raise ValueError("profile runs do not share the frozen protocol bytes")
    profile_summaries = [
        {
            "profile": item["configuration"]["profile"],
            "status": item["status"],
            "profile_result_sha256": item["result_sha256"],
            "profile_report": item,
        }
        for item in reports
    ]
    passed = all(
        sum(
            isinstance(claim, dict) and "pass" in claim
            for claim in summary["profile_report"]["acceptance"].values()
        )
        == 3
        and all(
            claim["pass"]
            for claim in summary["profile_report"]["acceptance"].values()
            if isinstance(claim, dict) and "pass" in claim
        )
        for summary in profile_summaries
    )
    report: dict[str, Any] = {
        "schema": "umcd.g5_tdl_ae_dmrs_density_family_confirmatory.v1",
        "status": "PASS_BOUNDED_TDL_AE_FIXTURE" if passed else "NOT_MET",
        "g5_09_closed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": reports[0]["git_revision"],
        "runtime": reports[0]["runtime"],
        "protocol_path": PROTOCOL.as_posix(),
        "protocol_sha256": _sha256(protocol_bytes),
        "source_sha256": source_hashes,
        "source_inventory_sha256": _canonical_sha256(source_hashes),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "family_claim_count": FAMILY_CLAIM_COUNT,
        "per_claim_alpha_bonferroni": PER_CLAIM_ALPHA,
        "profiles": list(PROFILES),
        "independent_channel_seeds_per_profile": list(CHANNEL_SEEDS),
        "conditions_per_profile": 3,
        "transport_blocks_are_nested_within_channel_seed": True,
        "profile_summaries": profile_summaries,
        "interpretation": (
            "A family pass supports only TDL-A-E at the frozen 100-ns, 3.5-GHz, "
            "120-km/h, 18-dB, 12-RB/30-kHz CPU PUSCH configuration. It is not "
            "measured-channel validation, general G5.09 closure, G3 qualification, "
            "or production RT authorization."
        ),
    }
    report["configuration_sha256"] = _canonical_sha256(
        {
            "profiles": list(PROFILES),
            "channel_seeds": list(CHANNEL_SEEDS),
            "family_claim_count": FAMILY_CLAIM_COUNT,
            "per_claim_alpha": PER_CLAIM_ALPHA,
            "profile_configurations": [item["configuration"] for item in reports],
        }
    )
    report["result_sha256"] = _canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing confirmation: {output}")
    report = run_family_confirmation()
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
