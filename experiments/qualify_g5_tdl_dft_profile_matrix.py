"""Run the frozen five-profile, family-wise signed-lag DFT confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.qualify_g5_tdl_c_dmrs_density import FAMILYWISE_ALPHA, run_confirmation

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path("docs/research/G5_TDL_AE_SIGNED_LAG_DFT_FAMILY_CONFIRMATORY_PROTOCOL_2026-10-24.md")
DEFAULT_OUTPUT = Path("docs/research/results/g5-tdl-ae-signed-lag-dft-family-confirm-20261024.json")
PROFILES = ("A", "B", "C", "D", "E")
CHANNEL_SEEDS = tuple(range(949_001, 949_081))
FAMILY_CLAIM_COUNT = len(PROFILES) * 3
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / FAMILY_CLAIM_COUNT
BOOTSTRAP_SEED_BASE = 949_101


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(encoded.encode())


def run_family_confirmation() -> dict[str, Any]:
    """Run all five profiles with one fixed estimator and one 15-claim family."""
    protocol_bytes = (ROOT / PROTOCOL).read_bytes()
    reports = [
        run_confirmation(
            profile=profile,
            channel_seeds=CHANNEL_SEEDS,
            per_claim_alpha=PER_CLAIM_ALPHA,
            family_claim_count=FAMILY_CLAIM_COUNT,
            bootstrap_seed=BOOTSTRAP_SEED_BASE + 2 * index,
            protocol_path=PROTOCOL,
            extra_source_paths=(
                "experiments/qualify_g5_tdl_dft_profile_matrix.py",
                "experiments/g5_cp_bounded_dft_estimator.py",
            ),
            channel_estimator_kind="finite_delay_dft",
            estimator_min_delay_samples=-6,
            estimator_max_delay_samples=7,
        )
        for index, profile in enumerate(PROFILES)
    ]
    expected_profiles = [f"TDL-{profile}" for profile in PROFILES]
    if [item["configuration"]["profile"] for item in reports] != expected_profiles:
        raise ValueError("profile runners returned an incomplete or reordered profile family")
    source_hashes = reports[0]["source_sha256"]
    if any(item["source_sha256"] != source_hashes for item in reports[1:]):
        raise ValueError("profile runs do not share one source inventory")
    protocol_digest = _sha256(protocol_bytes)
    if any(item["protocol_sha256"] != protocol_digest for item in reports):
        raise ValueError("profile runs do not share the frozen protocol bytes")

    claim_names = (
        "static_reference",
        "mobile_bler_reduction_position_0_minus_3",
        "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
    )
    passed = all(
        all(item["acceptance"].get(name, {}).get("pass") is True for name in claim_names)
        for item in reports
    )
    summaries = [
        {
            "profile": item["configuration"]["profile"],
            "status": item["status"],
            "profile_result_sha256": item["result_sha256"],
            "profile_report": item,
        }
        for item in reports
    ]
    family: dict[str, Any] = {
        "schema": "umcd.g5_tdl_ae_signed_lag_dft_family_confirmatory.v1",
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
        "channel_estimator": {
            "kind": "finite_delay_dft",
            "min_delay_samples": -6,
            "max_delay_samples": 7,
        },
        "transport_blocks_are_nested_within_channel_seed": True,
        "profile_summaries": summaries,
        "interpretation": (
            "A family pass supports only TDL-A-E at the frozen 100-ns, 3.5-GHz, "
            "120-km/h, 18-dB, 12-RB/30-kHz CPU PUSCH configuration and signed-lag "
            "DFT estimator. It is not measured-channel validation, general G5.09 "
            "closure, G3 qualification, or production RT authorization."
        ),
    }
    family["configuration_sha256"] = _canonical_sha256(
        {
            "profiles": list(PROFILES),
            "channel_seeds": list(CHANNEL_SEEDS),
            "family_claim_count": FAMILY_CLAIM_COUNT,
            "per_claim_alpha": PER_CLAIM_ALPHA,
            "channel_estimator": family["channel_estimator"],
            "profile_configurations": [item["configuration"] for item in reports],
        }
    )
    family["result_sha256"] = _canonical_sha256(family)
    return family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite confirmation: {output}")
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
