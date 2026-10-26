"""Run the frozen DM-RS density comparison on its disjoint seed set."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.g5_independent_channel_receiver import (
    clopper_pearson_upper_bound,
    cluster_bootstrap_interval,
)
from experiments.scan_g5_doppler_dmrs_density import run_dmrs_density_scan

SCHEMA = "umcd.g5_doppler_dmrs_density_confirmatory.v1"
CHANNEL_SEEDS = tuple(range(85_001, 85_065))
BLOCKS_PER_CHANNEL = 8
FAMILYWISE_ALPHA = 0.01
FAMILY_CLAIM_COUNT = 3
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 85_099
MAXIMUM_STATIC_REFERENCE_BLER = 0.05
MAXIMUM_STATIC_CHANNEL_FAILURE_UPPER_BOUND = 0.10
MINIMUM_MOBILE_BLER_REDUCTION = 0.10


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _source_fingerprints() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        root / "experiments/scan_g5_doppler_dmrs_density.py",
        root / "experiments/g5_independent_channel_receiver.py",
        root / "integrations/sionna/nr_pusch.py",
        root / "integrations/sionna/ofdm_impairments.py",
        root / "receivers/nr/pusch.py",
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _bler_by_channel(records: list[dict[str, Any]], position: str, condition: str) -> list[float]:
    return [
        1.0
        - sum(record["configuration_results"][position]["crc_pass_by_condition"][condition])
        / len(record["configuration_results"][position]["crc_pass_by_condition"][condition])
        for record in records
    ]


def _goodput_by_channel(
    records: list[dict[str, Any]], position: str, condition: str
) -> list[float]:
    return [
        sum(record["configuration_results"][position]["crc_pass_by_condition"][condition])
        * record["configuration_results"][position]["transport_block_bits"]
        / BLOCKS_PER_CHANNEL
        for record in records
    ]


def run_frozen_confirmation() -> dict[str, Any]:
    """Run the one-time frozen 64-channel comparison and evaluate its family."""
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "docs/research/G5_DOPPLER_DMRS_DENSITY_CONFIRMATORY_PROTOCOL_2026-10-24.md"
    )
    protocol_bytes = protocol_path.read_bytes()
    scan = run_dmrs_density_scan(
        channel_seeds=CHANNEL_SEEDS,
        blocks_per_channel=BLOCKS_PER_CHANNEL,
    )
    records = scan["channel_draw_records"]
    reference_bler_by_channel = _bler_by_channel(records, "0", "static")
    reference_failures = sum(rate > 0.0 for rate in reference_bler_by_channel)
    per_claim_alpha = FAMILYWISE_ALPHA / FAMILY_CLAIM_COUNT
    reference_upper_bound = clopper_pearson_upper_bound(
        reference_failures,
        len(records),
        alpha=per_claim_alpha,
    )
    mobile_bler_reduction = [
        low - high
        for low, high in zip(
            _bler_by_channel(records, "0", "750hz_continuous"),
            _bler_by_channel(records, "3", "750hz_continuous"),
            strict=True,
        )
    ]
    mobile_goodput_increase = [
        high - low
        for low, high in zip(
            _goodput_by_channel(records, "0", "750hz_continuous"),
            _goodput_by_channel(records, "3", "750hz_continuous"),
            strict=True,
        )
    ]
    bler_interval = cluster_bootstrap_interval(
        mobile_bler_reduction,
        seed=BOOTSTRAP_SEED,
        alpha=per_claim_alpha,
        replicates=BOOTSTRAP_REPLICATES,
    )
    goodput_interval = cluster_bootstrap_interval(
        mobile_goodput_increase,
        seed=BOOTSTRAP_SEED + 1,
        alpha=per_claim_alpha,
        replicates=BOOTSTRAP_REPLICATES,
    )
    mobile_bler_mean = sum(mobile_bler_reduction) / len(mobile_bler_reduction)
    mobile_goodput_mean = sum(mobile_goodput_increase) / len(mobile_goodput_increase)
    static_bler_position_3 = _bler_by_channel(records, "3", "static")
    static_bler_delta = [
        dense - sparse
        for dense, sparse in zip(static_bler_position_3, reference_bler_by_channel, strict=True)
    ]
    acceptance: dict[str, Any] = {
        "familywise_alpha": FAMILYWISE_ALPHA,
        "family_claim_count": FAMILY_CLAIM_COUNT,
        "per_claim_alpha": per_claim_alpha,
        "bootstrap_resampling_unit": "independent channel draw; eight blocks are nested",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed_bler": BOOTSTRAP_SEED,
        "bootstrap_seed_goodput": BOOTSTRAP_SEED + 1,
        "static_reference": {
            "additional_position_0_mean_bler": sum(reference_bler_by_channel)
            / len(reference_bler_by_channel),
            "channels_with_any_failure": reference_failures,
            "independent_channels": len(records),
            "one_sided_exact_family_adjusted_channel_failure_upper_bound": reference_upper_bound,
            "pass": sum(reference_bler_by_channel) / len(reference_bler_by_channel)
            <= MAXIMUM_STATIC_REFERENCE_BLER
            and reference_upper_bound <= MAXIMUM_STATIC_CHANNEL_FAILURE_UPPER_BOUND,
        },
        "mobile_bler_reduction_position_0_minus_3": {
            "equal_channel_mean": mobile_bler_mean,
            "minimum_required_mean_reduction": MINIMUM_MOBILE_BLER_REDUCTION,
            "family_adjusted_cluster_bootstrap_interval": list(bler_interval),
            "pass": mobile_bler_mean >= MINIMUM_MOBILE_BLER_REDUCTION and bler_interval[0] > 0,
        },
        "mobile_goodput_increase_position_3_minus_0_bits_per_slot": {
            "equal_channel_mean": mobile_goodput_mean,
            "family_adjusted_cluster_bootstrap_interval": list(goodput_interval),
            "pass": goodput_interval[0] > 0,
        },
        "static_bler_delta_position_3_minus_0_descriptive": {
            "equal_channel_mean": sum(static_bler_delta) / len(static_bler_delta),
            "channel_min_max": [min(static_bler_delta), max(static_bler_delta)],
        },
    }
    passed = all(
        acceptance[name]["pass"]
        for name in (
            "static_reference",
            "mobile_bler_reduction_position_0_minus_3",
            "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
        )
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_SYNTHETIC_FIXTURE_ONLY" if passed else "NOT_MET",
        "g5_09_closed": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_path": "docs/research/G5_DOPPLER_DMRS_DENSITY_CONFIRMATORY_PROTOCOL_2026-10-24.md",
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "source_sha256": _source_fingerprints(),
        "configuration": scan["configuration"],
        "acceptance": acceptance,
        "interpretation": "A pass supports only the frozen synthetic 6-RB/30-kHz channel and DM-RS comparison; it is not a general mobility or field-channel qualification.",
        "development_scan_result_sha256": "0cf7940e1f6248c651c63753b147ed9697db8435bb474d1436af7aeaf92f6095",
        "confirmatory_channel_draw_records": records,
    }
    result["configuration_sha256"] = _canonical_sha256(result["configuration"])
    result["result_sha256"] = _canonical_sha256(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/results/g5-doppler-dmrs-density-confirm-20261024.json"),
    )
    args = parser.parse_args()
    report = run_frozen_confirmation()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: status={report['status']}, acceptance={report['acceptance']}")


if __name__ == "__main__":
    main()
