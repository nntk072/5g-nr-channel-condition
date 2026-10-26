"""Independently verify the frozen G5 delay-domain DFT result artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.g5_independent_channel_receiver import (
    clopper_pearson_upper_bound,
    cluster_bootstrap_interval,
)
from experiments.qualify_g5_delay_domain_dft import (
    ALPHA,
    BLOCKS_PER_CHANNEL,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED_BLER,
    BOOTSTRAP_SEED_GOODPUT,
    CHANNEL_SEEDS,
    CLAIM_COUNT,
    _canonical_sha256,
    _source_fingerprints,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_report(report: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Check artifact bindings, design roster, retained outcomes, and inference."""
    _require(report.get("schema") == "umcd.g5_delay_domain_dft_confirmatory.v1", "bad schema")
    claimed_digest = report.get("result_sha256")
    digest_payload = dict(report)
    digest_payload.pop("result_sha256", None)
    _require(claimed_digest == _canonical_sha256(digest_payload), "result digest mismatch")
    protocol_path = root / report["protocol_path"]
    _require(
        hashlib.sha256(protocol_path.read_bytes()).hexdigest() == report["protocol_sha256"],
        "protocol digest mismatch",
    )
    _require(report["source_sha256"] == _source_fingerprints(), "source fingerprint mismatch")
    configuration = report["configuration"]
    records = report["confirmatory_channel_draw_records"]
    _require(configuration["channel_seeds"] == list(CHANNEL_SEEDS), "unexpected seed roster")
    _require(len(records) == len(CHANNEL_SEEDS), "wrong independent channel count")
    _require(configuration["blocks_per_channel"] == BLOCKS_PER_CHANNEL, "wrong block count")
    for expected_seed, record in zip(CHANNEL_SEEDS, records, strict=True):
        _require(record["channel_seed"] == expected_seed, "channel roster/order mismatch")
        _require(record["waveform_noise_seed"] == expected_seed + 2_000_000, "noise seed mismatch")
        _require(record["payload_seed"] == expected_seed + 1_000_000, "payload seed mismatch")
        _require(2 <= record["echo_delay_samples"] <= 8, "echo delay outside frozen support")
        results = record["configuration_results"]
        _require(set(results) == {"0", "3"}, "DM-RS settings missing")
        _require(
            results["0"]["waveform_noise_sha256"] == results["3"]["waveform_noise_sha256"],
            "paired noise mismatch",
        )
        for result in results.values():
            _require(result["payload_seed"] == record["payload_seed"], "payload pairing mismatch")
            _require(
                set(result["crc_pass_by_condition"]) == {"static", "750hz_continuous"},
                "condition mismatch",
            )
            for outcomes in result["crc_pass_by_condition"].values():
                _require(len(outcomes) == BLOCKS_PER_CHANNEL, "missing block-level outcomes")
                _require(
                    all(type(value) is bool for value in outcomes), "CRC values must be booleans"
                )

    def rates(position: str, condition: str) -> list[float]:
        return [
            1
            - sum(r["configuration_results"][position]["crc_pass_by_condition"][condition])
            / BLOCKS_PER_CHANNEL
            for r in records
        ]

    def goodput(position: str, condition: str) -> list[float]:
        return [
            sum(r["configuration_results"][position]["crc_pass_by_condition"][condition])
            * r["configuration_results"][position]["transport_block_bits"]
            / BLOCKS_PER_CHANNEL
            for r in records
        ]

    alpha = ALPHA / CLAIM_COUNT
    static = rates("0", "static")
    failures = sum(rate > 0 for rate in static)
    upper = clopper_pearson_upper_bound(failures, len(records), alpha=alpha)
    bler_delta = [
        a - b
        for a, b in zip(rates("0", "750hz_continuous"), rates("3", "750hz_continuous"), strict=True)
    ]
    goodput_delta = [
        a - b
        for a, b in zip(
            goodput("3", "750hz_continuous"), goodput("0", "750hz_continuous"), strict=True
        )
    ]
    bler_ci = cluster_bootstrap_interval(
        bler_delta, seed=BOOTSTRAP_SEED_BLER, alpha=alpha, replicates=BOOTSTRAP_REPLICATES
    )
    goodput_ci = cluster_bootstrap_interval(
        goodput_delta, seed=BOOTSTRAP_SEED_GOODPUT, alpha=alpha, replicates=BOOTSTRAP_REPLICATES
    )
    acceptance = report["acceptance"]
    _require(acceptance["bootstrap_replicates"] == BOOTSTRAP_REPLICATES, "bootstrap count mismatch")
    _require(
        acceptance["bootstrap_seed_bler"] == BOOTSTRAP_SEED_BLER, "BLER bootstrap seed mismatch"
    )
    _require(
        acceptance["bootstrap_seed_goodput"] == BOOTSTRAP_SEED_GOODPUT,
        "goodput bootstrap seed mismatch",
    )
    expected_static_pass = sum(static) / len(static) <= 0.05 and upper <= 0.10
    bler_mean = sum(bler_delta) / len(bler_delta)
    goodput_mean = sum(goodput_delta) / len(goodput_delta)
    _require(
        acceptance["static_reference"]["channels_with_any_failure"] == failures
        and abs(
            acceptance["static_reference"][
                "one_sided_exact_family_adjusted_channel_failure_upper_bound"
            ]
            - upper
        )
        < 1e-12
        and acceptance["static_reference"]["pass"] is expected_static_pass,
        "static claim does not recompute",
    )
    bler_claim = acceptance["mobile_bler_reduction_position_0_minus_3"]
    _require(abs(bler_claim["equal_channel_mean"] - bler_mean) < 1e-12, "BLER mean mismatch")
    _require(
        all(
            abs(a - b) < 1e-12
            for a, b in zip(
                bler_claim["family_adjusted_cluster_bootstrap_interval"], bler_ci, strict=True
            )
        ),
        "BLER interval mismatch",
    )
    _require(bler_claim["pass"] is (bler_mean >= 0.10 and bler_ci[0] > 0), "BLER decision mismatch")
    goodput_claim = acceptance["mobile_goodput_increase_position_3_minus_0_bits_per_slot"]
    _require(
        abs(goodput_claim["equal_channel_mean"] - goodput_mean) < 1e-12, "goodput mean mismatch"
    )
    _require(
        all(
            abs(a - b) < 1e-12
            for a, b in zip(
                goodput_claim["family_adjusted_cluster_bootstrap_interval"], goodput_ci, strict=True
            )
        ),
        "goodput interval mismatch",
    )
    _require(goodput_claim["pass"] is (goodput_ci[0] > 0), "goodput decision mismatch")
    pass_all = expected_static_pass and bler_mean >= 0.10 and bler_ci[0] > 0 and goodput_ci[0] > 0
    _require(
        report["status"] == ("PASS_NAMED_SYNTHETIC_CONFIGURATION_ONLY" if pass_all else "NOT_MET"),
        "overall status mismatch",
    )
    return {
        "status": "AUDIT_PASS",
        "independent_channels": len(records),
        "blocks_per_channel": BLOCKS_PER_CHANNEL,
        "claims_passed": sum(
            bool(acceptance[name]["pass"])
            for name in (
                "static_reference",
                "mobile_bler_reduction_position_0_minus_3",
                "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
            )
        ),
        "result_sha256": claimed_digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result",
        nargs="?",
        type=Path,
        default=Path("docs/research/results/g5-delay-domain-dft-confirm-20261024.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = json.loads(args.result.read_text(encoding="utf-8"))
    print(json.dumps(audit_report(report, root=root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
