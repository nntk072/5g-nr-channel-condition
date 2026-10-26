"""Audit the saved, hash-bound TDL-C DM-RS density confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

from experiments.g5_independent_channel_receiver import (
    clopper_pearson_upper_bound,
    cluster_bootstrap_interval,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = Path("docs/research/results/g5-tdl-c-dmrs-density-confirm-20261024.json")
PROTOCOL = Path("docs/research/G5_TDL_C_DMRS_DENSITY_CONFIRMATORY_PROTOCOL_2026-10-24.md")
EXPECTED_SEEDS = list(range(933_001, 933_065))
ALPHA = 0.01 / 3
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEEDS = {"bler": 933_099, "goodput": 933_100}
EXPECTED_SOURCE_PATHS = {
    "experiments/qualify_g5_tdl_c_dmrs_density.py",
    "experiments/g5_independent_channel_receiver.py",
    "experiments/link_curves.py",
    "integrations/sionna/nr_pusch.py",
    "integrations/sionna/tdl_pusch.py",
    "receivers/nr/pusch.py",
    "evaluation/statistics.py",
    PROTOCOL.as_posix(),
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Compute the two-sided 95% Wilson score interval for a binomial rate."""
    z = NormalDist().inv_cdf(0.975)
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    half_width = (
        z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    )
    return center - half_width, center + half_width


def audit_report(
    report: dict[str, Any],
    *,
    root: Path = ROOT,
    protocol_path: Path = PROTOCOL,
    expected_seeds: list[int] = EXPECTED_SEEDS,
    expected_profile: str = "C",
    expected_source_paths: set[str] = EXPECTED_SOURCE_PATHS,
    expected_channel_estimator: dict[str, Any] | None = None,
    bootstrap_seeds: dict[str, int] = BOOTSTRAP_SEEDS,
    familywise_alpha: float = 0.01,
    family_claim_count: int = 3,
) -> dict[str, Any]:
    """Check report/source/protocol hashes, complete pairing, and verdict math."""
    issues: list[str] = []
    recorded_result_hash = report.get("result_sha256")
    unhashed = dict(report)
    unhashed.pop("result_sha256", None)
    if recorded_result_hash != _canonical_sha256(unhashed):
        issues.append("result_sha256 does not match report contents")

    config = report.get("configuration")
    if not isinstance(config, dict):
        config = {}
        issues.append("configuration is missing")
    elif report.get("configuration_sha256") != _canonical_sha256(config):
        issues.append("configuration_sha256 does not match configuration")
    if config.get("independent_channel_seeds") != expected_seeds:
        issues.append("channel seed list differs from the frozen protocol")
    if config.get("frames_per_condition") != len(expected_seeds):
        issues.append("frame count differs from the frozen protocol")
    if config.get("profile") != f"TDL-{expected_profile}":
        issues.append("channel profile differs from the frozen protocol")
    if (
        expected_channel_estimator is not None
        and config.get("channel_estimator") != expected_channel_estimator
    ):
        issues.append("channel estimator differs from the frozen protocol")
    expected_configuration = {
        "delay_spread_s": 100e-9,
        "carrier_frequency_hz": 3.5e9,
        "speed_mps_mobile": 120.0 / 3.6,
        "resource_blocks": 12,
        "subcarrier_spacing_khz": 30,
        "configured_receive_grid_snr_db": 18.0,
        "frames_per_condition": len(expected_seeds),
        "batch_size": 1,
        "device": "cpu",
    }
    for field, expected in expected_configuration.items():
        if config.get(field) != expected:
            issues.append(f"configuration differs from protocol: {field}")
    if config.get("conditions") != {
        "static_position_0": {"speed_mps": 0.0, "dmrs_additional_position": 0},
        "mobile_position_0": {"speed_mps": 120.0 / 3.6, "dmrs_additional_position": 0},
        "mobile_position_3": {"speed_mps": 120.0 / 3.6, "dmrs_additional_position": 3},
    }:
        issues.append("condition matrix differs from the frozen protocol")

    protocol_file = root / protocol_path
    protocol_bytes = protocol_file.read_bytes() if protocol_file.is_file() else b""
    if report.get("protocol_sha256") != _sha256(protocol_bytes):
        issues.append("protocol hash differs from the checked-in protocol")
    if report.get("protocol_path") != protocol_path.as_posix():
        issues.append("protocol path is unexpected")

    source_hashes = report.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != expected_source_paths:
        issues.append("source inventory is incomplete or has unexpected paths")
        source_hashes = source_hashes if isinstance(source_hashes, dict) else {}
    for relative, expected_hash in source_hashes.items():
        source_path = root / relative
        if not source_path.is_file() or _sha256(source_path.read_bytes()) != expected_hash:
            issues.append(f"source hash mismatch: {relative}")
    if report.get("source_inventory_sha256") != _canonical_sha256(source_hashes):
        issues.append("source inventory hash is invalid")
    source_integrity_issues = (
        "source inventory is incomplete or has unexpected paths",
        "source inventory hash is invalid",
        "source hash mismatch:",
    )

    conditions = report.get("conditions")
    names = ("static_position_0", "mobile_position_0", "mobile_position_3")
    if not isinstance(conditions, dict) or set(conditions) != set(names):
        issues.append("condition results are incomplete")
        conditions = {}
    outcomes: dict[str, list[bool]] = {}
    goodput: dict[str, list[float]] = {}
    check_rows: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        condition = conditions.get(name)
        if not isinstance(condition, dict):
            issues.append(f"{name} result is missing")
            continue
        run_config = condition.get("configuration")
        point = condition.get("point")
        checks = condition.get("matched_realization_checks")
        if not isinstance(run_config, dict) or not isinstance(point, dict):
            issues.append(f"{name} configuration or point summary is missing")
            continue
        if (
            run_config.get("allow_cp_overrun") is not False
            or run_config.get("cp_covers_channel_memory") is not True
        ):
            issues.append(f"{name} does not prove CP covers channel memory")
        if run_config.get("channel_memory_samples", 0) > run_config.get(
            "cyclic_prefix_min_samples", -1
        ):
            issues.append(f"{name} channel memory exceeds the minimum CP")
        if (
            expected_channel_estimator is not None
            and run_config.get("channel_estimator") != expected_channel_estimator
        ):
            issues.append(f"{name} estimator differs from the frozen protocol")
        errors = point.get("frame_error_mask")
        if (
            not isinstance(errors, list)
            or len(errors) != len(expected_seeds)
            or not all(type(value) is bool for value in errors)
        ):
            issues.append(f"{name} lacks complete Boolean CRC outcomes")
            continue
        tbs = point.get("mean_transmitted_tb_bits_per_codeword")
        if isinstance(tbs, bool) or not isinstance(tbs, (int, float)) or tbs <= 0:
            issues.append(f"{name} has an invalid transport-block size")
            continue
        outcomes[name] = errors
        goodput[name] = [0.0 if error else float(tbs) for error in errors]
        failure_count = sum(errors)
        if point.get("frames") != len(expected_seeds):
            issues.append(f"{name} frame count does not match CRC outcomes")
        if point.get("bler_errors") != failure_count:
            issues.append(f"{name} CRC failure count does not match raw outcomes")
        if point.get("bler") != failure_count / len(expected_seeds):
            issues.append(f"{name} BLER does not match raw outcomes")
        interval = point.get("bler_wilson_95_ci")
        expected_interval = _wilson_interval(failure_count, len(expected_seeds))
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(
                not isinstance(value, (int, float)) or abs(float(value) - expected) > 1e-12
                for value, expected in zip(interval, expected_interval, strict=True)
            )
        ):
            issues.append(f"{name} Wilson interval does not match raw outcomes")
        reported_goodput = point.get("successfully_decoded_tb_bits_per_codeword")
        expected_goodput = sum(goodput[name]) / len(expected_seeds)
        if (
            not isinstance(reported_goodput, (int, float))
            or abs(float(reported_goodput) - expected_goodput) > 1e-9
        ):
            issues.append(f"{name} decoded-bit goodput does not match raw outcomes")
        if not isinstance(checks, list) or len(checks) != len(expected_seeds):
            issues.append(f"{name} lacks complete channel-seed records")
            continue
        check_rows[name] = checks
        if [item.get("seed") for item in checks] != expected_seeds:
            issues.append(f"{name} realization seeds differ from frozen seeds")

    channel_records = report.get("paired_channel_outcomes")
    if not isinstance(channel_records, list) or len(channel_records) != len(expected_seeds):
        issues.append("paired per-channel outcomes are missing or incomplete")
        channel_records = []

    if set(outcomes) == set(names) and set(check_rows) == set(names):
        for i, seed in enumerate(expected_seeds):
            static = check_rows["static_position_0"][i]
            mobile_zero = check_rows["mobile_position_0"][i]
            mobile_three = check_rows["mobile_position_3"][i]
            if mobile_zero.get("channel_truth_sha256") != mobile_three.get("channel_truth_sha256"):
                issues.append(f"mobile channel truth differs at seed {seed}")
            if static.get("transmit_bits_sha256") != mobile_zero.get("transmit_bits_sha256"):
                issues.append(f"position-0 payload differs between static/mobile at seed {seed}")
            if static.get("noise_real_seed") != mobile_zero.get("noise_real_seed") or static.get(
                "noise_imag_seed"
            ) != mobile_zero.get("noise_imag_seed"):
                issues.append(f"position-0 noise seeds differ between static/mobile at seed {seed}")
            for key in ("seed", "noise_real_seed", "noise_imag_seed"):
                if mobile_zero.get(key) != mobile_three.get(key):
                    issues.append(f"mobile paired {key} differs at seed {seed}")
            if len(channel_records) == len(expected_seeds):
                row = channel_records[i]
                if row.get("seed") != seed:
                    issues.append(f"paired row has wrong seed at index {i}")
                for name, field in (
                    ("static_position_0", "static_position_0_crc_error"),
                    ("mobile_position_0", "mobile_position_0_crc_error"),
                    ("mobile_position_3", "mobile_position_3_crc_error"),
                ):
                    if outcomes[name][i] != row.get(field):
                        issues.append(f"saved {field} differs at seed {seed}")
                for name, field in (
                    ("mobile_position_0", "mobile_position_0_goodput_bits_per_slot"),
                    ("mobile_position_3", "mobile_position_3_goodput_bits_per_slot"),
                ):
                    if goodput[name][i] != row.get(field):
                        issues.append(f"saved {field} differs at seed {seed}")
                if row.get("paired_bler_reduction") != float(
                    outcomes["mobile_position_0"][i]
                ) - float(outcomes["mobile_position_3"][i]):
                    issues.append(f"saved BLER contrast differs at seed {seed}")
                if (
                    row.get("paired_goodput_increase_bits_per_slot")
                    != goodput["mobile_position_3"][i] - goodput["mobile_position_0"][i]
                ):
                    issues.append(f"saved goodput contrast differs at seed {seed}")
                for name in names:
                    check = check_rows[name][i]
                    expected_truth = row.get(f"{name}_channel_truth_sha256")
                    if (
                        expected_truth is not None
                        and check.get("channel_truth_sha256") != expected_truth
                    ):
                        issues.append(f"saved channel-truth hash differs at seed {seed}: {name}")
                if row.get("mobile_position_0_transmit_bits_sha256") != mobile_zero.get(
                    "transmit_bits_sha256"
                ) or row.get("mobile_position_3_transmit_bits_sha256") != mobile_three.get(
                    "transmit_bits_sha256"
                ):
                    issues.append(f"saved payload hashes differ at seed {seed}")

    acceptance = report.get("acceptance")
    if not isinstance(acceptance, dict) or set(outcomes) != set(names):
        issues.append("acceptance data or all conditions are unavailable")
        passed = False
    else:
        per_claim_alpha = familywise_alpha / family_claim_count
        if acceptance.get("familywise_alpha") != familywise_alpha:
            issues.append("acceptance family-wise alpha differs from protocol")
        if (
            acceptance.get("family_claim_count") != family_claim_count
            or acceptance.get("per_claim_alpha") != per_claim_alpha
        ):
            issues.append("acceptance multiplicity allocation differs from protocol")
        if acceptance.get("bootstrap_replicates") != BOOTSTRAP_REPLICATES:
            issues.append("bootstrap replicate count differs from protocol")
        static_errors = sum(outcomes["static_position_0"])
        static_bler = static_errors / len(expected_seeds)
        static_upper = clopper_pearson_upper_bound(
            static_errors, len(expected_seeds), alpha=per_claim_alpha
        )
        bler_delta = [
            float(sparse) - float(dense)
            for sparse, dense in zip(
                outcomes["mobile_position_0"], outcomes["mobile_position_3"], strict=True
            )
        ]
        bler_ci = cluster_bootstrap_interval(
            bler_delta,
            seed=bootstrap_seeds["bler"],
            alpha=per_claim_alpha,
            replicates=BOOTSTRAP_REPLICATES,
        )
        goodput_delta = [
            dense - sparse
            for sparse, dense in zip(
                goodput["mobile_position_0"], goodput["mobile_position_3"], strict=True
            )
        ]
        goodput_ci = cluster_bootstrap_interval(
            goodput_delta,
            seed=bootstrap_seeds["goodput"],
            alpha=per_claim_alpha,
            replicates=BOOTSTRAP_REPLICATES,
        )
        expected_acceptance: dict[str, dict[str, Any]] = {
            "static_reference": {
                "position_0_bler": static_bler,
                "crc_failures": static_errors,
                "family_adjusted_one_sided_exact_failure_prevalence_upper_bound": static_upper,
            },
            "mobile_bler_reduction_position_0_minus_3": {
                "equal_channel_mean": sum(bler_delta) / len(bler_delta),
                "family_adjusted_paired_cluster_bootstrap_interval": list(bler_ci),
            },
            "mobile_goodput_increase_position_3_minus_0_bits_per_slot": {
                "equal_channel_mean": sum(goodput_delta) / len(goodput_delta),
                "family_adjusted_paired_cluster_bootstrap_interval": list(goodput_ci),
            },
        }
        for claim, fields in expected_acceptance.items():
            recorded = acceptance.get(claim)
            if not isinstance(recorded, dict):
                issues.append(f"acceptance claim missing: {claim}")
                continue
            for field, value in fields.items():
                if recorded.get(field) != value:
                    issues.append(f"acceptance value differs from raw outcomes: {claim}.{field}")
        pass_expectations = {
            "static_reference": static_bler <= 0.05 and static_upper <= 0.10,
            "mobile_bler_reduction_position_0_minus_3": (
                sum(bler_delta) / len(bler_delta) >= 0.10 and bler_ci[0] > 0
            ),
            "mobile_goodput_increase_position_3_minus_0_bits_per_slot": goodput_ci[0] > 0,
        }
        for claim, expected_pass in pass_expectations.items():
            if acceptance.get(claim, {}).get("pass") is not expected_pass:
                issues.append(f"acceptance verdict differs from raw outcomes: {claim}")
        passed = all(
            acceptance.get(claim, {}).get("pass") is True
            for claim in (
                "static_reference",
                "mobile_bler_reduction_position_0_minus_3",
                "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
            )
        )
        expected_status = f"PASS_BOUNDED_TDL_{expected_profile}_FIXTURE" if passed else "NOT_MET"
        if report.get("status") != expected_status:
            issues.append("overall status does not match frozen acceptance claims")
        if report.get("g5_09_closed") is not False:
            issues.append("bounded confirmation must not claim G5.09 closure")

    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "report_sha256": _sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ),
        "source_matches_worktree": not any(
            item.startswith(source_integrity_issues) for item in issues
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=DEFAULT_REPORT)
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = audit_report(report)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
