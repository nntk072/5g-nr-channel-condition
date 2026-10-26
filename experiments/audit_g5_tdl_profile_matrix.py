"""Audit retained outcomes from the G5 TDL-A–E DM-RS family confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = Path("docs/research/results/g5-tdl-ae-dmrs-density-family-confirm-20261024.json")
PROTOCOL = Path("docs/research/G5_TDL_AE_DMRS_DENSITY_FAMILY_CONFIRMATORY_PROTOCOL_2026-10-24.md")
REPLICATION_REPORT = Path(
    "docs/research/results/g5-tdl-ae-dmrs-density-family-reconfirm-20261024.json"
)
REPLICATION_PROTOCOL = Path(
    "docs/research/G5_TDL_AE_DMRS_DENSITY_FAMILY_REPLICATION_PROTOCOL_2026-10-24.md"
)
REPLICATION_V2_PROTOCOL = Path(
    "docs/research/G5_TDL_AE_DMRS_DENSITY_FAMILY_REPLICATION_V2_PROTOCOL_2026-10-24.md"
)
PROFILES = ("A", "B", "C", "D", "E")
SEEDS = list(range(934_001, 934_065))
FAMILY_ALPHA = 0.01
CLAIM_COUNT = 15
ALPHA = FAMILY_ALPHA / CLAIM_COUNT
REPLICATION_SEEDS = list(range(935_001, 935_081))
REPLICATION_V2_SEEDS = list(range(936_001, 936_081))
BOOTSTRAP_REPLICATES = 20_000
SOURCE_PATHS = {
    "experiments/qualify_g5_tdl_c_dmrs_density.py",
    "experiments/qualify_g5_tdl_profile_matrix.py",
    "experiments/g5_independent_channel_receiver.py",
    "experiments/link_curves.py",
    "integrations/sionna/nr_pusch.py",
    "integrations/sionna/tdl_pusch.py",
    "receivers/nr/pusch.py",
    "evaluation/statistics.py",
    PROTOCOL.as_posix(),
}
CONDITIONS = ("static_position_0", "mobile_position_0", "mobile_position_3")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(raw.encode())


def _same_number(actual: Any, expected: float) -> bool:
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
    )


def _bootstrap_interval(
    values: list[float],
    *,
    seed: int,
    alpha: float = ALPHA,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[float, float]:
    rng = random.Random(seed)
    count = len(values)
    estimates = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(replicates)
    )
    tail = alpha / 2.0
    low = estimates[math.floor(tail * replicates)]
    high = estimates[
        min(
            replicates - 1,
            math.ceil((1.0 - tail) * replicates) - 1,
        )
    ]
    return low, high


def _exact_upper(failures: int, count: int, *, alpha: float = ALPHA) -> float:
    if failures == count:
        return 1.0
    return float(beta.ppf(1.0 - alpha, failures + 1, count - failures))


def _audit_profile(
    profile_doc: dict[str, Any],
    *,
    profile: str,
    seeds: list[int] = SEEDS,
    family_alpha: float = FAMILY_ALPHA,
    claim_count: int = CLAIM_COUNT,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> list[str]:
    issues: list[str] = []
    alpha = family_alpha / claim_count
    recorded_digest = profile_doc.get("result_sha256")
    unhashed = dict(profile_doc)
    unhashed.pop("result_sha256", None)
    if recorded_digest != _canonical_sha256(unhashed):
        issues.append(f"TDL-{profile} profile result digest mismatch")
    config = profile_doc.get("configuration")
    if not isinstance(config, dict):
        issues.append(f"TDL-{profile} configuration missing")
        return issues
    if profile_doc.get("configuration_sha256") != _canonical_sha256(config):
        issues.append(f"TDL-{profile} configuration digest mismatch")
    expected_config = {
        "profile": f"TDL-{profile}",
        "delay_spread_s": 100e-9,
        "carrier_frequency_hz": 3.5e9,
        "speed_mps_mobile": 120.0 / 3.6,
        "resource_blocks": 12,
        "subcarrier_spacing_khz": 30,
        "configured_receive_grid_snr_db": 18.0,
        "independent_channel_seeds": seeds,
        "frames_per_condition": len(seeds),
        "batch_size": 1,
        "device": "cpu",
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            issues.append(f"TDL-{profile} configuration differs from protocol: {key}")
    expected_conditions = {
        "static_position_0": {"speed_mps": 0.0, "dmrs_additional_position": 0},
        "mobile_position_0": {"speed_mps": 120.0 / 3.6, "dmrs_additional_position": 0},
        "mobile_position_3": {"speed_mps": 120.0 / 3.6, "dmrs_additional_position": 3},
    }
    if config.get("conditions") != expected_conditions:
        issues.append(f"TDL-{profile} condition matrix differs from protocol")
    cp_checks = config.get("cp_coverage_by_condition")
    if not isinstance(cp_checks, dict) or set(cp_checks) != set(CONDITIONS):
        issues.append(f"TDL-{profile} CP coverage records are incomplete")
    elif any(item.get("cp_covers_channel_memory") is not True for item in cp_checks.values()):
        issues.append(f"TDL-{profile} has a cyclic-prefix coverage failure")

    conditions = profile_doc.get("conditions")
    if not isinstance(conditions, dict) or set(conditions) != set(CONDITIONS):
        issues.append(f"TDL-{profile} condition records are incomplete")
        return issues
    points: dict[str, dict[str, Any]] = {}
    checks: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        record = conditions[condition]
        if not isinstance(record, dict):
            issues.append(f"TDL-{profile} {condition} record malformed")
            continue
        run_config = record.get("configuration")
        point = record.get("point")
        rows = record.get("matched_realization_checks")
        if not isinstance(run_config, dict) or not isinstance(point, dict):
            issues.append(f"TDL-{profile} {condition} config/point missing")
            continue
        if (
            run_config.get("profile") != f"TDL-{profile}"
            or run_config.get("allow_cp_overrun") is not False
            or run_config.get("cp_covers_channel_memory") is not True
        ):
            issues.append(f"TDL-{profile} {condition} run config fails profile/CP checks")
        if not isinstance(rows, list) or len(rows) != len(seeds):
            issues.append(f"TDL-{profile} {condition} realization checks incomplete")
            continue
        if [row.get("seed") for row in rows] != seeds:
            issues.append(f"TDL-{profile} {condition} seed order differs from protocol")
        if point.get("snr_db") != 18.0:
            issues.append(f"TDL-{profile} {condition} SNR differs from protocol")
        errors = point.get("frame_error_mask")
        if (
            not isinstance(errors, list)
            or len(errors) != len(seeds)
            or not all(type(value) is bool for value in errors)
        ):
            issues.append(f"TDL-{profile} {condition} CRC outcomes malformed")
            continue
        points[condition] = point
        checks[condition] = rows

    if set(points) != set(CONDITIONS) or set(checks) != set(CONDITIONS):
        return issues
    for low, high in zip(checks["mobile_position_0"], checks["mobile_position_3"], strict=True):
        if any(
            low.get(key) != high.get(key)
            for key in ("seed", "channel_truth_sha256", "noise_real_seed", "noise_imag_seed")
        ):
            issues.append(f"TDL-{profile} mobile DM-RS conditions are not seed/truth/noise paired")
            break
    for static, mobile in zip(
        checks["static_position_0"], checks["mobile_position_0"], strict=True
    ):
        if any(
            static.get(key) != mobile.get(key)
            for key in ("transmit_bits_sha256", "noise_real_seed", "noise_imag_seed")
        ):
            issues.append(f"TDL-{profile} static/mobile position-0 pairing is invalid")
            break

    outcomes = profile_doc.get("paired_channel_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != len(seeds):
        issues.append(f"TDL-{profile} retained paired outcomes are incomplete")
        return issues
    static_errors = points["static_position_0"]["frame_error_mask"]
    mobile_zero = points["mobile_position_0"]["frame_error_mask"]
    mobile_three = points["mobile_position_3"]["frame_error_mask"]
    static_tbs = points["mobile_position_0"].get("mean_transmitted_tb_bits_per_codeword")
    dense_tbs = points["mobile_position_3"].get("mean_transmitted_tb_bits_per_codeword")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
        for value in (static_tbs, dense_tbs)
    ):
        issues.append(f"TDL-{profile} transport-block sizes are invalid")
        return issues
    assert isinstance(static_tbs, (int, float)) and not isinstance(static_tbs, bool)
    assert isinstance(dense_tbs, (int, float)) and not isinstance(dense_tbs, bool)
    bler_differences: list[float] = []
    goodput_differences: list[float] = []
    for index, (seed, row) in enumerate(zip(seeds, outcomes, strict=True)):
        if row.get("seed") != seed:
            issues.append(f"TDL-{profile} retained seed {index} is out of order")
            break
        expected = {
            "static_position_0_crc_error": static_errors[index],
            "mobile_position_0_crc_error": mobile_zero[index],
            "mobile_position_3_crc_error": mobile_three[index],
            "mobile_position_0_goodput_bits_per_slot": 0.0
            if mobile_zero[index]
            else float(static_tbs),
            "mobile_position_3_goodput_bits_per_slot": 0.0
            if mobile_three[index]
            else float(dense_tbs),
        }
        for key, value in expected.items():
            if row.get(key) != value:
                issues.append(f"TDL-{profile} retained outcome mismatch: {key} seed {seed}")
        difference = float(mobile_zero[index]) - float(mobile_three[index])
        goodput_difference = (
            expected["mobile_position_3_goodput_bits_per_slot"]
            - expected["mobile_position_0_goodput_bits_per_slot"]
        )
        if row.get("paired_bler_reduction") != difference:
            issues.append(f"TDL-{profile} paired BLER reduction mismatch at seed {seed}")
        if row.get("paired_goodput_increase_bits_per_slot") != goodput_difference:
            issues.append(f"TDL-{profile} paired goodput mismatch at seed {seed}")
        bler_differences.append(difference)
        goodput_differences.append(goodput_difference)

    acceptance = profile_doc.get("acceptance")
    if not isinstance(acceptance, dict):
        issues.append(f"TDL-{profile} acceptance block missing")
        return issues
    if (
        acceptance.get("familywise_alpha") != family_alpha
        or acceptance.get("family_claim_count") != claim_count
        or acceptance.get("per_claim_alpha") != alpha
        or acceptance.get("bootstrap_replicates") != bootstrap_replicates
    ):
        issues.append(f"TDL-{profile} family allocation differs from protocol")
    static_count = sum(static_errors)
    static_upper = _exact_upper(static_count, len(seeds), alpha=alpha)
    static_pass = static_count / len(seeds) <= 0.05 and static_upper <= 0.10
    static_claim = acceptance.get("static_reference", {})
    if (
        static_claim.get("crc_failures") != static_count
        or not _same_number(static_claim.get("position_0_bler"), static_count / len(seeds))
        or not _same_number(
            static_claim.get("family_adjusted_one_sided_exact_failure_prevalence_upper_bound"),
            static_upper,
        )
        or static_claim.get("pass") is not static_pass
    ):
        issues.append(f"TDL-{profile} static-reference acceptance math mismatch")

    bootstrap_bler_seed = acceptance.get("paired_bler_bootstrap_seed")
    bootstrap_goodput_seed = acceptance.get("paired_goodput_bootstrap_seed")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        for seed in (bootstrap_bler_seed, bootstrap_goodput_seed)
    ):
        issues.append(f"TDL-{profile} bootstrap seeds missing or invalid")
        return issues
    assert isinstance(bootstrap_bler_seed, int) and not isinstance(bootstrap_bler_seed, bool)
    assert isinstance(bootstrap_goodput_seed, int) and not isinstance(bootstrap_goodput_seed, bool)
    bler_interval = _bootstrap_interval(
        bler_differences,
        seed=bootstrap_bler_seed,
        alpha=alpha,
        replicates=bootstrap_replicates,
    )
    goodput_interval = _bootstrap_interval(
        goodput_differences,
        seed=bootstrap_goodput_seed,
        alpha=alpha,
        replicates=bootstrap_replicates,
    )
    bler_mean = sum(bler_differences) / len(bler_differences)
    goodput_mean = sum(goodput_differences) / len(goodput_differences)
    bler_claim = acceptance.get("mobile_bler_reduction_position_0_minus_3", {})
    goodput_claim = acceptance.get("mobile_goodput_increase_position_3_minus_0_bits_per_slot", {})
    expected_bler_pass = bler_mean >= 0.10 and bler_interval[0] > 0.0
    expected_goodput_pass = goodput_interval[0] > 0.0
    if (
        not _same_number(bler_claim.get("equal_channel_mean"), bler_mean)
        or bler_claim.get("family_adjusted_paired_cluster_bootstrap_interval")
        != list(bler_interval)
        or bler_claim.get("pass") is not expected_bler_pass
    ):
        issues.append(f"TDL-{profile} mobile BLER acceptance math mismatch")
    if (
        not _same_number(goodput_claim.get("equal_channel_mean"), goodput_mean)
        or goodput_claim.get("family_adjusted_paired_cluster_bootstrap_interval")
        != list(goodput_interval)
        or goodput_claim.get("pass") is not expected_goodput_pass
    ):
        issues.append(f"TDL-{profile} mobile goodput acceptance math mismatch")
    profile_pass = static_pass and expected_bler_pass and expected_goodput_pass
    expected_status = f"PASS_BOUNDED_TDL_{profile}_FIXTURE" if profile_pass else "NOT_MET"
    if profile_doc.get("status") != expected_status or profile_doc.get("g5_09_closed") is not False:
        issues.append(f"TDL-{profile} verdict or gate boundary mismatch")
    return issues


def audit_report(report: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Verify result and source digests, each profile, and the family verdict."""
    issues: list[str] = []
    is_replication = report.get("schema") == "umcd.g5_tdl_ae_dmrs_density_family_replication.v1"
    is_replication_v2 = report.get("schema") == "umcd.g5_tdl_ae_dmrs_density_family_replication.v2"
    expected_schema = (
        "umcd.g5_tdl_ae_dmrs_density_family_replication.v2"
        if is_replication_v2
        else "umcd.g5_tdl_ae_dmrs_density_family_replication.v1"
        if is_replication
        else "umcd.g5_tdl_ae_dmrs_density_family_confirmatory.v1"
    )
    seeds = (
        REPLICATION_V2_SEEDS
        if is_replication_v2
        else REPLICATION_SEEDS
        if is_replication
        else SEEDS
    )
    protocol = (
        REPLICATION_V2_PROTOCOL
        if is_replication_v2
        else REPLICATION_PROTOCOL
        if is_replication
        else PROTOCOL
    )
    claim_count = CLAIM_COUNT
    alpha = FAMILY_ALPHA / claim_count
    source_paths = (
        (SOURCE_PATHS - {PROTOCOL.as_posix()} - {"experiments/qualify_g5_tdl_profile_matrix.py"})
        | {protocol.as_posix(), "experiments/qualify_g5_tdl_profile_matrix_replication.py"}
        if is_replication or is_replication_v2
        else SOURCE_PATHS
    )
    unhashed = dict(report)
    recorded_result = unhashed.pop("result_sha256", None)
    if recorded_result != _canonical_sha256(unhashed):
        issues.append("family result digest mismatch")
    if report.get("schema") != expected_schema:
        issues.append("unexpected family report schema")
    if (
        report.get("familywise_alpha") != FAMILY_ALPHA
        or report.get("family_claim_count") != claim_count
    ):
        issues.append("family alpha or claim count differs from protocol")
    if report.get("per_claim_alpha_bonferroni") != alpha:
        issues.append("family per-claim alpha differs from protocol")
    if report.get("profiles") != list(PROFILES):
        issues.append("profile order or roster differs from protocol")
    if report.get("independent_channel_seeds_per_profile") != seeds:
        issues.append("seed roster differs from protocol")
    if (is_replication or is_replication_v2) and report.get(
        "independent_of_prior_64_seed_family"
    ) is not True:
        issues.append("replication independence declaration missing")
    if is_replication_v2 and report.get("invalid_runner_defect_run") != (
        "docs/research/results/g5-tdl-ae-dmrs-density-family-reconfirm-20261024.json"
    ):
        issues.append("revision-2 invalid-run disclosure missing")
    if report.get("g5_09_closed") is not False:
        issues.append("report incorrectly claims G5.09 closure")

    protocol_path = root / protocol
    protocol_bytes = protocol_path.read_bytes() if protocol_path.is_file() else b""
    if report.get("protocol_path") != protocol.as_posix() or report.get(
        "protocol_sha256"
    ) != _sha256(protocol_bytes):
        issues.append("protocol path or digest mismatch")
    source_hashes = report.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != source_paths:
        issues.append("source inventory is incomplete or unexpected")
        source_hashes = source_hashes if isinstance(source_hashes, dict) else {}
    for path, digest in source_hashes.items():
        candidate = root / path
        if not candidate.is_file() or _sha256(candidate.read_bytes()) != digest:
            issues.append(f"source hash mismatch: {path}")
    if report.get("source_inventory_sha256") != _canonical_sha256(source_hashes):
        issues.append("source inventory digest mismatch")

    summaries = report.get("profile_summaries")
    if not isinstance(summaries, list) or len(summaries) != len(PROFILES):
        issues.append("profile summaries incomplete")
        summaries = []
    profile_passes: list[bool] = []
    profile_configs: list[dict[str, Any]] = []
    for index, profile in enumerate(PROFILES):
        if index >= len(summaries) or not isinstance(summaries[index], dict):
            issues.append(f"TDL-{profile} summary missing")
            continue
        summary = summaries[index]
        profile_doc = summary.get("profile_report")
        if not isinstance(profile_doc, dict):
            issues.append(f"TDL-{profile} full profile report missing")
            continue
        if (
            summary.get("profile") != f"TDL-{profile}"
            or summary.get("status") != profile_doc.get("status")
            or summary.get("profile_result_sha256") != profile_doc.get("result_sha256")
        ):
            issues.append(f"TDL-{profile} wrapper binding mismatch")
        issues.extend(
            _audit_profile(
                profile_doc,
                profile=profile,
                seeds=seeds,
                family_alpha=FAMILY_ALPHA,
                claim_count=claim_count,
            )
        )
        profile_passes.append(profile_doc.get("status") == f"PASS_BOUNDED_TDL_{profile}_FIXTURE")
        config = profile_doc.get("configuration")
        if isinstance(config, dict):
            profile_configs.append(config)

    expected_configuration_digest = _canonical_sha256(
        {
            "profiles": list(PROFILES),
            "channel_seeds": seeds,
            "family_claim_count": claim_count,
            "per_claim_alpha": alpha,
            "profile_configurations": profile_configs,
        }
    )
    if report.get("configuration_sha256") != expected_configuration_digest:
        issues.append("family configuration digest mismatch")
    expected_status = (
        "PASS_BOUNDED_TDL_AE_FIXTURE"
        if len(profile_passes) == len(PROFILES) and all(profile_passes)
        else "NOT_MET"
    )
    if report.get("status") != expected_status:
        issues.append("family verdict does not match per-profile claims")
    source_matches = not any(issue.startswith("source hash mismatch:") for issue in issues)
    return {
        "schema": "umcd.g5_tdl_ae_dmrs_density_family_audit.v1",
        "status": "PASS_AUDIT" if not issues else "FAILED_AUDIT",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "g5_09_closed": False,
        "source_matches_worktree": source_matches,
        "profile_count": len(profile_passes),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = audit_report(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS_AUDIT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
