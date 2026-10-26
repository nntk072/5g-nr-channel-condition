"""Audit retained outcomes from the signed-lag DFT TDL-A–E family study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.audit_g5_tdl_c_dmrs_density import ROOT, _canonical_sha256
from experiments.audit_g5_tdl_dft_density import EXPECTED_ESTIMATOR, audit_dft_report

PROTOCOL = Path("docs/research/G5_TDL_AE_SIGNED_LAG_DFT_FAMILY_CONFIRMATORY_PROTOCOL_2026-10-24.md")
DEFAULT_REPORT = Path("docs/research/results/g5-tdl-ae-signed-lag-dft-family-confirm-20261024.json")
PROFILES = ("A", "B", "C", "D", "E")
EXPECTED_SEEDS = list(range(949_001, 949_081))
FAMILYWISE_ALPHA = 0.01
FAMILY_CLAIM_COUNT = 15
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / FAMILY_CLAIM_COUNT
BOOTSTRAP_SEED_BASE = 949_101
EXPECTED_SOURCE_PATHS = {
    "experiments/qualify_g5_tdl_c_dmrs_density.py",
    "experiments/g5_independent_channel_receiver.py",
    "experiments/link_curves.py",
    "integrations/sionna/nr_pusch.py",
    "integrations/sionna/tdl_pusch.py",
    "receivers/nr/pusch.py",
    "evaluation/statistics.py",
    "experiments/qualify_g5_tdl_dft_profile_matrix.py",
    "experiments/g5_cp_bounded_dft_estimator.py",
    PROTOCOL.as_posix(),
}
EXPECTED_CLAIMS = (
    "static_reference",
    "mobile_bler_reduction_position_0_minus_3",
    "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_family_report(report: dict[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Audit envelope, family allocation, every profile, and all raw outcomes."""
    issues: list[str] = []
    unhashed = dict(report)
    recorded_result_hash = unhashed.pop("result_sha256", None)
    if recorded_result_hash != _canonical_sha256(unhashed):
        issues.append("family result_sha256 does not match report contents")
    protocol_file = root / PROTOCOL
    try:
        protocol_digest = _sha256(protocol_file.read_bytes())
    except OSError:
        protocol_digest = ""
        issues.append("frozen family protocol is unavailable")
    if report.get("schema") != "umcd.g5_tdl_ae_signed_lag_dft_family_confirmatory.v1":
        issues.append("family schema identifier is invalid")
    if report.get("protocol_path") != PROTOCOL.as_posix():
        issues.append("family protocol path differs from the frozen protocol")
    if report.get("protocol_sha256") != protocol_digest:
        issues.append("family protocol digest differs from frozen protocol bytes")
    if report.get("familywise_alpha") != FAMILYWISE_ALPHA:
        issues.append("family-wise alpha differs from protocol")
    if report.get("family_claim_count") != FAMILY_CLAIM_COUNT:
        issues.append("family claim count differs from protocol")
    if report.get("per_claim_alpha_bonferroni") != PER_CLAIM_ALPHA:
        issues.append("per-claim alpha differs from protocol")
    if report.get("profiles") != list(PROFILES):
        issues.append("profile roster differs from protocol")
    if report.get("independent_channel_seeds_per_profile") != EXPECTED_SEEDS:
        issues.append("channel seed roster differs from protocol")
    if report.get("conditions_per_profile") != 3:
        issues.append("condition count differs from protocol")
    if report.get("channel_estimator") != EXPECTED_ESTIMATOR:
        issues.append("family estimator differs from protocol")
    if report.get("qualification_status") != "NOT_QUALIFIED":
        issues.append("bounded family report cannot qualify G5")
    if report.get("admission_possible") is not False or report.get("g5_09_closed") is not False:
        issues.append("bounded family report must not authorize admission or close G5.09")

    source_hashes = report.get("source_sha256")
    if not isinstance(source_hashes, dict):
        source_hashes = {}
        issues.append("family source inventory is missing")
    if report.get("source_inventory_sha256") != _canonical_sha256(source_hashes):
        issues.append("family source inventory digest mismatch")
    profile_summaries = report.get("profile_summaries")
    if not isinstance(profile_summaries, list) or len(profile_summaries) != len(PROFILES):
        issues.append("profile summary roster is incomplete")
        profile_summaries = []
    profile_statuses: dict[str, str] = {}
    profile_results: list[dict[str, Any]] = []
    for index, profile in enumerate(PROFILES):
        if index >= len(profile_summaries):
            continue
        summary = profile_summaries[index]
        expected_profile = f"TDL-{profile}"
        if not isinstance(summary, dict) or summary.get("profile") != expected_profile:
            issues.append(f"profile summary {index} is missing or out of order")
            continue
        profile_report = summary.get("profile_report")
        if not isinstance(profile_report, dict):
            issues.append(f"{expected_profile} retained report is missing")
            continue
        if summary.get("profile_result_sha256") != profile_report.get("result_sha256"):
            issues.append(f"{expected_profile} result digest does not match its summary")
        if summary.get("status") != profile_report.get("status"):
            issues.append(f"{expected_profile} status does not match its retained report")
        profile_statuses[expected_profile] = str(profile_report.get("status"))
        profile_results.append(profile_report)
        bootstrap_seeds = {
            "bler": BOOTSTRAP_SEED_BASE + 2 * index,
            "goodput": BOOTSTRAP_SEED_BASE + 2 * index + 1,
        }
        result = audit_dft_report(
            profile_report,
            root=root,
            protocol_path=PROTOCOL,
            expected_seeds=EXPECTED_SEEDS,
            expected_profile=profile,
            expected_source_paths=EXPECTED_SOURCE_PATHS,
            bootstrap_seeds=bootstrap_seeds,
            family_claim_count=FAMILY_CLAIM_COUNT,
        )
        if result["status"] != "PASS":
            issues.extend(f"{expected_profile}: {item}" for item in result["issues"])
        if profile_report.get("source_sha256") != source_hashes:
            issues.append(f"{expected_profile} source inventory differs from family inventory")

    passed = len(profile_results) == len(PROFILES) and all(
        all(
            profile_report.get("acceptance", {}).get(claim, {}).get("pass") is True
            for claim in EXPECTED_CLAIMS
        )
        for profile_report in profile_results
    )
    expected_status = "PASS_BOUNDED_TDL_AE_FIXTURE" if passed else "NOT_MET"
    if report.get("status") != expected_status:
        issues.append("family status does not match all 15 recomputed claims")
    if report.get("configuration_sha256") != _canonical_sha256(
        {
            "profiles": list(PROFILES),
            "channel_seeds": EXPECTED_SEEDS,
            "family_claim_count": FAMILY_CLAIM_COUNT,
            "per_claim_alpha": PER_CLAIM_ALPHA,
            "channel_estimator": EXPECTED_ESTIMATOR,
            "profile_configurations": [item.get("configuration") for item in profile_results],
        }
    ):
        issues.append("family configuration digest mismatch")
    return {
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "profile_statuses": profile_statuses,
        "source_matches_worktree": not any(
            "source hash" in issue or "source inventory" in issue for issue in issues
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=DEFAULT_REPORT)
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = audit_family_report(report)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
