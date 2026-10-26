"""Audit the preregistered TDL-C signed-lag DFT confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.audit_g5_tdl_c_dmrs_density import ROOT, audit_report

PROTOCOL = Path("docs/research/G5_TDL_C_SIGNED_LAG_DFT_CONFIRMATORY_PROTOCOL_2026-10-24.md")
DEFAULT_REPORT = Path("docs/research/results/g5-tdl-c-signed-lag-dft-confirm-20261024.json")
EXPECTED_SEEDS = list(range(948_001, 948_081))
BOOTSTRAP_SEEDS = {"bler": 948_099, "goodput": 948_100}
FAMILYWISE_ALPHA = 0.01
EXPECTED_ESTIMATOR = {
    "kind": "finite_delay_dft",
    "min_delay_samples": -6,
    "max_delay_samples": 7,
}
EXPECTED_SOURCE_PATHS = {
    "experiments/qualify_g5_tdl_c_dmrs_density.py",
    "experiments/g5_independent_channel_receiver.py",
    "experiments/link_curves.py",
    "integrations/sionna/nr_pusch.py",
    "integrations/sionna/tdl_pusch.py",
    "receivers/nr/pusch.py",
    "evaluation/statistics.py",
    "experiments/qualify_g5_tdl_dft_density.py",
    "experiments/g5_cp_bounded_dft_estimator.py",
    PROTOCOL.as_posix(),
}


def audit_dft_report(
    report: dict[str, Any],
    *,
    root: Path = ROOT,
    protocol_path: Path = PROTOCOL,
    expected_seeds: list[int] = EXPECTED_SEEDS,
    expected_profile: str = "C",
    expected_source_paths: set[str] = EXPECTED_SOURCE_PATHS,
    bootstrap_seeds: dict[str, int] = BOOTSTRAP_SEEDS,
    family_claim_count: int = 3,
) -> dict[str, Any]:
    """Recompute the paired result and enforce DFT-specific provenance."""
    return audit_report(
        report,
        root=root,
        protocol_path=protocol_path,
        expected_seeds=expected_seeds,
        expected_profile=expected_profile,
        expected_source_paths=expected_source_paths,
        expected_channel_estimator=EXPECTED_ESTIMATOR,
        bootstrap_seeds=bootstrap_seeds,
        familywise_alpha=FAMILYWISE_ALPHA,
        family_claim_count=family_claim_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=DEFAULT_REPORT)
    args = parser.parse_args()
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = audit_dft_report(report)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
