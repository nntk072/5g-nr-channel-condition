"""Run and retain the frozen G4 CDL dual-TX/dual-RX joint space-time protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / (
    "tests/test_integrations/test_g4_cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_native.py"
)
PROTOCOL = ROOT / (
    "docs/research/"
    "G4_07_CDL_JOINT_TX_RX_DUAL_TX_RX_SPATIOTEMPORAL_COVARIANCE_PROTOCOL_2026-11-10.md"
)
SEED_AUDIT = ROOT / (
    "docs/research/"
    "G4_CDL_JOINT_TX_RX_DUAL_TX_RX_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json"
)
REFERENCE = ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
EVALUATION = ROOT / "evaluation/g4_cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance.py"
PORT_ORDER_TEST = ROOT / "tests/test_integrations/test_g4_cdl_dual_vh_port_order_native.py"
MODELS = tuple("ABCDE")
SEED_START = 2_814_000
CASE_COUNT = 5
CLAIM_COUNT = 2_560
DRAW_COUNT = 16_384
FAMILYWISE_ALPHA = 0.01
MAXIMUM_HALF_WIDTH = 0.05


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sionna_version() -> str:
    for distribution in ("sionna", "sionna-no-rt"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise RuntimeError("neither sionna nor sionna-no-rt distribution is installed")


class _Collector:
    def __init__(self) -> None:
        self.cases: list[dict[str, Any]] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when != "call":
            return
        for key, value in report.user_properties:
            if key == "g4_cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_case":
                case = json.loads(value)
                case["test_nodeid"] = report.nodeid
                case["test_outcome"] = report.outcome
                self.cases.append(case)


def _validate_cases(cases: list[dict[str, Any]]) -> dict[str, float | int | bool]:
    if len(cases) != CASE_COUNT:
        raise ValueError(f"expected {CASE_COUNT} cases, observed {len(cases)}")
    by_model = {case["model"]: case for case in cases}
    if set(by_model) != set(MODELS):
        raise ValueError("profile roster differs from frozen protocol")
    claim_count = 0
    failed_claim_count = 0
    failed_case_count = 0
    maximum_half_width = 0.0
    maximum_absolute_error = 0.0
    for index, model in enumerate(MODELS):
        case = by_model[model]
        if (
            case["seed"] != SEED_START + index
            or case["draw_count"] != DRAW_COUNT
            or case["tx_element_count"] != 2
            or case["rx_element_count"] != 2
            or case["tx_port_count"] != 4
            or case["rx_port_count"] != 4
            or case["channel_port_count"] != 16
            or case["family_claim_count"] != CLAIM_COUNT
            or case["familywise_alpha"] != FAMILYWISE_ALPHA
            or case["test_outcome"] not in {"passed", "failed"}
            or case["batch_axis_semantics"] != "independent_stochastic_channel_draws"
        ):
            raise ValueError(f"frozen design/outcome mismatch for {model}")
        if len(case["claims"]) != 512:
            raise ValueError(f"expected 512 claims for {model}")
        case_failed_claims = 0
        for claim in case["claims"]:
            claim_count += 1
            half_width = float(claim["half_width"])
            absolute_error = float(claim["absolute_error"])
            maximum_half_width = max(maximum_half_width, half_width)
            maximum_absolute_error = max(maximum_absolute_error, absolute_error)
            expected_pass = absolute_error <= half_width and half_width <= MAXIMUM_HALF_WIDTH
            if bool(claim["passed"]) is not expected_pass:
                raise ValueError(f"claim pass arithmetic mismatch for {model}")
            if not expected_pass:
                failed_claim_count += 1
                case_failed_claims += 1
        expected_outcome = "failed" if case_failed_claims else "passed"
        if case["test_outcome"] != expected_outcome:
            raise ValueError(f"pytest outcome does not match retained claims for {model}")
        failed_case_count += bool(case_failed_claims)
    if claim_count != CLAIM_COUNT:
        raise ValueError(f"expected {CLAIM_COUNT} claims, observed {claim_count}")
    return {
        "case_count": len(cases),
        "claim_count": claim_count,
        "failed_case_count": failed_case_count,
        "failed_claim_count": failed_claim_count,
        "passed_claim_count": claim_count - failed_claim_count,
        "all_passed": failed_claim_count == 0,
        "maximum_half_width": maximum_half_width,
        "maximum_absolute_error": maximum_absolute_error,
    }


def _sionna_source_hashes() -> dict[str, str]:
    import sionna.phy.channel.tr38901.antenna as antenna
    import sionna.phy.channel.tr38901.channel_coefficients as coefficients

    return {
        "sionna/phy/channel/tr38901/antenna.py": _sha256(Path(antenna.__file__).resolve()),
        "sionna/phy/channel/tr38901/channel_coefficients.py": _sha256(
            Path(coefficients.__file__).resolve()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    sidecar = Path(str(output) + ".sha256")
    if output.exists() or sidecar.exists():
        parser.error("refusing to overwrite retained result or checksum")
    if subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    ):
        parser.error("refusing to run from a dirty worktree")
    seed_audit = json.loads(SEED_AUDIT.read_text(encoding="utf-8"))
    expected_seeds = list(range(SEED_START, SEED_START + CASE_COUNT))
    if (
        seed_audit.get("status") != "CLEAR_WITHIN_SCANNED_JSON_CORPUS"
        or seed_audit.get("reserved_seeds") != expected_seeds
        or seed_audit.get("collisions")
        or seed_audit.get("malformed_seed_fields")
        or seed_audit.get("incomplete_files")
    ):
        parser.error("fresh seed audit does not clear the frozen roster")

    collector = _Collector()
    exit_code = int(
        pytest.main(
            ["--no-cov", "-o", "addopts=", "-q", "-m", "native_sionna", str(TEST)],
            plugins=[collector],
        )
    )
    if exit_code not in (0, 1):
        parser.error(f"native pytest execution was incomplete with exit code {exit_code}")
    summary = _validate_cases(collector.cases)
    all_passed = bool(summary["all_passed"])
    if (exit_code == 0) is not all_passed:
        parser.error("pytest exit status does not match retained confidence claims")
    cases = sorted(collector.cases, key=lambda case: MODELS.index(case["model"]))
    report = {
        "schema": "loslab.g4_cdl_joint_tx_rx_dual_tx_rx_spatiotemporal_covariance_native_validation.v1",
        "date": "2026-11-10",
        "base_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip(),
        "status": (
            "PASS_WITHIN_BOUNDED_PROTOCOL_SCOPE"
            if all_passed
            else "FROZEN_FAILURE_WITHIN_BOUNDED_PROTOCOL_SCOPE"
        ),
        "all_passed": all_passed,
        "criteria_changed": False,
        "pytest_exit_code": exit_code,
        "profiles": list(MODELS),
        "velocity_mps": [10.0, 20.0, 20.0],
        "tx_element_count": 2,
        "rx_element_count": 2,
        "tx_port_count": 4,
        "rx_port_count": 4,
        "channel_port_count": 16,
        "draws_per_case": DRAW_COUNT,
        "case_count": CASE_COUNT,
        "simultaneous_claim_count": CLAIM_COUNT,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "maximum_interval_half_width": MAXIMUM_HALF_WIDTH,
        "maximum_observed_half_width": summary["maximum_half_width"],
        "maximum_observed_absolute_error": summary["maximum_absolute_error"],
        "failed_case_count": summary["failed_case_count"],
        "failed_claim_count": summary["failed_claim_count"],
        "passed_claim_count": summary["passed_claim_count"],
        "seed_audit": {
            "path": str(SEED_AUDIT.relative_to(ROOT)),
            "sha256": _sha256(SEED_AUDIT),
            "status": seed_audit["status"],
            "scanned_file_count": seed_audit["scanned_file_count"],
            "global_exclusivity_established": False,
        },
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": importlib.metadata.version("torch"),
            "sionna": _sionna_version(),
            "device": "cpu",
        },
        "sionna_source_hashes": _sionna_source_hashes(),
        "source_hashes": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                EVALUATION,
                TEST,
                PORT_ORDER_TEST,
                PROTOCOL,
                SEED_AUDIT,
                REFERENCE,
                Path(__file__).resolve(),
            )
        },
        "cases": cases,
        "qualification_claim": "none",
        "limitations": [
            "Bounded Sionna PHY 2.2.0 CPU model-consistency evidence for the frozen dual-TX/dual-RX joint space-time covariance family only.",
            "The family covers CDL-A through E, one oblique 30 m/s velocity, one 3.5 GHz carrier, one 0.7 ms lag, and two physical dual-VH elements at each zero-oriented endpoint.",
            "It does not establish arbitrary arrays or patterns, continuous orientation or mobility coverage, mutual coupling, measured-channel truth, RT/material truth, receiver validity, general 3GPP conformance, or overall G4 qualification.",
            "The seed audit establishes no global seed exclusivity.",
            "Production D1 generation and advanced training remain gated.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest, **summary}, sort_keys=True))
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
