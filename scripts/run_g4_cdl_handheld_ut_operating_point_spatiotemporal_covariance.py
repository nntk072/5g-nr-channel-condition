"""Run and retain the frozen physical-handheld CDL operating-point protocol."""

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
TEST = (
    ROOT
    / "tests/test_integrations/test_g4_cdl_handheld_ut_operating_point_spatiotemporal_covariance_native.py"
)
ORACLE_NATIVE_TEST = ROOT / "tests/test_integrations/test_g4_cdl_handheld_ut_oracle_native.py"
ORACLE_PORTABLE_TEST = (
    ROOT / "tests/test_evaluation/test_g4_cdl_handheld_ut_spatiotemporal_covariance.py"
)
PROTOCOL = (
    ROOT
    / "docs/research/G4_09_CDL_PHYSICAL_HANDHELD_UT_OPERATING_POINT_SPATIOTEMPORAL_COVARIANCE_PROTOCOL_2026-11-10.md"
)
SEED_AUDIT = (
    ROOT
    / "docs/research/G4_CDL_PHYSICAL_HANDHELD_UT_OPERATING_POINT_SPATIOTEMPORAL_COVARIANCE_SEED_AUDIT_2026-11-10.json"
)
REFERENCE = ROOT / "docs/research/G4_CDL_TAP_ANGLE_REFERENCE_2026-10-26.json"
EVALUATION = ROOT / "evaluation/g4_cdl_handheld_ut_spatiotemporal_covariance.py"
ADAPTER = ROOT / "integrations/sionna/cdl.py"
MODELS = tuple("ABCDE")
ROSTERS = ("tr38901-4", "tr38901")
PORT_COUNTS = {"tr38901-4": 4, "tr38901": 8}
OPERATING_POINTS = (
    (
        "zero_x_lowband",
        (0.0, 0.0, 0.0),
        (30.0, 0.0, 0.0),
        2.6e9,
        0.25e-3,
    ),
    (
        "tilted_y_midband",
        (-0.35, 0.22, -0.48),
        (0.0, -30.0, 0.0),
        3.5e9,
        1.0e-3,
    ),
    (
        "oblique_highband",
        (0.45, -0.25, 0.8),
        (-12.0, 24.0, 13.0),
        4.9e9,
        0.4e-3,
    ),
)
BS_ORIENTATION = (0.0, 0.0, 0.0)
SEED_START = 2_820_000
CASE_COUNT = len(MODELS) * len(OPERATING_POINTS) * len(ROSTERS)
CLAIM_COUNT = (
    len(MODELS)
    * len(OPERATING_POINTS)
    * sum(PORT_COUNTS[roster] * PORT_COUNTS[roster] * 2 for roster in ROSTERS)
)
DRAW_COUNT = 16_384
FAMILYWISE_ALPHA = 0.01
MAXIMUM_HALF_WIDTH = 0.05
DEVICE_DEPTH_M = 0.15
DEVICE_WIDTH_M = 0.07


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sionna_version() -> str:
    for distribution in ("sionna", "sionna-no-rt"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    raise RuntimeError("neither sionna nor sionna-no-rt distribution is installed")


def _seed_for(model: str, operating_point_index: int, roster_index: int) -> int:
    case_width = len(OPERATING_POINTS) * len(ROSTERS)
    return (
        SEED_START
        + MODELS.index(model) * case_width
        + operating_point_index * len(ROSTERS)
        + roster_index
    )


class _Collector:
    def __init__(self) -> None:
        self.cases: list[dict[str, Any]] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when != "call":
            return
        for key, value in report.user_properties:
            if key == "g4_cdl_handheld_ut_operating_point_spatiotemporal_covariance_case":
                case = json.loads(value)
                case["test_nodeid"] = report.nodeid
                case["test_outcome"] = report.outcome
                self.cases.append(case)


def _validate_cases(cases: list[dict[str, Any]]) -> dict[str, float | int | bool]:
    if len(cases) != CASE_COUNT:
        raise ValueError(f"expected {CASE_COUNT} cases, observed {len(cases)}")
    by_case = {
        (case["model"], case["operating_point_index"], case["roster_index"]): case for case in cases
    }
    expected_keys = {
        (model, operating_point_index, roster_index)
        for model in MODELS
        for operating_point_index in range(len(OPERATING_POINTS))
        for roster_index in range(len(ROSTERS))
    }
    if set(by_case) != expected_keys:
        raise ValueError(
            "profile/operating-point/handheld-roster family differs from frozen protocol"
        )

    claim_count = 0
    failed_claim_count = 0
    failed_case_count = 0
    maximum_half_width = 0.0
    maximum_absolute_error = 0.0
    for model in MODELS:
        for operating_point_index, operating_point in enumerate(OPERATING_POINTS):
            point_id, orientation, velocity, carrier_frequency_hz, lag_s = operating_point
            for roster_index, roster in enumerate(ROSTERS):
                key = (model, operating_point_index, roster_index)
                case = by_case[key]
                port_count = PORT_COUNTS[roster]
                if (
                    case["seed"] != _seed_for(model, operating_point_index, roster_index)
                    or case["draw_count"] != DRAW_COUNT
                    or case["operating_point_id"] != point_id
                    or case["roster"] != roster
                    or case["port_count"] != port_count
                    or case["carrier_frequency_hz"] != carrier_frequency_hz
                    or case["lag_s"] != lag_s
                    or case["ut_orientation_rad"] != list(orientation)
                    or case["bs_orientation_rad"] != list(BS_ORIENTATION)
                    or case["velocity_mps"] != list(velocity)
                    or case["device_depth_m"] != DEVICE_DEPTH_M
                    or case["device_width_m"] != DEVICE_WIDTH_M
                    or case["family_claim_count"] != CLAIM_COUNT
                    or case["familywise_alpha"] != FAMILYWISE_ALPHA
                    or case["test_outcome"] not in {"passed", "failed"}
                    or case["batch_axis_semantics"] != "independent_stochastic_channel_draws"
                ):
                    raise ValueError(f"frozen design/outcome mismatch for {key}")
                expected_case_claims = port_count * port_count * 2
                if len(case["claims"]) != expected_case_claims:
                    raise ValueError(f"expected {expected_case_claims} claims for {key}")
                case_failed_claims = 0
                for claim in case["claims"]:
                    claim_count += 1
                    half_width = float(claim["half_width"])
                    absolute_error = float(claim["absolute_error"])
                    maximum_half_width = max(maximum_half_width, half_width)
                    maximum_absolute_error = max(maximum_absolute_error, absolute_error)
                    expected_pass = (
                        absolute_error <= half_width and half_width <= MAXIMUM_HALF_WIDTH
                    )
                    if bool(claim["passed"]) is not expected_pass:
                        raise ValueError(f"claim pass arithmetic mismatch for {key}")
                    if not expected_pass:
                        failed_claim_count += 1
                        case_failed_claims += 1
                expected_outcome = "failed" if case_failed_claims else "passed"
                if case["test_outcome"] != expected_outcome:
                    raise ValueError(f"pytest outcome does not match retained claims for {key}")
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
    import sionna.phy.channel.tr38901.cdl as cdl
    import sionna.phy.channel.tr38901.channel_coefficients as coefficients

    return {
        "sionna/phy/channel/tr38901/antenna.py": _sha256(Path(antenna.__file__).resolve()),
        "sionna/phy/channel/tr38901/channel_coefficients.py": _sha256(
            Path(coefficients.__file__).resolve()
        ),
        "sionna/phy/channel/tr38901/cdl.py": _sha256(Path(cdl.__file__).resolve()),
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
        [
            "git",
            "-C",
            str(ROOT),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
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

    cases = sorted(
        collector.cases,
        key=lambda case: (
            MODELS.index(case["model"]),
            case["operating_point_index"],
            case["roster_index"],
        ),
    )
    report = {
        "schema": "loslab.g4_cdl_physical_handheld_ut_operating_point_spatiotemporal_covariance_native_validation.v1",
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
        "handheld_rosters": [
            {"name": roster, "port_count": PORT_COUNTS[roster]} for roster in ROSTERS
        ],
        "operating_points": [
            {
                "id": point_id,
                "ut_orientation_rad": list(orientation),
                "velocity_mps": list(velocity),
                "carrier_frequency_hz": carrier_frequency_hz,
                "lag_s": lag_s,
            }
            for point_id, orientation, velocity, carrier_frequency_hz, lag_s in OPERATING_POINTS
        ],
        "bs_orientation_rad": list(BS_ORIENTATION),
        "device_depth_m": DEVICE_DEPTH_M,
        "device_width_m": DEVICE_WIDTH_M,
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
                ADAPTER,
                TEST,
                ORACLE_NATIVE_TEST,
                ORACLE_PORTABLE_TEST,
                PROTOCOL,
                SEED_AUDIT,
                REFERENCE,
                Path(__file__).resolve(),
            )
        },
        "cases": cases,
        "qualification_claim": "none",
        "limitations": [
            "Bounded Sionna PHY 2.2.0 CPU model-consistency evidence for the frozen physical HandheldUTArray operating-point covariance family only.",
            "The family covers CDL-A through E, physical single-V tr38901-4 and tr38901 handheld rosters, default 0.15 m x 0.07 m handset dimensions, three preregistered finite orientation/velocity/carrier/lag operating points, and one single-V omni BS port.",
            "It does not establish dual-polarized HandheldUTArray behavior, arbitrary candidate subsets or handset dimensions, arbitrary arrays/topologies/patterns, continuous orientation/velocity/carrier/lag coverage, mutual coupling, measured-channel truth, RT/material truth, receiver validity, general 3GPP conformance, or overall G4 qualification.",
            "This physical-handheld family is distinct from the earlier PanelArray family whose 38.901-handheld label covered only the Table 7.3-2 element radiation law.",
            "The seed audit establishes no global seed exclusivity.",
            "Production D1 generation and advanced training remain gated.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {"output": str(output), "sha256": digest, **summary},
            sort_keys=True,
        )
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
