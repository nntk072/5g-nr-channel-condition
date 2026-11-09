"""Run and retain the frozen G4 TDL Rayleigh marginal protocol."""

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
from evaluation.g4_tdl_rayleigh_marginal import rayleigh_marginal_dkw_radius

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "tests/test_integrations/test_g4_tdl_rayleigh_marginal_native.py"
PROTOCOL = ROOT / "docs/research/G4_07_TDL_RAYLEIGH_MARGINAL_PROTOCOL_2026-10-27.md"
SEED_AUDIT = ROOT / "docs/research/G4_TDL_RAYLEIGH_MARGINAL_SEED_AUDIT_2026-11-09.json"
SOURCE_PATHS = (
    ROOT / "evaluation/g4_tdl_rayleigh_marginal.py",
    ROOT / "evaluation/g4_tdl_covariance_reference.py",
    ROOT / "integrations/sionna/tdl.py",
    TEST,
    PROTOCOL,
    ROOT / "docs/research/G4_SCALABLE_TDL_POWER_DELAY_REFERENCE_2026-10-26.json",
    ROOT / "docs/research/G4_FIXED_TDL_TAP_PROFILE_REFERENCE_2026-10-26.json",
)
MODELS = ("A", "B", "C", "D", "E", "A10", "A30", "B100", "C60", "C300", "D10", "D30")
SEED_START = 1_780_000
DRAW_COUNT = 32_768
FAMILY_TAP_COUNT = 177
FAMILYWISE_ALPHA = 0.01
MAXIMUM_SIMULTANEOUS_CDF_ERROR = 0.05


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
            if key == "g4_tdl_rayleigh_marginal_case":
                case = json.loads(value)
                case["test_nodeid"] = report.nodeid
                case["test_outcome"] = report.outcome
                self.cases.append(case)


def _validate_cases(cases: list[dict[str, Any]]) -> dict[str, float | int]:
    if len(cases) != len(MODELS):
        raise ValueError(f"expected {len(MODELS)} native profile records, observed {len(cases)}")
    by_model = {case["model"]: case for case in cases}
    if set(by_model) != set(MODELS):
        raise ValueError("native profile roster differs from the frozen protocol")
    radius = rayleigh_marginal_dkw_radius(DRAW_COUNT, FAMILY_TAP_COUNT, alpha=FAMILYWISE_ALPHA)
    claim_count = 0
    maximum_ks = 0.0
    maximum_upper_bound = 0.0
    for index, model in enumerate(MODELS):
        case = by_model[model]
        if case["seed"] != SEED_START + index or case["test_outcome"] != "passed":
            raise ValueError(f"native case identity/outcome mismatch for {model}")
        if case["batch_axis_semantics"] != "independent_stochastic_channel_draws":
            raise ValueError(f"batch-axis semantics mismatch for {model}")
        for claim in case["claims"]:
            claim_count += 1
            if claim["draw_count"] != DRAW_COUNT:
                raise ValueError(f"draw count mismatch for {model}")
            if abs(claim["confidence_radius"] - radius) > 1e-15:
                raise ValueError(f"family-wise DKW radius mismatch for {model}")
            if not claim["passed"] or claim["simultaneous_upper_bound"] > MAXIMUM_SIMULTANEOUS_CDF_ERROR:
                raise ValueError(f"frozen Rayleigh marginal criterion failed for {model}")
            maximum_ks = max(maximum_ks, float(claim["empirical_ks_distance"]))
            maximum_upper_bound = max(maximum_upper_bound, float(claim["simultaneous_upper_bound"]))
    if claim_count != FAMILY_TAP_COUNT:
        raise ValueError(f"expected {FAMILY_TAP_COUNT} claims, observed {claim_count}")
    return {
        "claim_count": claim_count,
        "confidence_radius": radius,
        "maximum_empirical_ks_distance": maximum_ks,
        "maximum_simultaneous_upper_bound": maximum_upper_bound,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    sidecar = Path(str(output) + ".sha256")
    if output.exists() or sidecar.exists():
        parser.error("refusing to overwrite retained result or checksum")
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=all"],
        text=True,
    )
    if status:
        parser.error("refusing to run from a dirty worktree")
    seed_audit = json.loads(SEED_AUDIT.read_text(encoding="utf-8"))
    expected_seeds = list(range(SEED_START, SEED_START + len(MODELS)))
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
    if exit_code:
        parser.error(f"native pytest execution failed with exit code {exit_code}")
    summary = _validate_cases(collector.cases)
    cases = sorted(collector.cases, key=lambda case: MODELS.index(case["model"]))
    claims = [
        {"model": case["model"], **claim}
        for case in cases
        for claim in case["claims"]
    ]
    report = {
        "schema": "loslab.g4_tdl_rayleigh_marginal_native_validation.v1",
        "date": "2026-11-09",
        "base_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "all_passed": True,
        "criteria_changed": False,
        "profiles": list(MODELS),
        "draws_per_profile": DRAW_COUNT,
        "simultaneous_claim_count": FAMILY_TAP_COUNT,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "confidence_radius": summary["confidence_radius"],
        "maximum_simultaneous_cdf_error": MAXIMUM_SIMULTANEOUS_CDF_ERROR,
        "maximum_empirical_ks_distance": summary["maximum_empirical_ks_distance"],
        "maximum_simultaneous_upper_bound": summary["maximum_simultaneous_upper_bound"],
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
        "source_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in SOURCE_PATHS},
        "profile_results": [
            {
                "model": case["model"],
                "seed": case["seed"],
                "native_tap_count": case["native_tap_count"],
                "rayleigh_tap_count": case["rayleigh_tap_count"],
                "sionna_version": case["sionna_version"],
                "batch_axis_semantics": case["batch_axis_semantics"],
                "parameter_table_source": case["parameter_table_source"],
                "parameter_table_sha256": case["parameter_table_sha256"],
                "maximum_empirical_ks_distance": max(claim["empirical_ks_distance"] for claim in case["claims"]),
                "maximum_simultaneous_upper_bound": max(claim["simultaneous_upper_bound"] for claim in case["claims"]),
                "passed": all(claim["passed"] for claim in case["claims"]),
            }
            for case in cases
        ],
        "claims": claims,
        "qualification_claim": "none",
        "limitations": [
            "This is a bounded Sionna PHY 2.2.0 CPU model-consistency result for the 177 ETSI-labeled Rayleigh-only taps in the frozen scalable/fixed TDL roster.",
            "The merged LOS-plus-diffuse first taps in scalable TDL-D/E and fixed D10/D30 are excluded from this Rayleigh marginal family.",
            "It does not establish joint Gaussianity, inter-tap independence, temporal/spatial dependence, specular LOS behavior, measured-channel truth, receiver validity, or overall G4 qualification.",
            "The seed audit establishes no global seed exclusivity.",
            "Production RT generation and advanced training remain gated.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.write_text(payload, encoding="utf-8", errors="strict")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "sha256": digest, **summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
