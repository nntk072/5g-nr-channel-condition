"""CLI regression tests for the analytical identifiability sweep."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_identifiability_sweep.py"


def test_identifiability_sweep_records_rank_deficiency_and_crlb(tmp_path):
    output = tmp_path / "sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(output),
            "--tone-counts",
            "32",
            "--snr-db",
            "20",
            "--separation-fractions",
            "0",
            "1",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text())
    assert result["schema"] == "loslab.identifiability_sweep.v1"
    assert len(result["cases"]) == 2
    assert result["cases"][0]["crlb_delay_std_ns"] is None
    assert result["cases"][0]["condition_number"] is None
    assert result["cases"][0]["absolute_delay_covariance_available"] is False
    assert result["cases"][1]["crlb_delay_std_ns"] is not None
    assert result["cases"][1]["rank"] == result["cases"][1]["parameter_count"]
    assert result["cases"][1]["unknown_timing_rank"] == 6
    assert result["cases"][1]["absolute_delay_covariance_available"] is False
    assert result["cases"][1]["unknown_timing_relative_delay_crlb_std_ns"] is not None


def test_identifiability_sweep_exposes_gain_phase_dependent_coincident_path_ambiguity(tmp_path):
    output = tmp_path / "phase-sweep.json"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(output),
            "--tone-counts",
            "32",
            "--snr-db",
            "20",
            "--separation-fractions",
            "0",
            "1",
            "--gain-phase-difference-deg",
            "0",
            "90",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    cases = json.loads(output.read_text())["cases"]
    assert len(cases) == 4
    zero_phase_coincident, quadrature_coincident, zero_phase_separated, _ = cases
    assert zero_phase_coincident["gain_phase_difference_deg"] == 0
    assert zero_phase_coincident["unknown_timing_relative_delay_crlb_std_ns"] is None
    assert quadrature_coincident["gain_phase_difference_deg"] == 90
    assert quadrature_coincident["unknown_timing_relative_delay_crlb_std_ns"] is not None
    assert zero_phase_separated["unknown_timing_relative_delay_crlb_std_ns"] is not None


def test_identifiability_sweep_refuses_to_overwrite(tmp_path):
    output = tmp_path / "existing.json"
    output.write_text("keep")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(output)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
    assert output.read_text() == "keep"


def test_identifiability_sweep_rejects_zero_bandwidth(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "invalid.json"),
            "--tone-counts",
            "2",
            "--tone-spacing-hz",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "tone_spacing_hz must be finite and positive" in result.stderr
