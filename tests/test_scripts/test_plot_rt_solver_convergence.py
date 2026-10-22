"""Seed-summary tests for the RT convergence figure generator."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PLOTTER = runpy.run_path(str(ROOT / "scripts" / "plot_rt_solver_convergence.py"))
summarize_seed_range = PLOTTER["summarize_seed_range"]
load_reports = PLOTTER["load_reports"]
material_label = PLOTTER["_material_label"]


def test_seed_summary_uses_median_and_full_seed_range():
    rows = [
        {"samples_per_src": 100, "value": 0.1},
        {"samples_per_src": 100, "value": 0.3},
        {"samples_per_src": 100, "value": 0.2},
        {"samples_per_src": 1000, "value": 0.01},
    ]
    x, median, low, high = summarize_seed_range(rows, x_field="samples_per_src", metric="value")
    np.testing.assert_array_equal(x, [100, 1000])
    np.testing.assert_allclose(median, [0.2, 0.01])
    np.testing.assert_allclose(low, [0.1, 0.01])
    np.testing.assert_allclose(high, [0.3, 0.01])


def test_log_error_summary_omits_exact_reference_zeros():
    rows = [
        {"samples_per_src": 100, "value": 0.0},
        {"samples_per_src": 100, "value": 0.02},
        {"samples_per_src": 1000, "value": 0.0},
    ]
    x, median, low, high = summarize_seed_range(
        rows, x_field="samples_per_src", metric="value", positive_only=True
    )
    np.testing.assert_array_equal(x, [100])
    np.testing.assert_allclose(median, [0.02])
    np.testing.assert_allclose(low, [0.02])
    np.testing.assert_allclose(high, [0.02])


def test_material_label_distinguishes_explicit_scattering_fixture():
    assert material_label({"material_scattering_overrides": {"screen-mat": 0.5}}) == (
        "S(screen-mat)=0.5"
    )
    assert material_label({}) == "scene-default scattering"


@pytest.mark.parametrize("version", ("v2", "v3"))
def test_loader_accepts_versioned_reports_with_absolute_delay(tmp_path: Path, version: str):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema": f"loslab.sionna_rt_solver_convergence.{version}",
                "configuration": {"normalize_delays_for_cfr": False},
                "runs": [],
            }
        ),
        encoding="utf-8",
    )
    assert load_reports([report])[0]["data"]["schema"].endswith(version)
