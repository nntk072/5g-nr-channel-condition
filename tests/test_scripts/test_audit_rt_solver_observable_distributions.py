"""Tests for paired-seed RT observable-distribution audits."""

from __future__ import annotations

import json
import lzma
import runpy
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = runpy.run_path(str(ROOT / "scripts" / "audit_rt_solver_observable_distributions.py"))
audit = AUDIT["audit_observable_distributions"]


def _row(seed: int, budget: int, *, los: bool = True, power: float = 2.0) -> dict[str, Any]:
    return {
        "link_id": "los-a" if los else "nlos-a",
        "seed": seed,
        "max_depth": 3,
        "samples_per_src": budget,
        "geometric_los": los,
        "relative_cfr_l2_to_max_budget_and_depth": 0.25 if budget == 100 else 0.0,
        "valid_path_count": budget // 100,
        "valid_non_diffuse_path_count": 10,
        "valid_diffuse_reflection_path_count": budget // 200,
        "cfr_mean_power": 4.0,
        "non_diffuse_path_power_sum": 1.0,
        "diffuse_path_power_sum": power,
        "power_weighted_rms_delay_spread_ns": 5.0,
        "non_diffuse_power_weighted_rms_delay_spread_ns": 1.0,
        "diffuse_power_weighted_rms_delay_spread_ns": 2.0,
        "non_diffuse_power_weighted_arrival_angular_spread_rad_3d": 0.1,
        "diffuse_power_weighted_arrival_angular_spread_rad_3d": 0.2,
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> Path:
    report = {
        "schema": "loslab.sionna_rt_solver_convergence.v4",
        "source_revision": "abc123",
        "scene": {"scene_tree_sha256": "scene-sha"},
        "script_sha256": "script-sha",
        "package_source_sha256": "package-sha",
        "configuration": {
            "normalize_delays_for_cfr": False,
            "samples_per_src_values": [100, 1000],
            "max_depth_values": [0, 3],
        },
        "runs": rows,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_summarizes_component_distributions_with_seed_bootstrap(tmp_path: Path):
    rows = []
    for seed, ref_power in ((1, 2.0), (2, 4.0), (3, 6.0)):
        rows.extend(
            (
                _row(seed, 100, power=ref_power * 0.5),
                _row(seed, 1000, power=ref_power),
            )
        )
    report = _write(tmp_path / "report.json", rows)

    result = audit(
        report,
        low_budget=100,
        reference_budget=1000,
        depth=3,
        bootstrap_draws=1000,
        bootstrap_seed=7,
    )

    diffuse_power = next(
        row
        for row in result["summaries"]
        if row["condition"] == "LOS" and row["metric"] == "diffuse_path_power_sum"
    )
    assert diffuse_power["valid_seed_pairs"] == 3
    assert diffuse_power["low_budget_median"] == pytest.approx(2.0)
    assert diffuse_power["reference_budget_median"] == pytest.approx(4.0)
    assert diffuse_power["paired_median_delta_low_minus_reference"] == pytest.approx(-2.0)
    assert diffuse_power["paired_median_relative_delta"] == pytest.approx(-0.5)
    assert result["configuration"]["resampling_unit"].startswith("paired solver seed")
    assert result["auditor"]["sha256"]

    cfr_error = next(
        row for row in result["summaries"] if row["metric"] == "cfr_relative_l2_to_reference"
    )
    assert cfr_error["paired_median_delta_low_minus_reference"] == pytest.approx(0.25)


def test_path_rates_are_normalized_by_budget_and_seed(tmp_path: Path):
    rows = [_row(seed, budget) for seed in (1, 2) for budget in (100, 1000)]
    report = _write(tmp_path / "report.json", rows)
    result = audit(report, low_budget=100, reference_budget=1000, depth=3, bootstrap_draws=200)
    rate = next(
        row
        for row in result["summaries"]
        if row["condition"] == "LOS" and row["metric"] == "valid_path_rate_per_1000_samples"
    )
    assert rate["low_budget_median"] == rate["reference_budget_median"] == 10.0
    assert rate["paired_median_delta_bootstrap_ci95"] == [0.0, 0.0]
    specular_count = next(
        row
        for row in result["summaries"]
        if row["condition"] == "LOS" and row["metric"] == "non_diffuse_path_count"
    )
    assert specular_count["low_budget_median"] == specular_count["reference_budget_median"] == 10.0


def test_rejects_unmatched_seed_sets_and_invalid_budget(tmp_path: Path):
    rows = [_row(1, 100), _row(2, 1000), _row(1, 1000)]
    report = _write(tmp_path / "report.json", rows)
    with pytest.raises(ValueError, match="same matched link/seed"):
        audit(report, low_budget=100, reference_budget=1000, depth=3)
    with pytest.raises(ValueError, match="reference_budget must exceed"):
        audit(report, low_budget=1000, reference_budget=100, depth=3)


def test_rejects_duplicate_budget_link_seed_row(tmp_path: Path):
    rows = [_row(1, 100), _row(1, 100), _row(1, 1000)]
    report = _write(tmp_path / "report.json", rows)
    with pytest.raises(ValueError, match="duplicate link/seed"):
        audit(report, low_budget=100, reference_budget=1000, depth=3)


def test_accepts_compact_v3_reports_without_cfr_vectors(tmp_path: Path):
    rows = [_row(seed, budget) for seed in (1, 2) for budget in (100, 1000)]
    report = _write(tmp_path / "report.json", rows)
    content = json.loads(report.read_text(encoding="utf-8"))
    content["schema"] = "loslab.sionna_rt_solver_convergence.v3"
    report.write_text(json.dumps(content), encoding="utf-8")

    result = audit(report, low_budget=100, reference_budget=1000, depth=3, bootstrap_draws=200)

    assert result["matched_link_seed_pairs"] == 2


def test_reads_xz_compressed_reports(tmp_path: Path):
    rows = [_row(seed, budget) for seed in (1, 2) for budget in (100, 1000)]
    report = _write(tmp_path / "report.json", rows)
    compressed = tmp_path / "report.json.xz"
    with lzma.open(compressed, "wt", encoding="utf-8") as stream:
        json.dump(json.loads(report.read_text(encoding="utf-8")), stream)

    result = audit(compressed, low_budget=100, reference_budget=1000, depth=3, bootstrap_draws=200)

    assert result["matched_link_seed_pairs"] == 2


def test_single_seed_does_not_claim_a_bootstrap_interval(tmp_path: Path):
    report = _write(tmp_path / "report.json", [_row(1, 100), _row(1, 1000)])
    result = audit(report, low_budget=100, reference_budget=1000, depth=3)
    metric = next(row for row in result["summaries"] if row["metric"] == "diffuse_path_power_sum")
    assert metric["valid_seed_pairs"] == 1
    assert metric["paired_median_delta_bootstrap_ci95"] is None
