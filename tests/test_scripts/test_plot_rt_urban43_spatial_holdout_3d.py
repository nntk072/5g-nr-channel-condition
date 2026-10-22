"""Regression test for the building and spatial-sample visualization."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_urban43_figure_shows_meshes_and_all_spatial_holdout_samples(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    module = runpy.run_path(str(ROOT / "scripts" / "plot_rt_urban43_spatial_holdout_3d.py"))
    audit_path = ROOT / "docs/research/results/rt-urban43-spatial-holdout-label-audit-20261022.json"

    result = module["make_figure"](audit_path, tmp_path / "urban43-spatial-holdout", ROOT)

    assert result["scene"]["building_mesh_count"] == 16
    assert result["transmitter_site_count"] == 3
    assert result["receiver_samples_per_site"] == {
        "west_x3": 40,
        "center_x0": 40,
        "east_x3": 40,
    }
    assert result["audit"]["label_counts"] == {
        "geometric_los": 60,
        "geometric_nlos": 60,
    }
    assert len(result["representative_links"]) == 6
    sidecar = json.loads((tmp_path / "urban43-spatial-holdout.json").read_text())
    assert sidecar["limitations"] == result["limitations"]
    for suffix in (".png", ".svg", ".json"):
        assert (tmp_path / f"urban43-spatial-holdout{suffix}").is_file()
