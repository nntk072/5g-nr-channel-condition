"""Tests for the City43 path-cap scene visualization and its provenance."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_city43_figure_uses_building_meshes_and_audited_link_endpoints(tmp_path: Path):
    pytest.importorskip("matplotlib")
    module = runpy.run_path(str(ROOT / "scripts" / "plot_rt_city43_cap_sweep_scene.py"))

    scene = "scenes/generated/research_scenes/city43/urban_00043.xml"
    report_path = tmp_path / "cap-report.json"
    report_path.write_text(
        json.dumps(
            {
                "configuration": {
                    "links": [
                        {
                            "link_id": "center-los",
                            "geometric_visibility_state": "visible",
                            "tx_position_m": [0.0, -55.0, 18.0],
                            "rx_position_m": [-8.590140794952246, -15.375005963836855, 1.5],
                        },
                        {
                            "link_id": "center-nlos",
                            "geometric_visibility_state": "blocked",
                            "tx_position_m": [0.0, -55.0, 18.0],
                            "rx_position_m": [24.35684045332106, -5.3057960716627575, 1.5],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    summary_path = tmp_path / "cap-summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "scene": {
                    "name": scene,
                    "scene_tree_sha256": "verified-fixture-scene-hash",
                },
                "inputs": [{"path": str(report_path)}],
            }
        ),
        encoding="utf-8",
    )

    result = module["make_figure"](summary_path, tmp_path / "city43", ROOT)

    assert result["scene"]["building_mesh_count"] == 16
    assert result["links"]["center-los"]["geometric_visibility_state"] == "visible"
    assert result["links"]["center-nlos"]["geometric_visibility_state"] == "blocked"
    assert result["links"]["center-nlos"]["rx_position_m"] == [
        24.35684045332106,
        -5.3057960716627575,
        1.5,
    ]
    for suffix in (".png", ".svg", ".json"):
        assert (tmp_path / f"city43{suffix}").is_file()


def test_city43_obj_reader_preserves_mesh_geometry():
    pytest.importorskip("matplotlib")
    module = runpy.run_path(str(ROOT / "scripts" / "plot_rt_city43_cap_sweep_scene.py"))
    mesh_path = ROOT / "scenes/generated/research_scenes/city43/meshes_00043/building_r01_c02.obj"

    vertices, faces = module["_read_mesh"](mesh_path)

    assert len(vertices) == 8
    assert len(faces) == 12
    assert vertices[0] == (10.012710, -39.202588, 0.0)
