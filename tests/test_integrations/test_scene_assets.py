from __future__ import annotations

from types import SimpleNamespace

import pytest
from integrations.sionna.scene_assets import (
    scene_asset_manifest,
    sionna_scene_material_manifest,
)


def test_scene_asset_manifest_hashes_recursive_mesh_material_and_texture_assets(tmp_path):
    scene = tmp_path / "scene.xml"
    mesh_dir = tmp_path / "meshes"
    texture_dir = tmp_path / "textures"
    mesh_dir.mkdir()
    texture_dir.mkdir()
    scene.write_text(
        '<scene><shape><string name="filename" value="meshes/building.obj"/></shape></scene>',
        encoding="utf-8",
    )
    (mesh_dir / "building.obj").write_text("mtllib building.mtl\n", encoding="utf-8")
    (mesh_dir / "building.mtl").write_text(
        "newmtl wall\nmap_Kd ../textures/wall.png\n", encoding="utf-8"
    )
    texture = texture_dir / "wall.png"
    texture.write_bytes(b"texture-v1")

    first = scene_asset_manifest(scene)
    assert first["complete"] is True
    assert first["asset_count"] == 4
    assert {item["path"] for item in first["assets"]} == {
        "scene.xml",
        "meshes/building.obj",
        "meshes/building.mtl",
        "textures/wall.png",
    }

    texture.write_bytes(b"texture-v2")
    second = scene_asset_manifest(scene)
    assert second["asset_set_sha256"] != first["asset_set_sha256"]


def test_scene_asset_manifest_marks_unresolved_uri_as_incomplete(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text(
        '<scene><texture><string name="filename" value="resource://missing.png"/>'
        "</texture></scene>",
        encoding="utf-8",
    )

    manifest = scene_asset_manifest(scene)

    assert manifest["complete"] is False
    assert manifest["unresolved_references"] == ["resource://missing.png"]


def test_sionna_material_manifest_tracks_active_parameters_bindings_and_callback():
    def frequency_update(frequency_hz):
        return frequency_hz

    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        itu_type="concrete",
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=frequency_update,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )

    first = sionna_scene_material_manifest(scene)
    material.scattering_coefficient = 0.5
    second = sionna_scene_material_manifest(scene)

    assert first["schema"] == "sionna_rt_material_state.v1"
    assert first["materials"]["wall-material"]["properties"]["relative_permittivity"] == 4.2
    assert first["object_material_bindings"] == {"wall-1": "wall-material"}
    callback = first["materials"]["wall-material"]["frequency_update_callback"]
    assert callback["code_sha256"]
    assert first != second


def test_sionna_material_manifest_fails_closed_on_unserializable_pattern_state():
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(extension=object()),
        frequency_update_callback=None,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )

    with pytest.raises(TypeError, match="unsupported material-state value"):
        sionna_scene_material_manifest(scene)
