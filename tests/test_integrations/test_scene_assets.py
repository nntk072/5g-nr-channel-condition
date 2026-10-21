from __future__ import annotations

from integrations.sionna.scene_assets import scene_asset_manifest


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
