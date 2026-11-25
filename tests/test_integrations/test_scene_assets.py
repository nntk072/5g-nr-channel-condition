from __future__ import annotations

import io
import sys
from types import CodeType, SimpleNamespace
from typing import Any, cast

import integrations.sionna.scene_assets as scene_assets_module
import numpy as np
import pytest
from integrations.sionna.scene_assets import (
    scene_asset_manifest,
    sionna_scene_material_manifest,
)

_CALLBACK_SCALE = 1.0
_UNSERIALIZABLE_CALLBACK_STATE = object()
_CALLBACK_GLOBAL_CYCLE: dict[str, object] = {}
_CALLBACK_GLOBAL_MAPPING = {"b": 2, "a": 1}
_CALLBACK_GLOBAL_SEQUENCE = (1, 2)
_CALLBACK_GLOBAL_SEQUENCE_CYCLE: list[object] = []
_CALLBACK_SELF: object
_CALLBACK_GLOBAL_CALLABLE = len


def _frequency_scale(frequency_hz: float) -> float:
    return frequency_hz * _CALLBACK_SCALE


def _frequency_update_with_global_helper(frequency_hz: float) -> float:
    return _frequency_scale(frequency_hz)


def _frequency_update_with_unsupported_global(frequency_hz: float) -> object:
    del frequency_hz
    return _UNSERIALIZABLE_CALLBACK_STATE


def _frequency_update_with_numpy_module(frequency_hz: float) -> float:
    return float(scene_assets_module.np.asarray(frequency_hz))


def _frequency_update_with_cyclic_global(frequency_hz: float) -> float:
    del frequency_hz
    return len(_CALLBACK_GLOBAL_CYCLE)


def _frequency_update_with_container_globals(frequency_hz: float) -> float:
    return frequency_hz + len(_CALLBACK_GLOBAL_MAPPING) + len(_CALLBACK_GLOBAL_SEQUENCE)


def _frequency_update_with_self_global(frequency_hz: float) -> float:
    return frequency_hz if _CALLBACK_SELF else 0.0


def _frequency_update_with_class_global(frequency_hz: float) -> object:
    return SimpleNamespace(value=frequency_hz)


def _frequency_update_with_callable_global(frequency_hz: float) -> int:
    return _CALLBACK_GLOBAL_CALLABLE((frequency_hz,))


def _frequency_update_with_cyclic_sequence(frequency_hz: float) -> float:
    del frequency_hz
    return len(_CALLBACK_GLOBAL_SEQUENCE_CYCLE)


_CALLBACK_SELF = _frequency_update_with_self_global


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
    asset_rows = cast(list[dict[str, Any]], first["assets"])
    assert {item["path"] for item in asset_rows} == {
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


def test_scene_asset_manifest_enforces_per_asset_and_total_byte_limits(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text("<scene/>", encoding="utf-8")

    with pytest.raises(ValueError, match="per-asset|configured byte limit"):
        scene_asset_manifest(scene, max_asset_bytes=3)
    with pytest.raises(ValueError, match="total byte limit"):
        scene_asset_manifest(scene, max_total_bytes=3)
    with pytest.raises(TypeError, match="max_asset_bytes"):
        scene_asset_manifest(scene, max_asset_bytes=True)


@pytest.mark.parametrize("name", ["max_asset_bytes", "max_total_bytes"])
def test_scene_asset_manifest_requires_positive_integer_limits(tmp_path, name):
    scene = tmp_path / "scene.xml"
    scene.write_text("<scene/>", encoding="utf-8")
    with pytest.raises(TypeError, match=name):
        scene_asset_manifest(scene, **{name: 1.5})
    with pytest.raises(ValueError, match=f"{name} must be positive"):
        scene_asset_manifest(scene, **{name: 0})


def test_bounded_asset_reader_stops_after_limit_plus_one_byte():
    reader = scene_assets_module._BoundedAssetReader(io.BytesIO(b"abc"), max_bytes=2)
    with pytest.raises(ValueError, match="configured byte limit"):
        reader.read()
    with pytest.raises(ValueError, match="configured byte limit"):
        reader.read()


def test_asset_digest_rejects_non_regular_files(tmp_path):
    with pytest.raises(ValueError, match="must be a regular file"):
        scene_assets_module._asset_references_and_digest(
            tmp_path, max_bytes=100, limit_error="fixture limit"
        )


def test_scene_asset_manifest_rejects_missing_scene_and_referenced_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match="scene XML not found"):
        scene_asset_manifest(tmp_path / "missing.xml")
    scene = tmp_path / "scene.xml"
    scene.write_text('<scene><shape filename="missing.obj"/></scene>', encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="scene asset not found"):
        scene_asset_manifest(scene)


def test_scene_asset_manifest_follows_mtl_options_and_reports_external_paths(tmp_path):
    scene = tmp_path / "scene.xml"
    external_texture = tmp_path.parent / f"{tmp_path.name}-texture.png"
    external_texture.write_bytes(b"external")
    (tmp_path / "model.obj").write_text("mtllib model.mtl\n", encoding="utf-8")
    (tmp_path / "model.mtl").write_text(
        f"map_Kd -s 1 1 1 {external_texture}\nbump -bm 0.2 {external_texture}\n",
        encoding="utf-8",
    )
    scene.write_text('<scene><shape filename="model.obj"/></scene>', encoding="utf-8")
    manifest = scene_asset_manifest(scene)
    assert manifest["complete"] is True
    assert manifest["asset_count"] == 4
    rows = cast(list[dict[str, Any]], manifest["assets"])
    assert f"external/{external_texture.name}" in {row["path"] for row in rows}


def test_scene_asset_manifest_includes_sidecar_and_skips_obj_comments(tmp_path):
    scene = tmp_path / "scene.xml"
    scene.write_text('<scene><shape filename="model.obj"/></scene>', encoding="utf-8")
    (tmp_path / "scene.manifest.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.obj").write_text("# only a comment\nmtllib model.mtl\n", encoding="utf-8")
    (tmp_path / "model.mtl").write_text("newmtl wall\n", encoding="utf-8")
    manifest = scene_asset_manifest(scene)
    assert manifest["asset_count"] == 4
    assert manifest["complete"] is True


def test_asset_digest_detects_content_growth_after_initial_stat(tmp_path, monkeypatch):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"x")
    monkeypatch.setattr(scene_assets_module.os, "read", lambda _fd, size: b"x" * size)
    with pytest.raises(ValueError, match="fixture limit"):
        scene_assets_module._asset_references_and_digest(
            asset, max_bytes=1, limit_error="fixture limit"
        )


def test_asset_digest_detects_change_during_inspection(tmp_path, monkeypatch):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"content")
    identities = iter(((1,), (1,), (2,), (3,)))
    monkeypatch.setattr(scene_assets_module, "_stat_identity", lambda _info: next(identities))
    with pytest.raises(ValueError, match="changed while being inspected"):
        scene_assets_module._asset_references_and_digest(
            asset, max_bytes=100, limit_error="fixture limit"
        )


def test_asset_digest_rejects_identity_change_while_opening(tmp_path, monkeypatch):
    asset = tmp_path / "asset.bin"
    asset.write_bytes(b"content")
    identities = iter(((1,), (2,)))
    monkeypatch.setattr(scene_assets_module, "_stat_identity", lambda _info: next(identities))
    with pytest.raises(ValueError, match="changed while opening"):
        scene_assets_module._asset_references_and_digest(
            asset, max_bytes=100, limit_error="fixture limit"
        )


def test_obj_parser_rejects_growth_after_initial_size_check(tmp_path, monkeypatch):
    asset = tmp_path / "asset.obj"
    asset.write_bytes(b"x")

    def growing_stream(descriptor, _mode):
        scene_assets_module.os.close(descriptor)
        return io.BytesIO(b"abc")

    monkeypatch.setattr(scene_assets_module.os, "fdopen", growing_stream)
    with pytest.raises(ValueError, match="fixture limit"):
        scene_assets_module._asset_references_and_digest(
            asset, max_bytes=1, limit_error="fixture limit"
        )


def test_scene_asset_manifest_rejects_overlong_obj_text_line(tmp_path):
    scene = tmp_path / "scene.xml"
    mesh = tmp_path / "mesh.obj"
    scene.write_text('<scene><shape filename="mesh.obj"/></scene>', encoding="utf-8")
    mesh.write_bytes(b"#" + b"x" * scene_assets_module.MAX_SCENE_ASSET_TEXT_LINE_BYTES)
    with pytest.raises(ValueError, match="text line exceeds the parser limit"):
        scene_asset_manifest(scene)


def test_scene_asset_manifest_hashes_without_read_bytes(tmp_path, monkeypatch):
    scene = tmp_path / "scene.xml"
    scene.write_text("<scene/>", encoding="utf-8")

    def fail_read_bytes(_path):
        raise AssertionError("scene asset hashing must stream through the open descriptor")

    monkeypatch.setattr(type(scene), "read_bytes", fail_read_bytes)
    manifest = scene_asset_manifest(scene)

    assert manifest["complete"] is True
    assert manifest["asset_count"] == 1


def test_sionna_material_manifest_tracks_active_parameters_bindings_and_callback():
    def frequency_update(frequency_hz: float) -> float:
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

    first_data = cast(dict[str, Any], first)
    assert first_data["schema"] == "sionna_rt_material_state.v1"
    assert first_data["materials"]["wall-material"]["properties"]["relative_permittivity"] == 4.2
    assert first["object_material_bindings"] == {"wall-1": "wall-material"}
    callback = first_data["materials"]["wall-material"]["frequency_update_callback"]
    assert callback["code_sha256"]
    assert first != second


def test_material_callback_manifest_binds_referenced_global_and_helper_function(monkeypatch):
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=_frequency_update_with_global_helper,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )

    first = sionna_scene_material_manifest(scene)
    callback = cast(dict[str, Any], first)["materials"]["wall-material"][
        "frequency_update_callback"
    ]
    helper = callback["referenced_globals"]["_frequency_scale"]
    assert helper["referenced_globals"]["_CALLBACK_SCALE"] == 1.0

    monkeypatch.setattr(sys.modules[__name__], "_CALLBACK_SCALE", 2.0)
    second = sionna_scene_material_manifest(scene)

    assert first != second


def test_material_callback_manifest_records_module_globals():
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=_frequency_update_with_numpy_module,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )
    record = cast(dict[str, Any], sionna_scene_material_manifest(scene))
    callback = record["materials"]["wall-material"]["frequency_update_callback"]
    assert callback["referenced_globals"]["scene_assets_module"]["module"] == (
        "integrations.sionna.scene_assets"
    )


def test_material_callback_manifest_records_container_globals():
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=_frequency_update_with_container_globals,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )
    callback = cast(dict[str, Any], sionna_scene_material_manifest(scene))["materials"][
        "wall-material"
    ]["frequency_update_callback"]
    globals_manifest = callback["referenced_globals"]
    assert globals_manifest["_CALLBACK_GLOBAL_MAPPING"]["mapping_items"]
    assert globals_manifest["_CALLBACK_GLOBAL_SEQUENCE"]["tuple"] == [1, 2]


def test_material_value_normalizes_numpy_and_code_constants():
    class ArrayValue:
        def numpy(self):
            return np.asarray([1, 2], dtype=np.int64)

    assert scene_assets_module._material_json_value(np.asarray(3)) == 3
    assert scene_assets_module._material_json_value(np.asarray([1, 2])) == [1, 2]
    assert scene_assets_module._material_json_value(np.float64(2.5)) == 2.5
    assert scene_assets_module._material_json_value(ArrayValue()) == [1, 2]
    assert scene_assets_module._code_constant(b"abc") == {"bytes_hex": "616263"}
    assert scene_assets_module._code_constant(Ellipsis) == {"ellipsis": True}
    assert scene_assets_module._material_json_value((1, 2)) == [1, 2]

    def outer():
        def inner():
            return 1

        return inner

    nested_code = next(value for value in outer.__code__.co_consts if isinstance(value, CodeType))
    assert "code" in scene_assets_module._code_constant(nested_code)
    assert scene_assets_module._code_constant((b"a", Ellipsis)) == [
        {"bytes_hex": "61"},
        {"ellipsis": True},
    ]


def test_material_value_rejects_nonfinite_values_and_nonstring_mapping_keys():
    with pytest.raises(ValueError, match="non-finite scalar"):
        scene_assets_module._material_json_value(float("inf"))
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        scene_assets_module._material_json_value({1: "invalid"})


def test_material_manifest_rejects_invalid_scene_mappings_material_names_and_bindings():
    with pytest.raises(TypeError, match="radio_materials and objects mappings"):
        sionna_scene_material_manifest(SimpleNamespace(radio_materials=(), objects=()))
    invalid_name_scene = SimpleNamespace(
        radio_materials={"": object()}, objects={}, frequency=3.5e9
    )
    with pytest.raises(ValueError, match="invalid radio material name"):
        sionna_scene_material_manifest(invalid_name_scene)
    material = SimpleNamespace(name="unlisted-material")
    listed_material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
    )
    unresolved_scene = SimpleNamespace(
        radio_materials={"wall-material": listed_material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )
    with pytest.raises(ValueError, match="unresolved radio material"):
        sionna_scene_material_manifest(unresolved_scene)


def test_callable_manifest_requires_python_code_and_detects_recursion():
    with pytest.raises(TypeError, match="expose Python code"):
        scene_assets_module._callable_manifest(object())
    record = scene_assets_module._callable_manifest(_frequency_update_with_self_global)
    recursive = record["referenced_globals"]["_CALLBACK_SELF"]
    assert "recursive_callable" in recursive


def test_material_callback_manifest_rejects_cyclic_global_state():
    _CALLBACK_GLOBAL_CYCLE.clear()


def test_material_callback_manifest_rejects_cyclic_global_sequence():
    _CALLBACK_GLOBAL_SEQUENCE_CYCLE.clear()
    _CALLBACK_GLOBAL_SEQUENCE_CYCLE.append(_CALLBACK_GLOBAL_SEQUENCE_CYCLE)
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=_frequency_update_with_cyclic_sequence,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )
    with pytest.raises(ValueError, match="cyclic global sequence"):
        sionna_scene_material_manifest(scene)
    _CALLBACK_GLOBAL_SEQUENCE_CYCLE.clear()


def test_material_callback_manifest_serializes_class_global():
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=_frequency_update_with_class_global,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )
    callback = cast(dict[str, Any], sionna_scene_material_manifest(scene))["materials"][
        "wall-material"
    ]["frequency_update_callback"]
    assert callback["referenced_globals"]["SimpleNamespace"]["class_qualname"] == (
        "SimpleNamespace"
    )


def test_material_callback_manifest_serializes_callable_global():
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=_frequency_update_with_callable_global,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )
    callback = cast(dict[str, Any], sionna_scene_material_manifest(scene))["materials"][
        "wall-material"
    ]["frequency_update_callback"]
    assert callback["referenced_globals"]["_CALLBACK_GLOBAL_CALLABLE"]["callable_qualname"] == "len"
    _CALLBACK_GLOBAL_CYCLE["self"] = _CALLBACK_GLOBAL_CYCLE
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=_frequency_update_with_cyclic_global,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )
    with pytest.raises(ValueError, match="cyclic global mapping"):
        sionna_scene_material_manifest(scene)
    _CALLBACK_GLOBAL_CYCLE.clear()


def test_material_callback_manifest_fails_closed_on_unsupported_global_state():
    material = SimpleNamespace(
        name="wall-material",
        relative_permittivity=4.2,
        conductivity=0.01,
        thickness=0.2,
        scattering_coefficient=0.25,
        xpd_coefficient=0.1,
        scattering_pattern=SimpleNamespace(roughness=0.4),
        frequency_update_callback=_frequency_update_with_unsupported_global,
    )
    scene = SimpleNamespace(
        radio_materials={"wall-material": material},
        objects={"wall-1": SimpleNamespace(radio_material=material)},
        frequency=3.5e9,
    )

    with pytest.raises(TypeError, match="unsupported global state"):
        sionna_scene_material_manifest(scene)


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
