import json
from xml.etree import ElementTree

import pytest
from propagation.geometry import CityConfig, cuboid_obj, make_city, on_obstacle


def test_cube_faces():
    m = cuboid_obj((0, 0, 0, 1, 2, 3))
    assert sum(line.startswith("v ") for line in m.splitlines()) == 8
    assert sum(line.startswith("f ") for line in m.splitlines()) == 12


def test_cube_degenerate_fails():
    with pytest.raises(ValueError):
        cuboid_obj((1, 0, 0, 0, 2, 3))


def test_city_config_invalid():
    with pytest.raises(ValueError):
        CityConfig(block_spacing_m=30, building_width_max_m=40)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"building_width_min_m": 0},
        {"building_width_min_m": 40, "building_width_max_m": 39},
        {"height_min_m": 0},
        {"height_min_m": 20, "height_max_m": 19},
        {"n_vehicles": -1},
    ],
)
def test_city_config_rejects_invalid_building_height_or_vehicle_count(kwargs):
    with pytest.raises(ValueError):
        CityConfig(**kwargs)


def test_building_occupancy():
    blocks = [{"bounds": [0, 0, 0, 5, 5, 10]}]
    assert on_obstacle(2, 2, blocks)
    assert not on_obstacle(20, 20, blocks)


def test_city_generation_and_reproducibility(tmp_path):
    c = CityConfig(grid_size=3, n_vehicles=2, seed=7)
    a = make_city(tmp_path / "first", c)
    b = make_city(tmp_path / "second", c)
    assert a.read_bytes() == b.read_bytes()
    m = json.loads(a.with_suffix(".manifest.json").read_text())
    assert len(m["buildings_and_vehicles"]) >= 9
    assert m["xml_sha256"]
    root = ElementTree.parse(a)
    assert len(root.findall(".//shape")) >= 10
    assert (a.parent / "meshes_00007/building_r00_c00.obj").is_file()


def test_different_scene_seeds(tmp_path):
    a = make_city(tmp_path / "same", CityConfig(grid_size=3, n_vehicles=0, seed=7))
    b = make_city(tmp_path / "same", CityConfig(grid_size=3, n_vehicles=0, seed=8))
    assert a.read_bytes() != b.read_bytes()


def test_city_generation_skips_vehicle_collision(tmp_path, monkeypatch):
    import propagation.geometry as geometry

    monkeypatch.setattr(geometry, "on_obstacle", lambda *_args, **_kwargs: True)
    scene_path = make_city(tmp_path, CityConfig(grid_size=2, n_vehicles=1, seed=11))
    manifest = json.loads(scene_path.with_suffix(".manifest.json").read_text())

    assert all(
        not item["name"].startswith("vehicle_") for item in manifest["buildings_and_vehicles"]
    )
