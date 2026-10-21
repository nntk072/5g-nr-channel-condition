import json
from xml.etree import ElementTree
import pytest
from loslab.geometry import CityConfig,cuboid_obj,on_obstacle,make_city

def test_cube_faces():
    m=cuboid_obj((0,0,0,1,2,3))
    assert sum(line.startswith('v ') for line in m.splitlines())==8
    assert sum(line.startswith('f ') for line in m.splitlines())==12

def test_cube_degenerate_fails():
    with pytest.raises(ValueError):cuboid_obj((1,0,0,0,2,3))

def test_city_config_invalid():
    with pytest.raises(ValueError):CityConfig(block_spacing_m=30,building_width_max_m=40)

def test_building_occupancy():
    blocks=[{'bounds':[0,0,0,5,5,10]}]
    assert on_obstacle(2,2,blocks)
    assert not on_obstacle(20,20,blocks)

def test_city_generation_and_reproducibility(tmp_path):
    c=CityConfig(grid_size=3,n_vehicles=2,seed=7)
    a=make_city(tmp_path/'first',c)
    b=make_city(tmp_path/'second',c)
    assert a.read_bytes()==b.read_bytes()
    m=json.loads(a.with_suffix('.manifest.json').read_text())
    assert len(m['buildings_and_vehicles'])>=9
    assert m['xml_sha256']
    root=ElementTree.parse(a)
    assert len(root.findall('.//shape'))>=10
    assert (a.parent/'meshes_00007/building_r00_c00.obj').is_file()

def test_different_scene_seeds(tmp_path):
    a=make_city(tmp_path/'same',CityConfig(grid_size=3,n_vehicles=0,seed=7))
    b=make_city(tmp_path/'same',CityConfig(grid_size=3,n_vehicles=0,seed=8))
    assert a.read_bytes()!=b.read_bytes()
