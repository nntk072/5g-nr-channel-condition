"""Deterministic parameterized urban-canyon scenes (Mitsuba XML + OBJ meshes).

These are useful *controlled synthetic ray-tracing geometries*, not accurate
reconstructions or field-validated physical environments. Test native Sionna
compatibility on the target host before running a publication experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from artifacts.provenance import atomic_json, sha256_file


@dataclass(frozen=True)
class CityConfig:
    grid_size: int = 5
    block_spacing_m: float = 55.0
    building_width_min_m: float = 23.0
    building_width_max_m: float = 35.0
    height_min_m: float = 15.0
    height_max_m: float = 62.0
    n_vehicles: int = 12
    seed: int = 42

    def __post_init__(self):
        if self.grid_size < 2 or self.block_spacing_m < self.building_width_max_m + 7:
            raise ValueError("Need at least 2 city blocks and a realistic street gap")
        if self.building_width_min_m <= 0 or self.building_width_max_m < self.building_width_min_m:
            raise ValueError("Invalid building widths")
        if self.height_min_m <= 0 or self.height_max_m < self.height_min_m or self.n_vehicles < 0:
            raise ValueError("Invalid heights / vehicle count")


def cuboid_obj(bounds: tuple[float, float, float, float, float, float]) -> str:
    """Return closed watertight cuboid OBJ mesh with outward-oriented triangles."""
    x0, y0, z0, x1, y1, z1 = map(float, bounds)
    if not (x0 < x1 and y0 < y1 and z0 < z1):
        raise ValueError("Positive cuboid dimensions required")
    vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    faces = [
        (1, 3, 2),
        (1, 4, 3),
        (5, 6, 7),
        (5, 7, 8),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 4, 8),
        (3, 8, 7),
        (4, 1, 5),
        (4, 5, 8),
    ]
    return (
        "# Procedural cuboid, meters\n"
        + "".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in vertices)
        + "".join(f"f {a} {b} {c}\n" for a, b, c in faces)
    )


def on_obstacle(x: float, y: float, blocks: list[dict], margin: float = 0.0) -> bool:
    """Reject receiver positions inside static solids; no oracle labels used."""
    return any(
        b["bounds"][0] - margin <= x <= b["bounds"][3] + margin
        and b["bounds"][1] - margin <= y <= b["bounds"][4] + margin
        for b in blocks
    )


def make_city(out_dir: str | Path, cfg: CityConfig = CityConfig()) -> Path:
    """Generate a fully materialized scene and sidecar reproducibility manifest.

    Vary seeds/heights/material ratios/vehicle placement across training and
    genuinely held-out evaluation scenes; never use one generated city as both.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    scene_filename = f"urban_{cfg.seed:05d}.xml"
    mesh_folder = root / f"meshes_{cfg.seed:05d}"
    mesh_folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    xml = ET.Element("scene", {"version": "3.0.0"})
    for material in ["concrete", "glass", "metal"]:
        # The Sionna RT preprocessor converts the mat-itu_* placeholder BSDFs
        # into the corresponding ITU radio materials.
        ET.SubElement(xml, "bsdf", {"type": "diffuse", "id": f"mat-itu_{material}"})
    obstacles = []

    def add_box(name, bounds, material):
        path = mesh_folder / f"{name}.obj"
        path.write_text(cuboid_obj(bounds), encoding="utf-8")
        element = ET.SubElement(xml, "shape", {"type": "obj", "id": name})
        ET.SubElement(element, "string", {"name": "filename", "value": str(path.relative_to(root))})
        ET.SubElement(element, "ref", {"id": f"mat-itu_{material}"})
        obstacles.append(
            {
                "name": name,
                "bounds": list(map(float, bounds)),
                "material": material,
                "mesh_sha256": sha256_file(path),
            }
        )

    half = (cfg.grid_size - 1) / 2
    centers = [(i - half) * cfg.block_spacing_m for i in range(cfg.grid_size)]
    for row, cy in enumerate(centers):
        for col, cx in enumerate(centers):
            w = float(rng.uniform(cfg.building_width_min_m, cfg.building_width_max_m))
            depth = float(rng.uniform(cfg.building_width_min_m, cfg.building_width_max_m))
            height = float(rng.uniform(cfg.height_min_m, cfg.height_max_m))
            material = str(rng.choice(["concrete", "glass", "metal"], p=[0.75, 0.20, 0.05]))
            add_box(
                f"building_r{row:02d}_c{col:02d}",
                (cx - w / 2, cy - depth / 2, 0.0, cx + w / 2, cy + depth / 2, height),
                material,
            )
    # Concrete ground supplies non-direct reflection for LOS channels.
    side = cfg.block_spacing_m * (cfg.grid_size + 1)
    # Ground is not an above-surface receiver obstruction.
    ground_name = f"ground_{cfg.seed:05d}"
    ground = mesh_folder / f"{ground_name}.obj"
    ground.write_text(
        cuboid_obj((-side / 2, -side / 2, -0.4, side / 2, side / 2, 0.0)), encoding="utf-8"
    )
    element = ET.SubElement(xml, "shape", {"type": "obj", "id": ground_name})
    ET.SubElement(element, "string", {"name": "filename", "value": str(ground.relative_to(root))})
    ET.SubElement(element, "ref", {"id": "mat-itu_concrete"})
    # Vehicles are simplified rectangular metallic bodies: controlled occluders.
    for j in range(cfg.n_vehicles):
        vertical = bool(j % 2)
        idx = int(rng.integers(0, len(centers) - 1))
        street = (centers[idx] + centers[idx + 1]) / 2
        distance = float(rng.uniform(min(centers), max(centers)))
        offset = float(rng.choice([-2.5, 2.5]))
        if vertical:
            cx = street + offset
            cy = distance
            hx = 0.95
            hy = 2.2
        else:
            cx = distance
            cy = street + offset
            hx = 2.2
            hy = 0.95
        box = (cx - hx, cy - hy, 0.05, cx + hx, cy + hy, 1.7)
        # Skip unlikely but possible collisions at cross-streets/corners.
        if on_obstacle(cx, cy, obstacles, margin=1.0):
            continue
        add_box(f"vehicle_{j:03d}", box, "metal")
    scene_path = root / scene_filename
    ET.indent(xml, space="  ")
    scene_path.write_text(ET.tostring(xml, encoding="unicode") + "\n", encoding="utf-8")
    sidecar = {
        "schema": "loslab.city.v1",
        "config": asdict(cfg),
        "xml_sha256": sha256_file(scene_path),
        "buildings_and_vehicles": obstacles,
        "coordinate_frame": "local Cartesian meters, z up",
        "material_semantics": "Sionna 2.2 ITU placeholders; test host compatibility",
        "truth_status": "synthetic geometry, not a measured reconstruction",
    }
    atomic_json(scene_path.with_suffix(".manifest.json"), sidecar)
    return scene_path
