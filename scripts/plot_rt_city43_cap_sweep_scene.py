"""Plot the exact City43 building meshes and the LOS/NLOS links in the path-cap sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from artifacts.provenance import atomic_json, sha256_file
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _read_mesh(path: Path) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """Read the small, triangulated Wavefront OBJ meshes used by the scene."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["v"]:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        elif fields[:1] == ["f"]:
            faces.append([int(value.split("/", maxsplit=1)[0]) - 1 for value in fields[1:]])
    if not vertices or not faces:
        raise ValueError(f"mesh has no vertices or faces: {path}")
    return vertices, faces


def make_figure(summary_path: Path, output: Path, repo_root: Path) -> dict[str, Any]:
    """Render the actual City43 building meshes around the two audited links."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scene_path = repo_root / summary["scene"]["name"]
    raw_report_path = Path(summary["inputs"][0]["path"])
    if not raw_report_path.is_file():
        raw_report_path = repo_root / "docs/research/results" / raw_report_path.name
    raw_report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    manifest_path = scene_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    buildings = [
        item
        for item in manifest["buildings_and_vehicles"]
        if str(item["name"]).startswith("building_")
    ]
    links = {item["link_id"]: item for item in raw_report["configuration"]["links"]}
    if {"center-los", "center-nlos"} - links.keys():
        raise ValueError("summary must contain the center-los and center-nlos links")

    fig = plt.figure(figsize=(17, 8.2), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.55, 1.0), wspace=0.16)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax2d = fig.add_subplot(grid[0, 1])
    materials = {
        "concrete": ("#98a6ad", "#52616a"),
        "glass": ("#76a9c4", "#345c70"),
        "metal": ("#bfad83", "#756644"),
    }

    for building in buildings:
        mesh_path = scene_path.parent / "meshes_00043" / f"{building['name']}.obj"
        vertices, faces = _read_mesh(mesh_path)
        polygons = [[vertices[index] for index in face] for face in faces]
        facecolor, edgecolor = materials.get(building["material"], materials["concrete"])
        ax3d.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.34,
                alpha=0.94,
            )
        )
        bounds = [float(value) for value in building["bounds"]]
        x0, y0, _, x1, y1, _ = bounds
        ax2d.fill(
            [x0, x1, x1, x0],
            [y0, y0, y1, y1],
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.7,
            alpha=0.85,
            zorder=1,
        )

    link_colors = {"center-los": "#07845f", "center-nlos": "#d55e00"}
    link_labels = {"center-los": "Geometric LOS (clear)", "center-nlos": "Geometric NLOS (blocked)"}
    all_x: list[float] = []
    all_y: list[float] = []
    for link_id in ("center-los", "center-nlos"):
        link = links[link_id]
        tx = [float(value) for value in link["tx_position_m"]]
        rx = [float(value) for value in link["rx_position_m"]]
        color = link_colors[link_id]
        all_x.extend((tx[0], rx[0]))
        all_y.extend((tx[1], rx[1]))
        line_style = "-" if link_id == "center-los" else "--"
        ax3d.plot(
            [tx[0], rx[0]],
            [tx[1], rx[1]],
            [tx[2], rx[2]],
            color=color,
            linewidth=2.4,
            linestyle=line_style,
            zorder=10,
        )
        ax2d.plot(
            [tx[0], rx[0]],
            [tx[1], rx[1]],
            color=color,
            linewidth=2.0,
            linestyle=line_style,
            zorder=4,
        )
        for axis in (ax3d,):
            axis.scatter(
                [tx[0]],
                [tx[1]],
                [tx[2]],
                marker="^",
                color="#4d477f",
                edgecolor="white",
                linewidth=1.0,
                s=115,
                depthshade=False,
                zorder=12,
            )
            axis.scatter(
                [rx[0]],
                [rx[1]],
                [rx[2]],
                marker="o",
                color=color,
                edgecolor="white",
                linewidth=0.9,
                s=85,
                depthshade=False,
                zorder=12,
            )
        ax2d.scatter(
            [tx[0]],
            [tx[1]],
            marker="^",
            color="#4d477f",
            edgecolor="white",
            linewidth=0.9,
            s=125,
            zorder=6,
        )
        ax2d.scatter(
            [rx[0]],
            [rx[1]],
            marker="o",
            color=color,
            edgecolor="white",
            linewidth=0.9,
            s=95,
            zorder=6,
        )
        ax2d.annotate(
            "TX (18 m)",
            (tx[0], tx[1]),
            xytext=(-3, 9),
            textcoords="offset points",
            fontsize=8,
            color="#393453",
            weight="bold",
            ha="center",
            zorder=7,
        )
        ax2d.annotate(
            "LOS RX" if link_id == "center-los" else "NLOS RX",
            (rx[0], rx[1]),
            xytext=(7, 5),
            textcoords="offset points",
            fontsize=8,
            color=color,
            weight="bold",
            zorder=7,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=link_colors[link_id],
            linewidth=2.5,
            linestyle="-" if link_id == "center-los" else "--",
            label=link_labels[link_id],
        )
        for link_id in ("center-los", "center-nlos")
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#98a6ad",
            markeredgecolor="#52616a",
            markersize=9,
            label="Building mesh (concrete / glass)",
        )
    )
    ax3d.set(
        title="3D scene: actual building meshes and audited links",
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="height z (m)",
        xlim=(-108, 108),
        ylim=(-108, 108),
        zlim=(0, 66),
    )
    ax3d.set_box_aspect((1.0, 1.0, 0.62))
    ax3d.view_init(elev=24, azim=-57)
    ax3d.grid(True, alpha=0.2)
    ax3d.legend(handles=handles, loc="upper left", fontsize=8, framealpha=0.96)

    ax2d.set(
        title="Top-down detail: the NLOS ray crosses a building",
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(-55, 58),
        ylim=(-70, 9),
        aspect="equal",
    )
    ax2d.grid(True, alpha=0.25, linestyle=":")
    ax2d.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.96)
    ax3d.yaxis.labelpad = 1
    fig.suptitle(
        "City43 path-cap experiment: one transmitter, two receiver positions",
        fontsize=14,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.005,
        "Left: all 16 City43 buildings rendered from their scene OBJ meshes. Right: close-up of the two audited links. "
        "LOS/NLOS denotes the direct-path geometry; "
        "materials use synthetic S=0.5 for this solver-cap experiment.",
        ha="center",
        fontsize=9,
        color="#40484d",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    png_path = output.with_suffix(".png")
    svg_path = output.with_suffix(".svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    result = {
        "schema": "loslab.rt_city43_cap_sweep_scene_figure.v1",
        "scene": {
            "path": summary["scene"]["name"],
            "scene_tree_sha256": summary["scene"]["scene_tree_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "building_mesh_count": len(buildings),
        },
        "input_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
        "links": {
            link_id: {
                "tx_position_m": links[link_id]["tx_position_m"],
                "rx_position_m": links[link_id]["rx_position_m"],
                "geometric_visibility_state": links[link_id]["geometric_visibility_state"],
            }
            for link_id in ("center-los", "center-nlos")
        },
        "figure": {
            "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
            "svg": {"path": str(svg_path), "sha256": sha256_file(svg_path)},
        },
    }
    atomic_json(output.with_suffix(".json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="path-cap summary JSON")
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = make_figure(args.summary, args.output_prefix, args.repo_root)
    print(f"Saved {result['scene']['building_mesh_count']} building meshes to {args.output_prefix}")


if __name__ == "__main__":
    main()
