"""Plot all audited TX/RX locations over the actual Urban43 building meshes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from artifacts.provenance import atomic_json, sha256_file
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _read_obj(path: Path) -> tuple[np.ndarray, list[list[int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["v"]:
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))
        elif fields[:1] == ["f"]:
            faces.append([int(value.split("/", maxsplit=1)[0]) - 1 for value in fields[1:]])
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh has no vertices or faces: {path}")
    return np.asarray(vertices, dtype=np.float64), faces


def make_figure(audit_path: Path, output_prefix: Path, repo_root: Path) -> dict[str, Any]:
    """Render the audited 120-link sample over all 16 Urban43 building meshes."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("audit_passed"):
        raise ValueError("spatial-holdout label audit did not pass")
    records = audit["records"]
    if len(records) != 120 or any(not row.get("matches") for row in records):
        raise ValueError("expected 120 audited records with independently verified labels")
    scene_path = Path(audit["scene"]["xml_path"])
    if not scene_path.is_absolute():
        scene_path = repo_root / scene_path
    manifest_path = scene_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    buildings = [
        item
        for item in manifest["buildings_and_vehicles"]
        if str(item["name"]).startswith("building_")
    ]
    if len(buildings) != 16:
        raise ValueError(f"expected 16 Urban43 building meshes, found {len(buildings)}")

    tx_by_id: dict[str, tuple[float, float, float]] = {}
    counts: Counter[str] = Counter()
    for row in records:
        tx_id = str(row["transmitter_id"])
        raw_tx = row["tx_position_m"]
        tx = (float(raw_tx[0]), float(raw_tx[1]), float(raw_tx[2]))
        if tx_id in tx_by_id and tx_by_id[tx_id] != tx:
            raise ValueError(f"transmitter {tx_id} changes position within the audit")
        tx_by_id[tx_id] = tx
        counts[tx_id] += 1
    if len(tx_by_id) != 3 or set(counts.values()) != {40}:
        raise ValueError("expected three transmitter sites with 40 receiver samples each")

    figure = plt.figure(figsize=(17, 8.4), layout="constrained")
    grid = figure.add_gridspec(1, 2, width_ratios=(1.35, 1.0))
    ax3d = figure.add_subplot(grid[0, 0], projection="3d")
    ax2d = figure.add_subplot(grid[0, 1])
    material_colors = {
        "concrete": ("#a7b5bd", "#61727a"),
        "glass": ("#83b7cf", "#426b7b"),
        "metal": ("#c7b58c", "#81704b"),
    }

    for building in buildings:
        mesh_path = scene_path.parent / "meshes_00043" / f"{building['name']}.obj"
        vertices, faces = _read_obj(mesh_path)
        polygons = [vertices[face] for face in faces]
        facecolor, edgecolor = material_colors.get(
            building["material"], material_colors["concrete"]
        )
        ax3d.add_collection3d(
            Poly3DCollection(
                polygons,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.25,
                alpha=0.52,
            )
        )
        bounds = [float(value) for value in building["bounds"]]
        x0, y0, _, x1, y1, _ = bounds
        ax2d.fill(
            (x0, x1, x1, x0),
            (y0, y0, y1, y1),
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.7,
            alpha=0.75,
        )

    rx_colors = {"geometric_los": "#07845f", "geometric_nlos": "#d55e00"}
    rx_markers = {"geometric_los": "o", "geometric_nlos": "x"}
    for state, color in rx_colors.items():
        selected = [row for row in records if row["oracle_label"] == state]
        rx = np.asarray([row["rx_position_m"] for row in selected], dtype=np.float64)
        ax3d.scatter(
            rx[:, 0],
            rx[:, 1],
            rx[:, 2],
            c=color,
            marker=rx_markers[state],
            s=22,
            alpha=0.86,
            depthshade=False,
        )
        ax2d.scatter(rx[:, 0], rx[:, 1], c=color, marker=rx_markers[state], s=20, alpha=0.8)

    # Draw one audited visible and one blocked direct segment from each site.
    representative_links: list[dict[str, Any]] = []
    for tx_id, tx in sorted(tx_by_id.items()):
        tx_color = {"west_x3": "#3949ab", "center_x0": "#7b1fa2", "east_x3": "#263238"}.get(
            tx_id, "#3949ab"
        )
        ax3d.scatter(*tx, marker="^", s=135, c=tx_color, edgecolor="white", linewidth=0.8)
        ax2d.scatter(tx[0], tx[1], marker="^", s=110, c=tx_color, edgecolor="white", linewidth=0.7)
        ax3d.text(tx[0], tx[1], tx[2] + 2.0, tx_id, color=tx_color, fontsize=8, weight="bold")
        for state in ("geometric_los", "geometric_nlos"):
            example = next(
                row
                for row in records
                if row["transmitter_id"] == tx_id and row["oracle_label"] == state
            )
            raw_rx = example["rx_position_m"]
            rx_position = [float(raw_rx[0]), float(raw_rx[1]), float(raw_rx[2])]
            color = rx_colors[state]
            style = "-" if state == "geometric_los" else "--"
            for axis in (ax3d,):
                axis.plot(
                    [tx[0], rx_position[0]],
                    [tx[1], rx_position[1]],
                    [tx[2], rx_position[2]],
                    color=color,
                    linestyle=style,
                    linewidth=1.6,
                    alpha=0.9,
                )
            ax2d.plot(
                [tx[0], rx_position[0]],
                [tx[1], rx_position[1]],
                color=color,
                linestyle=style,
                linewidth=1.4,
            )
            representative_links.append(
                {
                    "transmitter_id": tx_id,
                    "sample_index": int(example["sample_index"]),
                    "label": state,
                    "rx_position_m": rx_position,
                }
            )

    endpoint_handles = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor="#3949ab",
            label="TX site (3 sites)",
            markersize=9,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=rx_colors["geometric_los"],
            label="RX: geometric LOS (60)",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=rx_colors["geometric_nlos"],
            label="RX: geometric NLOS (60)",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            color=rx_colors["geometric_los"],
            linestyle="-",
            label="Example clear direct segment",
        ),
        Line2D(
            [0],
            [0],
            color=rx_colors["geometric_nlos"],
            linestyle="--",
            label="Example building-blocked segment",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#a7b5bd",
            markeredgecolor="#61727a",
            label="Actual building mesh",
            markersize=8,
        ),
    ]
    ax3d.set(
        title="3D Urban43 scene and all sampled endpoints",
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="height z (m)",
        xlim=(-108, 108),
        ylim=(-108, 108),
        zlim=(0, 66),
    )
    ax3d.set_box_aspect((1, 1, 0.52))
    ax3d.view_init(elev=30, azim=-58)
    ax3d.legend(handles=endpoint_handles, loc="upper left", fontsize=8, framealpha=0.94)
    ax2d.set(
        title="Top-down layout: 3 TX sites × 40 RX locations",
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(-108, 108),
        ylim=(-108, 108),
        aspect="equal",
    )
    ax2d.grid(True, alpha=0.22, linestyle=":")
    ax2d.legend(handles=endpoint_handles, loc="upper left", fontsize=8, framealpha=0.94)
    figure.suptitle(
        "Urban43 spatial-holdout sample: 120 links, 3 transmitter sites, 16 buildings",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.005,
        "All 120 RX locations use independently audited direct-segment labels (60 LOS / 60 NLOS). "
        "Six example TX–RX segments are drawn; labels describe geometric visibility, not measured propagation.",
        ha="center",
        fontsize=9,
        color="#40484d",
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    figure.savefig(png_path, dpi=220, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    plt.close(figure)
    result = {
        "schema": "loslab.urban43_spatial_holdout_3d_figure.v1",
        "audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "sample_count": len(records),
            "label_counts": dict(Counter(row["oracle_label"] for row in records)),
        },
        "scene": {
            "xml_path": str(scene_path),
            "scene_tree_sha256": audit["scene"]["scene_tree_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
            "building_mesh_count": len(buildings),
        },
        "transmitter_site_count": len(tx_by_id),
        "receiver_samples_per_site": dict(counts),
        "representative_links": representative_links,
        "limitations": [
            "This is one procedural City43 scene, not independent-scene or measured-channel evidence.",
            "Each site has one fixed TX and 40 sampled RX positions; it does not yet test TX-position holdout.",
            "Only six representative direct segments are drawn to keep the 3D figure readable.",
        ],
        "figure": {
            "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
            "svg": {"path": str(svg_path), "sha256": sha256_file(svg_path)},
        },
    }
    atomic_json(output_prefix.with_suffix(".json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("docs/research/results/rt-urban43-spatial-holdout-label-audit-20261022.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("docs/research/figures/rt-urban43-spatial-holdout-3d-20261022"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = make_figure(args.audit, args.output_prefix, args.repo_root)
    print(
        f"Saved {result['scene']['building_mesh_count']} building meshes, "
        f"{result['transmitter_site_count']} TX sites and {result['audit']['sample_count']} RX samples"
    )


if __name__ == "__main__":
    main()
