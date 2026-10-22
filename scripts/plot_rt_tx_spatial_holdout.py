"""Plot the audited City43 spatial holdout with its actual building geometry.

Usage:
    uv run --no-sync --locked python scripts/plot_rt_tx_spatial_holdout.py \
      outputs/rt-urban43-tx-spatial-holdout-20261022-v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets.channel import ChannelDataset
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CLASS_COLORS = {0: "#168a83", 1: "#e27a32"}
TX_COLORS = {"west_x3": "#5b4b9a", "center_x0": "#2773a8", "east_x3": "#b54c6b"}
TX_LABELS = {"west_x3": "West", "center_x0": "Center", "east_x3": "East"}


def _read_buildings(manifest: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return [
        item
        for item in payload["buildings_and_vehicles"]
        if str(item["name"]).startswith("building_")
    ]


def _cuboid_faces(bounds: list[float]) -> list[list[tuple[float, float, float]]]:
    x0, y0, z0, x1, y1, z1 = bounds
    points = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    return [
        [points[i] for i in face]
        for face in (
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
    ]


def make_figure(run_root: Path, output: Path) -> None:
    """Render a 3D overview and a top-down view of the exact holdout geometry."""
    dataset = ChannelDataset.load(run_root / "nr-dmrs-observations.npz")
    scene_path = Path(str(dataset.metadata[0]["scene"]))
    buildings = _read_buildings(scene_path.with_suffix(".manifest.json"))
    fig = plt.figure(figsize=(15.2, 7.4), layout="constrained")
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    for building in buildings:
        bounds = [float(value) for value in building["bounds"]]
        ax3d.add_collection3d(
            Poly3DCollection(
                _cuboid_faces(bounds),
                facecolor="#8795a1",
                edgecolor="#53616c",
                linewidth=0.42,
                alpha=0.28,
            )
        )
        x0, y0, _, x1, y1, _ = bounds
        ax2d.fill(
            [x0, x1, x1, x0],
            [y0, y0, y1, y1],
            facecolor="#aab4bb",
            edgecolor="#59666e",
            linewidth=0.55,
            alpha=0.76,
            zorder=1,
        )

    ids = np.asarray(
        [str(metadata["transmitter_id"]) for metadata in dataset.metadata], dtype=object
    )
    rx = np.asarray([metadata["rx"] for metadata in dataset.metadata], dtype=float)
    tx = np.asarray([metadata["tx"] for metadata in dataset.metadata], dtype=float)
    labels = np.asarray(dataset.y, dtype=int)
    tx_positions: dict[str, tuple[float, float, float]] = {}

    for transmitter_id in ("west_x3", "center_x0", "east_x3"):
        mask = ids == transmitter_id
        tx_position = tx[np.flatnonzero(mask)[0]]
        tx_positions[transmitter_id] = tuple(tx_position)
        first_metadata = dataset.metadata[np.flatnonzero(mask)[0]]
        center = np.asarray(first_metadata["scene_config"]["rx_center"], dtype=float)
        theta = np.linspace(0.0, 2.0 * np.pi, 240)
        circle_x = center[0] + 25.0 * np.cos(theta)
        circle_y = center[1] + 25.0 * np.sin(theta)
        ax2d.plot(circle_x, circle_y, color=TX_COLORS[transmitter_id], linewidth=1.4)
        ax2d.scatter(
            center[0],
            center[1],
            marker="+",
            color=TX_COLORS[transmitter_id],
            s=55,
            linewidth=1.8,
            zorder=4,
        )
        ax2d.annotate(
            f"{TX_LABELS[transmitter_id]} RX zone",
            (center[0], center[1] + 27.5),
            color=TX_COLORS[transmitter_id],
            fontsize=8,
            ha="center",
        )
        ax3d.plot(
            circle_x,
            circle_y,
            np.full_like(circle_x, 0.35),
            color=TX_COLORS[transmitter_id],
            linewidth=1.25,
            alpha=0.8,
        )
        ax3d.scatter(
            [tx_position[0]],
            [tx_position[1]],
            [tx_position[2]],
            marker="^",
            color=TX_COLORS[transmitter_id],
            edgecolor="white",
            linewidth=0.8,
            s=95,
            depthshade=False,
        )
        ax3d.text(
            tx_position[0],
            tx_position[1],
            tx_position[2] + 3.0,
            f"{TX_LABELS[transmitter_id]} TX",
            color=TX_COLORS[transmitter_id],
            fontsize=8,
        )

    class_names = {0: "Geometric LOS", 1: "Geometric NLOS"}
    for label, color in CLASS_COLORS.items():
        mask = labels == label
        ax3d.scatter(
            rx[mask, 0],
            rx[mask, 1],
            rx[mask, 2],
            color=color,
            edgecolor="white",
            linewidth=0.25,
            s=22,
            alpha=0.96,
            depthshade=False,
        )
        ax2d.scatter(
            rx[mask, 0],
            rx[mask, 1],
            color=color,
            edgecolor="white",
            linewidth=0.18,
            s=18,
            alpha=0.83,
            label=class_names[label],
            zorder=3,
        )

    for transmitter_id, tx_position in tx_positions.items():
        ax2d.scatter(
            tx_position[0],
            tx_position[1],
            marker="^",
            color=TX_COLORS[transmitter_id],
            edgecolor="white",
            linewidth=0.8,
            s=105,
            zorder=5,
        )
        ax2d.annotate(
            f"{TX_LABELS[transmitter_id]} TX",
            tx_position[:2],
            xytext=(5, 5),
            textcoords="offset points",
            color=TX_COLORS[transmitter_id],
            fontsize=8,
            weight="bold",
            zorder=6,
        )

    ax3d.set(
        title="Actual 3D scene geometry and sampled links",
        xlabel="x (m)",
        ylabel="y (m)",
        zlabel="height z (m)",
        xlim=(-110, 110),
        ylim=(-110, 110),
        zlim=(0, 72),
    )
    ax3d.set_box_aspect((1, 1, 0.68))
    ax3d.view_init(elev=25, azim=-58)
    ax3d.grid(True, alpha=0.2)
    legend_items = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_COLORS[label],
            markeredgecolor="white",
            markersize=7,
            label=class_names[label],
        )
        for label in (0, 1)
    ]
    legend_items.extend(
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=TX_COLORS[tx_id],
            markeredgecolor="white",
            markersize=8,
            label=f"{TX_LABELS[tx_id]} transmitter",
        )
        for tx_id in ("west_x3", "center_x0", "east_x3")
    )
    ax3d.legend(handles=legend_items, loc="upper left", fontsize=8, framealpha=0.95)

    ax2d.set(
        title="Top-down: building footprints and RX samples",
        xlabel="x (m)",
        ylabel="y (m)",
        xlim=(-110, 110),
        ylim=(-110, 110),
        aspect="equal",
    )
    ax2d.grid(True, alpha=0.25, linestyle=":")
    ax2d.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        "City43 spatial holdout: three TX locations with disjoint RX regions",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "Synthetic 4×4 building scene; 120 Sionna RT links (20 LOS + 20 NLOS per region). "
        "Labels are geometric direct-path oracles, not measured radio ground truth.",
        ha="center",
        fontsize=9,
        color="#40484d",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    svg_output = output.with_suffix(".svg")
    fig.savefig(svg_output, bbox_inches="tight")
    svg_text = svg_output.read_text(encoding="utf-8")
    svg_output.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/research/figures/rt-urban43-spatial-holdout-20261022.png"),
    )
    args = parser.parse_args()
    make_figure(args.run_root, args.output)
    print(f"Saved {args.output} and {args.output.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
