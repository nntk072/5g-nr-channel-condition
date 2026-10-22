"""Render a 3D endpoint-position QA figure from a ray-traced ChannelDataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import atomic_json, sha256_file, versions
from datasets.channel import ChannelDataset


def _scene_name(record: dict[str, Any]) -> str:
    """Return a stable short scene name for built-in names and scene-file paths."""
    value = str(record.get("scene", "")).replace("\\", "/")
    name = Path(value).stem
    if not name:
        raise ValueError("every metadata record must identify its scene")
    return name


def _resolve_scene_reference(name: str, raw_reference: str, rt: Any) -> Any:
    """Resolve embedded scene paths in this checkout, then Sionna built-ins."""
    path = Path(raw_reference).expanduser()
    if path.is_file():
        return path
    parts = path.parts
    if "scenes" in parts:
        relative = Path(*parts[parts.index("scenes") :])
        candidate = Path(__file__).resolve().parents[1] / relative
        if candidate.is_file():
            return candidate
    return getattr(rt.scene, name, name)


def _plot(dataset: ChannelDataset, output: Path, input_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as error:
        raise RuntimeError("Install matplotlib to render the endpoint-position figure") from error

    grouped: dict[tuple[str, tuple[float, float, float]], list[tuple[dict[str, Any], int]]] = {}
    scene_refs: dict[str, str] = {}
    for record, target in zip(dataset.metadata, dataset.y, strict=True):
        tx_key = tuple(float(value) for value in record.get("tx", ()))
        if len(tx_key) != 3:
            raise ValueError("every metadata record must have a 3D TX position")
        name = _scene_name(record)
        grouped.setdefault((name, tx_key), []).append((record, int(target)))
        scene_refs.setdefault(name, str(record.get("scene", name)))
    if not grouped:
        raise ValueError("dataset has no scene metadata")

    try:
        from integrations.sionna.backend import activate_rt_backend, select_rt_backend
    except ImportError as error:
        raise RuntimeError("Install Sionna RT 2.2 to render scene geometry") from error
    requested_variants = {str(record.get("mitsuba_variant", "auto")) for record in dataset.metadata}
    requested_variant = next(iter(requested_variants)) if len(requested_variants) == 1 else "auto"
    backend = select_rt_backend(requested_variant)
    activate_rt_backend(backend)
    import sionna.rt as rt

    meshes_by_scene: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    mesh_summaries: dict[str, dict[str, int]] = {}
    for name, raw_ref in scene_refs.items():
        scene_ref = _resolve_scene_reference(name, raw_ref, rt)
        scene = rt.load_scene(scene_ref)
        meshes: list[tuple[np.ndarray, np.ndarray]] = []
        vertex_count = 0
        triangle_count = 0
        for scene_object in scene.objects.values():
            mesh = scene_object.mi_mesh
            vertices = np.asarray(mesh.vertex_positions_buffer().numpy(), dtype=float).reshape(
                -1, 3
            )
            faces = np.asarray(mesh.faces_buffer().numpy(), dtype=np.int64).reshape(-1, 3)
            if not len(vertices) or not len(faces) or float(np.ptp(vertices[:, 2])) < 1e-3:
                continue
            meshes.append((vertices, faces))
            vertex_count += len(vertices)
            triangle_count += len(faces)
        if not meshes:
            raise ValueError(f"Sionna scene {name} contains no renderable triangle meshes")
        meshes_by_scene[name] = meshes
        mesh_summaries[name] = {
            "object_mesh_count": len(meshes),
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
        }

    keys = sorted(grouped)
    columns = min(3, len(keys))
    rows = math.ceil(len(keys) / columns)
    fig = plt.figure(figsize=(5.3 * columns, 5.0 * rows))
    fig.suptitle("Actual TX/RX positions in the Sionna RT dataset", y=0.99, fontsize=17)

    scene_summaries: list[dict[str, Any]] = []
    for panel, (name, tx_key) in enumerate(keys, start=1):
        records = grouped[(name, tx_key)]
        tx = np.asarray(records[0][0]["tx"], dtype=float)
        rx = np.asarray([record["rx"] for record, _ in records], dtype=float)
        if tx.shape != (3,) or rx.shape != (len(records), 3):
            raise ValueError(f"scene {name} has malformed TX/RX coordinates")
        if not np.isfinite(tx).all() or not np.isfinite(rx).all():
            raise ValueError(f"scene {name} has non-finite TX/RX coordinates")
        axis = fig.add_subplot(rows, columns, panel, projection="3d")
        scene_meshes = meshes_by_scene[name]
        geometry = np.concatenate([vertices for vertices, _ in scene_meshes])
        for vertices, faces in scene_meshes:
            axis.add_collection3d(
                Poly3DCollection(
                    vertices[faces],
                    facecolor="#94a3b8",
                    edgecolor="#475569",
                    linewidth=0.08,
                    alpha=0.2,
                    zsort="average",
                )
            )
        labels = np.asarray([label for _, label in records])
        for label, color, marker in ((1, "#16803c", "o"), (0, "#d97706", "x")):
            points = rx[labels == label]
            if len(points):
                axis.scatter(*points.T, c=color, marker=marker, s=27, alpha=0.9, depthshade=False)

        endpoints = np.vstack((rx, tx[None, :]))
        pad = max(2.0, float(np.ptp(endpoints[:, :2], axis=0).max()) * 0.08)
        x_limits = (float(endpoints[:, 0].min() - pad), float(endpoints[:, 0].max() + pad))
        y_limits = (float(endpoints[:, 1].min() - pad), float(endpoints[:, 1].max() + pad))
        z_min = min(0.0, float(geometry[:, 2].min()))
        z_max = max(3.0, float(geometry[:, 2].max()) + 1.0, float(endpoints[:, 2].max()) + 1.0)
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_zlim(z_min, z_max)
        axis.set_box_aspect(
            (
                max(1.0, x_limits[1] - x_limits[0]),
                max(1.0, y_limits[1] - y_limits[0]),
                max(1.0, z_max - z_min),
            )
        )
        axis.scatter(
            *tx,
            c="#1d4ed8",
            marker="*",
            s=165,
            edgecolor="white",
            linewidth=0.7,
            depthshade=False,
        )
        axis.view_init(elev=24, azim=-58)
        same_scene_txs = sum(scene_name == name for scene_name, _ in keys)
        title = name.replace("_", " ")
        if same_scene_txs > 1:
            title += f" · TX at ({tx[0]:g}, {tx[1]:g}, {tx[2]:g}) m"
        axis.set_title(title, fontsize=12, pad=3)
        axis.set_xlabel("x (m)", labelpad=-4)
        axis.set_ylabel("y (m)", labelpad=-4)
        axis.set_zlabel("height z (m)", labelpad=-2)
        axis.set_zticklabels([])
        axis.tick_params(labelsize=7, pad=0)
        axis.grid(True, alpha=0.22)
        axis.text2D(
            0.5,
            0.93,
            f"TX z={tx[2]:g} m | RX z={rx[0, 2]:g} m",
            transform=axis.transAxes,
            ha="center",
            fontsize=8,
            color="#374151",
        )
        scene_summaries.append(
            {
                "scene": name,
                "tx_position_m": list(tx_key),
                "sample_count": len(records),
                "geometric_los_count": int(np.sum(labels == 1)),
                "geometric_nlos_count": int(np.sum(labels == 0)),
                "rx_position_min_m": rx.min(axis=0).tolist(),
                "rx_position_max_m": rx.max(axis=0).tolist(),
                "scene_mesh": mesh_summaries[name],
            }
        )

    legend = [
        Patch(
            facecolor="#94a3b8",
            edgecolor="#475569",
            alpha=0.2,
            label="Sionna building/obstacle mesh",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#1d4ed8",
            markersize=13,
            label="TX: fixed transmitter per panel",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#16803c",
            markersize=8,
            label="RX: geometric LOS",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color="#d97706",
            linestyle="None",
            markersize=8,
            label="RX: geometric NLOS",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Building/obstacle meshes are shown translucently; propagation paths are not shown. "
        "Class-balanced samples do not represent deployment prevalence.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(top=0.87, bottom=0.10, left=0.03, right=0.97, wspace=0.15, hspace=0.24)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    atomic_json(
        output.with_suffix(output.suffix + ".json"),
        {
            "schema": "loslab.rt_endpoint_positions_figure.v1",
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "renderer_sha256": sha256_file(Path(__file__)),
            "figure_path": str(output),
            "figure_sha256": sha256_file(output),
            "rt_backend": backend.as_dict(),
            "dataset_label_semantics": dataset.label_semantics,
            "sample_count": len(dataset.y),
            "channel_shape": list(dataset.h.shape),
            "limitations": [
                "Exact Sionna scene meshes are shown; path polylines are not included.",
                "Any class balancing or candidate-selection policy is inherited from the source dataset.",
                "This view is a geometry QA artifact, not evidence of model performance.",
            ],
            "scenes": scene_summaries,
            "runtime_versions": versions(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Saved non-pickle ChannelDataset NPZ")
    parser.add_argument("output", type=Path, help="New PNG output path")
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".json")
    if args.output.exists() or sidecar.exists():
        parser.error("refusing to overwrite an existing figure or its provenance sidecar")
    dataset = ChannelDataset.load(args.dataset)
    if dataset.label_semantics != "geometric_direct_path_oracle":
        parser.error("dataset must use geometric_direct_path_oracle label semantics")
    _plot(dataset, args.output, args.dataset)
    print(f"Saved endpoint-position figure to {args.output}")


if __name__ == "__main__":
    main()
