"""Render stratified stored RT links with scene geometry and traced path overlays."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import atomic_json, sha256_file, versions
from datasets.channel import ChannelDataset
from integrations.sionna.adapter import count_valid_paths, has_valid_path

_INTERACTION_NAMES = {
    1: "specular",
    2: "diffuse",
    4: "refraction",
    8: "diffraction",
}
_INTERACTION_COLORS = {
    "LOS candidate": "#16a34a",
    "specular": "#dc2626",
    "diffuse": "#ea580c",
    "refraction": "#2563eb",
    "diffraction": "#9333ea",
    "mixed": "#7c3aed",
}


def extract_path_polylines(
    sources: np.ndarray,
    targets: np.ndarray,
    vertices: np.ndarray,
    interactions: np.ndarray,
    valid: np.ndarray,
) -> list[dict[str, Any]]:
    """Convert one-link Paths tensors into ordered world-coordinate polylines.

    Sionna stores path interactions and their positions along the leading
    maximum-depth axis. Zero-valued interaction entries are padding, so they
    must not be mistaken for vertices at the origin.
    """
    source = np.asarray(sources, dtype=float)
    target = np.asarray(targets, dtype=float)
    points = np.asarray(vertices, dtype=float)
    interaction = np.asarray(interactions)
    mask = np.asarray(valid, dtype=bool).reshape(-1)
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("single-link source and target must each have shape (3,)")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("vertices must have shape [depth, paths, 3]")
    if interaction.shape != points.shape[:2] or points.shape[1] != len(mask):
        raise ValueError("vertices, interactions, and valid-mask path dimensions differ")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("source and target coordinates must be finite")
    if not np.all(np.isfinite(points)):
        raise ValueError("path vertices must be finite")

    result: list[dict[str, Any]] = []
    for path_index in np.flatnonzero(mask):
        codes = [int(value) for value in interaction[:, path_index] if int(value) != 0]
        nodes = [source]
        nodes.extend(
            points[depth, path_index]
            for depth, code in enumerate(interaction[:, path_index])
            if int(code) != 0
        )
        nodes.append(target)
        names = sorted({_INTERACTION_NAMES.get(code, f"unknown-{code}") for code in codes})
        label = "LOS candidate" if not names else (names[0] if len(names) == 1 else "mixed")
        result.append(
            {
                "path_index": int(path_index),
                "coordinates_m": np.asarray(nodes, dtype=float),
                "interactions": names,
                "interaction_label": label,
                "direct_candidate": not codes,
            }
        )
    return result


def select_stratified_indices(
    metadata: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    labels: np.ndarray,
    *,
    examples_per_class: int = 2,
) -> list[int]:
    """Pick reproducible low-to-high path-count quantiles within each class."""
    y = np.asarray(labels).reshape(-1)
    if len(metadata) != len(y) or len(y) == 0:
        raise ValueError("metadata and labels must be nonempty and aligned")
    if (
        isinstance(examples_per_class, bool)
        or not isinstance(examples_per_class, int)
        or examples_per_class < 1
    ):
        raise ValueError("examples_per_class must be a positive integer")
    if set(np.unique(y).tolist()) != {0, 1}:
        raise ValueError("path QA requires both binary geometric-label classes")
    if any("valid_paths_full" not in row for row in metadata):
        raise ValueError("every row needs its stored valid_paths_full count")

    selected: list[int] = []
    for label in (0, 1):
        candidates = np.flatnonzero(y == label).tolist()
        candidates.sort(key=lambda index: (int(metadata[index]["valid_paths_full"]), index))
        quantiles = np.linspace(0, len(candidates) - 1, min(examples_per_class, len(candidates)))
        selected.extend(candidates[int(round(position))] for position in quantiles)
    return selected


def _scene_tree_sha256(scene_xml: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(scene_xml.parent.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(scene_xml.parent).as_posix().encode("utf8"))
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _object_bounds(obj: Any) -> tuple[np.ndarray, np.ndarray]:
    bbox = obj.mi_mesh.bbox()
    lower = np.asarray(bbox.min.numpy(), dtype=float).reshape(3)
    upper = np.asarray(bbox.max.numpy(), dtype=float).reshape(3)
    return lower, upper


def _draw_geometry(axis: Any, objects: Any, projection: tuple[int, int]) -> None:
    from matplotlib.patches import Rectangle

    x_axis, y_axis = projection
    for name, obj in objects.items():
        if "ground" in name.lower():
            continue
        lower, upper = _object_bounds(obj)
        width = float(upper[x_axis] - lower[x_axis])
        height = float(upper[y_axis] - lower[y_axis])
        if width <= 0 or height <= 0:
            # Thin walls/screens are visually represented at a small finite
            # width while retaining the exact scene coordinates in metadata.
            center = (float(lower[x_axis]) + float(upper[x_axis])) / 2
            width = max(width, 0.015)
            lower[x_axis] = center - width / 2
        if width > 250 or height > 250:
            continue
        axis.add_patch(
            Rectangle(
                (lower[x_axis], lower[y_axis]),
                width,
                height,
                facecolor="#64748b",
                edgecolor="#334155",
                linewidth=0.6,
                alpha=0.16,
                zorder=1,
            )
        )
        if width >= 1.0 and height >= 1.0:
            axis.text(
                lower[x_axis] + width / 2,
                lower[y_axis] + height / 2,
                name,
                fontsize=5,
                color="#475569",
                ha="center",
                va="center",
            )


def _render_overlay(
    *,
    scene: Any,
    paths: Any,
    polylines: list[dict[str, Any]],
    tx: np.ndarray,
    rx: np.ndarray,
    label: int,
    oracle_direct: bool,
    image_path: Path,
    scene_name: str,
    sample_index: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    plan = fig.add_subplot(1, 2, 1)
    elevation = fig.add_subplot(1, 2, 2)
    _draw_geometry(plan, scene.objects, (0, 1))
    _draw_geometry(elevation, scene.objects, (0, 2))
    for axis, projection in ((plan, (0, 1)), (elevation, (0, 2))):
        for path in polylines:
            coordinates = path["coordinates_m"]
            axis.plot(
                coordinates[:, projection[0]],
                coordinates[:, projection[1]],
                color=_INTERACTION_COLORS[path["interaction_label"]],
                alpha=0.48,
                linewidth=0.9,
                zorder=2,
            )
        axis.scatter(
            [tx[projection[0]], rx[projection[0]]],
            [tx[projection[1]], rx[projection[1]]],
            c=["#2563eb", "#16a34a"],
            edgecolors="white",
            linewidths=0.7,
            s=48,
            zorder=5,
        )
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, alpha=0.16)
        axis.set_xlabel(("x", "y", "z")[projection[0]] + " (m)")
        axis.set_ylabel(("x", "y", "z")[projection[1]] + " (m)")
    plan.set_title("Top view")
    elevation.set_title("Elevation (x–z)")
    handles = [
        plt.Line2D([], [], color=color, label=name, linewidth=2)
        for name, color in _INTERACTION_COLORS.items()
    ]
    handles.extend(
        [
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="#2563eb", label="TX"),
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="#16a34a", label="RX"),
        ]
    )
    fig.legend(handles=handles, loc="outside lower center", ncols=7, fontsize=8)
    direct_full = sum(path["direct_candidate"] for path in polylines)
    fig.suptitle(
        f"{scene_name} · sample {sample_index} · geometric label={label} · "
        f"depth-0 oracle direct={oracle_direct} · full direct candidates={direct_full} · "
        f"{count_valid_paths(paths)} valid paths\n"
        "Approximate AABB geometry/path overlay for QA; not physical validation"
    )
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def render_dataset_qa(
    input_path: Path,
    output_dir: Path,
    *,
    examples_per_class: int = 2,
) -> dict[str, Any]:
    """Rerun and render median-path-count examples from both geometric labels."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite RT path QA directory: {output_dir}")
    dataset = ChannelDataset.load(input_path)
    metadata = list(dataset.metadata)
    required = {"scene", "tx", "rx", "seed", "max_depth", "grid_config", "scene_config"}
    if any(not required.issubset(row) for row in metadata):
        raise ValueError("dataset metadata is missing RT link configuration fields")
    if len({str(row["scene"]) for row in metadata}) != 1:
        raise ValueError("one input dataset must contain links from one scene")
    if len({tuple(row["tx"]) for row in metadata}) != 1:
        raise ValueError("one input dataset must contain a single transmitter position")

    try:
        from integrations.sionna.backend import activate_rt_backend, select_rt_backend
        from integrations.sionna.candidate_audit import instrument_path_solver_candidate_counter

        variant = str(metadata[0]["mitsuba_variant"])
        if any(str(row["mitsuba_variant"]) != variant for row in metadata):
            raise ValueError("dataset contains multiple Mitsuba variants")
        backend_selection = select_rt_backend(variant)
        activate_rt_backend(backend_selection)
        import sionna.rt as rt
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            f"Could not initialize the dataset's Sionna RT backend: {error}"
        ) from error

    scene_name = str(metadata[0]["scene"])
    scene_ref = getattr(rt.scene, scene_name, scene_name)
    scene_xml = Path(str(scene_ref)).resolve()
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Sionna scene XML not found: {scene_xml}")
    scene = load_scene(str(scene_xml))
    first_row = metadata[0]
    grid = first_row["grid_config"]
    scene.frequency = float(grid["carrier_hz"])
    scene.bandwidth = int(grid["n_subcarriers"]) * float(grid["spacing_hz"])
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter(name="path_qa_tx", position=list(first_row["tx"])))
    receiver = Receiver(name="path_qa_rx", position=list(metadata[0]["rx"]))
    scene.add(receiver)
    solver = PathSolver(deterministic=True)
    candidate_counter = instrument_path_solver_candidate_counter(solver)
    selected = select_stratified_indices(
        metadata,
        dataset.y,
        examples_per_class=examples_per_class,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    tx = np.asarray(first_row["tx"], dtype=float)
    report_links: list[dict[str, Any]] = []

    for sample_index in selected:
        row = metadata[sample_index]
        rx = np.asarray(row["rx"], dtype=float)
        receiver.position = rx.tolist()
        seed = int(row["seed"])
        max_depth = int(row["max_depth"])
        kwargs = {
            "los": True,
            "specular_reflection": True,
            "diffuse_reflection": True,
            "refraction": True,
            "diffraction": False,
            "samples_per_src": int(row["scene_config"]["samples_per_src"]),
            "max_num_paths_per_src": int(
                row["scene_config"].get("max_num_paths_per_src", 1_000_000)
            ),
            "seed": seed,
        }
        paths = solver(scene, max_depth=max_depth, **kwargs)
        candidate_counts = candidate_counter.last_candidate_path_counts_per_source
        if candidate_counts is None:
            raise RuntimeError("RT path QA did not expose per-source candidate counts")
        if any(count >= kwargs["max_num_paths_per_src"] for count in candidate_counts):
            raise RuntimeError(
                "RT path QA encountered a candidate path cap-filled output: "
                f"sample={sample_index}, counts={candidate_counts}, "
                f"cap={kwargs['max_num_paths_per_src']}"
            )
        oracle = solver(
            scene,
            max_depth=0,
            los=True,
            specular_reflection=False,
            diffuse_reflection=False,
            refraction=False,
            diffraction=False,
            samples_per_src=int(row["scene_config"]["samples_per_src"]),
            max_num_paths_per_src=kwargs["max_num_paths_per_src"],
            seed=seed,
        )
        oracle_direct = has_valid_path(oracle)
        expected_label = int(dataset.y[sample_index])
        if oracle_direct != bool(expected_label):
            raise RuntimeError(
                f"Stored geometric label differs from depth-zero replay for {scene_name} "
                f"sample {sample_index}: stored={expected_label}, replay={oracle_direct}"
            )

        sources = np.asarray(paths.sources.numpy(), dtype=float).reshape(3, -1)[:, 0]
        targets = np.asarray(paths.targets.numpy(), dtype=float).reshape(3, -1)[:, 0]
        vertices = np.asarray(paths.vertices.numpy(), dtype=float).reshape(
            int(paths.vertices.shape[0]), -1, 3
        )
        interactions = np.asarray(paths.interactions.numpy()).reshape(
            int(paths.interactions.shape[0]), -1
        )
        valid = np.asarray(paths.valid.numpy(), dtype=bool).reshape(-1)
        polylines = extract_path_polylines(sources, targets, vertices, interactions, valid)
        direct_candidates = sum(path["direct_candidate"] for path in polylines)
        if oracle_direct and direct_candidates == 0:
            raise RuntimeError(
                f"Depth-zero oracle direct path missing from full solve for {scene_name} "
                f"sample {sample_index}"
            )
        if not oracle_direct and direct_candidates > 0:
            raise RuntimeError(
                f"Full solve indicates a direct path but depth-zero oracle does not for "
                f"{scene_name} sample {sample_index}"
            )

        image_name = f"{Path(input_path).stem}-sample-{sample_index:03d}-label-{expected_label}.png"
        image_path = output_dir / image_name
        _render_overlay(
            scene=scene,
            paths=paths,
            polylines=polylines,
            tx=tx,
            rx=rx,
            label=expected_label,
            oracle_direct=oracle_direct,
            image_path=image_path,
            scene_name=scene_name,
            sample_index=sample_index,
        )
        interaction_counts = Counter(name for path in polylines for name in path["interactions"])
        report_links.append(
            {
                "sample_index": sample_index,
                "group": str(dataset.group[sample_index]),
                "label": expected_label,
                "oracle_direct_replayed": oracle_direct,
                "full_valid_path_count": count_valid_paths(paths),
                "candidate_paths_per_source_before_validation": list(candidate_counts),
                "max_num_paths_per_src": kwargs["max_num_paths_per_src"],
                "full_direct_candidate_count": direct_candidates,
                "interaction_counts": dict(sorted(interaction_counts.items())),
                "ray_budget_samples_per_src": kwargs["samples_per_src"],
                "solver_seed": seed,
                "max_depth": max_depth,
                "rx_position_m": rx.tolist(),
                "image": image_name,
                "image_sha256": sha256_file(image_path),
            }
        )

    result = {
        "schema": "loslab.rt_path_visual_qa.v1",
        "evidence_grade": "VISUAL_SIMULATOR_QA_NOT_PHYSICAL_VALIDATION",
        "input": {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "links": len(dataset.y),
            "class_counts": {
                "0": int(np.count_nonzero(dataset.y == 0)),
                "1": int(np.count_nonzero(dataset.y == 1)),
            },
        },
        "scene": {
            "name": scene_name,
            "xml_sha256": sha256_file(scene_xml),
            "scene_tree_sha256": _scene_tree_sha256(scene_xml),
            "mitsuba_variant": variant,
            "rt_backend_selection": backend_selection.as_dict(),
        },
        "versions": versions(),
        "selection": {
            "policy": "Deterministic path-count quantiles selected separately within each geometric label; labels are not used to modify solver or visualization settings.",
            "examples_per_class": examples_per_class,
            "sample_indices": selected,
        },
        "label_replay_matches": True,
        "full_solver_direct_path_matches_depth_zero_oracle": True,
        "geometry_display": "Axis-aligned bounding rectangles projected from each Mitsuba object mesh bbox; the exact geometry and ray tracer remain authoritative.",
        "links": report_links,
        "limitations": [
            "The figures are a manual-review aid; AABB projections can hide mesh topology and path/label correctness still requires expert scene review.",
            "A depth-zero replay is a simulator oracle, not independently surveyed physical ground truth.",
            "The images do not qualify material parameters, stochastic convergence, receiver detectability, or field propagation.",
        ],
    }
    report_path = output_dir / "manifest.json"
    atomic_json(report_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--examples-per-class", type=int, default=2)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite RT path QA directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    reports = [
        render_dataset_qa(
            input_path,
            args.output_dir / input_path.stem,
            examples_per_class=args.examples_per_class,
        )
        for input_path in args.inputs
    ]
    atomic_json(
        args.output_dir / "manifest.json",
        {
            "schema": "loslab.rt_six_scene_path_visual_qa.v1",
            "evidence_grade": "VISUAL_SIMULATOR_QA_NOT_PHYSICAL_VALIDATION",
            "sources": [report["input"]["sha256"] for report in reports],
            "scene_reports": [
                {
                    "scene": report["scene"]["name"],
                    "scene_tree_sha256": report["scene"]["scene_tree_sha256"],
                    "report_sha256": sha256_file(
                        args.output_dir / input_path.stem / "manifest.json"
                    ),
                    "rendered_images": len(report["links"]),
                    "direct_oracle_label_matches": report["label_replay_matches"],
                    "full_solver_direct_matches_oracle": report[
                        "full_solver_direct_path_matches_depth_zero_oracle"
                    ],
                    "image_sha256": [link["image_sha256"] for link in report["links"]],
                }
                for input_path, report in zip(args.inputs, reports, strict=True)
            ],
            "versions": versions(),
            "limitations": [
                "Plots are QA evidence, not confirmation of physical scene fidelity or path truth.",
                "Human review and qualified materials are still required before P5 can pass.",
            ],
        },
    )
    print(
        f"Rendered {sum(len(report['links']) for report in reports)} path-QA overlays to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
