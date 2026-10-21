"""Re-solve every stored RT link at multiple ray budgets and compare its CFR."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import (
    atomic_json,
    package_source_sha256,
    sha256_file,
    versions,
)
from datasets.channel import ChannelDataset
from integrations.sionna.adapter import GridConfig, count_valid_paths, rt_cfr


def relative_l2_error(value: np.ndarray, reference: np.ndarray) -> float:
    """Return relative complex L2 error against a finite nonzero reference."""
    value = np.asarray(value)
    reference = np.asarray(reference)
    if (
        value.shape != reference.shape
        or not np.iscomplexobj(value)
        or not np.iscomplexobj(reference)
    ):
        raise ValueError("value and reference must be aligned complex arrays")
    denominator = float(np.linalg.norm(reference))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("reference must have finite nonzero energy")
    return float(np.linalg.norm(value - reference) / denominator)


def summarize_link_rows(rows: Sequence[dict[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    """Summarize budget errors and path-count changes by label and budget."""
    if not rows or not budgets:
        raise ValueError("rows and budgets must be nonempty")
    reference_budget = budgets[-1]
    reference_paths = {
        int(row["sample_index"]): int(row["valid_path_count"])
        for row in rows
        if row["samples_per_src"] == reference_budget
    }
    if len(reference_paths) != len({int(row["sample_index"]) for row in rows}):
        raise ValueError("each sample must have exactly one reference-budget row")

    summary: dict[str, Any] = {"reference_samples_per_src": reference_budget, "by_budget": {}}
    for budget in budgets:
        budget_rows = [row for row in rows if row["samples_per_src"] == budget]
        if set(int(row["sample_index"]) for row in budget_rows) != set(reference_paths):
            raise ValueError(f"budget {budget} does not cover the same samples as reference")
        errors = np.asarray(
            [float(row["relative_cfr_l2_to_reference"]) for row in budget_rows], dtype=float
        )
        changed = [
            row
            for row in budget_rows
            if int(row["valid_path_count"]) != reference_paths[int(row["sample_index"])]
        ]
        by_label: dict[str, Any] = {}
        for label in (0, 1):
            selected = [row for row in budget_rows if int(row["label"]) == label]
            if selected:
                label_errors = np.asarray(
                    [float(row["relative_cfr_l2_to_reference"]) for row in selected]
                )
                by_label[str(label)] = {
                    "links": len(selected),
                    "path_count_changes_vs_reference": sum(
                        int(row["valid_path_count"]) != reference_paths[int(row["sample_index"])]
                        for row in selected
                    ),
                    "max_relative_cfr_l2": float(np.max(label_errors)),
                    "p95_relative_cfr_l2": float(np.percentile(label_errors, 95)),
                }
        summary["by_budget"][str(budget)] = {
            "links": len(budget_rows),
            "path_count_changes_vs_reference": len(changed),
            "max_relative_cfr_l2": float(np.max(errors)),
            "median_relative_cfr_l2": float(np.median(errors)),
            "p95_relative_cfr_l2": float(np.percentile(errors, 95)),
            "by_label": by_label,
        }
    return summary


def _scene_tree_sha256(scene_xml: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(scene_xml.parent.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(scene_xml.parent).as_posix().encode("utf8"))
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def audit_dataset(
    input_path: Path,
    *,
    samples_per_src_values: Sequence[int],
    mitsuba_variant: str | None = None,
) -> dict[str, Any]:
    """Re-run each accepted dataset link with identical positions/seeds."""
    dataset = ChannelDataset.load(input_path)
    if not samples_per_src_values or any(value <= 0 for value in samples_per_src_values):
        raise ValueError("samples_per_src_values must be positive")
    if list(samples_per_src_values) != sorted(set(samples_per_src_values)):
        raise ValueError("samples_per_src_values must be unique and strictly increasing")
    metadata = list(dataset.metadata)
    required = {"scene", "tx", "rx", "seed", "max_depth", "grid_config", "scene_config"}
    if any(not required.issubset(row) for row in metadata):
        raise ValueError("dataset metadata is missing RT scene, link, solver, or grid fields")
    scene_names = {str(row["scene"]) for row in metadata}
    tx_positions = {tuple(row["tx"]) for row in metadata}
    depths = {int(row["max_depth"]) for row in metadata}
    grid_configs = {tuple(sorted(row["grid_config"].items())) for row in metadata}
    variants = {str(row["mitsuba_variant"]) for row in metadata}
    if any(len(values) != 1 for values in (scene_names, tx_positions, depths, grid_configs)):
        raise ValueError("one input NPZ must describe one scene, transmitter, depth, and grid")
    if len(variants) != 1:
        raise ValueError("dataset contains mixed Mitsuba variants")
    selected_variant = mitsuba_variant or next(iter(variants))
    if selected_variant != next(iter(variants)):
        raise ValueError("requested Mitsuba variant differs from stored dataset metadata")

    try:
        from integrations.sionna.backend import activate_rt_backend, select_rt_backend

        backend_selection = select_rt_backend(selected_variant)
        activate_rt_backend(backend_selection)
        import sionna.rt as rt
        from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            f"Could not initialize the dataset's Sionna RT backend: {error}"
        ) from error

    scene_name = next(iter(scene_names))
    scene_ref = getattr(rt.scene, scene_name, scene_name)
    scene_xml = Path(str(scene_ref)).resolve()
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Sionna scene XML not found: {scene_xml}")
    first_grid = metadata[0]["grid_config"]
    grid = GridConfig(
        n_subcarriers=int(first_grid["n_subcarriers"]),
        spacing_hz=float(first_grid["spacing_hz"]),
        carrier_hz=float(first_grid["carrier_hz"]),
        snr_db=float(first_grid["snr_db"]),
    )
    scene = load_scene(str(scene_xml))
    scene.frequency = grid.carrier_hz
    scene.bandwidth = grid.n_subcarriers * grid.spacing_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.add(Transmitter(name="budget_audit_tx", position=list(next(iter(tx_positions)))))
    receiver = Receiver(name="budget_audit_rx", position=list(metadata[0]["rx"]))
    scene.add(receiver)
    solver = PathSolver(deterministic=True)
    depth = next(iter(depths))

    records: list[dict[str, Any]] = []
    for sample_index, row in enumerate(metadata):
        receiver.position = [float(value) for value in row["rx"]]
        per_budget: list[tuple[int, np.ndarray, int, str]] = []
        for budget in samples_per_src_values:
            paths = solver(
                scene,
                max_depth=depth,
                los=True,
                specular_reflection=True,
                diffuse_reflection=True,
                refraction=True,
                diffraction=False,
                samples_per_src=budget,
                seed=int(row["seed"]),
            )
            cfr = rt_cfr(paths, grid).reshape(-1)
            path_count = count_valid_paths(paths)
            cfr_hash = hashlib.sha256(np.ascontiguousarray(cfr).tobytes()).hexdigest()
            per_budget.append((budget, cfr, path_count, cfr_hash))
        reference = per_budget[-1][1]
        for budget, cfr, path_count, cfr_hash in per_budget:
            records.append(
                {
                    "sample_index": sample_index,
                    "group": str(dataset.group[sample_index]),
                    "label": int(dataset.y[sample_index]),
                    "rx_position_m": [float(value) for value in row["rx"]],
                    "solver_seed": int(row["seed"]),
                    "max_depth": depth,
                    "samples_per_src": budget,
                    "valid_path_count": path_count,
                    "relative_cfr_l2_to_reference": relative_l2_error(cfr, reference),
                    "cfr_sha256": cfr_hash,
                }
            )

    runtime = versions()
    return {
        "schema": "loslab.rt_dataset_ray_budget_audit.v1",
        "evidence_grade": "FIXED_LINK_DATASET_SOLVER_DIAGNOSTIC",
        "source_revision": runtime.get("git_sha"),
        "script_sha256": sha256_file(__file__),
        "package_source_sha256": package_source_sha256(),
        "versions": runtime,
        "input": {
            "path": str(input_path.resolve()),
            "sha256": sha256_file(input_path),
            "links": len(dataset.y),
            "class_counts": {
                "0": int(np.count_nonzero(dataset.y == 0)),
                "1": int(np.count_nonzero(dataset.y == 1)),
            },
            "stored_generation_samples_per_src": int(
                metadata[0]["scene_config"]["samples_per_src"]
            ),
        },
        "scene": {
            "name": scene_name,
            "xml_sha256": sha256_file(scene_xml),
            "scene_tree_sha256": _scene_tree_sha256(scene_xml),
        },
        "configuration": {
            "mitsuba_variant": selected_variant,
            "rt_backend_selection": backend_selection.as_dict(),
            "deterministic_solver": True,
            "max_depth": depth,
            "samples_per_src_values": list(samples_per_src_values),
            "reference_samples_per_src": samples_per_src_values[-1],
            "tx_position_m": list(next(iter(tx_positions))),
            "grid_config": first_grid,
            "solver_flags": {
                "los": True,
                "specular_reflection": True,
                "diffuse_reflection": True,
                "refraction": True,
                "diffraction": False,
            },
        },
        "summary": summarize_link_rows(records, samples_per_src_values),
        "links": records,
        "limitations": [
            "The largest ray budget is a comparison reference, not validated physical truth.",
            "A single link's direct-path label is inherited from the original geometry oracle; this audit measures channel-solver sensitivity only.",
            "One finite set of receiver locations does not qualify the full geometry family or material model.",
            "This fixed-link rerun does not include antenna-orientation or geometry-precision sweeps.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument(
        "--samples-per-src",
        type=int,
        nargs="+",
        default=[200, 2000, 20000, 100000, 1000000],
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    if list(args.samples_per_src) != sorted(set(args.samples_per_src)):
        parser.error("--samples-per-src values must be unique and strictly increasing")
    audits = [
        audit_dataset(path, samples_per_src_values=args.samples_per_src) for path in args.inputs
    ]
    result = {
        "schema": "loslab.rt_multi_scene_ray_budget_audit.v1",
        "evidence_grade": "FIXED_LINK_DATASET_SOLVER_DIAGNOSTIC",
        "samples_per_src_values": args.samples_per_src,
        "scenes": audits,
        "interpretation_warning": (
            "This diagnostic compares solver outputs at the same saved links. It does not establish "
            "scene/material realism, path-label correctness, or model performance."
        ),
    }
    atomic_json(args.output, result)
    total_links = sum(scene["input"]["links"] for scene in audits)
    print(
        f"Saved ray-budget audit for {total_links} fixed links across {len(audits)} scenes to {args.output}"
    )


if __name__ == "__main__":
    main()
