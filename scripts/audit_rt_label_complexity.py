"""Audit accepted RT channel complexity and class-conditional selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets.channel import ChannelDataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "maximum": None, "p05": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
    }


def summarize_rt_dataset(path: Path) -> dict[str, Any]:
    """Summarize accepted path mechanisms and retain generator selection counts."""
    dataset = ChannelDataset.load(path)
    legacy_rows_are_geometric = all(
        str(metadata.get("geometry_label", "")).upper() == "GEOMETRIC_DIRECT_PATH_ORACLE"
        for metadata in dataset.metadata
    )
    if dataset.label_semantics != "geometric_direct_path_oracle" and not legacy_rows_are_geometric:
        raise ValueError(f"expected geometric RT labels in {path}, got {dataset.label_semantics}")
    scenes = set(map(str, dataset.domain.tolist()))
    if len(scenes) != 1:
        raise ValueError(f"expected one scene domain in {path}, got {sorted(scenes)}")
    generation = dataset.generation_report
    if generation.get("schema") != "loslab.raytraced_generation_report.v1":
        raise ValueError(f"missing recognized RT generation report in {path}")

    mechanism_counts: Counter[str] = Counter()
    path_histograms: dict[str, Counter[str]] = {
        "geometric_nlos": Counter(),
        "geometric_los": Counter(),
    }
    los_direct_to_indirect_db: list[float] = []
    direct_power_by_class: dict[str, list[float]] = {"geometric_nlos": [], "geometric_los": []}
    indirect_power_by_class: dict[str, list[float]] = {"geometric_nlos": [], "geometric_los": []}

    for index, (label_value, metadata) in enumerate(zip(dataset.y, dataset.metadata, strict=True)):
        label = int(label_value)
        class_name = "geometric_los" if label else "geometric_nlos"
        required = {
            "valid_paths_full",
            "valid_paths_direct_oracle",
            "los_plus_indirect_paths",
            "oracle_direct_power",
            "residual_indirect_power",
        }
        missing = required - set(metadata)
        if missing:
            raise ValueError(f"sample {index} in {path} lacks path diagnostics: {sorted(missing)}")
        full_paths = int(metadata["valid_paths_full"])
        direct_paths = int(metadata["valid_paths_direct_oracle"])
        direct_power = float(metadata["oracle_direct_power"])
        indirect_power = float(metadata["residual_indirect_power"])
        if full_paths < 0 or direct_paths < 0 or full_paths < direct_paths:
            raise ValueError(f"invalid path counts in sample {index} of {path}")
        if not math.isfinite(direct_power) or not math.isfinite(indirect_power):
            raise ValueError(f"nonfinite path power in sample {index} of {path}")
        if direct_power < 0 or indirect_power < 0:
            raise ValueError(f"negative path power in sample {index} of {path}")
        has_direct = direct_paths > 0
        mixed = bool(metadata["los_plus_indirect_paths"])
        if has_direct != bool(label):
            raise ValueError(
                f"geometry label and direct-path oracle disagree in sample {index} of {path}"
            )
        if mixed != bool(label and full_paths > direct_paths):
            raise ValueError(
                f"mixed-path flag disagrees with path counts in sample {index} of {path}"
            )

        if label:
            mechanism = "los_mixed" if mixed else "los_direct_only"
        else:
            mechanism = "nlos_indirect_paths" if full_paths else "nlos_no_path"
        mechanism_counts[mechanism] += 1
        path_histograms[class_name][str(full_paths)] += 1
        direct_power_by_class[class_name].append(direct_power)
        indirect_power_by_class[class_name].append(indirect_power)
        if label and indirect_power > 0 and direct_power > 0:
            los_direct_to_indirect_db.append(10.0 * math.log10(direct_power / indirect_power))

    attempts = generation["attempt_counts"]
    generation_digest = hashlib.sha256(
        json.dumps(generation, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    full_solver_calls = int(attempts["full_solver_calls"])
    empty_paths = int(attempts["full_solver_empty_paths"])
    accepted_total = len(dataset.y)
    if full_solver_calls < accepted_total or empty_paths > full_solver_calls:
        raise ValueError(f"inconsistent RT acceptance/outcome counters in {path}")
    if (
        sum(int(value) for value in generation["full_solver_outcomes"].values())
        != full_solver_calls
    ):
        raise ValueError(f"full-solver outcomes do not account for every solver call in {path}")
    if int(generation["full_solver_outcomes"].get("accepted", -1)) != accepted_total:
        raise ValueError(f"accepted dataset count disagrees with generation report in {path}")
    return {
        "scene_domain": next(iter(scenes)),
        "resolved_label_semantics": "geometric_direct_path_oracle",
        "source_path": str(path),
        "source_sha256": _sha256(path),
        "generation_report_sha256": generation_digest,
        "generator_environment": generation.get("environment", {}),
        "rt_backend_selection": generation.get("rt_backend_selection", {}),
        "n_samples": accepted_total,
        "accepted_class_counts": {
            "geometric_nlos": int(np.count_nonzero(dataset.y == 0)),
            "geometric_los": int(np.count_nonzero(dataset.y == 1)),
        },
        "accepted_mechanisms": dict(sorted(mechanism_counts.items())),
        "valid_full_path_count_histograms": {
            label: dict(sorted(histogram.items(), key=lambda item: int(item[0])))
            for label, histogram in path_histograms.items()
        },
        "direct_to_indirect_power_ratio_db_for_los_mixed": _distribution(los_direct_to_indirect_db),
        "path_power_by_geometric_label": {
            label: {
                "direct_power": _distribution(direct_power_by_class[label]),
                "indirect_power": _distribution(indirect_power_by_class[label]),
            }
            for label in ("geometric_nlos", "geometric_los")
        },
        "generation_selection": {
            "requested_per_class": generation["requested_per_class"],
            "attempt_counts": attempts,
            "full_solver_outcomes": generation["full_solver_outcomes"],
            "accepted_counts": generation["accepted_counts"],
            "accepted_per_full_solver_call": accepted_total / full_solver_calls,
            "empty_path_candidate_fraction": empty_paths / full_solver_calls,
            "selection_note": generation["selection_note"],
            "empty_path_class_attribution": "not recorded by legacy generation report",
        },
    }


def audit_rt_datasets(paths: list[Path]) -> dict[str, Any]:
    if not paths:
        raise ValueError("at least one RT dataset is required")
    scenes = [summarize_rt_dataset(path) for path in paths]
    if len({scene["scene_domain"] for scene in scenes}) != len(scenes):
        raise ValueError("each input must represent a distinct RT scene domain")
    return {
        "schema": "loslab.rt_label_complexity_audit.v1",
        "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
        "label_semantics": "geometric_direct_path_oracle",
        "audit_script_sha256": _sha256(Path(__file__).resolve()),
        "scene_count": len(scenes),
        "scenes": scenes,
        "reproduction": {
            "command": (
                "python scripts/audit_rt_label_complexity.py --input <rt-scene-1.npz> "
                "<rt-scene-2.npz> ... --output <new-audit.json>"
            )
        },
        "interpretation": (
            "Mechanism and power summaries describe only accepted samples under each recorded "
            "scene sampler, class quota, solver configuration, and LOS-multipath filter. "
            "Empty-path outcomes are candidate-level and legacy reports do not attribute them "
            "to LOS/NLOS separately; they are not exported observations."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    report = audit_rt_datasets(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {scene["scene_domain"]: scene["accepted_mechanisms"] for scene in report["scenes"]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
