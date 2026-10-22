"""Compare matched Sionna RT convergence reports across path-count caps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import atomic_json, sha256_file

METRICS = (
    ("valid_path_count", "Valid paths", "linear"),
    ("valid_diffuse_reflection_path_count", "Diffuse paths", "linear"),
    ("valid_path_power_sum", "Path coefficient power sum", "log"),
    ("power_weighted_rms_delay_spread_ns", "Power-weighted RMS delay spread (ns)", "linear"),
)


def _comparable_configuration(configuration: dict[str, Any], sample_budget: int) -> dict[str, Any]:
    """Normalize the two intentionally varied sweep axes before equality checks."""
    normalized = json.loads(json.dumps(configuration))
    normalized["max_num_paths_per_src"] = None
    normalized["samples_per_src_values"] = [sample_budget]
    return normalized


def summarize_cap_reports(
    report_paths: Sequence[Path], *, max_depth: int, sample_budget: int
) -> dict[str, Any]:
    """Validate matched reports and return compact per-seed and cap summaries."""
    if len(report_paths) < 2:
        raise ValueError("at least two cap reports are required")
    if max_depth < 0 or sample_budget < 1:
        raise ValueError("max_depth must be nonnegative and sample_budget positive")

    reports: list[dict[str, Any]] = []
    for path in report_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "loslab.sionna_rt_solver_convergence.v3":
            raise ValueError(f"unsupported report schema in {path}")
        configuration = data.get("configuration", {})
        if configuration.get("normalize_delays_for_cfr") is not False:
            raise ValueError(f"report does not use absolute-delay CFR in {path}")
        cap = configuration.get("max_num_paths_per_src")
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ValueError(f"report has an invalid path cap in {path}")
        reports.append({"path": path, "data": data, "cap": cap})

    reports.sort(key=lambda report: report["cap"])
    caps = [report["cap"] for report in reports]
    if len(set(caps)) != len(caps):
        raise ValueError("path caps must be unique")
    reference = reports[0]["data"]
    reference_configuration = _comparable_configuration(reference["configuration"], sample_budget)
    reference_scene_hash = reference["scene"].get("scene_tree_sha256")
    reference_rows: dict[tuple[str, int], dict[str, Any]] | None = None
    summary_rows: list[dict[str, Any]] = []

    for report in reports:
        data = report["data"]
        if data["scene"].get("scene_tree_sha256") != reference_scene_hash:
            raise ValueError("reports must use the same scene asset hash")
        if (
            _comparable_configuration(data["configuration"], sample_budget)
            != reference_configuration
        ):
            raise ValueError("reports differ in settings other than path cap/sample budget")
        selected = [
            row
            for row in data["runs"]
            if row["max_depth"] == max_depth and row["samples_per_src"] == sample_budget
        ]
        keyed = {(row["link_id"], row["seed"]): row for row in selected}
        if len(keyed) != len(selected) or not keyed:
            raise ValueError(f"report has duplicate or missing matched rows: {report['path']}")
        if reference_rows is None:
            reference_rows = keyed
        elif keyed.keys() != reference_rows.keys():
            raise ValueError("reports must have identical links and seeds at selected settings")
        for (link_id, seed), row in sorted(keyed.items()):
            summary_rows.append(
                {
                    "max_num_paths_per_src": report["cap"],
                    "link_id": link_id,
                    "seed": seed,
                    **{name: row[name] for name, _, _ in METRICS},
                    "power_weighted_arrival_angular_spread_deg": (
                        None
                        if row["power_weighted_arrival_angular_spread_rad_3d"] is None
                        else math.degrees(row["power_weighted_arrival_angular_spread_rad_3d"])
                    ),
                }
            )

    cap_summaries: list[dict[str, Any]] = []
    for cap in caps:
        for link in reference["configuration"]["links"]:
            link_id = link["link_id"]
            rows = [
                row
                for row in summary_rows
                if row["max_num_paths_per_src"] == cap and row["link_id"] == link_id
            ]
            cap_summaries.append(
                {
                    "max_num_paths_per_src": cap,
                    "link_id": link_id,
                    "geometric_visibility_state": link["geometric_visibility_state"],
                    "seed_count": len(rows),
                    "metrics": {
                        metric: {
                            "median": float(np.median(values)),
                            "min": min(values),
                            "max": max(values),
                        }
                        for metric in (
                            *(metric for metric, _, _ in METRICS),
                            "power_weighted_arrival_angular_spread_deg",
                        )
                        if (
                            values := [
                                float(row[metric]) for row in rows if row[metric] is not None
                            ]
                        )
                    },
                }
            )

    return {
        "schema": "loslab.rt_path_cap_sweep.v1",
        "scene": reference["scene"],
        "configuration": {
            "max_depth": max_depth,
            "samples_per_src": sample_budget,
            "seeds": reference["configuration"]["seeds"],
            "materials": reference["configuration"]["materials"],
            "material_scattering_overrides": reference["configuration"].get(
                "material_scattering_overrides", {}
            ),
            "mitsuba_variant": reference["configuration"]["mitsuba_variant"],
        },
        "inputs": [
            {
                "path": str(report["path"].resolve()),
                "sha256": sha256_file(report["path"]),
                "max_num_paths_per_src": report["cap"],
            }
            for report in reports
        ],
        "per_seed": summary_rows,
        "summaries": cap_summaries,
        "limitations": [
            "The largest path cap is a finite comparison reference, not ground truth.",
            "Only one synthetic thin-screen scene, one scattering override, and one depth/budget are compared.",
            "CFR vectors are not stored in the convergence reports, so this audit compares path and aggregate observables, not cross-cap complex CFR error.",
            "The S=0.5 material override is for mechanism testing, not calibrated material realism.",
        ],
    }


def plot_cap_sweep(summary: dict[str, Any], output_prefix: Path) -> dict[str, Any]:
    """Plot matched cap sensitivity and save the compact summary sidecar."""
    import matplotlib.pyplot as plt

    caps = sorted({int(row["max_num_paths_per_src"]) for row in summary["summaries"]})
    link_ids = list(dict.fromkeys(row["link_id"] for row in summary["summaries"]))
    colors = {link_id: plt.get_cmap("tab10")(index) for index, link_id in enumerate(link_ids)}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, (metric, label, scale) in zip(axes.flat, METRICS, strict=True):
        for link_id in link_ids:
            rows = [row for row in summary["summaries"] if row["link_id"] == link_id]
            rows.sort(key=lambda row: row["max_num_paths_per_src"])
            values = [row["metrics"][metric] for row in rows]
            medians = [value["median"] for value in values if value["median"] is not None]
            xs = [
                row["max_num_paths_per_src"]
                for row, value in zip(rows, values, strict=True)
                if value["median"] is not None
            ]
            lows = [value["min"] for value in values if value["median"] is not None]
            highs = [value["max"] for value in values if value["median"] is not None]
            if not xs:
                continue
            axis.errorbar(
                xs,
                medians,
                yerr=np.vstack((np.asarray(medians) - lows, np.asarray(highs) - medians)),
                marker="o",
                capsize=3,
                color=colors[link_id],
                label=link_id,
            )
        axis.set_xscale("log")
        axis.set_xticks(caps, labels=[f"{cap:,}" for cap in caps])
        if scale == "log":
            axis.set_yscale("log")
        axis.set_xlabel("Maximum paths per source")
        axis.set_ylabel(label)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

    override = summary["configuration"]["material_scattering_overrides"]
    override_text = ", ".join(f"{name} S={value:g}" for name, value in sorted(override.items()))
    fig.suptitle(
        f"Path-cap sensitivity | {summary['scene']['name']} | "
        f"depth {summary['configuration']['max_depth']} | "
        f"{summary['configuration']['samples_per_src']:,} samples | {override_text} (synthetic)"
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    summary["figure"] = {
        "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
        "svg": {"path": str(svg_path), "sha256": sha256_file(svg_path)},
    }
    atomic_json(output_prefix.with_suffix(".json"), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--max-depth", type=int, required=True)
    parser.add_argument("--sample-budget", type=int, required=True)
    args = parser.parse_args()
    summary = summarize_cap_reports(
        args.reports, max_depth=args.max_depth, sample_budget=args.sample_budget
    )
    plot_cap_sweep(summary, args.output_prefix)
    print(f"Saved path-cap comparison to {args.output_prefix}")


if __name__ == "__main__":
    main()
