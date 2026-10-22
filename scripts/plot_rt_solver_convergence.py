"""Plot provenance-backed Sionna RT convergence diagnostics from JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import atomic_json, sha256_file


def summarize_seed_range(
    rows: Sequence[dict[str, Any]],
    *,
    x_field: str,
    metric: str,
    positive_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted x, median, min, max over seeds, omitting undefined metrics."""
    x_values = sorted({int(row[x_field]) for row in rows})
    xs: list[int] = []
    medians: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    for x_value in x_values:
        values = [
            float(row[metric])
            for row in rows
            if int(row[x_field]) == x_value
            and row.get(metric) is not None
            and (not positive_only or float(row[metric]) > 0.0)
        ]
        if values:
            xs.append(x_value)
            medians.append(float(np.median(values)))
            lows.append(min(values))
            highs.append(max(values))
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(medians, dtype=np.float64),
        np.asarray(lows, dtype=np.float64),
        np.asarray(highs, dtype=np.float64),
    )


def load_reports(paths: Sequence[Path]) -> list[dict[str, Any]]:
    """Load only current geometry-labeled convergence report schema v2."""
    reports = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != "loslab.sionna_rt_solver_convergence.v2":
            raise ValueError(f"unsupported convergence report schema in {path}")
        if data.get("configuration", {}).get("normalize_delays_for_cfr") is not False:
            raise ValueError(f"report does not preserve absolute delay in {path}")
        reports.append({"path": path.resolve(), "data": data})
    if not reports:
        raise ValueError("at least one convergence report is required")
    return reports


def plot_reports(report_paths: Sequence[Path], output_prefix: Path) -> dict[str, Any]:
    """Render budget/depth curves with seed ranges and a hash sidecar."""
    import matplotlib.pyplot as plt

    reports = load_reports(report_paths)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0), constrained_layout=True)
    ax_budget, ax_depth, ax_delay, ax_angle = axes.flat
    colors = plt.get_cmap("tab10")
    line_styles = ("-", "--", ":", "-.")

    for report_index, report in enumerate(reports):
        data = report["data"]
        scene_name = data["scene"]["name"]
        config = data["configuration"]
        for link_index, link in enumerate(config["links"]):
            link_id = link["link_id"]
            label = (
                f"{scene_name} / {link_id} ({link['geometric_visibility_state']}; "
                f"depth {max(config['max_depth_values'])}, "
                f"reference {config['reference_samples_per_src']:,})"
            )
            color = colors(link_index % 10)
            line_style = line_styles[report_index % len(line_styles)]
            rows = [row for row in data["runs"] if row["link_id"] == link_id]
            by_depth = [row for row in rows if row["max_depth"] == max(config["max_depth_values"])]
            by_reference_budget = [
                row for row in rows if row["samples_per_src"] == config["reference_samples_per_src"]
            ]

            x, median, low, high = summarize_seed_range(
                by_depth,
                x_field="samples_per_src",
                metric="relative_cfr_l2_to_max_budget_and_depth",
                positive_only=True,
            )
            if len(x):
                ax_budget.errorbar(
                    x,
                    median,
                    yerr=np.vstack((median - low, high - median)),
                    marker="o",
                    capsize=3,
                    color=color,
                    linestyle=line_style,
                    label=label,
                )

            x, median, low, high = summarize_seed_range(
                by_reference_budget,
                x_field="max_depth",
                metric="relative_cfr_l2_to_max_budget_and_depth",
                positive_only=True,
            )
            if len(x):
                ax_depth.errorbar(
                    x,
                    median,
                    yerr=np.vstack((median - low, high - median)),
                    marker="o",
                    capsize=3,
                    color=color,
                    linestyle=line_style,
                    label=label,
                )

            x, median, low, high = summarize_seed_range(
                by_depth,
                x_field="samples_per_src",
                metric="power_weighted_rms_delay_spread_ns",
            )
            if len(x):
                ax_delay.errorbar(
                    x,
                    median,
                    yerr=np.vstack((median - low, high - median)),
                    marker="o",
                    capsize=3,
                    color=color,
                    linestyle=line_style,
                    label=label,
                )

            angle_rows = [
                {
                    **row,
                    "arrival_angular_spread_deg": (
                        None
                        if row["power_weighted_arrival_angular_spread_rad_3d"] is None
                        else float(np.rad2deg(row["power_weighted_arrival_angular_spread_rad_3d"]))
                    ),
                }
                for row in by_reference_budget
            ]
            x, median, low, high = summarize_seed_range(
                angle_rows,
                x_field="max_depth",
                metric="arrival_angular_spread_deg",
            )
            if len(x):
                ax_angle.errorbar(
                    x,
                    median,
                    yerr=np.vstack((median - low, high - median)),
                    marker="o",
                    capsize=3,
                    color=color,
                    linestyle=line_style,
                    label=label,
                )

    ax_budget.set_xscale("log")
    ax_budget.set_yscale("log")
    ax_budget.set_xlabel("Diffuse-ray samples per source")
    ax_budget.set_ylabel("Relative complex CFR L2 error")
    ax_budget.set_title("Ray-budget sensitivity at maximum depth")

    ax_depth.set_yscale("log")
    ax_depth.set_xlabel("Maximum interaction depth")
    ax_depth.set_ylabel("Relative complex CFR L2 error")
    ax_depth.set_title("Interaction-depth sensitivity at reference budget")

    ax_delay.set_xscale("log")
    ax_delay.set_xlabel("Diffuse-ray samples per source")
    ax_delay.set_ylabel("Power-weighted RMS delay spread (ns)")
    ax_delay.set_title("Delay-spread stability at maximum depth")

    ax_angle.set_xlabel("Maximum interaction depth")
    ax_angle.set_ylabel("Power-weighted arrival angular spread (degrees)")
    ax_angle.set_title("Angular-spread sensitivity at reference budget")

    legend_items: dict[str, Any] = {}
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        legend_items.update(zip(labels, handles, strict=True))
    if legend_items:
        fig.legend(
            list(legend_items.values()),
            list(legend_items),
            loc="outside lower center",
            ncols=2,
        )
    fig.suptitle("Sionna RT solver-budget convergence diagnostics", fontsize=15)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")

    sidecar = {
        "schema": "loslab.rt_solver_convergence_plot.v1",
        "inputs": [
            {
                "path": str(report["path"]),
                "sha256": sha256_file(report["path"]),
                "scene": report["data"]["scene"]["name"],
                "configuration": report["data"]["configuration"],
            }
            for report in reports
        ],
        "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
        "svg": {"path": str(svg_path), "sha256": sha256_file(svg_path)},
        "summary": "Error bars show min-to-max across deterministic seeds. Exact zero relative errors (identical to the selected reference CFR) are omitted from log-scale panels. The high-budget CFR is a comparison reference, not proven ground truth.",
    }
    atomic_json(output_prefix.with_suffix(".json"), sidecar)
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    sidecar = plot_reports(args.reports, args.output_prefix)
    print(f"Saved convergence plots to {sidecar['png']['path']} and {sidecar['svg']['path']}")


if __name__ == "__main__":
    main()
