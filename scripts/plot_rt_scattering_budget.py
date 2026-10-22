"""Plot diffuse-path budget sensitivity from an explicitly scattering Sionna RT report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from artifacts.provenance import atomic_json, sha256_file


def summarize_seed_range(
    rows: Sequence[dict[str, Any]], *, x_value: int, metric: str
) -> tuple[float, float, float] | None:
    """Return median/min/max over finite per-seed values for one budget."""
    values = [
        float(row[metric])
        for row in rows
        if int(row["samples_per_src"]) == x_value
        and row.get(metric) is not None
        and np.isfinite(float(row[metric]))
    ]
    if not values:
        return None
    return float(np.median(values)), min(values), max(values)


def load_report(path: Path) -> dict[str, Any]:
    """Require versioned material and path-component provenance."""
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema") != "loslab.sionna_rt_solver_convergence.v3":
        raise ValueError(f"unsupported scattering-budget report schema in {path}")
    configuration = report.get("configuration", {})
    if configuration.get("solver_flags", {}).get("diffuse_reflection") is not True:
        raise ValueError("diffuse reflection must be enabled in the solver")
    materials = configuration.get("materials", {})
    if not any(
        any(float(value) > 0.0 for value in item.get("scattering_coefficient", []))
        for item in materials.values()
    ):
        raise ValueError("report has no nonzero material scattering coefficient")
    for row in report.get("runs", []):
        required = (
            "valid_diffuse_reflection_path_count",
            "diffuse_path_power_sum",
            "diffuse_power_weighted_rms_delay_spread_ns",
            "diffuse_power_weighted_arrival_angular_spread_rad_3d",
        )
        if not all(key in row for key in required):
            raise ValueError("report is missing diffuse-component observables")
    return report


def plot_scattering_budget(report_path: Path, output_prefix: Path) -> dict[str, Any]:
    """Render per-component ray-budget summaries and a content-hash sidecar."""
    import matplotlib.pyplot as plt

    report = load_report(report_path)
    configuration = report["configuration"]
    depth = max(configuration["max_depth_values"])
    budgets = sorted(configuration["samples_per_src_values"])
    links = configuration["links"]
    colors = {"visible": "#188f86", "blocked": "#e7782f"}
    metrics = (
        ("valid_diffuse_reflection_path_count", "Diffuse path count", "linear"),
        ("diffuse_path_power_sum", "Diffuse coefficient power sum", "log"),
        ("diffuse_power_weighted_rms_delay_spread_ns", "Diffuse RMS delay spread (ns)", "linear"),
        (
            "diffuse_power_weighted_arrival_angular_spread_rad_3d",
            "Diffuse arrival angular spread (degrees)",
            "linear",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    for axis, (metric, ylabel, yscale) in zip(axes.flat, metrics, strict=True):
        for link in links:
            rows = [
                row
                for row in report["runs"]
                if row["link_id"] == link["link_id"] and row["max_depth"] == depth
            ]
            xs: list[int] = []
            medians: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for budget in budgets:
                summary = summarize_seed_range(rows, x_value=budget, metric=metric)
                if summary is None:
                    continue
                median, low, high = summary
                if yscale == "log" and median <= 0.0:
                    continue
                xs.append(budget)
                medians.append(float(np.rad2deg(median)) if metric.endswith("_rad_3d") else median)
                lows.append(float(np.rad2deg(low)) if metric.endswith("_rad_3d") else low)
                highs.append(float(np.rad2deg(high)) if metric.endswith("_rad_3d") else high)
            if xs:
                color = colors[link["geometric_visibility_state"]]
                axis.plot(xs, medians, marker="o", color=color, label=link["link_id"])
                if yscale == "log":
                    lower = [
                        low if low > 0.0 else max(median * 0.1, np.finfo(float).tiny)
                        for low, median in zip(lows, medians, strict=True)
                    ]
                else:
                    lower = lows
                axis.fill_between(xs, lower, highs, color=color, alpha=0.18)
        axis.set_xscale("log")
        if yscale == "log":
            axis.set_yscale("log")
        axis.set_xlabel("Path-solver ray samples per source")
        axis.set_ylabel(ylabel)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()

    figure_title = (
        f"Diffuse-path sampling sensitivity | {report['scene']['name']} | "
        f"max depth {depth} | scattering coefficients "
        + ", ".join(
            f"{name}:{','.join(str(x) for x in values['scattering_coefficient'])}"
            for name, values in configuration["materials"].items()
        )
    )
    fig.suptitle(figure_title)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    svg_path = output_prefix.with_suffix(".svg")
    fig.savefig(png_path, dpi=180)
    fig.savefig(svg_path)
    plt.close(fig)
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in svg_lines) + "\n", encoding="utf-8")

    sidecar = {
        "schema": "loslab.rt_scattering_budget_plot.v1",
        "input": {"path": str(report_path.resolve()), "sha256": sha256_file(report_path)},
        "configuration": configuration,
        "png": {"path": str(png_path), "sha256": sha256_file(png_path)},
        "svg": {"path": str(svg_path), "sha256": sha256_file(svg_path)},
        "summary": "Lines show the median across seeds; bands show the observed min/max and may use fewer seeds when a budget produces no diffuse paths. In the log-scaled diffuse-power panel, exact zero lower bounds are clipped to 10% of the median for visibility. The synthetic S=0.5 override demonstrates solver mechanics, not calibrated material scattering.",
    }
    atomic_json(output_prefix.with_suffix(".json"), sidecar)
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    sidecar = plot_scattering_budget(args.report, args.output_prefix)
    print(f"Saved scattering-budget plots to {sidecar['png']['path']} and {sidecar['svg']['path']}")


if __name__ == "__main__":
    main()
