"""Audit ray-budget changes in RT-observable distributions across solver seeds."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from artifacts.provenance import atomic_json, sha256_file

SCHEMAS = {
    "loslab.sionna_rt_solver_convergence.v3",
    "loslab.sionna_rt_solver_convergence.v4",
}
METRICS = {
    "valid_path_rate_per_1000_samples": ("valid_path_count", "rate"),
    "non_diffuse_path_count": ("valid_non_diffuse_path_count", "value"),
    "diffuse_path_rate_per_1000_samples": ("valid_diffuse_reflection_path_count", "rate"),
    "cfr_mean_power": ("cfr_mean_power", "value"),
    "non_diffuse_path_power_sum": ("non_diffuse_path_power_sum", "value"),
    "diffuse_path_power_sum": ("diffuse_path_power_sum", "value"),
    "total_rms_delay_spread_ns": ("power_weighted_rms_delay_spread_ns", "value"),
    "non_diffuse_rms_delay_spread_ns": (
        "non_diffuse_power_weighted_rms_delay_spread_ns",
        "value",
    ),
    "diffuse_rms_delay_spread_ns": ("diffuse_power_weighted_rms_delay_spread_ns", "value"),
    "non_diffuse_angular_spread_rad": (
        "non_diffuse_power_weighted_arrival_angular_spread_rad_3d",
        "value",
    ),
    "diffuse_angular_spread_rad": (
        "diffuse_power_weighted_arrival_angular_spread_rad_3d",
        "value",
    ),
}


def _load_report(path: Path) -> dict[str, Any]:
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                report = json.load(stream)
        else:
            report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, EOFError, gzip.BadGzipFile, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read convergence report {path}: {error}") from error
    if not isinstance(report, dict) or report.get("schema") not in SCHEMAS:
        raise ValueError(f"report must use a supported convergence schema: {path}")
    config = report.get("configuration")
    if not isinstance(config, dict) or config.get("normalize_delays_for_cfr") is not False:
        raise ValueError("report must retain absolute-delay CFR semantics")
    rows = report.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("report must contain solver runs")
    return report


def _index_rows(
    rows: list[dict[str, Any]], *, budget: int, depth: int
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("samples_per_src") != budget or row.get("max_depth") != depth:
            continue
        link_id, seed = row.get("link_id"), row.get("seed")
        if (
            not isinstance(link_id, str)
            or not link_id
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or not isinstance(row.get("geometric_los"), bool)
        ):
            raise ValueError("selected run rows need link_id, integer seed, and LOS boolean")
        key = (link_id, seed)
        if key in indexed:
            raise ValueError(f"duplicate link/seed row for budget {budget}: {key}")
        indexed[key] = row
    return indexed


def _metric_value(row: dict[str, Any], metric: str, budget: int) -> float | None:
    field, transform = METRICS[metric]
    value = row.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"run contains an invalid {field}")
    if value < 0:
        raise ValueError(f"run contains a negative {field}")
    return float(value) * (1000.0 / budget) if transform == "rate" else float(value)


def audit_observable_distributions(
    report_path: Path,
    *,
    low_budget: int,
    reference_budget: int,
    depth: int | None = None,
    bootstrap_draws: int = 10_000,
    bootstrap_seed: int = 20261022,
) -> dict[str, Any]:
    """Compare component observables and their seed distributions at one depth.

    Confidence intervals resample paired solver seeds. They quantify Monte Carlo
    seed uncertainty for this recorded scene/link set, not scene-level uncertainty.
    """
    if (
        isinstance(low_budget, bool)
        or isinstance(reference_budget, bool)
        or low_budget < 1
        or reference_budget <= low_budget
    ):
        raise ValueError("budgets must be positive and reference_budget must exceed low_budget")
    if (
        isinstance(bootstrap_draws, bool)
        or not isinstance(bootstrap_draws, int)
        or bootstrap_draws < 100
    ):
        raise ValueError("bootstrap_draws must be an integer >= 100")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")

    report = _load_report(report_path)
    config = report["configuration"]
    available = config.get("samples_per_src_values", [])
    if low_budget not in available or reference_budget not in available:
        raise ValueError("both requested budgets must appear in the same convergence report")
    depth_values = config.get("max_depth_values", [])
    selected_depth = max(depth_values) if depth is None else depth
    if isinstance(selected_depth, bool) or not isinstance(selected_depth, int):
        raise ValueError("depth must be an integer")
    if selected_depth not in depth_values:
        raise ValueError("depth is absent from the report configuration")

    low_rows = _index_rows(report["runs"], budget=low_budget, depth=selected_depth)
    reference_rows = _index_rows(report["runs"], budget=reference_budget, depth=selected_depth)
    if not low_rows or set(low_rows) != set(reference_rows):
        raise ValueError("requested budgets must contain the same matched link/seed rows")

    grouped: dict[tuple[bool, str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"low": [], "reference": []}
    )
    metric_pair_counts: dict[tuple[bool, str, int], int] = defaultdict(int)
    for key in sorted(low_rows):
        low, reference = low_rows[key], reference_rows[key]
        if low["geometric_los"] != reference["geometric_los"]:
            raise ValueError(f"geometric LOS label changed across budgets for {key}")
        is_los = low["geometric_los"]
        for metric in METRICS:
            low_value = _metric_value(low, metric, low_budget)
            reference_value = _metric_value(reference, metric, reference_budget)
            if low_value is None or reference_value is None:
                continue
            group = (is_los, metric, key[1])
            grouped[group]["low"].append(low_value)
            grouped[group]["reference"].append(reference_value)
            metric_pair_counts[(is_los, metric, selected_depth)] += 1

    group_keys = sorted({(is_los, metric) for is_los, metric, _ in grouped})
    rng = np.random.default_rng(bootstrap_seed)
    summaries: list[dict[str, Any]] = []
    for is_los, metric in group_keys:
        per_seed: dict[int, tuple[float, float]] = {}
        for (condition, name, seed), values in grouped.items():
            if condition == is_los and name == metric:
                per_seed[seed] = (
                    statistics.median(values["low"]),
                    statistics.median(values["reference"]),
                )
        seeds = sorted(per_seed)
        low_values = np.asarray([per_seed[seed][0] for seed in seeds], dtype=np.float64)
        reference_values = np.asarray([per_seed[seed][1] for seed in seeds], dtype=np.float64)
        deltas = low_values - reference_values
        ci95: list[float] | None = None
        if len(seeds) >= 2:
            bootstrap_indices = rng.integers(0, len(seeds), size=(bootstrap_draws, len(seeds)))
            bootstrap_delta = np.median(deltas[bootstrap_indices], axis=1)
            ci_low, ci_high = np.quantile(bootstrap_delta, [0.025, 0.975])
            ci95 = [float(ci_low), float(ci_high)]
        reference_median = float(np.median(reference_values))
        delta_median = float(np.median(deltas))
        summaries.append(
            {
                "condition": "LOS" if is_los else "NLOS",
                "metric": metric,
                "valid_seed_pairs": len(seeds),
                "valid_link_seed_pairs": metric_pair_counts[(is_los, metric, selected_depth)],
                "seeds": seeds,
                "low_budget_median": float(np.median(low_values)),
                "reference_budget_median": reference_median,
                "paired_median_delta_low_minus_reference": delta_median,
                "paired_median_relative_delta": (
                    delta_median / abs(reference_median) if reference_median != 0.0 else None
                ),
                "paired_median_delta_bootstrap_ci95": ci95,
            }
        )

    return {
        "schema": "loslab.rt_solver_observable_distribution_audit.v1",
        "evidence_grade": "descriptive paired-seed solver sensitivity; no tolerance or truth claim",
        "report": {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "source_revision": report.get("source_revision"),
            "scene_tree_sha256": report.get("scene", {}).get("scene_tree_sha256"),
            "solver_script_sha256": report.get("script_sha256"),
            "package_source_sha256": report.get("package_source_sha256"),
        },
        "auditor": {
            "script": Path(__file__).name,
            "sha256": sha256_file(Path(__file__)),
        },
        "configuration": {
            "low_samples_per_source": low_budget,
            "reference_samples_per_source": reference_budget,
            "max_depth": selected_depth,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
            "resampling_unit": "paired solver seed, after within-seed median across links",
        },
        "matched_link_seed_pairs": len(low_rows),
        "summaries": summaries,
        "limitations": [
            "A larger finite ray budget is not ground truth.",
            "Seed-level intervals do not estimate independent-scene or material uncertainty.",
            "Path-count rates are normalized by requested samples/source; they are not physical path counts.",
            "No pass/fail tolerance is applied; define tolerances before release admission.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--low-budget", type=int, required=True)
    parser.add_argument("--reference-budget", type=int, required=True)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing output: {args.output}")
    result = audit_observable_distributions(
        args.report,
        low_budget=args.low_budget,
        reference_budget=args.reference_budget,
        depth=args.depth,
        bootstrap_draws=args.bootstrap_draws,
        bootstrap_seed=args.bootstrap_seed,
    )
    atomic_json(args.output, result)
    brief = [
        {
            "condition": row["condition"],
            "metric": row["metric"],
            "low_minus_reference": row["paired_median_delta_low_minus_reference"],
            "ci95": row["paired_median_delta_bootstrap_ci95"],
            "seeds": row["valid_seed_pairs"],
        }
        for row in result["summaries"]
        if row["metric"] in {"diffuse_path_power_sum", "diffuse_rms_delay_spread_ns"}
    ]
    print(json.dumps(brief, sort_keys=True))


if __name__ == "__main__":
    main()
