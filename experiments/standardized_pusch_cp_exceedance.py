"""Paired repeated CPU probe for standardized PUSCH channels beyond the CP.

Exploratory diagnostic only; neither channel mode is physical ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import date, datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from evaluation.statistics import paired_cluster_binary_error_comparison

SCHEMA = "umcd.g5_cp_exceedance_receiver_probe.v5"
CURVE_SCHEMA = "umcd.g5_cp_exceedance_snr_curve.v4"


def wilson_interval(
    successes: int, trials: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int):
        raise TypeError("trials must be an integer")
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _summary(failures: int, trials: int, outcomes: list[bool]) -> dict[str, Any]:
    lower, upper = wilson_interval(failures, trials)
    return {
        "independent_channel_draws": trials,
        "block_errors": failures,
        "bler": failures / trials,
        "wilson_95_percent_interval": [lower, upper],
        "crc_pass_by_draw": outcomes,
    }


def _paired_table(left: Sequence[bool], right: Sequence[bool]) -> dict[str, int]:
    if len(left) != len(right) or not left:
        raise ValueError("paired outcomes must have equal nonzero lengths")
    pairs = list(zip(left, right, strict=True))
    return {
        "both_pass": sum(a and b for a, b in pairs),
        "left_only_pass": sum(a and not b for a, b in pairs),
        "right_only_pass": sum(not a and b for a, b in pairs),
        "both_fail": sum(not a and not b for a, b in pairs),
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _command_record(command: Sequence[str] | None) -> list[str]:
    """Return a validated argv record for the experiment invocation."""
    values = tuple(command) if command is not None else (sys.executable, *sys.argv)
    if not values or any(not isinstance(part, str) or not part.strip() for part in values):
        raise ValueError("command must contain nonempty string arguments")
    return list(values)


def _generated_at_utc() -> str:
    """Return an ISO-8601 UTC timestamp for a generated result record."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_snr_grid(snr_db: Sequence[float]) -> tuple[float, ...]:
    """Validate an ordered SNR grid for matched receiver comparisons."""
    if isinstance(snr_db, (str, bytes)):
        raise TypeError("snr_db must be a sequence of numeric values")
    if any(isinstance(value, bool) for value in snr_db):
        raise TypeError("snr_db values must be real numbers, not booleans")
    values = tuple(float(value) for value in snr_db)
    if len(values) < 2:
        raise ValueError("an SNR curve needs at least two points")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("snr_db values must be finite")
    if tuple(sorted(set(values))) != values:
        raise ValueError("snr_db values must be unique and strictly increasing")
    return values


def _parent_lineage(run: Any, records: Sequence[Any]) -> list[dict[str, Any]]:
    """Hash standardized channel truth and record each paired parent and seed."""
    if len(run.records) != len(records):
        raise ValueError("standardized parents and receiver records must align one-to-one")
    lineage: list[dict[str, Any]] = []
    for parent, receiver_record in zip(run.records, records, strict=True):
        digest = hashlib.sha256()
        for name in ("h_path", "tau_s", "cfr"):
            array = np.ascontiguousarray(getattr(parent, name))
            digest.update(name.encode("utf-8"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes())
        provenance = receiver_record.provenance
        parent_id = str(provenance["parent_subject_id"])
        if parent_id != parent.sample_id:
            raise RuntimeError("receiver record parent does not match its standardized draw")
        lineage.append(
            {
                "parent_subject_id": parent_id,
                "channel_truth_sha256": digest.hexdigest(),
                "pusch_transmit_seed": int(provenance["pusch_transmit_seed"]),
                "noise_seed": int(provenance["noise_seed"]),
            }
        )
    return lineage


def run_cp_exceedance_probe(
    *,
    samples: int = 16,
    snr_db: float = 60.0,
    model: str = "A",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run paired receiver views, counting each independent TDL draw once."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 4:
        raise ValueError("samples must be an integer >= 4")
    if model not in {"A", "B", "C", "D", "E"}:
        raise ValueError("model must be one of A, B, C, D, or E")
    if isinstance(snr_db, bool) or not isinstance(snr_db, (float, int)):
        raise TypeError("snr_db must be a real number")
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")

    from datasets.standardized import StandardizedFamily, standardized_run_from_channel
    from integrations.sionna.standardized_pusch import (
        build_standardized_pusch_dmrs_records,
        decode_standardized_pusch_records_for_diagnostic,
    )
    from integrations.sionna.tdl import generate_tdl_frequency_response
    from receivers.nr import NRPUSCHConfig

    channel_seed, transmit_seed, noise_seed = 20261023, 91002, 92002
    delay_spread_s = 1.0e-6
    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    channel = generate_tdl_frequency_response(
        model=model,
        seed=channel_seed,
        carrier_frequency_hz=3.5e9,
        subcarrier_spacing_hz=30_000.0,
        num_subcarriers=128,
        delay_spread_s=delay_spread_s,
        batch_size=samples,
        device="cpu",
    )
    run = standardized_run_from_channel(channel, family=StandardizedFamily.TDL)
    views = {
        mode: build_standardized_pusch_dmrs_records(
            run,
            snr_db=snr_db,
            seed=transmit_seed,
            noise_seed=noise_seed,
            receiver_config=config,
            device="cpu",
            channel_mode=mode,
        )
        for mode in ("time_domain", "ideal_cfr")
    }
    if any(len(records) != samples for records in views.values()):
        raise RuntimeError("PUSCH adapter did not preserve one record per channel draw")

    td_records, ideal_records = views["time_domain"], views["ideal_cfr"]
    if len({item.provenance["parent_subject_id"] for item in td_records}) != samples:
        raise RuntimeError("channel draws must have distinct parent identities")
    sampled_channel = td_records[0].provenance["sampled_time_channel"]
    cyclic_prefix_samples = int(sampled_channel["cyclic_prefix_length_samples"])
    sampling_frequency_hz = float(sampled_channel["sampling_frequency_hz"])
    cyclic_prefix_duration_s = cyclic_prefix_samples / sampling_frequency_hz
    max_excess_delay_s = float(channel.tau_s.max().item())
    if max_excess_delay_s <= cyclic_prefix_duration_s:
        raise RuntimeError("the physical TDL excess delay unexpectedly fits inside the CP")
    for td, ideal in zip(td_records, ideal_records, strict=True):
        for key in ("parent_subject_id", "pusch_transmit_seed", "noise_seed"):
            if td.provenance[key] != ideal.provenance[key]:
                raise RuntimeError(f"paired receiver views disagree on {key}")
        if td.provenance["sampled_time_channel"]["tap_span_fits_cyclic_prefix"]:
            raise RuntimeError("the time-domain channel unexpectedly fits inside the CP")

    outcomes: dict[str, list[bool]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for mode, records in views.items():
        for estimate_kind in ("dmrs_ls", "diagonal_cfr_oracle"):
            key = f"{mode}.{estimate_kind}"
            result = decode_standardized_pusch_records_for_diagnostic(
                records,
                receiver_config=config,
                estimate_source=estimate_kind,
            )
            passed = list(result)
            outcomes[key] = passed
            failures = samples - sum(passed)
            summaries[key] = _summary(failures, samples, passed)

    paired = {
        estimate: _paired_table(
            outcomes[f"time_domain.{estimate}"], outcomes[f"ideal_cfr.{estimate}"]
        )
        for estimate in ("dmrs_ls", "diagonal_cfr_oracle")
    }
    parent_ids = [item.provenance["parent_subject_id"] for item in td_records]
    parent_lineage = _parent_lineage(run, td_records)
    paired_bler_comparisons = {
        estimate: paired_cluster_binary_error_comparison(
            control_errors=[not passed for passed in outcomes[f"ideal_cfr.{estimate}"]],
            treatment_errors=[not passed for passed in outcomes[f"time_domain.{estimate}"]],
            groups=parent_ids,
            seed=channel_seed,
        )
        for estimate in ("dmrs_ls", "diagonal_cfr_oracle")
    }
    configuration = {
        "channel_family": f"Sionna TDL-{model} stochastic channel draws",
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30_000.0,
        "source_cfr_subcarriers": 128,
        "delay_spread_s": delay_spread_s,
        "max_excess_delay_s": max_excess_delay_s,
        "cyclic_prefix_samples": cyclic_prefix_samples,
        "sampling_frequency_hz": sampling_frequency_hz,
        "cyclic_prefix_duration_s": cyclic_prefix_duration_s,
        "pusch_resource_blocks": 6,
        "snr_db": snr_db,
        "channel_seed": channel_seed,
        "transmit_seed": transmit_seed,
        "noise_seed": noise_seed,
        "samples": samples,
        "trial_unit": "independent TDL draw; receiver views are paired descendants",
        "receiver_estimate": "DMRS LS or diagnostic diagonal-CFR oracle; oracle excluded from model inputs",
    }
    config_hash = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    root = Path(__file__).resolve().parents[1]
    source_paths = (
        Path("experiments/standardized_pusch_cp_exceedance.py"),
        Path("integrations/sionna/standardized_pusch.py"),
        Path("integrations/sionna/tdl.py"),
        Path("datasets/standardized.py"),
        Path("integrations/sionna/nr_pusch.py"),
        Path("evaluation/statistics.py"),
    )
    source_hashes = {
        str(path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in source_paths
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    return {
        "schema": SCHEMA,
        "status": "exploratory; does not pass G5 or approve production data",
        "run_date": date.today().isoformat(),
        "generated_at_utc": _generated_at_utc(),
        "command": _command_record(command),
        "repository_commit_at_run": revision.stdout.strip()
        if revision.returncode == 0
        else "unavailable",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": _package_version("torch"),
            "sionna": _package_version("sionna"),
        },
        "source_files_sha256": source_hashes,
        "source_sha256": source_hashes,
        "configuration": configuration,
        "configuration_sha256": config_hash,
        "outcomes": summaries,
        "paired_pass_table": paired,
        "paired_time_domain_minus_ideal_cfr_bler": paired_bler_comparisons,
        "parent_lineage": parent_lineage,
        "interpretation": [
            "One independent standardized channel draw is the trial unit; paired views are not extra trials.",
            "Paired BLER intervals resample complete independent channel-parent groups, retaining paired receiver outcomes.",
            "ideal_cfr removes time-domain CP spillover by construction; it is a diagnostic control, not propagation truth.",
            "diagonal_cfr_oracle separates DMRS-estimation error from diagonal-model mismatch; it is not a model input.",
            "Intervals are wide at this sample count; the probe does not qualify general NR receiver performance or close G5.09.",
        ],
    }


def run_cp_exceedance_snr_curve(
    *,
    samples: int = 16,
    snr_db: Sequence[float] = (60.0, 80.0, 100.0),
    model: str = "A",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run a matched CPU SNR curve over identical standardized channel parents.

    Each SNR point is regenerated from the same channel, transmit, and noise
    seeds. Parent channel-truth hashes and transmit/noise child seeds are
    compared explicitly before any curve is returned. The receiver's noise
    variance changes with SNR while its random stream remains paired.
    """
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 4:
        raise ValueError("samples must be an integer >= 4")
    snr_values = _validate_snr_grid(snr_db)
    reports = [
        run_cp_exceedance_probe(samples=samples, snr_db=snr, model=model, command=command)
        for snr in snr_values
    ]
    reference = reports[0]
    reference_configuration = dict(reference["configuration"])
    reference_configuration.pop("snr_db")
    reference_lineage = reference["parent_lineage"]
    for snr, report in zip(snr_values[1:], reports[1:], strict=True):
        candidate_configuration = dict(report["configuration"])
        candidate_configuration.pop("snr_db")
        if candidate_configuration != reference_configuration:
            raise RuntimeError("SNR points do not share the same frozen channel/receiver config")
        if report["parent_lineage"] != reference_lineage:
            raise RuntimeError(f"SNR point {snr:g} does not reuse identical parent channel draws")

    points = []
    point_hashes = []
    for report in reports:
        point = {
            "snr_db": report["configuration"]["snr_db"],
            "outcomes": report["outcomes"],
            "paired_pass_table": report["paired_pass_table"],
            "paired_time_domain_minus_ideal_cfr_bler": report[
                "paired_time_domain_minus_ideal_cfr_bler"
            ],
        }
        points.append(point)
        point_hashes.append(
            hashlib.sha256(
                json.dumps(point, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
                    "utf-8"
                )
            ).hexdigest()
        )
    curve_configuration = {
        "channel_family": reference_configuration["channel_family"],
        "channel_and_receiver_configuration": reference_configuration,
        "snr_db": list(snr_values),
        "samples_per_snr": samples,
        "independent_parent_count": len(reference_lineage),
        "channel_seed": reference_configuration["channel_seed"],
        "transmit_seed": reference_configuration["transmit_seed"],
        "noise_seed": reference_configuration["noise_seed"],
        "trial_unit": "one independent standardized channel parent, reused across SNR points",
        "snr_definition": (
            "post-channel, pre-beamforming receive-grid mean complex power divided by "
            "proper-complex AWGN variance; noise seeds are held fixed across SNR points"
        ),
    }
    curve_hash = hashlib.sha256(
        json.dumps(
            curve_configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    root = Path(__file__).resolve().parents[1]
    source_hashes = reference["source_files_sha256"]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=False, capture_output=True, text=True
    )
    return {
        "schema": CURVE_SCHEMA,
        "status": "exploratory; does not pass G5 or approve production data",
        "run_date": date.today().isoformat(),
        "generated_at_utc": _generated_at_utc(),
        "command": _command_record(command),
        "repository_commit_at_run": revision.stdout.strip()
        if revision.returncode == 0
        else "unavailable",
        "runtime": reference["runtime"],
        "source_files_sha256": source_hashes,
        "source_sha256": source_hashes,
        "configuration": curve_configuration,
        "configuration_sha256": curve_hash,
        "parent_lineage": reference_lineage,
        "point_report_sha256": point_hashes,
        "points": points,
        "interpretation": [
            "All SNR points use identical standardized channel truth, parent IDs, transmit seeds, and noise seeds.",
            "Different SNRs change the noise variance; they are paired receiver views, not independent channel evidence.",
            f"Parent-cluster intervals are conditional on the sampled TDL-{model} family and do not support scene/site generalization.",
            "The finite sinc-discretized time-domain TDL and ideal-CFR control are diagnostics, not physical truth.",
            "This is not a deployment BLER curve, standards certification, G5 pass, or G3 receiver qualification.",
        ],
    }


def write_report(report: dict[str, Any], path: str | Path) -> None:
    """Create the report immutably; an existing file is accepted only if identical."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    try:
        with destination.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError:
        if destination.read_bytes() != encoded:
            raise FileExistsError(f"refusing to replace a different report: {destination}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[60.0])
    parser.add_argument("--profile", choices=("A", "B", "C", "D", "E"), default="A")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    invocation = [
        "uv",
        "run",
        "--no-sync",
        "--locked",
        "python",
        "experiments/standardized_pusch_cp_exceedance.py",
        *(list(argv) if argv is not None else sys.argv[1:]),
    ]
    profile_suffix = "" if args.profile == "A" else f"-tdl-{args.profile.lower()}"
    if len(args.snr_db) == 1:
        report = run_cp_exceedance_probe(
            samples=args.samples,
            snr_db=args.snr_db[0],
            model=args.profile,
            command=invocation,
        )
        default_output = Path(
            "docs/research/results/g5-standardized-pusch-cp-exceedance"
            f"{profile_suffix}-cluster-bootstrap-v5-20261023.json"
        )
        print_summary = report["outcomes"]
    else:
        report = run_cp_exceedance_snr_curve(
            samples=args.samples,
            snr_db=args.snr_db,
            model=args.profile,
            command=invocation,
        )
        default_output = Path(
            "docs/research/results/g5-standardized-pusch-cp-exceedance"
            f"{profile_suffix}-snr-curve-v4-20261024.json"
        )
        print_summary = {str(point["snr_db"]): point["outcomes"] for point in report["points"]}
    output_path = args.output or default_output
    write_report(report, output_path)
    print(json.dumps({"output": str(output_path), "outcomes": print_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
