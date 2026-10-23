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
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from evaluation.statistics import paired_cluster_binary_error_comparison

SCHEMA = "umcd.g5_cp_exceedance_receiver_probe.v2"


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


def _summary(
    failures: int, trials: int, payload_mismatches: int, outcomes: list[bool]
) -> dict[str, Any]:
    lower, upper = wilson_interval(failures, trials)
    return {
        "independent_channel_draws": trials,
        "block_errors": failures,
        "bler": failures / trials,
        "wilson_95_percent_interval": [lower, upper],
        "payload_mismatches": payload_mismatches,
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


def run_cp_exceedance_probe(*, samples: int = 16) -> dict[str, Any]:
    """Run paired receiver views, counting each independent TDL draw once."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 4:
        raise ValueError("samples must be an integer >= 4")

    from datasets.standardized import StandardizedFamily, standardized_run_from_channel
    from integrations.sionna.standardized_pusch import (
        build_standardized_pusch_dmrs_records,
        decode_standardized_pusch_records_for_diagnostic,
    )
    from integrations.sionna.tdl import generate_tdl_frequency_response
    from receivers.nr import NRPUSCHConfig

    channel_seed, transmit_seed, noise_seed = 20261023, 91002, 92002
    snr_db, delay_spread_s = 60.0, 1.0e-6
    config = NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)
    channel = generate_tdl_frequency_response(
        model="A",
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
            result, payload_mismatches = decode_standardized_pusch_records_for_diagnostic(
                records,
                receiver_config=config,
                estimate_source=estimate_kind,
            )
            passed = list(result)
            outcomes[key] = passed
            failures = samples - sum(passed)
            lower, upper = wilson_interval(failures, samples)
            summaries[key] = {
                "independent_channel_draws": samples,
                "block_errors": failures,
                "bler": failures / samples,
                "wilson_95_percent_interval": [lower, upper],
                "payload_mismatches": payload_mismatches,
                "crc_pass_by_draw": passed,
            }

    paired = {
        estimate: _paired_table(
            outcomes[f"time_domain.{estimate}"], outcomes[f"ideal_cfr.{estimate}"]
        )
        for estimate in ("dmrs_ls", "diagonal_cfr_oracle")
    }
    parent_ids = [item.provenance["parent_subject_id"] for item in td_records]
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
        "channel_family": "Sionna TDL-A stochastic channel draws",
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
        "configuration": configuration,
        "configuration_sha256": config_hash,
        "outcomes": summaries,
        "paired_pass_table": paired,
        "paired_time_domain_minus_ideal_cfr_bler": paired_bler_comparisons,
        "interpretation": [
            "One independent standardized channel draw is the trial unit; paired views are not extra trials.",
            "Paired BLER intervals resample complete independent channel-parent groups, retaining paired receiver outcomes.",
            "ideal_cfr removes time-domain CP spillover by construction; it is a diagnostic control, not propagation truth.",
            "diagonal_cfr_oracle separates DMRS-estimation error from diagonal-model mismatch; it is not a model input.",
            "Intervals are wide at this sample count; the probe does not qualify general NR receiver performance or close G5.09.",
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/research/results/g5-standardized-pusch-cp-exceedance-cluster-bootstrap-20261023.json"
        ),
    )
    args = parser.parse_args(argv)
    report = run_cp_exceedance_probe(samples=args.samples)
    write_report(report, args.output)
    print(json.dumps({"output": str(args.output), "outcomes": report["outcomes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
