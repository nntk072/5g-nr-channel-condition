"""Build receiver-visible PUSCH/DMRS records from Sionna TDL or CDL draws."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from artifacts.provenance import package_source_sha256, sha256_file, versions
from datasets.nr_store import write_nr_record
from datasets.standardized import StandardizedFamily, standardized_run_from_channel
from integrations.sionna.standardized_pusch import build_standardized_pusch_dmrs_records
from receivers.nr import NRPUSCHConfig


def _publish_immutable_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish one immutable report, accepting exact replays only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"refusing to replace an immutable build report: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="directory for immutable record manifests/data")
    parser.add_argument("--family", choices=("tdl", "cdl"), default="tdl")
    parser.add_argument("--profile", choices=tuple("ABCDE"), default="A")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.seed < 0 or args.seed > 2**63 - 3:
        parser.error("--seed must be in [0, 2**63 - 3]")

    common = {
        "model": args.profile,
        "seed": args.seed,
        "carrier_frequency_hz": 3.5e9,
        "subcarrier_spacing_hz": 30_000.0,
        "num_subcarriers": 256,
        "delay_spread_s": 100e-9,
        "batch_size": args.samples,
        "device": args.device,
    }
    if args.family == "tdl":
        from integrations.sionna.tdl import generate_tdl_frequency_response

        channel: Any = generate_tdl_frequency_response(**common)
    else:
        from integrations.sionna.cdl import generate_cdl_frequency_response

        channel = generate_cdl_frequency_response(**common)

    family = StandardizedFamily(args.family)
    run = standardized_run_from_channel(channel, family=family)
    records = build_standardized_pusch_dmrs_records(
        run,
        snr_db=args.snr_db,
        seed=args.seed + 1,
        receiver_config=NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30),
        device=args.device,
    )
    manifests = [write_nr_record(args.output, record) for record in records]
    report = {
        "schema": "standardized-pusch-build.v1",
        "run_id": run.run_id,
        "family": args.family,
        "profile": args.profile,
        "profile_label_semantics": "proxy only; not geometric LOS/NLOS",
        "samples": len(records),
        "snr_db": args.snr_db,
        "device": args.device,
        "output": str(args.output.resolve()),
        "channel_generator_metadata": channel.metadata,
        "receiver_config": asdict(NRPUSCHConfig(n_size_bwp=6, subcarrier_spacing_khz=30)),
        "provenance": {
            "source_revision": versions()["git_sha"],
            "package_source_sha256": package_source_sha256(),
            "runner_sha256": sha256_file(__file__),
            "runtime_versions": versions(),
        },
        "record_manifests": [
            {
                **asdict(manifest),
                "manifest_path": (
                    f"manifests/{hashlib.sha256(manifest.sample_id.encode()).hexdigest()}.json"
                ),
            }
            for manifest in manifests
        ],
        "link_model": "ideal frequency-domain static channel; no ICI/ISI or impairments",
    }
    report_path = args.output / "build-report.json"
    _publish_immutable_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
