"""Development-only DM-RS density scan at a selected waveform-noise variance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from experiments import scan_g5_doppler_dmrs_density as dmrs_scan

SCHEMA = "umcd.g5_doppler_dmrs_noise_operating_point.v1"
DEFAULT_SEEDS = tuple(range(86_001, 86_017))


def _validate_operating_point(
    channel_seeds: tuple[int, ...], blocks_per_channel: int, waveform_noise_variance: float
) -> None:
    if not channel_seeds or len(set(channel_seeds)) != len(channel_seeds):
        raise ValueError("channel_seeds must be nonempty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in channel_seeds
    ):
        raise ValueError("channel_seeds must be nonnegative integers")
    if isinstance(blocks_per_channel, bool) or not isinstance(blocks_per_channel, int):
        raise TypeError("blocks_per_channel must be an integer")
    if blocks_per_channel < 1:
        raise ValueError("blocks_per_channel must be positive")
    if (
        isinstance(waveform_noise_variance, bool)
        or not isinstance(waveform_noise_variance, (float, int))
        or not math.isfinite(waveform_noise_variance)
        or waveform_noise_variance <= 0
    ):
        raise ValueError("waveform_noise_variance must be finite and positive")


def run_noise_operating_point(
    *,
    waveform_noise_variance: float,
    channel_seeds: tuple[int, ...] = DEFAULT_SEEDS,
    blocks_per_channel: int = dmrs_scan.BLOCKS_PER_CHANNEL,
) -> dict[str, Any]:
    """Run matched development channels without changing the historical scan source."""
    _validate_operating_point(channel_seeds, blocks_per_channel, waveform_noise_variance)
    previous_variance = dmrs_scan.WAVEFORM_NOISE_VARIANCE
    try:
        dmrs_scan.WAVEFORM_NOISE_VARIANCE = float(waveform_noise_variance)
        report = dmrs_scan.run_dmrs_density_scan(
            channel_seeds=channel_seeds,
            blocks_per_channel=blocks_per_channel,
        )
    finally:
        dmrs_scan.WAVEFORM_NOISE_VARIANCE = previous_variance

    root = Path(__file__).resolve().parents[1]
    report["schema"] = SCHEMA
    report["source_sha256"][Path(__file__).relative_to(root).as_posix()] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["configuration"]["purpose"] = "development_noise_operating_point_selection"
    report["status"] = "DEVELOPMENT_ONLY_NOT_QUALIFIED"
    report.pop("result_sha256", None)
    report["result_sha256"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waveform-noise-variance", type=float, required=True)
    parser.add_argument("--channel-seed-start", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument("--channel-count", type=int, default=len(DEFAULT_SEEDS))
    parser.add_argument("--blocks-per-channel", type=int, default=dmrs_scan.BLOCKS_PER_CHANNEL)
    args = parser.parse_args()
    if args.channel_count <= 0:
        parser.error("channel-count must be positive")
    seeds = tuple(range(args.channel_seed_start, args.channel_seed_start + args.channel_count))
    report = run_noise_operating_point(
        waveform_noise_variance=args.waveform_noise_variance,
        channel_seeds=seeds,
        blocks_per_channel=args.blocks_per_channel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.output}: noise_variance={args.waveform_noise_variance}, "
        f"channels={len(seeds)}, status={report['status']}"
    )


if __name__ == "__main__":
    main()
