"""Generate geometric LOS/NLOS RT datasets from full multi-path ray tracing.

Production generation requires a validated qualification approval manifest. Pilot runs are
bounded and write under the OS temporary directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate geometric LOS/NLOS RT datasets from full multi-path ray tracing"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scene", default="munich", help="Built-in scene name or scene file")
    parser.add_argument(
        "--tx-id",
        help="optional transmitter-location ID; creates a separate evaluation domain",
    )
    parser.add_argument("--tx", type=float, nargs=3, default=[8.5, 21.0, 27.0])
    parser.add_argument("--center", type=float, nargs=3, default=[44.0, 95.0, 1.5])
    parser.add_argument("--radius", type=float, default=90.0)
    parser.add_argument("--per-class", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--samples-per-src", type=int, default=120000)
    parser.add_argument("--block-size", type=float, default=30.0)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument(
        "--mitsuba-variant",
        default="auto",
        help="auto probes CUDA/OptiX then LLVM; use an exact variant for reproducible replay",
    )
    parser.add_argument("--allow-direct-only", action="store_true")
    parser.add_argument(
        "--cache-dir",
        help="optional immutable RT dataset cache; incomplete scene asset manifests bypass caching",
    )
    rt_policy = parser.add_mutually_exclusive_group()
    rt_policy.add_argument(
        "--pilot",
        action="store_true",
        help="allow bounded exploratory work with outputs under the OS temp directory",
    )
    rt_policy.add_argument(
        "--qualification-approval",
        type=Path,
        help="reserved; production stays disabled until a trusted signature keyring is configured",
    )
    parser.add_argument(
        "--qualification-approval-sha256",
        help="reserved while production authorization is disabled",
    )
    args = parser.parse_args()
    from artifacts.provenance import atomic_json
    from integrations.sionna.adapter import (
        RayScene,
        RaytraceGenerationError,
        authorize_rt_data_generation,
        generate_raytraced,
        generate_raytraced_cached,
    )

    cfg = RayScene(
        name=args.scene,
        tx_position=tuple(args.tx),
        transmitter_id=args.tx_id,
        rx_center=tuple(args.center),
        radius_m=args.radius,
        per_class=args.per_class,
        max_attempts=args.max_attempts,
        samples_per_src=args.samples_per_src,
        block_size_m=args.block_size,
        max_depth=args.max_depth,
        require_los_multipath=not args.allow_direct_only,
        mitsuba_variant=args.mitsuba_variant,
    )
    if args.pilot and (
        args.per_class > 20
        or args.max_attempts > 2000
        or args.samples_per_src > 5_000_000
        or args.max_depth > 3
    ):
        parser.error(
            "pilot RT generation is bounded to 20 samples/class, 2,000 attempts, "
            "5,000,000 samples/source, and depth 3"
        )
    if not args.pilot and (
        args.qualification_approval is None or args.qualification_approval_sha256 is None
    ):
        parser.error(
            "production RT generation requires a validated qualification approval manifest "
            "(--qualification-approval) and --qualification-approval-sha256"
        )
    if not args.pilot and not args.qualification_approval_sha256:
        parser.error("--qualification-approval-sha256 is required with production approval")
    try:
        destinations = [args.output]
        failure_path = Path(args.output).with_suffix(Path(args.output).suffix + ".failure.json")
        destinations.append(failure_path)
        if args.cache_dir:
            destinations.append(args.cache_dir)
        authorization = authorize_rt_data_generation(
            cfg,
            seed=args.seed,
            destinations=destinations,
            pilot=args.pilot,
            approval_manifest=args.qualification_approval,
            expected_approval_sha256=args.qualification_approval_sha256,
        )
        d = (
            generate_raytraced_cached(
                cfg,
                cache_directory=args.cache_dir,
                seed=args.seed,
                authorization=authorization,
                output_destination=args.output,
                failure_report_destination=failure_path,
            )
            if args.cache_dir
            else generate_raytraced(
                cfg,
                seed=args.seed,
                authorization=authorization,
                output_destination=args.output,
                failure_report_destination=failure_path,
            )
        )
    except RaytraceGenerationError as error:
        if failure_path.exists():
            raise FileExistsError(
                f"refusing to overwrite RT generation failure report: {failure_path}"
            ) from error
        try:
            atomic_json(failure_path, error.generation_report, overwrite=False)
            print(f"RT generation failed; attempt report saved to {failure_path}", file=sys.stderr)
        except OSError as report_error:
            print(
                f"RT generation failed; attempt report could not be saved to {failure_path}: "
                f"{report_error}",
                file=sys.stderr,
            )
        raise
    except (PermissionError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    d.save(args.output)
    print(f"GEOMETRY-LABELED RT: saved {len(d.y)} to {args.output}")
    return


if __name__ == "__main__":
    main()
