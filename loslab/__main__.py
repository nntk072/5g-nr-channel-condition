"""CLI for environment checks, channel data, NR observations and experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m loslab")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Report installed versions and optional RT capability")
    doctor.add_argument(
        "--probe-rt",
        action="store_true",
        help="run a tiny native probe and report the usable Mitsuba backend",
    )
    doctor.add_argument(
        "--mitsuba-variant",
        default="auto",
        help="backend for --probe-rt: auto (OptiX then LLVM), cuda, llvm, or exact variant",
    )
    city = sub.add_parser("city-scene", help="Generate reproducible 3D urban scene files")
    city.add_argument("--output-dir", required=True)
    city.add_argument("--seed", type=int, default=42)
    city.add_argument("--grid-size", type=int, default=5)
    city.add_argument("--vehicles", type=int, default=12)
    demo = sub.add_parser(
        "debug-data", help="Generate synthetic/debug smoke-test data, NOT results"
    )
    demo.add_argument("--output", required=True)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--scenes", type=int, default=5)
    demo.add_argument("--links-per-scene", type=int, default=32)
    phy = sub.add_parser(
        "phy-data", help="Generate 3GPP TDL/CDL statistical channels (proxy labels)"
    )
    phy.add_argument("--output", required=True)
    phy.add_argument("--seed", type=int, default=42)
    phy.add_argument("--samples-per-profile", type=int, default=80)
    phy.add_argument("--device", default="cpu")
    phy.add_argument("--spec-version", default="19.2")
    rt = sub.add_parser("rt-data", help="Generate geometric LOS/NLOS from full multi-path RT")
    rt.add_argument("--output", required=True)
    rt.add_argument("--seed", type=int, default=42)
    rt.add_argument("--scene", default="munich", help="Built-in scene name or scene file")
    rt.add_argument(
        "--tx-id",
        help="optional transmitter-location ID; creates a separate evaluation domain",
    )
    rt.add_argument("--tx", type=float, nargs=3, default=[8.5, 21.0, 27.0])
    rt.add_argument("--center", type=float, nargs=3, default=[44.0, 95.0, 1.5])
    rt.add_argument("--radius", type=float, default=90.0)
    rt.add_argument("--per-class", type=int, default=16)
    rt.add_argument("--max-attempts", type=int, default=2000)
    rt.add_argument("--samples-per-src", type=int, default=120000)
    rt.add_argument("--block-size", type=float, default=30.0)
    rt.add_argument("--max-depth", type=int, default=3)
    rt.add_argument(
        "--mitsuba-variant",
        default="auto",
        help="auto probes CUDA/OptiX then LLVM; use an exact variant for reproducible replay",
    )
    rt.add_argument("--allow-direct-only", action="store_true")
    rt.add_argument(
        "--cache-dir",
        help="optional immutable RT dataset cache; incomplete scene asset manifests bypass caching",
    )
    nr_observe = sub.add_parser(
        "nr-observe",
        help="Apply a Sionna PUSCH/DMRS receiver to saved RT CFR datasets",
    )
    nr_observe.add_argument("--input", nargs="+", required=True, help="RT dataset NPZ files")
    nr_observe.add_argument("--output", required=True, help="New observation dataset NPZ")
    nr_observe.add_argument("--snr-db", type=float, default=12.0)
    nr_observe.add_argument("--seed", type=int, default=20261031)
    nr_observe.add_argument("--device", default="cpu")
    nr_observe.add_argument("--n-size-bwp", type=int, default=6)
    nr_observe.add_argument("--subcarrier-spacing-khz", type=int, default=30)
    exp = sub.add_parser("experiment", help="Evaluate channel-condition baselines")
    exp.add_argument("--config", help="Versioned YAML/JSON experiment configuration")
    exp.add_argument("--input", nargs="+", help="One or more dataset NPZs")
    exp.add_argument("--output")
    exp.add_argument("--holdout", default=None, help="Unseen RT scene/domain, e.g. rt_munich")
    exp.add_argument(
        "--methods", nargs="+", default=["rf", "histgb", "mlp", "raw_mlp", "coral0", "coral"]
    )
    exp.add_argument(
        "--seeds", type=int, nargs="+", default=[11, 23, 47, 59, 71, 83, 97, 101, 109, 127]
    )
    exp.add_argument("--bootstrap", type=int, default=200)
    exp.add_argument("--device", default="cpu")
    exp.add_argument("--capacity", choices=("small", "medium", "large"), default="medium")
    exp.add_argument("--phase", action="store_true")
    exp.add_argument(
        "--observation", choices=["ideal_cfr", "comb_ls", "precomputed"], default="ideal_cfr"
    )
    exp.add_argument("--pilot-spacing", type=int, default=4)
    exp.add_argument("--pilot-snr-db", type=float, default=12.0)
    exp.add_argument("--observation-seed", type=int, default=2026)
    args = parser.parse_args()
    if args.command == "doctor":
        from .provenance import versions

        report: dict[str, object] = dict(versions())
        if args.probe_rt:
            from integrations.sionna.backend import select_rt_backend

            try:
                selection = select_rt_backend(args.mitsuba_variant)
            except (ImportError, RuntimeError) as error:
                report["rt_backend"] = {
                    "requested": args.mitsuba_variant,
                    "probe_passed": False,
                    "error": str(error),
                }
            else:
                report["rt_backend"] = selection.as_dict()
        print(json.dumps(report, indent=2))
        return
    if args.command == "city-scene":
        from .geometry import CityConfig, make_city

        p = make_city(
            args.output_dir,
            CityConfig(grid_size=args.grid_size, n_vehicles=args.vehicles, seed=args.seed),
        )
        print(f"Controlled synthetic 3D city: {p}")
        return
    if args.command == "debug-data":
        from .synthetic import debug_multipath

        d = debug_multipath(
            n_scenes=args.scenes, links_per_scene=args.links_per_scene, seed=args.seed
        )
        d.save(args.output)
        print(f"DEBUG DATA (not research evidence): saved {len(d.y)} to {args.output}")
        return
    if args.command == "phy-data":
        from .sionna2 import generate_standardized

        d = generate_standardized(
            per_profile=args.samples_per_profile,
            seed=args.seed,
            device=args.device,
            spec_version=args.spec_version,
        )
        d.save(args.output)
        print(f"3GPP PROFILE-PROXY LABELS: saved {len(d.y)} to {args.output}")
        return
    if args.command == "rt-data":
        from artifacts.provenance import atomic_json

        from .sionna2 import (
            RayScene,
            RaytraceGenerationError,
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
        try:
            d = (
                generate_raytraced_cached(cfg, cache_directory=args.cache_dir, seed=args.seed)
                if args.cache_dir
                else generate_raytraced(cfg, seed=args.seed)
            )
        except RaytraceGenerationError as error:
            failure_path = Path(args.output).with_suffix(Path(args.output).suffix + ".failure.json")
            if failure_path.exists():
                raise FileExistsError(
                    f"refusing to overwrite RT generation failure report: {failure_path}"
                ) from error
            atomic_json(failure_path, error.generation_report)
            print(f"RT generation failed; attempt report saved to {failure_path}", file=sys.stderr)
            raise
        d.save(args.output)
        print(f"GEOMETRY-LABELED RT: saved {len(d.y)} to {args.output}")
        return
    if args.command == "nr-observe":
        from datasets.channel import ChannelDataset
        from integrations.sionna.rt_dmrs import build_rt_dmrs_observation_dataset

        output_path = Path(args.output)
        if output_path.exists():
            raise FileExistsError(
                f"refusing to overwrite existing observation dataset: {output_path}"
            )
        sources = [ChannelDataset.load(path) for path in args.input]
        observed = build_rt_dmrs_observation_dataset(
            sources,
            snr_db=args.snr_db,
            seed=args.seed,
            device=args.device,
            n_size_bwp=args.n_size_bwp,
            subcarrier_spacing_khz=args.subcarrier_spacing_khz,
        )
        observed.save(output_path)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "samples": len(observed.y),
                    "shape": list(observed.h.shape),
                    "label_semantics": observed.label_semantics,
                    "observation_semantics": observed.generation_report["observation_semantics"],
                    "evidence_grade": "SIMULATION_ONLY_UNVALIDATED",
                    "warning": "RT/DMRS simulation is not measured radio evidence.",
                },
                indent=2,
            )
        )
        return
    if args.command == "experiment":
        from .experiment import run_experiment

        if args.config:
            from configuration.loader import load_config

            config = load_config(args.config)
            allowed = {
                "schema",
                "name",
                "inputs",
                "input",
                "output",
                "holdout",
                "methods",
                "seeds",
                "bootstrap",
                "include_phase",
                "device",
                "capacity",
                "observation",
                "pilot_spacing",
                "pilot_snr_db",
                "observation_seed",
            }
            unknown = set(config) - allowed
            if unknown:
                raise ValueError(f"Unknown experiment config keys: {sorted(unknown)}")
            if config.get("schema") != "loslab.experiment.v1":
                raise ValueError("Experiment config schema must be loslab.experiment.v1")
            inputs = config.get("inputs", config.get("input", args.input))
            output = config.get("output", args.output)
            options = {
                "holdout": config.get("holdout", args.holdout),
                "methods": config.get("methods", args.methods),
                "seeds": config.get("seeds", args.seeds),
                "bootstrap": config.get("bootstrap", args.bootstrap),
                "include_phase": config.get("include_phase", args.phase),
                "device": config.get("device", args.device),
                "capacity": config.get("capacity", args.capacity),
                "observation": config.get("observation", args.observation),
                "pilot_spacing": config.get("pilot_spacing", args.pilot_spacing),
                "pilot_snr_db": config.get("pilot_snr_db", args.pilot_snr_db),
                "observation_seed": config.get("observation_seed", args.observation_seed),
            }
        else:
            inputs = args.input
            output = args.output
            options = {
                "holdout": args.holdout,
                "methods": args.methods,
                "seeds": args.seeds,
                "bootstrap": args.bootstrap,
                "include_phase": args.phase,
                "device": args.device,
                "capacity": args.capacity,
                "observation": args.observation,
                "pilot_spacing": args.pilot_spacing,
                "pilot_snr_db": args.pilot_snr_db,
                "observation_seed": args.observation_seed,
            }
        if not inputs or not output:
            parser.error("experiment requires --input and --output, or --config with both")
        if isinstance(inputs, str):
            inputs = [inputs]
        result = run_experiment(inputs, output, **options)
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
