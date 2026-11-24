"""Evaluate channel-condition baselines on saved datasets from a versioned configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate channel-condition baselines")
    parser.add_argument("--config", help="Versioned YAML/JSON experiment configuration")
    parser.add_argument("--input", nargs="+", help="One or more dataset NPZs")
    parser.add_argument("--output")
    parser.add_argument("--holdout", default=None, help="Unseen RT scene/domain, e.g. rt_munich")
    parser.add_argument("--split-lock", type=Path, help="Frozen UMCD split registry JSON")
    parser.add_argument(
        "--split-signature", type=Path, help="OpenSSH detached signature for --split-lock"
    )
    parser.add_argument(
        "--split-trusted-signers", type=Path, help="Externally governed allowed_signers keyring"
    )
    parser.add_argument(
        "--split-signer-id", help="Trusted principal expected to sign the split lock"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["rf", "histgb", "mlp", "raw_mlp", "coral0", "coral"]
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[11, 23, 47, 59, 71, 83, 97, 101, 109, 127]
    )
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--capacity", choices=("small", "medium", "large"), default="medium")
    parser.add_argument("--phase", action="store_true")
    parser.add_argument(
        "--observation", choices=["ideal_cfr", "comb_ls", "precomputed"], default="ideal_cfr"
    )
    parser.add_argument("--pilot-spacing", type=int, default=4)
    parser.add_argument("--pilot-snr-db", type=float, default=12.0)
    parser.add_argument("--observation-seed", type=int, default=2026)
    parser.add_argument("--phase-randomization-seed", type=int, default=2027)
    parser.add_argument("--label-permutation-seed", type=int, default=2028)
    parser.add_argument("--row-shuffle-seed", type=int, default=2029)
    parser.add_argument("--attempt-index", type=int, default=0)
    parser.add_argument("--retry-of-manifest-sha256")
    args = parser.parse_args()
    from experiments.runner import run_experiment

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
            "phase_randomization_seed",
            "label_permutation_seed",
            "row_shuffle_seed",
            "split_lock_path",
            "split_signature_path",
            "trusted_split_signers_path",
            "split_signer_id",
            "prior_shift",
            "low_confidence_margin",
            "attempt_index",
            "retry_of_manifest_sha256",
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
            "phase_randomization_seed": config.get(
                "phase_randomization_seed", args.phase_randomization_seed
            ),
            "label_permutation_seed": config.get(
                "label_permutation_seed", args.label_permutation_seed
            ),
            "row_shuffle_seed": config.get("row_shuffle_seed", args.row_shuffle_seed),
            "split_lock_path": config.get("split_lock_path", args.split_lock),
            "split_signature_path": config.get("split_signature_path", args.split_signature),
            "trusted_split_signers_path": config.get(
                "trusted_split_signers_path", args.split_trusted_signers
            ),
            "split_signer_id": config.get("split_signer_id", args.split_signer_id),
            "prior_shift": config.get("prior_shift"),
            "low_confidence_margin": config.get("low_confidence_margin"),
            "attempt_index": config.get("attempt_index", args.attempt_index),
            "retry_of_manifest_sha256": config.get(
                "retry_of_manifest_sha256", args.retry_of_manifest_sha256
            ),
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
            "phase_randomization_seed": args.phase_randomization_seed,
            "label_permutation_seed": args.label_permutation_seed,
            "row_shuffle_seed": args.row_shuffle_seed,
            "split_lock_path": args.split_lock,
            "split_signature_path": args.split_signature,
            "trusted_split_signers_path": args.split_trusted_signers,
            "split_signer_id": args.split_signer_id,
            "prior_shift": None,
            "low_confidence_margin": None,
            "attempt_index": args.attempt_index,
            "retry_of_manifest_sha256": args.retry_of_manifest_sha256,
        }
    if not inputs or not output:
        parser.error("experiment requires --input and --output, or --config with both")
    if isinstance(inputs, str):
        inputs = [inputs]
    result = run_experiment(
        inputs,
        output,
        command_argv=tuple(sys.argv),
        command_source="cli_argv",
        **options,
    )
    print(json.dumps(result, indent=2))
    return


if __name__ == "__main__":
    main()
