"""Materialize a signed, non-copying six-cohort benchmark index from ChannelDatasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import jsonschema
from datasets.channel import ChannelDataset
from datasets.split_policy import (
    _reject_duplicate_json_keys,
    verify_openssh_signed_artifact,
    verify_signed_split_policy_registry_attestation,
)
from evaluation.benchmark_partitions import materialize_benchmark_manifest

from scripts.build_channel_split_registry import (
    _stable_file_bytes,
    _validate_npz_size,
    _write_exclusive,
)

_MAX_DATASET_BYTES = 8 * 1024 * 1024 * 1024
_MAX_COHORT_DECLARATION_BYTES = 16 * 1024 * 1024


def _attestation_dict(artifact: Any) -> dict[str, str]:
    return {
        "artifact_sha256": artifact.artifact_sha256,
        "signature_sha256": artifact.signature_sha256,
        "allowed_signers_sha256": artifact.allowed_signers_sha256,
        "signer_id": artifact.signer_id,
        "namespace": artifact.namespace,
    }


def build_benchmark_partition_manifest(
    *,
    registry_path: Path,
    registry_signature: Path,
    registry_allowed_signers: Path,
    registry_signer: str,
    declaration_path: Path,
    declaration_signature: Path,
    declaration_allowed_signers: Path,
    declaration_signer: str,
    datasets: Sequence[tuple[str, Path]],
    output: Path,
    max_dataset_bytes: int = _MAX_DATASET_BYTES,
) -> dict[str, Any]:
    """Verify both signed inventories and publish one exclusively created manifest."""
    if type(max_dataset_bytes) is not int or max_dataset_bytes <= 0:
        raise ValueError("max_dataset_bytes must be a positive integer")
    if not datasets:
        raise ValueError("at least one named dataset is required")
    destination = Path(output).absolute()
    all_inputs = [
        Path(path).absolute()
        for path in (
            registry_path,
            registry_signature,
            registry_allowed_signers,
            declaration_path,
            declaration_signature,
            declaration_allowed_signers,
        )
    ] + [Path(path).absolute() for _name, path in datasets]
    if len(set(all_inputs)) != len(all_inputs) or destination in all_inputs:
        raise ValueError("benchmark input paths must be distinct from each other and output")
    if destination.is_symlink() or destination.exists():
        raise FileExistsError("refusing to overwrite benchmark manifest")
    registry, registry_attestation = verify_signed_split_policy_registry_attestation(
        registry_path,
        registry_signature,
        trusted_allowed_signers_path=registry_allowed_signers,
        signer_id=registry_signer,
    )
    declaration_attestation = verify_openssh_signed_artifact(
        declaration_path,
        declaration_signature,
        trusted_allowed_signers_path=declaration_allowed_signers,
        signer_id=declaration_signer,
        namespace="umcd-benchmark-cohort-v1@github.com",
    )
    if len(declaration_attestation.payload) > _MAX_COHORT_DECLARATION_BYTES:
        raise ValueError("benchmark cohort declaration exceeds the 16-MiB limit")
    try:
        declaration = json.loads(
            declaration_attestation.payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("benchmark cohort declaration is not canonical UTF-8 JSON") from error
    schema_root = Path(__file__).resolve().parents[1] / "datasets" / "schema"
    declaration_schema = json.loads(
        (schema_root / "benchmark-cohort-declaration.v1.schema.json").read_text(encoding="utf-8")
    )
    try:
        jsonschema.validate(declaration, declaration_schema)
    except jsonschema.ValidationError as error:
        raise ValueError(
            f"benchmark cohort declaration violates its schema: {error.message}"
        ) from error

    loaded: list[tuple[str, str, ChannelDataset]] = []
    for name, path in sorted(datasets, key=lambda item: item[0]):
        payload = _stable_file_bytes(
            Path(path),
            label=f"benchmark ChannelDataset {name!r}",
            max_bytes=max_dataset_bytes,
        )
        _validate_npz_size(
            payload,
            label=f"benchmark ChannelDataset {name!r}",
            max_bytes=max_dataset_bytes,
        )
        loaded.append(
            (name, hashlib.sha256(payload).hexdigest(), ChannelDataset.load_bytes(payload))
        )
    manifest = materialize_benchmark_manifest(
        registry,
        declaration,
        loaded,
        registry_attestation=_attestation_dict(registry_attestation),
        declaration_attestation=_attestation_dict(declaration_attestation),
    )
    manifest_schema = json.loads(
        (schema_root / "benchmark-partition-manifest.v1.schema.json").read_text(encoding="utf-8")
    )
    try:
        jsonschema.validate(manifest, manifest_schema)
    except jsonschema.ValidationError as error:
        raise ValueError(
            f"benchmark partition manifest violates its schema: {error.message}"
        ) from error
    _write_exclusive(
        destination,
        (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    return manifest


def _dataset_arg(value: str) -> tuple[str, Path]:
    name, sep, path = value.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("--dataset must have NAME=PATH")
    return name, Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-signature", type=Path, required=True)
    parser.add_argument("--registry-allowed-signers", type=Path, required=True)
    parser.add_argument("--registry-signer", required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--declaration-signature", type=Path, required=True)
    parser.add_argument("--declaration-allowed-signers", type=Path, required=True)
    parser.add_argument("--declaration-signer", required=True)
    parser.add_argument("--dataset", type=_dataset_arg, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-dataset-bytes", type=int, default=_MAX_DATASET_BYTES)
    args = parser.parse_args(argv)
    try:
        manifest = build_benchmark_partition_manifest(
            registry_path=args.registry,
            registry_signature=args.registry_signature,
            registry_allowed_signers=args.registry_allowed_signers,
            registry_signer=args.registry_signer,
            declaration_path=args.declaration,
            declaration_signature=args.declaration_signature,
            declaration_allowed_signers=args.declaration_allowed_signers,
            declaration_signer=args.declaration_signer,
            datasets=args.dataset,
            output=args.output,
            max_dataset_bytes=args.max_dataset_bytes,
        )
    except (OSError, TypeError, ValueError, RecursionError) as error:
        parser.exit(2, f"error: {error}\n")
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "complete_six_cohort_coverage": manifest["complete_six_cohort_coverage"],
                "cohorts": manifest["cohorts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
