"""Generate a bounded, explicitly configured 3GPP scenario-GBSM run store."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from datasets.episode_store import _stable_file_bytes, write_episode_bundle
from datasets.scenario_gbsm import (
    scenario_gbsm_run_from_channel,
    scenario_gbsm_run_to_episode_bundles,
    write_scenario_gbsm_run,
)
from integrations.sionna.cdl import CDLArrayConfig
from integrations.sionna.scenario_gbsm import generate_scenario_gbsm_frequency_response

_MAX_CONFIG_BYTES = 1_048_576
_MAX_BATCH_SIZE = 128
_MAX_SUBCARRIERS = 4096
_MAX_TIME_STEPS = 256
_MAX_GRID_ELEMENTS = 2_000_000
_ROOT_FIELDS = {
    "schema", "run_id", "sample_ids", "scenario", "seed", "carrier_frequency_hz",
    "subcarrier_spacing_hz", "num_subcarriers", "topology", "generator",
}
_TOPOLOGY_REQUIRED = {"ut_loc", "bs_loc", "ut_velocities", "in_state", "los"}
_TOPOLOGY_OPTIONAL = {"ut_orientations", "bs_orientations"}
_GENERATOR_FIELDS = {
    "direction", "num_time_steps", "sampling_frequency_hz", "scenario_options",
    "enable_pathloss", "enable_shadow_fading", "spec_version", "device", "ut_array",
    "bs_array",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"configuration contains duplicate key {key!r}")
        result[key] = value
    return result


def _read_config(path: Path) -> tuple[dict[str, Any], str]:
    payload = _stable_file_bytes(
        path,
        label="scenario GBSM configuration",
        max_bytes=_MAX_CONFIG_BYTES,
    )
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("configuration must be valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError("configuration root must be an object")
    if set(value) != _ROOT_FIELDS:
        raise ValueError("configuration has missing or unknown root fields")
    if value["schema"] != "umcd.scenario_gbsm_generation.v1":
        raise ValueError("unsupported scenario-GBSM generation schema")
    return value, hashlib.sha256(payload).hexdigest()


def _validate_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_ids = config["sample_ids"]
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or len(sample_ids) > _MAX_BATCH_SIZE
        or any(
            not isinstance(item, str) or not item.strip() or item != item.strip()
            for item in sample_ids
        )
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError(f"sample_ids must contain 1–{_MAX_BATCH_SIZE} unique nonempty IDs")
    if (
        not isinstance(config["run_id"], str)
        or not config["run_id"].strip()
        or config["run_id"] != config["run_id"].strip()
    ):
        raise ValueError("run_id must be a nonempty string")
    if not isinstance(config["scenario"], str) or config["scenario"] not in {
        "UMi",
        "UMa",
        "RMa",
        "InH",
        "InF",
    }:
        raise ValueError("scenario must be one of UMi, UMa, RMa, InH, or InF")
    seed = config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for name in ("carrier_frequency_hz", "subcarrier_spacing_hz"):
        value = config[name]
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    subcarriers = config["num_subcarriers"]
    if (
        isinstance(subcarriers, bool) or not isinstance(subcarriers, int)
        or not 8 <= subcarriers <= _MAX_SUBCARRIERS
    ):
        raise ValueError(f"num_subcarriers must be in [8, {_MAX_SUBCARRIERS}]")
    topology = config["topology"]
    if (
        not isinstance(topology, dict)
        or not _TOPOLOGY_REQUIRED.issubset(topology)
        or set(topology) - (_TOPOLOGY_REQUIRED | _TOPOLOGY_OPTIONAL)
    ):
        raise ValueError("topology does not match the explicit per-drop topology contract")
    batch = len(sample_ids)
    for name in ("ut_loc", "bs_loc", "ut_velocities", "ut_orientations", "bs_orientations"):
        if name not in topology and name in _TOPOLOGY_OPTIONAL:
            continue
        rows = topology[name]
        if (
            not isinstance(rows, list) or len(rows) != batch
            or any(
                not isinstance(row, list) or len(row) != 3
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, (int, float))
                    or not math.isfinite(float(component))
                    for component in row
                )
                for row in rows
            )
        ):
            raise ValueError(f"topology.{name} must contain one finite 3-vector per sample")
    for name in ("in_state", "los"):
        values = topology[name]
        if not isinstance(values, list) or len(values) != batch or any(type(v) is not bool for v in values):
            raise ValueError(f"topology.{name} must contain one Boolean per sample")
    generator = config["generator"]
    if not isinstance(generator, dict) or set(generator) - _GENERATOR_FIELDS:
        raise ValueError("generator has unknown fields")
    time_steps = generator.get("num_time_steps", 1)
    if (
        isinstance(time_steps, bool) or not isinstance(time_steps, int)
        or not 1 <= time_steps <= _MAX_TIME_STEPS
    ):
        raise ValueError(f"generator.num_time_steps must be in [1, {_MAX_TIME_STEPS}]")
    sampling_frequency = generator.get("sampling_frequency_hz")
    if (
        isinstance(sampling_frequency, bool)
        or not isinstance(sampling_frequency, (int, float))
        or not math.isfinite(float(sampling_frequency))
        or sampling_frequency <= 0
    ):
        raise ValueError("generator.sampling_frequency_hz must be explicitly finite and positive")
    if batch * subcarriers * time_steps > _MAX_GRID_ELEMENTS:
        raise ValueError(f"batch × subcarriers × time_steps exceeds {_MAX_GRID_ELEMENTS} elements")
    for name in ("ut_array", "bs_array"):
        array = generator.get(name, {})
        if not isinstance(array, dict):
            raise ValueError(f"generator.{name} must be an object")
        array_config = CDLArrayConfig(**array)
        polarization_ports = 2 if array_config.polarization == "dual" else 1
        element_count = (
            array_config.num_rows_per_panel
            * array_config.num_cols_per_panel
            * array_config.num_rows
            * array_config.num_cols
            * polarization_ports
        )
        if element_count > 256:
            raise ValueError(f"generator.{name} may declare at most 256 antenna ports")
    options = generator.get("scenario_options", {})
    if not isinstance(options, dict):
        raise ValueError("generator.scenario_options must be an object")
    native_topology = {
        "ut_loc": [[row] for row in topology["ut_loc"]],
        "bs_loc": [[row] for row in topology["bs_loc"]],
        "ut_velocities": [[row] for row in topology["ut_velocities"]],
        "in_state": [[value] for value in topology["in_state"]],
        "los": [[[value]] for value in topology["los"]],
    }
    for name in _TOPOLOGY_OPTIONAL:
        if name in topology:
            native_topology[name] = [[row] for row in topology[name]]
    return native_topology, generator


def generate_and_store(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config, config_sha256 = _read_config(config_path)
    topology, generator = _validate_config(config)
    channel = generate_scenario_gbsm_frequency_response(
        scenario=config["scenario"],
        seed=config["seed"],
        carrier_frequency_hz=float(config["carrier_frequency_hz"]),
        subcarrier_spacing_hz=float(config["subcarrier_spacing_hz"]),
        num_subcarriers=config["num_subcarriers"],
        topology=topology,
        ut_array=CDLArrayConfig(**generator.get("ut_array", {})),
        bs_array=CDLArrayConfig(**generator.get("bs_array", {})),
        direction=generator.get("direction", "downlink"),
        num_time_steps=generator.get("num_time_steps", 1),
        sampling_frequency_hz=generator.get("sampling_frequency_hz", 1.0),
        scenario_options=generator.get("scenario_options", {}),
        enable_pathloss=generator.get("enable_pathloss", True),
        enable_shadow_fading=generator.get("enable_shadow_fading", True),
        spec_version=generator.get("spec_version", "19.2"),
        device=generator.get("device", "cpu"),
    )
    run = scenario_gbsm_run_from_channel(
        channel, run_id=config["run_id"], sample_ids=config["sample_ids"]
    )
    episode_root = output_dir / "episodes"
    bundles = scenario_gbsm_run_to_episode_bundles(run)
    if tuple(bundle.episode.episode_id for bundle in bundles) != tuple(
        record.sample_id for record in run.records
    ):
        raise ValueError("canonical episode bundles must exactly cover the scenario run roster")
    episode_pointers = [
        write_episode_bundle(episode_root, bundle)
        for bundle in bundles
    ]
    expected_episode_ids = tuple(record.sample_id for record in run.records)
    if tuple(pointer.episode_id for pointer in episode_pointers) != expected_episode_ids:
        raise ValueError("published episode manifests must exactly cover the scenario run roster")
    pointer = write_scenario_gbsm_run(output_dir, run)
    return {
        "schema": "umcd.scenario_gbsm_generation_report.v2",
        "status": "SYNTHETIC_STANDARDIZED_MODEL_DATA_NOT_QUALIFIED",
        "run_id": run.run_id,
        "scenario": config["scenario"],
        "sample_count": len(run.records),
        "configuration_file_sha256": config_sha256,
        "configuration_sha256": run.records[0].metadata["configuration_sha256"],
        "scenario_los_counts": {
            "los": sum(record.scenario_los for record in run.records),
            "nlos": sum(not record.scenario_los for record in run.records),
        },
        "geometric_los_labels_created": False,
        "store_manifest": {
            "run_id": pointer.run_id,
            "content_sha256": pointer.content_sha256,
            "relative_path": pointer.relative_path,
            "sample_ids": list(pointer.sample_ids),
            "schema_version": pointer.schema_version,
        },
        "episode_store_manifests": [
            {
                "episode_id": item.episode_id,
                "bundle_sha256": item.bundle_sha256,
                "relative_path": item.relative_path,
                "schema_version": item.schema_version,
            }
            for item in episode_pointers
        ],
        "interpretation": (
            "TR 38.901 scenario-GBSM simulation output; not RT path truth, measured data, "
            "statistical conformance, or release qualification"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="bounded v1 JSON config")
    parser.add_argument("--output-dir", type=Path, required=True, help="scenario run-store root")
    args = parser.parse_args()
    try:
        report = generate_and_store(args.config, args.output_dir)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\\n")
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
