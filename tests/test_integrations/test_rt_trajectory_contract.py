"""Framework-independent RT trajectory configuration checks."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from artifacts.provenance import canonical_hash
from integrations.sionna.d1_generation_gate import (
    authorize_d1_generation,
    require_d1_authorization,
)
from integrations.sionna.rt_identity import rt_run_identity
from integrations.sionna.rt_trajectory import (
    RTTrajectoryConfig,
    build_paired_frequency_rt_requests,
    build_rt_trajectory_generation_request,
    rt_trajectory_physical_config,
)
from propagation.scene_visibility import SceneTriangleInventory


def _trajectory_request_inputs() -> dict[str, object]:
    config = RTTrajectoryConfig(
        trajectory_id="route-a",
        scene="simple_reflector",
        transmitter_position_m=(1.0, 2.0, 3.0),
        receiver_start_m=(4.0, 5.0, 6.0),
        receiver_velocity_mps=(1.0, 0.0, 0.0),
        num_steps=2,
    )
    inventory = SceneTriangleInventory(
        object_names=("wall",),
        triangles_m=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        triangle_object_names=("wall",),
        scene_geometry_sha256="a" * 64,
        meters_per_unit=1.0,
    )
    return {
        "config": config,
        "spec_sha256": "b" * 64,
        "store_root": "/tmp/trajectory-store",
        "umcd_store_root": None,
        "package_source_digest": "c" * 64,
        "backend": {"backend": "llvm_cpu", "variant": "llvm_ad_mono_polarized"},
        "scene_ref": "simple_reflector",
        "scene_asset_manifest": {
            "asset_set_sha256": "d" * 64,
            "asset_count": 1,
            "complete": True,
        },
        "runtime_versions": {"sionna-rt": "2.2.0"},
        "material_manifest": {"wall": {"scattering_coefficient": 0.0}},
        "scene_geometry_inventory": inventory,
    }


def test_rt_trajectory_authorization_request_binds_active_scene_state() -> None:
    request = build_rt_trajectory_generation_request(**_trajectory_request_inputs())

    assert request["material_state_sha256"]
    assert request["scene_geometry"]["scene_geometry_sha256"] == "a" * 64
    assert request["config"]["solver_config"]["max_depth"] == 2
    assert request["config"]["antenna_config"]["node_orientation"]["policy"] == (
        "look_at_peer_each_step"
    )


def test_paired_frequency_request_plan_shares_scene_and_route_without_generating():
    jsonschema = pytest.importorskip("jsonschema")
    inputs = _trajectory_request_inputs()
    config = inputs.pop("config")
    inputs.pop("material_manifest")
    manifest = build_paired_frequency_rt_requests(
        pair_id="drop-a",
        base_config=config,
        carrier_frequencies_hz=(3.5e9, 28e9),
        material_manifests=(
            {"wall": {"frequency_hz": 3.5e9, "permittivity": 4.2}},
            {"wall": {"frequency_hz": 28e9, "permittivity": 3.1}},
        ),
        **inputs,
    )

    schema = json.loads(
        (
            Path(__file__).parents[2]
            / "datasets/schema/paired-frequency-rt-requests.v1.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(manifest)
    assert manifest["plan_sha256"] == canonical_hash(
        {key: value for key, value in manifest.items() if key != "plan_sha256"}
    )
    assert manifest["generation_executed"] is False
    assert [item["config"]["carrier_hz"] for item in manifest["requests"]] == [3.5e9, 28e9]
    assert [item["config"]["seed"] for item in manifest["requests"]] == [config.seed] * 2
    assert (
        manifest["requests"][0]["material_state_sha256"]
        != manifest["requests"][1]["material_state_sha256"]
    )
    assert manifest["shared_source"]["scene_geometry_sha256"] == "a" * 64


@pytest.mark.parametrize("spec_digest", [None, "A" * 64, "not-a-digest", "0" * 63])
def test_rt_trajectory_request_rejects_missing_or_noncanonical_spec_digest(
    spec_digest: object,
) -> None:
    inputs = _trajectory_request_inputs()
    inputs["spec_sha256"] = spec_digest

    with pytest.raises(ValueError, match="spec_sha256 must be a lowercase SHA-256 digest"):
        build_rt_trajectory_generation_request(**inputs)


def test_rt_trajectory_authorization_rejects_changed_material_or_geometry() -> None:
    request = build_rt_trajectory_generation_request(**_trajectory_request_inputs())
    authorization = authorize_d1_generation(request, pilot=True)
    changed_request = dict(request)
    changed_request["material_state_sha256"] = "f" * 64

    with pytest.raises(PermissionError, match="does not match"):
        require_d1_authorization(
            authorization, changed_request, expected_operation="rt_nr_trajectory"
        )


@pytest.mark.parametrize("changed", ["material", "geometry", "solver"])
def test_rt_trajectory_authorization_request_changes_with_scene_or_solver(
    changed: str,
) -> None:
    baseline = build_rt_trajectory_generation_request(**_trajectory_request_inputs())
    inputs = _trajectory_request_inputs()
    if changed == "material":
        inputs["material_manifest"] = {"wall": {"scattering_coefficient": 0.5}}
    elif changed == "geometry":
        inventory = inputs["scene_geometry_inventory"]
        inputs["scene_geometry_inventory"] = SceneTriangleInventory(
            object_names=inventory.object_names,
            triangles_m=np.concatenate(
                [
                    inventory.triangles_m,
                    np.asarray([[[1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0]]]),
                ]
            ),
            triangle_object_names=("wall", "wall"),
            scene_geometry_sha256="e" * 64,
            meters_per_unit=1.0,
        )
    else:
        config = inputs["config"]
        inputs["config"] = replace(config, max_depth=3)

    changed_request = build_rt_trajectory_generation_request(**inputs)
    assert canonical_hash(changed_request) != canonical_hash(baseline)


def test_rt_trajectory_cache_identity_binds_loaded_scene_geometry() -> None:
    inputs = _trajectory_request_inputs()
    config = inputs["config"]
    positions, _ = config.positions_and_times()
    inventory = inputs["scene_geometry_inventory"]
    rows = [
        {
            "path": "scene.xml",
            "bytes": 1,
            "sha256": hashlib.sha256(b"x").hexdigest(),
        }
    ]
    assets = {
        "scene_xml_path": "scene.xml",
        "asset_count": 1,
        "assets": rows,
        "asset_set_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "unresolved_references": [],
        "complete": True,
    }

    def identity(scene_inventory):
        return rt_run_identity(
            scene_asset_manifest=assets,
            physical_config=rt_trajectory_physical_config(config, positions, scene_inventory),
            material_config=inputs["material_manifest"],
            solver_config={"max_depth": config.max_depth},
            seed=config.seed,
            backend_selection=inputs["backend"],
            runtime_versions=inputs["runtime_versions"],
            package_source_sha256=inputs["package_source_digest"],
        )["cache_key_sha256"]

    first_key = identity(inventory)
    changed_inventory = SceneTriangleInventory(
        object_names=inventory.object_names,
        triangles_m=inventory.triangles_m,
        triangle_object_names=inventory.triangle_object_names,
        scene_geometry_sha256="e" * 64,
        meters_per_unit=inventory.meters_per_unit,
    )
    assert identity(changed_inventory) != first_key


def test_rt_trajectory_config_generates_ordered_constant_velocity_route() -> None:
    config = RTTrajectoryConfig(
        trajectory_id="route-a",
        scene="simple_reflector",
        transmitter_position_m=(1.0, 2.0, 3.0),
        receiver_start_m=(4.0, 5.0, 6.0),
        receiver_velocity_mps=(2.0, -1.0, 0.5),
        num_steps=3,
        sample_interval_s=0.25,
    )

    positions, times = config.positions_and_times()

    assert positions == ((4.0, 5.0, 6.0), (4.5, 4.75, 6.125), (5.0, 4.5, 6.25))
    assert times == (0.0, 0.25, 0.5)
    assert config.no_path_policy == "reject"


@pytest.mark.parametrize(
    "overrides",
    [
        {"trajectory_id": ""},
        {"scene": ""},
        {"num_steps": 0},
        {"sample_interval_s": 0.0},
        {"carrier_hz": float("nan")},
        {"snr_db": float("inf")},
        {"snr_db": None},
        {"tx_power_per_re_w": 1e-6},
        {"no_path_policy": "noise_only_absolute_power"},
        {"max_depth": -1},
        {"max_depth": True},
        {"samples_per_src": 0},
        {"max_num_paths_per_src": 0},
        {"max_num_paths_per_src": True},
        {"no_path_policy": "zero_is_ground_truth"},
        {"seed": True},
        {"receiver_velocity_mps": (float("nan"), 0.0, 0.0)},
    ],
)
def test_rt_trajectory_config_rejects_invalid_values(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "trajectory_id": "route-a",
        "scene": "simple_reflector",
        "transmitter_position_m": (1.0, 2.0, 3.0),
        "receiver_start_m": (4.0, 5.0, 6.0),
        "receiver_velocity_mps": (1.0, 0.0, 0.0),
    }
    values.update(overrides)

    with pytest.raises((TypeError, ValueError)):
        RTTrajectoryConfig(**values)  # type: ignore[arg-type]


def test_rt_trajectory_absolute_power_mode_requires_explicit_transmit_power() -> None:
    config = RTTrajectoryConfig(
        trajectory_id="physical-noise-route",
        scene="simple_reflector",
        transmitter_position_m=(1.0, 2.0, 3.0),
        receiver_start_m=(4.0, 5.0, 6.0),
        receiver_velocity_mps=(0.0, 0.0, 0.0),
        snr_db=None,
        tx_power_per_re_w=1e-6,
        no_path_policy="noise_only_absolute_power",
    )
    assert config.tx_power_per_re_w == 1e-6
    assert config.snr_db is None
