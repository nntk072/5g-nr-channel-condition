"""Fail-closed production RT dataset authorization tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from integrations.sionna.d1_generation_gate import (
    authorize_d1_generation,
    canonical_sha256,
    require_d1_authorization,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _approval(tmp_path: Path, request: dict, *, statuses: dict | None = None) -> Path:
    plan = tmp_path / "qualification-plan.json"
    report = tmp_path / "qualification-report.json"
    plan.write_text('{"plan":"frozen"}\n', encoding="utf-8")
    report.write_text('{"result":"passed"}\n', encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "schema": "loslab.d1_qualification_approval.v1",
                "decision": "approved",
                "approval_id": "approval-fixture-001",
                "approved_by": "independent-reviewer",
                "approved_at": "2026-10-23T12:00:00Z",
                "scope_request_sha256": canonical_sha256(request),
                "gate_statuses": statuses or {f"g{index}": "passed" for index in range(6)},
                "g7_release_scope": "internal_research",
                "g7_release_statuses": {},
                "qualification_artifacts": [
                    {
                        "role": "qualification_plan",
                        "path": plan.name,
                        "sha256": _sha256(plan),
                    },
                    {
                        "role": "qualification_report",
                        "path": report.name,
                        "sha256": _sha256(report),
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return approval


def test_production_is_denied_without_an_approval_manifest():
    with pytest.raises(PermissionError, match="trusted approval-signature keyring"):
        authorize_d1_generation({"operation": "rt_data"})


def test_explicit_pilot_authorizes_without_qualification_artifacts(tmp_path: Path):
    authorization = authorize_d1_generation(
        {
            "operation": "rt_data",
            "config": {
                "per_class": 20,
                "max_attempts": 40,
                "samples_per_src": 100,
                "max_depth": 3,
                "total_work_units": 4_000,
            },
            "destinations": ["/tmp/test-static"],
        },
        pilot=True,
    )

    assert authorization.mode == "pilot"
    assert authorization.approval_manifest_sha256 is None


def test_pilot_cannot_be_combined_with_an_approval(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot also supply"):
        authorize_d1_generation(
            {"operation": "rt_data"},
            pilot=True,
            approval_manifest=tmp_path / "approval.json",
            expected_approval_sha256="0" * 64,
        )


def test_pilot_requires_nonempty_fixed_scratch_destination(tmp_path: Path):
    request = {
        "operation": "rt_dmrs_observations",
        "sample_count": 1,
        "input_artifact_count": 0,
        "total_work_units": 1,
        "destinations": ["/var/tmp/outside"],
    }
    with pytest.raises(PermissionError, match="fixed scratch root"):
        authorize_d1_generation(request, pilot=True)
    request["destinations"] = []
    with pytest.raises(PermissionError, match="at least one scratch output"):
        authorize_d1_generation(request, pilot=True)


def test_forged_production_record_is_rejected_even_with_private_seal():
    import integrations.sionna.d1_generation_gate as gate

    request = {"operation": "rt_dmrs_observations"}
    forged = gate.D1GenerationAuthorization(
        "approved_production",
        canonical_sha256(request),
        "0" * 64,
        (),
        "rt_dmrs_observations",
        _seal=gate._AUTHORITY_SEAL,
    )
    with pytest.raises(PermissionError, match="unknown D1 generation authorization mode"):
        require_d1_authorization(forged, request)


def test_public_generation_apis_deny_direct_calls_without_gate_record(tmp_path: Path):
    from integrations.sionna.adapter import generate_raytraced, generate_raytraced_cached
    from integrations.sionna.rt_trajectory import RTTrajectoryConfig, generate_rt_nr_trajectory

    with pytest.raises(PermissionError, match="requires D1 authorization"):
        generate_raytraced()
    with pytest.raises(PermissionError, match="requires D1 authorization"):
        generate_raytraced_cached(cache_directory=tmp_path / "cache")
    config = RTTrajectoryConfig(
        trajectory_id="unauthorized-direct-call",
        scene="fixture-scene",
        transmitter_position_m=(0.0, 0.0, 1.0),
        receiver_start_m=(1.0, 0.0, 1.0),
        receiver_velocity_mps=(0.0, 0.0, 0.0),
    )
    with pytest.raises(PermissionError, match="requires D1 authorization"):
        generate_rt_nr_trajectory(config, tmp_path)


def test_umcd_publication_api_denies_direct_call_without_authorization(tmp_path: Path):
    from integrations.sionna.rt_network import write_sionna_rt_episode_bundle

    with pytest.raises(PermissionError, match="gate-issued D1 authorization"):
        write_sionna_rt_episode_bundle(
            tmp_path,
            None,
            (),
            run_identity={},
            material_config={},
            source_family="F03-RT-STATIC",
            producer="test",
        )


def test_signed_looking_approval_manifest_cannot_authorize_production(tmp_path: Path):
    request = {"operation": "rt_data", "seed": 5, "scene": "fixture"}
    approval = _approval(tmp_path, request)
    with pytest.raises(PermissionError, match="production D1 generation is disabled"):
        authorize_d1_generation(
            request, approval_manifest=approval, expected_approval_sha256=_sha256(approval)
        )


@pytest.mark.parametrize(
    "generation_request",
    [
        {
            "operation": "rt_data",
            "config": {
                "per_class": 21,
                "max_attempts": 2_000,
                "samples_per_src": 200,
                "max_depth": 2,
                "total_work_units": 4_000,
            },
            "destinations": ["/tmp/pilot"],
        },
        {
            "operation": "rt_data",
            "config": {
                "per_class": 1,
                "max_attempts": 2_001,
                "samples_per_src": 200,
                "max_depth": 2,
                "total_work_units": 4_002,
            },
            "destinations": ["/tmp/pilot"],
        },
        {
            "operation": "rt_data",
            "config": {
                "per_class": 1,
                "max_attempts": 2,
                "samples_per_src": 5_000_001,
                "max_depth": 2,
                "total_work_units": 4,
            },
            "destinations": ["/tmp/pilot"],
        },
    ],
)
def test_shared_pilot_authorization_enforces_numeric_rt_data_ceilings(
    generation_request: dict, tmp_path: Path
):
    generation_request["destinations"] = ["/tmp/pilot"]
    with pytest.raises(ValueError, match="pilot rt_data"):
        authorize_d1_generation(generation_request, pilot=True)


def test_shared_pilot_authorization_enforces_trajectory_ceilings(tmp_path: Path):
    with pytest.raises(ValueError, match="pilot rt_nr_trajectory num_steps"):
        authorize_d1_generation(
            {
                "operation": "rt_nr_trajectory",
                "config": {
                    "num_steps": 101,
                    "samples_per_src": 200,
                    "max_depth": 2,
                    "max_num_paths_per_src": 1_000,
                    "total_work_units": 20_200,
                },
                "destinations": ["/tmp/store"],
            },
            pilot=True,
        )


@pytest.mark.parametrize(
    ("generation_request", "message"),
    [
        (
            {
                "operation": "rt_data",
                "config": {
                    "per_class": 1,
                    "max_attempts": 1,
                    "samples_per_src": 1,
                    "max_depth": 1,
                    "total_work_units": True,
                },
                "destinations": ["/tmp/pilot"],
            },
            "pilot RT total_work_units",
        ),
        (
            {
                "operation": "rt_nr_trajectory",
                "config": {
                    "num_steps": 1,
                    "samples_per_src": 1,
                    "max_depth": 0,
                    "max_num_paths_per_src": 1,
                    "total_work_units": True,
                },
                "destinations": ["/tmp/pilot"],
            },
            "pilot trajectory total_work_units",
        ),
    ],
)
def test_pilot_authorization_rejects_boolean_work_count(generation_request: dict, message: str):
    with pytest.raises(ValueError, match=message):
        authorize_d1_generation(generation_request, pilot=True)


def test_public_release_status_claims_do_not_enable_production(tmp_path: Path):
    request = {"seed": 5}
    approval = _approval(tmp_path, request)
    value = json.loads(approval.read_text(encoding="utf-8"))
    value["g7_release_scope"] = "public_release"
    value["g7_release_statuses"] = {}
    approval.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")

    with pytest.raises(PermissionError, match="production D1 generation is disabled"):
        authorize_d1_generation(
            request,
            approval_manifest=approval,
            expected_approval_sha256=_sha256(approval),
        )


def test_loslab_rt_data_cli_fails_before_calling_the_solver(monkeypatch, capsys):
    import loslab.sionna2 as sionna2
    from loslab.__main__ import main

    def unexpected_generation(*args, **kwargs):
        raise AssertionError("solver must not run without authorization")

    monkeypatch.setattr(sionna2, "generate_raytraced", unexpected_generation)
    monkeypatch.setattr(sionna2, "generate_raytraced_cached", unexpected_generation)
    monkeypatch.setattr("sys.argv", ["loslab", "rt-data", "--output", "unused.npz"])

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "qualification approval manifest" in capsys.readouterr().err


def test_loslab_rt_data_cli_rejects_oversized_explicit_pilot(monkeypatch, capsys):
    from loslab.__main__ import main

    monkeypatch.setattr(
        "sys.argv",
        ["loslab", "rt-data", "--pilot", "--per-class", "21", "--output", "unused.npz"],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2
    assert "pilot RT generation is bounded" in capsys.readouterr().err


def test_rt_dmrs_library_builder_requires_matching_authorization():
    import numpy as np
    from datasets.channel import ChannelDataset
    from integrations.sionna.rt_dmrs import build_rt_dmrs_observation_dataset

    dataset = ChannelDataset(
        h=np.ones((2, 1, 16), dtype=np.complex64),
        y=np.asarray([0, 1], dtype=np.uint8),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["fixture", "fixture"]),
        source=np.asarray(["sionna_rt", "sionna_rt"]),
        metadata=({}, {}),
    )
    with pytest.raises(PermissionError, match="requires D1 generation authorization"):
        build_rt_dmrs_observation_dataset(
            [dataset], device="cpu", destination="/tmp/loslab-unauthorized-dmrs.npz"
        )


def test_standalone_rt_dmrs_builder_fails_closed_before_receiver_build(
    tmp_path: Path, monkeypatch, capsys
):
    import runpy

    import numpy as np
    from datasets.channel import ChannelDataset

    source = tmp_path / "source.npz"
    ChannelDataset(
        h=np.ones((2, 1, 16), dtype=np.complex64),
        y=np.asarray([0, 1], dtype=np.uint8),
        group=np.asarray(["a", "b"]),
        domain=np.asarray(["fixture", "fixture"]),
        source=np.asarray(["sionna_rt", "sionna_rt"]),
        metadata=({}, {}),
    ).save(source)
    namespace = runpy.run_path(
        "scripts/build_rt_dmrs_observation_dataset.py", run_name="rt_dmrs_cli_test"
    )
    namespace["build_rt_dmrs_observation_dataset"] = lambda *_args, **_kwargs: pytest.fail(
        "receiver builder must not run without authorization"
    )
    output = tmp_path / "observed.npz"
    monkeypatch.setattr("sys.argv", ["build_rt_dmrs_observation_dataset", str(output), str(source)])

    with pytest.raises(SystemExit) as error:
        namespace["main"]()

    assert error.value.code == 2
    assert "trusted approval-signature keyring" in capsys.readouterr().err
    assert not output.exists()


def test_trajectory_cli_fails_closed_without_approval(tmp_path: Path, monkeypatch, capsys):
    import runpy

    namespace = runpy.run_path(
        "scripts/generate_rt_nr_trajectory.py", run_name="trajectory_cli_test"
    )
    spec = tmp_path / "trajectory.json"
    spec.write_text(
        json.dumps(
            {
                "trajectory_id": "fixture",
                "scene": "simple_reflector",
                "transmitter_position_m": [0, 0, 3],
                "receiver_start_m": [1, 0, 1],
                "receiver_velocity_mps": [0, 0, 0],
            }
        ),
        encoding="utf-8",
    )
    namespace["generate_rt_nr_trajectory"] = lambda *args, **kwargs: pytest.fail(
        "trajectory solver must not run without authorization"
    )
    monkeypatch.setattr(
        "sys.argv", ["generate_rt_nr_trajectory", str(spec), str(tmp_path / "store")]
    )

    with pytest.raises(SystemExit) as error:
        namespace["main"]()

    assert error.value.code == 2
    assert "requires a validated qualification approval manifest" in capsys.readouterr().err
    assert not (tmp_path / "store").exists()


def test_trajectory_cli_preserves_explicit_pilot_workflow(tmp_path: Path, monkeypatch, capsys):
    import runpy

    namespace = runpy.run_path(
        "scripts/generate_rt_nr_trajectory.py", run_name="trajectory_cli_test"
    )
    spec = tmp_path / "trajectory.json"
    spec.write_text(
        json.dumps(
            {
                "trajectory_id": "fixture",
                "scene": "simple_reflector",
                "transmitter_position_m": [0, 0, 3],
                "receiver_start_m": [1, 0, 1],
                "receiver_velocity_mps": [0, 0, 0],
            }
        ),
        encoding="utf-8",
    )
    namespace["generate_rt_nr_trajectory"] = lambda *_args, **_kwargs: type(
        "Stored",
        (),
        {
            "trajectory_id": "fixture",
            "content_sha256": "abc",
            "manifest_relative_path": "fixture.json",
            "sample_ids": (),
        },
    )()
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_rt_nr_trajectory",
            str(spec),
            str(tmp_path / "store"),
            "--pilot",
        ],
    )

    namespace["main"]()

    assert json.loads(capsys.readouterr().out)["trajectory_id"] == "fixture"
