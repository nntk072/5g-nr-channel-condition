"""Regression definitions for retained, exact-byte-bound G3 joint reports."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from evaluation.g3_joint_acceptance import build_g3_joint_acceptance_family_v4
from evaluation.g3_qualification_contract import (
    build_g3_qualification_run_contract_v3,
    build_g3_qualification_run_contract_v4,
)

from evaluation import g3_joint_evaluation as joint
from evaluation import g3_joint_evidence_artifact as artifact
from evaluation import g3_joint_evidence_run_manifest as run_manifest
from evaluation import g3_path_equivalence_assurance as path_equivalence_assurance


def _payload() -> dict[str, Any]:
    family = build_g3_joint_acceptance_family_v4(
        family_id="artifact-family",
        material_expected_strata=[("concrete", "V", "20deg")],
        complex_csi_expected_strata=[("scene-a", "LOS")],
        propagation_expected_strata=[("scene-a", "LOS", "total")],
        receiver_impact_expected_strata=[("scene-a", "LOS")],
        path_equivalence_expected_scenarios=[("scene-a", "LOS", "episode-a")],
    )
    return {
        "schema": "loslab.g3_joint_evaluator_inputs.v1",
        "joint_acceptance_family": family,
        "material_calibration": {
            "payload": {"joint_acceptance_family": family, "scorer_input": "opaque"}
        },
        "complex_csi": {
            "records": [
                {
                    "family_id": "scene-a",
                    "condition": "LOS",
                    "episode_id": "episode-a",
                    "candidate_cfr": [{"$complex": [1.0, 0.0]}],
                    "challenger_cfr": [{"$complex": [1.0, 0.0]}],
                }
            ],
            "expected_strata": [["scene-a", "LOS"]],
        },
        "propagation_metrics": {
            "records": [
                {
                    "family_id": "scene-a",
                    "condition": "LOS",
                    "episode_id": "episode-a",
                }
            ],
            "expected_strata": [["scene-a", "LOS", "total"]],
            "effective_bandwidth_hz": 20e6,
        },
        "receiver_impact": {
            "records": [{"block_id": "block-a"}],
            "expected_strata": [["scene-a", "LOS"]],
            "candidate_config_id": "candidate-a",
            "challenger_config_id": "challenger-a",
            "snr_grid_db": [0.0, 5.0],
            "comparison_snr_db": 0.0,
        },
        "path_equivalence": {
            "records": [
                {
                    "family_id": "scene-a",
                    "condition": "LOS",
                    "episode_id": "episode-a",
                }
            ],
            "expected_scenarios": [["scene-a", "LOS", "episode-a"]],
        },
    }


def _payload_v2() -> dict[str, Any]:
    payload = _payload()
    payload["schema"] = "loslab.g3_joint_evaluator_inputs.v2"

    def config_entry(config_id: str, configuration: dict[str, Any]) -> dict[str, Any]:
        encoded = json.dumps(
            configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return {
            "config_id": config_id,
            "configuration": configuration,
            "configuration_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    waveform_configuration = {"mu": 1, "carrier_hz": 3.5e9}
    waveform_encoded = json.dumps(
        waveform_configuration, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    payload["receiver_execution"] = {
        "candidate": config_entry("candidate-a", {"receiver": "ls"}),
        "challenger": config_entry("challenger-a", {"receiver": "dft"}),
        "waveform": {
            "waveform_id": "pusch-fixture",
            "configuration": waveform_configuration,
            "configuration_sha256": hashlib.sha256(waveform_encoded).hexdigest(),
        },
        "link_budget_noise": {
            "tx_power_dbm": 23.0,
            "tx_gain_dbi": 0.0,
            "rx_gain_dbi": 0.0,
            "noise_figure_db": 7.0,
            "noise_temperature_k": 290.0,
            "noise_bandwidth_hz": 20e6,
        },
    }
    payload["receiver_impact"]["records"][0]["receiver_execution_context_sha256"] = (
        hashlib.sha256(
            json.dumps(
                payload["receiver_execution"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )
    return payload


def _payload_v3() -> dict[str, Any]:
    payload = _payload_v2()
    payload["schema"] = "loslab.g3_joint_evaluator_inputs.v3"
    payload["receiver_impact"]["records"][0].update(
        {"family_id": "scene-a", "condition": "LOS", "episode_id": "episode-a"}
    )
    return payload


def _payload_v4() -> dict[str, Any]:
    payload = _payload_v3()
    payload["schema"] = "loslab.g3_joint_evaluator_inputs.v4"
    candidate = payload["receiver_execution"]["candidate"]
    record = payload["receiver_impact"]["records"][0]
    record["configuration_id"] = candidate["config_id"]
    record["receiver_configuration_sha256"] = candidate["configuration_sha256"]
    path_record = payload["path_equivalence"]["records"][0]
    path_record["comparison"] = {
        "coverage_criterion": 0.95,
        "configuration": {
            "candidate_configuration_id": "candidate-a",
            "challenger_configuration_id": "challenger-a",
            "delay_tolerance_s": 1e-9,
        },
    }
    return payload


def _payload_v5() -> dict[str, Any]:
    payload = _payload_v4()
    payload["schema"] = "loslab.g3_joint_evaluator_inputs.v5"
    payload["path_equivalence"]["records"] = [
        {
            "family_id": "scene-a",
            "condition": "LOS",
            "episode_id": "episode-a",
            "candidate_paths": [
                {
                    "valid": True,
                    "path_kind": "specular",
                    "path_index": 0,
                    "interaction_type_masks": [1],
                    "object_ids": [7],
                    "primitive_ids": [2],
                    "interaction_vertices_m": [[1.0, 2.0, 3.0]],
                    "absolute_delay_s": 1e-7,
                    "theta_t_rad": 1.5707963267948966,
                    "phi_t_rad": 0.1,
                    "theta_r_rad": 1.5707963267948966,
                    "phi_r_rad": 0.2,
                    "coefficient_real": 0.05,
                    "coefficient_imag": 0.0,
                    "cfr_contribution_real": [0.05, 0.05],
                    "cfr_contribution_imag": [0.0, 0.0],
                }
            ],
            "challenger_paths": [
                {
                    "valid": True,
                    "path_kind": "specular",
                    "path_index": 0,
                    "interaction_type_masks": [1],
                    "object_ids": [7],
                    "primitive_ids": [2],
                    "interaction_vertices_m": [[1.0, 2.0, 3.0]],
                    "absolute_delay_s": 1e-7,
                    "theta_t_rad": 1.5707963267948966,
                    "phi_t_rad": 0.1,
                    "theta_r_rad": 1.5707963267948966,
                    "phi_r_rad": 0.2,
                    "coefficient_real": 0.05,
                    "coefficient_imag": 0.0,
                    "cfr_contribution_real": [0.05, 0.05],
                    "cfr_contribution_imag": [0.0, 0.0],
                }
            ],
            "candidate_full_cfr_real": [0.05, 0.05],
            "candidate_full_cfr_imag": [0.0, 0.0],
            "challenger_full_cfr_real": [0.05, 0.05],
            "challenger_full_cfr_imag": [0.0, 0.0],
        }
    ]
    payload["path_equivalence"]["comparison_configuration"] = {
        "candidate_configuration_id": "candidate-a",
        "challenger_configuration_id": "challenger-a",
        "vertex_tolerance_m": 0.01,
        "delay_tolerance_s": 1e-9,
        "direction_tolerance_rad": 0.01,
        "h_floor": 1e-12,
        "transmit_power_per_re_w": 1.0,
        "noise_power_per_re_w": 1e-6,
    }
    return payload


def _payload_v6() -> dict[str, Any]:
    payload = _payload_v5()
    payload["schema"] = "loslab.g3_joint_evaluator_inputs.v6"
    record = payload["path_equivalence"]["records"][0]
    for side in ("candidate", "challenger"):
        diffuse = dict(record[f"{side}_paths"][0])
        diffuse.update(
            {
                "path_kind": "diffuse_stochastic",
                "absolute_delay_s": 2e-7,
                "coefficient_real": 0.025,
                "cfr_contribution_real": [0.025, 0.025],
            }
        )
        record[f"{side}_paths"].append(diffuse)
        record[f"{side}_full_cfr_real"] = [0.075, 0.075]
    payload["path_equivalence"]["solver_budget_provenance"] = {
        "max_rays_per_link": 1000,
        "max_interactions": 8,
        "max_episodes": 20,
        "max_wall_seconds": 60.0,
        "overflow_policy": "record_censored_episode",
    }
    return payload


def _install_scorers(monkeypatch: pytest.MonkeyPatch) -> None:
    def material(payload: dict[str, Any]) -> dict[str, str]:
        del payload
        return {"status": "MATERIAL_NUMERICAL_ONLY"}

    def complex_csi(**kwargs: Any) -> dict[str, str]:
        del kwargs
        return {"status": "COMPLEX_CSI_NUMERICAL_ONLY"}

    def propagation(**kwargs: Any) -> dict[str, str]:
        del kwargs
        return {"status": "PROPAGATION_NUMERICAL_ONLY"}

    def receiver(**kwargs: Any) -> dict[str, str]:
        del kwargs
        return {"status": "RECEIVER_NUMERICAL_ONLY"}

    def path_equivalence(**kwargs: Any) -> dict[str, str]:
        del kwargs
        return {"status": "PATH_NUMERICAL_ONLY"}

    monkeypatch.setattr(joint, "evaluate_g3_material_calibration", material)
    monkeypatch.setattr(joint, "evaluate_complex_csi", complex_csi)
    monkeypatch.setattr(joint, "evaluate_propagation_metrics", propagation)
    monkeypatch.setattr(joint, "evaluate_receiver_impact", receiver)
    monkeypatch.setattr(joint, "evaluate_path_equivalence_roster", path_equivalence)


def _write_canonical(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _legacy_v1_report(report: dict[str, Any]) -> dict[str, Any]:
    bundle = dict(report["bundle"])
    bundle.pop("numerical_criterion_rollup")
    bundle["schema"] = "loslab.g3_joint_evaluator_bundle.v1"
    return {
        "schema": "loslab.g3_joint_evaluator_artifact.v1",
        "status": report["status"],
        "qualification_status": report["qualification_status"],
        "input_bytes_sha256": report["input_bytes_sha256"],
        "bundle_sha256": hashlib.sha256(artifact._canonical_json_bytes(bundle)).hexdigest(),
        "bundle": bundle,
    }


def test_joint_artifact_binds_exact_inputs_and_recomputes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "inputs.json"
    saved = tmp_path / "bundle.json"
    _write_canonical(source, _payload())

    report = artifact.evaluate_g3_joint_evidence_file(source)
    artifact_schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evaluator-artifact.v2.schema.json"
    )
    input_schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evaluator-input.v1.schema.json"
    )
    input_schema_v4_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evaluator-input.v4.schema.json"
    )
    input_schema_v5_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evaluator-input.v5.schema.json"
    )
    input_schema_v6_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evaluator-input.v6.schema.json"
    )
    jsonschema.Draft202012Validator(
        json.loads(artifact_schema_path.read_text(encoding="utf-8"))
    ).validate(report)
    jsonschema.Draft202012Validator(
        json.loads(input_schema_path.read_text(encoding="utf-8"))
    ).validate(_payload())
    jsonschema.Draft202012Validator(
        json.loads(input_schema_v4_path.read_text(encoding="utf-8"))
    ).validate(_payload_v4())
    jsonschema.Draft202012Validator(
        json.loads(input_schema_v5_path.read_text(encoding="utf-8"))
    ).validate(_payload_v5())
    jsonschema.Draft202012Validator(
        json.loads(input_schema_v6_path.read_text(encoding="utf-8"))
    ).validate(_payload_v6())
    artifact._write_new_artifact(saved, report)
    verified = artifact.verify_g3_joint_evidence_file(source, saved)

    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert verified["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert verified["input_bytes_sha256"] == report["input_bytes_sha256"]


def test_v4_schema_requires_row_configuration_identity() -> None:
    payload = _payload_v4()
    payload["receiver_impact"]["records"][0].pop("configuration_id")
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evaluator-input.v4.schema.json"
    )
    validator = jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )

    assert list(validator.iter_errors(payload))


def test_v4_schema_requires_path_comparison_binding() -> None:
    payload = _payload_v4()
    payload["path_equivalence"]["records"][0]["comparison"]["configuration"].pop(
        "delay_tolerance_s"
    )
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evaluator-input.v4.schema.json"
    )
    validator = jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )

    assert list(validator.iter_errors(payload))


def test_v5_schema_requires_raw_paths_and_frozen_configuration() -> None:
    payload = _payload_v5()
    payload["path_equivalence"]["comparison_configuration"]["h_floor"] = 0.0
    schema_path = (
        Path(__file__).resolve().parents[2]
        / "datasets/schema/g3-joint-evaluator-input.v5.schema.json"
    )
    validator = jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text(encoding="utf-8"))
    )

    assert list(validator.iter_errors(payload))


def test_contract_v4_binds_v6_diffuse_solver_budget_provenance() -> None:
    payload = _payload_v6()
    contract = _qualification_contract(payload, "f" * 64, raw_paths=True)
    payload["path_equivalence"]["solver_budget_provenance"] = dict(contract["solver_budget"])
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")

    run_manifest._verify_qualification_contract_bindings(contract, encoded)

    payload["path_equivalence"]["solver_budget_provenance"]["max_interactions"] += 1
    drifted = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="solver budget differs"):
        run_manifest._verify_qualification_contract_bindings(contract, drifted)


def test_contract_v4_binds_propagation_effective_bandwidth() -> None:
    payload = _payload_v6()
    contract = _qualification_contract(payload, "f" * 64, raw_paths=True)
    payload["path_equivalence"]["solver_budget_provenance"] = dict(contract["solver_budget"])
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    run_manifest._verify_qualification_contract_bindings(contract, encoded)

    payload["propagation_metrics"]["effective_bandwidth_hz"] = 40e6
    drifted = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="effective bandwidth differs"):
        run_manifest._verify_qualification_contract_bindings(contract, drifted)

    payload["propagation_metrics"]["effective_bandwidth_hz"] = 20_000_000
    typed_alias = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    with pytest.raises(ValueError, match="effective bandwidth differs"):
        run_manifest._verify_qualification_contract_bindings(contract, typed_alias)


def test_contract_v4_without_bandwidth_remains_diagnostic_only() -> None:
    payload = _payload_v6()
    contract = _qualification_contract(payload, "f" * 64, raw_paths=True)
    contract["task_scope"].pop("effective_bandwidth_hz")
    unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
    contract["contract_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    payload["path_equivalence"]["solver_budget_provenance"] = dict(contract["solver_budget"])
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")

    run_manifest._verify_qualification_contract_bindings(contract, encoded)
    with pytest.raises(ValueError, match="requires frozen propagation effective bandwidth"):
        run_manifest._verify_qualification_contract_bindings(
            contract, encoded, require_powered_design=True
        )


def test_joint_artifact_rejects_changed_input_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "inputs.json"
    saved = tmp_path / "bundle.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))

    changed = deepcopy(payload)
    changed["receiver_impact"]["comparison_snr_db"] = 1.0
    _write_canonical(source, changed)
    with pytest.raises(ValueError, match="exact-byte-bound recomputation"):
        artifact.verify_g3_joint_evidence_file(source, saved)


@pytest.mark.parametrize(
    "raw",
    [
        b'{"a":1,"a":2}\n',
        b'{"a":NaN}\n',
        b'{ "a": 1 }\n',
    ],
)
def test_joint_input_rejects_ambiguous_or_noncanonical_json(tmp_path: Path, raw: bytes) -> None:
    source = tmp_path / "inputs.json"
    source.write_bytes(raw)
    with pytest.raises(ValueError):
        artifact.evaluate_g3_joint_evidence_file(source)


def test_complex_json_encoding_rejects_nonfinite_components() -> None:
    with pytest.raises(ValueError, match="finite"):
        artifact._decode_json_value({"$complex": [float("inf"), 0.0]})


def test_artifact_writer_refuses_to_overwrite(tmp_path: Path) -> None:
    saved = tmp_path / "bundle.json"
    saved.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        artifact._write_new_artifact(saved, {"schema": artifact.ARTIFACT_SCHEMA})
    assert saved.read_text(encoding="utf-8") == "preserve"


def _write_run_manifest(
    path: Path,
    source: Path,
    saved: Path,
    *,
    wrong_input_hash: bool = False,
    qualification_contract: dict[str, Any] | None = None,
) -> None:
    run_id = (
        qualification_contract["run_id"]
        if qualification_contract is not None
        else "joint-run-001"
    )
    manifest = run_manifest.build_g3_joint_evidence_run_manifest(
        run_id, source, saved, qualification_contract=qualification_contract
    )
    if wrong_input_hash:
        manifest["input"]["sha256"] = "0" * 64
    _write_canonical(path, manifest)


def _sign_run_manifest(manifest_path: Path, directory: Path) -> tuple[Path, Path, str]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    signer_id = "g3-run-manifest-reviewer"
    private_key = directory / "g3-run-manifest-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = directory / "allowed_signers"
    allowed_signers.write_text(
        f'{signer_id} namespaces="{run_manifest.G3_RUN_MANIFEST_SIGNATURE_NAMESPACE}" '
        f"{public_key}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            run_manifest.G3_RUN_MANIFEST_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(f"{manifest_path}.sig"), allowed_signers, signer_id


def _qualification_contract(
    payload: dict[str, Any],
    source_bundle_sha256: str,
    *,
    candidate_config_id: str = "candidate-a",
    raw_paths: bool = False,
) -> dict[str, Any]:
    family = payload["joint_acceptance_family"]
    contract_spec: dict[str, Any] = {
            "run_id": "g3-run-contract-001",
            "joint_acceptance_family": family,
        "task_scope": {
                "task_id": "complex-csi-and-receiver-impact",
                "scenario_ids": ["scene-a"],
                "channel_conditions": ["LOS"],
                "carrier_hz": 3.5e9,
            **({"effective_bandwidth_hz": 20e6} if raw_paths else {}),
            },
            "candidate": {
                "config_id": candidate_config_id,
                "configuration": {"receiver": "ls"},
            },
            "challenger": {
                "config_id": "challenger-a",
                "configuration": {"receiver": "dft"},
            },
            "waveform": {
                "waveform_id": "pusch-fixture",
                "configuration": {"mu": 1, "carrier_hz": 3.5e9},
            },
            "link_budget_noise": {
                "tx_power_dbm": 23.0,
                "tx_gain_dbi": 0.0,
                "rx_gain_dbi": 0.0,
                "noise_figure_db": 7.0,
                "noise_temperature_k": 290.0,
                "noise_bandwidth_hz": 20e6,
            },
            "path_matching": {
                "delay_tolerance_s": 1e-9,
                "azimuth_tolerance_deg": 2.0,
                "elevation_tolerance_deg": 2.0,
                "coefficient_relative_tolerance": 0.05,
                "minimum_bidirectional_coverage": 0.95,
            },
            "sampling": {
                "episode_ids": ["episode-a"],
                "path_equivalence_scenarios": [["scene-a", "LOS", "episode-a"]],
                "unit_by_branch": {
                    "material_calibration": "independent_acquisition_unit",
                    "complex_csi": "independent_physical_episode",
                    "propagation_metrics": "independent_physical_episode",
                    "receiver_impact": "independent_transport_block_cluster",
                    "path_equivalence": "physical_episode",
                },
                "planned_units_by_branch": {
                    "material_calibration": 30,
                    "complex_csi": 60,
                    "propagation_metrics": 60,
                    "receiver_impact": 100,
                    "path_equivalence": 1,
                },
                "planned_units_by_stratum": {
                    "material_calibration": [
                        {"stratum": ["concrete", "V", "20deg"], "planned_units": 30}
                    ],
                    "complex_csi": [
                        {"stratum": ["scene-a", "LOS"], "planned_units": 60}
                    ],
                    "propagation_metrics": [
                        {"stratum": ["scene-a", "LOS", "total"], "planned_units": 60}
                    ],
                    "receiver_impact": [
                        {"stratum": ["scene-a", "LOS"], "planned_units": 100}
                    ],
                    "path_equivalence": [
                        {"stratum": ["scene-a", "LOS", "episode-a"], "planned_units": 1}
                    ],
                },
                "bootstrap_seed": 41,
                "bootstrap_draws": 10000,
            },
            "solver_budget": {
                "max_rays_per_link": 100000,
                "max_interactions": 5,
                "max_episodes": 200,
                "max_wall_seconds": 3600,
                "overflow_policy": "record_censored_episode",
            },
            "source_identity": {
                "git_revision": "c" * 40,
                "working_tree_dirty": True,
                "source_bundle_sha256": source_bundle_sha256,
            },
        }
    contract_spec["sample_size_design"] = _synthetic_complete_sample_size_design(family)
    if raw_paths:
        contract_spec["path_matching"].update(
            {
                "vertex_tolerance_m": 0.01,
                "direction_tolerance_rad": 0.01,
                "h_floor": 1e-12,
                "transmit_power_per_re_w": 1.0,
                "noise_power_per_re_w": 1e-6,
            }
        )
        return build_g3_qualification_run_contract_v4(contract_spec)
    return build_g3_qualification_run_contract_v3(contract_spec)


def _synthetic_complete_sample_size_design(family: dict[str, Any]) -> dict[str, Any]:
    branch_rosters = {
        "material_calibration": ("material_expected_strata", 20, "independent_acquisition_unit"),
        "complex_csi": ("complex_csi_expected_strata", 2, "independent_physical_episode"),
        "propagation_metrics": ("propagation_expected_strata", 2, "independent_physical_episode"),
        "receiver_impact": ("receiver_impact_expected_strata", 20, "independent_transport_block_cluster"),
        "path_equivalence": ("path_equivalence_expected_scenarios", 1, "physical_episode"),
    }
    branches = {
        branch: {
            "sampling_unit": unit,
            "required_units_by_stratum": [
                {"stratum": list(stratum), "required_units": count}
                for stratum in family[roster_field]
            ],
            "count_available": True,
            "component_status": "SCENARIO_CONDITIONAL_COUNT_AVAILABLE",
        }
        for branch, (roster_field, count, unit) in branch_rosters.items()
    }
    schemas = {
        "joint": "loslab.g3_joint_sample_size_planning.v1",
        "material_assurance": "loslab.g3_bootstrap_assurance_planning.v1",
        "complex_csi_assurance": "loslab.g3_complex_csi_bootstrap_assurance_planning.v1",
        "propagation_assurance": "loslab.g3_propagation_bootstrap_assurance_planning.v1",
        "receiver_assurance": "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1",
    }
    design: dict[str, Any] = {
        "schema": "loslab.g3_joint_sample_size_design_composition.v1",
        "status": "SCENARIO_CONDITIONAL_DESIGN_NOT_QUALIFICATION",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "joint_acceptance_family": {
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
        },
        "component_plans": {
            name: {"schema": schema, "sha256": "a" * 64}
            for name, schema in schemas.items()
        },
        "sampling_units_are_distinct": True,
        "single_combined_sample_size_is_reported": False,
        "branch_designs": branches,
        "all_branch_counts_available": True,
        "component_designs_claim_complete_powered_design": True,
        "branch_assurance_lower_bounds": {
            "material_calibration": 0.99,
            "complex_csi": 0.99,
            "propagation_metrics": 0.99,
            "receiver_impact": 0.99,
        },
        "joint_g3_assurance_lower_bound_union_bound": 0.96,
        "target_joint_g3_assurance": 0.8,
        "material_angular_maximum_assumption": {
            "supplied": True,
            "compatible_with_material_design": True,
            "maximum_units_per_stratum": 100,
        },
        "complete_precision_powered_g3_design": True,
        "complete_g3_sample_size": None,
        "blockers": [],
        "limitations": ["Synthetic test design; no scientific assumptions are verified."],
    }
    design["design_sha256"] = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return design


def test_run_manifest_hashes_and_replays_exact_inventory_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved)

    report = run_manifest.verify_g3_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["branch_count"] == 5
    assert report["evaluator_source_bundle_sha256"]
    assert {"pyproject.toml", "uv.lock"}.issubset(
        set(run_manifest.EVALUATOR_SOURCE_PATHS)
    )
    assert {
        run_manifest.INPUT_SCHEMA_V4_RELATIVE_PATH,
        run_manifest.INPUT_SCHEMA_V5_RELATIVE_PATH,
        run_manifest.INPUT_SCHEMA_V6_RELATIVE_PATH,
        run_manifest.PATH_DIAGNOSTIC_SCHEMA_V2_RELATIVE_PATH,
        run_manifest.QUALIFICATION_CONTRACT_SCHEMA_V4_RELATIVE_PATH,
        "evaluation/g3_path_equivalence.py",
        "evaluation/g3_propagation_assurance.py",
        "evaluation/g3_receiver_assurance.py",
    }.issubset(set(run_manifest.EVALUATOR_SOURCE_PATHS))
    assert run_manifest.INPUT_SCHEMA_V4_RELATIVE_PATH not in set(
        run_manifest.V1_EVALUATOR_SOURCE_PATHS
    )
    assert report["qualification_contract_bound"] is False


def test_strict_signed_reviewer_admission_rejects_unsigned_run_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved)

    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        manifest_path, require_signed_reviewer_admission=True
    )

    assert report["status"] == "INCONSISTENT"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["signed_reviewer_admission_status"] == "required"
    assert any(
        "strict signed reviewer admission requires" in error for error in report["errors"]
    )


def test_run_manifest_v1_keeps_its_frozen_source_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    _write_canonical(
        saved, _legacy_v1_report(artifact.evaluate_g3_joint_evidence_file(source))
    )
    manifest = run_manifest.build_g3_joint_evidence_run_manifest(
        "joint-run-001", source, saved, package_root=tmp_path
    )

    source_root = Path(__file__).resolve().parents[2]
    legacy_sources = []
    for relative_path in run_manifest.V1_EVALUATOR_SOURCE_PATHS:
        source_bytes = (source_root / relative_path).read_bytes()
        legacy_sources.append(
            {
                "path": relative_path,
                "size_bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            }
        )
    manifest.pop("runtime_identity")
    manifest["schema"] = "loslab.g3_joint_evidence_run_manifest.v1"
    manifest["evaluator_sources"] = legacy_sources
    manifest["evaluator_source_bundle_sha256"] = run_manifest._source_bundle_sha256(
        legacy_sources
    )
    manifest_path = tmp_path / "legacy-run-manifest.json"
    _write_canonical(manifest_path, manifest)

    report = run_manifest.verify_g3_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"


def test_run_manifest_v2_remains_readable_after_runtime_identity_v3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    _write_canonical(
        saved, _legacy_v1_report(artifact.evaluate_g3_joint_evidence_file(source))
    )
    manifest = run_manifest.build_g3_joint_evidence_run_manifest(
        "joint-run-001", source, saved, package_root=tmp_path
    )
    manifest.pop("runtime_identity")
    manifest["schema"] = "loslab.g3_joint_evidence_run_manifest.v2"
    manifest["evaluator_sources"] = [
        item
        for item in manifest["evaluator_sources"]
        if item["path"]
        not in {
            run_manifest.SCHEMA_V3_RELATIVE_PATH,
            run_manifest.ARTIFACT_SCHEMA_RELATIVE_PATH,
            "artifacts/provenance.py",
        }
    ]
    manifest["evaluator_source_bundle_sha256"] = run_manifest._source_bundle_sha256(
        manifest["evaluator_sources"]
    )
    manifest_path = tmp_path / "legacy-v2-run-manifest.json"
    _write_canonical(manifest_path, manifest)

    report = run_manifest.verify_g3_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["runtime_identity_status"] == "not_recorded"


def test_run_manifest_source_roster_sizes_match_versioned_schemas() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    schema_rosters = (
        (
            run_manifest.SCHEMA_RELATIVE_PATH,
            run_manifest.V2_EVALUATOR_SOURCE_PATHS,
        ),
        (
            run_manifest.SCHEMA_V3_RELATIVE_PATH,
            run_manifest.EVALUATOR_SOURCE_PATHS,
        ),
    )

    for relative_schema, source_paths in schema_rosters:
        schema = json.loads((repository_root / relative_schema).read_text(encoding="utf-8"))
        source_count = schema["properties"]["evaluator_sources"]
        assert source_count["minItems"] == len(source_paths)
        assert source_count["maxItems"] == len(source_paths)


def test_run_manifest_builder_rejects_malformed_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))

    with pytest.raises(ValueError, match="violates v3 schema"):
        run_manifest.build_g3_joint_evidence_run_manifest(
            "joint-run-001",
            source,
            saved,
            package_root=tmp_path,
            runtime_identity={"schema": "invalid"},
        )


def test_run_manifest_verifies_detached_signature_over_exact_manifest_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    signature_path, allowed_signers, signer_id = _sign_run_manifest(manifest_path, tmp_path)

    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        signer_id=signer_id,
    )

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["manifest_signature_status"] == "verified"
    assert report["manifest_signer_id"] == signer_id
    assert report["trust_file_custody"] == "not_established_by_this_verifier"


def test_run_manifest_cli_publishes_schema_bound_report_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    verification_path = tmp_path / "run-verification.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)

    exit_code = run_manifest.main(
        [
            str(manifest_path),
            "--package-root",
            str(tmp_path),
            "--report-output",
            str(verification_path),
        ]
    )

    report = json.loads(verification_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["schema"] == "loslab.g3_joint_evidence_run_verification.v1"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert len(report["report_schema_sha256"]) == 64
    assert run_manifest.main(
        [
            str(manifest_path),
            "--package-root",
            str(tmp_path),
            "--report-output",
            str(verification_path),
        ]
    ) == 2


def test_run_manifest_signature_resolves_external_signer_role_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    policy_path = tmp_path / "signer-policy.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    signature_path, allowed_signers, signer_id = _sign_run_manifest(manifest_path, tmp_path)
    _write_canonical(
        policy_path,
        {
            "schema": run_manifest.SIGNER_POLICY_SCHEMA,
            "policy_id": "test-role-policy",
            "signers": [
                {
                    "principal": signer_id,
                    "organization": "independent-lab",
                    "roles": ["independent_reviewer"],
                    "allowed_purposes": [run_manifest.SIGNER_POLICY_PURPOSE],
                }
            ],
        },
    )

    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=signer_id,
    )

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["signer_authorization_status"] == "authorized_for_run_manifest_verification"
    assert report["manifest_signer_roles"] == "independent_reviewer"
    assert report["manifest_signer_organization"] == "independent-lab"
    assert report["trusted_signer_policy_id"] == "test-role-policy"


@pytest.mark.parametrize("criterion_driven_contract", (False, True))
def test_strict_run_manifest_requires_criterion_driven_contract_v4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    criterion_driven_contract: bool,
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    policy_path = tmp_path / "signer-policy.json"
    payload = _payload_v6() if criterion_driven_contract else _payload_v4()
    if criterion_driven_contract:
        payload["path_equivalence"]["solver_budget_provenance"] = {
            "max_rays_per_link": 100000,
            "max_interactions": 5,
            "max_episodes": 200,
            "max_wall_seconds": 3600.0,
            "overflow_policy": "record_censored_episode",
        }
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        raw_paths=criterion_driven_contract,
    )
    _write_run_manifest(manifest_path, source, saved, qualification_contract=contract)
    manifest_signature, manifest_signers, manifest_signer_id = _sign_run_manifest(
        manifest_path, tmp_path
    )
    policy_value = {
        "schema": run_manifest.SIGNER_POLICY_SCHEMA,
        "policy_id": "custodian-signed-policy",
        "signers": [
            {
                "principal": manifest_signer_id,
                "organization": "independent-lab",
                "roles": ["independent_reviewer"],
                "allowed_purposes": [run_manifest.SIGNER_POLICY_PURPOSE],
            }
        ],
    }
    _write_canonical(policy_path, policy_value)

    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    policy_signer_id = "g3-policy-custodian"
    private_key = tmp_path / "g3-policy-custodian-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    policy_signers = tmp_path / "policy-allowed-signers"
    policy_signers.write_text(
        f'{policy_signer_id} namespaces="{run_manifest.G3_SIGNER_POLICY_SIGNATURE_NAMESPACE}" '
        f"{public_key}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            run_manifest.G3_SIGNER_POLICY_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(policy_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    policy_signature = Path(f"{policy_path}.sig")

    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        manifest_path,
        signature_path=manifest_signature,
        trusted_allowed_signers_path=manifest_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=manifest_signer_id,
        trusted_signer_policy_signature_path=policy_signature,
        trusted_policy_allowed_signers_path=policy_signers,
        policy_signer_id=policy_signer_id,
        require_signed_reviewer_admission=True,
    )

    if criterion_driven_contract:
        assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
        assert report["qualification_status"] == "NOT_QUALIFIED"
        assert report["signer_policy_signature_status"] == "verified"
        assert report["signer_policy_signer_id"] == policy_signer_id
        assert report["signer_policy_custody"] == "not_established_by_this_verifier"
        assert report["signed_reviewer_admission_status"] == (
            "signature_role_policy_and_contract_preflight_passed"
        )
    else:
        assert report["status"] == "INCONSISTENT"
        assert any(
            "requires qualification contract v4 with raw path criteria and a complete powered design"
            in error
            for error in report["errors"]
        )


@pytest.mark.parametrize(
    ("policy_identity", "message"),
    (
        (
            {"signer_id": "reviewer", "allowed_signers_sha256": "b" * 64},
            "custodian must differ",
        ),
        (
            {"signer_id": "custodian", "allowed_signers_sha256": "a" * 64},
            "keyring must differ",
        ),
    ),
)
def test_signer_policy_requires_independent_custodian_and_keyring(
    policy_identity: dict[str, str], message: str
) -> None:
    manifest_identity = {
        "signer_id": "reviewer",
        "allowed_signers_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match=message):
        run_manifest._require_independent_signer_policy_custody(
            manifest_identity, policy_identity
        )


def test_run_manifest_rejects_tampered_signed_signer_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    policy_path = tmp_path / "signer-policy.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    manifest_signature, manifest_signers, manifest_signer_id = _sign_run_manifest(
        manifest_path, tmp_path
    )
    _write_canonical(
        policy_path,
        {
            "schema": run_manifest.SIGNER_POLICY_SCHEMA,
            "policy_id": "signed-policy",
            "signers": [
                {
                    "principal": manifest_signer_id,
                    "organization": "independent-lab",
                    "roles": ["independent_reviewer"],
                    "allowed_purposes": [run_manifest.SIGNER_POLICY_PURPOSE],
                }
            ],
        },
    )
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    policy_signer_id = "g3-policy-custodian"
    private_key = tmp_path / "g3-policy-custodian-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    policy_signers = tmp_path / "policy-allowed-signers"
    policy_signers.write_text(
        f'{policy_signer_id} namespaces="{run_manifest.G3_SIGNER_POLICY_SIGNATURE_NAMESPACE}" '
        f"{public_key}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            run_manifest.G3_SIGNER_POLICY_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(policy_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    policy_signature = Path(f"{policy_path}.sig")
    _write_canonical(policy_path, {"schema": run_manifest.SIGNER_POLICY_SCHEMA})

    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        manifest_path,
        signature_path=manifest_signature,
        trusted_allowed_signers_path=manifest_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=manifest_signer_id,
        trusted_signer_policy_signature_path=policy_signature,
        trusted_policy_allowed_signers_path=policy_signers,
        policy_signer_id=policy_signer_id,
    )

    assert report["status"] == "INCONSISTENT"
    assert any("signature is invalid or signer is not trusted" in error for error in report["errors"])


def test_run_manifest_requires_all_signer_policy_signature_inputs_together(
    tmp_path: Path,
) -> None:
    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        tmp_path / "missing-manifest.json",
        trusted_signer_policy_signature_path=tmp_path / "policy.json.sig",
    )

    assert report["status"] == "INCONSISTENT"
    assert any("must be supplied together" in error for error in report["errors"])


def test_run_manifest_rejects_role_policy_without_run_manifest_purpose(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    policy_path = tmp_path / "signer-policy.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    signature_path, allowed_signers, signer_id = _sign_run_manifest(manifest_path, tmp_path)
    _write_canonical(
        policy_path,
        {
            "schema": run_manifest.SIGNER_POLICY_SCHEMA,
            "policy_id": "wrong-purpose-policy",
            "signers": [
                {
                    "principal": signer_id,
                    "organization": "independent-lab",
                    "roles": ["independent_reviewer"],
                    "allowed_purposes": ["other-purpose"],
                }
            ],
        },
    )

    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=signer_id,
    )

    assert report["status"] == "INCONSISTENT"
    assert any("unsupported run-manifest purpose" in error for error in report["errors"])


def test_run_manifest_signature_rejects_changed_manifest_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    _write_run_manifest(manifest_path, source, saved)
    signature_path, allowed_signers, signer_id = _sign_run_manifest(manifest_path, tmp_path)
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        manifest_path,
        signature_path=signature_path,
        trusted_allowed_signers_path=allowed_signers,
        signer_id=signer_id,
    )

    assert report["status"] == "INCONSISTENT"
    assert any("signature is invalid or signer is not trusted" in error for error in report["errors"])


def test_run_manifest_signature_options_must_be_supplied_together(tmp_path: Path) -> None:
    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        tmp_path / "manifest.json",
        signature_path=tmp_path / "manifest.json.sig",
    )

    assert report["status"] == "INCONSISTENT"
    assert any("must be supplied together" in error for error in report["errors"])


def test_run_manifest_binds_contract_to_family_receiver_and_path_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    manifest = run_manifest.build_g3_joint_evidence_run_manifest(
        "g3-run-contract-001", source, saved, qualification_contract=contract
    )
    manifest_path = tmp_path / "run-manifest.json"
    _write_canonical(manifest_path, manifest)

    report = run_manifest.verify_g3_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert report["qualification_status"] == "NOT_QUALIFIED"
    assert report["qualification_contract_bound"] is True
    assert report["qualification_contract_sha256"] == contract["contract_sha256"]


def test_preregistered_run_preflights_then_writes_artifact_and_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    contract_path = tmp_path / "qualification-contract.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    _write_canonical(contract_path, contract)

    result = artifact.run_g3_frozen_qualification_evaluation(
        source, contract_path, saved, manifest_path
    )

    assert result["status"] == "CONTRACT_BOUND_NUMERICAL_EVALUATION_NOT_QUALIFIED"
    assert result["qualification_status"] == "NOT_QUALIFIED"
    assert saved.is_file()
    assert manifest_path.is_file()
    receipt = run_manifest.verify_g3_joint_evidence_run_manifest(manifest_path)
    assert receipt["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert receipt["qualification_contract_bound"] is True


def test_v4_contract_binds_raw_path_input_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    monkeypatch.setattr(
        joint,
        "evaluate_path_equivalence_roster",
        path_equivalence_assurance.evaluate_path_equivalence_roster,
    )
    source = tmp_path / "joint-input-v6.json"
    contract_path = tmp_path / "qualification-contract-v4.json"
    saved = tmp_path / "joint-report.json"
    manifest_path = tmp_path / "run-manifest.json"
    payload = _payload_v6()
    challenger_path = payload["path_equivalence"]["records"][0]["challenger_paths"][0]
    challenger_path["coefficient_real"] = 0.04
    challenger_path["cfr_contribution_real"] = [0.04, 0.04]
    payload["path_equivalence"]["records"][0]["challenger_full_cfr_real"] = [0.04, 0.04]
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        raw_paths=True,
    )
    payload["path_equivalence"]["solver_budget_provenance"] = dict(contract["solver_budget"])
    _write_canonical(source, payload)
    _write_canonical(contract_path, contract)

    result = artifact.run_g3_frozen_qualification_evaluation(
        source, contract_path, saved, manifest_path
    )

    assert result["status"] == "CONTRACT_BOUND_NUMERICAL_EVALUATION_NOT_QUALIFIED"
    assert contract["schema"] == "loslab.g3_qualification_run_contract.v4"
    report = json.loads(saved.read_text(encoding="utf-8"))
    assert (
        report["bundle"]["evaluations"]["path_equivalence"]["scenarios"][0][
            "class_coverage_status"
        ]
        == "FAIL"
    )
    receipt = run_manifest.verify_g3_joint_evidence_run_manifest(manifest_path)
    assert receipt["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert receipt["qualification_contract_bound"] is True


def test_v4_contract_rejects_raw_path_configuration_drift_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint-input-v6.json"
    payload = _payload_v6()
    payload["path_equivalence"]["comparison_configuration"]["noise_power_per_re_w"] *= 2
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        raw_paths=True,
    )
    payload["path_equivalence"]["solver_budget_provenance"] = dict(contract["solver_budget"])
    _write_canonical(source, payload)
    contract_path = tmp_path / "qualification-contract-v4.json"
    _write_canonical(contract_path, contract)

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("raw path drift must be rejected before scoring")

    monkeypatch.setattr(joint, "evaluate_g3_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="differs from frozen contract v4"):
        artifact.run_g3_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


def test_preregistered_run_rejects_contract_before_running_scorers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint-input.json"
    contract_path = tmp_path / "qualification-contract.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        candidate_config_id="wrong-candidate",
    )
    _write_canonical(contract_path, contract)

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("an invalid contract must be rejected before evaluator execution")

    monkeypatch.setattr(joint, "evaluate_g3_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="receiver configurations differ"):
        artifact.run_g3_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


def test_contract_bound_run_rejects_legacy_v2_episode_unbound_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint-input.json"
    contract_path = tmp_path / "qualification-contract.json"
    payload = _payload_v2()
    _write_canonical(source, payload)
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    _write_canonical(contract_path, contract)

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("legacy input must be rejected before evaluator execution")

    monkeypatch.setattr(joint, "evaluate_g3_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="requires joint evaluator input v4"):
        artifact.run_g3_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


def test_contract_bound_run_rejects_v3_without_row_configuration_binding(
    tmp_path: Path,
) -> None:
    source = tmp_path / "joint-input.json"
    payload = _payload_v3()
    _write_canonical(source, payload)
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload, source_identity["evaluator_source_bundle_sha256"]
    )
    contract_path = tmp_path / "qualification-contract.json"
    _write_canonical(contract_path, contract)

    with pytest.raises(ValueError, match="requires joint evaluator input v4"):
        artifact.run_g3_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


def test_contract_preflight_rejects_receiver_row_bound_to_wrong_config(
    tmp_path: Path,
) -> None:
    source = tmp_path / "joint-input.json"
    payload = _payload_v4()
    payload["receiver_impact"]["records"][0]["receiver_configuration_sha256"] = "0" * 64
    _write_canonical(source, payload)
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload, source_identity["evaluator_source_bundle_sha256"]
    )
    contract_path = tmp_path / "qualification-contract.json"
    _write_canonical(contract_path, contract)

    with pytest.raises(ValueError, match="configuration binding differs from frozen contract"):
        artifact.run_g3_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


def test_contract_preflight_rejects_path_comparison_parameter_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "joint-input.json"
    payload = _payload_v4()
    payload["path_equivalence"]["records"][0]["comparison"]["configuration"][
        "delay_tolerance_s"
    ] = 2e-9
    _write_canonical(source, payload)
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload, source_identity["evaluator_source_bundle_sha256"]
    )
    contract_path = tmp_path / "qualification-contract.json"
    _write_canonical(contract_path, contract)

    with pytest.raises(ValueError, match="path-equivalence comparison 0 configuration differs"):
        artifact.run_g3_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


def test_contract_bound_run_rejects_receiver_episode_drift_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint-input.json"
    contract_path = tmp_path / "qualification-contract.json"
    payload = _payload_v4()
    payload["receiver_impact"]["records"][0]["episode_id"] = "episode-unregistered"
    _write_canonical(source, payload)
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    _write_canonical(contract_path, contract)

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("receiver episode drift must be rejected before evaluator execution")

    monkeypatch.setattr(joint, "evaluate_g3_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="receiver-impact observed episode roster differs"):
        artifact.run_g3_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


@pytest.mark.parametrize("mismatch", ["waveform", "link_budget_noise"])
def test_preregistered_run_rejects_execution_context_drift_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mismatch: str
) -> None:
    source = tmp_path / "joint-input.json"
    contract_path = tmp_path / "qualification-contract.json"
    payload = _payload_v4()
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])

    if mismatch == "waveform":
        waveform = payload["receiver_execution"]["waveform"]
        waveform["configuration"]["mu"] = 2
        encoded = json.dumps(
            waveform["configuration"], sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        waveform["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
    else:
        payload["receiver_execution"]["link_budget_noise"]["tx_power_dbm"] += 1.0
    _write_canonical(source, payload)
    _write_canonical(contract_path, contract)

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("execution-context drift must be rejected before scoring")

    monkeypatch.setattr(joint, "evaluate_g3_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="waveform, receiver configs, or link budget"):
        artifact.run_g3_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


@pytest.mark.parametrize("role", ["candidate", "challenger"])
def test_joint_input_rejects_execution_configuration_hash_drift(
    tmp_path: Path, role: str
) -> None:
    source = tmp_path / "joint-input.json"
    payload = _payload_v2()
    payload["receiver_execution"][role]["configuration"]["receiver"] = "changed"
    _write_canonical(source, payload)

    with pytest.raises(ValueError, match="configuration hash is inconsistent"):
        artifact.evaluate_g3_joint_evidence_file(source)


@pytest.mark.parametrize("record_binding", ["missing", "mismatch"])
def test_joint_input_rejects_receiver_record_without_execution_context_binding(
    tmp_path: Path, record_binding: str
) -> None:
    source = tmp_path / "joint-input.json"
    payload = _payload_v2()
    record = payload["receiver_impact"]["records"][0]
    if record_binding == "missing":
        record.pop("receiver_execution_context_sha256")
    else:
        record["receiver_execution_context_sha256"] = "0" * 64
    _write_canonical(source, payload)

    with pytest.raises(ValueError, match="not bound to the declared receiver execution context"):
        artifact.evaluate_g3_joint_evidence_file(source)


def test_run_manifest_rejects_contract_config_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    bad_contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        candidate_config_id="wrong-candidate",
    )
    with pytest.raises(ValueError, match="receiver configurations differ"):
        run_manifest.build_g3_joint_evidence_run_manifest(
            "g3-run-contract-001", source, saved, qualification_contract=bad_contract
        )


def test_v3_joint_input_rejects_unregistered_receiver_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    payload = _payload_v4()
    payload["receiver_impact"]["records"][0]["episode_id"] = "episode-unregistered"
    _write_canonical(source, payload)

    with pytest.raises(ValueError, match="observed episode roster differs"):
        artifact.evaluate_g3_joint_evidence_file(source)


def test_run_manifest_rejects_contract_source_bundle_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    bad_contract = _qualification_contract(payload, "e" * 64)

    with pytest.raises(ValueError, match="source bundle differs"):
        run_manifest.build_g3_joint_evidence_run_manifest(
            "g3-run-contract-001", source, saved, qualification_contract=bad_contract
        )


def test_run_manifest_rejects_contract_run_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    source_identity = run_manifest.g3_evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])

    with pytest.raises(ValueError, match="run_id differs"):
        run_manifest.build_g3_joint_evidence_run_manifest(
            "different-run-id", source, saved, qualification_contract=contract
        )


def test_run_manifest_rejects_inventory_digest_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved, wrong_input_hash=True)

    report = run_manifest.verify_g3_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any("size or SHA-256" in error for error in report["errors"])


def test_run_manifest_detects_locked_dependency_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved)

    repository_root = Path(__file__).resolve().parents[2]
    source_root = tmp_path / "source-snapshot"
    for relative_path in run_manifest.EVALUATOR_SOURCE_PATHS:
        destination = source_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository_root / relative_path, destination)
    drifted_source = source_root / "uv.lock"
    drifted_source.write_bytes(drifted_source.read_bytes() + b"\n# drift\n")

    report = run_manifest.verify_g3_joint_evidence_run_manifest(
        manifest_path, source_root=source_root
    )

    assert report["status"] == "INCONSISTENT"
    assert any("evaluator source bytes differ" in error for error in report["errors"])


def test_run_manifest_rejects_incomplete_scorer_source_roster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    _write_canonical(source, _payload())
    artifact._write_new_artifact(saved, artifact.evaluate_g3_joint_evidence_file(source))
    manifest_path = tmp_path / "run-manifest.json"
    _write_run_manifest(manifest_path, source, saved)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evaluator_sources"].pop()
    _write_canonical(manifest_path, manifest)

    report = run_manifest.verify_g3_joint_evidence_run_manifest(manifest_path)

    assert report["status"] == "INCONSISTENT"
    assert any("violates schema" in error for error in report["errors"])
