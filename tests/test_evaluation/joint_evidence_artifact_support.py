"""Shared payload and signing builders for the joint-evidence artifact tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from evaluation import joint_evidence_artifact as artifact
from evaluation import joint_evidence_run_manifest as run_manifest
from evaluation.acceptance import joint_evaluation as joint
from evaluation.acceptance import path_equivalence_assurance as path_equivalence_assurance
from evaluation.acceptance.joint_acceptance import build_joint_acceptance_family_v4
from evaluation.acceptance.qualification_contract import (
    build_qualification_run_contract_v3,
    build_qualification_run_contract_v4,
)


def _payload() -> dict[str, Any]:
    family = build_joint_acceptance_family_v4(
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
    payload["receiver_impact"]["records"][0]["receiver_execution_context_sha256"] = hashlib.sha256(
        json.dumps(
            payload["receiver_execution"],
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
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

    monkeypatch.setattr(joint, "evaluate_material_calibration", material)
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


def _write_run_manifest(
    path: Path,
    source: Path,
    saved: Path,
    *,
    wrong_input_hash: bool = False,
    qualification_contract: dict[str, Any] | None = None,
    runtime_inventory_path: Path | None = None,
) -> None:
    run_id = (
        qualification_contract["run_id"] if qualification_contract is not None else "joint-run-001"
    )
    manifest = run_manifest.build_joint_evidence_run_manifest(
        run_id,
        source,
        saved,
        qualification_contract=qualification_contract,
        runtime_inventory_path=runtime_inventory_path,
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
        f'{signer_id} namespaces="{run_manifest.RUN_MANIFEST_SIGNATURE_NAMESPACE}" {public_key}\n',
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            run_manifest.RUN_MANIFEST_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(f"{manifest_path}.sig"), allowed_signers, signer_id


def _runtime_inventory_fixture(path: Path) -> None:
    from integrations.runtime_inventory import build_runtime_inventory

    source_root = Path(__file__).resolve().parents[2]
    inventory = build_runtime_inventory(source_root, probe_loaded_native_libraries=True)
    identity = run_manifest._runtime_identity()
    inventory["python"]["version"] = identity["python"]
    inventory["python"]["implementation"] = identity["python_implementation"]
    inventory["platform"].update(
        {
            "system": identity["platform_system"],
            "release": identity["platform_release"],
            "machine": identity["machine"],
        }
    )
    inventory["source"]["git_commit"] = identity["source_revision"]
    inventory["source"]["tracked_worktree_clean"] = True
    for relative_path in ("pyproject.toml", "uv.lock"):
        inventory["source"][relative_path] = hashlib.sha256(
            (source_root / relative_path).read_bytes()
        ).hexdigest()
    for relative_path in (
        "integrations/runtime_inventory.py",
        "integrations/runtime_compatibility.py",
        "scripts/report_runtime_inventory.py",
    ):
        inventory["source"]["runtime_inventory_sources"][relative_path] = hashlib.sha256(
            (source_root / relative_path).read_bytes()
        ).hexdigest()
    distribution_by_name = {
        item["name"].lower().replace("_", "-"): item for item in inventory["distributions"]
    }
    runtime_packages = identity["packages"]
    package_candidates = {
        "numpy": ("numpy",),
        "scikit-learn": ("scikit-learn",),
        "torch": ("torch",),
        "sionna": ("sionna",),
        "sionna-rt": ("sionna-rt", "sionna-no-rt"),
        "sionna-phy": ("sionna-no-rt",),
    }
    for package_name, candidates in package_candidates.items():
        package_version = runtime_packages[package_name]
        matching = [distribution_by_name[name] for name in candidates]
        target = next(
            (
                item
                for item in matching
                if item.get("installed") is True and item.get("version") == package_version
            ),
            matching[0],
        )
        for distribution in matching:
            if distribution is target and package_version != "not installed":
                distribution["installed"] = True
                distribution["version"] = package_version
            else:
                distribution_name = distribution["name"]
                distribution.clear()
                distribution.update({"name": distribution_name, "installed": False})
    _write_canonical(path, inventory)


def _sign_runtime_inventory(inventory_path: Path, directory: Path) -> tuple[Path, Path, str]:
    ssh_keygen = shutil.which("ssh-keygen")
    if ssh_keygen is None:
        pytest.skip("OpenSSH is unavailable")
    signer_id = "g3-runtime-attestor"
    private_key = directory / "g3-runtime-attestor-key"
    subprocess.run(
        [ssh_keygen, "-q", "-t", "ed25519", "-N", "", "-f", str(private_key)],
        check=True,
        capture_output=True,
        text=True,
    )
    public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
    allowed_signers = directory / "runtime-allowed-signers"
    allowed_signers.write_text(
        f'{signer_id} namespaces="{run_manifest.RUNTIME_INVENTORY_SIGNATURE_NAMESPACE}" '
        f"{public_key}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            run_manifest.RUNTIME_INVENTORY_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(inventory_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(f"{inventory_path}.sig"), allowed_signers, signer_id


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
                "complex_csi": [{"stratum": ["scene-a", "LOS"], "planned_units": 60}],
                "propagation_metrics": [
                    {"stratum": ["scene-a", "LOS", "total"], "planned_units": 60}
                ],
                "receiver_impact": [{"stratum": ["scene-a", "LOS"], "planned_units": 100}],
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
        return build_qualification_run_contract_v4(contract_spec)
    return build_qualification_run_contract_v3(contract_spec)


def _synthetic_complete_sample_size_design(family: dict[str, Any]) -> dict[str, Any]:
    branch_rosters = {
        "material_calibration": ("material_expected_strata", 20, "independent_acquisition_unit"),
        "complex_csi": ("complex_csi_expected_strata", 2, "independent_physical_episode"),
        "propagation_metrics": ("propagation_expected_strata", 2, "independent_physical_episode"),
        "receiver_impact": (
            "receiver_impact_expected_strata",
            20,
            "independent_transport_block_cluster",
        ),
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
            name: {"schema": schema, "sha256": "a" * 64} for name, schema in schemas.items()
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
        "joint_assurance_lower_bound_union_bound": 0.96,
        "target_joint_assurance_lower_bound": 0.8,
        "material_angular_maximum_assumption": {
            "supplied": True,
            "compatible_with_material_design": True,
            "maximum_units_per_stratum": 100,
        },
        "complete_precision_powered_design": True,
        "complete_sample_size": None,
        "blockers": [],
        "limitations": ["Synthetic test design; no scientific assumptions are verified."],
    }
    design["design_sha256"] = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return design
