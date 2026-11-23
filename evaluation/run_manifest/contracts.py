"""Joint evaluator input schema and qualification contract bindings for run manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evaluation.acquisition_package import (
    MAX_SCHEMA_BYTES,
    _json_bytes,
)
from evaluation.qualification_contract import verify_qualification_run_contract
from evaluation.run_manifest.common import (
    _canonical_json_bytes,
)
from evaluation.run_manifest.constants import (
    INPUT_SCHEMA_RELATIVE_PATH,
    INPUT_SCHEMA_V2_RELATIVE_PATH,
    INPUT_SCHEMA_V3_RELATIVE_PATH,
    INPUT_SCHEMA_V4_RELATIVE_PATH,
    INPUT_SCHEMA_V5_RELATIVE_PATH,
    INPUT_SCHEMA_V6_RELATIVE_PATH,
)
from evaluation.run_manifest.sources import (
    _read_rooted_source_file,
)
from evaluation.sample_size_planning import verify_joint_sample_size_design


def _validate_joint_evaluator_input_schema(payload: Any, *, source_root: Path) -> str:
    """Validate a serialized joint input against its version-pinned source schema."""
    if not isinstance(payload, dict):
        raise ValueError("G3 joint evaluator input must be an object")
    schema_by_version = {
        "loslab.g3_joint_evaluator_inputs.v1": INPUT_SCHEMA_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v2": INPUT_SCHEMA_V2_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v3": INPUT_SCHEMA_V3_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v4": INPUT_SCHEMA_V4_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v5": INPUT_SCHEMA_V5_RELATIVE_PATH,
        "loslab.g3_joint_evaluator_inputs.v6": INPUT_SCHEMA_V6_RELATIVE_PATH,
    }
    schema_version = payload.get("schema")
    if not isinstance(schema_version, str):
        raise ValueError("G3 joint evaluator input schema must be a string")
    schema_path = schema_by_version.get(schema_version)
    if schema_path is None:
        raise ValueError("unsupported G3 joint evaluator input schema")
    schema = _json_bytes(
        _read_rooted_source_file(source_root, schema_path, limit=MAX_SCHEMA_BYTES),
        "G3 joint evaluator input schema",
    )
    import jsonschema

    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(payload))
    if errors:
        raise ValueError(
            "G3 joint evaluator input violates schema: "
            + "; ".join(error.message for error in errors[:10])
        )
    return schema_path


def _verify_qualification_contract_bindings(
    contract: dict[str, Any],
    input_bytes: bytes,
    *,
    require_powered_design: bool = False,
    require_raw_path_criteria: bool = False,
) -> str:
    verification = verify_qualification_run_contract(contract)
    contract_schema = contract.get("schema")
    if require_powered_design:
        if contract_schema not in {
            "loslab.g3_qualification_run_contract.v3",
            "loslab.g3_qualification_run_contract.v4",
        }:
            raise ValueError(
                "new contract-bound G3 evaluation requires qualification contract v3 or v4"
            )
        design_verification = verify_joint_sample_size_design(contract["sample_size_design"])
        if design_verification["complete_precision_powered_g3_design"] is not True:
            raise ValueError("contract-bound G3 evaluation requires a complete powered design")
    if require_raw_path_criteria and contract_schema != ("loslab.g3_qualification_run_contract.v4"):
        raise ValueError(
            "strict signed reviewer admission requires qualification contract v4 "
            "with raw path criteria and a complete powered design"
        )
    input_payload = _json_bytes(input_bytes, "G3 joint evaluator input")
    if not isinstance(input_payload, dict):
        raise ValueError("G3 joint evaluator input must be an object")
    input_schema = input_payload.get("schema")
    if contract_schema == "loslab.g3_qualification_run_contract.v3":
        allowed_input_schemas = {"loslab.g3_joint_evaluator_inputs.v4"}
    elif contract_schema == "loslab.g3_qualification_run_contract.v4":
        allowed_input_schemas = {"loslab.g3_joint_evaluator_inputs.v6"}
        if not require_powered_design:
            allowed_input_schemas.add("loslab.g3_joint_evaluator_inputs.v5")
    else:
        allowed_input_schemas = set()
    if input_schema not in allowed_input_schemas:
        raise ValueError(
            "contract-bound G3 evaluation requires input v4 for contract v3; "
            "new contract-v4 runs require input v6 with budget-bound diffuse diagnostics"
        )
    if input_payload.get("joint_acceptance_family") != contract["joint_acceptance_family"]:
        raise ValueError("qualification contract acceptance family differs from run input")
    receiver_input = input_payload.get("receiver_impact")
    if not isinstance(receiver_input, dict) or (
        receiver_input.get("candidate_config_id") != contract["candidate"]["config_id"]
        or receiver_input.get("challenger_config_id") != contract["challenger"]["config_id"]
    ):
        raise ValueError("qualification contract receiver configurations differ from run input")
    expected_execution = {
        "candidate": contract["candidate"],
        "challenger": contract["challenger"],
        "waveform": contract["waveform"],
        "link_budget_noise": contract["link_budget_noise"],
    }
    if input_payload.get("receiver_execution") != expected_execution:
        raise ValueError(
            "qualification contract waveform, receiver configs, or link budget differ from execution context"
        )
    propagation_input = input_payload.get("propagation_metrics")
    if not isinstance(propagation_input, dict):
        raise ValueError("qualification contract propagation-metric input must be an object")
    if contract_schema == "loslab.g3_qualification_run_contract.v4":
        frozen_bandwidth = contract["task_scope"].get("effective_bandwidth_hz")
        supplied_bandwidth = propagation_input.get("effective_bandwidth_hz")
        if frozen_bandwidth is None and require_powered_design:
            raise ValueError(
                "powered contract-v4 G3 evaluation requires frozen propagation effective bandwidth"
            )
        if frozen_bandwidth is not None and _canonical_json_bytes(
            frozen_bandwidth
        ) != _canonical_json_bytes(supplied_bandwidth):
            raise ValueError(
                "propagation effective bandwidth differs from frozen qualification task scope"
            )
    path_input = input_payload.get("path_equivalence")
    if (
        not isinstance(path_input, dict)
        or path_input.get("expected_scenarios")
        != (contract["sampling"]["path_equivalence_scenarios"])
    ):
        raise ValueError("qualification contract path roster differs from run input")
    path_records = path_input.get("records")
    if not isinstance(path_records, list) or not path_records:
        raise ValueError("path-equivalence input must contain a nonempty comparison roster")
    if contract_schema == "loslab.g3_qualification_run_contract.v4":
        expected_path_configuration = {
            "candidate_configuration_id": contract["candidate"]["config_id"],
            "challenger_configuration_id": contract["challenger"]["config_id"],
            **{
                key: contract["path_matching"][key]
                for key in (
                    "vertex_tolerance_m",
                    "delay_tolerance_s",
                    "direction_tolerance_rad",
                    "h_floor",
                    "transmit_power_per_re_w",
                    "noise_power_per_re_w",
                )
            },
        }
        if path_input.get("comparison_configuration") != expected_path_configuration:
            raise ValueError("raw path-equivalence configuration differs from frozen contract v4")
        if (
            input_schema == "loslab.g3_joint_evaluator_inputs.v6"
            and path_input.get("solver_budget_provenance") != contract["solver_budget"]
        ):
            raise ValueError("diffuse-path solver budget differs from frozen contract v4")
        for index, record in enumerate(path_records):
            if not isinstance(record, dict) or "comparison" in record:
                raise ValueError(
                    f"raw path-equivalence record {index} must omit supplied comparison"
                )
    else:
        expected_path_configuration = {
            "candidate_configuration_id": contract["candidate"]["config_id"],
            "challenger_configuration_id": contract["challenger"]["config_id"],
            "delay_tolerance_s": contract["path_matching"]["delay_tolerance_s"],
        }
        expected_coverage = contract["path_matching"]["minimum_bidirectional_coverage"]
        for index, record in enumerate(path_records):
            if not isinstance(record, dict) or not isinstance(record.get("comparison"), dict):
                raise ValueError(f"path-equivalence record {index} needs a comparison object")
            comparison = record["comparison"]
            configuration = comparison.get("configuration")
            if (
                not isinstance(configuration, dict)
                or any(
                    configuration.get(key) != expected
                    for key, expected in expected_path_configuration.items()
                )
                or comparison.get("coverage_criterion") != expected_coverage
            ):
                raise ValueError(
                    f"path-equivalence comparison {index} configuration differs from frozen contract"
                )
    expected_episodes = {tuple(item) for item in contract["sampling"]["path_equivalence_scenarios"]}
    receiver_records = receiver_input.get("records")
    if not isinstance(receiver_records, list) or not receiver_records:
        raise ValueError("receiver-impact input must include a nonempty episode-bound record list")
    observed_episodes: set[tuple[str, str, str]] = set()
    config_digests = {
        contract[role]["config_id"]: contract[role]["configuration_sha256"]
        for role in ("candidate", "challenger")
    }
    for index, record in enumerate(receiver_records):
        if not isinstance(record, dict):
            raise ValueError(f"receiver-impact record {index} must be an object")
        family_id = record.get("family_id")
        condition = record.get("condition")
        episode_id = record.get("episode_id")
        if (
            not isinstance(family_id, str)
            or not family_id.strip()
            or not isinstance(condition, str)
            or condition not in {"LOS", "NLOS"}
            or not isinstance(episode_id, str)
            or not episode_id.strip()
        ):
            raise ValueError(
                f"receiver-impact record {index} needs family_id, LOS/NLOS condition, "
                "and episode_id"
            )
        configuration_id = record.get("configuration_id")
        if (
            not isinstance(configuration_id, str)
            or configuration_id not in config_digests
            or record.get("receiver_configuration_sha256") != config_digests.get(configuration_id)
        ):
            raise ValueError(
                f"receiver-impact record {index} configuration binding differs from frozen contract"
            )
        observed_episodes.add((family_id, condition, episode_id))
    if observed_episodes != expected_episodes:
        raise ValueError(
            "receiver-impact observed episode roster differs from the frozen path roster"
        )
    return str(verification["contract_sha256"])
