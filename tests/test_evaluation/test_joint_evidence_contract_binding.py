"""Qualification-contract binding for joint-evidence run manifests, including raw path inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from evaluation import joint_evaluation as joint
from evaluation import joint_evidence_artifact as artifact
from evaluation import joint_evidence_run_manifest as run_manifest
from evaluation import path_equivalence_assurance as path_equivalence_assurance
from tests.test_evaluation.joint_evidence_artifact_support import (
    _install_scorers,
    _payload_v2,
    _payload_v3,
    _payload_v4,
    _payload_v5,
    _payload_v6,
    _qualification_contract,
    _runtime_inventory_fixture,
    _sign_run_manifest,
    _sign_runtime_inventory,
    _write_canonical,
    _write_run_manifest,
)


def test_v4_schema_requires_row_configuration_identity() -> None:
    payload = _payload_v4()
    payload["receiver_impact"]["records"][0].pop("configuration_id")
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/joint-evaluator-input.v4.schema.json"
    )
    validator = jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))

    assert list(validator.iter_errors(payload))


def test_v4_schema_requires_path_comparison_binding() -> None:
    payload = _payload_v4()
    payload["path_equivalence"]["records"][0]["comparison"]["configuration"].pop(
        "delay_tolerance_s"
    )
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/joint-evaluator-input.v4.schema.json"
    )
    validator = jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))

    assert list(validator.iter_errors(payload))


def test_v5_schema_requires_raw_paths_and_frozen_configuration() -> None:
    payload = _payload_v5()
    payload["path_equivalence"]["comparison_configuration"]["h_floor"] = 0.0
    schema_path = (
        Path(__file__).resolve().parents[2] / "datasets/schema/joint-evaluator-input.v5.schema.json"
    )
    validator = jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8")))

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
    runtime_inventory_path = tmp_path / "runtime-inventory.json"
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
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        raw_paths=criterion_driven_contract,
    )
    _runtime_inventory_fixture(runtime_inventory_path)
    _write_run_manifest(
        manifest_path,
        source,
        saved,
        qualification_contract=contract,
        runtime_inventory_path=runtime_inventory_path,
    )
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
        f'{policy_signer_id} namespaces="{run_manifest.SIGNER_POLICY_SIGNATURE_NAMESPACE}" '
        f"{public_key}\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            ssh_keygen,
            "-Y",
            "sign",
            "-n",
            run_manifest.SIGNER_POLICY_SIGNATURE_NAMESPACE,
            "-f",
            str(private_key),
            str(policy_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    policy_signature = Path(f"{policy_path}.sig")
    runtime_signature, runtime_signers, runtime_signer_id = _sign_runtime_inventory(
        runtime_inventory_path, tmp_path
    )

    report = run_manifest.verify_joint_evidence_run_manifest(
        manifest_path,
        signature_path=manifest_signature,
        trusted_allowed_signers_path=manifest_signers,
        trusted_signer_policy_path=policy_path,
        signer_id=manifest_signer_id,
        trusted_signer_policy_signature_path=policy_signature,
        trusted_policy_allowed_signers_path=policy_signers,
        policy_signer_id=policy_signer_id,
        runtime_inventory_signature_path=runtime_signature,
        trusted_runtime_inventory_signers_path=runtime_signers,
        runtime_inventory_signer_id=runtime_signer_id,
        require_signed_reviewer_admission=True,
    )

    if criterion_driven_contract:
        assert report["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
        assert report["qualification_status"] == "NOT_QUALIFIED"
        assert report["signer_policy_signature_status"] == "verified"
        assert report["signer_policy_signer_id"] == policy_signer_id
        assert report["signer_policy_custody"] == "not_established_by_this_verifier"
        assert report["runtime_inventory_signature_status"] == "verified"
        assert report["runtime_inventory_binding_status"] == "verified"
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
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        raw_paths=True,
    )
    payload["path_equivalence"]["solver_budget_provenance"] = dict(contract["solver_budget"])
    _write_canonical(source, payload)
    _write_canonical(contract_path, contract)

    result = artifact.run_frozen_qualification_evaluation(
        source, contract_path, saved, manifest_path
    )

    assert result["status"] == "CONTRACT_BOUND_NUMERICAL_EVALUATION_NOT_QUALIFIED"
    assert contract["schema"] == "loslab.g3_qualification_run_contract.v4"
    report = json.loads(saved.read_text(encoding="utf-8"))
    assert (
        report["bundle"]["evaluations"]["path_equivalence"]["scenarios"][0]["class_coverage_status"]
        == "FAIL"
    )
    receipt = run_manifest.verify_joint_evidence_run_manifest(manifest_path)
    assert receipt["status"] == "PASS_INPUT_SCOPED_RECOMPUTATION"
    assert receipt["qualification_contract_bound"] is True


def test_v4_contract_rejects_raw_path_configuration_drift_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "joint-input-v6.json"
    payload = _payload_v6()
    payload["path_equivalence"]["comparison_configuration"]["noise_power_per_re_w"] *= 2
    source_identity = run_manifest.evaluator_source_bundle_identity()
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

    monkeypatch.setattr(joint, "evaluate_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="differs from frozen contract v4"):
        artifact.run_frozen_qualification_evaluation(
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
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    _write_canonical(contract_path, contract)

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("legacy input must be rejected before evaluator execution")

    monkeypatch.setattr(joint, "evaluate_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="requires joint evaluator input v4"):
        artifact.run_frozen_qualification_evaluation(
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
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    contract_path = tmp_path / "qualification-contract.json"
    _write_canonical(contract_path, contract)

    with pytest.raises(ValueError, match="requires joint evaluator input v4"):
        artifact.run_frozen_qualification_evaluation(
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
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    contract_path = tmp_path / "qualification-contract.json"
    _write_canonical(contract_path, contract)

    with pytest.raises(ValueError, match="configuration binding differs from frozen contract"):
        artifact.run_frozen_qualification_evaluation(
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
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    contract_path = tmp_path / "qualification-contract.json"
    _write_canonical(contract_path, contract)

    with pytest.raises(ValueError, match="path-equivalence comparison 0 configuration differs"):
        artifact.run_frozen_qualification_evaluation(
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
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])
    _write_canonical(contract_path, contract)

    def unexpected_evaluation(*args: Any, **kwargs: Any) -> dict[str, str]:
        del args, kwargs
        raise AssertionError("receiver episode drift must be rejected before evaluator execution")

    monkeypatch.setattr(joint, "evaluate_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="receiver-impact observed episode roster differs"):
        artifact.run_frozen_qualification_evaluation(
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
    source_identity = run_manifest.evaluator_source_bundle_identity()
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

    monkeypatch.setattr(joint, "evaluate_material_calibration", unexpected_evaluation)

    with pytest.raises(ValueError, match="waveform, receiver configs, or link budget"):
        artifact.run_frozen_qualification_evaluation(
            source,
            contract_path,
            tmp_path / "joint-report.json",
            tmp_path / "run-manifest.json",
        )


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
        artifact.evaluate_joint_evidence_file(source)


def test_run_manifest_rejects_contract_config_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    source_identity = run_manifest.evaluator_source_bundle_identity()
    bad_contract = _qualification_contract(
        payload,
        source_identity["evaluator_source_bundle_sha256"],
        candidate_config_id="wrong-candidate",
    )
    with pytest.raises(ValueError, match="receiver configurations differ"):
        run_manifest.build_joint_evidence_run_manifest(
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
        artifact.evaluate_joint_evidence_file(source)


def test_run_manifest_rejects_contract_run_id_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_scorers(monkeypatch)
    source = tmp_path / "joint-input.json"
    saved = tmp_path / "joint-report.json"
    payload = _payload_v4()
    _write_canonical(source, payload)
    artifact._write_new_artifact(saved, artifact.evaluate_joint_evidence_file(source))
    source_identity = run_manifest.evaluator_source_bundle_identity()
    contract = _qualification_contract(payload, source_identity["evaluator_source_bundle_sha256"])

    with pytest.raises(ValueError, match="run_id differs"):
        run_manifest.build_joint_evidence_run_manifest(
            "different-run-id", source, saved, qualification_contract=contract
        )
