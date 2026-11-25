"""Qualification run contract: each rejected specification or verification input has its reason."""

from __future__ import annotations

from copy import deepcopy

import evaluation.acceptance.qualification_contract as contract_module
import pytest
from evaluation.acceptance.qualification_contract import (
    _validate_sample_size_design_binding,
    build_qualification_run_contract,
    build_qualification_run_contract_v3,
    build_qualification_run_contract_v4,
    verify_qualification_run_contract,
)
from tests.test_evaluation.test_qualification_contract import _complete_design, _spec


def test_specification_must_be_an_object() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        build_qualification_run_contract(["not", "an", "object"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda s: s["candidate"].update(config_id=s["challenger"]["config_id"]),
            "config_id values must differ",
        ),
        (
            lambda s: s["sampling"].update(path_equivalence_scenarios=[]),
            "must be a nonempty roster",
        ),
        (
            lambda s: s["sampling"].update(
                path_equivalence_scenarios=[
                    ["site-a", "LOS", "episode-01"],
                    ["site-a", "LOS", "episode-01"],
                ]
            ),
            "must be unique",
        ),
        (
            lambda s: s["sampling"].update(bootstrap_seed=-1),
            "bootstrap_seed must be a nonnegative integer",
        ),
        (
            lambda s: s["sampling"].update(bootstrap_draws=10),
            "bootstrap_draws must be an integer of at least 1000",
        ),
    ],
)
def test_sampling_and_scope_rules(mutate, message) -> None:
    spec = deepcopy(_spec())
    mutate(spec)
    with pytest.raises(ValueError, match=message):
        build_qualification_run_contract(spec)


def test_solver_budget_policy_and_positive_counts_are_checked() -> None:
    spec = deepcopy(_spec())
    spec["solver_budget"] = dict(spec.get("solver_budget", {}))
    spec["solver_budget"]["overflow_policy"] = "silently_drop"
    with pytest.raises(ValueError, match="overflow_policy is unsupported"):
        build_qualification_run_contract(spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("git_revision", "abc", "40-character lowercase Git SHA"),
        ("git_revision", "A" * 40, "40-character lowercase Git SHA"),
        ("working_tree_dirty", "no", "must be boolean"),
        ("source_bundle_sha256", "zz", "must be a SHA-256 digest"),
    ],
)
def test_source_identity_is_validated(field, value, message) -> None:
    spec = deepcopy(_spec())
    spec["source_identity"] = dict(spec.get("source_identity", {}))
    spec["source_identity"][field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        build_qualification_run_contract(spec)


def test_verification_requires_an_object_with_a_valid_digest_and_schema() -> None:
    with pytest.raises(TypeError, match="contract must be an object"):
        verify_qualification_run_contract("not-a-contract")
    contract = build_qualification_run_contract(deepcopy(_spec()))
    broken = dict(contract, contract_sha256="NOT-A-DIGEST")
    with pytest.raises(ValueError, match="contract_sha256 must be a SHA-256 digest"):
        verify_qualification_run_contract(broken)
    unsupported = dict(contract, schema="umcd.other.v1")
    with pytest.raises(
        ValueError, match="digest does not match|schema is unsupported|fields differ"
    ):
        verify_qualification_run_contract(unsupported)


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ("incomplete", "requires a complete precision- and power-based design"),
        ("family", "acceptance family differs from contract"),
        ("branches", "branch_designs must be an object"),
        ("strata", "sample-size strata differ"),
        ("floor", "powered design requirement is below the evaluator floor"),
        ("planned", "below its powered design requirement"),
    ),
)
def test_sample_size_design_binding_rejects_contract_drift(
    monkeypatch, change: str, message: str
) -> None:
    spec = _spec()
    base = build_qualification_run_contract(spec)
    design = _complete_design(spec)
    monkeypatch.setattr(
        contract_module,
        "verify_joint_sample_size_design",
        lambda _design: {"complete_precision_powered_design": change != "incomplete"},
    )
    if change == "family":
        design["joint_acceptance_family"] = {"family_id": "other", "family_sha256": "0" * 64}
    elif change == "branches":
        design["branch_designs"] = []
    elif change == "strata":
        design["branch_designs"]["complex_csi"]["required_units_by_stratum"][0]["stratum"] = [
            "different-site",
            "LOS",
        ]
    elif change == "floor":
        design["branch_designs"]["complex_csi"]["required_units_by_stratum"][0][
            "required_units"
        ] = 1
    elif change == "planned":
        design["branch_designs"]["complex_csi"]["required_units_by_stratum"][0][
            "required_units"
        ] = 61

    with pytest.raises(ValueError, match=message):
        _validate_sample_size_design_binding(base, design)


def test_v3_specification_requires_its_exact_field_roster() -> None:
    with pytest.raises(TypeError, match="specification must be an object"):
        build_qualification_run_contract_v3([])
    with pytest.raises(TypeError, match="specification must be an object"):
        build_qualification_run_contract_v4([])

    spec = _spec()
    with pytest.raises(ValueError, match="specification fields differ"):
        build_qualification_run_contract_v3(spec)

    v4_spec = _spec()
    v4_spec["sample_size_design"] = _complete_design(v4_spec)
    v4_spec["unexpected"] = True
    with pytest.raises(ValueError, match="qualification v4 specification fields differ"):
        build_qualification_run_contract_v4(v4_spec)

    spec["sample_size_design"] = _complete_design(spec)
    spec["unexpected"] = True
    with pytest.raises(ValueError, match="specification fields differ"):
        build_qualification_run_contract_v3(spec)


def test_verifier_rejects_unsupported_schema_with_a_valid_digest() -> None:
    contract = build_qualification_run_contract(_spec())
    contract["schema"] = "loslab.qualification_run_contract.v99"
    contract.pop("contract_sha256")
    contract["contract_sha256"] = contract_module._digest(contract)

    with pytest.raises(ValueError, match="schema is unsupported"):
        verify_qualification_run_contract(contract)


def _redigest_contract(contract: dict) -> None:
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = contract_module._digest(contract)


def _v3_contract() -> dict:
    spec = _spec()
    spec["sample_size_design"] = _complete_design(spec)
    return build_qualification_run_contract_v3(spec)


def test_verifier_rejects_frozen_field_roster_drift_with_valid_digest() -> None:
    contract = build_qualification_run_contract(_spec())
    del contract["source_identity"]
    _redigest_contract(contract)

    with pytest.raises(ValueError, match="fields differ from its frozen schema"):
        verify_qualification_run_contract(contract)


def test_verifier_rejects_malformed_embedded_design_and_digest_binding() -> None:
    contract = _v3_contract()
    contract["sample_size_design"] = []
    _redigest_contract(contract)
    with pytest.raises(ValueError, match="sample_size_design must be an object"):
        verify_qualification_run_contract(contract)

    contract = _v3_contract()
    contract["sample_size_design_sha256"] = "0" * 64
    _redigest_contract(contract)
    with pytest.raises(ValueError, match="sample_size_design_sha256 differs"):
        verify_qualification_run_contract(contract)


def test_verifier_reconstructs_every_frozen_contract_field() -> None:
    contract = build_qualification_run_contract(_spec())
    contract["qualification_status"] = "PASS"
    _redigest_contract(contract)

    with pytest.raises(ValueError, match="differs from deterministic reconstruction"):
        verify_qualification_run_contract(contract)
