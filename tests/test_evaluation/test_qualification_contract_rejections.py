"""Qualification run contract: each rejected specification or verification input has its reason."""

from __future__ import annotations

from copy import deepcopy

import pytest

from evaluation.acceptance.qualification_contract import (
    build_qualification_run_contract,
    verify_qualification_run_contract,
)
from tests.test_evaluation.test_qualification_contract import _spec


def test_specification_must_be_an_object() -> None:
    with pytest.raises(TypeError, match="must be an object"):
        build_qualification_run_contract(["not", "an", "object"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s["candidate"].update(config_id=s["challenger"]["config_id"]), "config_id values must differ"),
        (lambda s: s["sampling"].update(path_equivalence_scenarios=[]), "must be a nonempty roster"),
        (lambda s: s["sampling"].update(path_equivalence_scenarios=[["site-a", "LOS", "episode-01"], ["site-a", "LOS", "episode-01"]]), "must be unique"),
        (lambda s: s["sampling"].update(bootstrap_seed=-1), "bootstrap_seed must be a nonnegative integer"),
        (lambda s: s["sampling"].update(bootstrap_draws=10), "bootstrap_draws must be an integer of at least 1000"),
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
    with pytest.raises(ValueError, match="digest does not match|schema is unsupported|fields differ"):
        verify_qualification_run_contract(unsupported)
