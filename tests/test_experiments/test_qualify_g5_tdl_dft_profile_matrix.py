from __future__ import annotations

from typing import Any

from experiments import qualify_g5_tdl_dft_profile_matrix as matrix


def _profile_report(profile: str, *, passed: bool = True) -> dict[str, Any]:
    claims = (
        "static_reference",
        "mobile_bler_reduction_position_0_minus_3",
        "mobile_goodput_increase_position_3_minus_0_bits_per_slot",
    )
    return {
        "configuration": {"profile": f"TDL-{profile}"},
        "source_sha256": {"source.py": "a" * 64},
        "protocol_sha256": matrix._sha256((matrix.ROOT / matrix.PROTOCOL).read_bytes()),
        "status": f"PASS_BOUNDED_TDL_{profile}_FIXTURE" if passed else "NOT_MET",
        "acceptance": {name: {"pass": passed} for name in claims},
        "configuration_sha256": "b" * 64,
        "paired_channel_outcomes": [],
        "conditions": {},
        "result_sha256": "c" * 64,
        "git_revision": "fixture-revision",
        "runtime": {"device": "cpu"},
    }


def test_family_fixes_estimator_and_one_fifteen_claim_allocation(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_confirmation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _profile_report(kwargs["profile"])

    monkeypatch.setattr(matrix, "run_confirmation", fake_confirmation)

    family = matrix.run_family_confirmation()

    assert family["status"] == "PASS_BOUNDED_TDL_AE_FIXTURE"
    assert family["g5_09_closed"] is False
    assert family["family_claim_count"] == 15
    assert family["per_claim_alpha_bonferroni"] == 0.01 / 15
    assert family["profiles"] == ["A", "B", "C", "D", "E"]
    assert family["channel_estimator"] == {
        "kind": "finite_delay_dft",
        "min_delay_samples": -6,
        "max_delay_samples": 7,
    }
    assert len(calls) == 5
    assert all(call["family_claim_count"] == 15 for call in calls)
    assert all(call["channel_seeds"] == matrix.CHANNEL_SEEDS for call in calls)
    assert all(call["channel_estimator_kind"] == "finite_delay_dft" for call in calls)
    assert [call["bootstrap_seed"] for call in calls] == [949101, 949103, 949105, 949107, 949109]


def test_one_failed_claim_fails_the_whole_profile_family(monkeypatch) -> None:
    monkeypatch.setattr(
        matrix,
        "run_confirmation",
        lambda **kwargs: _profile_report(kwargs["profile"], passed=kwargs["profile"] != "D"),
    )

    family = matrix.run_family_confirmation()

    assert family["status"] == "NOT_MET"
    assert family["profile_summaries"][3]["status"] == "NOT_MET"


def test_seed_and_bootstrap_rosters_are_disjoint_from_tdl_c_dft_confirmation() -> None:
    assert matrix.CHANNEL_SEEDS == tuple(range(949001, 949081))
    assert set(matrix.CHANNEL_SEEDS).isdisjoint(range(948001, 948081))
    assert set(matrix.CHANNEL_SEEDS).isdisjoint(range(949101, 949111))
