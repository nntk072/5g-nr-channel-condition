from __future__ import annotations

from typing import Any

from experiments import qualify_g5_tdl_profile_matrix as matrix


def _profile_report(profile: str, *, passed: bool = True) -> dict[str, Any]:
    protocol_bytes = (matrix.ROOT / matrix.PROTOCOL).read_bytes()
    acceptance = {
        "static_reference": {"pass": passed},
        "mobile_bler_reduction_position_0_minus_3": {"pass": passed},
        "mobile_goodput_increase_position_3_minus_0_bits_per_slot": {"pass": passed},
    }
    return {
        "configuration": {"profile": f"TDL-{profile}"},
        "source_sha256": {"source.py": "a" * 64},
        "protocol_sha256": matrix._sha256(protocol_bytes),
        "status": f"PASS_BOUNDED_TDL_{profile}_FIXTURE" if passed else "NOT_MET",
        "acceptance": acceptance,
        "configuration_sha256": "b" * 64,
        "paired_channel_outcomes": [],
        "conditions": {},
        "result_sha256": "c" * 64,
        "git_revision": "fixture-revision",
        "runtime": {"device": "cpu"},
    }


def test_profile_family_applies_one_fifteen_claim_family(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_confirmation(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _profile_report(kwargs["profile"])

    monkeypatch.setattr(matrix, "run_confirmation", fake_confirmation)

    result = matrix.run_family_confirmation()

    assert result["status"] == "PASS_BOUNDED_TDL_AE_FIXTURE"
    assert result["g5_09_closed"] is False
    assert result["family_claim_count"] == 15
    assert result["per_claim_alpha_bonferroni"] == 0.01 / 15
    assert result["profiles"] == ["A", "B", "C", "D", "E"]
    assert len(calls) == 5
    assert all(call["family_claim_count"] == 15 for call in calls)
    assert all(call["channel_seeds"] == matrix.CHANNEL_SEEDS for call in calls)
    assert [call["bootstrap_seed"] for call in calls] == [934100, 934102, 934104, 934106, 934108]


def test_any_failed_profile_makes_family_not_met(monkeypatch) -> None:
    monkeypatch.setattr(
        matrix,
        "run_confirmation",
        lambda **kwargs: _profile_report(kwargs["profile"], passed=kwargs["profile"] != "D"),
    )

    result = matrix.run_family_confirmation()

    assert result["status"] == "NOT_MET"
    assert {item["profile"]: item["status"] for item in result["profile_summaries"]}["TDL-D"] == (
        "NOT_MET"
    )


def test_seed_range_is_disjoint_from_prior_tdl_c_confirmation() -> None:
    assert matrix.CHANNEL_SEEDS[0] == 934001
    assert matrix.CHANNEL_SEEDS[-1] == 934064
    assert set(matrix.CHANNEL_SEEDS).isdisjoint(range(933001, 933065))
