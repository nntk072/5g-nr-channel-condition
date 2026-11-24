"""Complex-CSI bootstrap assurance planner: input, scenario and work-bound rules."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from evaluation.acceptance.joint_acceptance import build_joint_acceptance_family
from evaluation.sample_size import bootstrap as bootstrap_module
from evaluation.sample_size.bootstrap import plan_complex_csi_bootstrap_assurance


def _family() -> dict[str, Any]:
    return build_joint_acceptance_family(
        family_id="csi-planning-rules",
        material_expected_strata=[("brick", "V", "normal")],
        complex_csi_expected_strata=[("urban", "LOS")],
    )


def _scenario(*, median: float = 0.005, p95: float = 0.02, failed: bool = False) -> dict[str, Any]:
    return {
        "assumed_cluster_summaries": [
            {"cluster_median_e_h": median, "cluster_p95_e_h": p95, "cluster_failed": failed}
        ],
        "probabilities": [1.0],
    }


def _kwargs(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "joint_acceptance_family": _family(),
        "assumed_iid_scenarios_by_stratum": {("urban", "LOS"): _scenario()},
        "candidate_clusters_per_stratum": [20, 120],
        "bootstrap_replicates": 1_000,
        "assurance_replicates": 100,
        "target_assurance": 0.80,
        "seed": 701,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param(
            {"assumed_iid_scenarios_by_stratum": []}, "must be a mapping", id="not-mapping"
        ),
        pytest.param(
            {"assumed_iid_scenarios_by_stratum": {("other", "LOS"): _scenario()}},
            "must exactly match the joint-family strata",
            id="wrong-strata",
        ),
        pytest.param(
            {"candidate_clusters_per_stratum": []}, "must contain integers >= 2", id="no-candidates"
        ),
        pytest.param(
            {"candidate_clusters_per_stratum": [1, 4]},
            "must contain integers >= 2",
            id="candidate-too-small",
        ),
        pytest.param(
            {"candidate_clusters_per_stratum": [True, 4]},
            "must contain integers >= 2",
            id="candidate-bool",
        ),
        pytest.param(
            {"candidate_clusters_per_stratum": [20, 20]}, "must be unique", id="duplicate-candidate"
        ),
        pytest.param(
            {"candidate_clusters_per_stratum": [120, 20]},
            "must be sorted ascending",
            id="unsorted-candidates",
        ),
        pytest.param(
            {"bootstrap_replicates": 999},
            "bootstrap_replicates must be an integer >= 1000",
            id="bootstrap-small",
        ),
        pytest.param(
            {"bootstrap_replicates": True},
            "bootstrap_replicates must be an integer >= 1000",
            id="bootstrap-bool",
        ),
        pytest.param(
            {"assurance_replicates": 99},
            "assurance_replicates must be an integer >= 100",
            id="assurance-small",
        ),
        pytest.param(
            {"assurance_replicates": 1.5},
            "assurance_replicates must be an integer >= 100",
            id="assurance-float",
        ),
        pytest.param({"seed": -1}, "seed must be a nonnegative integer", id="seed-negative"),
        pytest.param({"seed": False}, "seed must be a nonnegative integer", id="seed-bool"),
        pytest.param({"target_assurance": 1.0}, "target_assurance", id="target-one"),
        pytest.param(
            {"monte_carlo_confidence": 0.0}, "monte_carlo_confidence", id="confidence-zero"
        ),
    ],
)
def test_planner_rejects_invalid_arguments(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        plan_complex_csi_bootstrap_assurance(**_kwargs(**overrides))


def test_planner_bounds_total_resampling_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap_module, "_MAX_BOOTSTRAP_RESAMPLING_CELLS", 10)
    with pytest.raises(ValueError, match="exceeds the bounded work limit"):
        plan_complex_csi_bootstrap_assurance(**_kwargs())


def test_planner_bounds_peak_resample_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap_module, "_MAX_BOOTSTRAP_PEAK_CELLS", 100)
    with pytest.raises(ValueError, match="single CSI bootstrap resample exceeds"):
        plan_complex_csi_bootstrap_assurance(**_kwargs())


@pytest.mark.parametrize(
    ("scenario_change", "message"),
    [
        pytest.param(
            lambda s: s.update(probabilities=[0.5, 0.2]), "probabilit", id="probabilities-sum"
        ),
        pytest.param(
            lambda s: s.update(assumed_cluster_summaries=[]), "cluster", id="no-summaries"
        ),
        pytest.param(
            lambda s: s["assumed_cluster_summaries"][0].update(cluster_median_e_h=-1.0),
            "cluster",
            id="negative-median",
        ),
        pytest.param(
            lambda s: s["assumed_cluster_summaries"][0].update(cluster_p95_e_h=0.001),
            "p95",
            id="p95-below-median",
        ),
        pytest.param(
            lambda s: s["assumed_cluster_summaries"][0].update(cluster_failed="yes"),
            "cluster_failed",
            id="failed-not-boolean",
        ),
    ],
)
def test_scenario_atoms_must_be_admissible(scenario_change, message: str) -> None:
    scenario = deepcopy(_scenario())
    scenario_change(scenario)
    with pytest.raises(ValueError, match=message):
        plan_complex_csi_bootstrap_assurance(
            **_kwargs(assumed_iid_scenarios_by_stratum={("urban", "LOS"): scenario})
        )
