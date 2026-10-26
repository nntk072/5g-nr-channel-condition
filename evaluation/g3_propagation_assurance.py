"""Plan propagation-metric evaluator assurance from explicit hypothetical scenarios."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import beta

from evaluation.g3_joint_acceptance import (
    family_alpha_for_claim,
    validate_g3_joint_acceptance_family,
)
from evaluation.g3_propagation_metrics import evaluate_propagation_metrics

_METRICS = (
    "first_arrival_delay_s",
    "rms_delay_spread_s",
    "arrival_angular_spread_deg",
    "departure_angular_spread_deg",
)
_MAX_ASSURANCE_WORK = 100_000_000
_MAX_STUDY_EPISODES = 100_000


def _finite_number(value: Any, name: str, *, signed: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (not signed and result < 0.0):
        raise ValueError(f"{name} must be a finite {'signed ' if signed else 'nonnegative '}number")
    return result


def _validate_scenarios(
    raw_scenarios: Mapping[tuple[str, str, str], Mapping[str, Any]],
    *,
    expected_strata: set[tuple[str, str, str]],
) -> dict[tuple[str, str, str], tuple[list[list[dict[str, Any]]], np.ndarray]]:
    if not isinstance(raw_scenarios, Mapping) or set(raw_scenarios) != expected_strata:
        raise ValueError("propagation scenarios must exactly match the joint-family strata")
    validated: dict[tuple[str, str, str], tuple[list[list[dict[str, Any]]], np.ndarray]] = {}
    for stratum, raw in raw_scenarios.items():
        if not isinstance(raw, Mapping) or set(raw) != {
            "assumed_cluster_episodes",
            "probabilities",
        }:
            raise ValueError(
                "each propagation scenario must contain cluster episodes and probabilities"
            )
        atoms_raw = raw["assumed_cluster_episodes"]
        probabilities_raw = raw["probabilities"]
        if (
            not isinstance(atoms_raw, Sequence)
            or isinstance(atoms_raw, (str, bytes))
            or not atoms_raw
        ):
            raise ValueError("assumed_cluster_episodes must be a nonempty sequence")
        if (
            not isinstance(probabilities_raw, Sequence)
            or isinstance(probabilities_raw, (str, bytes))
            or len(probabilities_raw) != len(atoms_raw)
        ):
            raise ValueError("propagation probabilities must match the cluster atoms")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in probabilities_raw
        ):
            raise ValueError("propagation probabilities must be numeric weights")
        atoms = [
            _validate_cluster_atom(atom, stratum=stratum, atom_index=index)
            for index, atom in enumerate(atoms_raw)
        ]
        try:
            raw_probabilities = np.asarray(probabilities_raw)
            if raw_probabilities.dtype.kind == "b":
                raise ValueError("propagation probabilities must be numeric weights")
            probabilities = np.asarray(raw_probabilities, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("propagation probabilities must be numeric weights") from error
        if (
            probabilities.ndim != 1
            or not np.all(np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("propagation probabilities must be positive and sum to one")
        validated[stratum] = (atoms, probabilities)
    return validated


def _validate_cluster_atom(
    atom: Any,
    *,
    stratum: tuple[str, str, str],
    atom_index: int,
) -> list[dict[str, Any]]:
    if not isinstance(atom, Mapping) or set(atom) != {"episodes"}:
        raise ValueError(f"propagation atom {atom_index} must contain only episodes")
    episodes_raw = atom["episodes"]
    if (
        not isinstance(episodes_raw, Sequence)
        or isinstance(episodes_raw, (str, bytes))
        or not episodes_raw
    ):
        raise ValueError(f"propagation atom {atom_index} must contain at least one episode")
    return [
        _validate_episode(episode, stratum=stratum, atom_index=atom_index, episode_index=index)
        for index, episode in enumerate(episodes_raw)
    ]


def _validate_episode(
    episode: Any,
    *,
    stratum: tuple[str, str, str],
    atom_index: int,
    episode_index: int,
) -> dict[str, Any]:
    label = f"propagation atom {atom_index} episode {episode_index}"
    if not isinstance(episode, Mapping) or set(episode) != {"metrics"}:
        raise ValueError(f"{label} must contain only metrics")
    metrics = episode["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(_METRICS):
        raise ValueError(f"{label} metrics must contain exactly the four propagation metrics")
    normalized: dict[str, Any] = {}
    for name in _METRICS:
        item = metrics[name]
        if not isinstance(item, Mapping) or set(item) != {
            "candidate",
            "challenger",
            "availability",
            "reason",
        }:
            raise ValueError(f"{label} {name} has an invalid field set")
        state = item["availability"]
        candidate = item["candidate"]
        challenger = item["challenger"]
        reason = item["reason"]
        if not isinstance(state, str) or state not in {
            "resolved",
            "unresolved",
            "not_applicable",
        }:
            raise ValueError(f"{label} {name}.availability is invalid")
        if state == "resolved":
            signed = name == "first_arrival_delay_s"
            normalized[name] = {
                "availability": state,
                "candidate": _finite_number(candidate, f"{label} {name}.candidate", signed=signed),
                "challenger": _finite_number(
                    challenger, f"{label} {name}.challenger", signed=signed
                ),
                "reason": None,
            }
        elif state in {"unresolved", "not_applicable"}:
            if candidate is not None or challenger is not None:
                raise ValueError(f"{label} masked {name} values must both be null")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{label} masked {name} requires a reason")
            if state == "not_applicable" and stratum[2] == "total":
                raise ValueError("total-component propagation metrics cannot be not_applicable")
            normalized[name] = {
                "availability": state,
                "candidate": None,
                "challenger": None,
                "reason": reason.strip(),
            }
        else:
            raise ValueError(f"{label} {name}.availability is invalid")
    return {"metrics": normalized}


def plan_g3_propagation_bootstrap_assurance(
    *,
    joint_acceptance_family: dict[str, Any],
    assumed_iid_scenarios_by_stratum: Mapping[tuple[str, str, str], Mapping[str, Any]],
    candidate_clusters_per_stratum: Sequence[int],
    effective_bandwidth_hz: float,
    bootstrap_draws: int = 1_000,
    assurance_replicates: int = 100,
    target_assurance: float = 0.80,
    monte_carlo_confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """Estimate propagation scorer assurance under explicit finite IID scenarios.

    Each assumed atom is one hypothetical independent scene cluster containing
    paired candidate/challenger episode metrics. Every simulated study is
    scored by ``evaluate_propagation_metrics`` under the shared family. Exact
    Monte Carlo lower bounds and a union bound control finite assurance
    simulation uncertainty across strata and searched candidate counts.
    """
    family = validate_g3_joint_acceptance_family(joint_acceptance_family)
    if family["schema"] not in {
        "loslab.g3_joint_acceptance_family.v2",
        "loslab.g3_joint_acceptance_family.v3",
        "loslab.g3_joint_acceptance_family.v4",
    }:
        raise ValueError(
            "propagation bootstrap assurance requires a joint-family v2, v3, or v4 roster"
        )
    expected_strata = {
        (str(item[0]), str(item[1]), str(item[2])) for item in family["propagation_expected_strata"]
    }
    per_claim_alpha, family_claim_count = family_alpha_for_claim(
        family,
        evaluator="propagation_metrics",
        expected_strata=sorted(expected_strata),
    )
    if (
        isinstance(effective_bandwidth_hz, bool)
        or not isinstance(effective_bandwidth_hz, (int, float))
        or not math.isfinite(float(effective_bandwidth_hz))
        or float(effective_bandwidth_hz) <= 0.0
    ):
        raise ValueError("effective_bandwidth_hz must be a finite positive number")
    candidates = list(candidate_clusters_per_stratum)
    if (
        not candidates
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 2
            for value in candidates
        )
        or candidates != sorted(set(candidates))
    ):
        raise ValueError("candidate_clusters_per_stratum must be unique ascending integers >= 2")
    for name, value, minimum in (
        ("bootstrap_draws", bootstrap_draws, 1_000),
        ("assurance_replicates", assurance_replicates, 100),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    target = _probability(target_assurance, "target_assurance")
    mc_confidence = _probability(monte_carlo_confidence, "monte_carlo_confidence")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scenarios = _validate_scenarios(
        assumed_iid_scenarios_by_stratum,
        expected_strata=expected_strata,
    )
    max_episodes = max(len(atom) for atoms, _ in scenarios.values() for atom in atoms)
    work = (
        len(expected_strata)
        * len(candidates)
        * assurance_replicates
        * bootstrap_draws
        * max(candidates)
        * max_episodes
        * len(_METRICS)
    )
    if work > _MAX_ASSURANCE_WORK:
        raise ValueError("requested propagation assurance exceeds the bounded work limit")
    max_study_episodes = len(expected_strata) * max(candidates) * max_episodes
    if max_study_episodes > _MAX_STUDY_EPISODES:
        raise ValueError("a single propagation assurance study exceeds the bounded size limit")

    mc_alpha = (1.0 - mc_confidence) / (len(expected_strata) * len(candidates))
    rng = np.random.default_rng(seed)
    candidate_reports: list[dict[str, Any]] = []
    for cluster_count in candidates:
        successes = dict.fromkeys(sorted(expected_strata), 0)
        for _ in range(assurance_replicates):
            records: list[dict[str, Any]] = []
            for stratum in sorted(expected_strata):
                atoms, probabilities = scenarios[stratum]
                sampled_atoms = rng.choice(len(atoms), size=cluster_count, p=probabilities)
                for cluster_index, atom_index in enumerate(sampled_atoms):
                    for episode_index, episode in enumerate(atoms[int(atom_index)]):
                        records.append(
                            {
                                "family_id": stratum[0],
                                "condition": stratum[1],
                                "component": stratum[2],
                                "cluster_id": f"cluster-{cluster_index:08d}",
                                "episode_id": f"episode-{episode_index:08d}",
                                "candidate_configuration_id": "candidate",
                                "challenger_configuration_id": "challenger",
                                **episode,
                            }
                        )
            report = evaluate_propagation_metrics(
                records,
                expected_strata=sorted(expected_strata),
                effective_bandwidth_hz=float(effective_bandwidth_hz),
                joint_acceptance_family=family,
                bootstrap_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                bootstrap_draws=bootstrap_draws,
            )
            for stratum_report in report["strata"]:
                stratum = (
                    stratum_report["family_id"],
                    stratum_report["condition"],
                    stratum_report["component"],
                )
                statuses = [item["criteria_status"] for item in stratum_report["metrics"]]
                successes[stratum] += int(
                    all(status in {"PASS", "NOT_APPLICABLE"} for status in statuses)
                )
        lower_bounds: list[float] = []
        stratum_reports = []
        for stratum, count in sorted(successes.items()):
            lower = (
                0.0
                if count == 0
                else float(beta.ppf(mc_alpha, count, assurance_replicates - count + 1))
            )
            lower_bounds.append(lower)
            stratum_reports.append(
                {
                    "family_id": stratum[0],
                    "condition": stratum[1],
                    "component": stratum[2],
                    "cluster_count": cluster_count,
                    "joint_pass_count": count,
                    "assurance_estimate": count / assurance_replicates,
                    "assurance_one_sided_exact_lower_bound": lower,
                    "monte_carlo_confidence_level_bonferroni": 1.0 - mc_alpha,
                }
            )
        joint_lower = max(0.0, 1.0 - sum(1.0 - value for value in lower_bounds))
        candidate_reports.append(
            {
                "candidate_clusters_per_stratum": cluster_count,
                "stratum_assurance": stratum_reports,
                "joint_assurance_lower_bound_union_bound": joint_lower,
                "meets_target_assurance": joint_lower >= target,
            }
        )

    selected = next(
        (
            item["candidate_clusters_per_stratum"]
            for item in candidate_reports
            if item["meets_target_assurance"]
        ),
        None,
    )
    scenario_record = [
        {
            "stratum": list(key),
            "atoms": atoms,
            "probabilities": probabilities.tolist(),
        }
        for key, (atoms, probabilities) in sorted(scenarios.items())
    ]
    scenario_digest = hashlib.sha256(
        json.dumps(scenario_record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "schema": "loslab.g3_propagation_bootstrap_assurance_planning.v1",
        "status": "SCENARIO_CONDITIONAL_PLANNING_ONLY_NOT_QUALIFICATION",
        "qualification_status": "NOT_QUALIFIED",
        "admission_possible": False,
        "scenario_provenance_verified": False,
        "joint_acceptance_family": {
            "family_id": family["family_id"],
            "family_sha256": family["family_sha256"],
            "familywise_alpha": family["familywise_alpha"],
            "claim_count": family_claim_count,
            "per_claim_alpha_bonferroni": per_claim_alpha,
            "per_claim_confidence_level": 1.0 - per_claim_alpha,
        },
        "candidate_designs": candidate_reports,
        "selected_clusters_per_stratum_among_evaluated_candidates": selected,
        "selected_candidate_is_smallest_tested_not_global_minimum": True,
        "target_assurance": target,
        "monte_carlo_confidence": mc_confidence,
        "monte_carlo_alpha_bonferroni": mc_alpha,
        "bootstrap_draws_per_study": bootstrap_draws,
        "assurance_replicates_per_candidate_and_stratum": assurance_replicates,
        "candidate_search_count": len(candidates),
        "independent_strata_count": len(expected_strata),
        "scenario_sha256": scenario_digest,
        "seed": seed,
        "complete_g3_sample_size": None,
        "sampling_units_are_independent_scene_clusters": True,
        "limitations": [
            "All paired episode metrics and IID cluster probabilities are hypothetical and unverified.",
            "Each simulated study is evaluated by the current propagation evaluator and shared-family alpha.",
            "Exact Monte Carlo lower bounds account for assurance simulation uncertainty; a union bound combines strata.",
            "Propagation clusters remain separate from material acquisition units, CSI clusters, and NR receiver clusters.",
            "This planner never qualifies propagation, G3, a scene family, or authorizes production RT.",
        ],
        "bounded_work_cell_count": work,
        "maximum_study_episode_count": max_study_episodes,
    }


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return result
