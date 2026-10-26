"""Scenario-conditional assurance planning for the G3 paired NR receiver scorer.

The planner samples hypothetical independent channel-cluster outcomes and
delegates every simulated study to the production receiver evaluator. This
keeps pairing, nested block/cluster resampling, Bonferroni intervals, crossing
validity, and numerical acceptance rules aligned by construction. It never
reads measurements and cannot qualify a receiver or dataset.
"""

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
from evaluation.g3_numeric import finite_real_vector
from evaluation.g3_receiver_impact import evaluate_receiver_impact

_MAX_ASSURANCE_WORK = 250_000_000
_MAX_STUDY_OUTCOME_CELLS = 2_000_000


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be in (0, 1)")
    return result


def _validate_scenarios(
    scenarios: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    expected_strata: set[tuple[str, str]],
    snr_count: int,
) -> dict[tuple[str, str], tuple[list[dict[str, list[list[bool]]]], np.ndarray]]:
    if not isinstance(scenarios, Mapping) or set(scenarios) != expected_strata:
        raise ValueError("receiver scenarios must exactly match the joint-family strata")
    result: dict[tuple[str, str], tuple[list[dict[str, list[list[bool]]]], np.ndarray]] = {}
    required = {"assumed_cluster_outcomes", "probabilities"}
    outcome_fields = {"candidate_crc_errors", "challenger_crc_errors"}
    for stratum, raw in scenarios.items():
        if (
            not isinstance(stratum, tuple)
            or len(stratum) != 2
            or any(not isinstance(part, str) or not part.strip() for part in stratum)
            or stratum[1] not in {"LOS", "NLOS"}
        ):
            raise ValueError("receiver scenario keys must be (family_id, LOS or NLOS) tuples")
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ValueError(
                "each receiver scenario must contain cluster outcomes and probabilities"
            )
        atoms_raw = raw["assumed_cluster_outcomes"]
        probabilities_raw = raw["probabilities"]
        if (
            not isinstance(atoms_raw, Sequence)
            or isinstance(atoms_raw, (str, bytes))
            or not atoms_raw
        ):
            raise ValueError("assumed_cluster_outcomes must be a nonempty sequence")
        probabilities = finite_real_vector(probabilities_raw, "receiver scenario probabilities")
        if len(probabilities) != len(atoms_raw):
            raise ValueError("receiver scenario probabilities must match the outcome atoms")
        atoms: list[dict[str, list[list[bool]]]] = []
        for atom in atoms_raw:
            if not isinstance(atom, Mapping) or set(atom) != outcome_fields:
                raise ValueError("each receiver cluster atom has an invalid field set")
            normalized: dict[str, list[list[bool]]] = {}
            block_count: int | None = None
            for field in sorted(outcome_fields):
                raw_matrix = atom[field]
                matrix = np.asarray(raw_matrix, dtype=object)
                if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != snr_count:
                    raise ValueError(
                        f"{field} must be a nonempty transport-block × SNR boolean matrix"
                    )
                if any(not isinstance(value, (bool, np.bool_)) for value in matrix.flat):
                    raise ValueError(f"{field} values must be booleans")
                if block_count is not None and matrix.shape[0] != block_count:
                    raise ValueError("candidate and challenger block counts must match per cluster")
                block_count = int(matrix.shape[0])
                normalized[field] = matrix.astype(bool).tolist()
            atoms.append(normalized)
        if (
            np.any(probabilities <= 0.0)
            or abs(float(probabilities.sum()) - 1.0) > 1e-9
        ):
            raise ValueError("receiver scenario probabilities must be positive and sum to one")
        result[stratum] = atoms, probabilities
    return result


def plan_g3_receiver_impact_bootstrap_assurance(
    *,
    joint_acceptance_family: dict[str, Any],
    assumed_iid_scenarios_by_stratum: Mapping[tuple[str, str], Mapping[str, Any]],
    candidate_clusters_per_stratum: Sequence[int],
    snr_grid_db: Sequence[float],
    comparison_snr_db: float,
    bootstrap_replicates: int = 1_000,
    assurance_replicates: int = 100,
    target_assurance: float = 0.80,
    monte_carlo_confidence: float = 0.95,
    seed: int = 1,
) -> dict[str, Any]:
    """Estimate receiver-criterion assurance under explicit finite IID scenarios.

    Scenario atoms retain paired, per-transport-block CRC outcomes over the
    frozen SNR grid. Each assurance replicate samples independent channel
    clusters from those assumptions and invokes ``evaluate_receiver_impact``
    with the shared family, so all receiver intervals and point/precision rules
    are identical to the evaluator. The union bound combines strata; exact
    one-sided binomial lower bounds account for finite Monte Carlo replication.
    """
    family = validate_g3_joint_acceptance_family(joint_acceptance_family)
    if family["schema"] not in {
        "loslab.g3_joint_acceptance_family.v3",
        "loslab.g3_joint_acceptance_family.v4",
    }:
        raise ValueError("receiver bootstrap assurance requires a joint-family v3 or v4 roster")
    expected_strata = {
        (str(item[0]), str(item[1])) for item in family["receiver_impact_expected_strata"]
    }
    per_claim_alpha, family_claim_count = family_alpha_for_claim(
        family,
        evaluator="receiver_impact",
        expected_strata=sorted(expected_strata),
    )
    grid = finite_real_vector(snr_grid_db, "snr_grid_db")
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("snr_grid_db must contain at least two finite values")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("snr_grid_db must be strictly increasing")
    if (
        isinstance(comparison_snr_db, bool)
        or not isinstance(comparison_snr_db, (int, float))
        or not math.isfinite(float(comparison_snr_db))
        or float(comparison_snr_db) not in grid
    ):
        raise ValueError("comparison_snr_db must be an exact member of the frozen grid")
    candidates = list(candidate_clusters_per_stratum)
    if (
        not candidates
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 20
            for value in candidates
        )
        or candidates != sorted(set(candidates))
    ):
        raise ValueError("candidate_clusters_per_stratum must be unique ascending integers >= 20")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < 1_000
    ):
        raise ValueError("bootstrap_replicates must be an integer >= 1000")
    if (
        isinstance(assurance_replicates, bool)
        or not isinstance(assurance_replicates, int)
        or assurance_replicates < 100
    ):
        raise ValueError("assurance_replicates must be an integer >= 100")
    target = _probability(target_assurance, "target_assurance")
    mc_confidence = _probability(monte_carlo_confidence, "monte_carlo_confidence")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    scenarios = _validate_scenarios(
        assumed_iid_scenarios_by_stratum,
        expected_strata=expected_strata,
        snr_count=int(grid.size),
    )
    max_blocks = max(
        len(atom["candidate_crc_errors"]) for atoms, _ in scenarios.values() for atom in atoms
    )
    work_cells = (
        len(expected_strata)
        * len(candidates)
        * assurance_replicates
        * bootstrap_replicates
        * max(candidates)
        * max_blocks
        * int(grid.size)
    )
    if work_cells > _MAX_ASSURANCE_WORK:
        raise ValueError("requested receiver assurance exceeds the bounded work limit")
    if max(candidates) * max_blocks * int(grid.size) > _MAX_STUDY_OUTCOME_CELLS:
        raise ValueError("a single receiver assurance study exceeds the bounded size limit")

    alpha_per_stratum_bound = (1.0 - mc_confidence) / (len(expected_strata) * len(candidates))
    rng = np.random.default_rng(seed)
    candidate_reports: list[dict[str, Any]] = []
    for cluster_count in candidates:
        per_stratum_successes = dict.fromkeys(sorted(expected_strata), 0)
        for study_index in range(assurance_replicates):
            records: list[dict[str, Any]] = []
            for stratum in sorted(expected_strata):
                atoms, probabilities = scenarios[stratum]
                sampled = rng.choice(len(atoms), size=cluster_count, p=probabilities)
                for cluster_index, atom_index in enumerate(sampled):
                    atom = atoms[int(atom_index)]
                    candidate = atom["candidate_crc_errors"]
                    challenger = atom["challenger_crc_errors"]
                    for block_index in range(len(candidate)):
                        for snr_index, snr in enumerate(grid):
                            common = {
                                "family_id": stratum[0],
                                "condition": stratum[1],
                                "cluster_id": f"cluster-{cluster_index:08d}",
                                "transport_block_id": f"block-{block_index:08d}",
                                "snr_db": float(snr),
                            }
                            records.append(
                                {
                                    **common,
                                    "configuration_id": "candidate",
                                    "crc_error": candidate[block_index][snr_index],
                                    "decoder_outcome": "crc_error"
                                    if candidate[block_index][snr_index]
                                    else "crc_pass",
                                }
                            )
                            records.append(
                                {
                                    **common,
                                    "configuration_id": "challenger",
                                    "crc_error": challenger[block_index][snr_index],
                                    "decoder_outcome": "crc_error"
                                    if challenger[block_index][snr_index]
                                    else "crc_pass",
                                }
                            )
            report = evaluate_receiver_impact(
                records,
                expected_strata=sorted(expected_strata),
                candidate_config_id="candidate",
                challenger_config_id="challenger",
                snr_grid_db=grid.tolist(),
                comparison_snr_db=float(comparison_snr_db),
                joint_acceptance_family=family,
                bootstrap_seed=int(rng.integers(0, np.iinfo(np.int32).max)),
                bootstrap_draws=bootstrap_replicates,
            )
            for stratum_result in report["strata"]:
                stratum = (stratum_result["family_id"], stratum_result["condition"])
                passed = bool(
                    stratum_result["bler_criteria_pass"]
                    and stratum_result["snr_at_10_percent_criteria_pass"]
                )
                per_stratum_successes[stratum] += int(passed)
        stratum_reports: list[dict[str, Any]] = []
        lower_bounds: list[float] = []
        for stratum, successes in sorted(per_stratum_successes.items()):
            lower = (
                0.0
                if successes == 0
                else float(
                    beta.ppf(
                        alpha_per_stratum_bound,
                        successes,
                        assurance_replicates - successes + 1,
                    )
                )
            )
            lower_bounds.append(lower)
            stratum_reports.append(
                {
                    "family_id": stratum[0],
                    "condition": stratum[1],
                    "cluster_count": cluster_count,
                    "joint_pass_count": successes,
                    "assurance_estimate": successes / assurance_replicates,
                    "assurance_one_sided_exact_lower_bound": lower,
                    "monte_carlo_confidence_level_bonferroni": 1.0 - alpha_per_stratum_bound,
                }
            )
        joint_lower = max(0.0, 1.0 - sum(1.0 - value for value in lower_bounds))
        candidate_reports.append(
            {
                "candidate_clusters_per_stratum": cluster_count,
                "stratum_assurance": stratum_reports,
                "joint_assurance_lower_bound_union_bound": joint_lower,
                "monte_carlo_confidence_level_bonferroni": 1.0 - alpha_per_stratum_bound,
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
            "atoms": scenarios[key][0],
            "probabilities": scenarios[key][1].tolist(),
        }
        for key in sorted(scenarios)
    ]
    scenario_digest = hashlib.sha256(
        json.dumps(
            scenario_record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema": "loslab.g3_receiver_impact_bootstrap_assurance_planning.v1",
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
        "criteria": {
            "paired_bler_difference_absolute_margin": 0.01,
            "candidate_and_challenger_bler_interval_half_width_max": 0.01,
            "snr_at_10_percent_shift_absolute_margin_db": 0.5,
            "paired_crossing_bootstrap_valid_fraction_min": 0.95,
        },
        "candidate_designs": candidate_reports,
        "selected_clusters_per_stratum_among_evaluated_candidates": selected,
        "target_assurance": target,
        "monte_carlo_confidence": mc_confidence,
        "monte_carlo_alpha_bonferroni": alpha_per_stratum_bound,
        "bootstrap_replicates_per_study": bootstrap_replicates,
        "assurance_replicates_per_candidate_and_stratum": assurance_replicates,
        "candidate_search_count": len(candidates),
        "independent_strata_count": len(expected_strata),
        "scenario_sha256": scenario_digest,
        "seed": seed,
        "complete_g3_sample_size": None,
        "sampling_units_are_scene_clusters": True,
        "limitations": [
            "All per-block channel outcomes and IID cluster probabilities are hypothetical and unverified.",
            "Assurance is conditional on the supplied finite-support scenario and the current receiver evaluator implementation.",
            "The exact lower confidence bounds account for finite assurance Monte Carlo draws; a union bound combines the declared strata.",
            "This planner never qualifies NR receiver behavior, G3, a scene family, or authorizes production RT generation.",
        ],
    }
