"""Score paired G3 NR receiver-impact outcomes without claiming qualification."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from evaluation.g3_joint_acceptance import (
    FAMILYWISE_ALPHA,
    family_alpha_for_claim,
    validate_g3_joint_acceptance_family,
)
from evaluation.g3_numeric import finite_real_number, finite_real_vector

_BLER_MARGIN = 0.01
_SNR10_MARGIN_DB = 0.5
_INTERVALS_PER_STRATUM = 4
MIN_INDEPENDENT_CLUSTERS_PER_STRATUM = 20
_MAX_BOOTSTRAP_INDEX_CELLS = 2_000_000
_MAX_BOOTSTRAP_OUTPUT_CELLS = 8_000_000


def _crossing_snr(snr_grid: np.ndarray, bler: np.ndarray) -> float | None:
    """Return a uniquely bracketed 10% BLER crossing using linear interpolation."""
    if bler.ndim != 1 or bler.size != snr_grid.size or not np.all(np.isfinite(bler)):
        return None
    if np.any((bler < 0.0) | (bler > 1.0)) or np.any(np.diff(bler) > 1e-12):
        return None
    exact = np.flatnonzero(np.isclose(bler, 0.1, rtol=0.0, atol=1e-12))
    if exact.size:
        return float(snr_grid[int(exact[0])]) if exact.size == 1 else None
    crossings = np.flatnonzero((bler[:-1] > 0.1) & (bler[1:] < 0.1))
    if crossings.size != 1:
        return None
    left = int(crossings[0])
    fraction = (0.1 - bler[left]) / (bler[left + 1] - bler[left])
    return float(snr_grid[left] + fraction * (snr_grid[left + 1] - snr_grid[left]))


def _percentile_interval(values: np.ndarray, *, alpha: float) -> list[float]:
    tail_percent = 50.0 * alpha
    low, high = np.percentile(values, [tail_percent, 100.0 - tail_percent])
    return [float(low), float(high)]


def evaluate_receiver_impact(
    records: Sequence[dict[str, Any]],
    *,
    expected_strata: Sequence[tuple[str, str]],
    candidate_config_id: str,
    challenger_config_id: str,
    snr_grid_db: Sequence[float],
    comparison_snr_db: float,
    joint_acceptance_family: dict[str, Any] | None = None,
    bootstrap_seed: int = 20261024,
    bootstrap_draws: int = 20_000,
) -> dict[str, Any]:
    """Evaluate paired PUSCH block outcomes by independent channel cluster.

    Each record contains a stratum, cluster, transport block, SNR, configuration,
    CRC error boolean, and decoder outcome. Every block must occur for both
    configurations at every frozen SNR. Clusters receive equal weight; bootstrap
    replicates resample clusters and paired transport blocks within clusters.
    """
    grid = finite_real_vector(snr_grid_db, "snr_grid_db")
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("snr_grid_db must contain at least two finite values")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("snr_grid_db must be strictly increasing")
    comparison_snr = finite_real_number(comparison_snr_db, "comparison_snr_db")
    if comparison_snr not in grid:
        raise ValueError("comparison_snr_db must be an exact member of the frozen grid")
    if (
        not isinstance(candidate_config_id, str)
        or not candidate_config_id.strip()
        or not isinstance(challenger_config_id, str)
        or not challenger_config_id.strip()
        or candidate_config_id == challenger_config_id
    ):
        raise ValueError("candidate and challenger configuration IDs must be distinct and nonempty")
    if (
        isinstance(bootstrap_seed, bool)
        or not isinstance(bootstrap_seed, int)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if (
        isinstance(bootstrap_draws, bool)
        or not isinstance(bootstrap_draws, int)
        or bootstrap_draws < 1000
    ):
        raise ValueError("bootstrap_draws must be an integer of at least 1000")
    bootstrap_output_cells = bootstrap_draws * (2 * int(grid.size) + 3)
    if bootstrap_output_cells > _MAX_BOOTSTRAP_OUTPUT_CELLS:
        raise ValueError(
            "receiver-impact bootstrap output arrays exceed the configured allocation limit "
            f"({_MAX_BOOTSTRAP_OUTPUT_CELLS} float64 cells)"
        )
    if not isinstance(expected_strata, Sequence) or isinstance(expected_strata, (str, bytes)):
        raise ValueError("expected_strata must be a sequence of family/condition pairs")
    normalized_expected: list[tuple[str, str]] = []
    for item in expected_strata:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("each expected stratum must be a family ID and LOS/NLOS pair")
        family, condition = item
        if (
            not isinstance(family, str)
            or not family.strip()
            or family != family.strip()
            or not isinstance(condition, str)
            or condition not in {"LOS", "NLOS"}
        ):
            raise ValueError("expected_strata must contain family IDs paired with LOS or NLOS")
        normalized_expected.append((family, condition))
    expected = set(normalized_expected)
    if not expected:
        raise ValueError("expected_strata must contain family IDs paired with LOS or NLOS")
    if len(expected) != len(normalized_expected):
        raise ValueError("expected_strata must not contain duplicates")
    if not records:
        raise ValueError("records must be nonempty")
    if joint_acceptance_family is None:
        familywise_claim_count = len(expected) * _INTERVALS_PER_STRATUM
        per_claim_alpha = FAMILYWISE_ALPHA / familywise_claim_count
        family_reference = None
        confidence_scope = (
            "Bonferroni family across candidate/challenger BLER, paired BLER difference, "
            "and SNR@10% shift intervals for every expected stratum"
        )
    else:
        normalized_family = validate_g3_joint_acceptance_family(joint_acceptance_family)
        per_claim_alpha, familywise_claim_count = family_alpha_for_claim(
            normalized_family,
            evaluator="receiver_impact",
            expected_strata=sorted(expected),
        )
        family_reference = {
            "family_id": normalized_family["family_id"],
            "family_sha256": normalized_family["family_sha256"],
        }
        confidence_scope = (
            "shared G3 family across material, complex-CSI, propagation, and receiver-impact claims"
        )
    per_claim_confidence_level = 1.0 - per_claim_alpha

    keyed: dict[tuple[str, str, str, str, float, str], dict[str, Any]] = {}
    observed: set[tuple[str, str]] = set()
    block_keys: set[tuple[str, str, str, str]] = set()
    outcomes: dict[tuple[tuple[str, str], str], Counter[str]] = defaultdict(Counter)
    for record in records:
        try:
            family = record["family_id"]
            condition = record["condition"]
            cluster = record["cluster_id"]
            block = record["transport_block_id"]
            snr = record["snr_db"]
            config = record["configuration_id"]
            error = record["crc_error"]
            outcome = record["decoder_outcome"]
        except (KeyError, TypeError) as exc:
            raise ValueError("every record must contain all receiver outcome fields") from exc
        if (
            not isinstance(family, str)
            or not family.strip()
            or not isinstance(condition, str)
            or condition not in {"LOS", "NLOS"}
        ):
            raise ValueError("record family_id and condition are invalid")
        if (
            not isinstance(cluster, str)
            or not cluster.strip()
            or not isinstance(block, str)
            or not block.strip()
        ):
            raise ValueError("cluster_id and transport_block_id must be nonempty strings")
        if any(value != value.strip() for value in (family, cluster, block)):
            raise ValueError(
                "family_id, cluster_id, and transport_block_id must be canonical strings"
            )
        try:
            snr_value = finite_real_number(snr, "snr_db")
        except ValueError as exc:
            raise ValueError("snr_db must be a finite real number") from exc
        if snr_value not in grid:
            raise ValueError("record SNR is outside the frozen grid")
        if not isinstance(config, str) or not config.strip():
            raise ValueError("record configuration_id must be a nonempty string")
        if config not in {candidate_config_id, challenger_config_id}:
            raise ValueError("record contains an unexpected configuration_id")
        if not isinstance(error, (bool, np.bool_)):
            raise ValueError("crc_error must be boolean")
        if not isinstance(outcome, str) or not outcome.strip():
            raise ValueError("decoder_outcome must be a nonempty string")
        stratum = (family, condition)
        observed.add(stratum)
        key = (family, condition, cluster, block, snr_value, config)
        if key in keyed:
            raise ValueError("duplicate configuration/block/SNR receiver outcome")
        keyed[key] = record
        block_keys.add((family, condition, cluster, block))
        outcomes[(stratum, config)][outcome] += 1
    if observed != expected:
        raise ValueError("observed family × LOS/NLOS strata do not match expected_strata")

    outcome_digest = hashlib.sha256()
    for (family, condition, cluster, block, snr, config), record in sorted(keyed.items()):
        canonical_record = {
            "family_id": family,
            "condition": condition,
            "cluster_id": cluster,
            "transport_block_id": block,
            "snr_db": snr,
            "configuration_id": config,
            "crc_error": bool(record["crc_error"]),
            "decoder_outcome": record["decoder_outcome"],
        }
        encoded_record = json.dumps(
            canonical_record, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        outcome_digest.update(encoded_record)
        outcome_digest.update(b"\n")
    outcome_records_sha256 = outcome_digest.hexdigest()

    result_strata: list[dict[str, Any]] = []
    rng = np.random.default_rng(bootstrap_seed)
    for family, condition in sorted(expected):
        cluster_blocks: dict[str, set[str]] = defaultdict(set)
        for fam, cond, cluster, block in block_keys:
            if (fam, cond) == (family, condition):
                cluster_blocks[cluster].add(block)
        clusters = sorted(cluster_blocks)
        if len(clusters) < MIN_INDEPENDENT_CLUSTERS_PER_STRATUM:
            raise ValueError(
                "each stratum requires at least "
                f"{MIN_INDEPENDENT_CLUSTERS_PER_STRATUM} independent clusters"
            )
        cluster_outcomes: dict[str, dict[str, np.ndarray]] = {
            config: {} for config in (candidate_config_id, challenger_config_id)
        }
        for cluster in clusters:
            blocks = cluster_blocks[cluster]
            for config in cluster_outcomes:
                values = np.empty((len(blocks), len(grid)), dtype=np.float64)
                for block_index, block in enumerate(sorted(blocks)):
                    for snr_index, snr in enumerate(grid):
                        key = (family, condition, cluster, block, float(snr), config)
                        if key not in keyed:
                            raise ValueError("candidate/challenger outcomes are not fully paired")
                        values[block_index, snr_index] = float(bool(keyed[key]["crc_error"]))
                cluster_outcomes[config][cluster] = values

        candidate_blocks = [cluster_outcomes[candidate_config_id][cluster] for cluster in clusters]
        challenger_blocks = [
            cluster_outcomes[challenger_config_id][cluster] for cluster in clusters
        ]
        candidate = np.stack([values.mean(axis=0) for values in candidate_blocks])
        challenger = np.stack([values.mean(axis=0) for values in challenger_blocks])
        candidate_point = candidate.mean(axis=0)
        challenger_point = challenger.mean(axis=0)
        comparison_index = int(np.flatnonzero(grid == comparison_snr_db)[0])
        candidate_crossing = _crossing_snr(grid, candidate_point)
        challenger_crossing = _crossing_snr(grid, challenger_point)

        max_blocks = max(len(values) for values in candidate_blocks)
        chunk_size = max(
            1,
            _MAX_BOOTSTRAP_INDEX_CELLS // max(max_blocks * len(grid), len(clusters)),
        )
        candidate_boot = np.empty((bootstrap_draws, len(grid)), dtype=np.float64)
        challenger_boot = np.empty_like(candidate_boot)
        delta_boot = np.empty(bootstrap_draws, dtype=np.float64)
        candidate_cross_boot = np.full(bootstrap_draws, np.nan)
        challenger_cross_boot = np.full(bootstrap_draws, np.nan)
        for start in range(0, bootstrap_draws, chunk_size):
            stop = min(start + chunk_size, bootstrap_draws)
            cluster_counts = rng.multinomial(
                len(clusters), np.full(len(clusters), 1.0 / len(clusters)), size=stop - start
            )
            candidate_values = np.zeros((stop - start, len(grid)), dtype=np.float64)
            challenger_values = np.zeros_like(candidate_values)
            for cluster_index, (candidate_rows, challenger_rows) in enumerate(
                zip(candidate_blocks, challenger_blocks, strict=True)
            ):
                block_count = len(candidate_rows)
                block_indices = rng.integers(0, block_count, size=(stop - start, block_count))
                cluster_multiplicity = cluster_counts[:, cluster_index, None]
                candidate_values += (
                    candidate_rows[block_indices].mean(axis=1) * cluster_multiplicity
                )
                challenger_values += (
                    challenger_rows[block_indices].mean(axis=1) * cluster_multiplicity
                )
            candidate_values /= len(clusters)
            challenger_values /= len(clusters)
            # Weighted means of CRC indicators are probabilities in [0, 1].
            # Roundoff can put exact boundary values just outside that domain,
            # which would incorrectly invalidate otherwise valid crossings.
            np.clip(candidate_values, 0.0, 1.0, out=candidate_values)
            np.clip(challenger_values, 0.0, 1.0, out=challenger_values)
            candidate_boot[start:stop] = candidate_values
            challenger_boot[start:stop] = challenger_values
            c_values = candidate_values
            h_values = challenger_values
            delta_boot[start:stop] = h_values[:, comparison_index] - c_values[:, comparison_index]
            for offset, (c_curve, h_curve) in enumerate(
                zip(c_values, h_values, strict=True), start=start
            ):
                c_crossing = _crossing_snr(grid, c_curve)
                h_crossing = _crossing_snr(grid, h_curve)
                candidate_cross_boot[offset] = np.nan if c_crossing is None else c_crossing
                challenger_cross_boot[offset] = np.nan if h_crossing is None else h_crossing

        delta = float(challenger_point[comparison_index] - candidate_point[comparison_index])
        delta_ci = [
            float(value)
            for value in np.clip(_percentile_interval(delta_boot, alpha=per_claim_alpha), -1.0, 1.0)
        ]
        candidate_bler_ci = [
            float(value)
            for value in np.clip(
                _percentile_interval(candidate_boot[:, comparison_index], alpha=per_claim_alpha),
                0.0,
                1.0,
            )
        ]
        challenger_bler_ci = [
            float(value)
            for value in np.clip(
                _percentile_interval(challenger_boot[:, comparison_index], alpha=per_claim_alpha),
                0.0,
                1.0,
            )
        ]
        candidate_half_width = max(
            candidate_point[comparison_index] - candidate_bler_ci[0],
            candidate_bler_ci[1] - candidate_point[comparison_index],
        )
        challenger_half_width = max(
            challenger_point[comparison_index] - challenger_bler_ci[0],
            challenger_bler_ci[1] - challenger_point[comparison_index],
        )
        crossing_valid = np.isfinite(candidate_cross_boot) & np.isfinite(challenger_cross_boot)
        crossing_valid_fraction = float(crossing_valid.mean())
        snr_shift: float | None = None
        snr_shift_ci: list[float] | None = None
        if candidate_crossing is not None and challenger_crossing is not None:
            snr_shift = float(challenger_crossing - candidate_crossing)
        if crossing_valid_fraction >= 0.95:
            snr_shift_ci = _percentile_interval(
                challenger_cross_boot[crossing_valid] - candidate_cross_boot[crossing_valid],
                alpha=per_claim_alpha,
            )
        snr_pass = (
            snr_shift is not None
            and snr_shift_ci is not None
            and abs(snr_shift) <= _SNR10_MARGIN_DB
            and snr_shift_ci[0] >= -_SNR10_MARGIN_DB
            and snr_shift_ci[1] <= _SNR10_MARGIN_DB
        )
        bler_pass = (
            abs(delta) <= _BLER_MARGIN
            and delta_ci[0] >= -_BLER_MARGIN
            and delta_ci[1] <= _BLER_MARGIN
            and candidate_half_width <= _BLER_MARGIN
            and challenger_half_width <= _BLER_MARGIN
        )
        result_strata.append(
            {
                "family_id": family,
                "condition": condition,
                "independent_clusters": len(clusters),
                "transport_block_counts_per_cluster": sorted(
                    {len(value) for value in cluster_blocks.values()}
                ),
                "comparison_snr_db": comparison_snr_db,
                "candidate_bler": float(candidate_point[comparison_index]),
                "challenger_bler": float(challenger_point[comparison_index]),
                "challenger_minus_candidate_bler": delta,
                "paired_bler_delta_confidence_interval": delta_ci,
                "candidate_bler_cluster_block_bootstrap_confidence_interval": candidate_bler_ci,
                "challenger_bler_cluster_block_bootstrap_confidence_interval": challenger_bler_ci,
                "candidate_bler_interval_half_width": float(candidate_half_width),
                "challenger_bler_interval_half_width": float(challenger_half_width),
                "candidate_snr_at_10_percent_bler_db": candidate_crossing,
                "challenger_snr_at_10_percent_bler_db": challenger_crossing,
                "challenger_minus_candidate_snr_at_10_percent_db": snr_shift,
                "snr_at_10_percent_shift_confidence_interval_db": snr_shift_ci,
                "interval_confidence_level": per_claim_confidence_level,
                "per_claim_alpha_bonferroni": per_claim_alpha,
                "valid_paired_crossing_bootstrap_fraction": crossing_valid_fraction,
                "bler_criteria_pass": bool(bler_pass),
                "snr_at_10_percent_criteria_pass": bool(snr_pass),
                "decoder_outcome_counts": {
                    candidate_config_id: dict(
                        sorted(outcomes[((family, condition), candidate_config_id)].items())
                    ),
                    challenger_config_id: dict(
                        sorted(outcomes[((family, condition), challenger_config_id)].items())
                    ),
                },
            }
        )

    numeric_pass = all(
        row["bler_criteria_pass"] and row["snr_at_10_percent_criteria_pass"]
        for row in result_strata
    )
    return {
        "schema": "g3_receiver_impact_diagnostic_v2",
        "status": "NUMERICAL_CRITERIA_ONLY_NOT_QUALIFIED"
        if numeric_pass
        else "NUMERICAL_CRITERIA_NOT_MET_OR_UNRESOLVED",
        "candidate_configuration_id": candidate_config_id,
        "challenger_configuration_id": challenger_config_id,
        "snr_grid_db": grid.tolist(),
        "comparison_snr_db": comparison_snr_db,
        "snr_at_10_percent_method": "unique monotone grid-bracketed crossing; linear interpolation; no extrapolation",
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_method": "equal-weight cluster resampling followed by paired block resampling within cluster",
        "bootstrap_output_float64_cells": bootstrap_output_cells,
        "bootstrap_output_float64_cell_limit": _MAX_BOOTSTRAP_OUTPUT_CELLS,
        "outcome_record_count": len(keyed),
        "outcome_records_sha256": outcome_records_sha256,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "familywise_claim_count": familywise_claim_count,
        "per_claim_alpha_bonferroni": per_claim_alpha,
        "per_claim_confidence_level": per_claim_confidence_level,
        "joint_acceptance_family": family_reference,
        "confidence_scope": confidence_scope,
        "strata": result_strata,
        "limitations": [
            "The caller must freeze the comparison SNR before examining candidate/challenger outcomes.",
            "Cluster IDs must represent independent channel realizations; this is not externally verified.",
            "CRC outcomes and decoder labels are preserved as supplied; this scorer does not validate the receiver implementation.",
            "A numerical pass cannot establish G5 qualification, physical validity, or authorize production RT generation.",
        ],
    }
