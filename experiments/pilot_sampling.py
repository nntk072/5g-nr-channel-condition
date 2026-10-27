"""Deterministic, input-scoped pilot sampling and outcome coverage reports."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_DESIGN_SCHEMA = "umcd.pilot_design.v1"
_CANDIDATE_SCHEMA = "umcd.pilot_candidate_roster.v1"
_SAMPLE_SCHEMA = "umcd.pilot_sample.v1"
_OUTCOME_SCHEMA = "umcd.pilot_outcomes.v1"
_AUDIT_SCHEMA = "umcd.pilot_audit.v1"
_DIFFICULTIES = frozenset({"low", "medium", "high"})
_PRIORS = frozenset({"natural_prevalence", "enriched_stress"})
_OUTCOMES = frozenset({"included", "no_path", "empty", "rejected", "failed"})
_DIGEST_LENGTH = 64


def _implementation_source_hashes() -> dict[str, str]:
    module_path = Path(__file__).resolve()
    cli_path = module_path.with_name("pilot_cli.py")
    return {
        "implementation_source_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "cli_source_sha256": hashlib.sha256(cli_path.read_bytes()).hexdigest(),
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a nonempty canonical string")
    return value


def _digest(value: Any, *, field: str) -> str:
    result = _text(value, field=field)
    if len(result) != _DIGEST_LENGTH or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return result


def _population_probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite probability in (0, 1]")
    try:
        probability = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be a finite probability in (0, 1]") from error
    if not math.isfinite(probability) or not 0 < probability <= 1:
        raise ValueError(f"{field} must be a finite probability in (0, 1]")
    return probability


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        row["channel_family"],
        row["antenna_topology"],
        row["difficulty"],
        row["prior_kind"],
    )


def _cell_id(row: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "channel_family": row["channel_family"],
            "antenna_topology": row["antenna_topology"],
            "difficulty": row["difficulty"],
            "prior_kind": row["prior_kind"],
        }
    )


def _validate_design(design: Any) -> dict[str, Any]:
    if not isinstance(design, dict) or set(design) != {
        "schema",
        "sampling_seed",
        "prior_provenance_sha256",
        "mandatory_channel_families",
        "mandatory_antenna_topologies",
        "mandatory_difficulties",
        "required_prior_kinds",
        "cells",
    }:
        raise ValueError("pilot design must contain exactly the v1 fields")
    if design["schema"] != _DESIGN_SCHEMA:
        raise ValueError("unsupported pilot design schema")
    seed = design["sampling_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**128:
        raise ValueError("sampling_seed must be a nonnegative integer below 2^128")
    _digest(design["prior_provenance_sha256"], field="prior_provenance_sha256")
    required_lists = {
        "mandatory_channel_families": None,
        "mandatory_antenna_topologies": None,
        "mandatory_difficulties": _DIFFICULTIES,
        "required_prior_kinds": _PRIORS,
    }
    for field, allowed in required_lists.items():
        values = design[field]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{field} must be a nonempty list")
        for value in values:
            if allowed is None:
                _text(value, field=field)
            elif value not in allowed:
                raise ValueError(f"{field} contains an unsupported value")
        if len(values) != len(set(values)):
            raise ValueError(f"{field} values must be unique")
    if not {"low", "high"}.issubset(design["mandatory_difficulties"]):
        raise ValueError("mandatory_difficulties must include low and high")
    if set(design["required_prior_kinds"]) != _PRIORS:
        raise ValueError("required_prior_kinds must include natural and enriched priors")
    cells = design["cells"]
    if not isinstance(cells, list) or not cells:
        raise ValueError("pilot design cells must be a nonempty list")
    keys: set[tuple[str, str, str, str]] = set()
    ids: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
            "channel_family",
            "antenna_topology",
            "difficulty",
            "prior_kind",
            "independent_groups",
            "target_population_probability",
        }:
            raise ValueError("each pilot cell must contain exactly the v1 fields")
        for field in ("channel_family", "antenna_topology"):
            _text(cell[field], field=f"cell.{field}")
        if cell["difficulty"] not in _DIFFICULTIES:
            raise ValueError("cell.difficulty must be low, medium, or high")
        if cell["prior_kind"] not in _PRIORS:
            raise ValueError("cell.prior_kind must be natural_prevalence or enriched_stress")
        target_probability = cell["target_population_probability"]
        if cell["prior_kind"] == "natural_prevalence":
            _population_probability(
                target_probability, field="natural-prevalence target probability"
            )
        elif target_probability is not None:
            raise ValueError("enriched-stress cells must not claim a population probability")
        count = cell["independent_groups"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("cell.independent_groups must be a positive integer")
        key = _cell_key(cell)
        identifier = _cell_id(cell)
        if key in keys or identifier in ids:
            raise ValueError("pilot design cells must be unique")
        keys.add(key)
        ids.add(identifier)
    natural_probability = sum(
        cell["target_population_probability"]
        for cell in cells
        if cell["prior_kind"] == "natural_prevalence"
    )
    if not math.isclose(natural_probability, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("natural-prevalence target probabilities must sum to one")
    if {cell["channel_family"] for cell in cells} != set(design["mandatory_channel_families"]):
        raise ValueError("pilot cells do not cover the mandatory channel-family roster exactly")
    if {cell["antenna_topology"] for cell in cells} != set(design["mandatory_antenna_topologies"]):
        raise ValueError("pilot cells do not cover the mandatory antenna-topology roster exactly")
    if {cell["difficulty"] for cell in cells} != set(design["mandatory_difficulties"]):
        raise ValueError("pilot cells do not cover the mandatory difficulty roster exactly")
    if {cell["prior_kind"] for cell in cells} != set(design["required_prior_kinds"]):
        raise ValueError("pilot cells do not cover the natural/enriched prior roster exactly")
    family_topology_pairs = {(cell["channel_family"], cell["antenna_topology"]) for cell in cells}
    for family, topology in family_topology_pairs:
        observed = {
            (cell["difficulty"], cell["prior_kind"])
            for cell in cells
            if (cell["channel_family"], cell["antenna_topology"]) == (family, topology)
        }
        required = {
            (difficulty, prior)
            for difficulty in design["mandatory_difficulties"]
            for prior in design["required_prior_kinds"]
        }
        if observed != required:
            raise ValueError(
                f"pilot cells must cross every mandatory difficulty and prior for "
                f"{family!r}/{topology!r}"
            )
    return design


def _validate_candidates(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, dict) or set(candidates) != {"schema", "candidates"}:
        raise ValueError("candidate roster must contain exactly the v1 fields")
    if candidates["schema"] != _CANDIDATE_SCHEMA or not isinstance(candidates["candidates"], list):
        raise ValueError("unsupported pilot candidate roster")
    rows = candidates["candidates"]
    if not rows:
        raise ValueError("pilot candidate roster must not be empty")
    ids: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "independent_group_id",
            "channel_family",
            "antenna_topology",
            "difficulty",
            "prior_kind",
            "source_record_sha256",
        }:
            raise ValueError("each pilot candidate must contain exactly the v1 fields")
        for field in ("candidate_id", "independent_group_id", "channel_family", "antenna_topology"):
            _text(row[field], field=f"candidate.{field}")
        if row["candidate_id"] in ids:
            raise ValueError("candidate_id values must be unique")
        ids.add(row["candidate_id"])
        if row["difficulty"] not in _DIFFICULTIES or row["prior_kind"] not in _PRIORS:
            raise ValueError("candidate difficulty/prior_kind is unsupported")
        _digest(row["source_record_sha256"], field="candidate.source_record_sha256")
        validated.append(dict(row))
    return validated


def build_pilot_sample(design: Any, candidate_roster: Any) -> dict[str, Any]:
    """Select independent groups by stable SHA-256 rank within frozen cells.

    The source roster must already reflect the declared natural or enriched
    sampling process. This function does not estimate deployment prevalence.
    """
    design = _validate_design(design)
    candidates = _validate_candidates(candidate_roster)
    roster_sha256 = _sha256(candidate_roster)
    selected: list[dict[str, str]] = []
    cells: list[dict[str, Any]] = []
    for specification in sorted(design["cells"], key=lambda item: _cell_id(item)):
        key = _cell_key(specification)
        eligible_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            if _cell_key(candidate) == key:
                eligible_by_group[candidate["independent_group_id"]].append(candidate)
        ranked_groups = sorted(
            eligible_by_group,
            key=lambda group: hashlib.sha256(
                f"{design['sampling_seed']}\0{_cell_id(specification)}\0{group}".encode()
            ).digest(),
        )
        chosen_groups = ranked_groups[: specification["independent_groups"]]
        cell_selected: list[dict[str, str]] = []
        for group in chosen_groups:
            options = sorted(eligible_by_group[group], key=lambda item: item["candidate_id"])
            choice = min(
                options,
                key=lambda candidate: hashlib.sha256(
                    f"{design['sampling_seed']}\0{_cell_id(specification)}\0{group}\0{candidate['candidate_id']}".encode()
                ).digest(),
            )
            entry = {
                "candidate_id": choice["candidate_id"],
                "independent_group_id": group,
                "cell_id": _cell_id(specification),
                "source_record_sha256": choice["source_record_sha256"],
            }
            selected.append(entry)
            cell_selected.append(entry)
        cells.append(
            {
                **{
                    field: specification[field]
                    for field in ("channel_family", "antenna_topology", "difficulty", "prior_kind")
                },
                "cell_id": _cell_id(specification),
                "requested_independent_groups": specification["independent_groups"],
                "target_population_probability": specification["target_population_probability"],
                "available_independent_groups": len(eligible_by_group),
                "selected_independent_groups": len(cell_selected),
                "shortfall": specification["independent_groups"] - len(cell_selected),
                "selected_candidate_ids": [entry["candidate_id"] for entry in cell_selected],
            }
        )
    selected_by_prior: dict[str, int] = defaultdict(int)
    for cell in cells:
        selected_by_prior[cell["prior_kind"]] += cell["selected_independent_groups"]
    for cell in cells:
        selected_total = selected_by_prior[cell["prior_kind"]]
        sample_share = (
            cell["selected_independent_groups"] / selected_total if selected_total else None
        )
        target_probability = cell["target_population_probability"]
        cell["sample_share_within_prior"] = sample_share
        cell["design_importance_weight"] = (
            target_probability / sample_share
            if target_probability is not None and sample_share is not None and sample_share > 0
            else None
        )
    complete = all(cell["shortfall"] == 0 for cell in cells)
    return {
        "schema": _SAMPLE_SCHEMA,
        **_implementation_source_hashes(),
        "status": "SAMPLE_ROSTER_COMPLETE" if complete else "INCOMPLETE_DESIGN",
        "qualification_status": "NOT_QUALIFIED",
        "sampling_algorithm": "sha256_rank_without_replacement_v1",
        "sampling_seed": design["sampling_seed"],
        "design_sha256": _sha256(design),
        "candidate_roster_sha256": roster_sha256,
        "prior_provenance_sha256": design["prior_provenance_sha256"],
        "candidate_count": len(candidates),
        "selected_candidate_count": len(selected),
        "selected_unique_independent_group_count": len(
            {item["independent_group_id"] for item in selected}
        ),
        "selected_candidates": selected,
        "cells": cells,
        "limitations": [
            "The supplied candidate roster is input-scoped and is not authenticated as complete or representative.",
            "Natural-prevalence claims require independent evidence that the source roster was sampled from the declared deployment prior.",
            "Enriched-stress selection is a coverage design and does not estimate natural prevalence.",
            "The sampler seed is not audited against experimental seed inventories and does not authorize a scientific run.",
            "This sampler launches no channel, receiver, RT, or model computation.",
        ],
    }


def _validate_outcomes(outcomes: Any) -> list[dict[str, Any]]:
    if not isinstance(outcomes, dict) or set(outcomes) != {"schema", "outcomes"}:
        raise ValueError("pilot outcomes must contain exactly the v1 fields")
    if outcomes["schema"] != _OUTCOME_SCHEMA or not isinstance(outcomes["outcomes"], list):
        raise ValueError("unsupported pilot outcomes schema")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in outcomes["outcomes"]:
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "outcome",
            "runtime_seconds",
            "storage_bytes",
            "result_sha256",
        }:
            raise ValueError("each pilot outcome must contain exactly the v1 fields")
        candidate_id = _text(row["candidate_id"], field="outcome.candidate_id")
        if candidate_id in seen:
            raise ValueError("pilot outcomes must contain at most one row per candidate")
        seen.add(candidate_id)
        if row["outcome"] not in _OUTCOMES:
            raise ValueError("pilot outcome value is unsupported")
        runtime = row["runtime_seconds"]
        if isinstance(runtime, bool) or not isinstance(runtime, (int, float)) or runtime < 0:
            raise ValueError("runtime_seconds must be a finite nonnegative number")
        try:
            runtime_value = float(runtime)
        except OverflowError as error:
            raise ValueError("runtime_seconds must be a finite nonnegative number") from error
        if not math.isfinite(runtime_value):
            raise ValueError("runtime_seconds must be a finite nonnegative number")
        storage = row["storage_bytes"]
        if (
            isinstance(storage, bool)
            or not isinstance(storage, int)
            or storage < 0
            or storage > 2**63 - 1
        ):
            raise ValueError("storage_bytes must be an integer in [0, 2^63-1]")
        _digest(row["result_sha256"], field="outcome.result_sha256")
        validated.append(dict(row))
    return validated


def _validate_sample(sample: Any) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "implementation_source_sha256",
        "cli_source_sha256",
        "status",
        "qualification_status",
        "sampling_algorithm",
        "sampling_seed",
        "design_sha256",
        "candidate_roster_sha256",
        "prior_provenance_sha256",
        "candidate_count",
        "selected_candidate_count",
        "selected_unique_independent_group_count",
        "selected_candidates",
        "cells",
        "limitations",
    }
    if not isinstance(sample, dict) or set(sample) != expected_fields:
        raise ValueError("pilot sample report has an invalid field roster")
    if sample["schema"] != _SAMPLE_SCHEMA or sample["qualification_status"] != "NOT_QUALIFIED":
        raise ValueError("unsupported or qualification-eligible pilot sample report")
    _digest(sample["implementation_source_sha256"], field="implementation_source_sha256")
    _digest(sample["cli_source_sha256"], field="cli_source_sha256")
    if sample["sampling_algorithm"] != "sha256_rank_without_replacement_v1":
        raise ValueError("unsupported pilot sampling algorithm")
    if sample["status"] not in {"SAMPLE_ROSTER_COMPLETE", "INCOMPLETE_DESIGN"}:
        raise ValueError("unsupported pilot sample status")
    for field in ("design_sha256", "candidate_roster_sha256", "prior_provenance_sha256"):
        _digest(sample[field], field=field)
    seed = sample["sampling_seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**128:
        raise ValueError("pilot sample sampling_seed is invalid")
    for field in (
        "candidate_count",
        "selected_candidate_count",
        "selected_unique_independent_group_count",
    ):
        count = sample[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"pilot sample {field} is invalid")
    if sample["candidate_count"] < sample["selected_candidate_count"]:
        raise ValueError("pilot sample selected more candidates than the roster")
    if not isinstance(sample["limitations"], list) or not sample["limitations"]:
        raise ValueError("pilot sample limitations must be a nonempty list")
    selected = sample["selected_candidates"]
    cells = sample["cells"]
    if not isinstance(selected, list) or not isinstance(cells, list) or not cells:
        raise ValueError("pilot sample selected candidates/cells are malformed")
    if len(selected) != sample["selected_candidate_count"]:
        raise ValueError("pilot sample selected-candidate count is inconsistent")
    selected_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    selected_ids: set[str] = set()
    for row in selected:
        if not isinstance(row, dict) or set(row) != {
            "candidate_id",
            "independent_group_id",
            "cell_id",
            "source_record_sha256",
        }:
            raise ValueError("pilot selected-candidate row has an invalid field roster")
        candidate_id = _text(row["candidate_id"], field="selected.candidate_id")
        _text(row["independent_group_id"], field="selected.independent_group_id")
        _digest(row["cell_id"], field="selected.cell_id")
        _digest(row["source_record_sha256"], field="selected.source_record_sha256")
        if candidate_id in selected_ids:
            raise ValueError("pilot sample contains duplicate selected candidate IDs")
        selected_ids.add(candidate_id)
        selected_by_cell[row["cell_id"]].append(row)
    observed_cell_ids: set[str] = set()
    globally_selected_groups = {row["independent_group_id"] for row in selected}
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
            "channel_family",
            "antenna_topology",
            "difficulty",
            "prior_kind",
            "cell_id",
            "requested_independent_groups",
            "target_population_probability",
            "available_independent_groups",
            "selected_independent_groups",
            "sample_share_within_prior",
            "design_importance_weight",
            "shortfall",
            "selected_candidate_ids",
        }:
            raise ValueError("pilot sample cell has an invalid field roster")
        cell_id = _text(cell["cell_id"], field="cell.cell_id")
        if cell_id in observed_cell_ids or cell_id != _cell_id(cell):
            raise ValueError("pilot sample cell identity is duplicate or inconsistent")
        observed_cell_ids.add(cell_id)
        for field in ("channel_family", "antenna_topology"):
            _text(cell[field], field=f"cell.{field}")
        if cell["difficulty"] not in _DIFFICULTIES or cell["prior_kind"] not in _PRIORS:
            raise ValueError("pilot sample cell factor is unsupported")
        target_probability = cell["target_population_probability"]
        if cell["prior_kind"] == "natural_prevalence":
            _population_probability(
                target_probability, field="sample natural-prior population probability"
            )
        elif target_probability is not None:
            raise ValueError("sample enriched prior must not declare a population probability")
        counts = [
            cell["requested_independent_groups"],
            cell["available_independent_groups"],
            cell["selected_independent_groups"],
            cell["shortfall"],
        ]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts
        ):
            raise ValueError("pilot sample cell counts must be nonnegative integers")
        requested, available, selected_count, shortfall = counts
        if requested == 0 or available < selected_count or selected_count > requested:
            raise ValueError("pilot sample cell counts are inconsistent")
        if shortfall != requested - selected_count:
            raise ValueError("pilot sample cell shortfall is inconsistent")
        ids = cell["selected_candidate_ids"]
        rows = selected_by_cell.get(cell_id, [])
        if not isinstance(ids, list) or len(rows) != selected_count:
            raise ValueError("pilot sample cell selected roster is inconsistent")
        for candidate_id in ids:
            _text(candidate_id, field="cell.selected_candidate_ids[]")
        if len(ids) != len(set(ids)) or set(ids) != {row["candidate_id"] for row in rows}:
            raise ValueError("pilot sample cell selected roster is inconsistent")
        groups = [row["independent_group_id"] for row in rows]
        if len(groups) != len(set(groups)):
            raise ValueError("pilot sample repeats an independent group within a cell")
    selected_by_prior: dict[str, int] = defaultdict(int)
    for cell in cells:
        selected_by_prior[cell["prior_kind"]] += cell["selected_independent_groups"]
    natural_probabilities = 0.0
    for cell in cells:
        target_probability = cell["target_population_probability"]
        sample_total = selected_by_prior[cell["prior_kind"]]
        selected_count = cell["selected_independent_groups"]
        expected_share = selected_count / sample_total if sample_total else None
        observed_share = cell["sample_share_within_prior"]
        if expected_share is None:
            if observed_share is not None:
                raise ValueError("empty pilot prior must have null sample shares")
        elif (
            isinstance(observed_share, bool)
            or not isinstance(observed_share, (int, float))
            or not math.isclose(observed_share, expected_share, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("pilot cell sample share is inconsistent")
        expected_weight = (
            target_probability / expected_share
            if target_probability is not None and expected_share is not None and expected_share > 0
            else None
        )
        observed_weight = cell["design_importance_weight"]
        if expected_weight is None:
            if observed_weight is not None:
                raise ValueError("cell without a target population probability has a weight")
        elif (
            isinstance(observed_weight, bool)
            or not isinstance(observed_weight, (int, float))
            or not math.isclose(observed_weight, expected_weight, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError("pilot cell design importance weight is inconsistent")
        if target_probability is not None:
            natural_probabilities += target_probability
    if not math.isclose(natural_probabilities, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("pilot sample natural-prior probabilities do not sum to one")
    if not set(selected_by_cell).issubset(observed_cell_ids):
        raise ValueError("pilot selected roster references an undeclared cell")
    if len(globally_selected_groups) != sample["selected_unique_independent_group_count"]:
        raise ValueError("pilot sample unique independent-group count is inconsistent")
    expected_status = (
        "SAMPLE_ROSTER_COMPLETE"
        if all(cell["shortfall"] == 0 for cell in cells)
        else "INCOMPLETE_DESIGN"
    )
    if sample["status"] != expected_status:
        raise ValueError("pilot sample status does not match cell shortfalls")
    return sample


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def audit_pilot_outcomes(sample: Any, outcomes: Any) -> dict[str, Any]:
    """Bind observed pilot outcomes and cost coverage to an exact sample roster."""
    sample = _validate_sample(sample)
    rows = _validate_outcomes(outcomes)
    selected = sample.get("selected_candidates")
    cells = sample.get("cells")
    if not isinstance(selected, list) or not isinstance(cells, list):
        raise ValueError("pilot sample report has malformed selected roster or cells")
    selected_by_id = {row["candidate_id"]: row for row in selected}
    if len(selected_by_id) != len(selected):
        raise ValueError("pilot sample report has duplicate candidate IDs")
    outcomes_by_id = {row["candidate_id"]: row for row in rows}
    extra = set(outcomes_by_id) - set(selected_by_id)
    if extra:
        raise ValueError("pilot outcomes include candidates outside the frozen sample")
    outcomes_sha256 = _sha256(outcomes)
    outcomes_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_id, outcome in outcomes_by_id.items():
        outcomes_by_cell[selected_by_id[candidate_id]["cell_id"]].append(outcome)
    cell_reports: list[dict[str, Any]] = []
    for cell in cells:
        observed = outcomes_by_cell[cell["cell_id"]]
        counts = Counter(row["outcome"] for row in observed)
        runtime = [float(row["runtime_seconds"]) for row in observed]
        storage = [float(row["storage_bytes"]) for row in observed]
        cell_reports.append(
            {
                **{
                    field: cell[field]
                    for field in (
                        "channel_family",
                        "antenna_topology",
                        "difficulty",
                        "prior_kind",
                        "cell_id",
                        "target_population_probability",
                        "sample_share_within_prior",
                        "design_importance_weight",
                    )
                },
                "requested_independent_groups": cell["requested_independent_groups"],
                "selected_independent_groups": cell["selected_independent_groups"],
                "target_population_probability": cell["target_population_probability"],
                "sample_share_within_prior": cell["sample_share_within_prior"],
                "design_importance_weight": cell["design_importance_weight"],
                "observed_candidate_count": len(observed),
                "not_run_count": cell["selected_independent_groups"] - len(observed),
                "outcome_counts": {outcome: counts[outcome] for outcome in sorted(_OUTCOMES)},
                "runtime_seconds": {
                    "total": float(sum(runtime)),
                    "p50": _percentile(runtime, 0.50),
                    "p90": _percentile(runtime, 0.90),
                    "p99": _percentile(runtime, 0.99),
                },
                "storage_bytes": {
                    "total": int(sum(storage)),
                    "p50": _percentile(storage, 0.50),
                    "p90": _percentile(storage, 0.90),
                    "p99": _percentile(storage, 0.99),
                },
            }
        )
    complete = sample.get("status") == "SAMPLE_ROSTER_COMPLETE" and all(
        row["not_run_count"] == 0 for row in cell_reports
    )
    return {
        "schema": _AUDIT_SCHEMA,
        **_implementation_source_hashes(),
        "status": "COMPLETE_DIAGNOSTIC" if complete else "INCOMPLETE_DIAGNOSTIC",
        "qualification_status": "NOT_QUALIFIED",
        "sample_report_sha256": _sha256(sample),
        "outcomes_sha256": outcomes_sha256,
        "selected_candidate_count": len(selected),
        "observed_candidate_count": len(rows),
        "not_run_candidate_count": len(selected) - len(rows),
        "outcome_counts": {
            outcome: sum(row["outcome"] == outcome for row in rows) for outcome in sorted(_OUTCOMES)
        },
        "runtime_seconds_total": float(sum(float(row["runtime_seconds"]) for row in rows)),
        "storage_bytes_total": sum(row["storage_bytes"] for row in rows),
        "cells": cell_reports,
        "limitations": [
            "This report covers only supplied sample and outcome manifests; it does not authenticate corpus completeness or prior representativeness.",
            "Observed runtime and storage costs are conditional on the recorded hardware, software, cache state, and artifact accounting supplied by the run producer.",
            "Outcome accounting does not establish that failures, no-path labels, or rejection reasons are scientifically correct.",
            "A complete diagnostic report is not a production-range decision, scientific qualification, or authorization for D1 generation or training.",
        ],
    }
