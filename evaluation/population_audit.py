"""Expected-population accounting for inclusion, omission, and failure outcomes."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_ROSTER_SCHEMA = "loslab.expected_population_roster.v1"
_DISPOSITIONS = frozenset({"included", "excluded", "rejected", "no_path", "empty"})
_ROSTER_FIELDS = frozenset({"schema", "roster_id", "units"})
_UNIT_FIELDS = frozenset(
    {"unit_id", "family_id", "domain_id", "independent_group_id", "disposition", "reason"}
)


def validate_expected_population_roster(roster: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an exact, caller-frozen sample roster and outcome dispositions."""

    if not isinstance(roster, Mapping) or set(roster) != _ROSTER_FIELDS:
        raise ValueError("population roster must have exactly schema, roster_id, and units")
    if roster["schema"] != _ROSTER_SCHEMA:
        raise ValueError(f"population roster schema must be {_ROSTER_SCHEMA!r}")
    roster_id = roster["roster_id"]
    if not isinstance(roster_id, str) or not roster_id or roster_id != roster_id.strip():
        raise ValueError("population roster roster_id must be a canonical nonempty string")
    units = roster["units"]
    if not isinstance(units, list) or not units:
        raise ValueError("population roster units must be a nonempty array")
    seen: set[str] = set()
    canonical_units = []
    for index, unit in enumerate(units):
        if not isinstance(unit, Mapping) or set(unit) != _UNIT_FIELDS:
            raise ValueError(f"population roster unit {index} has an invalid field roster")
        values = dict(unit)
        for field in ("unit_id", "family_id", "domain_id", "independent_group_id"):
            value = values[field]
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"population roster unit {index} {field} must be canonical")
        if values["unit_id"] in seen:
            raise ValueError(f"duplicate expected population unit ID: {values['unit_id']}")
        seen.add(values["unit_id"])
        if values["disposition"] not in _DISPOSITIONS:
            raise ValueError(f"population roster unit {index} has an unsupported disposition")
        reason = values["reason"]
        if values["disposition"] == "included":
            if reason is not None:
                raise ValueError("included population units must have a null reason")
        elif not isinstance(reason, str) or not reason or reason != reason.strip():
            raise ValueError("non-included population units require a canonical reason")
        canonical_units.append(values)
    return {"schema": _ROSTER_SCHEMA, "roster_id": roster_id, "units": canonical_units}


def audit_expected_population(
    roster: Mapping[str, Any],
    *,
    observed_sample_ids: Sequence[str],
    observed_family_ids: Sequence[str],
    observed_domain_ids: Sequence[str],
    observed_group_ids: Sequence[str],
) -> dict[str, Any]:
    """Compare observed dataset rows with a frozen roster and report selection gaps.

    Every observed row must have metadata ``sample_id`` that joins to one expected
    ``included`` unit. Source family, domain, and group values must agree exactly.
    Non-included outcomes remain visible in the report and are not treated as
    missing rows. This verifies a declared roster, not its completeness or truth.
    """

    frozen = validate_expected_population_roster(roster)
    arrays = [
        np.asarray(observed_sample_ids, dtype=object),
        np.asarray(observed_family_ids, dtype=object),
        np.asarray(observed_domain_ids, dtype=object),
        np.asarray(observed_group_ids, dtype=object),
    ]
    if any(vector.ndim != 1 for vector in arrays) or any(
        vector.shape != arrays[0].shape for vector in arrays[1:]
    ):
        raise ValueError("observed sample, family, domain, and group IDs must align")
    if any(
        not isinstance(value, str) or not value or value != value.strip()
        for vector in arrays
        for value in vector.tolist()
    ):
        raise ValueError("observed sample, family, domain, and group IDs must be canonical")

    sample_ids, family_ids, domain_ids, group_ids = (vector.astype(str) for vector in arrays)
    expected = {unit["unit_id"]: unit for unit in frozen["units"]}
    observed_indices: dict[str, int] = {}
    duplicate_observed: list[str] = []
    unrostered: list[str] = []
    unexpected_disposition: list[str] = []
    identity_mismatches: list[dict[str, str]] = []
    for index, sample_id in enumerate(sample_ids.tolist()):
        if sample_id in observed_indices:
            duplicate_observed.append(sample_id)
            continue
        observed_indices[sample_id] = index
        unit = expected.get(sample_id)
        if unit is None:
            unrostered.append(sample_id)
            continue
        if unit["disposition"] != "included":
            unexpected_disposition.append(sample_id)
        for expected_field, observed_value in (
            ("family_id", family_ids[index]),
            ("domain_id", domain_ids[index]),
            ("independent_group_id", group_ids[index]),
        ):
            if unit[expected_field] != observed_value:
                identity_mismatches.append(
                    {
                        "unit_id": sample_id,
                        "field": expected_field,
                        "expected": unit[expected_field],
                        "observed": observed_value,
                    }
                )

    missing_included = sorted(
        unit["unit_id"]
        for unit in frozen["units"]
        if unit["disposition"] == "included" and unit["unit_id"] not in observed_indices
    )
    disposition_counts = Counter(unit["disposition"] for unit in frozen["units"])
    roster_digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()
    observed_digest_payload = [
        {
            "sample_id": sample_id,
            "family_id": family_id,
            "domain_id": domain_id,
            "independent_group_id": group_id,
        }
        for sample_id, family_id, domain_id, group_id in zip(
            sample_ids.tolist(),
            family_ids.tolist(),
            domain_ids.tolist(),
            group_ids.tolist(),
            strict=True,
        )
    ]
    observed_digest = hashlib.sha256(
        json.dumps(
            observed_digest_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()

    def stratified_counts(field: str) -> list[dict[str, Any]]:
        values = sorted({unit[field] for unit in frozen["units"]})
        output = []
        for value in values:
            rows = [unit for unit in frozen["units"] if unit[field] == value]
            row_counts = Counter(unit["disposition"] for unit in rows)
            output.append(
                {
                    "stratum_id": value,
                    "expected_unit_count": len(rows),
                    "included_unit_count": row_counts["included"],
                    "excluded_unit_count": row_counts["excluded"],
                    "rejected_unit_count": row_counts["rejected"],
                    "no_path_unit_count": row_counts["no_path"],
                    "empty_unit_count": row_counts["empty"],
                }
            )
        return output

    complete = not (
        missing_included
        or duplicate_observed
        or unrostered
        or unexpected_disposition
        or identity_mismatches
    )
    return {
        "schema": "loslab.expected_population_audit.v1",
        "status": "ROSTER_MATCH" if complete else "ROSTER_MISMATCH",
        "roster_id": frozen["roster_id"],
        "roster_sha256": roster_digest,
        "observed_identity_roster_sha256": observed_digest,
        "expected_unit_count": len(frozen["units"]),
        "observed_row_count": int(len(sample_ids)),
        "expected_disposition_counts": {
            disposition: int(disposition_counts[disposition])
            for disposition in sorted(_DISPOSITIONS)
        },
        "included_missing_unit_ids": missing_included,
        "duplicate_observed_sample_ids": sorted(set(duplicate_observed)),
        "unrostered_observed_sample_ids": sorted(set(unrostered)),
        "observed_for_nonincluded_unit_ids": sorted(set(unexpected_disposition)),
        "identity_mismatches": sorted(
            identity_mismatches, key=lambda row: (row["unit_id"], row["field"])
        ),
        "by_family": stratified_counts("family_id"),
        "by_domain": stratified_counts("domain_id"),
        "roster_source": None,
        "dataset_inputs": [],
        "limitations": [
            "The roster is caller-declared; this audit does not prove it covers the target population.",
            "Disposition reasons are declarations and are not independently verified.",
            "Group identifiers are checked for identity consistency, not physical independence.",
        ],
    }
