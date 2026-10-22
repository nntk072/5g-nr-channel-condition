"""Deterministic, scene-grouped data splits with immutable lock files."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class IndependentSplit:
    """A versioned assignment of samples and groups to independent splits."""

    sample_to_split: Mapping[str, str]
    group_to_split: Mapping[str, str]
    seed: str
    group_field: str

    def __post_init__(self) -> None:
        if not self.sample_to_split or not self.group_to_split:
            raise ValueError("split assignments must not be empty")
        if not self.seed.strip() or not self.group_field.strip():
            raise ValueError("seed and group_field must be non-empty")
        allowed = {"train", "validation", "calibration", "test"}
        if set(self.sample_to_split.values()) - allowed:
            raise ValueError("split names must be train, validation, calibration, or test")
        if set(self.group_to_split.values()) - allowed:
            raise ValueError("split names must be train, validation, calibration, or test")
        object.__setattr__(self, "sample_to_split", MappingProxyType(dict(self.sample_to_split)))
        object.__setattr__(self, "group_to_split", MappingProxyType(dict(self.group_to_split)))

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-compatible representation."""
        return {
            "schema_version": 1,
            "seed": self.seed,
            "group_field": self.group_field,
            "group_to_split": dict(sorted(self.group_to_split.items())),
            "sample_to_split": dict(sorted(self.sample_to_split.items())),
        }


def make_independent_split(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    group_field: str = "scene_group_id",
    sample_id_field: str = "sample_id",
    fractions: Mapping[str, float] | None = None,
) -> IndependentSplit:
    """Assign complete scene groups using a stable hash ordering.

    The rounded group counts approximate requested fractions. Groups are
    ordered by SHA256(seed, group ID), so the result is stable across Python
    versions and does not depend on process hash randomization.
    """
    proportions_input = dict(fractions or {"train": 0.7, "validation": 0.15, "test": 0.15})
    if set(proportions_input) not in (
        {"train", "validation", "test"},
        {"train", "validation", "calibration", "test"},
    ):
        raise ValueError(
            "fractions must define train/validation/test or train/validation/calibration/test"
        )
    split_order = tuple(
        name for name in ("train", "validation", "calibration", "test") if name in proportions_input
    )
    proportions = {name: float(proportions_input[name]) for name in split_order}
    if (
        any(not math.isfinite(value) or value <= 0 for value in proportions.values())
        or abs(sum(proportions.values()) - 1) > 1e-9
    ):
        raise ValueError("split fractions must be positive and sum to one")
    if not seed.strip() or not records:
        raise ValueError("seed and records must be non-empty")

    sample_groups: dict[str, str] = {}
    for record in records:
        sample_id = str(record.get(sample_id_field, "")).strip()
        group_id = str(record.get(group_field, "")).strip()
        if not sample_id or not group_id:
            raise ValueError(f"each record needs non-empty {sample_id_field} and {group_field}")
        if sample_id in sample_groups:
            raise ValueError(f"duplicate sample ID: {sample_id}")
        sample_groups[sample_id] = group_id

    groups = sorted(set(sample_groups.values()))
    if len(groups) < len(split_order):
        minimum = "three" if len(split_order) == 3 else "4"
        raise ValueError(
            f"at least {minimum} independent groups ({len(split_order)}) are required for "
            f"{len(split_order)}-way splits"
        )
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}\0{group}".encode()).digest(),
    )
    # Give each split at least one group, then allocate remaining slots by
    # largest-remainder rounding to keep small registries useful.
    exact = {name: len(groups) * value for name, value in proportions.items()}
    counts = {name: max(1, int(exact[name])) for name in proportions}
    while sum(counts.values()) > len(groups):
        candidates = [name for name in counts if counts[name] > 1]
        name = max(candidates, key=lambda key: counts[key] - exact[key])
        counts[name] -= 1
    while sum(counts.values()) < len(groups):
        name = max(proportions, key=lambda key: exact[key] - counts[key])
        counts[name] += 1

    group_to_split: dict[str, str] = {}
    offset = 0
    for name in split_order:
        for group_id in ordered[offset : offset + counts[name]]:
            group_to_split[group_id] = name
        offset += counts[name]
    sample_to_split = {
        sample_id: group_to_split[group_id] for sample_id, group_id in sample_groups.items()
    }
    return IndependentSplit(sample_to_split, group_to_split, seed, group_field)


def audit_group_leakage(split: IndependentSplit, records: Sequence[Mapping[str, Any]]) -> None:
    """Raise on duplicate/missing samples or groups crossing split boundaries."""
    observed: dict[str, str] = {}
    observed_samples: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        group_id = str(record.get(split.group_field, ""))
        if sample_id in observed_samples:
            raise ValueError(f"duplicate sample ID in audit records: {sample_id}")
        observed_samples.add(sample_id)
        if sample_id not in split.sample_to_split:
            raise ValueError(f"sample is absent from frozen split: {sample_id}")
        expected = split.sample_to_split[sample_id]
        if group_id not in split.group_to_split or split.group_to_split[group_id] != expected:
            raise ValueError(
                f"scene-group leakage or changed group identity for sample {sample_id}"
            )
        if group_id in observed and observed[group_id] != expected:
            raise ValueError(f"group appears in multiple splits: {group_id}")
        observed[group_id] = expected
    if set(observed) != set(split.group_to_split):
        raise ValueError("records do not cover exactly the groups in the frozen split")
    if observed_samples != set(split.sample_to_split):
        raise ValueError("records do not cover exactly the samples in the frozen split")


def freeze_split_lock(path: str | Path, split: IndependentSplit) -> str:
    """Create an immutable JSON lock; return its SHA256 or verify existing lock."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(split.to_dict(), indent=2, sort_keys=True) + "\n"
    encoded = payload.encode()
    digest = hashlib.sha256(encoded).hexdigest()
    if destination.exists():
        existing = destination.read_bytes()
        if existing != encoded:
            raise FileExistsError(f"refusing to modify existing split lock: {destination}")
        return hashlib.sha256(existing).hexdigest()

    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temp:
            temp.write(encoded)
            temp.flush()
            os.fsync(temp.fileno())
            temp_path = temp.name
        # Hard-link publication is atomic and fails if another writer won.
        os.link(temp_path, destination)
    except FileExistsError:
        if destination.read_bytes() != encoded:
            raise FileExistsError(f"refusing to modify existing split lock: {destination}")
    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)
    return digest
