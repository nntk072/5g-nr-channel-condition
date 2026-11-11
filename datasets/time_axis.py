"""Strict real-valued time-axis validation shared by dataset contracts."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Iterable


def canonical_time_axis(values: Iterable[object], *, name: str) -> tuple[float, ...]:
    """Return finite, strictly increasing real timestamps without coercing aliases."""
    try:
        raw_times = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of real timestamps") from error
    if not raw_times:
        raise ValueError(f"{name} must be a non-empty finite time axis")
    if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_times):
        raise TypeError(f"{name} values must be real numeric timestamps")
    try:
        times = tuple(float(value) for value in raw_times)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} values must be finite real timestamps") from error
    if any(not math.isfinite(value) for value in times):
        raise ValueError(f"{name} values must be finite real timestamps")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return times


@dataclass(frozen=True, slots=True)
class TimeAxisAlignment:
    """One-to-one timestamp matches from a reference axis to an observed axis."""

    reference_times_s: tuple[float, ...]
    observed_times_s: tuple[float, ...]
    tolerance_s: float
    observed_indices: tuple[int | None, ...]
    residuals_s: tuple[float | None, ...]
    unmatched_observed_indices: tuple[int, ...]

    @property
    def complete(self) -> bool:
        """Whether every reference timestamp has an observed match."""
        return all(index is not None for index in self.observed_indices)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible report retaining unmatched samples and residuals."""
        return {
            "schema": "umcd.time_axis_alignment.v1",
            "method": "unique_match_within_absolute_tolerance_no_interpolation",
            "reference_times_s": list(self.reference_times_s),
            "observed_times_s": list(self.observed_times_s),
            "tolerance_s": self.tolerance_s,
            "observed_indices": list(self.observed_indices),
            "residuals_s": list(self.residuals_s),
            "unmatched_reference_indices": [
                index
                for index, observed_index in enumerate(self.observed_indices)
                if observed_index is None
            ],
            "unmatched_observed_indices": list(self.unmatched_observed_indices),
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, value: object) -> TimeAxisAlignment:
        """Rebuild an alignment only when every stored claim matches replay."""
        required = {
            "schema",
            "method",
            "reference_times_s",
            "observed_times_s",
            "tolerance_s",
            "observed_indices",
            "residuals_s",
            "unmatched_reference_indices",
            "unmatched_observed_indices",
            "complete",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("time-axis alignment fields do not match v1 contract")
        if value.get("schema") != "umcd.time_axis_alignment.v1":
            raise ValueError("unsupported time-axis alignment schema")
        if value.get("method") != "unique_match_within_absolute_tolerance_no_interpolation":
            raise ValueError("unsupported time-axis alignment method")
        tolerance = value.get("tolerance_s")
        if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
            raise TypeError("time-axis alignment tolerance_s must be real")
        replay = align_time_axes(
            value.get("reference_times_s", ()),
            value.get("observed_times_s", ()),
            tolerance_s=tolerance,
        )
        expected_bytes = json.dumps(
            replay.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        observed_bytes = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if expected_bytes != observed_bytes:
            raise ValueError("time-axis alignment report differs from deterministic replay")
        return replay


@dataclass(frozen=True, slots=True)
class MultiSourceTimeAlignment:
    """Named source clocks aligned to one canonical episode frame grid."""

    reference_times_s: tuple[float, ...]
    source_alignments: tuple[tuple[str, TimeAxisAlignment], ...]

    def __post_init__(self) -> None:
        reference = canonical_time_axis(self.reference_times_s, name="reference_times_s")
        rows = tuple(self.source_alignments)
        if not rows:
            raise ValueError("source_alignments must contain at least one named source")
        names: list[str] = []
        normalized: list[tuple[str, TimeAxisAlignment]] = []
        for item in rows:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError("source_alignments entries must be (source_id, alignment) pairs")
            source_id, alignment = item
            if (
                not isinstance(source_id, str)
                or not source_id.strip()
                or source_id != source_id.strip()
            ):
                raise ValueError("source IDs must be non-empty canonical strings")
            if not isinstance(alignment, TimeAxisAlignment):
                raise TypeError("every source alignment must be a TimeAxisAlignment")
            replay = TimeAxisAlignment.from_dict(alignment.to_dict())
            if replay.reference_times_s != reference:
                raise ValueError("all source alignments must use the shared reference axis")
            names.append(source_id)
            normalized.append((source_id, replay))
        if len(set(names)) != len(names):
            raise ValueError("source IDs must be unique")
        normalized.sort(key=lambda item: item[0])
        object.__setattr__(self, "reference_times_s", reference)
        object.__setattr__(self, "source_alignments", tuple(normalized))

    @property
    def complete(self) -> bool:
        """Whether every named source supplies every reference frame."""
        return all(alignment.complete for _, alignment in self.source_alignments)

    @property
    def sha256(self) -> str:
        """Canonical digest used to bind generated channels to this alignment."""
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return all source matches, residuals, missing frames, and extra samples."""
        return {
            "schema": "umcd.multi_source_time_alignment.v1",
            "method": "independent_unique_match_per_source_no_interpolation",
            "reference_times_s": list(self.reference_times_s),
            "sources": [
                {"source_id": source_id, "alignment": alignment.to_dict()}
                for source_id, alignment in self.source_alignments
            ],
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, value: object) -> MultiSourceTimeAlignment:
        """Rebuild a named-source report, rejecting altered or unsupported claims."""
        if not isinstance(value, dict) or set(value) != {
            "schema",
            "method",
            "reference_times_s",
            "sources",
            "complete",
        }:
            raise ValueError("multi-source alignment fields do not match v1 contract")
        if (
            value.get("schema") != "umcd.multi_source_time_alignment.v1"
            or value.get("method") != "independent_unique_match_per_source_no_interpolation"
        ):
            raise ValueError("unsupported multi-source alignment schema or method")
        sources = value.get("sources")
        if not isinstance(sources, list) or not sources:
            raise ValueError("multi-source alignment sources must be a non-empty list")
        rows: list[tuple[str, TimeAxisAlignment]] = []
        for source in sources:
            if not isinstance(source, dict) or set(source) != {"source_id", "alignment"}:
                raise ValueError("multi-source alignment entry has an invalid field roster")
            rows.append((source["source_id"], TimeAxisAlignment.from_dict(source["alignment"])))
        report = cls(tuple(value["reference_times_s"]), tuple(rows))
        expected = json.dumps(report.to_dict(), sort_keys=True, separators=(",", ":"))
        actual = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if expected != actual:
            raise ValueError("multi-source alignment differs from deterministic replay")
        return report


def align_named_time_axes(
    reference_times_s: Iterable[object],
    observed_time_axes: Mapping[str, Iterable[object]],
    *,
    tolerance_s: Real,
    require_complete: bool = False,
) -> MultiSourceTimeAlignment:
    """Align each named source clock independently to one episode axis.

    Samples are matched one-to-one within the stated absolute tolerance. No
    interpolation, clock fitting, or cross-source sample reuse is performed.
    """
    if not isinstance(observed_time_axes, Mapping) or not observed_time_axes:
        raise ValueError("observed_time_axes must be a non-empty source mapping")
    if any(not isinstance(source_id, str) for source_id in observed_time_axes):
        raise TypeError("observed_time_axes keys must be source ID strings")
    reference = canonical_time_axis(reference_times_s, name="reference_times_s")
    rows = tuple(
        (
            source_id,
            align_time_axes(
                reference,
                observed_times_s,
                tolerance_s=tolerance_s,
                require_complete=require_complete,
            ),
        )
        for source_id, observed_times_s in sorted(observed_time_axes.items())
    )
    return MultiSourceTimeAlignment(reference, rows)


def align_time_axes(
    reference_times_s: Iterable[object],
    observed_times_s: Iterable[object],
    *,
    tolerance_s: Real,
    require_complete: bool = False,
) -> TimeAxisAlignment:
    """Match two monotone time axes without interpolation or sample reuse.

    Each reference timestamp must have at most one observed timestamp within the
    caller-declared absolute tolerance. Missing matches are reported unless
    ``require_complete`` is true. An observed sample can satisfy only one
    reference timestamp; extra observed samples are retained as unmatched rows.
    The returned residual for a match is ``observed_time - reference_time``.
    """
    reference = canonical_time_axis(reference_times_s, name="reference_times_s")
    observed = canonical_time_axis(observed_times_s, name="observed_times_s")
    if isinstance(tolerance_s, bool) or not isinstance(tolerance_s, Real):
        raise TypeError("tolerance_s must be a real number")
    tolerance = float(tolerance_s)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_s must be finite and non-negative")
    if not isinstance(require_complete, bool):
        raise TypeError("require_complete must be a boolean")

    observed_indices: list[int | None] = []
    residuals: list[float | None] = []
    used: set[int] = set()
    for reference_time in reference:
        start = bisect_left(observed, reference_time - tolerance)
        stop = bisect_right(observed, reference_time + tolerance)
        if stop - start > 1:
            raise ValueError(
                "timestamp tolerance matches multiple observed samples for a reference sample"
            )
        if start == stop:
            observed_indices.append(None)
            residuals.append(None)
            continue
        observed_index = start
        if observed_index in used:
            raise ValueError("one observed timestamp cannot align to multiple reference samples")
        used.add(observed_index)
        observed_indices.append(observed_index)
        residuals.append(observed[observed_index] - reference_time)

    if require_complete and any(index is None for index in observed_indices):
        missing = [
            index for index, observed_index in enumerate(observed_indices) if observed_index is None
        ]
        raise ValueError(f"time-axis alignment is missing reference samples: {missing}")
    return TimeAxisAlignment(
        reference_times_s=reference,
        observed_times_s=observed,
        tolerance_s=tolerance,
        observed_indices=tuple(observed_indices),
        residuals_s=tuple(residuals),
        unmatched_observed_indices=tuple(
            index for index in range(len(observed)) if index not in used
        ),
    )
