"""Validated, non-pickle datasets with separately stored oracle/metadata.

The model receives only H (no class label, profile, scene, distance or path oracle).
Each channel observation is [receive_antenna_links, subcarriers].
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model_input_audit import audit_model_input_payload


@dataclass(frozen=True, slots=True)
class ChannelModelInputs:
    """CFR-only feature source without labels or dataset metadata.

    Whether this CFR is ideal, estimated, or precomputed is declared by the
    calling experiment; this wrapper does not claim that ideal CFR is a
    receiver-observable quantity.
    """

    observed_cfr: np.ndarray

    def __post_init__(self) -> None:
        observed = np.asarray(self.observed_cfr)
        if (
            observed.ndim != 3
            or any(size < 1 for size in observed.shape)
            or not np.iscomplexobj(observed)
        ):
            raise ValueError("observed_cfr must be complex [samples, antennas, subcarriers]")
        if not np.isfinite(observed).all():
            raise ValueError("observed_cfr must be nonempty and finite")
        # Expose a read-only view without duplicating a potentially large corpus.
        visible = observed.view()
        visible.setflags(write=False)
        object.__setattr__(self, "observed_cfr", visible)


@dataclass(frozen=True)
class ChannelDataset:
    h: np.ndarray  # [N, antennas_or_links, frequency_bins] complex
    y: np.ndarray  # [N] binary target interpreted only with label_semantics
    group: np.ndarray  # [N] receiver location / link trajectory ID
    domain: np.ndarray  # [N] domain (tdl, cdl, scene ID)
    source: np.ndarray  # [N] source identifier (synthetic/tdl/cdl/rt)
    metadata: tuple[dict, ...]  # optional provenance only, NEVER used as features
    generation_report: dict = field(default_factory=dict)  # dataset-level QA/provenance
    label_semantics: str = "unspecified_legacy"

    def with_split_child_binding(self, child_id: str, child_kind: str) -> "ChannelDataset":
        """Return a child artifact whose every row carries one frozen child identity.

        All rows must already identify the same ``split_subject_id`` parent.
        The returned dataset adds ``split_child_id`` and ``split_child_kind``
        to every row's metadata so a signed child-artifact audit can verify
        semantic coverage without exposing metadata as model features.
        """
        if not isinstance(child_id, str) or not child_id or child_id != child_id.strip():
            raise ValueError("child_id must be a nonempty canonical string")
        if not isinstance(child_kind, str) or not child_kind or child_kind != child_kind.strip():
            raise ValueError("child_kind must be a nonempty canonical string")
        parents: set[str] = set()
        bound_metadata: list[dict] = []
        for row_index, metadata in enumerate(self.metadata):
            if not isinstance(metadata, dict):
                raise ValueError(f"row {row_index} metadata must be an object")
            parent = metadata.get("split_subject_id")
            if not isinstance(parent, str) or not parent or parent != parent.strip():
                raise ValueError(
                    f"row {row_index} lacks a nonempty canonical split_subject_id"
                )
            parents.add(parent)
            existing_child = metadata.get("split_child_id")
            existing_kind = metadata.get("split_child_kind")
            if (existing_child is None) != (existing_kind is None):
                raise ValueError(f"row {row_index} has incomplete split child metadata")
            if existing_child is not None and (
                existing_child != child_id or existing_kind != child_kind
            ):
                raise ValueError(f"row {row_index} is already bound to a different split child")
            bound_metadata.append(
                {
                    **metadata,
                    "split_subject_id": parent,
                    "split_child_id": child_id,
                    "split_child_kind": child_kind,
                }
            )
        if len(parents) != 1:
            raise ValueError("a split child artifact must contain rows from exactly one parent")
        return ChannelDataset(
            self.h,
            self.y,
            self.group,
            self.domain,
            self.source,
            tuple(bound_metadata),
            {
                "schema": "loslab.dataset_derivation.v1",
                "operation": "bind_split_child",
                "child_id": child_id,
                "child_kind": child_kind,
                "parent_subject_id": next(iter(parents)),
                "row_count": len(bound_metadata),
                "parent_generation_report": self.generation_report,
            },
            self.label_semantics,
        )

    def model_inputs(self) -> ChannelModelInputs:
        """Return the CFR-only feature payload, isolated from labels and metadata."""
        return audit_model_input_payload(ChannelModelInputs(observed_cfr=self.h))

    def __post_init__(self) -> None:
        h = np.asarray(self.h)
        n = len(h)
        if h.ndim != 3 or not np.iscomplexobj(h) or h.shape[1] < 1 or h.shape[2] < 8:
            raise ValueError("h must be complex [samples, antennas, >=8 subcarriers]")
        if n < 1 or not np.all(np.isfinite(h)):
            raise ValueError("channel must be finite and non-empty")
        for name in ("y", "group", "domain", "source"):
            a = np.asarray(getattr(self, name))
            if a.shape != (n,):
                raise ValueError(f"{name} must be length {n}")
        if not np.all(np.isin(self.y, [0, 1])):
            raise ValueError("labels must be 0 or 1")
        if len(self.metadata) != n:
            raise ValueError("metadata count mismatch")
        if (
            not isinstance(self.label_semantics, str)
            or not self.label_semantics
            or self.label_semantics != self.label_semantics.strip()
        ):
            raise ValueError("label_semantics must be a non-empty string")
        if any(
            not isinstance(group_id, str)
            or not group_id
            or group_id != group_id.strip()
            for group_id in self.group
        ):
            raise ValueError("group IDs must be nonempty canonical strings")
        try:
            report = json.loads(json.dumps(self.generation_report, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise TypeError("generation_report must contain finite JSON-compatible values") from exc
        if not isinstance(report, dict):
            raise TypeError("generation_report must be a JSON object")
        reported_semantics = report.get("label_semantics")
        if reported_semantics is not None and reported_semantics != self.label_semantics:
            raise ValueError("label_semantics disagrees with generation_report")
        object.__setattr__(self, "generation_report", report)

    def select(self, indices: np.ndarray) -> "ChannelDataset":
        ii = np.asarray(indices, dtype=np.int64)
        return ChannelDataset(
            self.h[ii],
            self.y[ii],
            self.group[ii],
            self.domain[ii],
            self.source[ii],
            tuple(self.metadata[int(i)] for i in ii),
            {
                "schema": "loslab.dataset_derivation.v1",
                "operation": "select",
                "selected_samples": int(len(ii)),
                "selected_indices": ii.tolist(),
                "parent_generation_report": self.generation_report,
            },
            self.label_semantics,
        )

    @classmethod
    def concatenate(cls, datasets: list["ChannelDataset"]) -> "ChannelDataset":
        if not datasets:
            raise ValueError("No datasets")
        shapes = {(d.h.shape[1], d.h.shape[2]) for d in datasets}
        if len(shapes) != 1:
            raise ValueError(f"Incompatible antennas and subcarriers: {shapes}")
        semantics = {dataset.label_semantics for dataset in datasets}
        if len(semantics) != 1:
            raise ValueError(f"Incompatible binary label semantics: {sorted(semantics)}")
        return cls(
            np.concatenate([d.h for d in datasets]),
            np.concatenate([d.y for d in datasets]),
            np.concatenate([d.group for d in datasets]),
            np.concatenate([d.domain for d in datasets]),
            np.concatenate([d.source for d in datasets]),
            tuple(m for d in datasets for m in d.metadata),
            {
                "schema": "loslab.dataset_derivation.v1",
                "operation": "concatenate",
                "components": [d.generation_report for d in datasets],
            },
            datasets[0].label_semantics,
        )

    def save(self, path: str | Path, *, exclusive: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(
                tmp,
                h=self.h.astype(np.complex64),
                y=np.asarray(self.y, dtype=np.uint8),
                group=np.asarray(self.group, dtype=str),
                domain=np.asarray(self.domain, dtype=str),
                source=np.asarray(self.source, dtype=str),
                metadata_json=np.asarray(json.dumps(self.metadata, sort_keys=True)),
                generation_report_json=np.asarray(
                    json.dumps(self.generation_report, sort_keys=True, allow_nan=False)
                ),
                label_semantics_json=np.asarray(self.label_semantics),
            )
            if exclusive:
                os.link(tmp, path)
            else:
                os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path: str | Path) -> "ChannelDataset":
        return cls.load_bytes(Path(path).read_bytes())

    @classmethod
    def load_bytes(cls, payload: bytes) -> "ChannelDataset":
        """Parse exactly the immutable byte sequence used for provenance hashing."""
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            metadata = tuple(json.loads(str(z["metadata_json"].item())))
            report = (
                json.loads(str(z["generation_report_json"].item()))
                if "generation_report_json" in z.files
                else {}
            )
            label_semantics = (
                str(z["label_semantics_json"].item())
                if "label_semantics_json" in z.files
                else "unspecified_legacy"
            )
            return cls(
                z["h"],
                z["y"],
                z["group"],
                z["domain"],
                z["source"],
                metadata,
                report,
                label_semantics,
            )
