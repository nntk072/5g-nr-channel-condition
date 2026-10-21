"""Validated, non-pickle datasets with separately stored oracle/metadata.

The model receives only H (no class label, profile, scene, distance or path oracle).
Each channel observation is [receive_antenna_links, subcarriers].
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ChannelDataset:
    h: np.ndarray  # [N, antennas_or_links, frequency_bins] complex
    y: np.ndarray  # [N] 1=geometric LOS, 0=NLOS or profile proxy
    group: np.ndarray  # [N] receiver location / link trajectory ID
    domain: np.ndarray  # [N] domain (tdl, cdl, scene ID)
    source: np.ndarray  # [N] source identifier (synthetic/tdl/cdl/rt)
    metadata: tuple[dict, ...]  # optional provenance only, NEVER used as features
    generation_report: dict = field(default_factory=dict)  # dataset-level QA/provenance

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
        if any(not str(g).strip() for g in self.group):
            raise ValueError("empty group ID is forbidden")
        try:
            report = json.loads(json.dumps(self.generation_report, sort_keys=True, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise TypeError("generation_report must contain finite JSON-compatible values") from exc
        if not isinstance(report, dict):
            raise TypeError("generation_report must be a JSON object")
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
        )

    @classmethod
    def concatenate(cls, datasets: list["ChannelDataset"]) -> "ChannelDataset":
        if not datasets:
            raise ValueError("No datasets")
        shapes = {(d.h.shape[1], d.h.shape[2]) for d in datasets}
        if len(shapes) != 1:
            raise ValueError(f"Incompatible antennas and subcarriers: {shapes}")
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
        )

    def save(self, path: str | Path) -> None:
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
            )
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path: str | Path) -> "ChannelDataset":
        with np.load(path, allow_pickle=False) as z:
            metadata = tuple(json.loads(str(z["metadata_json"].item())))
            report = (
                json.loads(str(z["generation_report_json"].item()))
                if "generation_report_json" in z.files
                else {}
            )
            return cls(z["h"], z["y"], z["group"], z["domain"], z["source"], metadata, report)
