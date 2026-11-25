"""CDL profile metadata must be derived from the exact bytes it identifies."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from integrations.sionna.cdl import _profile_metadata


class _Order:
    def __init__(self, indices: list[int]) -> None:
        self.indices = indices

    def detach(self) -> _Order:
        return self

    def cpu(self) -> _Order:
        return self

    def tolist(self) -> list[int]:
        return self.indices


class _TensorOps:
    @staticmethod
    def as_tensor(values: list[float], *, dtype: Any, device: str) -> list[float]:
        del dtype, device
        return values

    @staticmethod
    def argsort(values: list[float]) -> _Order:
        return _Order(sorted(range(len(values)), key=values.__getitem__))


class _Resource:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.read_count = 0

    def read_bytes(self) -> bytes:
        self.read_count += 1
        return self.payload


class _ProfileResources:
    def __init__(self, resource: _Resource) -> None:
        self.resource = resource

    def parameter_file(self, filename: str, version: str) -> _Resource:
        assert filename == "CDL-A.json"
        assert version == "19.2"
        return self.resource

    @staticmethod
    def load_json(resource: _Resource) -> dict[str, Any]:
        del resource
        raise AssertionError("the table must be parsed from the hashed byte snapshot")


def test_cdl_profile_metadata_parses_and_hashes_one_resource_snapshot() -> None:
    payload = json.dumps(
        {
            "powers": [0.0, -3.0],
            "los": False,
            "delays": [1.0, 0.0],
            "aod": [10.0, 20.0],
            "aoa": [30.0, 40.0],
            "zod": [50.0, 60.0],
            "zoa": [70.0, 80.0],
            "cASD": 5.0,
            "cASA": 6.0,
            "cZSD": 7.0,
            "cZSA": 8.0,
            "xpr": 9.0,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    resource = _Resource(payload)

    metadata = _profile_metadata(
        "A",
        "19.2",
        100e-9,
        _ProfileResources(resource),
        _TensorOps(),
        "cpu",
        "float64",
    )

    assert resource.read_count == 1
    assert metadata["parameter_table_sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["nominal_cluster_delays_s"] == [0.0, 100e-9]
    assert metadata["nominal_cluster_angles_deg"]["aod"] == [20.0, 10.0]


def test_cdl_los_profile_metadata_separates_direct_component_from_diffuse_clusters() -> None:
    payload = json.dumps(
        {
            "powers": [0.0, -3.0, -6.0],
            "los": True,
            "delays": [0.0, 1.0, 2.0],
            "aod": [11.0, 21.0, 31.0],
            "aoa": [12.0, 22.0, 32.0],
            "zod": [13.0, 23.0, 33.0],
            "zoa": [14.0, 24.0, 34.0],
            "cASD": 5.0,
            "cASA": 6.0,
            "cZSD": 7.0,
            "cZSA": 8.0,
            "xpr": 9.0,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    metadata = _profile_metadata(
        "A",
        "19.2",
        100e-9,
        _ProfileResources(_Resource(payload)),
        _TensorOps(),
        "cpu",
        "float64",
    )

    assert metadata["profile_los_component"] is True
    assert metadata["nominal_los_angles_deg"] == {
        "aod": 11.0,
        "aoa": 12.0,
        "zod": 13.0,
        "zoa": 14.0,
    }
    assert metadata["nominal_cluster_delays_s"] == [100e-9, 200e-9]
